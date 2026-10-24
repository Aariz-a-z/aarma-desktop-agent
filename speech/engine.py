"""The speech engine: what gets said, by which voice, on a worker thread.

Everything the rest of AARMA needs from voice output is here. The engine owns
four responsibilities the providers deliberately do not:

**Choosing a provider.** Preference order comes from the user's mode setting and
actual connectivity, and the fallback is *explicit* -- when the natural voice is
unreachable and the Windows voice takes over, the caller is told so. Falling
back silently is how a user concludes the product simply sounds bad.

**Never speaking a language the voice cannot speak.** If an offline Kannada
voice does not exist, the engine says so and offers the alternatives rather
than reading Kannada text with an English voice, which is not a fallback but a
different failure (specification section 7).

**Keeping the interface responsive.** Synthesis and playback happen on a single
worker thread, so the GUI never waits for audio (section 13). One worker also
means two replies can never talk over each other.

**Filtering.** Every utterance passes through :mod:`speech.text` first, so
credentials and raw tool payloads cannot reach a speaker (section 12), and the
``speaker`` permission is checked before any audio is produced.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Sequence

from config.settings import ConfigManager
from speech.speaker import (
    NullSpeechSpeaker,
    SpeechRequest,
    SpeechResult,
    SpeechSpeaker,
)
from speech.text import prepare_for_speech, preview_line
from speech.voices import (
    VoiceAvailability,
    VoiceQuality,
    VoiceStyle,
    get_profile,
)
from utils.logger import get_logger

logger = get_logger("speech.engine")

_SHUTDOWN = object()


class SpeechPurpose(str, Enum):
    """Why something is being said. Affects delivery, never content."""

    REPLY = "reply"                # An answer to something the user asked.
    NOTIFICATION = "notification"  # Arrives unprompted: reminders, alerts.
    PREVIEW = "preview"            # The Test Voice button.


#: Notifications are delivered a little slower and calmer than replies: the
#: listener was not expecting to be spoken to and needs a moment to tune in.
_PURPOSE_STYLE: dict[SpeechPurpose, Optional[VoiceStyle]] = {
    SpeechPurpose.REPLY: None,
    SpeechPurpose.NOTIFICATION: VoiceStyle.CALM,
    SpeechPurpose.PREVIEW: None,
}


@dataclass(frozen=True)
class SpeechOutcome:
    """What actually happened, in terms the interface can show a user."""

    ok: bool
    purpose: SpeechPurpose = SpeechPurpose.REPLY
    provider: str = ""
    voice_name: str = ""
    quality: VoiceQuality = VoiceQuality.UNAVAILABLE
    online: bool = False
    error: str = ""
    error_code: str = ""
    interrupted: bool = False
    #: Set when the engine had to use something other than the preferred voice.
    notice: str = ""
    #: Set when the spoken text differed from the text on screen.
    redacted: bool = False
    condensed: bool = False


@dataclass
class _Task:
    text: str
    purpose: SpeechPurpose
    language: str
    voice_id: str
    style: VoiceStyle
    redacted: bool = False
    condensed: bool = False


OutcomeCallback = Callable[[SpeechOutcome], None]
StateCallback = Callable[[], None]


@dataclass
class _Callbacks:
    started: List[StateCallback] = field(default_factory=list)
    finished: List[OutcomeCallback] = field(default_factory=list)


class SpeechEngine:
    """Owns the voice: provider choice, the queue, and the worker thread."""

    def __init__(
        self,
        config: ConfigManager,
        *,
        providers: Optional[Sequence[SpeechSpeaker]] = None,
        connectivity=None,
        permissions=None,
        autostart: bool = True,
    ) -> None:
        self._config = config
        self._connectivity = connectivity
        self._permissions = permissions
        self._providers: List[SpeechSpeaker] = list(
            providers if providers is not None else default_providers()
        )
        self._queue: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        self._callbacks = _Callbacks()
        self._speaking = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.RLock()
        self._active: Optional[SpeechSpeaker] = None

        if autostart:
            self.start()

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker = threading.Thread(target=self._run, name="aarma-speech",
                                            daemon=True)
            self._worker.start()
        logger.info("Speech engine started", extra={"context": {
            "providers": [p.name for p in self._providers],
        }})

    def shutdown(self, timeout: float = 3.0) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        self.stop()
        self._queue.put(_SHUTDOWN)
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)
        logger.info("Speech engine stopped")

    # ------------------------------------------------------------- properties

    @property
    def enabled(self) -> bool:
        """Whether spoken replies are switched on *and* permitted."""
        return bool(self._config.config.voice.enabled) and self._permitted()

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def providers(self) -> List[SpeechSpeaker]:
        return list(self._providers)

    def _permitted(self) -> bool:
        if self._permissions is None:
            return True
        from security.permissions import Permission

        return self._permissions.is_granted(Permission.SPEAKER)

    def _online(self) -> bool:
        if self._connectivity is None:
            return True
        try:
            return bool(self._connectivity.is_online())
        except Exception:  # noqa: BLE001 - connectivity must never break speech
            return False

    # ----------------------------------------------------------- subscriptions

    def on_started(self, callback: StateCallback) -> None:
        """Called when an utterance begins. Fires on the worker thread."""
        self._callbacks.started.append(callback)

    def on_finished(self, callback: OutcomeCallback) -> None:
        """Called when an utterance ends, succeeds or fails. On the worker thread."""
        self._callbacks.finished.append(callback)

    def _emit_started(self) -> None:
        for callback in list(self._callbacks.started):
            try:
                callback()
            except Exception:  # noqa: BLE001 - a listener must not break speech
                logger.exception("Speech start listener raised")

    def _emit_finished(self, outcome: SpeechOutcome) -> None:
        for callback in list(self._callbacks.finished):
            try:
                callback(outcome)
            except Exception:  # noqa: BLE001
                logger.exception("Speech finish listener raised")

    # ------------------------------------------------------ provider selection

    def _preference_order(self) -> List[SpeechSpeaker]:
        """Providers to try, best first, for the current mode and connectivity."""
        mode = (self._config.config.voice.provider_mode or "auto").strip().lower()
        online_providers = [p for p in self._providers if p.requires_internet]
        offline_providers = [p for p in self._providers if not p.requires_internet]

        if mode == "offline":
            return offline_providers
        if mode == "online":
            return online_providers
        # auto: prefer the natural voice when there is a connection to reach it.
        if self._online():
            return online_providers + offline_providers
        return offline_providers

    def resolve(self, *, language: Optional[str] = None,
                voice_id: Optional[str] = None,
                style: Optional[VoiceStyle] = None) -> tuple[Optional[SpeechSpeaker], VoiceAvailability, str]:
        """Choose a provider without speaking.

        Returns ``(provider, availability, notice)``. ``notice`` is non-empty
        when the choice is not the preferred one, so the caller can explain the
        difference instead of leaving the user to wonder.
        """
        settings = self._config.config
        language = language or settings.language.speech_output
        voice_id = voice_id or settings.assistant.voice_id
        style = style or self.current_style()

        order = self._preference_order()
        if not order:
            return None, self._no_provider_availability(language), ""

        first_failure: Optional[VoiceAvailability] = None
        for index, provider in enumerate(order):
            availability = provider.describe_voice(language, voice_id, style)
            if availability.available:
                notice = ""
                if index > 0 and first_failure is not None:
                    notice = self._fallback_notice(first_failure, availability)
                return provider, availability, notice
            if first_failure is None:
                first_failure = availability

        return None, first_failure or self._no_provider_availability(language), ""

    def _fallback_notice(self, preferred: VoiceAvailability,
                         chosen: VoiceAvailability) -> str:
        """Explain, in one sentence, why this is not the best voice."""
        if preferred.online and not chosen.online:
            return (f"Using the {chosen.quality.label.lower()} offline voice — "
                    "the natural voice needs an internet connection.")
        return f"Using the {chosen.quality.label.lower()} voice — {preferred.reason}"

    def _no_provider_availability(self, language: str) -> VoiceAvailability:
        mode = (self._config.config.voice.provider_mode or "auto").lower()
        if mode == "online":
            reason = "Voice is set to online only, and no online voice is available."
            suggestion = "Switch voice mode to Automatic in Settings."
        elif mode == "offline":
            reason = "Voice is set to offline only, and no offline voice is available."
            suggestion = "Switch voice mode to Automatic, or install a Windows voice."
        else:
            reason = "No speech voice is available on this computer."
            suggestion = "Connect to the internet, or install a Windows voice."
        return VoiceAvailability(available=False, provider="none",
                                 quality=VoiceQuality.UNAVAILABLE, language=language,
                                 reason=reason, suggestion=suggestion)

    def availability(self) -> VoiceAvailability:
        """What the user would hear right now, for the Settings screen."""
        if not self._permitted():
            return VoiceAvailability(
                available=False, provider="none", quality=VoiceQuality.UNAVAILABLE,
                language=self._config.config.language.speech_output,
                reason="The Speaker permission is switched off.",
                suggestion="Enable 'Speaker' in Settings → Permissions.",
            )
        _provider, availability, _notice = self.resolve()
        return availability

    def current_style(self) -> VoiceStyle:
        raw = (self._config.config.voice.style or "").strip().lower()
        try:
            return VoiceStyle(raw)
        except ValueError:
            return get_profile(self._config.config.assistant.voice_id).default_style

    # ------------------------------------------------------------------ speech

    def speak(self, text: str, *, purpose: SpeechPurpose = SpeechPurpose.REPLY,
              interrupt: bool = True, language: Optional[str] = None,
              voice_id: Optional[str] = None) -> bool:
        """Queue ``text`` to be spoken. Returns False if it will not be.

        Returns immediately -- synthesis happens on the worker. ``interrupt``
        replaces anything currently queued, which is the right default for
        replies: two answers talking over each other helps nobody.
        """
        if not self.enabled and purpose is not SpeechPurpose.PREVIEW:
            return False
        if purpose is SpeechPurpose.PREVIEW and not self._permitted():
            return False
        if purpose is SpeechPurpose.NOTIFICATION and \
                not self._config.config.voice.speak_notifications:
            return False

        prepared = prepare_for_speech(text)
        if not prepared.speakable:
            return False

        settings = self._config.config
        style = _PURPOSE_STYLE.get(purpose) or self.current_style()
        task = _Task(
            text=prepared.text,
            purpose=purpose,
            language=language or settings.language.speech_output,
            voice_id=voice_id or settings.assistant.voice_id,
            style=style,
            redacted=prepared.redacted,
            condensed=prepared.condensed,
        )

        if interrupt:
            self.stop()
        self._queue.put(task)
        return True

    def speak_preview(self, *, voice_id: Optional[str] = None,
                      language: Optional[str] = None) -> bool:
        """Say the Test Voice line using the current assistant's name.

        The name comes from configuration, so a renamed assistant introduces
        itself correctly (section 15).
        """
        name = self._config.config.assistant.name
        return self.speak(preview_line(name), purpose=SpeechPurpose.PREVIEW,
                          interrupt=True, voice_id=voice_id, language=language)

    def stop(self) -> None:
        """Silence the assistant and discard anything queued."""
        self._drain()
        with self._lock:
            provider = self._active
        if provider is not None:
            provider.stop()

    def _drain(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _SHUTDOWN:      # Never discard the shutdown signal.
                self._queue.put(item)
                return

    # ------------------------------------------------------------------ worker

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                return
            if not isinstance(item, _Task):
                continue
            try:
                self._perform(item)
            except Exception:  # noqa: BLE001 - the worker must survive anything
                logger.exception("Speech task failed")
                self._emit_finished(SpeechOutcome(
                    ok=False, purpose=item.purpose,
                    error="Something went wrong while speaking.",
                    error_code="internal_error"))

    def _perform(self, task: _Task) -> None:
        provider, availability, notice = self.resolve(
            language=task.language, voice_id=task.voice_id, style=task.style)

        if provider is None or not availability.available:
            logger.info("No voice available for request", extra={"context": {
                "language": task.language, "reason": availability.reason}})
            self._emit_finished(SpeechOutcome(
                ok=False, purpose=task.purpose, quality=VoiceQuality.UNAVAILABLE,
                error=availability.reason, error_code="voice_unavailable",
                notice=availability.suggestion))
            return

        request = SpeechRequest(
            text=task.text,
            language=task.language,
            voice_id=task.voice_id,
            style=task.style,
            rate=self._config.config.voice.rate,
            volume=self._config.config.voice.volume,
            pitch=self._config.config.voice.pitch,
        )

        with self._lock:
            self._active = provider
        self._speaking.set()
        self._emit_started()
        try:
            result: SpeechResult = provider.speak(request)
        finally:
            self._speaking.clear()
            with self._lock:
                self._active = None

        logger.info("Speech finished", extra={"context": {
            "provider": result.provider, "ok": result.ok,
            "interrupted": result.interrupted, "purpose": task.purpose.value}})

        self._emit_finished(SpeechOutcome(
            ok=result.ok,
            purpose=task.purpose,
            provider=result.provider,
            voice_name=result.voice.name if result.voice else availability.voice_name,
            quality=result.voice.quality if result.voice else availability.quality,
            online=availability.online,
            error=result.error,
            error_code=result.error_code,
            interrupted=result.interrupted,
            notice=notice,
            redacted=task.redacted,
            condensed=task.condensed,
        ))


def default_providers() -> List[SpeechSpeaker]:
    """The providers shipped in this build, best quality first.

    Construction never fails: a provider that cannot run on this machine simply
    reports itself unavailable, and the engine skips it.
    """
    providers: List[SpeechSpeaker] = []

    try:
        from speech.providers.edge import EdgeNeuralSpeaker

        providers.append(EdgeNeuralSpeaker())
    except Exception:  # noqa: BLE001
        logger.exception("Could not create the neural speech provider")

    try:
        from speech.providers.sapi import SapiSpeaker

        providers.append(SapiSpeaker())
    except Exception:  # noqa: BLE001
        logger.exception("Could not create the Windows speech provider")

    if not providers:
        providers.append(NullSpeechSpeaker())
    return providers
