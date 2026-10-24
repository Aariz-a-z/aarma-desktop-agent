"""Online neural speech using Microsoft's Edge voices.

This is the provider that delivers the voice quality Phase 2A is actually about:
natural prosody, comfortable for long listening, and available in every language
in AARMA's catalogue -- English, Hindi, Kannada, Telugu, Tamil, Malayalam, Urdu,
Arabic, Spanish and French. Nothing offline on a stock Windows machine comes
close, which is why the engine prefers it whenever a connection exists.

**An honest caveat, recorded here rather than buried.** ``edge-tts`` speaks to
the endpoint Microsoft Edge's Read Aloud feature uses. It needs no account and
no key, but it is not a documented public API and Microsoft could change it. For
a capstone and for personal use that is a reasonable trade for the quality; for
a commercial release it should be swapped for Azure Speech (the same voices,
under a supported contract) or a local neural engine. The provider interface
exists precisely so that swap costs one file.

Audio arrives as MP3 and is played through :mod:`speech.audio`, which uses
Windows' own decoder -- so the natural voice adds no codec to the installer.
"""

from __future__ import annotations

import asyncio
import gc
import tempfile
import threading
import uuid
from pathlib import Path
from typing import List, Optional

from speech.audio import AudioPlayback, create_playback
from speech.speaker import SpeechRequest, SpeechResult, SpeechSpeaker, VoiceDescriptor
from speech.voices import (
    ONLINE_VOICES,
    STYLE_PROSODY,
    VoiceAvailability,
    VoiceQuality,
    VoiceStyle,
    language_display_name,
    resolve_online_voice,
)
from utils.logger import get_logger

logger = get_logger("speech.edge")

#: Give up rather than leave the user waiting in silence.
SYNTHESIS_TIMEOUT_SECONDS = 20.0


def _shutdown_loop(loop) -> None:
    """Close ``loop`` in an order that leaves nothing pointing at it.

    ``edge-tts`` speaks to Microsoft's service over a websocket, so each
    synthesis creates an ``aiohttp`` session and, on Windows' Proactor loop, a
    ``_ProactorSocketTransport``. ``Communicate.save()`` returns as soon as the
    audio is written, but that transport is closed *asynchronously*: its
    ``close()`` only schedules ``_call_connection_lost`` via ``call_soon``.

    Closing the loop immediately therefore leaves a live transport whose
    ``_loop`` is closed. When the collector eventually reaches it, its
    ``__del__`` calls ``close()`` -> ``call_soon()`` -> ``RuntimeError: Event
    loop is closed``, printed as "Exception ignored in" at interpreter
    shutdown. Measured on this machine: 5 of 5 runs of five syntheses.

    The order below is what actually fixes it, verified by isolating each step:

    1. cancel and collect anything still pending (a timed-out ``wait_for``
       leaves a cancelled task behind);
    2. **force collection while the loop is still open**, so the session and
       transport ``__del__`` methods run against a live loop and their
       ``call_soon`` succeeds -- this is the step that matters, and without it
       the rest is not enough (measured: 4 of 8 runs still failed);
    3. run the loop through ``shutdown_asyncgens`` and
       ``shutdown_default_executor``, which executes the close callbacks that
       step 2 queued;
    4. only then close.

    Nothing here suppresses the error: the transport close genuinely happens,
    it simply happens while it is still legal.
    """
    try:
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        gc.collect()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
    except Exception:  # noqa: BLE001 - teardown must not mask a synthesis result
        logger.debug("Speech event loop drain was incomplete")
    finally:
        loop.close()


def _percent(multiplier: float) -> str:
    """Edge expresses rate and volume as a signed percentage of natural."""
    return f"{int(round((multiplier - 1.0) * 100)):+d}%"


def _hertz(semitones: float) -> str:
    """Edge expresses pitch as a signed hertz offset."""
    return f"{int(round(semitones * 5)):+d}Hz"


class EdgeNeuralSpeaker(SpeechSpeaker):
    """Neural text-to-speech over the network."""

    name = "neural_online"
    display_name = "Natural online voice"
    requires_internet = True
    quality = VoiceQuality.NEURAL

    def __init__(self, playback: Optional[AudioPlayback] = None,
                 cache_dir: Optional[Path] = None) -> None:
        self._playback = playback or create_playback()
        self._cache_dir = cache_dir
        self._stop_requested = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ availability

    @staticmethod
    def is_installed() -> bool:
        """Whether the client library is present, without importing it fully."""
        import importlib.util

        return importlib.util.find_spec("edge_tts") is not None

    def is_available(self) -> bool:
        """Installed, and with something to play the audio through.

        Deliberately *not* a network check: connectivity is the engine's
        concern, and probing here would put a network round trip in front of
        every settings refresh.
        """
        return self.is_installed() and self._playback.is_available()

    def _work_dir(self) -> Path:
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            return self._cache_dir
        directory = Path(tempfile.gettempdir()) / "aarma-speech"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def describe_voice(self, language: str, voice_id: str,
                       style: Optional[VoiceStyle] = None) -> VoiceAvailability:
        if not self.is_installed():
            return VoiceAvailability(
                available=False, provider=self.name, quality=VoiceQuality.UNAVAILABLE,
                language=language, online=True,
                reason="The natural-voice component isn't installed.",
                suggestion="Run: pip install edge-tts",
            )
        if not self._playback.is_available():
            return VoiceAvailability(
                available=False, provider=self.name, quality=VoiceQuality.UNAVAILABLE,
                language=language, online=True,
                reason="No audio output device is available.",
            )

        voice = resolve_online_voice(language, voice_id, style)
        if voice is None:
            return VoiceAvailability(
                available=False, provider=self.name, quality=VoiceQuality.UNAVAILABLE,
                language=language, online=True,
                reason=(f"There's no natural {language_display_name(language)} voice "
                        "in this build."),
            )
        return VoiceAvailability(
            available=True, provider=self.name, quality=self.quality,
            voice_name=voice.name, language=voice.language, online=True,
        )

    def voices(self, language: str = "en") -> List[VoiceDescriptor]:
        base = (language or "en").split("-")[0].lower()
        table = ONLINE_VOICES.get(base, {})
        return [
            VoiceDescriptor(voice_id=name, display_name=name, gender=gender,
                            language=base, quality=VoiceQuality.NEURAL,
                            provider=self.name)
            for gender, names in table.items()
            for name in names
        ]

    # ------------------------------------------------------------------ speech

    def speak(self, request: SpeechRequest) -> SpeechResult:
        options = request.normalised()
        self._stop_requested.clear()

        if not self.is_installed():
            return SpeechResult.failure(
                self.name, "The natural-voice component isn't installed.",
                error_code="provider_unavailable")

        voice = resolve_online_voice(options.language, options.voice_id, options.style)
        if voice is None:
            return SpeechResult.failure(
                self.name,
                "I don't have a natural "
                f"{language_display_name(options.language)} voice available.",
                error_code="voice_unavailable")

        prosody = STYLE_PROSODY.get(options.style, STYLE_PROSODY[VoiceStyle.CLEAR])
        audio_path = self._work_dir() / f"speech-{uuid.uuid4().hex[:12]}.mp3"

        try:
            self._synthesise(
                text=options.text,
                voice=voice.name,
                rate=_percent(options.rate * prosody.rate_scale),
                volume=_percent(options.volume + 0.1),  # 0.9 is our natural point.
                pitch=_hertz(options.pitch + prosody.pitch_offset),
                destination=audio_path,
            )
        except asyncio.TimeoutError:
            return SpeechResult.failure(self.name, "The voice service took too long.",
                                        error_code="timeout")
        except Exception as exc:  # noqa: BLE001 - network/client errors vary widely
            logger.error("Neural synthesis failed",
                         extra={"context": {"voice": voice.name,
                                            "error": type(exc).__name__}})
            return SpeechResult.failure(
                self.name, "I couldn't reach the natural voice service.",
                error_code="synthesis_failed")

        if self._stop_requested.is_set():
            audio_path.unlink(missing_ok=True)
            return SpeechResult.cancelled(self.name)

        try:
            played = self._playback.play(audio_path)
        finally:
            audio_path.unlink(missing_ok=True)

        if self._stop_requested.is_set():
            return SpeechResult.cancelled(self.name)
        if not played:
            return SpeechResult.failure(self.name, "I couldn't play that out loud.",
                                        error_code="playback_failed")
        return SpeechResult.success(self.name, voice)

    def _synthesise(self, *, text: str, voice: str, rate: str, volume: str,
                    pitch: str, destination: Path) -> None:
        """Render ``text`` to ``destination``.

        ``edge-tts`` is asyncio-based; a private loop is created for this call
        because the engine's worker thread has none of its own and must not
        disturb any loop the host application is running.
        """
        import edge_tts

        async def run() -> None:
            communicate = edge_tts.Communicate(text, voice, rate=rate,
                                               volume=volume, pitch=pitch)
            await asyncio.wait_for(communicate.save(str(destination)),
                                   timeout=SYNTHESIS_TIMEOUT_SECONDS)

        with self._lock:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(run())
            finally:
                _shutdown_loop(loop)

    def stop(self) -> None:
        self._stop_requested.set()
        self._playback.stop()

    def unavailable_reason(self) -> str:
        if not self.is_installed():
            return "The natural-voice component isn't installed. Run: pip install edge-tts"
        if not self._playback.is_available():
            return "No audio output device is available."
        return "The natural voice needs an internet connection."
