"""The listening engine: microphone in, validated text out.

This layer **never executes anything**. It produces a
:class:`~speech.listener.TranscriptionResult` and stops. The interface then
feeds ``result.text`` into :meth:`core.engine.AarmaEngine.handle_text`, exactly
as if it had been typed — so speech travels the same intent, permission, risk
and registry path as every other command, and there is no second execution route
to secure (specification sections 16 and 17).

Three rules shape everything here:

**Never fabricate.** Silence, noise, a short blip or a low-confidence guess all
end as a refusal with a reason, never as a command. The confidence gate is the
recogniser's own probability, compared against a configurable threshold.

**Never claim to be listening when it is not.** Every transition is published,
so the interface can show the true state rather than an optimistic one.

**Never block the interface.** Capture and transcription run on a worker
thread; listeners are called from it and the GUI marshals them across.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from config.settings import ConfigManager
from speech.capture import (
    CalibrationReport,
    DeviceResolution,
    MicrophoneCapture,
    MicrophoneDevice,
    list_devices,
    resolve_device,
)
from speech.diagnostics import DiagnosticLog, RecognitionDiagnostic
from speech.listener import (
    NullSpeechRecognizer,
    RecognizerAvailability,
    SpeechRecognizer,
    TranscriptionResult,
    confidence_floor,
)
from speech.vad import VadTuning
from utils.logger import get_logger

logger = get_logger("speech.input")


class ListeningState(str, Enum):
    """What the microphone is doing, in terms the interface can show."""

    DISABLED = "disabled"        # Permission off, or the user switched it off.
    UNAVAILABLE = "unavailable"  # No microphone, or no recogniser.
    IDLE = "idle"                # Ready, not listening.
    LISTENING = "listening"      # Capturing audio right now.
    PROCESSING = "processing"    # Audio captured; working out what was said.
    ERROR = "error"              # Something went wrong; message explains.

    @property
    def label(self) -> str:
        return {
            ListeningState.DISABLED: "Microphone off",
            ListeningState.UNAVAILABLE: "No microphone",
            ListeningState.IDLE: "Idle",
            ListeningState.LISTENING: "Listening…",
            ListeningState.PROCESSING: "Understanding…",
            ListeningState.ERROR: "Microphone problem",
        }[self]

    @property
    def is_busy(self) -> bool:
        return self in (ListeningState.LISTENING, ListeningState.PROCESSING)


@dataclass(frozen=True)
class ListeningReadiness:
    """Whether the microphone can be used, and what to say if not."""

    ready: bool
    state: ListeningState
    message: str = ""
    suggestion: str = ""
    processes_locally: bool = True
    needs_setup: bool = False
    device_name: str = ""

    @property
    def privacy_label(self) -> str:
        return ("Audio is processed on this computer and never uploaded."
                if self.processes_locally
                else "Audio is sent to an online service.")


StateCallback = Callable[[ListeningState, str], None]
LevelCallback = Callable[[float], None]
ResultCallback = Callable[[TranscriptionResult], None]


class ListeningEngine:
    """Owns the microphone, the recogniser and the listening state machine."""

    def __init__(
        self,
        config: ConfigManager,
        *,
        recognizer: Optional[SpeechRecognizer] = None,
        capture: Optional[MicrophoneCapture] = None,
        permissions=None,
        speech_engine=None,
    ) -> None:
        self._config = config
        self._permissions = permissions
        self._recognizer = recognizer if recognizer is not None else default_recognizer(config)
        self._capture = capture
        # Used to keep the assistant's own voice out of the microphone.
        self._speech_engine = speech_engine
        self._owned_capture: Optional[MicrophoneCapture] = None
        self._owned_device: Optional[int] = None
        self._device_notice = ""
        self._reported_fallback: Optional[str] = None
        #: Recent recognition attempts, for the diagnostics panel.
        self.diagnostics = DiagnosticLog()
        self._state = ListeningState.IDLE
        self._lock = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self._state_callbacks: List[StateCallback] = []
        self._level_callbacks: List[LevelCallback] = []
        self._result_callbacks: List[ResultCallback] = []

    # ------------------------------------------------------------- properties

    @property
    def recognizer(self) -> SpeechRecognizer:
        return self._recognizer

    @property
    def state(self) -> ListeningState:
        with self._lock:
            return self._state

    @property
    def is_listening(self) -> bool:
        return self.state.is_busy

    def devices(self) -> List[MicrophoneDevice]:
        return list_devices()

    def _permitted(self) -> bool:
        if self._permissions is None:
            return True
        from security.permissions import Permission

        return self._permissions.is_granted(Permission.MICROPHONE)

    def resolve_device(self) -> DeviceResolution:
        """Which microphone will actually be used, resolved by stable name."""
        return resolve_device(self._config.config.speech_input.device_name)

    def _capture_backend(self) -> MicrophoneCapture:
        """The capture object, rebuilt only when the chosen device changes.

        It must be *the same instance* across calls: ``cancel()`` sets a flag on
        the object that is recording, so handing out a fresh one each time would
        cancel something that was never listening — and the real recording would
        run to its timeout regardless of the user pressing stop.

        The device is resolved by name each time, so a microphone that was
        unplugged and replugged (and so changed index) is still found, and a
        device that has genuinely disappeared falls back to the Windows default
        instead of leaving AARMA attached to nothing.
        """
        if self._capture is not None:
            return self._capture

        resolution = self.resolve_device()
        device_index = resolution.index
        if resolution.fell_back and resolution.requested_name != self._reported_fallback:
            self._reported_fallback = resolution.requested_name
            self._device_notice = resolution.notice()
            logger.info("Microphone fell back to the default",
                        extra={"context": {"requested": resolution.requested_name}})

        with self._lock:
            if self._owned_capture is None or self._owned_device != device_index:
                if self._owned_capture is not None:
                    self._owned_capture.close()
                self._owned_capture = MicrophoneCapture(device_index=device_index)
                self._owned_device = device_index
            return self._owned_capture

    def take_device_notice(self) -> str:
        """Any pending "I switched microphones" message, consumed once."""
        notice, self._device_notice = self._device_notice, ""
        return notice

    def diagnose(self, *, seconds: float = 1.2):
        """Probe every device and report what is actually arriving."""
        from speech.capture import diagnose as run_diagnosis

        return run_diagnosis(self._config.config.speech_input.device_name,
                             seconds=seconds)

    def tuning(self) -> VadTuning:
        """Detector settings for the user's chosen sensitivity."""
        sensitivity = self._config.config.speech_input.vad_sensitivity
        return VadTuning().scaled(sensitivity=sensitivity)

    def vocabulary(self) -> List[str]:
        """Words the recogniser should expect but probably does not know.

        Just the assistant's name, and the name in its wake word if the user
        has changed one without the other. Measured on the ``small`` model:
        without this, "Asma" was transcribed as "Asthma" in every one of eight
        attempts across four acoustic conditions -- the assistant could not
        hear its own name.

        Deliberately *only* names. This list becomes the decoder's prompt, and
        conversation history or user content in a decoder prompt would bias the
        recognition of whatever is said next.
        """
        identity = self._config.config.assistant
        words: List[str] = []
        for candidate in (identity.name, *(identity.wake_word or "").split()):
            token = (candidate or "").strip()
            if token and token.lower() not in {w.lower() for w in words}:
                words.append(token)
        return words

    def preload(self) -> None:
        """Warm the recogniser in the background.

        The recommended model takes a few seconds to load. Doing it on the
        first command makes the first thing a user ever tries the slowest, so
        it happens quietly at startup instead.
        """
        recognizer = self._recognizer
        if not hasattr(recognizer, "preload"):
            return

        def warm() -> None:
            try:
                if getattr(recognizer, "is_model_downloaded", lambda: True)():
                    recognizer.preload()
            except Exception:  # noqa: BLE001 - warming is best-effort
                logger.debug("Model preload failed")

        threading.Thread(target=warm, name="aarma-stt-warm", daemon=True).start()

    # ------------------------------------------------------------ echo guard

    def _assistant_is_speaking(self) -> bool:
        engine = self._speech_engine
        if engine is None:
            return False
        try:
            return bool(engine.is_speaking)
        except Exception:  # noqa: BLE001
            return False

    def _silence_assistant(self) -> None:
        """Stop the assistant talking before opening the microphone.

        Without this the microphone hears "Opening Chrome." and transcribes the
        assistant's own reply as a command (specification section 10).
        """
        engine = self._speech_engine
        if engine is None:
            return
        try:
            if engine.is_speaking:
                engine.stop()
        except Exception:  # noqa: BLE001
            logger.debug("Could not stop speech before listening")

    # -------------------------------------------------------------- readiness

    def readiness(self) -> ListeningReadiness:
        """Can the microphone be used right now, and if not, why not?"""
        settings = self._config.config.speech_input

        if not settings.enabled:
            return ListeningReadiness(
                False, ListeningState.DISABLED,
                "Speech input is switched off.",
                "Turn on 'Listen for voice commands' in Settings.")

        if not self._permitted():
            # Section 4: the permission system is not bypassed, and the message
            # says exactly where to change it.
            return ListeningReadiness(
                False, ListeningState.DISABLED,
                "I don't have permission to use the microphone.",
                "Enable microphone access in Permissions.")

        # Windows hands a silent stream to desktop apps whose microphone access
        # is switched off, rather than failing -- so it has to be checked
        # explicitly, or a permission problem looks like a broken microphone.
        from speech.diagnostics import check_windows_permission

        windows = check_windows_permission()
        if not windows.allowed:
            return ListeningReadiness(
                False, ListeningState.UNAVAILABLE,
                "Windows is blocking desktop apps from using the microphone.",
                windows.advice())

        capture = self._capture_backend()
        reason = capture.unavailable_reason()
        if reason:
            return ListeningReadiness(False, ListeningState.UNAVAILABLE, reason,
                                      "You can keep using text input.")

        language = self._config.config.language.speech_input
        availability: RecognizerAvailability = self._recognizer.availability(language)
        if not availability.available:
            return ListeningReadiness(
                False, ListeningState.UNAVAILABLE, availability.reason,
                availability.suggestion,
                processes_locally=availability.processes_locally)

        device = self._current_device()
        return ListeningReadiness(
            True, ListeningState.IDLE,
            availability.reason, availability.suggestion,
            processes_locally=availability.processes_locally,
            needs_setup=availability.needs_setup,
            device_name=device.name if device else "")

    def _current_device(self) -> Optional[MicrophoneDevice]:
        return self.resolve_device().device

    # ------------------------------------------------------------ subscription

    def on_state_changed(self, callback: StateCallback) -> None:
        """Called on every state transition. Fires on the worker thread."""
        self._state_callbacks.append(callback)

    def on_level(self, callback: LevelCallback) -> None:
        """Called with the input level while recording. On the audio thread."""
        self._level_callbacks.append(callback)

    def remove_level_listener(self, callback: LevelCallback) -> None:
        """Unsubscribe. A dialog that has closed must stop being called."""
        try:
            self._level_callbacks.remove(callback)
        except ValueError:
            pass

    def on_result(self, callback: ResultCallback) -> None:
        """Called once per listening session, success or failure."""
        self._result_callbacks.append(callback)

    def _set_state(self, state: ListeningState, message: str = "") -> None:
        with self._lock:
            self._state = state
        for callback in list(self._state_callbacks):
            try:
                callback(state, message)
            except Exception:  # noqa: BLE001 - a listener must not break capture
                logger.exception("Listening state listener raised")

    def _emit_level(self, level: float) -> None:
        for callback in list(self._level_callbacks):
            try:
                callback(level)
            except Exception:  # noqa: BLE001
                pass

    def _emit_result(self, result: TranscriptionResult) -> None:
        for callback in list(self._result_callbacks):
            try:
                callback(result)
            except Exception:  # noqa: BLE001
                logger.exception("Transcription listener raised")

    # ------------------------------------------------------------- listening

    def start_listening(self) -> bool:
        """Begin one listening session. Returns False if it cannot start."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                logger.debug("Already listening; ignoring start")
                return False

            readiness = self.readiness()
            if not readiness.ready:
                self._set_state(readiness.state, readiness.message)
                self._emit_result(TranscriptionResult.unavailable(
                    readiness.message, provider=self._recognizer.name))
                return False

            self._worker = threading.Thread(target=self._run, name="aarma-listen",
                                            daemon=True)
            self._worker.start()
        return True

    def calibrate(self, *, on_phase=None, on_level=None) -> CalibrationReport:
        """Run the microphone check. Blocks; call from a worker thread."""
        if not self._permitted():
            return CalibrationReport(
                error="I don't have permission to use the microphone.")
        capture = self._capture_backend()
        self._set_state(ListeningState.LISTENING)
        try:
            report = capture.calibrate(on_phase=on_phase, on_level=on_level,
                                       tuning=self.tuning())
        finally:
            self._set_state(ListeningState.IDLE)
        return report

    def cancel(self) -> None:
        """Stop listening now. Safe from any thread and when idle."""
        capture = self._capture_backend()
        capture.cancel()
        self._recognizer.cancel()

    def shutdown(self, timeout: float = 3.0) -> None:
        """Cancel, wait, and release the device."""
        self.cancel()
        with self._lock:
            worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)
        # Close the stream explicitly: a held device blocks other applications
        # and is exactly the "stale microphone" the rebuild set out to remove.
        with self._lock:
            owned, self._owned_capture = self._owned_capture, None
        if owned is not None:
            owned.close()
        if self._capture is not None:
            close = getattr(self._capture, "close", None)
            if callable(close):
                close()

    def _run(self) -> None:
        result, record = self.listen_once()
        self._finish(result, record)

    def listen_once(self, *, audio_path: Optional[Path] = None
                    ) -> Tuple[TranscriptionResult, RecognitionDiagnostic]:
        """One complete capture-and-recognise cycle, run synchronously.

        Blocks; call from a worker thread. Returns the result *and* the
        :class:`~speech.diagnostics.RecognitionDiagnostic` behind it, and
        deliberately does **not** notify the result listeners -- so the speech
        test can measure "Open Chrome." without opening Chrome.

        ``audio_path`` writes the captured utterance to that exact file. It is
        for the benchmark, which needs a recording it can re-analyse without
        asking someone to say all twenty commands again. It is a *parameter*
        rather than the ``save_debug_audio`` setting on purpose: a benchmark
        must not switch on a persistent privacy-relevant preference behind the
        user's back, and normal conversation stays exactly as it was.
        """
        settings = self._config.config.speech_input
        language = self._config.config.language.speech_input
        capture = self._capture_backend()
        device = self._current_device()

        # perf_counter, not monotonic: on Windows the monotonic clock ticks
        # about every 15 ms, which is coarser than the thing being measured.
        started = time.perf_counter()
        record = RecognitionDiagnostic(
            device=device.name if device else "",
            language=language,
            model=getattr(self._recognizer, "model_size", self._recognizer.name))

        try:
            # Silence the assistant and let the speakers settle, so its own
            # voice cannot be captured as a command.
            self._silence_assistant()
            if settings.post_speech_delay > 0:
                deadline = time.monotonic() + settings.post_speech_delay
                while time.monotonic() < deadline and self._assistant_is_speaking():
                    time.sleep(0.02)

            self._set_state(ListeningState.LISTENING)
            captured = capture.record(
                max_seconds=settings.max_seconds,
                silence_timeout=settings.silence_timeout,
                no_speech_timeout=settings.no_speech_timeout,
                tuning=self.tuning(),
                on_level=self._emit_level,
            )

            record.capture_seconds = captured.raw_duration
            record.speech_seconds = captured.speech_seconds
            record.speech_start_sec = captured.speech_start_sec
            record.speech_end_sec = captured.speech_end_sec
            record.trailing_silence_sec = captured.trailing_silence_sec
            record.rms = captured.rms
            record.peak = captured.peak
            record.noise_rms = captured.noise_rms
            record.clipped = captured.clipped
            record.speech_detected = captured.has_speech

            if captured.cancelled:
                record.reason = "cancelled"
                return self._record_outcome(TranscriptionResult.cancelled(
                    provider=self._recognizer.name, language=language),
                    record, started)

            if not captured.ok:
                # Silence, a blip, or a device failure -- reported as-is. There
                # is nothing to transcribe and nothing to guess from.
                record.reason = captured.error_code or "capture_failed"
                record.detail = captured.error
                outcome = (TranscriptionResult.nothing_heard(
                    provider=self._recognizer.name, language=language,
                    duration=captured.duration)
                    if captured.error_code in ("no_speech", "empty_audio")
                    else TranscriptionResult.failed(
                        captured.error or "I couldn't record from the microphone.",
                        error_code=captured.error_code or "capture_failed",
                        provider=self._recognizer.name, language=language))
                return self._record_outcome(outcome, record, started)

            if audio_path is not None:
                record.audio_path = write_wav(captured.audio, audio_path)
            elif settings.save_debug_audio:
                record.audio_path = _save_debug_audio(captured.audio)

            self._set_state(ListeningState.PROCESSING)
            result = self._recognizer.transcribe(
                captured.audio, language=language, sample_rate=16_000,
                vocabulary=self.vocabulary())

            # The raw text, before any validation touched it. Without this the
            # difference between "recognition was wrong" and "our rules threw
            # away a correct reading" cannot be told apart.
            record.raw_text = result.raw_text or result.text
            record.confidence = result.confidence
            record.average_logprob = result.average_logprob
            record.no_speech_prob = result.no_speech_prob
            record.detected_language = result.detected_language
            if not result.ok and result.error_code:
                record.reason = result.error_code
                record.detail = result.error

            return self._record_outcome(
                self._apply_confidence_gate(result, settings.confidence_threshold,
                                            record),
                record, started)
        except Exception as exc:  # noqa: BLE001 - the worker must survive anything
            logger.exception("Listening session failed")
            record.reason = "internal_error"
            record.detail = f"{type(exc).__name__}: {exc}"
            return self._record_outcome(TranscriptionResult.failed(
                "Something went wrong while listening.",
                error_code="internal_error", provider=self._recognizer.name),
                record, started)

    def _apply_confidence_gate(self, result: TranscriptionResult,
                               threshold: float,
                               record: Optional[RecognitionDiagnostic] = None
                               ) -> TranscriptionResult:
        """Refuse a transcription the recogniser is not sure enough about.

        Only applied when the provider actually supplied a confidence -- a
        missing confidence is left alone rather than replaced with a guess
        (section 10) -- and scaled by utterance length, because short commands
        score systematically lower even when perfectly transcribed.
        """
        if not result.ok or result.confidence is None:
            return result

        floor = confidence_floor(result.text, threshold)
        if record is not None:
            record.threshold_used = floor
        if result.confidence >= floor:
            return result

        logger.info("Rejected low-confidence transcription", extra={"context": {
            "confidence": round(result.confidence, 3),
            "threshold": round(floor, 3), "base_threshold": threshold,
            "words": len(result.text.split()),
            "characters": len(result.text),   # Never the text itself.
        }})
        if record is not None:
            record.reason = "low_confidence"
            record.detail = (f"confidence {result.confidence:.3f} below "
                             f"{floor:.3f} for {len(result.text.split())} word(s)")
        return TranscriptionResult.unclear(
            language=result.language, confidence=result.confidence,
            duration=result.duration, provider=result.provider,
            processed_locally=result.processed_locally,
            average_logprob=result.average_logprob,
            no_speech_prob=result.no_speech_prob,
            detected_language=result.detected_language,
            raw_text=result.text)

    def _record_outcome(self, result: TranscriptionResult,
                        record: RecognitionDiagnostic, started: float
                        ) -> Tuple[TranscriptionResult, RecognitionDiagnostic]:
        """Complete the diagnostic record and file it.

        The user gets one friendly sentence; this is where the *exact* reason
        goes, so "I couldn't understand that clearly" is never the only thing
        anyone can find out (section 6). The transcription itself stays out of
        the log line -- only its shape.
        """
        record.latency = time.perf_counter() - started
        record.accepted = result.ok
        record.text = result.text
        if result.ok:
            record.reason = ""
        elif not record.reason:
            record.reason = result.error_code or "unknown"

        self.diagnostics.add(record)
        logger.info("Recognition finished", extra={"context": {
            "accepted": record.accepted, "reason": record.reason or "-",
            "detail": record.detail or "-",
            "confidence": record.confidence_label,
            "threshold": (round(record.threshold_used, 3)
                          if record.threshold_used is not None else "-"),
            "words": len(record.raw_text.split()),
            "speech_seconds": round(record.speech_seconds, 2),
            "latency": round(record.latency, 2),
        }})
        return result, record

    def _finish(self, result: TranscriptionResult,
                record: Optional[RecognitionDiagnostic] = None) -> None:
        if result.outcome.value == "error":
            self._set_state(ListeningState.ERROR, result.error)
        else:
            self._set_state(ListeningState.IDLE)
        self._emit_result(result)


#: How many diagnostic recordings are kept before the oldest is deleted. A
#: debugging aid must not quietly turn into an unbounded archive of a room.
DEBUG_AUDIO_KEEP = 20


def write_wav(audio, path: Path) -> str:
    """Write captured audio to ``path`` as 16 kHz mono PCM. Never raises.

    Returns the path written, or ``""`` on failure -- a recording is a
    diagnostic aid, and one that can break the thing it documents is worse
    than none.
    """
    try:
        import wave

        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.clip(np.asarray(audio, dtype="float32"), -1.0, 1.0)
        pcm = (samples * 32767.0).astype("<i2")

        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(pcm.tobytes())

        logger.info("Saved audio", extra={"context": {
            "seconds": round(samples.size / 16_000, 2)}})   # Never the path.
        return str(path)
    except Exception:  # noqa: BLE001 - a recording must never break listening
        logger.debug("Could not save audio")
        return ""


def _save_debug_audio(audio) -> str:
    """Write one captured utterance to the diagnostics folder.

    Reached solely when the user has switched **Save diagnostic audio** on --
    it is off by default, and the recording stays on this machine. Nothing is
    uploaded, nothing is attached to a log, and only the most recent
    :data:`DEBUG_AUDIO_KEEP` files are retained.
    """
    try:
        from utils.paths import get_paths

        folder = get_paths().diagnostic_dir
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = folder / f"speech-{stamp}-{int(time.monotonic() * 1000) % 1000:03d}.wav"
        written = write_wav(audio, path)
        if not written:
            return ""

        for stale in sorted(folder.glob("speech-*.wav"))[:-DEBUG_AUDIO_KEEP]:
            try:
                stale.unlink()
            except OSError:
                pass
        return written
    except Exception:  # noqa: BLE001 - a debugging aid must never break listening
        logger.debug("Could not save diagnostic audio")
        return ""


def default_recognizer(config: ConfigManager) -> SpeechRecognizer:
    """The recogniser shipped in this build.

    Offline only, by choice. Speech-to-text sends a recording of the room, so an
    online fallback is not the harmless convenience it is for speech *output* --
    it is a privacy decision, and it should be an explicit one rather than
    something that happens automatically when a model is missing. The interface
    is ready for an online provider whenever that choice is offered.
    """
    try:
        from speech.providers.whisper_local import LocalWhisperRecognizer

        return LocalWhisperRecognizer(config.config.speech_input.model_size)
    except Exception:  # noqa: BLE001
        logger.exception("Could not create the speech recogniser")
        return NullSpeechRecognizer()
