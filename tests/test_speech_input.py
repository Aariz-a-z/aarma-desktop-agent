"""Speech input: capture, recognition, refusal, permission and privacy.

No test needs a microphone, a speech model or a network. Capture and
recognition are replaced with fakes, which is the only way to assert the
property that matters most: **AARMA never invents a transcription.** Silence,
noise, a cancelled session, a provider failure and a low-confidence guess must
all end as refusals — never as a command reaching the engine.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence

import pytest

from config.settings import ConfigManager, SpeechInputSettings, load_config
from speech.capture import CaptureResult, MicrophoneDevice
from speech.input_engine import ListeningEngine, ListeningState
from speech.listener import (
    NullSpeechRecognizer,
    RecognitionOutcome,
    RecognizerAvailability,
    SpeechRecognizer,
    TranscriptionResult,
    confidence_floor,
    confidence_from_logprob,
)


# ------------------------------------------------------------------ fakes

class FakeCapture:
    """Stands in for the microphone. Records how it was driven."""

    def __init__(self, result: Optional[CaptureResult] = None, *,
                 unavailable: str = "", duration: float = 0.0,
                 levels: Sequence[float] = ()) -> None:
        self._result = result if result is not None else CaptureResult(
            audio=_fake_audio(), duration=1.5, peak=0.4, speech_seconds=1.2)
        self._unavailable = unavailable
        self._duration = duration
        self._levels = list(levels)
        self.calls: List[dict] = []
        self.cancel_calls = 0
        self._cancelled = threading.Event()

    def unavailable_reason(self) -> str:
        return self._unavailable

    def record(self, *, max_seconds=15.0, silence_timeout=1.2,
               no_speech_timeout=6.0, min_speech_seconds=0.25,
               tuning=None, on_level=None, on_speech_started=None) -> CaptureResult:
        self.calls.append({"max_seconds": max_seconds,
                           "silence_timeout": silence_timeout,
                           "no_speech_timeout": no_speech_timeout,
                           "tuning": tuning})
        self._cancelled.clear()
        for level in self._levels:
            if on_level is not None:
                on_level(level)
        if self._duration:
            deadline = time.time() + self._duration
            while time.time() < deadline:
                if self._cancelled.is_set():
                    return CaptureResult(cancelled=True)
                time.sleep(0.01)
        if self._cancelled.is_set():
            return CaptureResult(cancelled=True)
        return self._result

    def cancel(self) -> None:
        self.cancel_calls += 1
        self._cancelled.set()

    def close(self) -> None:
        self.close_calls = getattr(self, "close_calls", 0) + 1


class FakeRecognizer(SpeechRecognizer):
    """A recogniser whose every answer is dictated by the test."""

    name = "fake"
    display_name = "Fake recogniser"
    processes_locally = True

    def __init__(self, result: Optional[TranscriptionResult] = None, *,
                 available: bool = True, languages: Sequence[str] = ("en", "hi"),
                 reason: str = "Fake is unavailable.",
                 needs_setup: bool = False) -> None:
        self._result = result
        self._available = available
        self._languages = tuple(languages)
        self._reason = reason
        self._needs_setup = needs_setup
        self.calls: List[dict] = []
        self.cancel_calls = 0

    def is_available(self) -> bool:
        return self._available

    def availability(self, language: str = "en") -> RecognizerAvailability:
        if not self._available:
            return RecognizerAvailability(False, self.name, reason=self._reason,
                                          suggestion="Try text input.")
        if language.split("-")[0] not in self._languages:
            return RecognizerAvailability(
                False, self.name,
                reason=f"No recognition for {language}.",
                suggestion="Change the speech input language.")
        return RecognizerAvailability(True, self.name, needs_setup=self._needs_setup)

    def supported_languages(self) -> Sequence[str]:
        return self._languages

    def transcribe(self, audio, *, language: str = "en",
                   sample_rate: int = 16_000,
                   vocabulary: Sequence[str] = ()) -> TranscriptionResult:
        self.calls.append({"language": language, "sample_rate": sample_rate,
                           "audio_is_none": audio is None,
                           "vocabulary": tuple(vocabulary)})
        if self._result is not None:
            return self._result
        return TranscriptionResult.transcribed(
            "open chrome", language=language, confidence=0.8, provider=self.name)

    def cancel(self) -> None:
        self.cancel_calls += 1


class FakePermissions:
    def __init__(self, granted: bool = True) -> None:
        self.granted = granted

    def is_granted(self, _permission) -> bool:
        return self.granted


def _fake_audio(seconds: float = 1.5):
    import numpy as np

    return np.zeros(int(16_000 * seconds), dtype="float32")


def wait_for(items: List, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if items:
            return items[0]
        time.sleep(0.01)
    return None


@pytest.fixture
def capture() -> FakeCapture:
    return FakeCapture()


@pytest.fixture
def recognizer() -> FakeRecognizer:
    return FakeRecognizer()


@pytest.fixture
def listening(config: ConfigManager, capture, recognizer):
    engine = ListeningEngine(config, recognizer=recognizer, capture=capture,
                             permissions=FakePermissions(True))
    yield engine
    engine.shutdown()


@pytest.fixture
def results(listening: ListeningEngine) -> List[TranscriptionResult]:
    collected: List[TranscriptionResult] = []
    listening.on_result(collected.append)
    return collected


@pytest.fixture
def states(listening: ListeningEngine) -> List[ListeningState]:
    collected: List[ListeningState] = []
    listening.on_state_changed(lambda state, _message: collected.append(state))
    return collected


# ------------------------------------------------------------ result types

class TestTranscriptionResult:
    def test_transcribed_text_is_actionable(self) -> None:
        result = TranscriptionResult.transcribed("open chrome", confidence=0.9)
        assert result.ok and result.is_actionable
        assert result.outcome is RecognitionOutcome.TRANSCRIBED

    @pytest.mark.parametrize("factory", [
        TranscriptionResult.nothing_heard,
        TranscriptionResult.unclear,
        TranscriptionResult.cancelled,
    ])
    def test_refusals_are_never_actionable(self, factory) -> None:
        result = factory()
        assert not result.ok
        assert result.text == ""
        assert result.error

    def test_empty_text_is_not_actionable_even_if_marked_transcribed(self) -> None:
        """A blank transcription is not a command."""
        assert not TranscriptionResult.transcribed("   ").ok

    def test_failure_carries_a_code_for_the_log_and_prose_for_the_user(self) -> None:
        result = TranscriptionResult.failed("I couldn't process that audio.",
                                            error_code="transcription_failed")
        assert result.error_code == "transcription_failed"
        assert "couldn't" in result.error

    def test_confidence_is_none_when_not_supplied(self) -> None:
        assert TranscriptionResult.transcribed("hello").confidence is None

    def test_detected_language_is_preserved_when_a_provider_reports_it(self) -> None:
        result = TranscriptionResult.transcribed("hello", language="en",
                                                 detected_language="hi")
        assert result.detected_language == "hi"

    def test_confidence_derivation_matches_calibration(self) -> None:
        """Derived from the model's own log probability, not invented."""
        assert confidence_from_logprob(-0.255) == pytest.approx(0.775, abs=0.01)
        assert confidence_from_logprob(-0.755) == pytest.approx(0.470, abs=0.01)
        assert 0.0 <= confidence_from_logprob(-99.0) <= 1.0
        assert confidence_from_logprob(0.0) == pytest.approx(1.0)


# -------------------------------------------------------------- readiness

class TestReadiness:
    def test_ready_when_everything_is_present(self, listening) -> None:
        readiness = listening.readiness()
        assert readiness.ready
        assert readiness.state is ListeningState.IDLE

    def test_microphone_permission_is_enforced(self, config, capture, recognizer) -> None:
        """Section 4: speech input must not bypass the permission system."""
        engine = ListeningEngine(config, recognizer=recognizer, capture=capture,
                                 permissions=FakePermissions(False))
        readiness = engine.readiness()
        assert not readiness.ready
        assert readiness.state is ListeningState.DISABLED
        assert "permission" in readiness.message.lower()
        assert "Permissions" in readiness.suggestion

    def test_denied_permission_prevents_capture_entirely(
        self, config, capture, recognizer
    ) -> None:
        engine = ListeningEngine(config, recognizer=recognizer, capture=capture,
                                 permissions=FakePermissions(False))
        assert engine.start_listening() is False
        assert capture.calls == []

    def test_disabled_setting_prevents_listening(self, listening, config,
                                                 capture) -> None:
        config.update_speech_input(enabled=False)
        assert listening.start_listening() is False
        assert capture.calls == []

    def test_missing_microphone_is_reported(self, config, recognizer) -> None:
        engine = ListeningEngine(config, recognizer=recognizer,
                                 capture=FakeCapture(unavailable="No microphone was detected."),
                                 permissions=FakePermissions(True))
        readiness = engine.readiness()
        assert not readiness.ready
        assert readiness.state is ListeningState.UNAVAILABLE
        assert "microphone" in readiness.message.lower()
        assert "text input" in readiness.suggestion.lower()

    def test_unavailable_recognizer_is_reported(self, config, capture) -> None:
        engine = ListeningEngine(config, recognizer=FakeRecognizer(available=False),
                                 capture=capture, permissions=FakePermissions(True))
        readiness = engine.readiness()
        assert not readiness.ready
        assert readiness.state is ListeningState.UNAVAILABLE

    def test_unsupported_language_is_reported(self, listening, config) -> None:
        config.update_language(speech_input="kn")
        readiness = listening.readiness()
        assert not readiness.ready
        assert "kn" in readiness.message or "recognition" in readiness.message.lower()

    def test_privacy_state_is_exposed(self, listening) -> None:
        """Section 15: the user must be able to tell where audio is processed."""
        readiness = listening.readiness()
        assert readiness.processes_locally
        assert "this computer" in readiness.privacy_label

    def test_online_provider_privacy_is_labelled_differently(self) -> None:
        remote = RecognizerAvailability(True, "cloud", processes_locally=False)
        assert "online service" in remote.privacy_label

    def test_one_time_setup_is_surfaced(self, config, capture) -> None:
        engine = ListeningEngine(config, recognizer=FakeRecognizer(needs_setup=True),
                                 capture=capture, permissions=FakePermissions(True))
        readiness = engine.readiness()
        assert readiness.ready and readiness.needs_setup


# ------------------------------------------------------------- the session

class TestListeningSession:
    def test_successful_transcription_reaches_the_result_callback(
        self, listening, results
    ) -> None:
        assert listening.start_listening()
        result = wait_for(results)
        assert result is not None and result.ok
        assert result.text == "open chrome"

    def test_states_move_idle_listening_processing_idle(
        self, listening, states, results
    ) -> None:
        listening.start_listening()
        wait_for(results)
        assert ListeningState.LISTENING in states
        assert ListeningState.PROCESSING in states
        assert states[-1] is ListeningState.IDLE

    def test_configured_language_reaches_the_recognizer(
        self, listening, config, recognizer, results
    ) -> None:
        config.update_language(speech_input="hi")
        listening.start_listening()
        wait_for(results)
        assert recognizer.calls[0]["language"] == "hi"

    def test_capture_settings_reach_the_microphone(
        self, listening, config, capture, results
    ) -> None:
        config.update_speech_input(max_seconds=9.0, silence_timeout=2.5)
        listening.start_listening()
        wait_for(results)
        assert capture.calls[0]["max_seconds"] == pytest.approx(9.0)
        assert capture.calls[0]["silence_timeout"] == pytest.approx(2.5)

    def test_levels_are_published_while_recording(self, config, recognizer) -> None:
        engine = ListeningEngine(config, recognizer=recognizer,
                                 capture=FakeCapture(levels=[0.01, 0.2, 0.05]),
                                 permissions=FakePermissions(True))
        levels: List[float] = []
        results: List[TranscriptionResult] = []
        engine.on_level(levels.append)
        engine.on_result(results.append)
        try:
            engine.start_listening()
            wait_for(results)
            assert levels == [0.01, 0.2, 0.05]
        finally:
            engine.shutdown()

    def test_a_second_start_is_ignored_while_busy(self, config, recognizer) -> None:
        """One microphone, one session: no overlapping capture."""
        engine = ListeningEngine(config, recognizer=recognizer,
                                 capture=FakeCapture(duration=1.0),
                                 permissions=FakePermissions(True))
        try:
            assert engine.start_listening() is True
            time.sleep(0.15)
            assert engine.start_listening() is False
        finally:
            engine.shutdown()

    def test_cancel_stops_capture_and_reports_cancellation(
        self, config, recognizer
    ) -> None:
        capture = FakeCapture(duration=3.0)
        engine = ListeningEngine(config, recognizer=recognizer, capture=capture,
                                 permissions=FakePermissions(True))
        results: List[TranscriptionResult] = []
        engine.on_result(results.append)
        try:
            engine.start_listening()
            time.sleep(0.2)
            engine.cancel()
            result = wait_for(results)
            assert result is not None
            assert result.outcome is RecognitionOutcome.CANCELLED
            assert capture.cancel_calls >= 1
        finally:
            engine.shutdown()

    def test_cancelled_session_never_transcribes(self, config, recognizer) -> None:
        capture = FakeCapture(duration=3.0)
        engine = ListeningEngine(config, recognizer=recognizer, capture=capture,
                                 permissions=FakePermissions(True))
        results: List[TranscriptionResult] = []
        engine.on_result(results.append)
        try:
            engine.start_listening()
            time.sleep(0.2)
            engine.cancel()
            wait_for(results)
            assert recognizer.calls == []
        finally:
            engine.shutdown()

    def test_cancel_reaches_the_object_that_is_recording(self, config,
                                                         recognizer) -> None:
        """A fresh capture object per call means stop cancels the wrong one.

        Caught by the real-microphone test: pressing stop left the recording
        running to its timeout because `cancel()` acted on a different instance.
        """
        engine = ListeningEngine(config, recognizer=recognizer,
                                 permissions=FakePermissions(True))
        first = engine._capture_backend()
        second = engine._capture_backend()
        assert first is second

    def test_changing_the_device_rebuilds_capture(self, config, recognizer,
                                                  monkeypatch) -> None:
        from speech.capture import MicrophoneDevice

        devices = [
            MicrophoneDevice(0, "Built-in Microphone", 2, 48000.0, is_default=True),
            MicrophoneDevice(3, "USB Headset", 1, 44100.0),
        ]
        monkeypatch.setattr("speech.capture.list_devices", lambda: devices)

        engine = ListeningEngine(config, recognizer=recognizer,
                                 permissions=FakePermissions(True))
        first = engine._capture_backend()
        config.update_speech_input(device_name="USB Headset")
        assert engine._capture_backend() is not first

    def test_shutdown_leaves_no_capture_running(self, config, recognizer) -> None:
        """Section 12: no background audio capture may survive."""
        capture = FakeCapture(duration=5.0)
        engine = ListeningEngine(config, recognizer=recognizer, capture=capture,
                                 permissions=FakePermissions(True))
        engine.start_listening()
        time.sleep(0.2)
        engine.shutdown(timeout=3.0)
        assert capture.cancel_calls >= 1
        assert not engine.is_listening

    def test_a_listener_that_raises_does_not_break_the_session(
        self, listening, results
    ) -> None:
        listening.on_result(lambda _r: (_ for _ in ()).throw(RuntimeError("boom")))
        listening.start_listening()
        assert wait_for(results) is not None


# -------------------------------------------------- refusals, no fabrication

class TestNeverFabricates:
    """The central guarantee of Phase 2B (specification sections 10 and 11)."""

    def test_silence_produces_no_command(self, config, recognizer) -> None:
        capture = FakeCapture(CaptureResult(
            audio=None, duration=3.0, speech_seconds=0.0,
            error="I didn't hear anything.", error_code="no_speech"))
        engine = ListeningEngine(config, recognizer=recognizer, capture=capture,
                                 permissions=FakePermissions(True))
        results: List[TranscriptionResult] = []
        engine.on_result(results.append)
        try:
            engine.start_listening()
            result = wait_for(results)
            assert result.outcome is RecognitionOutcome.NO_SPEECH
            assert result.text == ""
            assert recognizer.calls == []   # Nothing was even sent to recognise.
        finally:
            engine.shutdown()

    def test_capture_failure_produces_no_command(self, config, recognizer) -> None:
        capture = FakeCapture(CaptureResult(
            error="I couldn't record from the microphone.",
            error_code="capture_failed"))
        engine = ListeningEngine(config, recognizer=recognizer, capture=capture,
                                 permissions=FakePermissions(True))
        results: List[TranscriptionResult] = []
        engine.on_result(results.append)
        try:
            engine.start_listening()
            result = wait_for(results)
            assert not result.ok
            assert result.text == ""
        finally:
            engine.shutdown()

    def test_recognizer_failure_produces_no_command(self, config, capture) -> None:
        engine = ListeningEngine(
            config,
            recognizer=FakeRecognizer(TranscriptionResult.failed("Model exploded.")),
            capture=capture, permissions=FakePermissions(True))
        results: List[TranscriptionResult] = []
        engine.on_result(results.append)
        try:
            engine.start_listening()
            result = wait_for(results)
            assert not result.ok and result.text == ""
        finally:
            engine.shutdown()

    def test_low_confidence_is_refused_rather_than_guessed(
        self, config, capture
    ) -> None:
        engine = ListeningEngine(
            config,
            recognizer=FakeRecognizer(TranscriptionResult.transcribed(
                "delete everything in my documents folder now", confidence=0.2)),
            capture=capture, permissions=FakePermissions(True))
        results: List[TranscriptionResult] = []
        engine.on_result(results.append)
        try:
            engine.start_listening()
            result = wait_for(results)
            assert result.outcome is RecognitionOutcome.LOW_CONFIDENCE
            assert result.text == ""            # The guess is discarded, not passed on.
            assert "clearly" in result.error
            assert result.confidence == pytest.approx(0.2)
            # The discarded reading is still available to a developer, so
            # "misheard" and "rules rejected a correct reading" stay distinct.
            assert result.raw_text == "delete everything in my documents folder now"
        finally:
            engine.shutdown()

    def test_confidence_threshold_is_configurable(self, listening, config,
                                                  capture) -> None:
        listening._recognizer = FakeRecognizer(
            TranscriptionResult.transcribed(
                "open chrome and search for the weather", confidence=0.5))
        config.update_speech_input(confidence_threshold=0.8)
        results: List[TranscriptionResult] = []
        listening.on_result(results.append)
        listening.start_listening()
        assert wait_for(results).outcome is RecognitionOutcome.LOW_CONFIDENCE

        config.update_speech_input(confidence_threshold=0.3)
        results.clear()
        listening.start_listening()
        assert wait_for(results).ok

    def test_missing_confidence_is_not_invented(self, listening) -> None:
        """Section 10: no fabricated confidence for providers that omit it."""
        result = listening._apply_confidence_gate(
            TranscriptionResult.transcribed("open chrome", confidence=None), 0.9)
        assert result.ok
        assert result.confidence is None


class TestShortCommandsAreNotPenalised:
    """A legitimate short command must not be refused for being short.

    ``avg_logprob`` is a mean over tokens, so one- and two-word utterances score
    systematically lower than sentences even when transcribed perfectly.
    Measured locally, a flat 0.45 gate refused *correct* readings of "Asma"
    (0.379), "Help." (0.447) and "Open Chrome." (0.400) -- exactly the commands
    a voice assistant exists to handle.
    """

    def test_floor_scales_with_utterance_length(self) -> None:
        assert confidence_floor("Asma", 0.40) < confidence_floor("Open Chrome now", 0.40)
        assert (confidence_floor("Open Chrome now", 0.40)
                < confidence_floor("open chrome and search for the weather", 0.40))

    def test_long_utterances_keep_the_full_threshold(self) -> None:
        base = 0.45
        assert confidence_floor("one two three four five six", base) == base

    def test_measured_short_commands_would_now_be_accepted(self, listening) -> None:
        """The exact readings and scores measured on this machine."""
        for text, confidence in (("Asma", 0.379), ("Help.", 0.447),
                                 ("Open Chrome.", 0.400), ("Stop.", 0.366),
                                 ("Hello Asma", 0.469)):
            result = listening._apply_confidence_gate(
                TranscriptionResult.transcribed(text, confidence=confidence), 0.35)
            assert result.ok, f"{text!r} at {confidence} was refused"

    def test_a_genuinely_unsure_short_reading_is_still_refused(self, listening) -> None:
        result = listening._apply_confidence_gate(
            TranscriptionResult.transcribed("blurgh", confidence=0.05), 0.35)
        assert result.outcome is RecognitionOutcome.LOW_CONFIDENCE


# ----------------------------------------------------- recogniser contract

class TestNullRecognizer:
    def test_never_claims_availability_or_text(self) -> None:
        recognizer = NullSpeechRecognizer()
        assert not recognizer.is_available()
        result = recognizer.transcribe(_fake_audio(), language="en")
        assert result.outcome is RecognitionOutcome.UNAVAILABLE
        assert result.text == ""

    def test_suggests_text_input_as_the_fallback(self) -> None:
        availability = NullSpeechRecognizer().availability("en")
        assert not availability.available
        assert "text input" in availability.suggestion.lower()


class TestLocalRecognizerDegradesGracefully:
    """The shipped recogniser must be safe to interrogate with no model present."""

    def test_constructs_and_reports_without_loading_a_model(self) -> None:
        from speech.providers.whisper_local import LocalWhisperRecognizer

        recognizer = LocalWhisperRecognizer("base")
        assert recognizer.processes_locally is True
        assert recognizer.model_size == "base"
        recognizer.availability("en")          # Must not raise.
        recognizer.is_model_downloaded()       # Must not raise.

    def test_rejects_an_unsupported_language_without_audio(self) -> None:
        from speech.providers.whisper_local import LocalWhisperRecognizer

        result = LocalWhisperRecognizer().transcribe(_fake_audio(), language="klingon")
        assert result.outcome is RecognitionOutcome.UNAVAILABLE
        assert result.text == ""

    def test_empty_audio_is_refused_before_the_model_loads(self) -> None:
        from speech.providers.whisper_local import LocalWhisperRecognizer

        for audio in (None,):
            result = LocalWhisperRecognizer().transcribe(audio, language="en")
            assert result.outcome is RecognitionOutcome.NO_SPEECH

    def test_invalid_model_size_falls_back_to_the_default(self) -> None:
        from speech.providers.whisper_local import DEFAULT_MODEL_SIZE, LocalWhisperRecognizer

        assert LocalWhisperRecognizer("enormous").model_size == DEFAULT_MODEL_SIZE

    def test_declares_the_catalogue_languages(self) -> None:
        from speech.providers.whisper_local import LocalWhisperRecognizer

        recognizer = LocalWhisperRecognizer()
        for language in ("en", "hi", "kn", "te", "ta", "ml"):
            assert recognizer.supports_language(language)

    def test_distinguishes_verified_from_merely_supported(self) -> None:
        """Honesty: Indic accuracy was poor at the default model size."""
        from speech.providers.whisper_local import LocalWhisperRecognizer

        recognizer = LocalWhisperRecognizer()
        assert recognizer.is_language_verified("en")
        assert not recognizer.is_language_verified("kn")


# ------------------------------------------------------------ capture layer

class TestCaptureLayer:
    def test_device_descriptor_marks_the_default(self) -> None:
        device = MicrophoneDevice(1, "Realtek Microphone", 2, 44100.0, is_default=True)
        assert "default" in device.display_name

    def test_capture_result_reports_speech_presence(self) -> None:
        assert not CaptureResult(error="x").ok
        assert not CaptureResult(audio=_fake_audio(), speech_seconds=0.0).has_speech
        assert CaptureResult(audio=_fake_audio(), speech_seconds=1.0).has_speech

    def test_cancelled_capture_is_not_ok(self) -> None:
        assert not CaptureResult(audio=_fake_audio(), cancelled=True).ok

    def test_enumeration_never_raises(self) -> None:
        from speech.capture import has_microphone, list_devices

        list_devices()
        has_microphone()


# ----------------------------------------------------------- configuration

class TestSpeechInputConfiguration:
    def test_defaults_are_sensible(self, config: ConfigManager) -> None:
        settings = config.config.speech_input
        assert settings.enabled is True
        # 'small' is the benchmarked default: the only size that handled
        # distant speech and non-English languages acceptably.
        assert settings.model_size == "small"
        assert settings.device_index is None
        assert 0.0 < settings.confidence_threshold < 1.0
        assert 0.0 <= settings.vad_sensitivity <= 1.0
        # Failure must not cost the full recording limit.
        assert settings.no_speech_timeout < settings.max_seconds

    def test_settings_persist_across_a_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        first = ConfigManager(path)
        first.load()
        first.update_speech_input(model_size="small", device_index=3,
                                  max_seconds=20.0, confidence_threshold=0.7,
                                  enabled=False)

        reloaded = load_config(path)
        settings = reloaded.config.speech_input
        assert settings.model_size == "small"
        assert settings.device_index == 3
        assert settings.max_seconds == pytest.approx(20.0)
        assert settings.confidence_threshold == pytest.approx(0.7)
        assert settings.enabled is False

    def test_a_config_written_before_speech_input_existed_still_loads(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "config.json"
        path.write_text('{"assistant": {"name": "Nova"}}', encoding="utf-8")
        manager = load_config(path)
        assert manager.config.assistant.name == "Nova"
        assert manager.config.speech_input == SpeechInputSettings()


# ------------------------------------------------- engine integration

class TestEngineIntegration:
    """Transcribed text must travel the ordinary, audited command path."""

    def test_transcribed_text_runs_through_the_normal_pipeline(self, engine) -> None:
        spoken = TranscriptionResult.transcribed("what time is it", confidence=0.9)
        response = engine.handle_text(spoken.text, source="voice")
        assert response.ok
        assert response.action == "get_time"

    def test_spoken_commands_obey_permissions(self, engine) -> None:
        from security.permissions import Permission

        engine.permissions.revoke(Permission.FILESYSTEM)
        response = engine.handle_text("list files in Documents", source="voice")
        assert not response.ok
        assert "permission" in response.text.lower()

    def test_spoken_high_risk_commands_still_require_confirmation(
        self, engine, sandbox
    ) -> None:
        (sandbox / "disposable.txt").write_text("bye", encoding="utf-8")
        response = engine.handle_text("delete disposable.txt", source="voice")
        assert response.kind.value == "confirmation"
        assert (sandbox / "disposable.txt").exists()

    def test_voice_source_is_recorded_in_the_audit_trail(self, engine) -> None:
        engine.handle_text("what time is it", source="voice")
        entry = engine.database.recent_actions(limit=1)[0]
        assert entry.source == "voice"

    def test_listening_engine_is_lazy(self, engine) -> None:
        """A text-only session must not load a speech model."""
        assert engine.listening_started is False

    def test_listening_engine_shuts_down_with_the_engine(self, config, database,
                                                         resolver) -> None:
        from core.engine import AarmaEngine

        instance = AarmaEngine(config, database=database, path_resolver=resolver)
        instance.listening  # noqa: B018 - constructs it
        assert instance.listening_started
        instance.shutdown()
        assert instance.listening_started is False
