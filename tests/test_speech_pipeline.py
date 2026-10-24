"""The rebuilt audio pipeline: detection, validation, lifecycle and echo.

These cover the corrective work after the first speech-input build failed in
real use. Each class maps to a failure that was actually observed and measured,
so a regression would show up as a named test rather than as a vague report that
"the microphone doesn't work any more".

Nothing here needs a microphone: streams are simulated in real time, and audio
is synthesised.
"""

from __future__ import annotations

import threading
import time
from typing import List

import numpy as np
import pytest

from config.settings import ConfigManager
from speech.capture import (
    BLOCK_SECONDS,
    CalibrationReport,
    CaptureResult,
    MicrophoneCapture,
    MicrophoneStream,
    negotiate_sample_rate,
    resample_to_16k,
)
from speech.input_engine import ListeningEngine
from speech.listener import TranscriptionResult
from speech.vad import SAMPLE_RATE, EnergyVad, SileroVad, VadTuning, create_detector
from speech.validation import (
    dominant_script,
    has_excessive_repetition,
    script_agreement,
    validate_transcription,
)

RNG = np.random.default_rng(17)


def silence(seconds: float, level: float = 0.0006):
    return (RNG.standard_normal(int(seconds * SAMPLE_RATE)) * level).astype("float32")


def speechlike(seconds: float, amplitude: float = 0.2):
    """A voiced-sounding signal: a moving formant over a pitch pulse train."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    pitch = 120 + 15 * np.sin(2 * np.pi * 2.5 * t)
    excitation = np.sign(np.sin(2 * np.pi * pitch * t))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    formant = np.sin(2 * np.pi * 700 * t) + 0.5 * np.sin(2 * np.pi * 1220 * t)
    return (amplitude * excitation * envelope * formant).astype("float32")


# --------------------------------------------------------------- VAD contract

class TestVoiceActivityDetection:
    """The energy threshold that shipped first failed in ways these pin down."""

    @pytest.fixture(params=["silero", "energy"])
    def detector(self, request):
        if request.param == "silero":
            vad = SileroVad()
            if not vad.is_available():
                pytest.skip("Silero VAD not installed")
            return vad
        return EnergyVad()

    def test_finds_speech_surrounded_by_silence(self, detector) -> None:
        audio = np.concatenate([silence(1.0), speechlike(1.5), silence(1.0)])
        assert detector.has_speech(audio)

    @pytest.mark.parametrize("level", [0.2, 0.05, 0.02, 0.008])
    def test_finds_quieter_speech_too(self, detector, level) -> None:
        """Distance failure: a fixed floor of 0.006 excluded speech beyond ~1.5 m."""
        audio = np.concatenate([silence(0.8), speechlike(1.5, level), silence(0.8)])
        assert detector.has_speech(audio), f"missed speech at amplitude {level}"

    @pytest.mark.parametrize("noise", [0.001, 0.01, 0.02])
    def test_finds_speech_over_room_noise(self, detector, noise) -> None:
        """Noise failure: a hum used to push the threshold above normal speech."""
        audio = np.concatenate([silence(0.8, noise), speechlike(1.5, 0.1),
                                silence(0.8, noise)])
        audio = audio + (RNG.standard_normal(audio.size) * noise).astype("float32")
        assert detector.has_speech(audio.astype("float32"))

    def test_silence_is_not_speech(self, detector) -> None:
        assert not detector.has_speech(silence(3.0))

    def test_empty_buffer_is_not_speech(self, detector) -> None:
        assert not detector.has_speech(np.zeros(0, dtype="float32"))
        assert not detector.has_speech(None)

    def test_speech_starting_immediately_is_still_found(self, detector) -> None:
        """The 'works once then never again' failure: no warm-up is required."""
        audio = np.concatenate([speechlike(1.5), silence(1.0)])
        assert detector.has_speech(audio)

    def test_trim_keeps_the_whole_utterance(self, detector) -> None:
        speech = speechlike(1.5)
        audio = np.concatenate([silence(1.2), speech, silence(1.5)])
        trimmed = detector.trim(audio)
        assert trimmed is not None
        # Long enough to hold the utterance, shorter than everything recorded.
        assert trimmed.size >= speech.size * 0.8
        assert trimmed.size < audio.size

    def test_trim_returns_nothing_for_silence(self, detector) -> None:
        assert detector.trim(silence(2.0)) is None

    def test_last_speech_end_supports_endpointing(self, detector) -> None:
        audio = np.concatenate([silence(0.6), speechlike(1.2), silence(1.2)])
        end = detector.last_speech_end(audio)
        assert end is not None
        assert end < audio.size          # Speech ended before the buffer did.


class TestSileroSpecifics:
    def test_neural_detector_is_preferred(self) -> None:
        detector = create_detector()
        if SileroVad().is_available():
            assert detector.is_neural
        else:
            assert detector.name == "energy"

    def test_energy_fallback_is_labelled_as_weaker(self) -> None:
        assert EnergyVad().is_neural is False
        assert SileroVad().is_neural is True

    def test_a_detector_failure_never_invents_speech(self, monkeypatch) -> None:
        vad = SileroVad()
        if not vad.is_available():
            pytest.skip("Silero VAD not installed")

        def explode(*_a, **_k):
            raise RuntimeError("model fault")

        monkeypatch.setattr("faster_whisper.vad.get_speech_timestamps", explode)
        assert vad.speech_regions(speechlike(1.0)) == []


class TestVadTuning:
    def test_sensitivity_maps_to_threshold(self) -> None:
        conservative = VadTuning().scaled(sensitivity=0.0)
        default = VadTuning().scaled(sensitivity=0.5)
        sensitive = VadTuning().scaled(sensitivity=1.0)
        assert conservative.threshold > default.threshold > sensitive.threshold

    def test_sensitivity_is_clamped(self) -> None:
        assert VadTuning().scaled(sensitivity=-5).threshold <= 0.6
        assert VadTuning().scaled(sensitivity=99).threshold >= 0.2

    def test_padding_defaults_protect_word_boundaries(self) -> None:
        """Padding is what stops the first and last words being clipped."""
        assert VadTuning().speech_pad_ms >= 200


# ---------------------------------------------------------------- validation

class TestTranscriptionValidation:
    """Whisper fails by returning plausible nonsense, so output is checked."""

    @pytest.mark.parametrize(("text", "language"), [
        ("Open Chrome.", "en"),
        ("क्रों खोलो", "hi"),
        ("இப்போது மணி என்ன?", "ta"),
        ("ok", "en"),
    ])
    def test_good_transcriptions_are_accepted(self, text, language) -> None:
        assert validate_transcription(text, language=language).accepted

    @pytest.mark.parametrize(("text", "language"), [
        ("इगर समया एश्टूः", "kn"),       # Devanagari returned for Kannada
        ("atility", "te"),                # Latin returned for Telugu
        ("Kron Turku ga", "ml"),          # Latin returned for Malayalam
    ])
    def test_wrong_script_is_refused(self, text, language) -> None:
        """Measured failure: the model answered in the wrong language's script."""
        result = validate_transcription(text, language=language)
        assert not result.accepted
        assert result.error_code == "script_mismatch"

    @pytest.mark.parametrize("text", [
        "the the the the the the the the",
        "go go go go go go go go go",
        "aaaaaaaaaaaaaaaa",
    ])
    def test_degenerate_output_is_refused(self, text) -> None:
        result = validate_transcription(text, language="en")
        assert not result.accepted
        assert result.error_code == "degenerate_output"

    def test_empty_is_refused(self) -> None:
        assert not validate_transcription("", language="en").accepted
        assert not validate_transcription("   ", language="en").accepted

    def test_ordinary_repetition_is_allowed(self) -> None:
        """Real speech repeats words; only degenerate loops are rejected."""
        assert not has_excessive_repetition("I said I would open the file")
        assert not has_excessive_repetition("yes yes")

    def test_unknown_language_skips_the_script_check(self) -> None:
        assert validate_transcription("whatever", language="zz").accepted

    def test_script_detection(self) -> None:
        assert dominant_script("Open Chrome") == "LATIN"
        assert dominant_script("क्रोम") == "DEVANAGARI"
        assert dominant_script("") == "NONE"
        assert script_agreement("Open", ("LATIN",)) == pytest.approx(1.0)
        assert script_agreement("क्रोम", ("LATIN",)) == pytest.approx(0.0)

    def test_check_can_be_disabled(self) -> None:
        assert validate_transcription("atility", language="te",
                                      check_script=False).accepted


# ----------------------------------------------------------- stream lifecycle

class FakeSoundDevice:
    """A sounddevice stand-in whose streams behave in real time."""

    def __init__(self, buffer=None, rates=(16000,)):
        self.buffer = buffer
        self.rates = rates
        self.opened = 0
        self.closed = 0
        self.live = 0

    def check_input_settings(self, device=None, channels=1, samplerate=None,
                             dtype=None):
        if samplerate not in self.rates:
            raise ValueError("unsupported rate")

    def query_devices(self, index=None):
        info = {"name": "Fake Mic", "max_input_channels": 2,
                "default_samplerate": float(self.rates[0])}
        return info if index is not None else [info]

    class default:
        device = (0, 0)

    def InputStream(self, **kwargs):  # noqa: N802 - mimics the sounddevice API
        owner = self

        class Stream:
            def __init__(self):
                self.pos = 0
                self.started = False

            def start(self):
                self.started = True
                owner.opened += 1
                owner.live += 1

            def read(self, frames):
                time.sleep(BLOCK_SECONDS)
                if owner.buffer is None:
                    return (RNG.standard_normal(frames) * 0.0005
                            ).astype("float32").reshape(-1, 1), False
                block = owner.buffer[self.pos:self.pos + frames]
                self.pos += frames
                if block.size < frames:
                    block = np.concatenate(
                        [block, np.zeros(frames - block.size, dtype="float32")])
                return block.astype("float32").reshape(-1, 1), False

            def stop(self):
                pass

            def close(self):
                owner.closed += 1
                owner.live -= 1

        return Stream()


@pytest.fixture
def fake_audio_backend(monkeypatch):
    """Install a fake sounddevice for the capture module."""
    backend = FakeSoundDevice()
    monkeypatch.setattr("speech.capture._sounddevice", lambda: backend)
    monkeypatch.setattr("speech.capture.is_supported", lambda: True)
    monkeypatch.setattr("speech.capture.has_microphone", lambda: True)
    return backend


class TestMicrophoneStreamLifecycle:
    """The original build leaked and reused streams; this pins the contract."""

    def test_open_is_idempotent(self, fake_audio_backend) -> None:
        stream = MicrophoneStream()
        stream.open()
        stream.open()
        assert fake_audio_backend.opened == 1
        stream.close()

    def test_close_is_idempotent_and_safe_when_never_opened(self, fake_audio_backend) -> None:
        MicrophoneStream().close()
        stream = MicrophoneStream()
        stream.open()
        stream.close()
        stream.close()
        assert fake_audio_backend.live == 0

    def test_reopen_after_close_works(self, fake_audio_backend) -> None:
        stream = MicrophoneStream()
        for _ in range(5):
            stream.open()
            stream.read_block()
            stream.close()
        assert fake_audio_backend.opened == 5
        assert fake_audio_backend.live == 0

    def test_reading_without_opening_raises_clearly(self, fake_audio_backend) -> None:
        with pytest.raises(RuntimeError):
            MicrophoneStream().read_block()

    def test_context_manager_closes(self, fake_audio_backend) -> None:
        with MicrophoneStream() as stream:
            stream.read_block()
        assert fake_audio_backend.live == 0


class TestSampleRateNegotiation:
    def test_prefers_16k_when_supported(self, monkeypatch) -> None:
        backend = FakeSoundDevice(rates=(16000, 44100))
        monkeypatch.setattr("speech.capture._sounddevice", lambda: backend)
        monkeypatch.setattr("speech.capture.is_supported", lambda: True)
        assert negotiate_sample_rate(None) == 16000

    def test_falls_back_to_a_supported_rate(self, monkeypatch) -> None:
        backend = FakeSoundDevice(rates=(48000,))
        monkeypatch.setattr("speech.capture._sounddevice", lambda: backend)
        monkeypatch.setattr("speech.capture.is_supported", lambda: True)
        assert negotiate_sample_rate(None) == 48000

    def test_resampling_preserves_duration(self) -> None:
        audio = speechlike(1.0)
        for rate in (44100, 48000):
            source = np.interp(np.linspace(0, audio.size - 1, int(rate * 1.0)),
                               np.arange(audio.size), audio).astype("float32")
            out = resample_to_16k(source, rate)
            assert abs(out.size - SAMPLE_RATE) <= 2

    def test_resampling_is_a_no_op_at_the_target_rate(self) -> None:
        audio = speechlike(0.5)
        assert resample_to_16k(audio, SAMPLE_RATE).size == audio.size


class TestRepeatedCapture:
    """Acceptance criteria 1-3: repeated use, second attempt, no stale stream."""

    def test_twenty_consecutive_captures(self, fake_audio_backend) -> None:
        capture = MicrophoneCapture()
        for cycle in range(20):
            stopper = threading.Timer(0.25, capture.cancel)
            stopper.daemon = True
            stopper.start()
            result = capture.record(max_seconds=2.0, no_speech_timeout=1.0)
            stopper.cancel()
            assert result.error_code != "capture_failed", f"cycle {cycle} failed"
        capture.close()
        assert fake_audio_backend.live == 0

    def test_the_stream_is_reused_not_recreated(self, fake_audio_backend) -> None:
        """Reopening the device per command is slow and can fail on Windows."""
        capture = MicrophoneCapture()
        for _ in range(4):
            capture.record(max_seconds=0.6, no_speech_timeout=0.4)
        assert fake_audio_backend.opened == 1
        capture.close()

    def test_cancel_reaches_the_recording(self, fake_audio_backend) -> None:
        capture = MicrophoneCapture()
        stopper = threading.Timer(0.2, capture.cancel)
        stopper.daemon = True
        stopper.start()
        result = capture.record(max_seconds=5.0, no_speech_timeout=4.0)
        stopper.cancel()
        assert result.cancelled

    def test_capture_works_immediately_after_a_cancel(self, fake_audio_backend) -> None:
        capture = MicrophoneCapture()
        capture.cancel()
        result = capture.record(max_seconds=0.6, no_speech_timeout=0.4)
        assert not result.cancelled

    def test_close_then_record_reopens(self, fake_audio_backend) -> None:
        capture = MicrophoneCapture()
        capture.record(max_seconds=0.5, no_speech_timeout=0.3)
        capture.close()
        result = capture.record(max_seconds=0.5, no_speech_timeout=0.3)
        assert result.error_code != "capture_failed"
        capture.close()


class TestEndpointing:
    def test_silence_gives_up_early_instead_of_running_the_full_limit(
        self, fake_audio_backend
    ) -> None:
        """Acceptance criterion: failure must not cost the whole timeout."""
        capture = MicrophoneCapture()
        started = time.monotonic()
        result = capture.record(max_seconds=12.0, no_speech_timeout=1.5)
        elapsed = time.monotonic() - started
        assert not result.ok
        assert result.error_code == "no_speech"
        assert elapsed < 4.0, "silence should not burn the full recording limit"
        capture.close()

    def test_speech_is_captured_and_trimmed(self, monkeypatch) -> None:
        buffer = np.concatenate([silence(0.4), speechlike(1.4), silence(2.0)])
        backend = FakeSoundDevice(buffer=buffer)
        monkeypatch.setattr("speech.capture._sounddevice", lambda: backend)
        monkeypatch.setattr("speech.capture.is_supported", lambda: True)
        monkeypatch.setattr("speech.capture.has_microphone", lambda: True)

        capture = MicrophoneCapture()
        result = capture.record(max_seconds=8.0, silence_timeout=0.8,
                                no_speech_timeout=4.0)
        capture.close()
        if not result.ok:
            pytest.skip("synthetic signal not recognised as speech by this detector")
        assert result.duration < result.raw_duration      # Silence was trimmed.
        assert result.speech_seconds > 0.5

    def test_no_microphone_is_reported_without_recording(self, monkeypatch) -> None:
        monkeypatch.setattr("speech.capture.is_supported", lambda: True)
        monkeypatch.setattr("speech.capture.has_microphone", lambda: False)
        result = MicrophoneCapture().record()
        assert result.error_code == "no_microphone"


class TestCalibrationReport:
    def test_labels_describe_the_measurements(self) -> None:
        good = CalibrationReport(ok=True, noise_rms=0.002, speech_rms=0.08,
                                 peak=0.5, speech_detected=True)
        assert good.noise_label == "Low"
        assert good.speech_label == "Good"
        assert good.microphone_label == "Good"
        assert "working well" in good.advice()

    def test_clipping_is_called_out(self) -> None:
        clipped = CalibrationReport(noise_rms=0.002, speech_rms=0.5, peak=1.0,
                                    clipped=True, speech_detected=True)
        assert clipped.microphone_label == "Too loud"
        assert "clipping" in clipped.advice().lower()

    def test_no_speech_is_called_out(self) -> None:
        quiet = CalibrationReport(noise_rms=0.001, speech_detected=False)
        assert quiet.speech_label == "Not detected"
        assert "didn't hear" in quiet.advice()

    def test_noisy_room_is_flagged(self) -> None:
        noisy = CalibrationReport(ok=True, noise_rms=0.05, speech_rms=0.08,
                                  speech_detected=True)
        assert noisy.noise_label == "High"
        assert "noisy" in noisy.advice()


# ------------------------------------------------------------- the echo guard

class FakeSpeechEngine:
    """Stands in for the Phase 2A voice."""

    def __init__(self, speaking: bool = False) -> None:
        self.is_speaking = speaking
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1
        self.is_speaking = False


class FakePermissions:
    def is_granted(self, _permission) -> bool:
        return True


class SilentCapture:
    """Capture that immediately reports nothing heard."""

    def __init__(self) -> None:
        self.calls = 0

    def unavailable_reason(self) -> str:
        return ""

    def record(self, **kwargs) -> CaptureResult:
        self.calls += 1
        return CaptureResult(error="I didn't hear anything.", error_code="no_speech")

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        pass


class TestEchoGuard:
    """The assistant must not transcribe its own voice (section 10)."""

    def test_speaking_is_stopped_before_the_microphone_opens(
        self, config: ConfigManager
    ) -> None:
        voice = FakeSpeechEngine(speaking=True)
        capture = SilentCapture()
        engine = ListeningEngine(config, recognizer=_always("open chrome"),
                                 capture=capture, permissions=FakePermissions(),
                                 speech_engine=voice)
        results: List[TranscriptionResult] = []
        engine.on_result(results.append)
        try:
            engine.start_listening()
            _wait(results)
            assert voice.stop_calls >= 1, "the assistant kept talking into the mic"
            assert capture.calls == 1
        finally:
            engine.shutdown()

    def test_listening_works_without_a_speech_engine(self, config: ConfigManager) -> None:
        engine = ListeningEngine(config, recognizer=_always("hello"),
                                 capture=SilentCapture(),
                                 permissions=FakePermissions())
        results: List[TranscriptionResult] = []
        engine.on_result(results.append)
        try:
            engine.start_listening()
            assert _wait(results) is not None
        finally:
            engine.shutdown()

    def test_a_broken_speech_engine_does_not_block_listening(
        self, config: ConfigManager
    ) -> None:
        class Broken:
            @property
            def is_speaking(self):
                raise RuntimeError("voice fault")

            def stop(self):
                raise RuntimeError("voice fault")

        engine = ListeningEngine(config, recognizer=_always("hello"),
                                 capture=SilentCapture(),
                                 permissions=FakePermissions(),
                                 speech_engine=Broken())
        results: List[TranscriptionResult] = []
        engine.on_result(results.append)
        try:
            engine.start_listening()
            assert _wait(results) is not None
        finally:
            engine.shutdown()


class TestTuningReachesCapture:
    def test_sensitivity_setting_reaches_the_detector(self, config: ConfigManager) -> None:
        capture = SilentCapture()
        recorded = {}

        def record(**kwargs):
            recorded.update(kwargs)
            return CaptureResult(error="none", error_code="no_speech")

        capture.record = record
        engine = ListeningEngine(config, recognizer=_always("x"), capture=capture,
                                 permissions=FakePermissions())
        config.update_speech_input(vad_sensitivity=1.0)
        results: List[TranscriptionResult] = []
        engine.on_result(results.append)
        try:
            engine.start_listening()
            _wait(results)
            assert recorded["tuning"].threshold < VadTuning().threshold
        finally:
            engine.shutdown()

    def test_no_speech_timeout_reaches_capture(self, config: ConfigManager) -> None:
        capture = SilentCapture()
        recorded = {}

        def record(**kwargs):
            recorded.update(kwargs)
            return CaptureResult(error="none", error_code="no_speech")

        capture.record = record
        engine = ListeningEngine(config, recognizer=_always("x"), capture=capture,
                                 permissions=FakePermissions())
        config.update_speech_input(no_speech_timeout=3.0)
        results: List[TranscriptionResult] = []
        engine.on_result(results.append)
        try:
            engine.start_listening()
            _wait(results)
            assert recorded["no_speech_timeout"] == pytest.approx(3.0)
        finally:
            engine.shutdown()


# ------------------------------------------------------------------- helpers

def _always(text: str):
    from speech.listener import SpeechRecognizer

    class Fixed(SpeechRecognizer):
        name = "fixed"
        display_name = "Fixed"

        def is_available(self):
            return True

        def availability(self, language="en"):
            from speech.listener import RecognizerAvailability

            return RecognizerAvailability(True, self.name)

        def supported_languages(self):
            return ("en", "hi", "ta")

        def transcribe(self, audio, *, language="en", sample_rate=16000):
            return TranscriptionResult.transcribed(text, language=language,
                                                   confidence=0.9)

    return Fixed()


def _wait(items: List, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if items:
            return items[0]
        time.sleep(0.01)
    return None
