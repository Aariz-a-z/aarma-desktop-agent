"""Enrollment's recording gate, and who owns the microphone while it runs.

The bug these exist to prevent came back with a very specific shape: a recording
that the rest of AARMA would happily transcribe was refused by enrollment, with
"too quiet", forever. The cause was an absolute peak threshold of 0.02. Across
the 320 wake-phrase recordings already in this machine's corpus the quietest
usable clip peaks at 0.0141, and ambient clips containing no speech at all reach
0.47 — so absolute level does not separate speech from silence at all. It only
measures how much gain the sound card applies.

Every threshold is now relative to the room measured at the start of the
session, and every bound below is checked against the *minimum* observed in
those 320 real recordings. If someone tightens one, these fail.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest

from speech.wakeword_enrollment import (
    CLIPPED_PEAK,
    MIN_PEAK_OVER_FLOOR,
    MIN_SNR_DB,
    MIN_SPEECH_SECONDS,
    SILENT_PEAK,
    ClipQuality,
    EnrollmentRecorder,
    assess_clip,
)

#: Measured on this machine: a silent room floors at about 1.6e-5 to 6e-5 RMS.
ROOM_FLOOR = 3.0e-5

#: The worst case in the corpus of 320 real recordings, per distance.
#: (peak, rms, speech_seconds, snr_db)
WORST_REAL = {
    "near-mic": (0.0833, 0.0110, 1.10, 24.0),
    "0.5m": (0.0291, 0.0040, 1.08, 15.0),
    "1m": (0.0715, 0.0060, 1.05, 18.0),
    "2m": (0.0476, 0.0035, 0.98, 11.0),
    "3m": (0.0141, 0.0016, 0.92, 7.4),
}


def _measure(peak: float, rms: float, speech: float,
             snr: Optional[float], vad: bool = True) -> dict:
    return {"peak": peak, "rms": rms, "speech_seconds": speech,
            "snr_db": snr, "vad_triggered": vad, "duration_sec": 3.2}


def _audio(level: float = 0.01, seconds: float = 3.2):
    return np.full(int(seconds * 16000), level, dtype=np.float32)


class TestRealRecordingsAreAccepted:
    """The regression that matters: nothing the corpus contains may be refused."""

    @pytest.mark.parametrize("distance", sorted(WORST_REAL))
    def test_the_worst_real_recording_at_each_distance(self, distance: str) -> None:
        peak, rms, speech, snr = WORST_REAL[distance]
        quality = assess_clip(_audio(), _measure(peak, rms, speech, snr), ROOM_FLOOR)
        assert quality.accepted, f"{distance}: {quality.code} — {quality.reason}"
        assert quality.code == "ok"

    def test_the_level_that_used_to_be_rejected_is_accepted(self) -> None:
        # Below the old absolute 0.02 gate, but 500x the room floor: this is
        # ordinary speech on a low-gain microphone, not silence.
        quality = assess_clip(_audio(), _measure(0.015, 0.002, 1.1, 18.0), ROOM_FLOOR)
        assert quality.accepted
        assert quality.peak < 0.02

    def test_a_quiet_microphone_is_not_punished_for_its_gain(self) -> None:
        """Halving every level must not change the verdict.

        This is the whole point of measuring against the room rather than
        against a constant: the same voice through a quieter sound card is the
        same voice.
        """
        loud = assess_clip(_audio(), _measure(0.12, 0.011, 1.4, 24.0), ROOM_FLOOR)
        quiet = assess_clip(_audio(), _measure(0.012, 0.0011, 1.4, 24.0),
                            ROOM_FLOOR / 10.0)
        assert loud.accepted and quiet.accepted


class TestBadRecordingsAreStillRefused:
    def test_digital_silence(self) -> None:
        quality = assess_clip(np.zeros(51200, dtype=np.float32),
                              _measure(0.0, 0.0, 0.0, None, vad=False), ROOM_FLOOR)
        assert not quality.accepted
        assert quality.code == "silent"
        assert quality.all_zero is True

    def test_a_muted_microphone(self) -> None:
        quality = assess_clip(_audio(1e-5), _measure(0.0001, 0.00002, 0.0, None,
                                                    vad=False), ROOM_FLOOR)
        assert not quality.accepted
        assert quality.code == "silent"
        assert "muted" in quality.advice.lower()

    def test_room_noise_with_no_phrase_in_it(self) -> None:
        quality = assess_clip(_audio(0.05),
                              _measure(0.19, 0.005, 0.0, None, vad=False), ROOM_FLOOR)
        assert not quality.accepted
        assert quality.code == "no_speech"

    def test_half_a_phrase(self) -> None:
        quality = assess_clip(_audio(), _measure(0.10, 0.01, 0.20, 20.0), ROOM_FLOOR)
        assert not quality.accepted
        assert quality.code == "too_short"

    def test_speech_buried_in_the_room(self) -> None:
        # Loud in absolute terms, but the room is louder still.
        quality = assess_clip(_audio(), _measure(0.30, 0.09, 1.2, 2.0), 0.05)
        assert not quality.accepted
        assert quality.code == "low_snr"

    def test_clipped_audio(self) -> None:
        clipped = np.full(51200, 1.0, dtype=np.float32)
        quality = assess_clip(clipped, _measure(1.0, 0.9, 1.5, 40.0), ROOM_FLOOR)
        assert not quality.accepted
        assert quality.code == "clipped"

    def test_no_audio_at_all(self) -> None:
        quality = assess_clip(np.zeros(0, dtype=np.float32),
                              _measure(0.0, 0.0, 0.0, None, vad=False), ROOM_FLOOR)
        assert not quality.accepted
        assert quality.code == "silent"


class TestThresholdsStayBelowWhatWasMeasured:
    """Guard rails: the bounds must remain looser than real recordings need."""

    def test_peak_over_floor(self) -> None:
        # Minimum observed across 320 real recordings was 14.8.
        assert MIN_PEAK_OVER_FLOOR <= 14.8

    def test_snr(self) -> None:
        assert MIN_SNR_DB <= 7.4          # minimum observed

    def test_speech_duration(self) -> None:
        assert MIN_SPEECH_SECONDS <= 0.92  # minimum observed

    def test_the_silence_floor_is_far_below_speech(self) -> None:
        # A silent room on this machine peaks around 0.0001; the quietest real
        # recording peaks at 0.0141. The floor must sit between them.
        assert 0.0002 < SILENT_PEAK < 0.0141

    def test_clipping_is_only_at_the_rail(self) -> None:
        assert CLIPPED_PEAK >= 0.95


class TestDiagnostics:
    def test_every_field_the_report_needs_is_present(self) -> None:
        quality = assess_clip(_audio(), _measure(0.12, 0.011, 1.4, 24.0), ROOM_FLOOR)
        line = quality.diagnostics()
        for field in ("sr=", "samples=", "dur=", "peak=", "rms=", "speech=",
                      "ratio=", "snr=", "peak/floor=", "accepted"):
            assert field in line, f"{field} missing from {line}"

    def test_a_rejection_names_its_reason(self) -> None:
        quality = assess_clip(_audio(), _measure(0.10, 0.01, 0.20, 20.0), ROOM_FLOOR)
        assert "rejected:too_short" in quality.diagnostics()

    def test_diagnostics_carry_no_audio(self) -> None:
        quality = assess_clip(_audio(), _measure(0.12, 0.011, 1.4, 24.0), ROOM_FLOOR)
        assert "array" not in quality.diagnostics()
        assert len(quality.diagnostics()) < 200

    def test_speech_ratio_is_reported(self) -> None:
        quality = assess_clip(_audio(seconds=3.2),
                              _measure(0.12, 0.011, 1.6, 24.0), ROOM_FLOOR)
        assert quality.speech_ratio == pytest.approx(0.5, abs=0.02)


class TestReadiness:
    def _recorder(self, floor: float) -> EnrollmentRecorder:
        recorder = EnrollmentRecorder.__new__(EnrollmentRecorder)
        recorder.floor_rms = floor
        return recorder

    def test_a_normal_room_is_ready(self) -> None:
        usable, why = self._recorder(3e-5).readiness()
        assert usable is True
        assert "ready" in why

    def test_a_dead_device_is_reported_before_prompting(self) -> None:
        usable, why = self._recorder(0.0).readiness()
        assert usable is False
        assert "nothing" in why

    def test_digital_silence_is_named_as_muting(self) -> None:
        usable, why = self._recorder(1e-9).readiness()
        assert usable is False
        assert "muted" in why

    def test_a_very_loud_room_is_refused(self) -> None:
        usable, why = self._recorder(0.2).readiness()
        assert usable is False
        assert "loud" in why


class TestOneFrontEndCannotDriftFromTheOther:
    def test_both_front_ends_call_the_same_capture(self) -> None:
        """Neither the console nor the dialog may judge a recording itself.

        Checked by source inspection rather than behaviour, because the failure
        being prevented is someone *adding* a second opinion later.
        """
        import inspect

        import tools.enrol_wakeword as cli
        import ui.wakeword_enrol as gui

        for module in (cli, gui):
            source = inspect.getsource(module)
            assert "recorder.capture(" in source or "_recorder.capture(" in source
            # The old absolute test must not reappear in either front end.
            assert "0.02" not in source
            assert "vad_triggered" not in source

    def test_the_clip_carries_the_verdict_rather_than_the_caller_deciding(self) -> None:
        quality = assess_clip(_audio(), _measure(0.12, 0.011, 1.4, 24.0), ROOM_FLOOR)
        assert isinstance(quality, ClipQuality)
        assert hasattr(quality, "accepted")
        assert hasattr(quality, "reason")
        assert hasattr(quality, "advice")


class TestMicrophoneOwnership:
    def test_pausing_the_wake_loop_releases_the_device_first(self, engine,
                                                             monkeypatch) -> None:
        """The detector must be stopped before anything else opens the mic."""
        from speech.wakeword import WakeWordDetector, WakeWordProvider, WakeWordRegistry

        events = []

        class _Detector(WakeWordDetector):
            name = "owns-the-mic"

            def __init__(self, **kwargs) -> None:
                self.running = False

            def is_available(self) -> bool:
                return True

            def supports_phrase(self, phrase: str) -> bool:
                return phrase.strip().lower() == "hey aariz"

            def start(self, phrase, callback) -> bool:
                events.append("open")
                self.running = True
                return True

            def stop(self) -> None:
                if self.running:
                    events.append("release")
                self.running = False

        registry = WakeWordRegistry()
        registry.register(WakeWordProvider(phrase="hey aariz",
                                           factory=lambda **kw: _Detector(**kw)))
        monkeypatch.setattr("core.engine._wake_registry", lambda: registry)

        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        assert engine.start_wake_loop() is True
        with engine.wake_paused():
            events.append("enrollment-opens-mic")
        events.append("done")

        assert events == ["open", "release", "enrollment-opens-mic", "open", "done"]
        # Released strictly before enrollment touched the device.
        assert events.index("release") < events.index("enrollment-opens-mic")
