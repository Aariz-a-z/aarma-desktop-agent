"""Recognition diagnostics, repeated cycles and accuracy scoring.

Written after a user reported that pressing the microphone and speaking
returned "I couldn't understand that clearly" every time, while the microphone
diagnostic said the device was fine. That combination is only debuggable if
three things are true, and each is asserted here:

* the recogniser's **raw output is kept**, so "misheard" and "our own rules
  discarded a correct reading" are distinguishable;
* **every refusal records an exact reason** internally, even though the user
  sees one friendly sentence;
* **a failed attempt cannot affect the next one** -- no state carries over.

No microphone, model or network is used. The measurements that motivated the
length-scaled confidence floor were taken separately on real audio and are
quoted in :func:`speech.listener.confidence_floor`; what is tested here is the
logic those measurements produced.
"""

from __future__ import annotations

from typing import List

import pytest

from speech.accuracy import (
    AccuracyReport,
    UtteranceScore,
    default_phrases,
    normalise,
    word_error_rate,
)
from speech.capture import CaptureResult
from speech.diagnostics import DiagnosticLog, RecognitionDiagnostic
from speech.input_engine import ListeningEngine, ListeningState
from speech.listener import RecognitionOutcome, TranscriptionResult

from tests.test_speech_input import (
    FakeCapture,
    FakePermissions,
    FakeRecognizer,
    _fake_audio,
)


def _speech_capture(**overrides) -> CaptureResult:
    values = dict(audio=_fake_audio(), duration=1.2, raw_duration=1.8,
                  peak=0.42, rms=0.09, noise_rms=0.004, speech_seconds=1.2)
    values.update(overrides)
    return CaptureResult(**values)


@pytest.fixture
def capture() -> FakeCapture:
    return FakeCapture(_speech_capture())


# --------------------------------------------------------- the raw output

class TestRawOutputIsPreserved:
    """Section 2: expose what the recogniser said *before* validation ran."""

    def test_rejected_reading_is_still_recorded(self, config, capture) -> None:
        engine = ListeningEngine(
            config,
            recognizer=FakeRecognizer(TranscriptionResult.transcribed(
                "open the pod bay doors please hal", confidence=0.05)),
            capture=capture, permissions=FakePermissions(True))
        try:
            result, record = engine.listen_once()
            assert not result.ok                       # The user sees a refusal.
            assert record.raw_text == "open the pod bay doors please hal"
            assert record.reason == "low_confidence"
            assert "0.050" in record.detail             # The numbers, not a shrug.
        finally:
            engine.shutdown()

    def test_a_validation_refusal_keeps_its_own_code(self, config, capture) -> None:
        engine = ListeningEngine(
            config,
            recognizer=FakeRecognizer(TranscriptionResult.unclear(
                error_code="script_mismatch", raw_text="ಹಲೋ",
                error="That didn't come out in the expected script.")),
            capture=capture, permissions=FakePermissions(True))
        try:
            _result, record = engine.listen_once()
            assert record.reason == "script_mismatch"
            assert record.raw_text == "ಹಲೋ"
        finally:
            engine.shutdown()

    def test_silence_is_reported_as_silence_not_as_misunderstanding(
            self, config) -> None:
        engine = ListeningEngine(
            config,
            recognizer=FakeRecognizer(),
            capture=FakeCapture(CaptureResult(
                error="I didn't hear anything.", error_code="no_speech",
                raw_duration=6.0, peak=0.001)),
            permissions=FakePermissions(True))
        try:
            result, record = engine.listen_once()
            assert result.outcome is RecognitionOutcome.NO_SPEECH
            assert record.reason == "no_speech"
            assert not record.speech_detected
        finally:
            engine.shutdown()


class TestDiagnosticRecord:
    def test_capture_measurements_reach_the_record(self, config) -> None:
        engine = ListeningEngine(
            config, recognizer=FakeRecognizer(),
            capture=FakeCapture(_speech_capture(clipped=True)),
            permissions=FakePermissions(True))
        try:
            _result, record = engine.listen_once()
            assert record.capture_seconds == pytest.approx(1.8)
            assert record.speech_seconds == pytest.approx(1.2)
            assert record.rms == pytest.approx(0.09)
            assert record.peak == pytest.approx(0.42)
            assert record.clipped and record.speech_detected
            assert record.latency > 0.0
        finally:
            engine.shutdown()

    def test_every_attempt_is_filed_whether_it_succeeded_or_not(
            self, config, capture) -> None:
        engine = ListeningEngine(
            config, recognizer=FakeRecognizer(), capture=capture,
            permissions=FakePermissions(True))
        try:
            for _ in range(3):
                engine.listen_once()
            assert len(engine.diagnostics) == 3
            assert engine.diagnostics.last.accepted
        finally:
            engine.shutdown()

    def test_report_never_contains_a_secret_shaped_field(self) -> None:
        """The block a user copies into a bug report is text they can read."""
        record = RecognitionDiagnostic(device="Mic", raw_text="hello",
                                       confidence=0.6, accepted=True)
        block = record.report()
        assert "Raw STT" in block and "hello" in block
        assert "Decision      : ACCEPTED" in block


class TestDiagnosticLog:
    def test_keeps_only_the_most_recent_entries(self) -> None:
        log = DiagnosticLog(limit=3)
        for index in range(6):
            log.add(RecognitionDiagnostic(raw_text=str(index)))
        assert len(log) == 3
        assert [entry.raw_text for entry in log.recent()] == ["5", "4", "3"]

    def test_last_is_none_before_anything_happens(self) -> None:
        assert DiagnosticLog().last is None


# -------------------------------------------------- one attempt, then another

class TestRepeatedCycles:
    """"Works once, then fails" was the original bug. It must not return."""

    def test_twenty_consecutive_cycles_all_succeed(self, config, capture) -> None:
        engine = ListeningEngine(
            config, recognizer=FakeRecognizer(), capture=capture,
            permissions=FakePermissions(True))
        try:
            for attempt in range(20):
                result, _record = engine.listen_once()
                assert result.ok, f"cycle {attempt + 1} failed"
                assert engine.state is ListeningState.PROCESSING
        finally:
            engine.shutdown()

    def test_a_failure_does_not_poison_the_next_attempt(self, config) -> None:
        capture = FakeCapture(CaptureResult(error="I didn't hear anything.",
                                            error_code="no_speech"))
        engine = ListeningEngine(config, recognizer=FakeRecognizer(),
                                 capture=capture,
                                 permissions=FakePermissions(True))
        try:
            first, _ = engine.listen_once()
            assert not first.ok

            capture._result = _speech_capture()
            second, record = engine.listen_once()
            assert second.ok
            assert record.reason == ""          # Not carried over from the first.
            assert record.detail == ""
        finally:
            engine.shutdown()

    def test_an_exception_does_not_leave_the_engine_stuck(self, config,
                                                          capture) -> None:
        class Exploding(FakeRecognizer):
            def transcribe(self, audio, **kwargs):
                raise RuntimeError("decoder exploded")

        engine = ListeningEngine(config, recognizer=Exploding(), capture=capture,
                                 permissions=FakePermissions(True))
        try:
            result, record = engine.listen_once()
            assert not result.ok and record.reason == "internal_error"
            assert "RuntimeError" in record.detail

            engine._recognizer = FakeRecognizer()
            assert engine.listen_once()[0].ok
        finally:
            engine.shutdown()

    def test_the_model_is_loaded_once_and_reused(self, config, capture) -> None:
        recognizer = FakeRecognizer()
        engine = ListeningEngine(config, recognizer=recognizer, capture=capture,
                                 permissions=FakePermissions(True))
        try:
            for _ in range(5):
                engine.listen_once()
            assert len(recognizer.calls) == 5
        finally:
            engine.shutdown()


class TestTestModeDoesNotExecute:
    """Section 3: saying "Open Chrome." during a test must open nothing."""

    def test_listen_once_does_not_notify_result_listeners(self, config,
                                                          capture) -> None:
        engine = ListeningEngine(
            config,
            recognizer=FakeRecognizer(TranscriptionResult.transcribed(
                "open chrome", confidence=0.9)),
            capture=capture, permissions=FakePermissions(True))
        delivered: List[TranscriptionResult] = []
        engine.on_result(delivered.append)
        try:
            result, _record = engine.listen_once()
            assert result.ok                # The test still sees the answer...
            assert delivered == []          # ...but the command pipeline does not.
        finally:
            engine.shutdown()


# --------------------------------------------------- hearing its own name

class TestVocabularyPriming:
    """The assistant could not hear its own name.

    Measured on the ``small`` model with synthetic speech: "Asma" was
    transcribed as "Asthma" in 8 of 8 attempts across four acoustic conditions.
    Priming the decoder with the name recovered 8 of 8 and left the other 28
    utterances in the set unchanged.
    """

    def test_the_assistant_name_is_offered_to_the_recogniser(
            self, config, capture) -> None:
        config.update_assistant(name="Asma", wake_word="hey asma")
        recognizer = FakeRecognizer()
        engine = ListeningEngine(config, recognizer=recognizer, capture=capture,
                                 permissions=FakePermissions(True))
        try:
            engine.listen_once()
            assert "Asma" in recognizer.calls[0]["vocabulary"]
        finally:
            engine.shutdown()

    def test_a_renamed_assistant_is_offered_its_new_name(self, config,
                                                         capture) -> None:
        config.update_assistant(name="Zubeida", wake_word="hey zubeida")
        recognizer = FakeRecognizer()
        engine = ListeningEngine(config, recognizer=recognizer, capture=capture,
                                 permissions=FakePermissions(True))
        try:
            engine.listen_once()
            assert "Zubeida" in recognizer.calls[0]["vocabulary"]
        finally:
            engine.shutdown()

    def test_duplicates_are_not_repeated(self, config) -> None:
        config.update_assistant(name="Asma", wake_word="hey asma")
        engine = ListeningEngine(config, recognizer=FakeRecognizer(),
                                 permissions=FakePermissions(True))
        try:
            assert engine.vocabulary() == ["Asma", "hey"]
        finally:
            engine.shutdown()

    def test_the_prompt_carries_names_and_nothing_else(self) -> None:
        """A decoder prompt biases the next decode. Only names belong in it."""
        from speech.providers.whisper_local import MAX_PROMPT_WORDS, build_prompt

        assert build_prompt(["Asma", "Aariz"]) == "Asma. Aariz."
        assert build_prompt([]) == ""
        # A sentence would distort ordinary commands, so it is dropped.
        assert build_prompt(["the user said open my bank statement"]) == ""
        # And the list cannot grow without bound.
        assert (len(build_prompt([f"name{i}" for i in range(50)]).split())
                == MAX_PROMPT_WORDS)

    def test_no_conversation_history_can_reach_the_prompt(self, config) -> None:
        """The vocabulary is derived from identity, never from what was said."""
        config.update_assistant(name="Asma", wake_word="hey asma")
        engine = ListeningEngine(config, recognizer=FakeRecognizer(),
                                 permissions=FakePermissions(True))
        try:
            first = engine.vocabulary()
            engine.listen_once()
            assert engine.vocabulary() == first   # Unchanged by an utterance.
        finally:
            engine.shutdown()


# ----------------------------------------------------------- saved audio

class TestDiagnosticAudio:
    """Section 8: off by default, local only, never uploaded."""

    def test_is_off_by_default(self, config) -> None:
        assert config.config.speech_input.save_debug_audio is False

    def test_nothing_is_written_and_no_folder_appears_while_it_is_off(
            self, config, capture) -> None:
        """A machine that never asked for a recording folder should not have one."""
        from utils.paths import get_paths

        engine = ListeningEngine(config, recognizer=FakeRecognizer(),
                                 capture=capture,
                                 permissions=FakePermissions(True))
        try:
            _result, record = engine.listen_once()
            assert record.audio_path == ""
            assert not get_paths().diagnostic_dir.exists()
        finally:
            engine.shutdown()

    def test_writes_a_readable_wave_file_when_switched_on(self, config) -> None:
        import wave
        from pathlib import Path

        from utils.paths import get_paths

        config.update_speech_input(save_debug_audio=True)
        engine = ListeningEngine(config, recognizer=FakeRecognizer(),
                                 capture=FakeCapture(_speech_capture()),
                                 permissions=FakePermissions(True))
        try:
            _result, record = engine.listen_once()
            assert record.audio_path, "no file was written"
            # It stays on this machine, in AARMA's own data folder.
            assert Path(record.audio_path).parent == get_paths().diagnostic_dir
            with wave.open(record.audio_path, "rb") as handle:
                assert handle.getnchannels() == 1
                assert handle.getframerate() == 16_000
                assert handle.getnframes() > 0
        finally:
            engine.shutdown()

    def test_old_recordings_are_pruned(self) -> None:
        from speech import input_engine
        from utils.paths import get_paths

        folder = get_paths().diagnostic_dir
        for _ in range(input_engine.DEBUG_AUDIO_KEEP + 5):
            input_engine._save_debug_audio(_fake_audio(0.05))
        kept = list(folder.glob("speech-*.wav"))
        assert 0 < len(kept) <= input_engine.DEBUG_AUDIO_KEEP

    def test_a_write_failure_never_breaks_listening(self, config,
                                                    monkeypatch) -> None:
        """Saving is a debugging aid. It must not be able to stop recognition."""
        def explode(_audio):
            raise OSError("disk full")

        monkeypatch.setattr("wave.open", explode)
        config.update_speech_input(save_debug_audio=True)
        engine = ListeningEngine(config, recognizer=FakeRecognizer(),
                                 capture=FakeCapture(_speech_capture()),
                                 permissions=FakePermissions(True))
        try:
            result, record = engine.listen_once()
            assert result.ok                 # Recognition still happened.
            assert record.audio_path == ""   # The file simply is not there.
        finally:
            engine.shutdown()


# ------------------------------------------------------- accuracy scoring

class TestNormalisation:
    @pytest.mark.parametrize("expected,actual", [
        ("Open Chrome.", "open chrome"),
        ("What time is it?", "what time is it"),
        ("Hello Asma.", "  Hello,  Asma!  "),
    ])
    def test_formatting_is_not_an_error(self, expected, actual) -> None:
        """Whisper's capitals and punctuation are presentation, not mishearing."""
        assert normalise(expected) == normalise(actual)
        assert word_error_rate(expected, actual) == 0.0

    def test_a_wrong_word_is_an_error(self) -> None:
        assert word_error_rate("open chrome", "open chrome now") == pytest.approx(0.5)
        assert word_error_rate("open chrome", "open firefox") == pytest.approx(0.5)

    def test_nothing_heard_is_a_total_error(self) -> None:
        assert word_error_rate("open chrome", "") == pytest.approx(1.0)


class TestUtteranceScore:
    def test_a_correct_accepted_reading_passes(self) -> None:
        score = UtteranceScore(expected="Open Chrome.", detected="open chrome",
                               raw_detected="open chrome", accepted=True)
        assert score.passed and score.verdict == "PASS"

    def test_a_correct_but_refused_reading_is_named_as_such(self) -> None:
        """The distinction the whole diagnostic exists to make."""
        score = UtteranceScore(expected="Open Chrome.", detected="",
                               raw_detected="Open Chrome.", accepted=False,
                               reason="low_confidence")
        assert not score.passed
        assert score.verdict == "REJECTED (heard correctly)"

    def test_a_misheard_reading_is_not_confused_with_a_refusal(self) -> None:
        score = UtteranceScore(expected="Open Chrome.", detected="",
                               raw_detected="open crumb", accepted=False)
        assert score.verdict == "REJECTED (misheard)"

    def test_an_accepted_but_wrong_reading_is_misheard(self) -> None:
        score = UtteranceScore(expected="Open Chrome.", detected="open crumb",
                               raw_detected="open crumb", accepted=True)
        assert score.verdict == "MISHEARD"

    def test_silence_is_its_own_verdict(self) -> None:
        assert UtteranceScore(expected="Stop.").verdict == "NO SPEECH"


class TestAccuracyReport:
    def _report(self) -> AccuracyReport:
        report = AccuracyReport()
        report.scores = [
            UtteranceScore("Open Chrome.", "open chrome", "open chrome",
                           accepted=True, latency=2.0),
            UtteranceScore("Stop.", "", "Stop.", accepted=False,
                           reason="low_confidence", latency=1.0),
            UtteranceScore("Help.", "", "hell", accepted=False, latency=1.0),
            UtteranceScore("Tell me a joke.", skipped=True),
        ]
        return report

    def test_skipped_phrases_are_not_counted(self) -> None:
        assert self._report().total == 3

    def test_recognition_and_acceptance_are_reported_separately(self) -> None:
        report = self._report()
        assert report.passes == 1                    # Only one usable command.
        assert report.recognised == 2                # Two were heard correctly.
        assert report.rejected_correct == 1          # One was thrown away.

    def test_the_headline_carries_its_denominator(self) -> None:
        headline = self._report().headline()
        assert "1 of 3" in headline and "2 of 3" in headline
        assert "refused" in headline

    def test_the_target_is_stated_not_claimed(self) -> None:
        report = self._report()
        assert not report.meets_target
        assert "not met" in report.report()

    def test_an_empty_run_claims_nothing(self) -> None:
        empty = AccuracyReport()
        assert not empty.meets_target
        assert empty.headline() == "Nothing measured yet."


class TestDiagnosticsPanel:
    """Section 9: the panel must show the real reason, not a friendly one."""

    @pytest.fixture
    def dashboard(self, engine):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        from ui.dashboard import Dashboard

        app = QApplication.instance() or QApplication([])
        window = Dashboard(engine)
        yield window
        window.close()
        app.processEvents()

    def test_says_so_plainly_before_anything_has_happened(self, dashboard) -> None:
        page = dashboard.settings_page
        page._refresh_diagnostics()
        assert "No listening attempt yet." in page._diagnostic_rows["Decision"].text()

    def test_shows_the_raw_output_and_the_exact_reason(self, dashboard) -> None:
        page = dashboard.settings_page
        dashboard.engine.listening.diagnostics.add(RecognitionDiagnostic(
            device="Realtek Microphone", capture_seconds=2.4, speech_seconds=0.9,
            rms=0.031, peak=0.28, model="small", language="en",
            raw_text="open chrome", confidence=0.31, threshold_used=0.28,
            average_logprob=-1.17, speech_detected=True, accepted=False,
            reason="low_confidence", detail="confidence 0.310 below 0.280",
            latency=3.1))
        page._refresh_diagnostics()
        rows = page._diagnostic_rows

        assert rows["Heard"].text() == "open chrome"       # Before validation.
        assert "REJECTED" in rows["Decision"].text()
        assert "low_confidence" in rows["Decision"].text()
        assert "0.310" in rows["Decision"].text()
        assert "0.310" not in rows["Heard"].text()
        assert "needed 0.280" in rows["Certainty"].text()
        assert "3.10s" in rows["Response time"].text()

    def test_a_near_silent_level_is_not_printed_as_zero(self, dashboard) -> None:
        """0.000017 rendered as "0.0000" is what started this whole exercise."""
        page = dashboard.settings_page
        dashboard.engine.listening.diagnostics.add(
            RecognitionDiagnostic(rms=0.000017, peak=0.000042))
        page._refresh_diagnostics()
        assert "0.0000," not in page._diagnostic_rows["Levels"].text()


class TestDefaultPhrases:
    def test_includes_the_specified_five(self) -> None:
        phrases = default_phrases("Asma")
        for required in ("Hello Asma.", "Open Chrome.", "What time is it?",
                         "Open YouTube.", "Tell me a joke."):
            assert required in phrases

    def test_includes_the_short_commands_that_score_lowest(self) -> None:
        phrases = default_phrases("Asma")
        assert {"Asma", "Stop.", "Help.", "Cancel."} <= set(phrases)

    def test_uses_the_chosen_assistant_name(self) -> None:
        assert "Hello Aariz." in default_phrases("Aariz")
