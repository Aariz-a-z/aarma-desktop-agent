"""The live benchmark: non-execution, durability, and export integrity.

The single property everything else rests on is that **a benchmark run launches
nothing**. If one command opens Chrome, the run is not a measurement, it is an
accident with a log file. That promise is enforced in three independent places
and each is tested here:

1. ``AarmaEngine.execute(..., dry_run=True)`` returns at its resolution step,
   after every gate has been evaluated, before the handler is called.
2. ``AarmaEngine._invoke`` -- the only place in AARMA that calls
   ``spec.handler`` -- raises :class:`~core.engine.BenchmarkExecutionError` if
   it is ever entered with that flag set.
3. ``BenchmarkRun.export`` refuses to write the file unless every record
   carries ``executed: false``.

The tests also pin the thing that would silently invalidate everything: that
``dry_run`` defaults to False and that an ordinary command is completely
unaffected by its existence.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from actions.registry import ActionRequest
from benchmark.runner import (
    STATUS_SKIPPED,
    BenchmarkRun,
    ExportRefused,
    find_incomplete,
)
from core.engine import BenchmarkExecutionError


def _grant_all(engine) -> None:
    from security.permissions import Permission

    for permission in Permission:
        engine.permissions.grant(permission)


def _revoke_all(engine) -> None:
    from security.permissions import Permission

    for permission in Permission:
        engine.permissions.revoke(permission)


def _record(command_id: int, *, executed: bool = False, correct: bool = True,
            intent: str = "open_application", expected: str = "open_application",
            status: str = "completed", result: str = "SUCCESS",
            layer: str = "NONE", attempt: int = 1) -> Dict[str, Any]:
    return {
        "command_id": command_id, "attempt": attempt, "status": status,
        "excluded": False, "expected_intent": expected,
        "stt_correct": correct,
        "intent": {"name": intent, "parameters": {}},
        "action": {"selected": intent, "executed": executed},
        "final_result": result, "failure_layer": layer,
    }


# ------------------------------------------------------- nothing executes

class TestNothingExecutes:
    def test_benchmark_dispatch_resolves_chrome_and_launches_nothing(
            self, engine, monkeypatch) -> None:
        """The test named in the phase brief.

        ``open_application`` is the most dangerous entry in the benchmark set:
        it is the one command that would visibly start a program. The launcher
        is replaced with a tripwire so that a regression fails the test rather
        than opening a browser on someone's desktop.
        """
        launched: list = []
        monkeypatch.setattr("actions.applications.launch_entry",
                            lambda *a, **k: launched.append(a) or True,
                            raising=False)
        monkeypatch.setattr("subprocess.Popen",
                            lambda *a, **k: pytest.fail(
                                f"benchmark spawned a process: {a}"))
        monkeypatch.setattr("os.startfile",
                            lambda *a, **k: pytest.fail(
                                f"benchmark opened a file: {a}"), raising=False)

        _grant_all(engine)
        response = engine.execute(
            ActionRequest(action="open_application",
                          parameters={"name": "chrome"}, source="benchmark"),
            dry_run=True)

        resolution = response.data["resolution"]
        assert response.ok
        assert resolution["selected"] == "open_application"
        assert resolution["risk_level"] in ("Low", "Medium", "High")
        assert resolution["executed"] is False
        assert launched == []

    def test_the_executor_raises_if_it_is_ever_entered(self, engine) -> None:
        """The tripwire. A silent no-op would hide the bug it exists to catch."""
        spec = engine.registry.require("get_time")
        with pytest.raises(BenchmarkExecutionError, match="get_time"):
            engine._invoke(spec, {}, "benchmark", dry_run=True)

    def test_resolution_still_evaluates_every_gate(self, engine) -> None:
        """Permissions are checked and refused, not waved through."""
        _revoke_all(engine)
        response = engine.execute(
            ActionRequest(action="take_screenshot", parameters={},
                          source="benchmark"),
            dry_run=True)
        assert not response.ok
        assert "resolution" not in (response.data or {})

    def test_no_confirmation_token_is_left_behind(self, engine) -> None:
        """A parked confirmation would swallow the user's next yes or no."""
        _grant_all(engine)
        before = engine.pending_confirmation_token
        for spec in engine.registry.all():
            if spec.risk_label != "High":
                continue
            engine.execute(ActionRequest(action=spec.name, parameters={},
                                         source="benchmark"), dry_run=True)
        assert engine.pending_confirmation_token == before

    def test_handle_text_dry_run_writes_no_transcript_or_audit(
            self, engine) -> None:
        _grant_all(engine)
        session = engine.session_id
        before_messages = len(engine.database.recent_messages(limit=200, session_id=session))
        before_actions = len(engine.database.recent_actions(limit=200))

        response = engine.handle_text("what time is it", source="benchmark",
                                      dry_run=True)
        assert response.ok

        assert len(engine.database.recent_messages(limit=200, session_id=session)) == before_messages
        assert len(engine.database.recent_actions(limit=200)) == before_actions


class TestLivePathIsUnchanged:
    """``dry_run`` must be invisible to every existing caller."""

    def test_it_defaults_to_false_everywhere(self) -> None:
        import inspect

        from core.engine import AarmaEngine

        for name in ("execute", "handle_text", "_route", "_invoke"):
            signature = inspect.signature(getattr(AarmaEngine, name))
            assert signature.parameters["dry_run"].default is False, name

    def test_an_ordinary_command_still_executes(self, engine) -> None:
        _grant_all(engine)
        response = engine.handle_text("what time is it")
        assert response.ok
        assert "[dry run]" not in response.text

    def test_an_ordinary_command_is_still_transcribed_and_audited(
            self, engine) -> None:
        _grant_all(engine)
        before = len(engine.database.recent_actions(limit=200))
        engine.handle_text("what time is it")
        assert len(engine.database.recent_actions(limit=200)) > before


# --------------------------------------------------------------- the run

class TestRunDurability:
    def test_each_attempt_is_on_disk_immediately(self) -> None:
        """If the app dies at command 17, sixteen must survive."""
        run = BenchmarkRun(run_id="20260101-000000")
        for index in range(1, 4):
            run.append(_record(index))
        assert run.traces_path.exists()
        reloaded = BenchmarkRun.load(run.run_id)
        assert len(reloaded.records) == 3

    def test_retries_are_all_kept_and_the_last_one_scores(self) -> None:
        run = BenchmarkRun(run_id="20260101-000001")
        run.append(_record(3, correct=False, attempt=1))
        run.append(_record(3, correct=False, attempt=2))
        run.append(_record(3, correct=True, attempt=3))
        assert len(run.attempts_for(3)) == 3      # Drift stays visible.
        finals = run.final_attempts()
        assert len(finals) == 1 and finals[0]["attempt"] == 3
        assert run.summary()["total_attempts"] == 3

    def test_next_attempt_counts_from_what_is_on_disk(self) -> None:
        run = BenchmarkRun(run_id="20260101-000002")
        assert run.next_attempt(5) == 1
        run.append(_record(5))
        assert run.next_attempt(5) == 2

    def test_skipped_commands_leave_every_denominator_alone(self) -> None:
        run = BenchmarkRun(run_id="20260101-000003")
        run.append(_record(1))
        run.append(_record(2, status=STATUS_SKIPPED))
        summary = run.summary()
        assert summary["scored_commands"] == 1
        assert summary["skipped"] == 1
        assert summary["stt_accuracy"]["total"] == 1

    def test_an_incomplete_run_is_offered_for_resume(self) -> None:
        run = BenchmarkRun(run_id="20260101-000004")
        for index in range(1, 17):
            run.append(_record(index))
        found = find_incomplete(expected=20)
        assert found is not None and found.run_id == run.run_id
        assert len(found.completed_ids()) == 16

    def test_a_finished_run_is_not_offered_for_resume(self) -> None:
        run = BenchmarkRun(run_id="20260101-000005")
        for index in range(1, 21):
            run.append(_record(index))
        run.export()
        assert find_incomplete(expected=20) is None


class TestExportIntegrity:
    def test_refuses_to_write_if_anything_executed(self) -> None:
        run = BenchmarkRun(run_id="20260101-000006")
        run.append(_record(1))
        run.append(_record(2, executed=True))
        with pytest.raises(ExportRefused, match="executed=false"):
            run.export()
        assert not run.export_path.exists()

    def test_a_missing_executed_flag_is_also_refused(self) -> None:
        """Absent is not the same as False, and must not be treated as it."""
        run = BenchmarkRun(run_id="20260101-000007")
        record = _record(1)
        record["action"] = {"selected": "get_time"}
        run.append(record)
        with pytest.raises(ExportRefused):
            run.export()

    def test_a_clean_run_exports_with_metadata_and_mapping(self) -> None:
        run = BenchmarkRun(run_id="20260101-000008")
        run.metadata = {"stt_model": "small", "gate_base": 0.35}
        run.append(_record(1))
        path = run.export(mapping={"intents": {"web_search": "search_web"}})
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["run_id"] == run.run_id
        assert payload["metadata"]["stt_model"] == "small"
        assert payload["intent_mapping"]["intents"]["web_search"] == "search_web"
        assert payload["records"][0]["action"]["executed"] is False

    def test_the_three_rates_are_reported_separately(self) -> None:
        """Never merged into one number."""
        run = BenchmarkRun(run_id="20260101-000009")
        run.append(_record(1, correct=True, result="SUCCESS"))
        run.append(_record(2, correct=True, intent="unknown",
                           result="FAILED", layer="INTENT"))
        run.append(_record(3, correct=False, result="FAILED", layer="STT"))
        summary = run.summary()
        assert summary["stt_accuracy"] == {"passed": 2, "total": 3}
        assert summary["intent_accuracy_on_correct_transcripts"] == {
            "passed": 1, "total": 2}
        assert summary["end_to_end_resolution"] == {"passed": 1, "total": 3}
        assert summary["failure_layers"] == {"INTENT": 1, "STT": 1}

    def test_intent_accuracy_ignores_commands_the_recogniser_got_wrong(
            self) -> None:
        """A mishearing must not be counted against the intent engine."""
        run = BenchmarkRun(run_id="20260101-000010")
        run.append(_record(1, correct=False, intent="unknown",
                           result="FAILED", layer="STT"))
        assert run.summary()["intent_accuracy_on_correct_transcripts"]["total"] == 0

    def test_audio_can_be_deleted_without_touching_the_records(self) -> None:
        run = BenchmarkRun(run_id="20260101-000011")
        run.append(_record(1))
        run.prepare()
        (run.directory / "01_01.wav").write_bytes(b"RIFF")
        assert run.delete_audio() == 1
        assert run.traces_path.exists()
        assert len(BenchmarkRun.load(run.run_id).records) == 1


class TestRecordShape:
    def test_a_skipped_record_is_marked_and_carries_executed_false(self) -> None:
        from benchmark.harness import BenchmarkSuite
        from ui.voice_benchmark import skipped_record

        command = BenchmarkSuite().commands()[0]
        record = skipped_record(command, attempt=1)
        assert record["status"] == STATUS_SKIPPED
        assert record["action"]["executed"] is False
        assert record["command_id"] == command.id

    def test_the_watchdog_is_longer_than_the_no_speech_timeout(self,
                                                               config) -> None:
        """Otherwise the watchdog would cut off normal recordings."""
        from ui.voice_benchmark import RECORDING_WATCHDOG_SECONDS

        assert (RECORDING_WATCHDOG_SECONDS
                > config.config.speech_input.no_speech_timeout)


class TestBenchmarkAudioIsOptIn:
    def test_listen_once_writes_nothing_unless_a_path_is_given(self,
                                                               config) -> None:
        """Normal conversation must persist no audio, as before."""
        import inspect

        from speech.input_engine import ListeningEngine

        signature = inspect.signature(ListeningEngine.listen_once)
        assert signature.parameters["audio_path"].default is None

    def test_the_save_debug_audio_setting_is_untouched_by_the_benchmark(
            self, config) -> None:
        assert config.config.speech_input.save_debug_audio is False
