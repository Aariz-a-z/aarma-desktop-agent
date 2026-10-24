"""The text-only intent benchmark, run as a test suite.

This is the measurement that needs no microphone, and it has to come first. A
user reported that most spoken commands fail. That has two possible causes with
opposite fixes, and feeding known-perfect text straight into the intent engine
separates them: if accuracy is already poor here, the speech layer is not the
primary problem and no work on the microphone will help.

The command set, the paraphrase groups and the spec-to-production name mapping
are data files under ``benchmark/``. The mapping is applied only when building
what we expect to see -- nothing in the runtime pipeline reads it, so the
benchmark measures production as it stands rather than a version renamed to
agree with the benchmark.

Commands whose spec intent has no production equivalent are excluded from the
denominator and asserted separately. Counting them as failures would blame the
engine for a capability nobody built.
"""

from __future__ import annotations

from typing import List

import pytest

from ai.intent import IntentEngine
from benchmark.harness import (
    NO_PRODUCTION_INTENT,
    BenchmarkSuite,
    TextResult,
    run_paraphrases,
    run_short_controls,
    run_text_only,
    score,
)


@pytest.fixture(scope="module")
def suite() -> BenchmarkSuite:
    return BenchmarkSuite()


@pytest.fixture
def engine(config) -> IntentEngine:
    return IntentEngine(config)


def _table(results: List[TextResult]) -> str:
    lines = [f"{'input':<42}{'normalized':<38}{'matched':<20}"
             f"{'expected':<20}{'verdict'}"]
    for result in results:
        lines.append(
            f"{result.text[:41]:<42}{result.normalized[:37]:<38}"
            f"{result.matched_intent[:19]:<20}"
            f"{(result.expected_intent or NO_PRODUCTION_INTENT)[:19]:<20}"
            f"{result.verdict}")
    return "\n".join(lines)


# ------------------------------------------------------------ the mapping

class TestMapping:
    """The translation table must be visible and expectation-side only."""

    def test_spec_names_resolve_to_production_names(self, suite) -> None:
        assert suite.production_intent("open_website") == "open_url"
        assert suite.production_intent("web_search") == "search_web"
        assert suite.production_intent("wake_greeting") == "greeting"

    def test_unmapped_names_pass_through(self, suite) -> None:
        assert suite.production_intent("get_time") == "get_time"

    def test_parameter_keys_are_renamed(self, suite) -> None:
        assert suite.production_parameters(
            "open_application", {"app": "chrome"}) == {"name": "chrome"}
        assert suite.production_parameters(
            "list_files", {"path": "Documents"}) == {"location": "Documents"}

    def test_open_url_expects_a_bare_shortcut_token(self, suite) -> None:
        """Read out of the code, not assumed.

        ``ai.intent._validate_url`` returns ``raw.lower()`` when the word is a
        key of ``actions.browser.SITE_SHORTCUTS``, and the handler looks the
        token up in that same table. So production carries ``youtube``, not
        ``youtube.com`` and not a full URL.
        """
        from actions.browser import SITE_SHORTCUTS

        assert "youtube" in SITE_SHORTCUTS and "google" in SITE_SHORTCUTS
        assert suite.production_parameters(
            "open_website", {"site": "youtube"}) == {"url": "youtube"}

    def test_every_command_maps_to_a_real_intent_or_is_flagged(self, suite) -> None:
        known = suite.known_intents()
        for command in suite.commands():
            if command.resolvable:
                assert command.intent in known, command.text
            else:
                assert NO_PRODUCTION_INTENT in command.mapping_note

    def test_the_runtime_pipeline_does_not_read_the_mapping(self) -> None:
        """Guards the one rule that keeps this a measurement, not a rewrite.

        Checked by parsing imports rather than grepping for the word: the
        dispatcher legitimately *mentions* the benchmark in comments explaining
        its dry-run flag, and a substring match would flag that as a violation
        while still missing an import written any other way.
        """
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        runtime = [root / "ai" / "intent.py", root / "ai" / "planner.py",
                   root / "core" / "engine.py", root / "speech" / "input_engine.py",
                   root / "actions" / "registry.py"]
        for path in runtime:
            source = path.read_text(encoding="utf-8")
            # No runtime module may load the expectation-side data files.
            assert "intent_mapping" not in source, path.name
            assert "commands.json" not in source, path.name

            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    assert not name.split(".")[0] == "benchmark", (
                        f"{path.name} imports the benchmark package")


# ---------------------------------------------------- the 20-command set

class TestCommandSet:
    def test_the_file_holds_exactly_twenty_commands(self, suite) -> None:
        assert len(suite.commands()) == 20

    def test_reports_accuracy_with_its_denominator(self, engine, suite,
                                                   capsys) -> None:
        """Not an assertion of a target -- a printed measurement."""
        results = run_text_only(engine, suite)
        summary = score(results)
        with capsys.disabled():
            print("\n\n=== TEXT-ONLY INTENT BENCHMARK ===")
            print(_table(results))
            print(f"\npassed {summary['passed']}/{summary['total']} "
                  f"({summary['accuracy'] * 100:.0f}%), "
                  f"{summary['excluded']} excluded as {NO_PRODUCTION_INTENT}")
        assert summary["total"] + summary["excluded"] == 20

    def test_no_command_crashes_the_engine(self, engine, suite) -> None:
        for result in run_text_only(engine, suite):
            assert isinstance(result.matched_intent, str) and result.matched_intent

    def test_excluded_commands_are_not_counted_as_failures(self, engine,
                                                           suite) -> None:
        results = run_text_only(engine, suite)
        summary = score(results)
        assert all(r.excluded for r in summary["excluded_rows"])
        assert not any(r.excluded for r in summary["failures"])


# -------------------------------------------------------------- paraphrases

class TestParaphrases:
    """Whether the engine understands meaning or only memorised sentences."""

    def test_reports_each_group(self, engine, suite, capsys) -> None:
        groups = run_paraphrases(engine, suite)
        with capsys.disabled():
            print("\n=== PARAPHRASE GROUPS ===")
            for name, results in groups.items():
                passed = sum(1 for r in results if r.passed)
                print(f"\n{name}: {passed}/{len(results)}")
                print(_table(results))
        assert groups

    def test_short_controls_are_recorded_not_asserted(self, engine, suite,
                                                      capsys) -> None:
        """"stop" and "cancel" have no production intent. Show what they do."""
        results = run_short_controls(engine, suite)
        with capsys.disabled():
            print("\n=== SHORT CONTROLS ===")
            print(_table(results))
        assert len(results) == 4


# ------------------------------------------------------------------ trace

class TestTraceRecord:
    """The trace has to distinguish "did not run" from "ran and found nothing"."""

    def test_unrun_stages_are_null_not_missing(self) -> None:
        from utils.trace import PipelineTrace

        record = PipelineTrace(mode="text").finalise().to_dict()
        for stage in ("audio", "vad", "stt", "normalization", "validation",
                      "intent", "action"):
            assert stage in record, stage
            assert record[stage] is None, stage

    def test_blames_the_first_wrong_stage_not_the_last(self) -> None:
        """A mishearing guarantees an intent miss. The bug is still the STT."""
        from utils.trace import IntentStage, PipelineTrace, SttStage

        record = PipelineTrace(
            mode="benchmark", expected_text="Open Chrome.",
            expected_intent="open_application",
            stt=SttStage(raw_text="open crumb", confidence=0.6),
            intent=IntentStage(name="unknown", parameters={}),
        ).finalise()
        assert record.failure_layer == "STT"
        assert "open crumb" in record.failure_detail

    def test_a_validation_reject_is_not_blamed_on_intent(self) -> None:
        from utils.trace import PipelineTrace, SttStage, ValidationStage

        record = PipelineTrace(
            mode="benchmark", expected_text="Help.", expected_intent="help",
            stt=SttStage(raw_text="Help."),
            validation=ValidationStage(result="REJECT", raw_confidence=0.19,
                                       gate_base=0.35, gate_effective=0.193,
                                       reason="raw_confidence 0.190 < 0.193"),
        ).finalise()
        assert record.failure_layer == "VALIDATION"

    def test_records_the_effective_gate_not_only_the_constant(self) -> None:
        """Without this the trace cannot explain any rejection."""
        from utils.trace import PipelineTrace, ValidationStage

        record = PipelineTrace(validation=ValidationStage(
            result="PASS", raw_confidence=0.58, gate_base=0.35,
            gate_effective=0.28,
            length_scale_input={"duration_sec": 1.42, "token_count": 4},
        )).finalise().to_dict()
        assert record["validation"]["gate_effective"] == 0.28
        assert record["validation"]["length_scale_input"]["token_count"] == 4

    def test_case_and_punctuation_are_not_scored_as_mishearing(self) -> None:
        from utils.trace import PipelineTrace, SttStage

        record = PipelineTrace(expected_text="Open Chrome.",
                               stt=SttStage(raw_text="open chrome"))
        assert record.stt_correct
        assert not record.stt_exact

    def test_round_trips_through_jsonl(self, tmp_path) -> None:
        from utils import trace as trace_module
        from utils.trace import PipelineTrace, read_all

        path = tmp_path / "pipeline_trace.jsonl"
        original = trace_module.trace_path
        trace_module.trace_path = lambda: path
        try:
            trace_module.write(PipelineTrace(mode="benchmark",
                                             expected_text="Help.").finalise())
            trace_module.write(PipelineTrace(mode="benchmark",
                                             expected_text="Hello.").finalise())
        finally:
            trace_module.trace_path = original
        records = read_all(path)
        assert len(records) == 2
        assert records[0]["expected_text"] == "Help."

    def test_a_write_failure_never_propagates(self, monkeypatch) -> None:
        from utils import trace as trace_module
        from utils.trace import PipelineTrace

        monkeypatch.setattr(trace_module, "trace_path",
                            lambda: (_ for _ in ()).throw(OSError("no disk")))
        assert trace_module.write(PipelineTrace()) is None


class TestIntentInstrumentation:
    def test_a_match_records_how_it_was_made(self, engine) -> None:
        assert engine.parse("what time is it").match_method == "pattern"

    def test_an_unmatched_utterance_records_no_method(self, engine) -> None:
        intent = engine.parse("xyzzy plugh frobnicate")
        assert intent.name == "unknown"
        assert intent.match_method == "none"
