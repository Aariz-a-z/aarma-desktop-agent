"""Run the text-only intent benchmark and print the tables.

Needs no microphone, no speech model and no network. It answers one question:
**how well does intent matching work when the transcription is perfect?**

If accuracy here is already poor, speech recognition is not the primary problem
and work on the microphone will not help. That is the whole reason this runs
before anything touches audio.

Usage::

    python tools/intent_benchmark.py
    python tools/intent_benchmark.py --trace     # also write logs/pipeline_trace.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.intent import IntentEngine  # noqa: E402
from benchmark.harness import (  # noqa: E402
    NO_PRODUCTION_INTENT,
    BenchmarkSuite,
    TextResult,
    run_paraphrases,
    run_short_controls,
    run_text_only,
    score,
)
from config.settings import load_config  # noqa: E402

COLUMNS = (("input", 40), ("normalized", 34), ("matched", 20),
           ("expected", 20), ("method", 9), ("verdict", 24))


def _header() -> str:
    line = "".join(f"{name:<{width}}" for name, width in COLUMNS)
    return line + "\n" + "-" * len(line)


def _row(result: TextResult) -> str:
    expected = result.expected_intent or NO_PRODUCTION_INTENT
    cells = (result.text, result.normalized, result.matched_intent,
             expected, result.match_method, result.verdict)
    return "".join(f"{str(cell)[:width - 1]:<{width}}"
                   for cell, (_name, width) in zip(cells, COLUMNS))


def _print_table(results: List[TextResult]) -> None:
    print(_header())
    for result in results:
        print(_row(result))


def _emit_traces(results: List[TextResult], mode: str) -> None:
    """Write one trace record per row, with the audio stages null."""
    from utils.trace import ActionStage, IntentStage, NormalizationStage, PipelineTrace
    from utils.trace import write as write_trace

    from actions.registry import build_default_registry
    from benchmark.harness import META_INTENTS

    registry = build_default_registry()
    path = None
    for result in results:
        spec = (registry.require(result.matched_intent)
                if result.matched_intent in registry else None)
        record = PipelineTrace(
            mode=mode,
            expected_text=result.text,
            expected_intent=result.expected_intent,
            expected_parameters=result.expected_parameters,
            # audio / vad / stt stay None: they genuinely did not run.
            normalization=NormalizationStage(normalized_text=result.normalized),
            intent=IntentStage(
                name=result.matched_intent,
                parameters=result.matched_parameters,
                match_method=result.match_method,
                match_score=result.match_score,
            ),
            action=ActionStage(
                selected=(spec.name if spec else
                          (result.matched_intent
                           if result.matched_intent in META_INTENTS else None)),
                risk_level=(spec.risk_label if spec else None),
                would_require_confirmation=(spec.risk_label == "High"
                                            if spec else False),
                executed=False,
            ),
        )
        path = write_trace(record.finalise()) or path
    if path:
        print(f"\nTraces written to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Text-only intent benchmark")
    parser.add_argument("--trace", action="store_true",
                        help="also append records to logs/pipeline_trace.jsonl")
    args = parser.parse_args()

    engine = IntentEngine(load_config())
    suite = BenchmarkSuite()

    print("=" * 100)
    print("TEXT-ONLY INTENT BENCHMARK — no microphone, no speech model")
    print("=" * 100)

    commands = run_text_only(engine, suite)
    print("\n## The 20-command set\n")
    _print_table(commands)
    summary = score(commands)
    print(f"\nIntent accuracy: {summary['passed']}/{summary['total']} "
          f"({summary['accuracy'] * 100:.0f}%)"
          + (f"   [{summary['excluded']} excluded as {NO_PRODUCTION_INTENT}]"
             if summary["excluded"] else ""))

    if summary["excluded_rows"]:
        print("\n### Excluded — no production intent exists\n")
        for result in summary["excluded_rows"]:
            print(f"  #{result.command_id}  {result.text:<34}"
                  f"{result.mapping_note}   (resolved to "
                  f"{result.matched_intent!r})")

    print("\n## Paraphrase groups\n")
    group_totals: Dict[str, str] = {}
    for name, results in run_paraphrases(engine, suite).items():
        group_score = score(results)
        group_totals[name] = (f"{group_score['passed']}/{group_score['total']}")
        print(f"\n{name}  —  {group_totals[name]}")
        _print_table(results)

    print("\n## Short controls\n")
    controls = run_short_controls(engine, suite)
    _print_table(controls)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"20-command intent accuracy : {summary['passed']}/{summary['total']}"
          f" ({summary['accuracy'] * 100:.0f}%)")
    for name, total in group_totals.items():
        print(f"  paraphrase {name:<28}: {total}")

    failures = summary["failures"]
    if failures:
        print(f"\nFailing commands ({len(failures)}):")
        for result in failures:
            print(f"  {result.text:<38} -> {result.matched_intent:<20}"
                  f"expected {result.expected_intent}")

    if args.trace:
        _emit_traces(commands, mode="text")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
