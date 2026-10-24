"""Re-analyse a recorded benchmark run offline. No microphone, no new speech.

A run was recorded against a configuration nobody intended (``tiny`` at gate
0.9). The audio survived, so most of that run is recoverable without asking
anyone to say twenty commands again. Three independent analyses live here, and
they are reported as three separate numbers because they measure three
different things.

``--audit``   Reconciles the attempts and checks the *call-site* values in the
              records against the run metadata. Answers "is the export even
              telling the truth?" without trusting the metadata to answer it.

``--intent``  Feeds every recorded ``stt.raw_text`` straight into the unmodified
              intent engine. This is intent accuracy on **real speech output**
              rather than typed strings, and it needs no model and no config
              change -- the transcripts are already on disk, including the ones
              the confidence gate threw away.

``--decode``  Re-decodes the saved WAVs with an explicitly chosen model and gate
              base, giving a clean A/B on identical audio.

Nothing here writes configuration, and nothing here executes an action.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_RATE = 16_000


def load_run(run_id: Optional[str] = None):
    from benchmark.runner import BenchmarkRun, run_root

    if run_id is None:
        candidates = sorted(p for p in run_root().iterdir() if p.is_dir())
        if not candidates:
            raise SystemExit(f"No benchmark runs under {run_root()}")
        run_id = candidates[-1].name
    return BenchmarkRun.load(run_id)


def read_wav(path: Path):
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        rate = handle.getframerate()
    audio = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    if rate != SAMPLE_RATE:
        from speech.capture import resample_to_16k

        audio = resample_to_16k(audio, rate)
    return audio


# --------------------------------------------------------------------- audit

def audit(run) -> None:
    """Reconcile every attempt, and check metadata against the call sites."""
    from speech.listener import confidence_floor

    meta = run.metadata
    records = run.records
    finals = {r["command_id"]: r for r in records if r.get("command_id")}

    print("=" * 78)
    print("ATTEMPT RECONCILIATION")
    print("=" * 78)
    per_command = Counter(r["command_id"] for r in records)
    retried = {cid: n for cid, n in per_command.items() if n > 1}
    print(f"total attempt records        : {len(records)}")
    print(f"distinct commands            : {len(per_command)}")
    print(f"commands needing >1 attempt  : {len(retried)}  -> {dict(sorted(retried.items()))}")
    print(f"extra attempts beyond first  : {len(records) - len(per_command)}")
    print(f"skipped                      : "
          f"{sum(1 for r in records if r.get('status') == 'skipped')}")

    print("\nFINAL ATTEMPTS ONLY (what scoring used):")
    layers = Counter(r.get("failure_layer") for r in finals.values())
    for layer, count in sorted(layers.items(), key=lambda kv: -kv[1]):
        print(f"  {layer or 'NONE':<12}{count}")
    print(f"  {'TOTAL':<12}{len(finals)}")

    print("\nALL 41 ATTEMPTS (including superseded retries):")
    all_layers = Counter(r.get("failure_layer") for r in records)
    for layer, count in sorted(all_layers.items(), key=lambda kv: -kv[1]):
        print(f"  {layer or 'NONE':<12}{count}")

    print()
    print("=" * 78)
    print("METADATA vs CALL SITE")
    print("=" * 78)
    print(f"metadata stt_model : {meta.get('stt_model')!r}")
    print(f"metadata gate_base : {meta.get('gate_base')!r}")

    models = Counter((r.get("stt") or {}).get("model") for r in records)
    print(f"stt.model recorded per attempt : {dict(models)}")

    mismatches = []
    checked = 0
    for record in records:
        stt = record.get("stt") or {}
        validation = record.get("validation") or {}
        effective = validation.get("gate_effective")
        raw = stt.get("raw_text")
        if effective is None or not raw:
            continue
        checked += 1
        expected = confidence_floor(raw, meta.get("gate_base"))
        if abs(expected - effective) > 1e-6:
            mismatches.append((record["command_id"], record["attempt"], raw,
                               effective, expected))

    print(f"\ngate_effective checked against confidence_floor(raw_text, "
          f"{meta.get('gate_base')}) on {checked} attempt(s)")
    if mismatches:
        print(f"  MISMATCHES: {len(mismatches)}")
        for cid, attempt, raw, got, want in mismatches[:10]:
            print(f"    cmd {cid} att {attempt}: recorded {got:.4f}, "
                  f"recomputed {want:.4f}  ({raw!r})")
    else:
        print("  no mismatches — the gate actually applied matches the metadata")

    print("\nWhat gate base 0.35 would have produced for the same utterances:")
    print(f"{'cmd':<5}{'raw_text':<34}{'conf':>7}{'gate@0.9':>10}"
          f"{'gate@0.35':>11}{'@0.9':>8}{'@0.35':>8}")
    print("-" * 83)
    would_pass = 0
    scored = 0
    for cid in sorted(finals):
        record = finals[cid]
        stt = record.get("stt") or {}
        raw, confidence = stt.get("raw_text"), stt.get("confidence")
        if not raw or confidence is None:
            continue
        scored += 1
        high = confidence_floor(raw, 0.9)
        low = confidence_floor(raw, 0.35)
        passes_high, passes_low = confidence >= high, confidence >= low
        would_pass += passes_low
        print(f"{cid:<5}{raw[:33]:<34}{confidence:>7.3f}{high:>10.3f}"
              f"{low:>11.3f}{'PASS' if passes_high else 'REJECT':>8}"
              f"{'PASS' if passes_low else 'REJECT':>8}")
    print(f"\ngate 0.35 would have admitted {would_pass}/{scored} of the "
          "transcriptions that reached the gate")


# ------------------------------------------------------------ 5b: intent only

def replay_intent(run) -> None:
    """Every recorded transcript, straight into the unmodified intent engine."""
    from ai.intent import IntentEngine
    from benchmark.harness import BenchmarkSuite
    from config.settings import load_config
    from speech.accuracy import normalise

    suite = BenchmarkSuite()
    commands = {c.id: c for c in suite.commands()}
    engine = IntentEngine(load_config())
    finals = {r["command_id"]: r for r in run.records if r.get("command_id")}

    print("=" * 96)
    print("5b — INTENT ACCURACY ON THE RECORDED TRANSCRIPTS (no model, no config change)")
    print("=" * 96)
    print(f"{'cmd':<5}{'raw transcript':<36}{'matched':<20}{'expected':<20}"
          f"{'verdict'}")
    print("-" * 96)

    considered = correct_stt = intent_ok = 0
    failures: List[str] = []
    for cid in sorted(finals):
        record = finals[cid]
        command = commands.get(cid)
        raw = ((record.get("stt") or {}).get("raw_text") or "").strip()
        if command is None or record.get("status") == "skipped":
            continue
        considered += 1
        if not raw:
            print(f"{cid:<5}{'(nothing transcribed)':<36}{'—':<20}"
                  f"{command.intent or '—':<20}NO TRANSCRIPT")
            continue

        heard_right = normalise(raw) == normalise(command.text)
        correct_stt += heard_right
        intent = engine.parse(raw)
        matches = intent.name == command.intent
        params_ok = all(
            key in intent.parameters
            and str(intent.parameters[key]).strip().casefold()
            == str(want).strip().casefold()
            for key, want in (command.parameters or {}).items())
        ok = matches and params_ok
        intent_ok += ok and heard_right

        verdict = ("PASS" if ok else
                   "FAIL (params)" if matches else
                   "FAIL (no match)" if intent.name == "unknown" else
                   "FAIL (wrong intent)")
        if not heard_right:
            verdict += "  [misheard]"
        if not ok:
            failures.append(f"  cmd {cid:>2}  {raw!r} -> {intent.name} "
                            f"{dict(intent.parameters)}  expected "
                            f"{command.intent} {command.parameters}")
        print(f"{cid:<5}{raw[:35]:<36}{intent.name[:19]:<20}"
              f"{(command.intent or '—')[:19]:<20}{verdict}")

    # Intent on every transcript, and intent on correctly-heard ones only:
    # separate numbers, because mixing them blames the intent engine for
    # mishearings.
    all_ok = 0
    all_total = 0
    for cid in sorted(finals):
        record = finals[cid]
        command = commands.get(cid)
        raw = ((record.get("stt") or {}).get("raw_text") or "").strip()
        if command is None or not raw:
            continue
        all_total += 1
        intent = engine.parse(raw)
        if intent.name == command.intent and all(
                key in intent.parameters
                and str(intent.parameters[key]).strip().casefold()
                == str(want).strip().casefold()
                for key, want in (command.parameters or {}).items()):
            all_ok += 1

    print()
    print(f"transcripts available            : {all_total} / {considered}")
    print(f"transcribed correctly            : {correct_stt} / {considered}")
    print(f"INTENT on every transcript       : {all_ok} / {all_total}")
    print(f"INTENT on correct transcripts    : {intent_ok} / {correct_stt}")
    if failures:
        print("\nIntent failures:")
        for line in failures:
            print(line)


# ------------------------------------------------------------- 5a: re-decode

def replay_decode(run, model_size: str, gate_base: float,
                  vocabulary: List[str]) -> Dict[str, Any]:
    """Re-decode the saved WAVs with an explicit model and gate."""
    from speech.accuracy import normalise
    from speech.listener import confidence_floor
    from speech.providers.whisper_local import LocalWhisperRecognizer
    from benchmark.harness import BenchmarkSuite

    suite = BenchmarkSuite()
    commands = {c.id: c for c in suite.commands()}
    finals = {r["command_id"]: r for r in run.records if r.get("command_id")}

    recognizer = LocalWhisperRecognizer(model_size)
    print(f"\nLoading {model_size}…")
    recognizer.preload()

    rows: List[Dict[str, Any]] = []
    for cid in sorted(finals):
        record = finals[cid]
        command = commands.get(cid)
        if command is None:
            continue
        wav = run.audio_path(cid, record.get("attempt", 1))
        if not wav.exists():
            continue
        audio = read_wav(wav)
        result = recognizer.transcribe(audio, language="en",
                                       vocabulary=vocabulary)
        raw = result.raw_text or result.text
        confidence = result.confidence
        gate = confidence_floor(raw, gate_base) if raw else None
        admitted = bool(result.ok and confidence is not None
                        and gate is not None and confidence >= gate)
        rows.append({
            "command_id": cid, "expected": command.text, "raw": raw,
            "confidence": confidence, "gate": gate,
            "stt_correct": bool(raw) and normalise(raw) == normalise(command.text),
            "admitted": admitted,
        })
    return {"model": model_size, "gate_base": gate_base, "rows": rows}


def print_decode(report: Dict[str, Any]) -> None:
    rows = report["rows"]
    print("=" * 100)
    print(f"5a — RE-DECODE  model={report['model']}  gate_base={report['gate_base']}")
    print("=" * 100)
    print(f"{'cmd':<5}{'expected':<32}{'heard':<32}{'conf':>7}{'gate':>7}"
          f"{'stt':>6}{'gate':>8}")
    print("-" * 100)
    for row in rows:
        confidence = ("-" if row["confidence"] is None
                      else f"{row['confidence']:.3f}")
        gate = "-" if row["gate"] is None else f"{row['gate']:.3f}"
        print(f"{row['command_id']:<5}{row['expected'][:31]:<32}"
              f"{(row['raw'] or '(nothing)')[:31]:<32}"
              f"{confidence:>7}{gate:>7}"
              f"{'OK' if row['stt_correct'] else 'MISS':>6}"
              f"{'PASS' if row['admitted'] else 'REJECT':>8}")
    total = len(rows)
    stt_ok = sum(1 for r in rows if r["stt_correct"])
    admitted = sum(1 for r in rows if r["admitted"])
    both = sum(1 for r in rows if r["admitted"] and r["stt_correct"])
    print(f"\nSTT correct              : {stt_ok} / {total}")
    print(f"Admitted by the gate     : {admitted} / {total}")
    print(f"Correct AND admitted     : {both} / {total}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline benchmark replay")
    parser.add_argument("--run", default=None, help="run id (default: newest)")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--intent", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--models", nargs="*", default=["small"])
    parser.add_argument("--gate", type=float, default=0.35)
    parser.add_argument("--out", default=None, help="write JSON results here")
    args = parser.parse_args()

    run = load_run(args.run)
    print(f"Run {run.run_id} — {len(run.records)} attempt records\n")

    if args.audit:
        audit(run)
    if args.intent:
        replay_intent(run)

    reports = []
    if args.decode:
        for model in args.models:
            report = replay_decode(run, model, args.gate, ["Asma", "hey"])
            print_decode(report)
            reports.append(report)

    if args.out and reports:
        Path(args.out).write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
