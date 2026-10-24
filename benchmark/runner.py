"""Run state for the live voice benchmark: identity, durability, export.

Three properties this file exists to guarantee.

**Nothing is lost.** Every attempt is appended to ``traces.jsonl`` the moment
the command completes. If the application dies at command 17, sixteen are still
on disk and the run can be resumed rather than re-spoken.

**Nothing executed.** :func:`export` refuses to write ``run.json`` if any record
carries ``executed: true``. The engine already refuses structurally; this is the
second lock, on the artefact rather than the dispatch path, because a benchmark
whose export cannot be trusted is worse than no benchmark.

**Every attempt is kept.** Retries are recorded with an increasing ``attempt``
number and scoring uses the last one, but the earlier attempts stay in the file.
Whether a command needs three tries is itself a measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

STATUS_OK = "completed"
STATUS_SKIPPED = "skipped"


class ExportRefused(RuntimeError):
    """Raised when an export would claim a run in which something executed."""


def run_root() -> Path:
    from utils.paths import get_paths

    return get_paths().log_dir / "benchmark"


def new_run_id() -> str:
    """Sortable, readable, and unique enough for one machine."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class BenchmarkRun:
    """One pass through the command set, on disk from the first command."""

    run_id: str = field(default_factory=new_run_id)
    metadata: Dict[str, Any] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)

    # ----------------------------------------------------------------- paths

    @property
    def directory(self) -> Path:
        return run_root() / self.run_id

    @property
    def traces_path(self) -> Path:
        return self.directory / "traces.jsonl"

    @property
    def export_path(self) -> Path:
        return self.directory / "run.json"

    def audio_path(self, command_id: int, attempt: int) -> Path:
        return self.directory / f"{command_id:02d}_{attempt:02d}.wav"

    def prepare(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory

    # --------------------------------------------------------------- writing

    def append(self, record: Dict[str, Any]) -> None:
        """Persist one attempt immediately. Crash-safety over batching."""
        self.prepare()
        self.records.append(record)
        with self.traces_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    # -------------------------------------------------------------- progress

    def attempts_for(self, command_id: int) -> List[Dict[str, Any]]:
        return [r for r in self.records if r.get("command_id") == command_id]

    def next_attempt(self, command_id: int) -> int:
        return len(self.attempts_for(command_id)) + 1

    def final_attempts(self) -> List[Dict[str, Any]]:
        """The last attempt at each command -- what scoring uses."""
        latest: Dict[int, Dict[str, Any]] = {}
        for record in self.records:
            command_id = record.get("command_id")
            if command_id is not None:
                latest[command_id] = record
        return [latest[key] for key in sorted(latest)]

    def completed_ids(self) -> List[int]:
        return sorted({r["command_id"] for r in self.records
                       if r.get("command_id") is not None})

    # ---------------------------------------------------------------- export

    def summary(self) -> Dict[str, Any]:
        """Three rates, computed from the recorded run. Never merged into one.

        ``intent_accuracy`` is deliberately measured only over commands whose
        transcription was correct: mixing in the mishearings would report the
        intent engine as broken for something the recogniser did.
        """
        finals = [r for r in self.final_attempts()
                  if r.get("status") != STATUS_SKIPPED
                  and not r.get("excluded")]
        total = len(finals)
        stt_ok = [r for r in finals if r.get("stt_correct")]
        end_to_end = [r for r in finals if r.get("final_result") == "SUCCESS"]
        intent_ok = [r for r in stt_ok
                     if (r.get("intent") or {}).get("name")
                     == r.get("expected_intent")]

        layers: Dict[str, int] = {}
        for record in finals:
            if record.get("final_result") != "SUCCESS":
                layer = record.get("failure_layer") or "UNKNOWN"
                layers[layer] = layers.get(layer, 0) + 1

        return {
            "scored_commands": total,
            "skipped": sum(1 for r in self.final_attempts()
                           if r.get("status") == STATUS_SKIPPED),
            "excluded": sum(1 for r in self.final_attempts()
                            if r.get("excluded")),
            "total_attempts": len(self.records),
            "stt_accuracy": {"passed": len(stt_ok), "total": total},
            "intent_accuracy_on_correct_transcripts": {
                "passed": len(intent_ok), "total": len(stt_ok)},
            "end_to_end_resolution": {"passed": len(end_to_end), "total": total},
            "failure_layers": layers,
        }

    def assert_nothing_executed(self) -> None:
        """The export-side lock. Refuses rather than warns."""
        offenders = [
            r.get("command_id")
            for r in self.records
            if (r.get("action") or {}).get("executed") is not False
        ]
        if offenders:
            raise ExportRefused(
                "Refusing to export: "
                f"{len(offenders)} record(s) do not carry executed=false "
                f"(commands {offenders}). A benchmark run in which an action "
                "may have run is not a valid measurement."
            )

    def export(self, *, mapping: Optional[Dict[str, Any]] = None) -> Path:
        self.assert_nothing_executed()
        self.prepare()
        payload = {
            "run_id": self.run_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "metadata": self.metadata,
            "intent_mapping": mapping,
            "summary": self.summary(),
            "records": self.records,
        }
        with self.export_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return self.export_path

    def delete_audio(self) -> int:
        """Remove every recording from this run. Returns how many went."""
        removed = 0
        for wav in sorted(self.directory.glob("*.wav")):
            try:
                wav.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    # -------------------------------------------------------------- resuming

    @classmethod
    def load(cls, run_id: str) -> "BenchmarkRun":
        run = cls(run_id=run_id)
        if run.traces_path.exists():
            with run.traces_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        run.records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        meta = run.directory / "metadata.json"
        if meta.exists():
            try:
                run.metadata = json.loads(meta.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return run

    def save_metadata(self) -> None:
        self.prepare()
        (self.directory / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8")


def find_incomplete(expected: int = 20) -> Optional[BenchmarkRun]:
    """The most recent run that never reached the last command, if any."""
    root = run_root()
    if not root.is_dir():
        return None
    for directory in sorted(root.iterdir(), reverse=True):
        if not directory.is_dir() or not (directory / "traces.jsonl").exists():
            continue
        if (directory / "run.json").exists():
            return None          # Newest run finished; nothing to resume.
        run = BenchmarkRun.load(directory.name)
        if len(run.completed_ids()) < expected:
            return run
        return None
    return None
