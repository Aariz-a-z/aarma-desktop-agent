"""One structured record per utterance, covering the whole pipeline.

Written because "most spoken commands fail" is not a diagnosis. An utterance
crosses six stages before anything happens, each can fail, and from the outside
every failure produces the same shrug. This record makes the stages separately
visible so the *first* one that went wrong can be named.

**The first wrong stage, not the last loud one.** If speech recognition mishears
"open chrome" as "open crumb", intent matching will also fail — but the intent
engine is not the bug. :meth:`PipelineTrace.classify` walks the stages in
execution order and stops at the first that produced a wrong or empty result.

**Every stage writes its field, even on failure.** A stage that never ran gets
``null``, not a missing key, so a reader can tell "did not run" from "ran and
found nothing". That distinction is the whole reason for the file.

**Raw text is never overwritten.** ``stt.raw_text`` is exactly what the
recogniser returned; anything cleaned up lives in ``normalization``. Losing the
raw string would make a validation rejection indistinguishable from a
mishearing.

This module sits in :mod:`utils` deliberately: the record is written from the
speech layer, the intent layer and the engine, and none of those may import each
other. It depends on nothing but the standard library.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Stage names, in execution order. :meth:`PipelineTrace.classify` relies on
#: this ordering, so it is the single definition of "which came first".
STAGES: tuple[str, ...] = ("VAD", "STT", "VALIDATION", "INTENT", "ACTION")

NO_FAILURE = "NONE"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AudioStage:
    duration_sec: Optional[float] = None
    sample_rate: Optional[int] = None
    peak_amplitude: Optional[float] = None
    #: Only written for benchmark utterances, so a failure can be re-analysed
    #: without asking the person to say it all again.
    wav_path: Optional[str] = None


@dataclass
class VadStage:
    speech_detected: Optional[bool] = None
    speech_start_sec: Optional[float] = None
    speech_end_sec: Optional[float] = None
    trailing_silence_sec: Optional[float] = None


@dataclass
class SttStage:
    engine: Optional[str] = None
    model: Optional[str] = None
    #: Exactly what the recogniser returned. Never cleaned, never replaced.
    raw_text: Optional[str] = None
    confidence: Optional[float] = None
    #: Empty unless the recogniser is asked for n-best. Beam size is currently
    #: 1, so this is ``[]`` -- recorded as a fact, not left implicit.
    alternatives: List[str] = field(default_factory=list)
    latency_ms: Optional[int] = None


@dataclass
class NormalizationStage:
    normalized_text: Optional[str] = None


@dataclass
class ValidationStage:
    """Why a transcription was kept or thrown away.

    ``gate_effective`` is the number that actually decided it. The configured
    threshold is scaled by utterance length, so recording only the constant
    would leave a rejection unexplainable -- which is the failure mode this
    whole record exists to prevent.
    """

    result: Optional[str] = None            # PASS | REJECT
    raw_confidence: Optional[float] = None
    gate_base: Optional[float] = None
    gate_effective: Optional[float] = None
    length_scale_input: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


@dataclass
class IntentStage:
    name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    #: How the match was made: ``pattern`` today. Recorded rather than assumed,
    #: so adding synonym or fuzzy matching later shows up in the data.
    match_method: Optional[str] = None
    match_score: Optional[float] = None


@dataclass
class ActionStage:
    selected: Optional[str] = None
    risk_level: Optional[str] = None
    would_require_confirmation: Optional[bool] = None
    #: Always False in benchmark mode. The benchmark resolves and stops.
    executed: bool = False


@dataclass
class PipelineTrace:
    """Everything that happened to one utterance."""

    mode: str = "live"                       # live | benchmark | text
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=_now)
    #: What the speaker was asked to say. ``None`` outside the benchmark.
    expected_text: Optional[str] = None
    #: Expected intent/parameters after benchmark-side mapping. Expectation
    #: only -- nothing in the runtime path reads these.
    expected_intent: Optional[str] = None
    expected_parameters: Optional[Dict[str, Any]] = None

    audio: Optional[AudioStage] = None
    vad: Optional[VadStage] = None
    stt: Optional[SttStage] = None
    normalization: Optional[NormalizationStage] = None
    validation: Optional[ValidationStage] = None
    intent: Optional[IntentStage] = None
    action: Optional[ActionStage] = None

    final_result: str = "FAILED"             # SUCCESS | FAILED
    failure_layer: str = NO_FAILURE
    #: One line naming what went wrong, for a human reading the file.
    failure_detail: str = ""

    # ------------------------------------------------------------ classifying

    def classify(self) -> str:
        """Name the first stage that produced a wrong or empty result.

        Order matters more than severity here. A mishearing guarantees an intent
        miss, so blaming the intent engine for it would send the next round of
        work to the wrong layer.
        """
        if self.vad is not None and self.vad.speech_detected is False:
            self.failure_detail = "no speech found in the recording"
            return "VAD"

        if self.stt is not None:
            if not (self.stt.raw_text or "").strip():
                self.failure_detail = "recogniser returned no text"
                return "STT"
            if self.expected_text is not None and not self.stt_correct:
                self.failure_detail = (
                    f"heard {self.stt.raw_text!r}, expected {self.expected_text!r}")
                return "STT"

        if self.validation is not None and self.validation.result == "REJECT":
            self.failure_detail = self.validation.reason or "validation refused"
            return "VALIDATION"

        if self.intent is not None:
            if self.intent.name in (None, "", "unknown"):
                self.failure_detail = "no intent matched"
                return "INTENT"
            if self.expected_intent is not None:
                if self.intent.name != self.expected_intent:
                    self.failure_detail = (
                        f"matched {self.intent.name!r}, "
                        f"expected {self.expected_intent!r}")
                    return "INTENT"
                if not self.parameters_match:
                    self.failure_detail = (
                        f"parameters {self.intent.parameters!r}, "
                        f"expected {self.expected_parameters!r}")
                    return "INTENT"

        if self.action is not None and not self.action.selected:
            self.failure_detail = "intent resolved to no registered action"
            return "ACTION"

        self.failure_detail = ""
        return NO_FAILURE

    def finalise(self) -> "PipelineTrace":
        """Compute the verdict fields. Call once, after every stage has run."""
        self.failure_layer = self.classify()
        self.final_result = "SUCCESS" if self.failure_layer == NO_FAILURE else "FAILED"
        return self

    # -------------------------------------------------------------- scoring

    @property
    def stt_exact(self) -> bool:
        """Character-for-character, including case and punctuation."""
        if self.stt is None or self.expected_text is None:
            return False
        return (self.stt.raw_text or "") == self.expected_text

    @property
    def stt_correct(self) -> bool:
        """Correct after case and punctuation are set aside.

        Whisper's capitals and full stops are formatting decisions, not
        hearing. Scoring them as errors would report a working recogniser as
        broken.
        """
        if self.stt is None or self.expected_text is None:
            return False
        from speech.accuracy import normalise

        return normalise(self.stt.raw_text or "") == normalise(self.expected_text)

    @property
    def parameters_match(self) -> bool:
        """Whether the extracted parameters match what was expected.

        Compared case-insensitively on string values: the intent layer passes
        the user's own casing through, and the action layer lowercases where it
        matters. Only the keys that were *expected* are checked, so an extra
        optional parameter is not counted as a failure.
        """
        if self.expected_parameters is None:
            return True
        actual = self.intent.parameters if self.intent else None
        if actual is None:
            return not self.expected_parameters
        for key, want in self.expected_parameters.items():
            if key not in actual:
                return False
            got = actual[key]
            if isinstance(want, str) and isinstance(got, str):
                if want.strip().casefold() != got.strip().casefold():
                    return False
            elif want != got:
                return False
        return True

    # -------------------------------------------------------------- writing

    def to_dict(self) -> Dict[str, Any]:
        """A plain dictionary with every key present, ``None`` where unrun."""
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "expected_text": self.expected_text,
            "expected_intent": self.expected_intent,
            "expected_parameters": self.expected_parameters,
            "audio": asdict(self.audio) if self.audio else None,
            "vad": asdict(self.vad) if self.vad else None,
            "stt": asdict(self.stt) if self.stt else None,
            "normalization": asdict(self.normalization) if self.normalization else None,
            "validation": asdict(self.validation) if self.validation else None,
            "intent": asdict(self.intent) if self.intent else None,
            "action": asdict(self.action) if self.action else None,
            "final_result": self.final_result,
            "failure_layer": self.failure_layer,
            "failure_detail": self.failure_detail,
        }


_WRITE_LOCK = threading.Lock()


def trace_path() -> Path:
    from utils.paths import get_paths

    return get_paths().log_dir / "pipeline_trace.jsonl"


def audio_dir() -> Path:
    from utils.paths import get_paths

    return get_paths().log_dir / "audio"


def write(record: PipelineTrace) -> Optional[Path]:
    """Append one record to ``logs/pipeline_trace.jsonl``.

    Never raises. A diagnostic that can break the thing it is diagnosing is
    worse than no diagnostic, so a failure here is swallowed and reported as
    ``None`` rather than propagated into the listening path.
    """
    try:
        path = trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with _WRITE_LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return path
    except Exception:  # noqa: BLE001 - instrumentation must not break the pipeline
        from utils.logger import get_logger

        get_logger("utils.trace").debug("Could not write pipeline trace")
        return None


def read_all(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every record written so far. Malformed lines are skipped, not fatal."""
    target = path or trace_path()
    if not target.exists():
        return []
    records: List[Dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
