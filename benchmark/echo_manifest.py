"""Which TTS-echo clips may be counted, and a guard against counting the rest.

The echo test asks whether AARMA's own voice falsely wakes it. A clip the
detector could never have fired on answers that question trivially and in the
wrong direction: it produces "zero false wakes" because nothing was ever
audible, not because the detector resisted anything. Pooling weak clips into the
denominator is therefore not a small inaccuracy -- it manufactures a passing
result.

So the corpus is classified once, by :mod:`tools.audit_tts_echo`, into three
labels, and this module is the only sanctioned way to read that classification:

``VALID_FOR_ECHO_TEST``   loud enough, VAD-triggering, and demonstrably
                          containing the phrase. Counts.
``BORDERLINE_LOW_LEVEL``  audible and intelligible but below the level band of
                          the real positive clips. Reported separately, never
                          folded into the headline.
``INVALID_NEAR_SILENT``   fails level, VAD, or content. Excluded entirely.

Two labels would have forced the middle group into one bucket or the other and
quietly biased the result, which is why there are three.

:func:`label_for` **raises** on a clip that is missing from the manifest or
carries no label. Silently including it would inflate the denominator; silently
skipping it would hide that the corpus and the manifest have drifted apart.
Neither failure is one anybody would notice in a summary table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID = "VALID_FOR_ECHO_TEST"
BORDERLINE = "BORDERLINE_LOW_LEVEL"
INVALID = "INVALID_NEAR_SILENT"

LABELS = frozenset({VALID, BORDERLINE, INVALID})

#: Below this many valid clips, an echo false-positive rate is not worth
#: quoting -- a single clip either fires or does not, and 0/1 is not evidence
#: of resistance to self-triggering.
MINIMUM_VALID_CLIPS = 20

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "corpus" / "tts_echo_manifest.json"


class EchoManifestError(RuntimeError):
    """The manifest and the corpus disagree, or a clip has no verdict."""


class EchoManifest:
    """The classification, loaded once and queried by filename."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data
        self._by_name: Dict[str, Dict[str, Any]] = {}
        for clip in data.get("clips", []):
            name = Path(clip.get("file", "")).name
            if not name:
                raise EchoManifestError("manifest entry has no 'file'")
            label = clip.get("label")
            if label not in LABELS:
                raise EchoManifestError(
                    f"{name}: label {label!r} is not one of {sorted(LABELS)}")
            if not clip.get("reason"):
                raise EchoManifestError(f"{name}: every entry needs a 'reason'")
            # Same filename in both batches: the re-record overwrote the
            # originals in place, so batch is part of the identity.
            self._by_name.setdefault(name, clip)
            self._by_name[f"{clip.get('batch')}/{name}"] = clip

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EchoManifest":
        target = path or MANIFEST_PATH
        if not target.is_file():
            raise EchoManifestError(
                f"No TTS-echo manifest at {target}. Run tools/audit_tts_echo.py "
                "before any echo measurement -- clips must be classified first.")
        try:
            return cls(json.loads(target.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise EchoManifestError(f"{target} is not valid JSON: {exc}") from exc

    @property
    def thresholds(self) -> Dict[str, Any]:
        return dict(self._data.get("thresholds", {}))

    def label_for(self, clip_path) -> str:
        """The verdict for one clip. Raises if it has none.

        Deliberately not ``.get(name, INVALID)``: an unclassified clip is a
        disagreement between the corpus and the manifest, and the run should
        stop and be looked at rather than quietly drop a sample.
        """
        name = Path(clip_path).name
        entry = self._by_name.get(name)
        if entry is None:
            raise EchoManifestError(
                f"{name} is not in the TTS-echo manifest. Re-run "
                "tools/audit_tts_echo.py; a clip may never be counted, or "
                "skipped, without an explicit verdict.")
        return entry["label"]

    def entries(self, label: Optional[str] = None) -> List[Dict[str, Any]]:
        clips = list(self._data.get("clips", []))
        return [c for c in clips if c["label"] == label] if label else clips

    def valid_clips(self) -> List[Dict[str, Any]]:
        """The only clips an echo false-positive rate may be computed over."""
        return self.entries(VALID)

    def counts(self) -> Dict[str, int]:
        return {label: len(self.entries(label)) for label in sorted(LABELS)}

    def is_measurable(self) -> bool:
        return len(self.valid_clips()) >= MINIMUM_VALID_CLIPS

    def readiness(self) -> str:
        """One sentence a report can quote without softening it."""
        valid = len(self.valid_clips())
        if valid >= MINIMUM_VALID_CLIPS:
            return (f"{valid} valid clips: enough to quote an echo "
                    "false-activation rate.")
        return (f"Only {valid} valid clip(s) against a minimum of "
                f"{MINIMUM_VALID_CLIPS}. The TTS echo test is NOT measurable "
                "on this corpus; no echo rate may be reported.")


def assert_measurable(manifest: Optional[EchoManifest] = None) -> EchoManifest:
    """Refuse to start an echo measurement that cannot mean anything."""
    active = manifest or EchoManifest.load()
    if not active.is_measurable():
        raise EchoManifestError(active.readiness())
    return active
