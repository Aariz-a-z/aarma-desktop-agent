"""Scoring a transcription against what was actually said.

Separated from the interface so the measurement can be tested without a screen,
and separated from :mod:`speech.validation` because the two answer different
questions. Validation asks *"is this output shaped like a transcription at
all?"*; this asks *"is it the right one?"* — which only has an answer when a
human has told AARMA what they said.

**Recognition is scored, not intent.** "Open Chrome." transcribed as "open
chrome" is a perfect result: the punctuation and capitals are Whisper's
formatting, not a mishearing. Whether the words then map to an action is the
intent layer's business, and conflating the two is how a working recogniser
gets blamed for an unmapped phrase (specification section 15).

The word error rate is the standard measure: substitutions plus deletions plus
insertions, over the number of words expected. It is reported as measured, with
the sample size beside it, because an accuracy figure without a denominator is
not a measurement.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

#: Everything that is formatting rather than a word.
_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Reduce a phrase to the words in it, for comparison only.

    Case, punctuation and spacing are Whisper's presentation choices. Comparing
    them would report a correct transcription as an error.
    """
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", folded)).strip()


def words(text: str) -> List[str]:
    normalised = normalise(text)
    return normalised.split() if normalised else []


def edit_distance(expected: Sequence[str], actual: Sequence[str]) -> int:
    """Levenshtein distance over word lists."""
    if not expected:
        return len(actual)
    if not actual:
        return len(expected)

    previous = list(range(len(actual) + 1))
    for i, want in enumerate(expected, start=1):
        current = [i]
        for j, got in enumerate(actual, start=1):
            current.append(min(
                previous[j] + 1,            # deletion
                current[j - 1] + 1,         # insertion
                previous[j - 1] + (want != got),  # substitution
            ))
        previous = current
    return previous[-1]


def word_error_rate(expected: str, actual: str) -> float:
    """Errors per expected word. 0.0 is perfect; can exceed 1.0."""
    want, got = words(expected), words(actual)
    if not want:
        return 0.0 if not got else 1.0
    return edit_distance(want, got) / len(want)


@dataclass
class UtteranceScore:
    """One phrase, spoken once, scored against what the speaker intended."""

    expected: str
    detected: str = ""
    raw_detected: str = ""          # Before validation, when they differ.
    confidence: Optional[float] = None
    threshold: Optional[float] = None
    speech_seconds: float = 0.0
    latency: float = 0.0
    accepted: bool = False          # Did AARMA return it as a command?
    reason: str = ""                # Why not, if not.
    skipped: bool = False

    @property
    def exact(self) -> bool:
        """Whether the words match, ignoring case and punctuation."""
        return bool(self.detected) and normalise(self.detected) == normalise(self.expected)

    @property
    def raw_exact(self) -> bool:
        """Whether the *recogniser* got it right, before validation ran."""
        heard = self.raw_detected or self.detected
        return bool(heard) and normalise(heard) == normalise(self.expected)

    @property
    def error_rate(self) -> float:
        return word_error_rate(self.expected, self.detected or self.raw_detected)

    @property
    def verdict(self) -> str:
        """The short label shown beside the row."""
        if self.skipped:
            return "SKIPPED"
        if self.accepted and self.exact:
            return "PASS"
        if self.accepted:
            return "MISHEARD"
        if self.raw_exact:
            # The recogniser was right and AARMA discarded it. This is a
            # different bug from mishearing, and naming it differently is the
            # whole point of keeping the raw text.
            return "REJECTED (heard correctly)"
        if self.raw_detected:
            return "REJECTED (misheard)"
        return "NO SPEECH"

    @property
    def passed(self) -> bool:
        return self.accepted and self.exact


@dataclass
class AccuracyReport:
    """A whole test run. Reports what was measured, never what was hoped for."""

    scores: List[UtteranceScore] = field(default_factory=list)
    #: What the run is being measured against. Not a claim that it was met.
    target: float = 0.95

    @property
    def attempted(self) -> List[UtteranceScore]:
        return [s for s in self.scores if not s.skipped]

    @property
    def total(self) -> int:
        return len(self.attempted)

    @property
    def passes(self) -> int:
        return sum(1 for s in self.attempted if s.passed)

    @property
    def recognised(self) -> int:
        """Got the words right, whether or not AARMA then accepted them."""
        return sum(1 for s in self.attempted if s.raw_exact)

    @property
    def rejected_correct(self) -> int:
        """Correct transcriptions AARMA threw away. Should be zero."""
        return sum(1 for s in self.attempted if s.raw_exact and not s.accepted)

    @property
    def accuracy(self) -> float:
        return (self.passes / self.total) if self.total else 0.0

    @property
    def recognition_accuracy(self) -> float:
        return (self.recognised / self.total) if self.total else 0.0

    @property
    def mean_error_rate(self) -> float:
        rates = [s.error_rate for s in self.attempted]
        return sum(rates) / len(rates) if rates else 0.0

    @property
    def mean_latency(self) -> float:
        values = [s.latency for s in self.attempted if s.latency > 0]
        return sum(values) / len(values) if values else 0.0

    @property
    def meets_target(self) -> bool:
        return self.total > 0 and self.accuracy >= self.target

    def headline(self) -> str:
        """One honest sentence, with the denominator attached."""
        if not self.total:
            return "Nothing measured yet."
        line = (f"{self.passes} of {self.total} correct "
                f"({self.accuracy * 100:.0f}%). "
                f"Recognition alone: {self.recognised} of {self.total} "
                f"({self.recognition_accuracy * 100:.0f}%).")
        if self.rejected_correct:
            line += (f" {self.rejected_correct} correct transcription(s) were "
                     "refused by AARMA's own checks.")
        return line

    def report(self) -> str:
        """A block the user can copy into a bug report."""
        lines = ["Speech recognition test", "=" * 23, self.headline(), ""]
        for score in self.scores:
            lines.append(f"Expected   : {score.expected}")
            if score.skipped:
                lines.append("Result     : skipped")
                lines.append("")
                continue
            lines.append(f"Detected   : {score.detected or '(nothing returned)'}")
            if score.raw_detected and score.raw_detected != score.detected:
                lines.append(f"Raw output : {score.raw_detected}")
            confidence = ("n/a" if score.confidence is None
                          else f"{score.confidence:.3f}")
            threshold = ("n/a" if score.threshold is None
                         else f"{score.threshold:.3f}")
            lines.append(f"Confidence : {confidence} (needed {threshold})")
            lines.append(f"Speech     : {score.speech_seconds:.2f}s  "
                         f"Latency: {score.latency:.2f}s")
            lines.append(f"Verdict    : {score.verdict}"
                         + (f" — {score.reason}" if score.reason else ""))
            lines.append("")
        if self.total:
            lines.append(f"Mean word error rate: {self.mean_error_rate * 100:.1f}%")
            lines.append(f"Mean latency: {self.mean_latency:.2f}s")
            lines.append(f"Target: {self.target * 100:.0f}% — "
                         + ("met" if self.meets_target else "not met"))
        return "\n".join(lines)


def default_phrases(assistant_name: str = "Asma") -> List[str]:
    """The phrases the speech test asks for.

    The first five are the specified set. The last four are one- and two-word
    commands, included because short utterances score lowest and are exactly
    what a voice assistant is asked for most often — a test made only of
    sentences would miss the failure this whole exercise started from.
    """
    return [
        f"Hello {assistant_name}.",
        "Open Chrome.",
        "What time is it?",
        "Open YouTube.",
        "Tell me a joke.",
        assistant_name,
        "Stop.",
        "Help.",
        "Cancel.",
    ]
