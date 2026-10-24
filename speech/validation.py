"""Post-recognition validation: catching output that should not be trusted.

Whisper-family models do not fail by returning nothing. They fail by returning
something *plausible-looking*, which is worse: an invented command is acted on,
a missing one is not. Benchmarking on this machine produced three distinct
failure shapes, and each has a check here.

**Wrong script.** Asked for Kannada, the model returned Devanagari ("इगर समया
एश्टूः") or Latin ("atility"). It is transliterating or hallucinating in another
language entirely. Accepting that would put nonsense into the command pipeline,
so a script that does not match the requested language is refused
(specification sections 13 and 15).

**Low confidence.** Correct results scored 0.63-0.79; the failures scored
0.04-0.55. The confidence gate in :mod:`speech.input_engine` uses this.

**Degenerate output.** Models under-fed with audio emit repeated tokens
("the the the the") or long runs of one character. Repetition detection catches
these.

None of these checks *fix* a bad transcription. They convert it into an honest
refusal, which is the only safe outcome.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from utils.logger import get_logger

logger = get_logger("speech.validation")

#: The Unicode script each language is written in. Used to detect a model
#: answering in the wrong one -- the clearest signal that it did not actually
#: understand the audio.
LANGUAGE_SCRIPTS: Dict[str, Tuple[str, ...]] = {
    "en": ("LATIN",),
    "es": ("LATIN",),
    "fr": ("LATIN",),
    "hi": ("DEVANAGARI",),
    "kn": ("KANNADA",),
    "te": ("TELUGU",),
    "ta": ("TAMIL",),
    "ml": ("MALAYALAM",),
    "ur": ("ARABIC",),
    "ar": ("ARABIC",),
}

#: A transcription shorter than this is not worth script-checking.
MIN_SCRIPT_CHECK_CHARS = 4
#: Proportion of letters that must belong to the expected script.
SCRIPT_AGREEMENT = 0.5

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class ValidationResult:
    """Whether a transcription is safe to treat as a command."""

    accepted: bool
    reason: str = ""
    error_code: str = ""
    detected_script: str = ""

    @classmethod
    def accept(cls, script: str = "") -> "ValidationResult":
        return cls(accepted=True, detected_script=script)

    @classmethod
    def reject(cls, reason: str, error_code: str,
               script: str = "") -> "ValidationResult":
        return cls(accepted=False, reason=reason, error_code=error_code,
                   detected_script=script)


def dominant_script(text: str) -> str:
    """The Unicode script most of the letters belong to."""
    counts: Dict[str, int] = {}
    for character in text or "":
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        script = name.split()[0] if name else "UNKNOWN"
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "NONE"
    return max(counts, key=counts.get)


def script_agreement(text: str, expected: Sequence[str]) -> float:
    """Fraction of letters written in one of the expected scripts."""
    total = 0
    matching = 0
    for character in text or "":
        if not character.isalpha():
            continue
        total += 1
        name = unicodedata.name(character, "")
        script = name.split()[0] if name else "UNKNOWN"
        if script in expected:
            matching += 1
    return (matching / total) if total else 0.0


def has_excessive_repetition(text: str) -> bool:
    """Detect degenerate output: the same word or character over and over."""
    words = _WORD_RE.findall((text or "").lower())
    if len(words) >= 6:
        # One word making up most of a long utterance is decoder looping,
        # not speech.
        most_common = max((words.count(word) for word in set(words)), default=0)
        if most_common / len(words) >= 0.6:
            return True
        # Or the same short phrase repeated back to back.
        for size in (2, 3):
            if len(words) >= size * 3:
                chunk = words[:size]
                repeats = sum(1 for i in range(0, len(words) - size + 1, size)
                              if words[i:i + size] == chunk)
                if repeats >= 3:
                    return True
    stripped = re.sub(r"\s+", "", text or "")
    if len(stripped) >= 8:
        if re.search(r"(.)\1{7,}", stripped):
            return True
    return False


def validate_transcription(text: str, *, language: str,
                           check_script: bool = True) -> ValidationResult:
    """Decide whether ``text`` is a usable transcription of ``language`` speech.

    Rejections are deliberately conservative in one direction only: it is
    always safe to ask someone to repeat themselves, and never safe to act on
    something the recogniser did not really hear.
    """
    candidate = (text or "").strip()
    if not candidate:
        return ValidationResult.reject(
            "I didn't hear anything. Please try again.", "no_speech")

    if has_excessive_repetition(candidate):
        logger.info("Rejected repetitive transcription",
                    extra={"context": {"characters": len(candidate)}})
        return ValidationResult.reject(
            "I couldn't understand that clearly. Please try again.",
            "degenerate_output")

    script = dominant_script(candidate)
    if not check_script:
        return ValidationResult.accept(script)

    expected = LANGUAGE_SCRIPTS.get((language or "en").split("-")[0].lower())
    if not expected:
        return ValidationResult.accept(script)

    letters = sum(1 for character in candidate if character.isalpha())
    if letters < MIN_SCRIPT_CHECK_CHARS:
        return ValidationResult.accept(script)

    agreement = script_agreement(candidate, expected)
    if agreement < SCRIPT_AGREEMENT:
        from speech.voices import language_display_name

        logger.info("Rejected transcription in the wrong script", extra={"context": {
            "language": language, "expected": expected, "detected": script,
            "agreement": round(agreement, 2)}})
        return ValidationResult.reject(
            f"That didn't come through as {language_display_name(language)}. "
            "Please try again, or change the speech language in Settings.",
            "script_mismatch", script)

    return ValidationResult.accept(script)
