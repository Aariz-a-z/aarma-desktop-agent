"""The speech-to-text provider interface.

Everything above this line depends only on :class:`SpeechRecognizer`, so an
offline model, an online service or a future streaming engine are
interchangeable to the rest of AARMA.

**The rule this interface exists to enforce: a recogniser never guesses.**
:class:`TranscriptionResult` can say "I heard nothing", "I heard something but
not clearly", or "I heard this, with this confidence" -- and those are different
states, not degrees of the same one. A provider that cannot supply a confidence
leaves it ``None``; nothing downstream may invent one, and a result that fails
its confidence threshold is refused rather than passed to the engine as if it
were a command (specification section 10).

The interface is language-aware from the outset (section 8) and carries a
``detected_language`` field that providers may fill in, so automatic detection
can be added later without changing the type (section 9).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from utils.logger import get_logger

logger = get_logger("speech.listener")


class RecognitionOutcome(str, Enum):
    """What a recognition attempt actually produced."""

    TRANSCRIBED = "transcribed"       # Usable text.
    NO_SPEECH = "no_speech"           # Silence, noise, or too short.
    LOW_CONFIDENCE = "low_confidence"  # Heard something; not sure enough to act.
    CANCELLED = "cancelled"           # The user stopped it.
    UNAVAILABLE = "unavailable"       # No provider, model or device.
    ERROR = "error"                   # Something failed.


@dataclass(frozen=True)
class TranscriptionResult:
    """One recognition attempt, successful or not."""

    outcome: RecognitionOutcome
    text: str = ""
    language: str = "en"
    #: Filled in only when the provider reports it. Never inferred.
    detected_language: Optional[str] = None
    #: 0.0-1.0 where the provider supplies a usable signal, otherwise ``None``.
    confidence: Optional[float] = None
    #: What the recogniser returned *before* validation. Kept even when the
    #: result is refused, so "the model misheard" and "our rules discarded a
    #: correct reading" can be told apart.
    raw_text: str = ""
    #: The provider's own scores, for diagnostics. Never invented.
    average_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    duration: float = 0.0
    provider: str = ""
    #: True when audio was processed on this machine and never left it.
    processed_locally: bool = True
    error: str = ""
    error_code: str = ""

    @property
    def ok(self) -> bool:
        """Whether this result carries text safe to act on."""
        return self.outcome is RecognitionOutcome.TRANSCRIBED and bool(self.text.strip())

    @property
    def is_actionable(self) -> bool:
        return self.ok

    @classmethod
    def transcribed(cls, text: str, **kwargs) -> "TranscriptionResult":
        return cls(outcome=RecognitionOutcome.TRANSCRIBED, text=text, **kwargs)

    @classmethod
    def nothing_heard(cls, **kwargs) -> "TranscriptionResult":
        kwargs.setdefault("error", "I didn't hear anything. Please try again.")
        return cls(outcome=RecognitionOutcome.NO_SPEECH, error_code="no_speech", **kwargs)

    @classmethod
    def unclear(cls, **kwargs) -> "TranscriptionResult":
        kwargs.setdefault(
            "error", "I couldn't understand that clearly. Please try again.")
        # Overridable, so a script mismatch keeps its own code for the log
        # while still being an "unclear" outcome to the user.
        kwargs.setdefault("error_code", "low_confidence")
        return cls(outcome=RecognitionOutcome.LOW_CONFIDENCE, **kwargs)

    @classmethod
    def cancelled(cls, **kwargs) -> "TranscriptionResult":
        kwargs.setdefault("error", "Listening stopped.")
        return cls(outcome=RecognitionOutcome.CANCELLED, error_code="cancelled", **kwargs)

    @classmethod
    def failed(cls, error: str, *, error_code: str = "recognition_failed",
               **kwargs) -> "TranscriptionResult":
        return cls(outcome=RecognitionOutcome.ERROR, error=error,
                   error_code=error_code, **kwargs)

    @classmethod
    def unavailable(cls, error: str, *, error_code: str = "unavailable",
                    **kwargs) -> "TranscriptionResult":
        return cls(outcome=RecognitionOutcome.UNAVAILABLE, error=error,
                   error_code=error_code, **kwargs)


@dataclass(frozen=True)
class RecognizerAvailability:
    """Whether recognition can run, and what the user should know about it."""

    available: bool
    provider: str
    processes_locally: bool = True
    #: True when the provider needs a one-time download before first use.
    needs_setup: bool = False
    reason: str = ""
    suggestion: str = ""

    @property
    def privacy_label(self) -> str:
        """Where the audio goes. Shown in the interface, not buried in docs."""
        return ("Audio is processed on this computer." if self.processes_locally
                else "Audio is sent to an online service.")


def confidence_floor(text: str, base_threshold: float) -> float:
    """The confidence a transcription of this length must actually reach.

    A flat threshold is wrong, and measurably so. Whisper's ``avg_logprob`` is a
    *mean over tokens*, and short utterances carry few tokens with the first
    ones inherently uncertain — so brief commands score systematically lower
    than sentences even when they are transcribed perfectly. Measured on this
    machine across four acoustic conditions:

    ==========  ===================  ==========================
    Words       Confidence observed  Correct at the low end?
    ==========  ===================  ==========================
    1-2         0.366 - 0.638        yes, including 0.366
    4           0.637 - 0.742        yes
    7           0.699 - 0.717        yes
    ==========  ===================  ==========================

    A flat 0.45 gate therefore rejected *correct* readings of "Asma", "Help."
    and "Open Chrome." while passing every long sentence — penalising exactly
    the short commands a voice assistant exists to handle. Scaling the
    requirement by length fixes that without loosening the gate on the long
    utterances where a low score really does mean trouble.
    """
    words = len((text or "").split())
    if words <= 2:
        return base_threshold * 0.55
    if words <= 4:
        return base_threshold * 0.80
    return base_threshold


def confidence_from_logprob(average_logprob: float) -> float:
    """Turn a mean per-token log probability into a 0-1 confidence.

    This is a **derivation, not an invention**: ``avg_logprob`` is the model's
    own mean log probability, so its exponential is the corresponding mean
    probability. Calibrated against real samples on this machine -- clean speech
    scores about 0.60-0.78, while a misrecognition of noisy audio scored 0.47 --
    which is what the default threshold is set against.
    """
    try:
        return max(0.0, min(1.0, math.exp(average_logprob)))
    except (OverflowError, ValueError):
        return 0.0


class SpeechRecognizer(ABC):
    """A speech-to-text backend."""

    #: Stable identifier stored in configuration.
    name: str = "abstract"
    #: Shown in Settings. Never provider jargon.
    display_name: str = "Abstract"
    #: True when audio never leaves the machine.
    processes_locally: bool = True

    @abstractmethod
    def is_available(self) -> bool:
        """True when this recogniser could run right now."""

    @abstractmethod
    def availability(self, language: str = "en") -> RecognizerAvailability:
        """Report readiness for ``language`` without recognising anything."""

    @abstractmethod
    def supported_languages(self) -> Sequence[str]:
        """Base language codes this recogniser accepts."""

    @abstractmethod
    def transcribe(self, audio, *, language: str = "en",
                   sample_rate: int = 16_000,
                   vocabulary: Sequence[str] = ()) -> TranscriptionResult:
        """Convert captured audio into text. Blocks; call on a worker thread.

        ``vocabulary`` names words the recogniser is unlikely to know but is
        likely to hear -- in practice, the assistant's own name. A recogniser
        may use it to bias decoding, or ignore it entirely.

        **It carries a fixed word list and nothing else.** No conversation
        history, no previous transcription, no user content: priming a decoder
        with what was said before makes it repeat itself, and priming it with
        private text puts that text inside the recognition of the next thing
        anyone says in the room.

        Must never raise: failures are reported as a
        :class:`TranscriptionResult`. Must never return invented text.
        """

    def cancel(self) -> None:
        """Abandon an in-progress transcription, if the backend supports it."""

    def supports_language(self, language: str) -> bool:
        base = (language or "en").split("-")[0].lower()
        return base in self.supported_languages()

    def unavailable_reason(self) -> str:
        return f"{self.display_name} is not available."


class NullSpeechRecognizer(SpeechRecognizer):
    """Used when no recogniser can run. Never fabricates a transcription."""

    name = "none"
    display_name = "No speech recognition"

    def __init__(self, reason: str = "Speech recognition isn't available.") -> None:
        self._reason = reason

    def is_available(self) -> bool:
        return False

    def availability(self, language: str = "en") -> RecognizerAvailability:
        return RecognizerAvailability(
            available=False, provider=self.name, reason=self._reason,
            suggestion="You can keep using text input.")

    def supported_languages(self) -> Sequence[str]:
        return ()

    def transcribe(self, audio, *, language: str = "en",
                   sample_rate: int = 16_000,
                   vocabulary: Sequence[str] = ()) -> TranscriptionResult:
        return TranscriptionResult.unavailable(self._reason, provider=self.name,
                                               language=language)

    def unavailable_reason(self) -> str:
        return self._reason


#: Kept so Phase 1 imports continue to resolve; the interface is now real.
SpeechListener = SpeechRecognizer
NullSpeechListener = NullSpeechRecognizer
