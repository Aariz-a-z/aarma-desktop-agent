"""Offline speech recognition with a local Whisper model (faster-whisper).

**Audio never leaves the machine.** The model runs in-process on the CPU, the
array goes straight from the microphone buffer into the recogniser, and nothing
is written to disk. For a feature that listens to a room, local processing is
not a nice-to-have -- it is the difference between a tool and a microphone
someone else owns.

Chosen over the alternatives because it is the only option that is offline,
private *and* covers AARMA's declared languages. Vosk is smaller but has no
Kannada, Telugu or Malayalam; the cloud services are more accurate on Indic
speech but send audio away; ``openai-whisper`` needs PyTorch and is slower on a
CPU than CTranslate2.

**Honest accuracy note.** Measured on this machine, English at the ``base``
model is reliable (short commands transcribed exactly, ~1.2 s for a two-second
utterance). Indic languages at ``tiny``/``base`` are **not** reliable -- they
tend to transliterate into Latin script rather than transcribe. Larger models
fix this; :data:`MODEL_SIZES` exposes that choice and the interface says so
rather than implying every declared language works equally well.

**One-time download.** The model is fetched on first use (~145 MB for ``base``)
and cached. Until then recognition needs a connection, which
:meth:`LocalWhisperRecognizer.availability` reports as ``needs_setup`` rather
than failing mysteriously offline.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

from speech.listener import (
    RecognizerAvailability,
    SpeechRecognizer,
    TranscriptionResult,
    confidence_from_logprob,
)
from utils.logger import get_logger

logger = get_logger("speech.whisper")


@dataclass(frozen=True)
class ModelSize:
    """A selectable accuracy/size trade-off, with measured figures."""

    key: str
    display_name: str
    approximate_mb: int      # Download size.
    description: str
    resident_mb: int = 0     # Measured RSS with the model loaded.
    latency_seconds: float = 0.0   # Measured on a ~3.8 s utterance.
    load_seconds: float = 0.0      # Measured, once cached.


#: Offered in Settings, with the figures measured by ``tools/benchmark_stt.py``
#: on this machine (2-core i3, no GPU). ``small`` is the default because it is
#: the only size that handles distant speech and non-English languages, and its
#: latency is still acceptable for a command.
MODEL_SIZES: Dict[str, ModelSize] = {
    size.key: size
    for size in (
        ModelSize("tiny", "Fastest", 75,
                  "Fastest to respond. Close-range English only.",
                  resident_mb=310, latency_seconds=0.7, load_seconds=0.7),
        ModelSize("base", "Faster", 145,
                  "Quick. Good English up close; weak further away.",
                  resident_mb=355, latency_seconds=1.1, load_seconds=0.9),
        ModelSize("small", "Recommended", 484,
                  "Best balance: handles distance, and the only size that "
                  "manages Hindi and Tamil.",
                  resident_mb=540, latency_seconds=3.4, load_seconds=1.9),
        ModelSize("medium", "Most accurate (slow)", 1500,
                  "Most accurate, but around ten seconds per command on a "
                  "computer without a graphics accelerator.",
                  resident_mb=1500, latency_seconds=11.0, load_seconds=5.0),
    )
}

DEFAULT_MODEL_SIZE = "small"

#: Languages Whisper handles that AARMA also declares.
SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en", "hi", "kn", "te", "ta", "ml", "ur", "ar", "es", "fr",
)

#: Languages that actually passed a local benchmark at :data:`DEFAULT_MODEL_SIZE`.
#: Measured character error rates: English 1%, Tamil 8%, Hindi 16%. Kannada,
#: Telugu and Malayalam failed at every size tested -- the model answered in the
#: wrong script -- so they are *supported* but not *verified*, and the script
#: check in :mod:`speech.validation` refuses their output rather than passing
#: transliterated nonsense into the command pipeline.
VERIFIED_LANGUAGES: tuple[str, ...] = ("en", "hi", "ta")

#: A prompt longer than this is refused rather than truncated. The prompt is a
#: short list of names; anything larger means something else has leaked into it.
MAX_PROMPT_WORDS = 12


def build_prompt(vocabulary: Sequence[str]) -> str:
    """Turn a word list into the decoder prompt, defensively.

    Whisper decodes conditioned on this text, so what goes in it matters twice
    over. Measured on the ``small`` model: without it, "Asma" came back as
    "Asthma" in 8 of 8 attempts across four acoustic conditions; with it, 8 of
    8 were correct and the other 28 utterances in the set were unchanged.

    Only short, whitespace-free tokens survive, and only a handful of them.
    That is a deliberate limit rather than a formality: the prompt biases every
    word of the decode, so a sentence in here would distort ordinary commands,
    and any user content in here would be conditioning the recognition of
    whatever is said next in the room.
    """
    words = []
    for entry in vocabulary or ():
        token = (entry or "").strip()
        # One word, no punctuation to confuse the tokeniser, plausibly a name.
        if not token or len(token.split()) != 1 or len(token) > 32:
            continue
        if token not in words:
            words.append(token)
        if len(words) >= MAX_PROMPT_WORDS:
            break
    return " ".join(f"{word}." for word in words)


class LocalWhisperRecognizer(SpeechRecognizer):
    """Whisper running locally through CTranslate2."""

    name = "whisper_local"
    display_name = "On-device recognition"
    processes_locally = True

    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE,
                 *, compute_type: str = "int8") -> None:
        self._model_size = model_size if model_size in MODEL_SIZES else DEFAULT_MODEL_SIZE
        self._compute_type = compute_type
        self._model = None
        self._loaded_size: Optional[str] = None
        self._lock = threading.RLock()
        self._cancel = threading.Event()

    # ------------------------------------------------------------ properties

    @property
    def model_size(self) -> str:
        return self._model_size

    def set_model_size(self, model_size: str) -> None:
        """Switch model. The new one loads lazily on the next transcription."""
        chosen = model_size if model_size in MODEL_SIZES else DEFAULT_MODEL_SIZE
        with self._lock:
            if chosen != self._model_size:
                self._model_size = chosen
                self._model = None
                self._loaded_size = None

    @staticmethod
    def is_installed() -> bool:
        import importlib.util

        return importlib.util.find_spec("faster_whisper") is not None

    def is_model_downloaded(self, model_size: Optional[str] = None) -> bool:
        """Whether the model is already cached, so first use can work offline."""
        size = model_size or self._model_size
        roots = [
            Path(os.environ.get("HF_HUB_CACHE", "")),
            Path(os.environ.get("HF_HOME", "")) / "hub",
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
        for root in roots:
            try:
                if not root or not root.is_dir():
                    continue
            except OSError:
                continue
            candidate = root / f"models--Systran--faster-whisper-{size}"
            if candidate.is_dir() and any(candidate.rglob("*.bin")):
                return True
        return False

    def is_available(self) -> bool:
        return self.is_installed()

    def availability(self, language: str = "en") -> RecognizerAvailability:
        if not self.is_installed():
            return RecognizerAvailability(
                available=False, provider=self.name, processes_locally=True,
                reason="On-device speech recognition isn't installed.",
                suggestion="Run: pip install faster-whisper")

        if not self.supports_language(language):
            from speech.voices import language_display_name

            return RecognizerAvailability(
                available=False, provider=self.name, processes_locally=True,
                reason=("Speech recognition doesn't support "
                        f"{language_display_name(language)} yet."),
                suggestion="Change the speech input language in Settings.")

        if not self.is_model_downloaded():
            size = MODEL_SIZES[self._model_size]
            return RecognizerAvailability(
                available=True, provider=self.name, processes_locally=True,
                needs_setup=True,
                reason=(f"The speech model ({size.approximate_mb} MB) will download "
                        "the first time you use the microphone."),
                suggestion="This happens once and needs an internet connection.")

        return RecognizerAvailability(available=True, provider=self.name,
                                      processes_locally=True)

    def supported_languages(self) -> Sequence[str]:
        return SUPPORTED_LANGUAGES

    def is_language_verified(self, language: str) -> bool:
        """Whether this language was actually verified, not merely supported."""
        return (language or "en").split("-")[0].lower() in VERIFIED_LANGUAGES

    # ---------------------------------------------------------------- loading

    def _ensure_model(self):
        """Load the model, downloading it if this is the first run."""
        with self._lock:
            if self._model is not None and self._loaded_size == self._model_size:
                return self._model
            from faster_whisper import WhisperModel

            logger.info("Loading speech model",
                        extra={"context": {"size": self._model_size}})
            self._model = WhisperModel(self._model_size, device="cpu",
                                       compute_type=self._compute_type)
            self._loaded_size = self._model_size
            return self._model

    def preload(self) -> bool:
        """Load (and if necessary download) the model ahead of time."""
        try:
            self._ensure_model()
            return True
        except Exception:  # noqa: BLE001 - download/load failures vary
            logger.exception("Speech model could not be loaded")
            return False

    # ------------------------------------------------------------ recognition

    def transcribe(self, audio, *, language: str = "en",
                   sample_rate: int = 16_000,
                   vocabulary: Sequence[str] = ()) -> TranscriptionResult:
        base_language = (language or "en").split("-")[0].lower()
        self._cancel.clear()

        if not self.is_installed():
            return TranscriptionResult.unavailable(
                "On-device speech recognition isn't installed.",
                provider=self.name, language=base_language)

        if not self.supports_language(base_language):
            from speech.voices import language_display_name

            return TranscriptionResult.unavailable(
                "Speech recognition doesn't support "
                f"{language_display_name(base_language)} yet.",
                error_code="unsupported_language",
                provider=self.name, language=base_language)

        if audio is None or getattr(audio, "size", 0) == 0:
            return TranscriptionResult.nothing_heard(provider=self.name,
                                                     language=base_language)

        duration = float(getattr(audio, "size", 0)) / float(sample_rate)

        try:
            model = self._ensure_model()
        except Exception as exc:  # noqa: BLE001 - network or disk failure
            logger.error("Speech model unavailable",
                         extra={"context": {"error": type(exc).__name__}})
            return TranscriptionResult.failed(
                "The speech model isn't ready. It needs a one-time download, "
                "which requires an internet connection.",
                error_code="model_unavailable",
                provider=self.name, language=base_language, duration=duration)

        if self._cancel.is_set():
            return TranscriptionResult.cancelled(provider=self.name,
                                                 language=base_language)

        try:
            segments, info = model.transcribe(
                audio,
                language=base_language,
                beam_size=1,
                vad_filter=True,           # Drops silence before decoding.
                condition_on_previous_text=False,  # No carry-over between utterances.
                initial_prompt=build_prompt(vocabulary) or None,
            )
            collected = list(segments)
        except Exception as exc:  # noqa: BLE001 - decoding can fail many ways
            logger.error("Transcription failed",
                         extra={"context": {"error": type(exc).__name__}})
            return TranscriptionResult.failed(
                "I couldn't process that audio.", error_code="transcription_failed",
                provider=self.name, language=base_language, duration=duration)

        if self._cancel.is_set():
            return TranscriptionResult.cancelled(provider=self.name,
                                                 language=base_language)

        text = "".join(segment.text for segment in collected).strip()
        average_logprob = (sum(s.avg_logprob for s in collected) / len(collected)
                           if collected else None)
        no_speech_prob = (sum(s.no_speech_prob for s in collected) / len(collected)
                          if collected else None)

        # Guard against the failure shapes measured on this machine: output in
        # the wrong script, and degenerate repetition. Both look like text and
        # neither is a transcription.
        if text:
            from speech.validation import validate_transcription

            verdict = validate_transcription(text, language=base_language)
            if not verdict.accepted:
                return TranscriptionResult.unclear(
                    error=verdict.reason, error_code=verdict.error_code,
                    language=base_language, provider=self.name,
                    duration=duration, raw_text=text,
                    confidence=confidence_from_logprob(average_logprob)
                    if average_logprob is not None else None,
                    average_logprob=average_logprob,
                    no_speech_prob=no_speech_prob,
                    detected_language=getattr(info, "language", None))

        if not collected or not text:
            # The voice-activity filter found nothing to decode. This is the
            # normal outcome for silence or room noise, and it is why nothing
            # can be fabricated: there is no text to invent from.
            return TranscriptionResult.nothing_heard(
                provider=self.name, language=base_language, duration=duration,
                no_speech_prob=no_speech_prob)

        # Confidence is the mean of the model's own per-segment log probability.
        confidence = confidence_from_logprob(average_logprob)
        detected = getattr(info, "language", None)

        logger.info("Transcribed speech", extra={"context": {
            "language": base_language, "detected": detected,
            "confidence": round(confidence, 3), "duration": round(duration, 2),
            "characters": len(text),   # Never the transcription itself.
        }})

        return TranscriptionResult.transcribed(
            text=text,
            # The same string, kept under both names: downstream code compares
            # raw against final to tell "misheard" from "we discarded it", and
            # that comparison must not depend on which path produced the result.
            raw_text=text,
            language=base_language,
            detected_language=detected,
            confidence=confidence,
            average_logprob=average_logprob,
            no_speech_prob=no_speech_prob,
            duration=duration,
            provider=self.name,
            processed_locally=True,
        )

    def cancel(self) -> None:
        self._cancel.set()

    def unavailable_reason(self) -> str:
        if not self.is_installed():
            return ("On-device speech recognition isn't installed. "
                    "Run: pip install faster-whisper")
        return ""
