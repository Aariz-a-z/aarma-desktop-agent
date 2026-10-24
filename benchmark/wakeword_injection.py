"""File-injection wake-word benchmark: synthesised speech straight into the backends.

**SYNTHETIC / FILE-INJECTION — NOT REAL-WORLD RESULT.** Every number this module
produces comes from text-to-speech audio fed directly to a recogniser. No
microphone, no speakers, no room. It cannot tell you what happens when somebody
speaks across a kitchen, and it must never be quoted as detection accuracy.

What it *can* answer, deterministically and repeatably:

* does AARMA's own spoken output contain something a wake detector would fire on;
* do the four names behave differently from each other;
* how do the candidate backends compare on identical audio;
* what does each cost in CPU, memory and processing time.

The acoustic path could not answer any of these, because Windows' echo
cancellation removes AARMA's voice from the microphone before it is recorded.
This is the half of the echo question that is measurable today.

Since the architecture is local STT plus transcript matching, **the match policy
is the operating point**, not an afterthought. It is defined in one place
(:class:`MatchPolicy`), recorded in every export, and applied identically to
every backend so the comparison is about recognition, not about four different
matchers.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

SAMPLE_RATE = 16_000

#: The four names. Scored separately, always. They are phonetically unalike --
#: "Aariz" and "Zara" carry quite different risks from "Asma" -- and an average
#: across them would hide exactly the differences worth knowing.
NAMES: Tuple[str, ...] = ("Asma", "Aariz", "Nova", "Zara")

#: Voices used to synthesise each phrase, so a result is not one voice's quirk.
VOICES: Tuple[Tuple[str, str], ...] = (
    ("aria", "en-US-AriaNeural"),
    ("guy", "en-US-GuyNeural"),
    ("sonia", "en-GB-SoniaNeural"),
)


@dataclass(frozen=True)
class Phrase:
    """One sentence to synthesise, and what the detector should do with it."""

    text: str
    #: "positive" -- the wake phrase, must fire for ``name``.
    #: "negative" -- must not fire for any name.
    kind: str
    #: Which name this phrase concerns, for per-name reporting.
    name: str
    group: str

    @property
    def slug(self) -> str:
        safe = re.sub(r"[^a-z0-9]+", "_", self.text.lower()).strip("_")
        return f"{self.kind[:3]}_{self.name.lower()}_{safe[:28]}"


def build_phrases() -> List[Phrase]:
    """The injection set: 4 positives, 16 negatives."""
    phrases: List[Phrase] = [
        Phrase(f"Hey {name}.", "positive", name, "wake_phrase") for name in NAMES
    ]

    # Things AARMA itself says. If any of these fires the detector, the
    # assistant wakes itself up every time it answers.
    responses = [
        ("Asma is ready.", "Asma"),
        ("I am Asma.", "Asma"),
        ("Aariz, that is your name.", "Aariz"),
        ("Nova, I've opened Chrome for you.", "Nova"),
        ("I couldn't find Zara in your contacts.", "Zara"),
    ]
    phrases += [Phrase(t, "negative", n, "assistant_response") for t, n in responses]

    # The bare name with no wake word. The single most important negative:
    # hearing its own name mentioned must not wake the assistant.
    phrases += [Phrase(f"{name}.", "negative", name, "bare_name") for name in NAMES]

    # Ordinary sentences that happen to contain a name.
    sentences = [
        ("Asma is a common name.", "Asma"),
        ("Nova is a car.", "Nova"),
        ("I met Zara yesterday.", "Zara"),
        ("Aariz called me this morning.", "Aariz"),
        ("My friend Asma is coming over later.", "Asma"),
        ("Have you seen the new Nova model?", "Nova"),
        ("Zara has a shop in the city centre.", "Zara"),
    ]
    phrases += [Phrase(t, "negative", n, "sentence_with_name") for t, n in sentences]
    return phrases


# ----------------------------------------------------------------- matching

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return _SPACE.sub(" ", _PUNCT.sub(" ", folded)).strip()


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class MatchPolicy:
    """When a transcript counts as the wake phrase.

    Recorded in every export because it *is* the operating point. Two settings
    carry all the weight:

    ``require_hey``  With it on, "Asma is ready" cannot fire -- only "hey asma"
                     can. Off, the assistant wakes whenever its own name is
                     spoken, which is what the negative set is here to catch.
    ``max_distance`` Local STT does not spell names reliably: "asma" comes back
                     as "osma", "as ma", "asthma". A strict equality match would
                     report a working recogniser as a total failure, so a small
                     edit distance is allowed -- and swept, because it trades
                     misses against false wakes.
    """

    require_hey: bool = True
    max_distance: int = 1
    #: Allow the name to arrive split across two tokens ("as ma").
    join_split_tokens: bool = True

    def label(self) -> str:
        return (f"require_hey={self.require_hey}, max_distance={self.max_distance}, "
                f"join_split={self.join_split_tokens}")

    def _name_hit(self, token: str, name: str) -> bool:
        return edit_distance(token, name) <= self.max_distance

    def matches(self, transcript: str, name: str) -> bool:
        """Whether ``transcript`` contains the wake phrase for ``name``."""
        tokens = normalise(transcript).split()
        target = normalise(name)
        if not tokens or not target:
            return False

        for index, token in enumerate(tokens):
            candidates = [token]
            if self.join_split_tokens and index + 1 < len(tokens):
                candidates.append(token + tokens[index + 1])
            for candidate in candidates:
                if not self._name_hit(candidate, target):
                    continue
                if not self.require_hey:
                    return True
                if index > 0 and edit_distance(tokens[index - 1], "hey") <= 1:
                    return True
        return False

    def fired_for(self, transcript: str) -> List[str]:
        """Every name this transcript would wake. Usually zero or one."""
        return [n for n in NAMES if self.matches(transcript, n)]


DEFAULT_POLICY = MatchPolicy()


# ---------------------------------------------------------------- backends

class Backend:
    """A candidate wake-detection backend, driven from audio in memory."""

    key = "abstract"
    label = "Abstract"

    def load(self) -> None:
        """Load models. Timed separately from recognition."""

    def transcribe(self, audio, sample_rate: int = SAMPLE_RATE) -> str:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label}


class VoskBackend(Backend):
    """Vosk small English model, optionally restricted to a runtime grammar.

    The grammar is the lever that separates this from general dictation: given
    a word list, the decoder cannot emit anything else, which should cut both
    CPU and false wakes. Benchmarked both ways because that delta is the point.
    """

    def __init__(self, model_path: str, *, grammar: bool = True) -> None:
        self.model_path = model_path
        self.grammar = grammar
        self.key = "vosk_grammar" if grammar else "vosk_unconstrained"
        self.label = ("Vosk small + runtime grammar" if grammar
                      else "Vosk small, full vocabulary")
        self._model = None

    #: Only these words may be emitted when the grammar is active. ``[unk]``
    #: is required by Vosk so out-of-grammar audio has somewhere to go.
    def grammar_json(self) -> str:
        import json

        words = ["hey"] + [n.lower() for n in NAMES] + ["[unk]"]
        return json.dumps(words)

    def load(self) -> None:
        import vosk

        vosk.SetLogLevel(-1)
        self._model = vosk.Model(self.model_path)

    def transcribe(self, audio, sample_rate: int = SAMPLE_RATE) -> str:
        import json

        import numpy as np
        import vosk

        recognizer = (vosk.KaldiRecognizer(self._model, sample_rate,
                                           self.grammar_json())
                      if self.grammar
                      else vosk.KaldiRecognizer(self._model, sample_rate))
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        recognizer.AcceptWaveform(pcm)
        result = json.loads(recognizer.FinalResult())
        return (result.get("text") or "").strip()

    def describe(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label,
                "model_path": self.model_path,
                "grammar": self.grammar_json() if self.grammar else None}


class WhisperBackend(Backend):
    """VAD-gated faster-whisper.

    Silero runs first and the decoder only sees audio it accepted, which is
    what makes an always-on design plausible at all: the expensive stage is
    duty-cycled behind a cheap one. The gate is measured as well as used --
    ``vad_opened`` records whether the decoder ran.
    """

    def __init__(self, size: str) -> None:
        self.size = size
        self.key = f"whisper_{size}"
        self.label = f"VAD-gated faster-whisper ({size})"
        self._recognizer = None
        self._detector = None
        self._tuning = None
        self.vad_opened = 0
        self.vad_total = 0

    def load(self) -> None:
        from speech.providers.whisper_local import LocalWhisperRecognizer
        from speech.vad import VadTuning, create_detector

        self._recognizer = LocalWhisperRecognizer(self.size)
        self._recognizer.preload()
        self._detector = create_detector()
        self._tuning = VadTuning()

    def transcribe(self, audio, sample_rate: int = SAMPLE_RATE) -> str:
        self.vad_total += 1
        if not self._detector.has_speech(audio, self._tuning):
            return ""                      # Gate stayed shut; decoder never ran.
        self.vad_opened += 1
        result = self._recognizer.transcribe(
            audio, language="en",
            vocabulary=[n for n in NAMES] + ["hey"])
        return (result.raw_text or result.text or "").strip()

    def describe(self) -> Dict[str, Any]:
        from speech.vad import VadTuning

        tuning = VadTuning()
        return {"key": self.key, "label": self.label, "model_size": self.size,
                "compute_type": "int8", "device": "cpu", "beam_size": 1,
                "vad": asdict(tuning),
                "vad_duty_cycle": (self.vad_opened / self.vad_total
                                   if self.vad_total else None)}


# ----------------------------------------------------------------- scoring

@dataclass
class ClipResult:
    """One clip through one backend."""

    clip: str
    text: str
    kind: str
    name: str
    group: str
    voice: str
    backend: str
    transcript: str = ""
    fired: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def fired_any(self) -> bool:
        return bool(self.fired)

    @property
    def correct(self) -> bool:
        if self.kind == "positive":
            return self.name in self.fired
        return not self.fired

    @property
    def outcome(self) -> str:
        if self.kind == "positive":
            return "TP" if self.name in self.fired else "FN"
        return "FP" if self.fired else "TN"


def score(results: Sequence[ClipResult]) -> Dict[str, Any]:
    """Counts with denominators attached. Never a single blended number."""
    positives = [r for r in results if r.kind == "positive"]
    negatives = [r for r in results if r.kind == "negative"]
    tp = sum(1 for r in positives if r.outcome == "TP")
    fn = sum(1 for r in positives if r.outcome == "FN")
    fp = sum(1 for r in negatives if r.outcome == "FP")
    tn = sum(1 for r in negatives if r.outcome == "TN")
    latencies = sorted(r.latency_ms for r in results)

    def pct(n: int, d: int) -> Optional[float]:
        return (n / d) if d else None

    return {
        "true_positives": tp, "false_negatives": fn,
        "false_positives": fp, "true_negatives": tn,
        "positive_total": len(positives), "negative_total": len(negatives),
        "detection_rate": pct(tp, len(positives)),
        "false_positive_rate": pct(fp, len(negatives)),
        "latency_ms": {
            "mean": (sum(latencies) / len(latencies)) if latencies else None,
            "p50": latencies[len(latencies) // 2] if latencies else None,
            "p95": (latencies[int(len(latencies) * 0.95) - 1]
                    if len(latencies) >= 20 else None),
            "max": latencies[-1] if latencies else None,
        },
    }


def score_by_name(results: Sequence[ClipResult]) -> Dict[str, Dict[str, Any]]:
    """Per name, because averaging the four would hide the differences."""
    return {name: score([r for r in results if r.name == name]) for name in NAMES}


def run_backend(backend: Backend, clips: Sequence[Dict[str, Any]]
                ) -> Tuple[List[ClipResult], float]:
    """Feed every clip through one backend. Returns results and load seconds."""
    import wave

    import numpy as np

    started = time.perf_counter()
    backend.load()
    load_seconds = time.perf_counter() - started

    results: List[ClipResult] = []
    for clip in clips:
        with wave.open(clip["path"], "rb") as handle:
            rate = handle.getframerate()
            audio = (np.frombuffer(handle.readframes(handle.getnframes()),
                                   dtype="<i2").astype("float32") / 32768.0)
        begin = time.perf_counter()
        transcript = backend.transcribe(audio, rate)
        elapsed = (time.perf_counter() - begin) * 1000.0
        results.append(ClipResult(
            clip=clip["path"], text=clip["text"], kind=clip["kind"],
            name=clip["name"], group=clip["group"], voice=clip["voice"],
            backend=backend.key, transcript=transcript,
            fired=DEFAULT_POLICY.fired_for(transcript), latency_ms=elapsed))
    return results, load_seconds
