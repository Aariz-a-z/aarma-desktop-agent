"""Benchmark speech-recognition models on this machine.

Model choice should be a measurement, not a guess: the right model depends on
the CPU it will actually run on, and a model that is more accurate but takes ten
seconds is useless for a voice assistant. This measures accuracy, latency and
memory for each candidate and prints a table to choose from.

    python tools/benchmark_stt.py                  # tiny, base, small
    python tools/benchmark_stt.py --models base small medium
    python tools/benchmark_stt.py --languages en hi kn

Reference audio is synthesised with the Phase 2A neural voices and cached.
**That is a proxy, not a substitute for human speech**: synthetic voices are
cleaner and more consistent than a person in a room, so the numbers here are an
upper bound. Use ``tools/benchmark_microphone.py`` to measure your own voice.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODELS = ("tiny", "base", "small")
SAMPLE_RATE = 16_000

#: Short commands and one conversational sentence per language. English carries
#: more because it is the primary interface language.
PHRASES: Dict[str, List[Tuple[str, str]]] = {
    "en": [
        ("en-IN-NeerjaNeural", "Open Chrome."),
        ("en-IN-PrabhatNeural", "What time is it?"),
        ("en-US-AvaNeural", "Create a folder called AI Project in Documents."),
        ("en-GB-SoniaNeural", "Open YouTube and search for Python tutorials."),
        ("en-US-AndrewNeural",
         "Could you remind me tomorrow morning to submit the assignment?"),
    ],
    "hi": [
        ("hi-IN-SwaraNeural", "क्रोम खोलो।"),
        ("hi-IN-MadhurNeural", "अभी समय क्या हुआ है?"),
        ("hi-IN-SwaraNeural", "मुझे कल सुबह याद दिलाना।"),
    ],
    "kn": [
        ("kn-IN-SapnaNeural", "ಕ್ರೋಮ್ ತೆರೆಯಿರಿ."),
        ("kn-IN-GaganNeural", "ಈಗ ಸಮಯ ಎಷ್ಟು?"),
        ("kn-IN-SapnaNeural", "ನಾಳೆ ಬೆಳಿಗ್ಗೆ ನನಗೆ ನೆನಪಿಸಿ."),
    ],
    "te": [
        ("te-IN-ShrutiNeural", "క్రోమ్ తెరవండి."),
        ("te-IN-MohanNeural", "ఇప్పుడు సమయం ఎంత?"),
        ("te-IN-ShrutiNeural", "రేపు ఉదయం నాకు గుర్తు చేయండి."),
    ],
    "ta": [
        ("ta-IN-PallaviNeural", "குரோம் திறக்கவும்."),
        ("ta-IN-ValluvarNeural", "இப்போது மணி என்ன?"),
        ("ta-IN-PallaviNeural", "நாளை காலை எனக்கு நினைவூட்டு."),
    ],
    "ml": [
        ("ml-IN-SobhanaNeural", "ക്രോം തുറക്കുക."),
        ("ml-IN-MidhunNeural", "ഇപ്പോൾ സമയം എത്ര?"),
        ("ml-IN-SobhanaNeural", "നാളെ രാവിലെ എന്നെ ഓർമ്മിപ്പിക്കുക."),
    ],
}

#: Conditions applied to English, to probe distance and noise.
CONDITIONS: Tuple[Tuple[str, float, float], ...] = (
    ("clean", 1.00, 0.0000),      # Close talking, quiet room.
    ("quiet", 0.15, 0.0008),      # Roughly arm's length or beyond.
    ("noisy", 0.50, 0.0200),      # Moderate background noise.
    ("far", 0.06, 0.0050),        # Distant and quiet.
)


# --------------------------------------------------------------------- scoring

def _normalise(text: str) -> str:
    """Lower-case, strip punctuation and collapse spaces, script-aware."""
    cleaned = unicodedata.normalize("NFKC", text or "").lower()
    kept = [ch for ch in cleaned if not unicodedata.category(ch).startswith("P")]
    return " ".join("".join(kept).split())


def _edit_distance(a: Sequence, b: Sequence) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (item_a != item_b)))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = _normalise(reference).split()
    hyp = _normalise(hypothesis).split()
    return 1.0 if not ref else _edit_distance(ref, hyp) / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Better than WER for scripts that do not separate words the same way."""
    ref = _normalise(reference).replace(" ", "")
    hyp = _normalise(hypothesis).replace(" ", "")
    return 1.0 if not ref else _edit_distance(ref, hyp) / len(ref)


def script_of(text: str) -> str:
    """The dominant Unicode script, to detect transliteration."""
    counts: Dict[str, int] = {}
    for ch in text or "":
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        script = name.split()[0] if name else "UNKNOWN"
        counts[script] = counts.get(script, 0) + 1
    return max(counts, key=counts.get) if counts else "NONE"


# ------------------------------------------------------------------- fixtures

def synthesise(cache: Path) -> Dict[str, List[Tuple[str, Path]]]:
    """Render every phrase with a neural voice, cached on disk."""
    import edge_tts

    cache.mkdir(parents=True, exist_ok=True)
    plan: Dict[str, List[Tuple[str, Path]]] = {}

    async def run() -> None:
        for language, entries in PHRASES.items():
            plan[language] = []
            for index, (voice, text) in enumerate(entries):
                path = cache / f"{language}-{index}.mp3"
                if not path.exists():
                    await edge_tts.Communicate(text, voice).save(str(path))
                plan[language].append((text, path))

    asyncio.run(run())
    return plan


def decode(path: Path):
    import av
    import numpy as np

    container = av.open(str(path))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    for frame in container.decode(audio=0):
        for out in resampler.resample(frame):
            chunks.append(out.to_ndarray().flatten())
    container.close()
    return (np.concatenate(chunks).astype("float32") / 32768.0)


def apply_condition(audio, gain: float, noise: float, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    lead = np.zeros(int(0.4 * SAMPLE_RATE), dtype="float32")
    tail = np.zeros(int(0.6 * SAMPLE_RATE), dtype="float32")
    signal = np.concatenate([lead, (audio * gain).astype("float32"), tail])
    if noise:
        signal = signal + (rng.standard_normal(signal.size) * noise).astype("float32")
    return signal.astype("float32")


def process_rss_mb() -> float:
    """Resident memory of this process, in MB."""
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize / 1e6
    except Exception:  # noqa: BLE001 - diagnostics only
        return 0.0


# ------------------------------------------------------------------ benchmark

@dataclass
class Measurement:
    model: str
    language: str
    condition: str
    reference: str
    hypothesis: str
    wer: float
    cer: float
    latency: float
    audio_seconds: float
    confidence: Optional[float]
    script: str
    script_matches: bool


@dataclass
class ModelSummary:
    model: str
    load_seconds: float = 0.0
    resident_mb: float = 0.0
    measurements: List[Measurement] = field(default_factory=list)

    def rows_for(self, language: str, condition: Optional[str] = None) -> List[Measurement]:
        return [m for m in self.measurements
                if m.language == language and (condition is None or m.condition == condition)]

    @staticmethod
    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0


def benchmark(models: Sequence[str], languages: Sequence[str],
              cache: Path, verbose: bool = True) -> List[ModelSummary]:
    from faster_whisper import WhisperModel

    from speech.listener import confidence_from_logprob

    fixtures = synthesise(cache)
    summaries: List[ModelSummary] = []

    for model_name in models:
        if verbose:
            print(f"\n{'=' * 74}\nMODEL: {model_name}\n{'=' * 74}")
        baseline = process_rss_mb()
        started = time.time()
        try:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
        except Exception as exc:  # noqa: BLE001
            print(f"  could not load {model_name}: {type(exc).__name__}: {exc}")
            continue
        load_seconds = time.time() - started
        summary = ModelSummary(model=model_name, load_seconds=load_seconds)

        for language in languages:
            if language not in fixtures:
                continue
            # Conditions are only varied for English; other languages are
            # measured clean, so a poor score cannot be blamed on the noise.
            conditions = CONDITIONS if language == "en" else CONDITIONS[:1]
            for condition_name, gain, noise in conditions:
                for reference, path in fixtures[language]:
                    audio = apply_condition(decode(path), gain, noise)
                    t0 = time.time()
                    segments, _info = model.transcribe(
                        audio, language=language, beam_size=1, vad_filter=True,
                        condition_on_previous_text=False)
                    collected = list(segments)
                    latency = time.time() - t0
                    hypothesis = "".join(s.text for s in collected).strip()
                    confidence = (confidence_from_logprob(
                        sum(s.avg_logprob for s in collected) / len(collected))
                        if collected else None)

                    reference_script = script_of(reference)
                    hypothesis_script = script_of(hypothesis)
                    summary.measurements.append(Measurement(
                        model=model_name, language=language,
                        condition=condition_name, reference=reference,
                        hypothesis=hypothesis,
                        wer=word_error_rate(reference, hypothesis),
                        cer=character_error_rate(reference, hypothesis),
                        latency=latency,
                        audio_seconds=audio.size / SAMPLE_RATE,
                        confidence=confidence, script=hypothesis_script,
                        script_matches=(hypothesis_script == reference_script)))

            if verbose:
                clean = summary.rows_for(language, "clean")
                if clean:
                    print(f"  {language}: WER {ModelSummary.mean([m.wer for m in clean]):.2%}"
                          f"  CER {ModelSummary.mean([m.cer for m in clean]):.2%}"
                          f"  script kept "
                          f"{sum(m.script_matches for m in clean)}/{len(clean)}")

        summary.resident_mb = max(0.0, process_rss_mb() - baseline)
        summaries.append(summary)
        del model

    return summaries


def report(summaries: Sequence[ModelSummary], languages: Sequence[str]) -> None:
    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    print(f"{'model':<8}{'load':>7}{'RAM':>9}{'latency':>9}{'RTF':>7}   English WER by condition")
    for summary in summaries:
        english = [m for m in summary.measurements if m.language == "en"]
        latency = ModelSummary.mean([m.latency for m in english])
        audio = ModelSummary.mean([m.audio_seconds for m in english])
        rtf = latency / audio if audio else 0.0
        parts = []
        for condition, _gain, _noise in CONDITIONS:
            rows = summary.rows_for("en", condition)
            if rows:
                parts.append(f"{condition} {ModelSummary.mean([m.wer for m in rows]):.0%}")
        print(f"{summary.model:<8}{summary.load_seconds:>6.1f}s"
              f"{summary.resident_mb:>8.0f}M{latency:>8.2f}s{rtf:>7.2f}   "
              + "  ".join(parts))

    print(f"\n{'model':<8}" + "".join(f"{lang:>16}" for lang in languages))
    for summary in summaries:
        cells = []
        for language in languages:
            rows = summary.rows_for(language, "clean")
            if not rows:
                cells.append(f"{'-':>16}")
                continue
            cer = ModelSummary.mean([m.cer for m in rows])
            kept = sum(m.script_matches for m in rows)
            cells.append(f"{f'CER {cer:.0%} {kept}/{len(rows)}':>16}")
        print(f"{summary.model:<8}" + "".join(cells))
    print("\n  RTF = seconds of compute per second of audio (lower is better).")
    print("  'n/m' = phrases returned in the expected script (transcribed, not transliterated).")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark speech recognition models.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--languages", nargs="+", default=list(PHRASES))
    parser.add_argument("--cache", default=None, help="where to keep reference audio")
    parser.add_argument("--json", default=None, help="write raw measurements here")
    args = parser.parse_args(argv)

    cache = Path(args.cache) if args.cache else REPO_ROOT / "data" / "benchmark-audio"
    summaries = benchmark(args.models, args.languages, cache)
    report(summaries, args.languages)

    if args.json:
        payload = [{"model": s.model, "load_seconds": s.load_seconds,
                    "resident_mb": s.resident_mb,
                    "measurements": [asdict(m) for m in s.measurements]}
                   for s in summaries]
        Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
