"""Generate TTS audio and run the file-injection wake-word benchmark.

SYNTHETIC / FILE-INJECTION -- NOT REAL-WORLD RESULT.

Audio is written once into its own directory and reused by every backend, so
all four configurations are compared on byte-identical input. Nothing here
touches the microphone corpus.

    python tools/wakeword_injection.py generate
    python tools/wakeword_injection.py run
    python tools/wakeword_injection.py sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.wakeword_injection import (  # noqa: E402
    DEFAULT_POLICY,
    VOICES,
    MatchPolicy,
    VoskBackend,
    WhisperBackend,
    build_phrases,
    run_backend,
    score,
    score_by_name,
)

VOSK_MODEL = Path.home() / ".cache" / "vosk" / "vosk-model-small-en-us-0.15"
VOSK_SHA256 = "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"


def injection_root() -> Path:
    """Its own directory. The microphone corpus is never written to."""
    from utils.paths import get_paths

    return get_paths().log_dir / "wakeword_injection"


def audio_dir() -> Path:
    return injection_root() / "audio"


def manifest_path() -> Path:
    return injection_root() / "clips.json"


# --------------------------------------------------------------- generation

def generate(args) -> int:
    """Synthesise every phrase in every voice, once."""
    from speech.audio import NullAudioPlayback
    from speech.providers.edge import EdgeNeuralSpeaker

    folder = audio_dir()
    folder.mkdir(parents=True, exist_ok=True)
    # NullAudioPlayback: synthesise to file, never play. Nothing reaches the
    # speakers, so nothing can reach the microphone.
    speaker = EdgeNeuralSpeaker(playback=NullAudioPlayback())
    if not speaker.is_installed():
        print("edge-tts is not installed; cannot generate audio.")
        return 2

    phrases = build_phrases()
    clips: List[Dict[str, Any]] = []
    total = len(phrases) * len(VOICES)
    index = 0
    print(f"Generating {total} clips ({len(phrases)} phrases x {len(VOICES)} voices)")

    for phrase in phrases:
        for voice_key, voice_name in VOICES:
            index += 1
            wav = folder / f"{phrase.slug}__{voice_key}.wav"
            if wav.exists() and not args.force:
                clips.append(_entry(wav, phrase, voice_key))
                continue
            mp3 = folder / f"{wav.stem}.mp3"
            try:
                speaker._synthesise(text=phrase.text, voice=voice_name,
                                    rate="+0%", volume="+0%", pitch="+0Hz",
                                    destination=mp3)
                _to_wav(mp3, wav)
            finally:
                mp3.unlink(missing_ok=True)
            clips.append(_entry(wav, phrase, voice_key))
            print(f"  [{index}/{total}] {phrase.kind:<8} {voice_key:<6} "
                  f"{phrase.text}")

    manifest_path().write_text(json.dumps({
        "note": "SYNTHETIC / FILE-INJECTION - NOT REAL-WORLD RESULT",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "voices": {k: v for k, v in VOICES},
        "clips": clips,
    }, indent=2), encoding="utf-8")
    print(f"\n{len(clips)} clips -> {folder}")
    print(f"manifest -> {manifest_path()}")
    return 0


def _entry(wav: Path, phrase, voice_key: str) -> Dict[str, Any]:
    return {"path": str(wav), "text": phrase.text, "kind": phrase.kind,
            "name": phrase.name, "group": phrase.group, "voice": voice_key}


def _to_wav(mp3: Path, wav: Path) -> None:
    """Decode to 16 kHz mono PCM, the format every backend expects."""
    import av
    import numpy as np

    container = av.open(str(mp3))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16_000)
    chunks = []
    for frame in container.decode(audio=0):
        for out in resampler.resample(frame):
            chunks.append(out.to_ndarray().flatten())
    container.close()
    audio = np.concatenate(chunks).astype("float32") / 32768.0
    # A little lead-in and tail so the voice activity detector sees an onset
    # rather than a buffer that starts mid-word.
    pad = np.zeros(int(0.3 * 16_000), dtype="float32")
    from speech.input_engine import write_wav

    write_wav(np.concatenate([pad, audio, pad]), wav)


# ------------------------------------------------------------- measurement

def _memory_mb() -> tuple:
    """Working set and peak working set, in MB.

    The process handle must be declared as HANDLE: ctypes defaults a return
    value to C ``int``, which truncates a 64-bit handle and makes the call fail
    silently, reporting 0.0 for everything.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi.dll")
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                          ctypes.byref(counters), counters.cb):
            return (0.0, 0.0)
        return (counters.WorkingSetSize / 1e6, counters.PeakWorkingSetSize / 1e6)
    except Exception:  # noqa: BLE001
        return (0.0, 0.0)


def _rss_mb() -> float:
    return _memory_mb()[0]


def _cpu_seconds() -> float:
    times = os.times()
    return times.user + times.system


def load_clips() -> List[Dict[str, Any]]:
    path = manifest_path()
    if not path.is_file():
        raise SystemExit(f"No clips at {path}. Run 'generate' first.")
    return json.loads(path.read_text(encoding="utf-8"))["clips"]


def build_backends() -> List:
    if not VOSK_MODEL.is_dir():
        raise SystemExit(f"Vosk model missing at {VOSK_MODEL}")
    return [
        VoskBackend(str(VOSK_MODEL), grammar=True),
        VoskBackend(str(VOSK_MODEL), grammar=False),
        WhisperBackend("tiny"),
        WhisperBackend("small"),
    ]


def run(args) -> int:
    clips = load_clips()
    print("=" * 100)
    print("WAKE-WORD FILE-INJECTION BENCHMARK")
    print("SYNTHETIC / FILE-INJECTION - NOT REAL-WORLD RESULT")
    print("=" * 100)
    print(f"clips: {len(clips)}  |  match policy: {DEFAULT_POLICY.label()}\n")

    report: Dict[str, Any] = {
        "note": "SYNTHETIC / FILE-INJECTION - NOT REAL-WORLD RESULT",
        "match_policy": asdict(DEFAULT_POLICY),
        "clip_count": len(clips),
        "backends": {},
    }

    for backend in build_backends():
        baseline_rss = _rss_mb()
        cpu_before = _cpu_seconds()
        wall_before = time.perf_counter()
        results, load_seconds = run_backend(backend, clips)
        wall = time.perf_counter() - wall_before
        cpu = _cpu_seconds() - cpu_before
        rss, peak_rss = _memory_mb()

        summary = score(results)
        per_name = score_by_name(results)
        report["backends"][backend.key] = {
            "label": backend.label,
            "describe": backend.describe(),
            "load_seconds": round(load_seconds, 2),
            "wall_seconds": round(wall, 2),
            "cpu_seconds": round(cpu, 2),
            "cpu_ratio": round(cpu / wall, 2) if wall else None,
            "rss_mb_after": round(rss, 1),
            "rss_delta_mb": round(rss - baseline_rss, 1),
            "peak_rss_mb": round(peak_rss, 1),
            "summary": summary,
            "per_name": per_name,
            "results": [asdict(r) for r in results],
        }
        _print_backend(backend, report["backends"][backend.key])

    out = injection_root() / "results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nfull results -> {out}")
    return 0


def _print_backend(backend, data: Dict[str, Any]) -> None:
    s = data["summary"]
    print("-" * 100)
    print(f"{data['label']}")
    print("-" * 100)
    print(f"  TP {s['true_positives']}/{s['positive_total']}   "
          f"FN {s['false_negatives']}   "
          f"FP {s['false_positives']}/{s['negative_total']}   "
          f"TN {s['true_negatives']}")
    lat = s["latency_ms"]
    p95 = "n/a" if lat["p95"] is None else f"{lat['p95']:.0f}"
    print(f"  latency ms: mean {lat['mean']:.0f}  p50 {lat['p50']:.0f}  "
          f"p95 {p95}  max {lat['max']:.0f}")
    print(f"  load {data['load_seconds']}s   wall {data['wall_seconds']}s   "
          f"cpu {data['cpu_seconds']}s ({data['cpu_ratio']}x)   "
          f"RSS {data['rss_mb_after']} MB (+{data['rss_delta_mb']})")
    print("  per name:")
    for name, ns in data["per_name"].items():
        rate = ns["detection_rate"]
        fpr = ns["false_positive_rate"]
        print(f"    {name:<7} TP {ns['true_positives']}/{ns['positive_total']}"
              f"   FP {ns['false_positives']}/{ns['negative_total']}"
              f"   detect {'n/a' if rate is None else f'{rate*100:.0f}%'}"
              f"   FP-rate {'n/a' if fpr is None else f'{fpr*100:.0f}%'}")
    misses = [r for r in data["results"] if r["kind"] == "positive"
              and r["name"] not in r["fired"]]
    fps = [r for r in data["results"] if r["kind"] == "negative" and r["fired"]]
    if misses:
        print("  MISSED wake phrases:")
        for r in misses:
            print(f"    {r['name']:<7} {r['voice']:<6} heard {r['transcript']!r}")
    if fps:
        print("  FALSE WAKES:")
        for r in fps:
            print(f"    {r['name']:<7} {r['voice']:<6} {r['text']!r} -> "
                  f"{r['transcript']!r} fired {r['fired']}")


def sweep(args) -> int:
    """Match-policy sweep. The policy is the operating point, so it is swept."""
    results_path = injection_root() / "results.json"
    if not results_path.is_file():
        raise SystemExit("Run 'run' first.")
    stored = json.loads(results_path.read_text(encoding="utf-8"))

    print("=" * 100)
    print("MATCH-POLICY SWEEP (re-scoring stored transcripts; no re-decoding)")
    print("SYNTHETIC / FILE-INJECTION - NOT REAL-WORLD RESULT")
    print("=" * 100)
    header = f"{'backend':<28}{'policy':<46}{'TP':>7}{'FN':>5}{'FP':>5}{'TN':>5}"
    print(header)
    print("-" * len(header))
    for key, data in stored["backends"].items():
        for require_hey in (True, False):
            for distance in (0, 1, 2):
                policy = MatchPolicy(require_hey=require_hey, max_distance=distance)
                tp = fn = fp = tn = 0
                for r in data["results"]:
                    fired = policy.fired_for(r["transcript"])
                    if r["kind"] == "positive":
                        if r["name"] in fired:
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if fired:
                            fp += 1
                        else:
                            tn += 1
                print(f"{key:<28}{policy.label():<46}{tp:>7}{fn:>5}{fp:>5}{tn:>5}")
    return 0


def stability(args) -> int:
    """Repeat one backend N times to check for drift between runs."""
    clips = load_clips()
    backend = {b.key: b for b in build_backends()}[args.backend]
    print(f"Repeating {backend.label} x{args.runs}  "
          "(SYNTHETIC / FILE-INJECTION - NOT REAL-WORLD RESULT)\n")
    print(f"{'run':>4}{'TP':>6}{'FP':>6}{'wall s':>9}{'RSS MB':>9}")
    print("-" * 34)
    first = None
    for run_index in range(1, args.runs + 1):
        began = time.perf_counter()
        results, _load = run_backend(backend, clips)
        wall = time.perf_counter() - began
        s = score(results)
        signature = tuple(sorted((r.clip, tuple(r.fired)) for r in results))
        first = first or signature
        print(f"{run_index:>4}{s['true_positives']:>6}{s['false_positives']:>6}"
              f"{wall:>9.1f}{_rss_mb():>9.1f}"
              + ("" if signature == first else "   <- DIFFERENT DECISIONS"))
    print("\nidentical decisions across all runs"
          if first else "")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TTS file-injection wake benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate"); g.add_argument("--force", action="store_true")
    g.set_defaults(func=generate)
    r = sub.add_parser("run"); r.set_defaults(func=run)
    s = sub.add_parser("sweep"); s.set_defaults(func=sweep)
    st = sub.add_parser("stability")
    st.add_argument("--backend", default="vosk_grammar")
    st.add_argument("--runs", type=int, default=3)
    st.set_defaults(func=stability)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
