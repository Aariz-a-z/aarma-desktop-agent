"""Record the wake-word evaluation corpus once, so it can be replayed forever.

Phase 2B was rescued because the audio survived a botched configuration. This
tool applies that lesson before the mistake rather than after: **one recording
session, then both candidates are scored offline on identical audio**, and every
retune of the match policy costs nothing.

Nothing here detects a wake word and nothing here is the production detector.
It records, measures levels, and writes a manifest. It reuses AARMA's existing
:class:`~speech.capture.MicrophoneStream` rather than opening a second capture
implementation, and it changes no setting.

Five modes, run in this order:

``distance-check``  Establishes whether 2 m and 3 m are even usable with this
                    laptop's microphone, with numbers, before an hour is spent
                    recording at a distance that turns out to be silence.
``positives``       The prompted wake phrases, one at a time, with retry/skip.
``negatives``       Near-misses, bare names, Phase 2B commands, conversation.
``ambient``         Long unattended capture, chunked. This is the only thing
                    that makes false-activations-per-hour meaningful.
``tts-echo``        AARMA speaking through the speakers, re-captured by the
                    microphone. File injection is generated separately and does
                    not need the room.

Every clip is written as 16 kHz mono WAV with a line in ``manifest.jsonl``
carrying phrase, distance, condition, rep, levels and timestamp. The session
resumes: completed cells are skipped on restart.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_RATE = 16_000

#: The four names under test. All four run the identical path -- there are no
#: per-name models in this architecture -- but they are scored separately
#: because their phonetics differ and averaging would hide that.
NAMES = ("Asma", "Aariz", "Nova", "Zara")

#: Ordered nearest-first, so the feasibility check degrades in one direction.
DISTANCES = ("near-mic", "0.5m", "1m", "2m", "3m")

CONDITIONS = ("normal", "quiet", "noisy", "fast")

CONDITION_HINT = {
    "normal": "normal speaking volume, as you would to a person beside you",
    "quiet": "quietly, as if someone is asleep in the next room",
    "noisy": "normal volume, but start some background noise first "
             "(music, a video, a fan)",
    "fast": "quickly and naturally, the way you would actually say it",
}

#: Must NOT trigger. Grouped because the groups fail for different reasons and
#: are reported separately.
NEGATIVES: Dict[str, List[str]] = {
    "near_miss": ["Hey Alma", "Hey Osman", "Asthma", "Hey Nora",
                  "Hey Sarah", "Hey Zahra", "Hey Nova Scotia"],
    "bare_name": ["Asma is a common name", "Nova is a car",
                  "I met Zara yesterday", "Aariz called me"],
    "command": ["Open Chrome.", "What time is it?", "Hello.",
                "Take a screenshot."],
    "conversation": [
        "So I was thinking we could go out for dinner later tonight",
        "The weather has been really strange this week, hasn't it",
        "Can you remind me to call the bank tomorrow afternoon",
    ],
}

#: Spoken by AARMA's own voice, played through the speakers, re-captured.
TTS_ECHO_PHRASES = [
    "Hey Asma is ready.",
    "I am Asma.",
    "Hey Aariz, that is your name.",
    "Nova, I've opened Chrome for you.",
    "I couldn't find Zara in your contacts.",
]

#: The fixed sentence used for the distance feasibility check.
DISTANCE_PHRASE = "Hey Asma, what time is it?"


#: Filenames are kept short on purpose. Clips now live under
#: ``<kind>/<run_id>/``, and on Windows the full path still has to fit inside
#: MAX_PATH (260 characters) -- a 40-character stem plus a deep corpus root was
#: enough to push the longest phrase over it and lose the write.
STEM_LIMIT = 24


def safe_stem(text: str, limit: int = STEM_LIMIT) -> str:
    """A filesystem-safe, length-capped fragment of ``text``."""
    return "".join(c if c.isalnum() else "_" for c in text)[:limit].strip("_")


def echo_stem(phrase: str, rep: int) -> str:
    """Filename stem for one TTS-echo clip.

    Factored out so the regression test can exercise exactly the naming the
    recorder uses, rather than a copy of it that could drift apart from the
    thing it is meant to protect.
    """
    return f"echo_{safe_stem(phrase)}_{rep:02d}"


def corpus_root() -> Path:
    from utils.paths import get_paths

    return get_paths().log_dir / "wakeword_corpus"


@dataclass
class ClipRecord:
    """One recorded clip and what was measured while recording it."""

    kind: str                      # positive | negative | ambient | tts_echo | distance
    #: The exact file this record describes. Unique for the life of the corpus.
    path: str
    #: Which invocation of the tool produced it. Older records predate run ids
    #: and carry "", which is why this has a default rather than being required.
    run_id: str = ""
    #: Bumped when a prompt is recorded more than once inside one run, so a
    #: retry adds a take instead of replacing the one before it.
    take: int = 1
    phrase: str = ""
    name: str = ""                 # Which of the four names, for positives.
    distance: str = ""
    condition: str = ""
    group: str = ""                # Negative subgroup.
    rep: int = 0
    duration_sec: float = 0.0
    peak: float = 0.0
    rms: float = 0.0
    speech_seconds: float = 0.0
    speech_start_sec: Optional[float] = None
    speech_end_sec: Optional[float] = None
    vad_triggered: bool = False
    snr_db: Optional[float] = None
    room_floor_rms: Optional[float] = None
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    #: Assigned when the corpus is split, never by the recorder.
    split: str = ""

    @property
    def cell(self) -> str:
        """Identity of this prompt, used to resume without re-recording."""
        return f"{self.kind}|{self.name}|{self.phrase}|{self.distance}|" \
               f"{self.condition}|{self.group}|{self.rep}"


class ClipWriteError(RuntimeError):
    """Raised when a clip could not be written to disk.

    ``write_wav`` reports failure by returning an empty string, which the
    recorder used to store as the record's path -- so a clip that was never
    written appeared in the manifest as though it had been. A phantom entry is
    worse than a loud failure: it inflates every count computed from the
    manifest and is invisible until someone tries to open the file.
    """


class ClipOverwriteError(RuntimeError):
    """Raised rather than replacing audio that already exists.

    A second ``tts-echo`` session once wrote over the first because both used
    ``echo_<phrase>_<rep>.wav``: fifteen recordings were destroyed and only a
    backup taken for another reason saved them. Recorded audio is the one thing
    in this project that cannot be regenerated, so the writer refuses instead of
    trusting the caller to have picked a fresh name.
    """


class Corpus:
    """The on-disk corpus: WAV files plus an append-only manifest.

    Files live under ``<kind>/<run_id>/``. The run id is the invocation's
    timestamp, so a second session sits beside the first instead of on top of
    it, and every previous run stays readable forever.
    """

    def __init__(self, root: Optional[Path] = None,
                 run_id: Optional[str] = None) -> None:
        self.root = root or corpus_root()
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.records: List[ClipRecord] = []
        self._load()

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.jsonl"

    # ------------------------------------------------------------- addressing

    def run_dir(self, kind: str) -> Path:
        """Where this invocation's clips of ``kind`` go."""
        return self.root / kind / self.run_id

    def reserve(self, kind: str, stem: str) -> Tuple[Path, int]:
        """A path guaranteed not to exist yet, plus its take number.

        Two collisions are possible and both are handled here rather than by
        every caller: another *run* (impossible now that the run id is in the
        path) and a *retry inside this run*, which appends ``-take2``,
        ``-take3`` and so on. Nothing is ever replaced.
        """
        folder = self.run_dir(kind)
        candidate = folder / f"{stem}.wav"
        take = 1
        while candidate.exists():
            take += 1
            candidate = folder / f"{stem}-take{take}.wav"
        return candidate, take

    def write_clip(self, audio, path: Path) -> str:
        """Write one clip, refusing to overwrite anything already there."""
        from speech.input_engine import write_wav

        if path.exists():
            raise ClipOverwriteError(
                f"{path} already exists. Recorded audio is never replaced -- "
                "use Corpus.reserve() to obtain a fresh path.")
        path.parent.mkdir(parents=True, exist_ok=True)
        written = write_wav(audio, path)
        if not written or not path.is_file():
            raise ClipWriteError(
                f"Could not write {path} ({len(str(path))}-character path). "
                "The recording was NOT saved. On Windows a path over 260 "
                "characters fails here; shorten the corpus root or the phrase.")
        return written

    def _load(self) -> None:
        if not self.manifest_path.exists():
            return
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.records.append(ClipRecord(**json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue

    def append(self, record: ClipRecord) -> None:
        """Persist immediately. A session interrupted at clip 200 keeps 199."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.records.append(record)
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            handle.flush()

    def done_cells(self) -> set:
        return {r.cell for r in self.records}

    def count(self, kind: str) -> int:
        return sum(1 for r in self.records if r.kind == kind)

    def summary(self) -> str:
        lines = [f"Corpus at {self.root}", ""]
        for kind in ("distance", "positive", "negative", "ambient", "tts_echo"):
            records = [r for r in self.records if r.kind == kind]
            seconds = sum(r.duration_sec for r in records)
            lines.append(f"  {kind:<12}{len(records):>5} clips"
                         f"{seconds / 60:>9.1f} min")
        return "\n".join(lines)


# --------------------------------------------------------------- recording

class Recorder:
    """Fixed-duration and press-to-stop capture on the existing stream."""

    def __init__(self, device_index: Optional[int] = None) -> None:
        """``device_index`` overrides the configured microphone for one run.

        Used to compare capture paths: the same physical Realtek microphone is
        reachable through MME, DirectSound and WASAPI, and they do not behave
        alike -- MME was measured zeroing 76% of samples where WASAPI zeroed
        none. This is a recording-tool option only; it changes nothing about
        which device AARMA itself uses.
        """
        from speech.capture import MicrophoneStream, resolve_device
        from speech.vad import VadTuning, create_detector

        if device_index is None:
            resolution = resolve_device(None)
            index = resolution.index
            self.device_name = (resolution.device.name if resolution.device
                                else "default")
        else:
            import sounddevice as sd

            info = sd.query_devices(device_index)
            api = sd.query_hostapis()[info["hostapi"]]["name"]
            index = device_index
            self.device_name = f"{info['name']}  [{api}, index {device_index}]"
        self._stream = MicrophoneStream(index)
        self._detector = create_detector()
        self._tuning = VadTuning()
        self._stop = threading.Event()

    def open(self) -> None:
        self._stream.open()

    def close(self) -> None:
        self._stream.close()

    @property
    def sample_rate(self) -> int:
        return self._stream.sample_rate

    def record(self, seconds: float):
        """Capture for a fixed duration. Returns 16 kHz mono float32."""
        import numpy as np

        from speech.capture import resample_to_16k

        chunks = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not self._stop.is_set():
            block, _overflow = self._stream.read_block()
            chunks.append(block.copy())
        if not chunks:
            return np.zeros(0, dtype="float32")
        audio = np.concatenate(chunks).astype("float32")
        if self._stream.sample_rate != SAMPLE_RATE:
            audio = resample_to_16k(audio, self._stream.sample_rate)
        return audio

    def measure(self, audio, floor_rms: Optional[float] = None) -> Dict[str, object]:
        """Levels, VAD boundaries and SNR for one buffer."""
        import numpy as np

        if audio is None or audio.size == 0:
            return {"duration_sec": 0.0, "peak": 0.0, "rms": 0.0,
                    "speech_seconds": 0.0, "vad_triggered": False,
                    "speech_start_sec": None, "speech_end_sec": None,
                    "snr_db": None}

        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio ** 2)))
        regions = self._detector.speech_regions(audio, self._tuning)
        speech_seconds = sum(r.duration for r in regions)

        snr = None
        if floor_rms and floor_rms > 0:
            if regions:
                speech = audio[regions[0].start:regions[-1].end]
                speech_rms = float(np.sqrt(np.mean(speech ** 2))) if speech.size else rms
            else:
                speech_rms = rms
            if speech_rms > 0:
                snr = 20.0 * math.log10(speech_rms / floor_rms)

        return {
            "duration_sec": round(audio.size / SAMPLE_RATE, 3),
            "peak": round(peak, 6),
            "rms": round(rms, 6),
            "speech_seconds": round(speech_seconds, 3),
            "vad_triggered": bool(regions),
            "speech_start_sec": (round(regions[0].start / SAMPLE_RATE, 3)
                                 if regions else None),
            "speech_end_sec": (round(regions[-1].end / SAMPLE_RATE, 3)
                               if regions else None),
            "snr_db": round(snr, 1) if snr is not None else None,
        }

    def room_floor(self, seconds: float = 2.0) -> float:
        """The quiet level of the room, used as the SNR reference."""
        import numpy as np

        audio = self.record(seconds)
        return float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0


def ask(prompt: str, options: str = "[Enter]=record  s=skip  q=quit") -> str:
    try:
        return input(f"{prompt}\n   {options} > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "q"


# ------------------------------------------------------------ step 1: range

def run_distance_check(args) -> int:
    """Measure whether each distance is usable at all, before recording 300 clips."""
    corpus = Corpus()
    recorder = Recorder()
    recorder.open()
    print("=" * 78)
    print("DISTANCE FEASIBILITY CHECK")
    print("=" * 78)
    print(f"Microphone: {recorder.device_name}  ({recorder.sample_rate} Hz)")
    print("\nThis establishes the maximum usable distance with numbers, so we do")
    print("not spend an hour recording at a distance that is really silence.\n")

    try:
        print("First, two seconds of room tone. Please stay quiet…")
        input("   [Enter] when ready > ")
        floor = recorder.room_floor(2.0)
        print(f"   room floor RMS = {floor:.6f}\n")

        results = []
        for distance in DISTANCES:
            answer = ask(f"Move to {distance} and say:  \"{DISTANCE_PHRASE}\"",
                         "[Enter]=record 4s  s=skip this distance  q=quit")
            if answer == "q":
                break
            if answer == "s":
                print("   skipped\n")
                continue

            print("   recording…")
            audio = recorder.record(4.0)
            metrics = recorder.measure(audio, floor)
            path, take = corpus.reserve(
                "distance", f"distance-{distance.replace('.', '_')}")
            written = corpus.write_clip(audio, path)

            corpus.append(ClipRecord(
                kind="distance", path=written, run_id=corpus.run_id, take=take,
                phrase=DISTANCE_PHRASE, distance=distance,
                room_floor_rms=round(floor, 8), **metrics))
            results.append((distance, metrics))
            print(f"   peak {metrics['peak']:.5f}   rms {metrics['rms']:.5f}   "
                  f"SNR {metrics['snr_db']} dB   "
                  f"VAD {'YES' if metrics['vad_triggered'] else 'NO'}\n")

        print("=" * 78)
        print(f"{'distance':<12}{'peak':>10}{'rms':>10}{'SNR dB':>9}{'VAD':>6}   verdict")
        print("-" * 78)
        feasible = []
        for distance, m in results:
            snr = m["snr_db"]
            ok = bool(m["vad_triggered"]) and snr is not None and snr >= 10.0
            if ok:
                feasible.append(distance)
            verdict = ("usable" if ok else
                       "NOT FEASIBLE — no voice detected" if not m["vad_triggered"]
                       else f"NOT FEASIBLE — SNR {snr} dB below 10 dB")
            print(f"{distance:<12}{m['peak']:>10.5f}{m['rms']:>10.5f}"
                  f"{(snr if snr is not None else float('nan')):>9.1f}"
                  f"{'Y' if m['vad_triggered'] else 'N':>6}   {verdict}")

        (corpus.root / "feasible_distances.json").write_text(
            json.dumps({"feasible": feasible, "room_floor_rms": floor,
                        "device": recorder.device_name}, indent=2),
            encoding="utf-8")
        print(f"\nUsable distances: {feasible or 'NONE'}")
        print("The positives run will prompt only these. Re-run this check to change it.")
    finally:
        recorder.close()
    return 0


def feasible_distances(corpus: Corpus) -> List[str]:
    path = corpus.root / "feasible_distances.json"
    if not path.exists():
        return list(DISTANCES)
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("feasible") or []
    except json.JSONDecodeError:
        return list(DISTANCES)


# --------------------------------------------------------- step 2: positives

def run_positives(args) -> int:
    corpus = Corpus()
    recorder = Recorder()
    distances = args.distances or feasible_distances(corpus)
    if not distances:
        print("No feasible distances recorded. Run 'distance-check' first.")
        return 2

    done = corpus.done_cells()
    plan = [(name, distance, condition, rep)
            for distance in distances
            for condition in CONDITIONS
            for name in NAMES
            for rep in range(1, args.reps + 1)]
    todo = [item for item in plan
            if f"positive|{item[0]}|Hey {item[0]}|{item[1]}|{item[2]}||{item[3]}"
            not in done]

    print("=" * 78)
    print(f"POSITIVES — {len(todo)} of {len(plan)} remaining")
    print("=" * 78)
    print("Say each phrase once, naturally, then wait. Recording is 3 seconds.")
    print("Retry anything you fluff — retries are kept and cost nothing.\n")

    recorder.open()
    floor = None
    try:
        print("Two seconds of room tone first. Please stay quiet…")
        input("   [Enter] when ready > ")
        floor = recorder.room_floor(2.0)
        print(f"   room floor RMS = {floor:.6f}\n")

        current_cell = None
        index = 0
        while index < len(todo):
            name, distance, condition, rep = todo[index]
            if (distance, condition) != current_cell:
                current_cell = (distance, condition)
                print("\n" + "-" * 78)
                print(f"  DISTANCE: {distance}     CONDITION: {condition}")
                print(f"  {CONDITION_HINT[condition]}")
                print("-" * 78)

            phrase = f"Hey {name}"
            answer = ask(f'[{index + 1}/{len(todo)}]  Say:  "{phrase}"  (rep {rep})',
                         "[Enter]=record  r=retry last  s=skip  q=quit")
            if answer == "q":
                break
            if answer == "s":
                index += 1
                continue
            if answer == "r":
                index = max(0, index - 1)
                continue

            print("   recording…")
            audio = recorder.record(args.seconds)
            metrics = recorder.measure(audio, floor)
            safe = distance.replace(".", "_")
            path, take = corpus.reserve(
                "positives", f"{name.lower()}_{safe}_{condition}_{rep:02d}")
            written = corpus.write_clip(audio, path)
            corpus.append(ClipRecord(
                kind="positive", path=written, run_id=corpus.run_id, take=take,
                phrase=phrase, name=name, distance=distance,
                condition=condition, rep=rep,
                room_floor_rms=round(floor, 8), **metrics))

            flag = "" if metrics["vad_triggered"] else "   ⚠ no voice detected"
            print(f"   peak {metrics['peak']:.5f}  SNR {metrics['snr_db']} dB{flag}")
            index += 1
    finally:
        recorder.close()
    print(f"\n{corpus.summary()}")
    return 0


def run_negatives(args) -> int:
    corpus = Corpus()
    recorder = Recorder()
    done = corpus.done_cells()

    plan = [(group, phrase, rep)
            for group, phrases in NEGATIVES.items()
            for phrase in phrases
            for rep in range(1, args.reps + 1)]
    todo = [item for item in plan
            if f"negative|||{args.distance}||{item[0]}|{item[2]}" not in done
            and f"negative||{item[1]}|{args.distance}||{item[0]}|{item[2]}" not in done]

    print("=" * 78)
    print(f"NEGATIVES — {len(todo)} clips.  None of these may ever trigger a wake.")
    print("=" * 78)
    print(f"Speak all of these at {args.distance}, normal volume.\n")

    recorder.open()
    try:
        print("Room tone first. Please stay quiet…")
        input("   [Enter] when ready > ")
        floor = recorder.room_floor(2.0)
        print(f"   room floor RMS = {floor:.6f}\n")

        index = 0
        while index < len(todo):
            group, phrase, rep = todo[index]
            answer = ask(f'[{index + 1}/{len(todo)}]  ({group})  Say:  "{phrase}"',
                         "[Enter]=record  r=retry last  s=skip  q=quit")
            if answer == "q":
                break
            if answer == "s":
                index += 1
                continue
            if answer == "r":
                index = max(0, index - 1)
                continue

            print("   recording…")
            audio = recorder.record(args.seconds)
            metrics = recorder.measure(audio, floor)
            path, take = corpus.reserve(
                "negatives", f"{safe_stem(group, 12)}_{safe_stem(phrase)}_{rep:02d}")
            written = corpus.write_clip(audio, path)
            corpus.append(ClipRecord(
                kind="negative", path=written, run_id=corpus.run_id, take=take,
                phrase=phrase, group=group, distance=args.distance, rep=rep,
                room_floor_rms=round(floor, 8), **metrics))
            print(f"   peak {metrics['peak']:.5f}  SNR {metrics['snr_db']} dB")
            index += 1
    finally:
        recorder.close()
    print(f"\n{corpus.summary()}")
    return 0


# ----------------------------------------------------------- step 2: ambient

def run_ambient(args) -> int:
    """Unattended chunked capture. This is what makes per-hour rates real."""
    corpus = Corpus()
    recorder = Recorder()
    folder = corpus.run_dir("ambient")
    chunk = args.chunk_minutes * 60.0
    total = args.hours * 3600.0

    print("=" * 78)
    print("AMBIENT CAPTURE")
    print("=" * 78)
    print(f"Recording up to {args.hours:g} h in {args.chunk_minutes:g}-minute chunks.")
    print(f"Files land in {folder}")
    print("\nLeave the room normal: fan, keyboard, music, a video, conversation.")
    print("Do NOT say any of the four wake phrases during this capture.")
    print("Press Ctrl+C at any time — completed chunks are already saved.\n")
    input("   [Enter] to begin > ")

    recorder.open()
    captured = 0.0
    index = 0
    try:
        while captured < total:
            index += 1
            seconds = min(chunk, total - captured)
            started = datetime.now().strftime("%H:%M:%S")
            print(f"   chunk {index:03d} started {started} "
                  f"({seconds / 60:.0f} min)…", flush=True)
            audio = recorder.record(seconds)
            metrics = recorder.measure(audio)
            path, take = corpus.reserve("ambient", f"ambient-{index:03d}")
            written = corpus.write_clip(audio, path)
            corpus.append(ClipRecord(
                kind="ambient", path=written, run_id=corpus.run_id, take=take,
                group=args.label, **metrics))
            captured += metrics["duration_sec"]
            print(f"   saved {path.name}  "
                  f"{metrics['duration_sec'] / 60:.1f} min  "
                  f"peak {metrics['peak']:.5f}  "
                  f"total {captured / 3600:.2f} h")
    except KeyboardInterrupt:
        print("\n   stopped by user — completed chunks are saved")
    finally:
        recorder.close()
    print(f"\nAmbient captured: {captured / 3600:.2f} hours")
    return 0


# ---------------------------------------------------------- step 2: TTS echo

def run_tts_echo(args) -> int:
    """AARMA speaks through the speakers; the microphone records the room.

    File injection is generated separately and needs no room. This is the path
    that actually fails in practice, because the speakers are in the same room
    as the microphone and file injection cannot show that.
    """
    from config.settings import load_config
    from speech.engine import SpeechEngine

    corpus = Corpus()
    speech = SpeechEngine(load_config())
    recorder = Recorder(getattr(args, "device", None))
    print("=" * 78)
    print("TTS ACOUSTIC LOOP")
    print("=" * 78)
    print("AARMA will speak each phrase through your speakers while the")
    print("microphone records the room. Set the volume to what you would")
    print("actually use, and do not speak yourself.\n")
    print("If your speakers are muted or the volume is very low, this measures")
    print("nothing — the whole point is the sound reaching the microphone.\n")
    input("   [Enter] to begin > ")

    # ``speak`` queues and returns immediately, so polling ``is_speaking``
    # straight afterwards can read False before the worker has even started.
    # The finished callback is the only reliable end-of-speech signal.
    spoken = threading.Event()
    speech.on_finished(lambda _outcome: spoken.set())
    speech.start()

    # ``--limit`` exists so the capture can be proved on a single clip before
    # committing a whole set to a room that may not be delivering audio at all.
    phrases = TTS_ECHO_PHRASES[:args.limit] if args.limit else TTS_ECHO_PHRASES

    recorder.open()
    try:
        for index, phrase in enumerate(phrases, start=1):
            for rep in range(1, args.reps + 1):
                print(f"   [{index}/{len(phrases)}] rep {rep}: "
                      f"speaking {phrase!r}…")
                captured: List = []

                def capture() -> None:
                    captured.append(recorder.record(args.seconds))

                worker = threading.Thread(target=capture, daemon=True)
                worker.start()
                time.sleep(0.2)          # Let capture start before audio does.

                spoken.clear()
                if not speech.speak(phrase, interrupt=True):
                    print("      ⚠ speech engine refused to speak — "
                          "is voice output enabled and permitted?")
                spoken.wait(timeout=args.seconds)
                worker.join(timeout=args.seconds + 5)

                audio = captured[0] if captured else None
                metrics = recorder.measure(audio)
                path, take = corpus.reserve("tts_echo",
                                            echo_stem(phrase, rep))
                written = corpus.write_clip(audio, path)
                corpus.append(ClipRecord(
                    kind="tts_echo", path=written, run_id=corpus.run_id,
                    take=take, phrase=phrase, rep=rep,
                    group="acoustic_loop", **metrics))
                print(f"      peak {metrics['peak']:.5f}  "
                      f"VAD {'YES' if metrics['vad_triggered'] else 'no'}")
                time.sleep(0.4)
    finally:
        recorder.close()
        speech.shutdown()
    print(f"\n{corpus.summary()}")
    return 0


def run_status(args) -> int:
    corpus = Corpus()
    print(corpus.summary())
    print(f"\nFeasible distances: {feasible_distances(corpus)}")
    positives = [r for r in corpus.records if r.kind == "positive"]
    if positives:
        print("\nPositives per name:")
        for name in NAMES:
            rows = [r for r in positives if r.name == name]
            no_vad = sum(1 for r in rows if not r.vad_triggered)
            print(f"  {name:<8}{len(rows):>4} clips"
                  f"   {no_vad} with no voice detected")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record the wake-word evaluation corpus")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("distance-check",
                                  help="measure usable microphone range first")
    check.set_defaults(func=run_distance_check)

    positives = subparsers.add_parser("positives", help="record the wake phrases")
    positives.add_argument("--reps", type=int, default=4)
    positives.add_argument("--seconds", type=float, default=3.0)
    positives.add_argument("--distances", nargs="*", default=None)
    positives.set_defaults(func=run_positives)

    negatives = subparsers.add_parser("negatives", help="record the hard negatives")
    negatives.add_argument("--reps", type=int, default=3)
    negatives.add_argument("--seconds", type=float, default=4.0)
    negatives.add_argument("--distance", default="1m")
    negatives.set_defaults(func=run_negatives)

    ambient = subparsers.add_parser("ambient", help="unattended chunked capture")
    ambient.add_argument("--hours", type=float, default=4.0)
    ambient.add_argument("--chunk-minutes", type=float, default=5.0)
    ambient.add_argument("--label", default="room")
    ambient.set_defaults(func=run_ambient)

    echo = subparsers.add_parser("tts-echo", help="speakers to microphone loop")
    echo.add_argument("--reps", type=int, default=3)
    echo.add_argument("--seconds", type=float, default=6.0)
    echo.add_argument("--limit", type=int, default=0,
                      help="record only the first N phrases (0 = all)")
    echo.add_argument("--device", type=int, default=None,
                      help="input device index, to compare capture paths")
    echo.set_defaults(func=run_tts_echo)

    status = subparsers.add_parser("status", help="what has been recorded so far")
    status.set_defaults(func=run_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
