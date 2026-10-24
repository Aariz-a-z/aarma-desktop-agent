"""Teach AARMA a new wake phrase from the user's own voice.

Renaming the assistant to Friday does not make it able to hear "Hey Friday".
Both stages of the detector are specific to one phrase: stage 1 matches against
enrollment embeddings, and stage 2 is a classifier trained to tell that phrase
from everything else.  This module builds both from recordings the user makes
here, and refuses to install the result unless it survives a test on data it
was never trained on.

What it reuses, and why that matters
------------------------------------
Nothing here opens a microphone of its own: recording goes through
:class:`tools.wakeword_corpus.Recorder`, which is the existing
:class:`~speech.capture.MicrophoneStream` plus the existing VAD.  The features
and the model come from :mod:`speech.wakeword_cascade` -- the same
``_log_mel`` and the same architecture the detector will run at inference.
Training against a private copy of the front end is how Phase 13 lost 25 of 40
detections to a windowing mismatch nobody could see; importing the real one
makes that class of bug impossible rather than unlikely.

The recipe is Phase 15's, unchanged: log-mel 40, four conv blocks, mean+max
pooling, one logit, trained from scratch on augmented positives against speech
and noise negatives.  Only the data is new.  The *thresholds* are not inherited
-- 0.10 was calibrated for one particular model of one particular phrase, and
reusing it for another would be a guess dressed as a measurement.

Honest limits, by construction
------------------------------
* Fifteen recordings from one person in one room is a small, narrow dataset.
  A detector built from it is validated for that voice in that room and nothing
  more; :func:`quality_gate` reports what was actually measured and the numbers
  are small.
* Far-field robustness is *claimed only if measured*.  If the user records at
  one distance, the report says so.
* The gate is a floor, not a benchmark.  Passing it means "not obviously
  broken", which is the most that a few minutes of enrollment can establish.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from config.settings import normalise_wake_phrase, wake_phrase_for_name
from speech.wakeword_store import (
    STORE,
    CustomWakeWord,
    CustomWakeWordError,
    CustomWakeWordStore,
    validate_reference,
)
from utils.logger import get_logger

logger = get_logger("speech.wakeword_enrollment")

SAMPLE_RATE = 16_000

#: How many recordings to ask for.  Below :data:`MIN_RECORDINGS` the split
#: leaves nothing to validate on, and a detector that cannot be validated must
#: not be installed.
TARGET_RECORDINGS = 25
MIN_RECORDINGS = 15
MAX_RECORDINGS = 40

#: Recordings held back from training and from threshold selection, used once
#: to decide whether the detector works.  A fraction, with a floor.
HELD_OUT_FRACTION = 0.25
MIN_HELD_OUT = 3

#: Recordings used to choose the threshold.  Separate from both the training
#: set (which the model memorises) and the held-out set (which must stay
#: untouched for the gate to mean anything).
CALIBRATION_FRACTION = 0.2
MIN_CALIBRATION = 2
MIN_TRAIN_CLIPS = 5

#: Seconds captured per prompt.  The detector scores a 1.5 s window inside a
#: 2.75 s context, so the clip has to comfortably contain the phrase plus room
#: on both sides.
CLIP_SECONDS = 3.2

#: Training conditions, copied from the frozen Phase-15 configuration.  These
#: simulate distance and noise so that ten close recordings still produce a
#: detector that survives a few metres -- but simulation is not measurement,
#: which is why the gate reports real recordings separately.
CONDITIONS: Dict[str, Dict[str, Optional[float]]] = {
    "clean":       {"t60": 0.0,  "snr": None, "gain_db": 0.0},
    "near":        {"t60": 0.0,  "snr": 30.0, "gain_db": 0.0},
    "1m_like":     {"t60": 0.20, "snr": 25.0, "gain_db": -3.0},
    "2m_like":     {"t60": 0.35, "snr": 18.0, "gain_db": -7.0},
    "3m_like":     {"t60": 0.50, "snr": 14.0, "gain_db": -11.0},
    "reverberant": {"t60": 0.65, "snr": 20.0, "gain_db": -9.0},
    "noisy":       {"t60": 0.15, "snr": 10.0, "gain_db": -4.0},
}
REPEATS_PER_CONDITION = 4
EPOCHS = 40
LEARNING_RATE = 0.002
BATCH_SIZE = 64
DEFAULT_SEED = 2026

#: Regularisation: heavier settings were tried and REJECTED after measuring
#: precision as well as recall.
#:
#: Dropout 0.5 with weight decay 1e-2 and label smoothing 0.05 raised recall on
#: unseen recordings from 0.50 to 0.72, which looked like a clear win -- until
#: the same model was put through the full gate, where it accepted 179 of 240
#: recordings of *other people's* wake words and 4,761 room-noise windows an
#: hour against the shipped detector's 98.1.  It had simply learned to say yes.
#:
#: The lesson is recorded here rather than the settings: a change measured on
#: recall alone is not measured.  Anything proposed for this model has to be
#: judged on the whole gate, and the gate is what caught this one.
WEIGHT_DECAY = 1e-4

#: Candidate stage-2 thresholds, the Phase-15 grid.
THRESHOLD_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                  0.60, 0.70, 0.80, 0.90, 0.95)

#: Ambient recordings the trainer may learn from.  The corpus is ordered, and
#: Phase 15 declared files 1-8 the dev set; everything past that is held out and
#: is only ever scored.  Training on the evaluation ambient would make the
#: false-activation number meaningless, so the split is enforced here rather
#: than left to callers.
AMBIENT_DEV_FILES = 8

#: How many window-sized slices of that dev ambient to add as negatives.
#:
#: Without them a custom detector fires on *digital near-silence*: the audit
#: found its highest-scoring room-noise windows peaking at 0.0001 with an RMS of
#: 0.00001, scored 0.97-0.99, where the shipped detector scores 0.0000. The
#: cause is that every other negative contains *something* -- a name, a
#: sentence, LibriSpeech, MUSAN noise -- while the log-mel front end applies
#: per-utterance CMVN, which scales an empty room's noise floor up to unit
#: variance and hands the model a pattern nothing has told it to reject.
#:
#: Six hundred slices is roughly a quarter of the speech-like negatives, enough
#: to cover the region without swamping the phonetic ones.
SILENCE_NEGATIVE_WINDOWS = 600

#: Stage 1 is few-shot and is not retrained, so it keeps the operating point it
#: was measured at.
STAGE1_THRESHOLD = 0.70

#: The shipped "Hey Aariz" verifier's own acceptance rate on 3.03 h of recorded
#: room audio, measured with :func:`_ambient_fires` at its frozen threshold of
#: 0.10, over the 29 held-out ambient recordings -- the same material the gate
#: scores a candidate on, so bar and measurement come from the same audio (the
#: figure over all 37 files, including the 8 the trainer may use, is 98.1).
#: It is the anchor for the noise check below, and it is deliberately a
#: measurement rather than a chosen number: the first version of this gate used
#: a plausible-looking 12/h, which would have rejected the validated detector
#: this project already ships.
#:
#: It is high because it is the *verifier alone*.  In the real cascade the
#: always-on stage proposes about 1.3 candidates an hour, so almost none of
#: these windows are ever scored -- Phase 15 measured 280/h standalone against
#: roughly 0.42/h in the cascade.  That is also why passing this check says
#: nothing about the cascade's false-activation rate, which remains unmeasured
#: for any newly enrolled phrase.
SHIPPED_STANDALONE_FA_PER_HOUR = 87.2

#: A candidate may be no worse at ignoring room noise than the model already in
#: use.  Rounded up from the anchor so that measurement noise on a three-hour
#: recording does not decide the verdict.
MAX_STANDALONE_FA_PER_HOUR = 100.0

#: The share of held-back recordings a detector must recognise before it is
#: installed.  Set from the product requirement rather than from what a given
#: enrollment happens to achieve.
MIN_HELD_OUT_RECALL = 0.90


# ------------------------------------------------------------------- prompts

@dataclass(frozen=True)
class EnrollmentPrompt:
    """One recording to make: what to say, and how."""

    index: int
    phrase: str
    distance: str
    condition: str
    hint: str

    @property
    def label(self) -> str:
        return f"{self.index}. {self.distance}, {self.condition}"


_VARIATION = (
    ("near", "normal", "normal speaking volume, close to the laptop"),
    ("near", "normal", "again, normally -- do not try to match the last one"),
    ("near", "quiet", "quietly, as if someone were asleep nearby"),
    ("near", "fast", "quickly and naturally"),
    ("0.5m", "normal", "about half a metre away, normal volume"),
    ("0.5m", "loud", "half a metre away, a little louder"),
    ("1m", "normal", "about a metre away, normal volume"),
    ("1m", "normal", "a metre away again, slightly different intonation"),
    ("1m", "fast", "a metre away, spoken quickly"),
    ("near", "normal", "close again, normal volume"),
    ("0.5m", "quiet", "half a metre away, quietly"),
    ("1m", "quiet", "a metre away, quietly"),
    ("near", "loud", "close, a little louder than usual"),
    ("0.5m", "fast", "half a metre away, spoken quickly"),
    ("1m", "loud", "a metre away, a little louder"),
    # Beyond fifteen the plan keeps adding variety rather than repeating
    # itself: pace, emphasis and where the phrase sits in the recording all
    # change what the model sees, and recall was measured to rise with the
    # number of genuinely different examples, not with the number of takes.
    ("near", "slow", "close, but slowly and deliberately"),
    ("0.5m", "normal", "half a metre away, as if asking a question"),
    ("1m", "slow", "a metre away, slowly"),
    ("near", "normal", "close, starting as soon as recording begins"),
    ("0.5m", "loud", "half a metre away, as if calling across the room"),
    ("1m", "normal", "a metre away, leaving a pause before you start"),
    ("near", "fast", "close and quick, the way you would in a hurry"),
    ("0.5m", "slow", "half a metre away, slowly"),
    ("1m", "fast", "a metre away, quickly"),
    ("near", "quiet", "close and quiet, almost under your breath"),
    ("0.5m", "normal", "half a metre away, flat and unemphatic"),
    ("1m", "normal", "a metre away, with the stress on the second word"),
)


def enrollment_plan(phrase: str, count: int = TARGET_RECORDINGS
                    ) -> Tuple[EnrollmentPrompt, ...]:
    """The recordings to ask for, varied on purpose.

    Fifteen identical recordings teach the model one delivery of one phrase,
    and it then fails the first time the user says it while walking away.  The
    plan therefore moves the speaker and changes the delivery, and repeats the
    pattern if more than fifteen are requested.
    """
    phrase = normalise_wake_phrase(phrase)
    count = max(MIN_RECORDINGS, min(int(count), MAX_RECORDINGS))
    prompts = []
    for i in range(count):
        distance, condition, hint = _VARIATION[i % len(_VARIATION)]
        prompts.append(EnrollmentPrompt(index=i + 1, phrase=phrase,
                                        distance=distance, condition=condition,
                                        hint=hint))
    return tuple(prompts)



def _stratified_split(clips: Sequence["EnrollmentClip"], rng, *,
                      held_count: int, calib_count: int):
    """Assign recordings to train / calibration / held-out, proportionally.

    Two properties matter, and the previous version had only the first:

    * each group -- one distance and delivery, "1m quiet" -- is shuffled, so
      which of its recordings goes where is random;
    * each *split* gets a share of every group, so the held-out set reflects
      the range the user recorded instead of one corner of it.

    The second is what was missing.  Groups are dealt round-robin, and each
    successive recording goes to whichever split is furthest below its target
    share, so consecutive picks -- which come from different groups -- land in
    different splits.

    Returns ``(train_idx, calib_idx, held_idx)`` as sets of positions.
    """
    groups: Dict[str, List[int]] = {}
    for index, clip in enumerate(clips):
        key = f"{clip.prompt.distance}|{clip.prompt.condition}"
        groups.setdefault(key, []).append(index)
    for key in groups:
        rng.shuffle(groups[key])

    keys = list(groups)
    rng.shuffle(keys)
    order: List[int] = []
    while any(groups.values()):
        for key in keys:
            if groups[key]:
                order.append(groups[key].pop())
        rng.shuffle(keys)

    train_count = max(0, len(clips) - held_count - calib_count)
    targets = {"held": held_count, "calib": calib_count, "train": train_count}
    taken = {"held": 0, "calib": 0, "train": 0}
    buckets: Dict[str, set] = {"held": set(), "calib": set(), "train": set()}
    for index in order:
        # Whichever split is proportionally emptiest takes the next recording.
        name = min((n for n in targets if taken[n] < targets[n]),
                   key=lambda n: (taken[n] / targets[n]) if targets[n] else 1.0,
                   default="train")
        buckets[name].add(index)
        taken[name] += 1
    return buckets["train"], buckets["calib"], buckets["held"]

# ----------------------------------------------------------------- recording

#: Enrollment's own recording-quality gate.
#:
#: The first version rejected anything whose peak fell below an absolute 0.02.
#: That was wrong, and measurably so: across the 320 wake-phrase recordings
#: already in this machine's corpus -- made by this same Recorder, on this same
#: microphone -- the quietest usable clip peaks at 0.0141, and ambient room
#: recordings containing no speech at all reach 0.47.  Absolute level does not
#: separate speech from silence; it only measures how much gain the sound card
#: happens to apply, which differs per machine and per Windows setting.
#:
#: Every number below is instead taken from the *minimum* observed across those
#: 320 known-good recordings, with headroom, so that a clip the rest of AARMA
#: would happily transcribe is never refused here.
#:
#:   measured minimum        threshold used
#:   peak / room floor 14.8  ->  8.0
#:   SNR 7.4 dB              ->  4.0 dB
#:   voiced speech 0.92 s    ->  0.45 s
#:
#: The one absolute number left is a dead-microphone check, and it is three
#: orders of magnitude below speech: a silent room on this machine measures a
#: peak of about 0.0001, so anything under 0.002 means the device is muted,
#: disconnected or returning zeros -- not that the speaker was quiet.
SILENT_PEAK = 0.002
MIN_PEAK_OVER_FLOOR = 8.0
MIN_SNR_DB = 4.0
MIN_SPEECH_SECONDS = 0.45
CLIPPED_PEAK = 0.99

#: Above this fraction of clipped samples the waveform is distorted enough that
#: training on it teaches the model the distortion.
MAX_CLIPPED_FRACTION = 0.01


@dataclass(frozen=True)
class ClipQuality:
    """What was measured in one recording, and whether it can be used.

    The single structured result both front ends render: the console prints it,
    the dialog shows it, and neither decides anything for itself.  That is what
    stops the two from drifting into disagreeing about what is acceptable.
    """

    accepted: bool
    code: str                  # ok | silent | no_speech | too_short | low_snr | clipped
    reason: str = ""           # what went wrong, in a sentence
    advice: str = ""           # what to do about it
    sample_rate: int = SAMPLE_RATE
    samples: int = 0
    duration: float = 0.0
    peak: float = 0.0
    rms: float = 0.0
    speech_seconds: float = 0.0
    speech_ratio: float = 0.0
    snr_db: Optional[float] = None
    floor_rms: float = 0.0
    peak_over_floor: Optional[float] = None
    clipped_fraction: float = 0.0
    all_zero: bool = False

    def diagnostics(self) -> str:
        """One line of numbers, for the log and for --diagnose.

        Deliberately only measurements: no audio, and nothing that could
        reconstruct what was said.
        """
        parts = [f"sr={self.sample_rate}", f"samples={self.samples}",
                 f"dur={self.duration:.2f}s", f"peak={self.peak:.4f}",
                 f"rms={self.rms:.5f}", f"speech={self.speech_seconds:.2f}s",
                 f"ratio={self.speech_ratio:.2f}"]
        if self.snr_db is not None:
            parts.append(f"snr={self.snr_db:.1f}dB")
        if self.peak_over_floor is not None:
            parts.append(f"peak/floor={self.peak_over_floor:.1f}")
        if self.all_zero:
            parts.append("ALL-ZERO")
        parts.append("accepted" if self.accepted else f"rejected:{self.code}")
        return "  ".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def assess_clip(audio, measure: Dict[str, Any],
                floor_rms: float = 0.0) -> ClipQuality:
    """Decide whether one recording is worth training on.

    Ordered so the *most diagnostic* failure is the one reported: a muted
    device is not "too quiet", and a clip with no speech in it is not "poor
    signal to noise".  Each rejection names something the user can act on.
    """
    import numpy as np

    samples = int(getattr(audio, "size", 0) or 0)
    duration = samples / float(SAMPLE_RATE) if samples else 0.0
    peak = float(measure.get("peak") or 0.0)
    rms = float(measure.get("rms") or 0.0)
    speech_seconds = float(measure.get("speech_seconds") or 0.0)
    snr_db = measure.get("snr_db")
    snr_db = float(snr_db) if snr_db is not None else None
    ratio = (speech_seconds / duration) if duration > 0 else 0.0
    over_floor = (peak / floor_rms) if floor_rms > 0 else None
    all_zero = bool(samples and not np.any(np.asarray(audio)))
    clipped_fraction = (float(np.mean(np.abs(np.asarray(audio)) >= CLIPPED_PEAK))
                        if samples else 0.0)

    common = dict(samples=samples, duration=round(duration, 3), peak=peak, rms=rms,
                  speech_seconds=speech_seconds, speech_ratio=round(ratio, 3),
                  snr_db=snr_db, floor_rms=floor_rms,
                  peak_over_floor=(round(over_floor, 1) if over_floor else None),
                  clipped_fraction=round(clipped_fraction, 4), all_zero=all_zero)

    def bad(code: str, reason: str, advice: str) -> ClipQuality:
        return ClipQuality(accepted=False, code=code, reason=reason,
                           advice=advice, **common)

    if samples == 0:
        return bad("silent", "the microphone returned no audio at all",
                   "Check that the right input device is selected.")
    if all_zero or peak < SILENT_PEAK:
        return bad("silent", "the microphone returned silence",
                   "It may be muted or disabled in Windows sound settings, or "
                   "another application may be holding it.")
    if clipped_fraction > MAX_CLIPPED_FRACTION:
        return bad("clipped", "that recording is distorted from being too loud",
                   "Move back a little, or lower the microphone level.")
    if not measure.get("vad_triggered") or speech_seconds <= 0.0:
        return bad("no_speech", "no speech was detected in that recording",
                   "Wait for the prompt, then say the phrase.")
    if speech_seconds < MIN_SPEECH_SECONDS:
        return bad("too_short", f"only {speech_seconds:.2f} s of speech was heard",
                   "Say the whole phrase, without rushing it.")
    if over_floor is not None and over_floor < MIN_PEAK_OVER_FLOOR:
        return bad("low_snr",
                   f"the phrase was only {over_floor:.1f} times the room's own "
                   "level, which is too close to the background",
                   "Move closer to the microphone, or quieten the room.")
    if snr_db is not None and snr_db < MIN_SNR_DB:
        return bad("low_snr", f"the signal-to-noise ratio was {snr_db:.1f} dB",
                   "Move closer to the microphone, or quieten the room.")

    return ClipQuality(accepted=True, code="ok", **common)


@dataclass
class EnrollmentClip:
    """One recorded example, with what the VAD measured in it."""

    prompt: EnrollmentPrompt
    audio: Any                              # np.ndarray, 16 kHz mono float32
    measure: Dict[str, Any] = field(default_factory=dict)
    quality: Optional[ClipQuality] = None

    @property
    def usable(self) -> bool:
        """Whether there is speech in it at a level worth training on."""
        return self.quality is not None and self.quality.accepted

    def rejection(self) -> str:
        return "" if self.quality is None or self.quality.accepted \
            else self.quality.reason

    def advice(self) -> str:
        return "" if self.quality is None or self.quality.accepted \
            else self.quality.advice


class EnrollmentRecorder:
    """The existing corpus recorder, opened once for a whole session.

    A thin wrapper, kept so that callers never touch the microphone directly
    and so the session always closes the device even when a prompt raises.
    """

    def __init__(self, device_index: Optional[int] = None) -> None:
        from tools.wakeword_corpus import Recorder

        self._recorder = Recorder(device_index)
        self._open = False
        self.floor_rms: float = 0.0
        self._floor_peak: float = 0.0
        self._last_room_peak: float = 0.0

    def __enter__(self) -> "EnrollmentRecorder":
        self._recorder.open()
        self._open = True
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._open:
            self._open = False
            try:
                self._recorder.close()
            except Exception:  # noqa: BLE001 - releasing must not raise
                logger.debug("Enrollment microphone would not close cleanly")

    def measure_room(self, seconds: float = 1.5) -> float:
        """Sample the quiet room, and remember it as the reference level.

        Every later judgement is made relative to this rather than to an
        absolute number, which is what makes the gate work on a microphone with
        a different gain from the one it was written on.
        """
        self.floor_rms = self._recorder.room_floor(seconds)
        return self.floor_rms

    def readiness(self) -> Tuple[bool, str]:
        """Whether the device is actually delivering audio, before prompting.

        Catches the failure that is most confusing to meet halfway through
        enrollment -- a muted or absent input -- while it can still be reported
        as one clear sentence instead of fifteen rejected recordings.
        """
        if self.floor_rms <= 0.0:
            return False, ("the microphone returned nothing at all; check that "
                           "the right input device is selected and not muted")
        if self.floor_rms < 1e-7:
            return False, ("the microphone is returning digital silence; it is "
                           "probably muted in Windows sound settings")
        if self.floor_rms > 0.05:
            return False, (f"the room is very loud (floor RMS {self.floor_rms:.4f}); "
                           "enrollment needs somewhere quieter")
        return True, "microphone ready"

    def capture(self, prompt: EnrollmentPrompt,
                seconds: float = CLIP_SECONDS) -> EnrollmentClip:
        """Record one example and judge it.  The single recording entry point.

        Both front ends call this and neither judges anything itself, so the
        console and the settings dialog cannot come to different conclusions
        about the same audio.
        """
        audio = self._recorder.record(seconds)
        measure = self._recorder.measure(audio, self.floor_rms or None)
        quality = assess_clip(audio, measure, self.floor_rms)
        logger.info("Enrollment clip: %s", quality.diagnostics(), extra={
            "context": {"prompt": prompt.index, "code": quality.code}})
        return EnrollmentClip(prompt=prompt, audio=audio, measure=measure,
                              quality=quality)

    @property
    def device_name(self) -> str:
        return getattr(self._recorder, "device_name", "default")

    @property
    def sample_rate(self) -> int:
        return int(getattr(self._recorder, "sample_rate", SAMPLE_RATE))



# --------------------------------------------------------------- persistence

def enrolment_sessions_root() -> Path:
    from utils.paths import get_paths

    return get_paths().log_dir / "enrolment"


class EnrollmentSession:
    """Somewhere for one enrollment attempt's recordings to live.

    Enrollment used to hold the recordings in memory and drop them the moment
    it finished.  When a run failed validation -- which is the case that most
    needs looking at -- fifteen recordings of the user's voice went with it, and
    the only way to ask "why did held-out clip 3 score 0.33?" was to record
    fifteen more.

    Clips are written as they are accepted, so an interrupted session keeps
    what it already had, and the verdict is written beside them afterwards.
    Recordings of the user's own voice, kept under their data directory with
    everything else they recorded.
    """

    def __init__(self, phrase: str, *, root: Optional[Path] = None) -> None:
        from speech.wakeword_store import slug_for_phrase

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = root or enrolment_sessions_root()
        self.directory = base / f"{slug_for_phrase(phrase) or 'wake'}-{stamp}"
        self.phrase = normalise_wake_phrase(phrase)
        self._saved: List[Dict[str, Any]] = []

    def save_clip(self, index: int, clip: "EnrollmentClip") -> Optional[Path]:
        """Write one accepted recording.  Never raises: losing a clip is bad,
        but failing the whole enrollment because a disk write failed is worse."""
        from speech.input_engine import write_wav

        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"clip-{index:02d}.wav"
            if not write_wav(clip.audio, path):
                raise OSError("write_wav reported failure")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Enrollment recording could not be saved", extra={
                "context": {"index": index, "error": str(exc)}})
            return None
        self._saved.append({
            "index": index, "file": path.name,
            "prompt": {"distance": clip.prompt.distance,
                       "condition": clip.prompt.condition,
                       "hint": clip.prompt.hint},
            "quality": clip.quality.as_dict() if clip.quality else {},
            "measure": {k: v for k, v in clip.measure.items()
                        if isinstance(v, (int, float, bool, str)) or v is None},
        })
        self._write_manifest()
        return path

    def save_result(self, result: "EnrollmentResult") -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / "result.json").write_text(
                json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Enrollment result could not be saved", extra={
                "context": {"error": str(exc)}})

    def _write_manifest(self) -> None:
        try:
            (self.directory / "session.json").write_text(json.dumps({
                "phrase": self.phrase,
                "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "clips": self._saved,
            }, indent=2, default=str), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Enrollment manifest could not be written", extra={
                "context": {"error": str(exc)}})


def load_session(directory: Path) -> Tuple[str, List["EnrollmentClip"]]:
    """Read a saved session back as clips, for re-running without re-recording.

    Returns ``(phrase, clips)``.  Every clip is re-assessed rather than trusting
    the stored verdict, so a change to the quality gate is reflected on replay.
    """
    directory = Path(directory)
    manifest = directory / "session.json"
    if not manifest.is_file():
        raise CustomWakeWordError(f"no enrollment session at {directory}")
    blob = json.loads(manifest.read_text(encoding="utf-8"))
    phrase = normalise_wake_phrase(blob.get("phrase", ""))

    clips: List[EnrollmentClip] = []
    for entry in blob.get("clips", []):
        path = directory / str(entry.get("file", ""))
        if not path.is_file():
            continue
        audio = _read_wav(str(path))
        prompt_raw = entry.get("prompt") or {}
        prompt = EnrollmentPrompt(index=int(entry.get("index", len(clips) + 1)),
                                  phrase=phrase,
                                  distance=str(prompt_raw.get("distance", "")),
                                  condition=str(prompt_raw.get("condition", "")),
                                  hint=str(prompt_raw.get("hint", "")))
        measure = dict(entry.get("measure") or {})
        floor = float((entry.get("quality") or {}).get("floor_rms") or 0.0)
        clips.append(EnrollmentClip(prompt=prompt, audio=audio, measure=measure,
                                    quality=assess_clip(audio, measure, floor)))
    if not clips:
        raise CustomWakeWordError(f"no recordings found in {directory}")
    return phrase, clips

# ------------------------------------------------------------- negative data

@dataclass(frozen=True)
class NegativePool:
    """The audio a custom detector must learn *not* to fire on.

    Positives alone cannot produce a usable detector: a model shown only "Hey
    Friday" learns "audio exists".  What keeps it honest is other people's
    names, ordinary speech and room noise -- and, specifically, the user's own
    recordings of *other* wake words, which are the hardest negatives there are.
    """

    impostor_clips: Tuple[Dict[str, Any], ...] = ()
    negative_clips: Tuple[Dict[str, Any], ...] = ()
    ambient_clips: Tuple[Dict[str, Any], ...] = ()
    speech_files: Tuple[str, ...] = ()
    noise_files: Tuple[str, ...] = ()
    sources: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_clips(self) -> int:
        return len(self.impostor_clips) + len(self.negative_clips)

    def sufficient(self) -> Tuple[bool, str]:
        """Whether there is enough here to train something trustworthy."""
        if self.total_clips < 20:
            return False, (f"only {self.total_clips} negative recordings are "
                           "available; a detector trained on this would fire on "
                           "almost anything")
        if not self.noise_files and not self.speech_files and self.total_clips < 60:
            return False, ("no licensed speech or noise corpus was found and the "
                           "local negatives are too few to stand in for one")
        return True, ""


def discover_negatives(*, corpus_root: Optional[Path] = None,
                       librispeech: Optional[Path] = None,
                       musan_noise: Optional[Path] = None,
                       wake_phrase: str = "") -> NegativePool:
    """Collect negative audio from what is already on this machine.

    The wake-word corpus recorded for the Aariz work is the first source: the
    user's own voice saying other names, plus 54 deliberate distractors and the
    ambient recordings.  LibriSpeech and MUSAN are used when a path to them is
    given or an earlier research run left them where they can be found; both
    are CC-BY 4.0 and their use is recorded in the model's provenance.

    Nothing is downloaded.
    """
    from tools.wakeword_corpus import corpus_root as default_corpus_root

    root = Path(corpus_root) if corpus_root else default_corpus_root()
    manifest = root / "manifest.jsonl"
    impostors: List[Dict[str, Any]] = []
    negatives: List[Dict[str, Any]] = []
    ambient: List[Dict[str, Any]] = []
    target = normalise_wake_phrase(wake_phrase)

    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = row.get("path") or ""
            if not path or not Path(path).is_file():
                continue
            kind = row.get("kind")
            if kind == "positive":
                # Every recorded phrase that is not the one being enrolled is an
                # impostor, including the shipped names.
                if normalise_wake_phrase(row.get("phrase", "")) != target:
                    impostors.append(row)
            elif kind in ("negative", "tts_echo"):
                negatives.append(row)
            elif kind == "ambient":
                ambient.append(row)

    speech = _find_audio(librispeech, ("*.flac", "*.wav"), limit=600,
                         fallbacks=("vwwork/corpus/LibriSpeech", "corpus/LibriSpeech"))
    noise = _find_audio(musan_noise, ("*.wav",), limit=930,
                        fallbacks=("vwwork/corpus/musan/noise", "corpus/musan/noise"))

    sources = {
        "wake_word_corpus": {
            "root": str(root), "licence": "the user's own recordings",
            "impostor_clips": len(impostors), "negative_clips": len(negatives),
            "ambient_clips": len(ambient),
        },
        "librispeech": {"files": len(speech), "licence": "CC-BY 4.0",
                        "root": speech[0].rsplit("LibriSpeech", 1)[0] + "LibriSpeech"
                        if speech else ""},
        "musan_noise": {"files": len(noise), "licence": "CC-BY 4.0",
                        "root": noise[0].rsplit("noise", 1)[0] + "noise" if noise else ""},
    }
    return NegativePool(impostor_clips=tuple(impostors), negative_clips=tuple(negatives),
                        ambient_clips=tuple(ambient), speech_files=tuple(speech),
                        noise_files=tuple(noise), sources=sources)


def _find_audio(explicit: Optional[Path], patterns: Sequence[str], *,
                limit: int, fallbacks: Sequence[str]) -> Tuple[str, ...]:
    """Files under an explicit path, or under a known earlier research location.

    The fallbacks exist because the licensed corpora were already fetched for
    the Phase-15 work and re-downloading a gigabyte to enrol a wake word would
    be absurd.  They are *looked for*, never created, and their absence is not
    an error.
    """
    import os

    roots: List[Path] = []
    if explicit:
        roots.append(Path(explicit))
    else:
        temp = Path(os.environ.get("LOCALAPPDATA", "")) / "Temp" if os.name == "nt" \
            else Path("/tmp")
        for tail in fallbacks:
            roots.append(temp / tail)
    found: List[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                found.append(str(path))
                if len(found) >= limit:
                    return tuple(found)
        if found:
            break
    return tuple(found)


# ------------------------------------------------------------- audio helpers


class _Skipped:
    """Counts audio a loader could not read, so a silent hole becomes a number.

    The MUSAN negatives were skipped on every single file for weeks because the
    failure had nowhere to be reported.  A count that reaches the log is the
    cheapest thing that would have caught it.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self.count = 0
        self.first: str = ""

    def note(self, path: object, error: Exception) -> None:
        self.count += 1
        if not self.first:
            self.first = f"{type(error).__name__}: {error}"
        logger.debug("Unreadable %s entry: %s (%s)", self.source, path, error)

    def report(self, total: int) -> None:
        if not self.count:
            return
        message = ("every %s entry failed to load (%d of %d): %s"
                   if self.count >= total else
                   "%s: %d of %d entries could not be read: %s")
        logger.warning(message, self.source, self.count, total, self.first)


def _read_wav(path: str):
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE and rate > 0:
        step = max(1, rate // SAMPLE_RATE)
        audio = audio[::step]
    return np.asarray(audio, dtype=np.float32)


def _centred(audio, start_sec, end_sec, window: int):
    """The training window, centred on the speech the VAD found."""
    import numpy as np

    if start_sec is None or end_sec is None:
        centre = audio.size // 2
    else:
        centre = int(((float(start_sec) + float(end_sec)) / 2.0) * SAMPLE_RATE)
    begin = max(0, min(centre - window // 2, max(0, audio.size - window)))
    chunk = audio[begin:begin + window]
    return np.pad(chunk, (0, window - chunk.size)) if chunk.size < window else chunk


def _rir(t60: float, rng):
    """A synthetic exponential-decay room impulse response."""
    import numpy as np

    n = max(2, int(t60 * SAMPLE_RATE))
    t = np.arange(n) / SAMPLE_RATE
    h = rng.normal(0, 1, n).astype(np.float32) * np.exp(-6.9078 * t / max(t60, 1e-6))
    h[0] = 1.0
    return h / (float(np.linalg.norm(h)) + 1e-8)


def _apply_condition(x, cond, noise_pool, rng):
    """Reverberate, add noise and change level, as Phase 15 did."""
    import numpy as np

    y = np.asarray(x, dtype=np.float32)
    if cond["t60"]:
        y = np.convolve(y, _rir(float(cond["t60"]), rng))[: x.size]
        peak = float(np.max(np.abs(y))) + 1e-8
        y = y / peak * (float(np.max(np.abs(x))) + 1e-8)
    if cond["snr"] is not None and noise_pool:
        noise = noise_pool[int(rng.integers(0, len(noise_pool)))]
        if noise.size == 0:
            noise = np.zeros(y.size, dtype=np.float32)
        if noise.size < y.size:
            noise = np.tile(noise, int(np.ceil(y.size / max(1, noise.size))))
        off = int(rng.integers(0, max(1, noise.size - y.size)))
        noise = noise[off:off + y.size]
        py = float(np.mean(y ** 2))
        pn = float(np.mean(noise ** 2)) + 1e-12
        y = y + math.sqrt(py / (pn * 10 ** (float(cond["snr"]) / 10.0))) * noise
    gain = float(cond["gain_db"]) + float(rng.uniform(-3, 3))
    return np.clip(y * 10 ** (gain / 20.0), -1.0, 1.0)


#: Noise clips are read once into memory and reused.  Enrollment mixes noise
#: into a few thousand training examples, and re-reading a MUSAN file for each
#: one turns a three-minute step into a twenty-minute one for no benefit: the
#: augmentation needs *varied* noise, not *unique* noise.
NOISE_POOL_FILES = 60

#: How many noise windows to draw from that pool.  More than the number of
#: files, because each is long and several non-overlapping windows from one
#: recording are still different noise.
NOISE_NEGATIVE_WINDOWS = 400


def _load_noise_pool(paths: Sequence[str], rng, limit: int = NOISE_POOL_FILES):
    """A handful of noise recordings, decoded once."""
    import numpy as np

    if not paths:
        return []
    chosen = rng.choice(len(paths), size=min(limit, len(paths)), replace=False)
    pool = []
    for i in chosen:
        try:
            audio = _read_wav(paths[int(i)])
        except Exception:  # noqa: BLE001 - a bad file is simply not used
            continue
        if audio.size:
            pool.append(np.asarray(audio, dtype=np.float32))
    return pool


#: Recordings of other wake words are the most valuable negatives there are, but
#: there are hundreds of them and each is augmented seven ways.  Sampling evenly
#: across the sorted list keeps every name, distance and condition represented
#: while holding enrollment to a few minutes.
MAX_IMPOSTOR_CLIPS = 180


def _spread(rows: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if len(rows) <= limit:
        return list(rows)
    step = len(rows) / float(limit)
    return [rows[int(i * step)] for i in range(limit)]


def _features(window):
    """The tail the verifier scores, through the detector's own front end."""
    from speech.wakeword_cascade import TAIL_SECONDS, _log_mel

    tail = int(TAIL_SECONDS * SAMPLE_RATE)
    return _log_mel(window[-tail:])



def split_ambient(negatives: "NegativePool"):
    """``(dev, held_out)`` ambient recordings, in file order.

    Only the dev half may be trained on.  The held-out half is what the false
    activation figure is measured on, and a model that had seen it would be
    reporting its own training performance.
    """
    rows = sorted(negatives.ambient_clips, key=lambda r: str(r.get("path", "")))
    return rows[:AMBIENT_DEV_FILES], rows[AMBIENT_DEV_FILES:]


def _silence_windows(rows, rng, *, limit: int = SILENCE_NEGATIVE_WINDOWS):
    """Window-sized slices of empty room, sampled across the dev recordings."""
    import numpy as np

    from speech.wakeword_cascade import WINDOW_SAMPLES

    catalogue = []
    cache = {}
    skipped = _Skipped("ambient (silence negatives)")
    for row in rows:
        path = row.get("path")
        if not path:
            continue
        try:
            audio = _read_wav(path)
        except Exception as exc:  # noqa: BLE001
            skipped.note(path, exc)
            continue
        if audio.size < WINDOW_SAMPLES:
            continue
        cache[path] = audio
        hop = WINDOW_SAMPLES // 2
        for start in range(0, audio.size - WINDOW_SAMPLES, hop):
            catalogue.append((path, start))
    if not catalogue:
        return []
    skipped.report(len(rows))
    picks = rng.choice(len(catalogue), size=min(limit, len(catalogue)),
                       replace=False)
    return [cache[catalogue[int(i)][0]][catalogue[int(i)][1]:
                                        catalogue[int(i)][1] + WINDOW_SAMPLES]
            for i in picks]

# ---------------------------------------------------------------- the result

@dataclass
class GateCheck:
    """One measured property of the candidate detector."""

    name: str
    passed: bool
    detail: str
    value: Optional[float] = None
    numerator: Optional[int] = None
    denominator: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "passed": self.passed,
                               "detail": self.detail}
        if self.value is not None:
            out["value"] = round(float(self.value), 6)
        if self.denominator:
            out["count"] = f"{self.numerator}/{self.denominator}"
        return out


@dataclass
class EnrollmentResult:
    """Everything the workflow produced, whether or not it was installed."""

    phrase: str
    display_name: str
    passed: bool
    installed: bool
    checks: List[GateCheck] = field(default_factory=list)
    stage2_threshold: float = 0.0
    stage1_threshold: float = STAGE1_THRESHOLD
    summary: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    model: Optional[CustomWakeWord] = None

    def failures(self) -> List[GateCheck]:
        return [c for c in self.checks if not c.passed]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phrase": self.phrase, "display_name": self.display_name,
            "passed": self.passed, "installed": self.installed,
            "stage1_threshold": self.stage1_threshold,
            "stage2_threshold": self.stage2_threshold,
            "checks": [c.as_dict() for c in self.checks],
            "summary": self.summary, "message": self.message,
        }


#: Reported to the caller as the workflow moves, so a console or a dialog can
#: show progress without either of them knowing how enrollment works.
Progress = Callable[[str, str], None]


def _noop(stage: str, message: str) -> None:
    logger.info("Enrollment: %s", message, extra={"context": {"stage": stage}})


# ------------------------------------------------------------------ training

def build_detector(clips: Sequence[EnrollmentClip], *, phrase: str,
                   negatives: NegativePool, work_dir: Path,
                   seed: int = DEFAULT_SEED, epochs: int = EPOCHS,
                   progress: Optional[Progress] = None) -> Dict[str, Any]:
    """Train stage 2 and enrol stage 1 from the recordings.

    Returns the artefact paths and everything measured along the way.  Nothing
    is installed here: :func:`enrol` decides that, after the gate.
    """
    import numpy as np
    import torch
    import torch.nn as nn

    from speech.wakeword_cascade import WINDOW_SAMPLES, _FarFieldVerifier

    say = progress or _noop
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    usable = [c for c in clips if c.usable]
    if len(usable) < MIN_RECORDINGS:
        raise CustomWakeWordError(
            f"only {len(usable)} of {len(clips)} recordings are usable; "
            f"at least {MIN_RECORDINGS} are needed")

    # Three ways, not two.  The model reaches a training loss of about 1e-4 on
    # a set this small -- it memorises its own examples and scores them 1.0 --
    # so a threshold picked from them saturates at the top of the grid and the
    # detector then hears nothing it has not already heard.  The calibration
    # clips exist to show the generalisation gap; the held-out clips are then
    # never consulted until the gate, so the gate remains a measurement.
    held_count = max(MIN_HELD_OUT, int(round(len(usable) * HELD_OUT_FRACTION)))
    calib_count = max(MIN_CALIBRATION, int(round(len(usable) * CALIBRATION_FRACTION)))
    spare = len(usable) - MIN_TRAIN_CLIPS
    if held_count + calib_count > spare:
        held_count = max(MIN_HELD_OUT, spare // 2)
        calib_count = max(MIN_CALIBRATION, spare - held_count)
    _train_idx, calib_idx, held_idx = _stratified_split(
        usable, rng, held_count=held_count, calib_count=calib_count)
    train_clips = [c for i, c in enumerate(usable)
                   if i not in held_idx and i not in calib_idx]
    calib_clips = [c for i, c in enumerate(usable) if i in calib_idx]
    held_clips = [c for i, c in enumerate(usable) if i in held_idx]
    if not train_clips or not calib_clips or not held_clips:
        raise CustomWakeWordError(
            f"{len(usable)} usable recordings cannot be split into training, "
            "calibration and test sets; please record more")

    say("split", f"{len(train_clips)} recordings to learn from, "
                 f"{len(calib_clips)} to set the threshold with, "
                 f"{len(held_clips)} kept back to test with")

    def clip_window(clip: EnrollmentClip):
        return _centred(clip.audio, clip.measure.get("speech_start_sec"),
                        clip.measure.get("speech_end_sec"), WINDOW_SAMPLES)

    train_windows = [clip_window(c) for c in train_clips]
    calib_windows = [clip_window(c) for c in calib_clips]
    held_windows = [clip_window(c) for c in held_clips]

    # ---- stage 1: enrol, do not train ---------------------------------
    say("stage1", "enrolling the phrase with the always-on detector")
    reference = _build_reference(train_windows)
    reference_path = work_dir / "reference.npz"
    np.savez_compressed(reference_path, embeddings=reference)
    validate_reference(reference_path)

    # ---- stage 2: build the training set -------------------------------
    say("dataset", "building the training set from your recordings")
    noise_pool = _load_noise_pool(negatives.noise_files, rng)
    conditions = list(CONDITIONS.values())

    features: List[Any] = []
    labels: List[float] = []
    for window in train_windows:
        for cond in conditions:
            for _ in range(REPEATS_PER_CONDITION):
                features.append(_features(_apply_condition(window, cond, noise_pool, rng)))
                labels.append(1.0)

    impostor_rows = (_spread(sorted(negatives.impostor_clips,
                                    key=lambda r: str(r.get("path", ""))),
                             MAX_IMPOSTOR_CLIPS)
                     + list(negatives.negative_clips))
    for row in impostor_rows:
        try:
            audio = _read_wav(row["path"])
        except Exception:  # noqa: BLE001 - a missing clip is not fatal
            continue
        if audio.size < WINDOW_SAMPLES:
            continue
        base = _centred(audio, row.get("speech_start_sec"),
                        row.get("speech_end_sec"), WINDOW_SAMPLES)
        for cond in conditions:
            features.append(_features(_apply_condition(base, cond, noise_pool, rng)))
            labels.append(0.0)

    speech_skipped = _Skipped("LibriSpeech negatives")
    for path in negatives.speech_files:
        try:
            audio = _read_wav(path)
        except Exception as exc:  # noqa: BLE001
            speech_skipped.note(path, exc)
            continue
        if audio.size < WINDOW_SAMPLES:
            continue
        start = int(rng.integers(0, audio.size - WINDOW_SAMPLES))
        features.append(_features(audio[start:start + WINDOW_SAMPLES]))
        labels.append(0.0)

    # Empty room, explicitly labelled as "not the wake word".  See
    # SILENCE_NEGATIVE_WINDOWS: without these the detector fires on silence.
    dev_ambient, _held_ambient = split_ambient(negatives)
    silence = _silence_windows(dev_ambient, rng)
    for window in silence:
        features.append(_features(window))
        labels.append(0.0)
    if silence:
        say("dataset", f"{len(silence)} windows of empty room added as negatives")

    speech_skipped.report(len(negatives.speech_files))

    # The pool is already decoded by _load_noise_pool, so these are arrays and
    # not paths.  Reading them as files raised every time and the bare except
    # swallowed it, which meant no noise negative ever reached training -- a
    # silent hole this loop existed to fill.  Several windows are taken per
    # clip, since the pool is deliberately a small number of long recordings.
    usable_noise = [a for a in noise_pool if getattr(a, "size", 0) >= WINDOW_SAMPLES]
    noise_added = 0
    for _ in range(NOISE_NEGATIVE_WINDOWS if usable_noise else 0):
        audio = usable_noise[int(rng.integers(0, len(usable_noise)))]
        start = int(rng.integers(0, audio.size - WINDOW_SAMPLES + 1))
        features.append(_features(audio[start:start + WINDOW_SAMPLES]))
        labels.append(0.0)
        noise_added += 1
    if noise_pool and not usable_noise:
        logger.warning("The noise pool decoded %d clips but none were long "
                       "enough to use", len(noise_pool))

    # A source that contributed nothing is the MUSAN bug's signature.  It is
    # not fatal -- a machine may genuinely lack a corpus -- but it is never
    # something to discover months later from a false-activation rate.
    for source, expected, got in (
            ("noise", bool(noise_pool), noise_added),
            ("silence", bool(dev_ambient), len(silence)),
    ):
        if expected and not got:
            logger.warning("No %s negatives reached training; the detector will "
                           "not have been taught to reject them", source)
            say("dataset", f"WARNING: no {source} negatives were usable")

    frames = min(f.shape[1] for f in features)
    x = np.stack([f[:, :frames] for f in features]).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)
    positives, negative_count = int(y.sum()), int((1 - y).sum())
    if negative_count < positives:
        raise CustomWakeWordError(
            "there is less negative audio than positive; the detector would "
            "learn to say yes to everything")
    say("dataset", f"{x.shape[0]:,} training examples "
                   f"({positives:,} positive, {negative_count:,} negative)")

    # ---- stage 2: train -------------------------------------------------
    model = _FarFieldVerifier()
    xt = torch.from_numpy(x).unsqueeze(1)
    yt = torch.from_numpy(y)
    weight = torch.tensor([negative_count / max(1.0, float(positives))],
                          dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    generator = torch.Generator().manual_seed(seed)

    started = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(xt.shape[0], generator=generator)
        total = 0.0
        for i in range(0, perm.numel(), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimiser.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimiser.step()
            total += float(loss) * idx.numel()
        schedule.step()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            say("train", f"epoch {epoch + 1}/{epochs}, loss {total / perm.numel():.4f}")
    train_seconds = time.perf_counter() - started
    model.eval()

    verifier_path = work_dir / "verifier.pt"
    torch.save({"state_dict": model.state_dict(), "seed": seed, "n_mels": 40,
                "frames": int(frames), "conditions": CONDITIONS,
                "params": sum(p.numel() for p in model.parameters())},
               verifier_path)

    def score(window) -> float:
        feature = _features(window)
        feature = (feature[:, :frames] if feature.shape[1] >= frames
                   else np.pad(feature, ((0, 0), (0, frames - feature.shape[1]))))
        tensor = torch.from_numpy(feature).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(model(tensor))[0])

    return {
        "verifier_path": verifier_path,
        "reference_path": reference_path,
        "score": score,
        "frames": int(frames),
        "seed": seed,
        "train_seconds": round(train_seconds, 1),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "checkpoint_bytes": verifier_path.stat().st_size,
        "train_clips": train_clips,
        "calib_clips": calib_clips,
        "held_clips": held_clips,
        "train_windows": train_windows,
        "calib_windows": calib_windows,
        "held_windows": held_windows,
        "training_examples": int(x.shape[0]),
        "training_positives": positives,
        "training_negatives": negative_count,
        "silence_negatives": len(silence),
        "noise_negatives": noise_added,
        "reference_rows": int(reference.shape[0]),
    }


def _build_reference(windows: Sequence[Any]):
    """Stage-1 enrollment embeddings for the recorded windows.

    EfficientWord-Net is few-shot: it is not retrained for a new phrase, it is
    *shown* the phrase.  This is that step, and it is why stage 1 needs no
    training data beyond the user's own recordings.
    """
    import numpy as np

    from eff_word_net.audio_processing import Resnet50_Arc_loss

    backend = Resnet50_Arc_loss()
    vectors = []
    for window in windows:
        vector = backend.audioToVector(np.asarray(window, dtype=np.float32))
        vectors.append(np.asarray(vector, dtype=np.float32).reshape(-1))
    if not vectors:
        raise CustomWakeWordError("no usable recordings to enrol stage 1 with")
    return np.stack(vectors).astype(np.float32)


# ---------------------------------------------------------------- validation

def calibrate_threshold(built: Dict[str, Any], negatives: NegativePool, *,
                        rng_seed: int = DEFAULT_SEED) -> Tuple[float, Dict[str, Any]]:
    """Pick the stage-2 threshold on the calibration recordings.

    Not on the training ones.  The model drives its training loss to about 1e-4
    and scores its own examples 1.0, so calibrating there picks the top of the
    grid and produces a detector that recognises nothing new -- measured, on
    real recordings, before this was changed.  The calibration clips were never
    trained on, so their scores show the gap that actually matters.

    The held-out recordings are still not consulted: a threshold chosen to make
    the test look good is not a measurement.  Candidates come from the frozen
    grid and the rule is fixed in advance -- the highest threshold that still
    accepts at least 90% of the calibration windows.
    """
    import numpy as np

    score = built["score"]
    rng = np.random.default_rng(rng_seed)
    conditions = list(CONDITIONS.values())
    noise_pool = _load_noise_pool(negatives.noise_files, rng)

    windows = built.get("calib_windows") or built.get("train_windows") or []
    positive_scores = []
    for window in windows:
        positive_scores.append(score(window))
        for cond in conditions:
            positive_scores.append(score(_apply_condition(window, cond, noise_pool, rng)))
    if not positive_scores:
        return float(THRESHOLD_GRID[0]), {"method": "no calibration windows"}

    chosen = THRESHOLD_GRID[0]
    for threshold in THRESHOLD_GRID:
        keep = sum(1 for s in positive_scores if s >= threshold) / len(positive_scores)
        if keep >= 0.90:
            chosen = threshold
        else:
            break

    return float(chosen), {
        "method": ("highest threshold on the frozen grid that keeps >=90% of the "
                   "augmented CALIBRATION windows -- recordings the model was "
                   "not trained on; the held-out set was not used"),
        "grid": list(THRESHOLD_GRID),
        "calibration_recordings": len(windows),
        "calibration_windows": len(positive_scores),
        "median_calibration_score": round(float(np.median(positive_scores)), 4),
    }


def quality_gate(built: Dict[str, Any], negatives: NegativePool, *,
                 threshold: float, phrase: str,
                 progress: Optional[Progress] = None) -> List[GateCheck]:
    """Measure the candidate detector on data it has never seen.

    Conservative by design.  Every check reports its own numerator and
    denominator because the numbers are small, and a small denominator is
    something the user should see rather than something to hide behind a
    percentage.
    """
    import numpy as np

    from speech.wakeword_cascade import WINDOW_SAMPLES

    say = progress or _noop
    score = built["score"]
    checks: List[GateCheck] = []

    # 1. Does it recognise the held-out recordings of its own phrase?
    say("gate", "testing against the recordings kept back")
    held = built["held_windows"]
    hits = sum(1 for w in held if score(w) >= threshold)
    recall = hits / len(held) if held else 0.0
    # 90%, not 70%: a wake word you have to repeat is not one you would keep
    # using. The bar is deliberately hard to clear on a small enrollment, and a
    # detector that cannot clear it is not one to install.
    checks.append(GateCheck(
        name="held_out_recall", passed=bool(held) and recall >= MIN_HELD_OUT_RECALL,
        numerator=hits, denominator=len(held), value=recall,
        detail=(f"woke on {hits} of {len(held)} recordings it was not trained "
                f"on ({recall:.0%}; {MIN_HELD_OUT_RECALL:.0%} required)"
                if held else "no recordings were kept back")))

    # 2. Does it stay quiet for other names?  The one that matters most: a
    #    detector that answers to someone else's wake word is worse than none.
    say("gate", "testing against other names and distractors")
    impostor_fires, impostor_total = _count_fires(
        negatives.impostor_clips, score, threshold, WINDOW_SAMPLES)
    checks.append(GateCheck(
        name="other_names", passed=impostor_fires == 0,
        numerator=impostor_fires, denominator=impostor_total,
        value=(impostor_fires / impostor_total) if impostor_total else 0.0,
        detail=(f"woke on {impostor_fires} of {impostor_total} recordings of "
                "other wake words" if impostor_total
                else "no other-name recordings were available to test with")))

    # 3. Ordinary speech and commands.
    negative_fires, negative_total = _count_fires(
        negatives.negative_clips, score, threshold, WINDOW_SAMPLES)
    checks.append(GateCheck(
        name="ordinary_speech", passed=negative_fires == 0,
        numerator=negative_fires, denominator=negative_total,
        value=(negative_fires / negative_total) if negative_total else 0.0,
        detail=(f"woke on {negative_fires} of {negative_total} distractor "
                "recordings" if negative_total
                else "no distractor recordings were available to test with")))

    # 4. Silence and room noise, densely.  Stage 1 would gate most of these out
    #    in the real loop, so this is an upper bound -- which is what a gate
    #    wants.  It is the only check with a denominator large enough to mean
    #    much, and it is still not a false-activation rate.
    # Only the ambient the model never saw.  Scoring the dev half would be
    # reporting training performance as a false-activation rate.
    say("gate", "testing against recorded room noise")
    _dev_ambient, held_ambient = split_ambient(negatives)
    fires, hours = _ambient_fires(held_ambient, score, threshold,
                                  WINDOW_SAMPLES)
    per_hour = (fires / hours) if hours > 0 else 0.0
    checks.append(GateCheck(
        name="room_noise",
        passed=(hours <= 0) or per_hour <= MAX_STANDALONE_FA_PER_HOUR,
        numerator=fires, denominator=int(round(hours * 3600)), value=per_hour,
        detail=(f"{fires} acceptances in {hours:.2f} h of recorded room audio "
                f"({per_hour:.1f}/h, verifier alone; the shipped detector "
                f"measures {SHIPPED_STANDALONE_FA_PER_HOUR}/h the same way)"
                if hours > 0
                else "no ambient recordings were available to test with")))

    # 5. Enough separation to be worth installing at all.
    if held:
        margin = float(np.median([score(w) for w in held])) - threshold
        checks.append(GateCheck(
            name="margin", passed=margin > 0.05, value=margin,
            detail=f"median held-out score sits {margin:+.3f} from the threshold"))

    return checks


def _count_fires(rows: Sequence[Dict[str, Any]], score, threshold: float,
                 window: int) -> Tuple[int, int]:
    fires = total = 0
    skipped = _Skipped("gate negatives")
    for row in rows:
        try:
            audio = _read_wav(row["path"])
        except Exception as exc:  # noqa: BLE001
            skipped.note(row.get("path"), exc)
            continue
        if audio.size < window:
            continue
        total += 1
        centred = _centred(audio, row.get("speech_start_sec"),
                           row.get("speech_end_sec"), window)
        if score(centred) >= threshold:
            fires += 1
    skipped.report(len(rows))
    return fires, total


def _ambient_fires(rows: Sequence[Dict[str, Any]], score, threshold: float,
                   window: int, hop_seconds: float = 0.5) -> Tuple[int, float]:
    """Slide over the ambient recordings and count acceptances."""
    hop = int(hop_seconds * SAMPLE_RATE)
    fires = 0
    samples = 0
    skipped = _Skipped("ambient")
    for row in rows:
        try:
            audio = _read_wav(row["path"])
        except Exception as exc:  # noqa: BLE001
            skipped.note(row.get("path"), exc)
            continue
        samples += audio.size
        for start in range(0, max(1, audio.size - window), hop):
            if score(audio[start:start + window]) >= threshold:
                fires += 1
    skipped.report(len(rows))
    return fires, samples / SAMPLE_RATE / 3600.0


# --------------------------------------------------------------- the workflow

def enrol(*, display_name: str, clips: Sequence[EnrollmentClip],
          negatives: Optional[NegativePool] = None,
          store: Optional[CustomWakeWordStore] = None,
          phrase: str = "", seed: int = DEFAULT_SEED, epochs: int = EPOCHS,
          progress: Optional[Progress] = None,
          install: bool = True) -> EnrollmentResult:
    """Train, validate and -- only if it passes -- install a custom detector.

    ``install=False`` runs everything and reports, without touching the store;
    used by the tests and by anyone who wants to see the numbers first.
    """
    import tempfile

    say = progress or _noop
    phrase = normalise_wake_phrase(phrase) or wake_phrase_for_name(display_name)
    store = store or STORE
    result = EnrollmentResult(phrase=phrase, display_name=display_name,
                              passed=False, installed=False)

    if not phrase:
        result.message = "That name cannot be turned into a wake phrase."
        return result

    if negatives is None:
        say("negatives", "looking for audio to teach it what not to react to")
        negatives = discover_negatives(wake_phrase=phrase)
    enough, why = negatives.sufficient()
    if not enough:
        result.message = (f"I can't train a wake word safely: {why}. "
                          "Nothing has been changed.")
        logger.warning("Enrollment refused for lack of negatives", extra={
            "context": {"phrase": phrase, "reason": why}})
        return result

    with tempfile.TemporaryDirectory(prefix="aarma-enrol-") as tmp:
        work = Path(tmp)
        try:
            built = build_detector(clips, phrase=phrase, negatives=negatives,
                                   work_dir=work, seed=seed, epochs=epochs,
                                   progress=progress)
        except CustomWakeWordError as exc:
            result.message = str(exc)
            return result

        say("threshold", "calibrating the confidence threshold")
        threshold, calibration = calibrate_threshold(built, negatives, rng_seed=seed)
        result.stage2_threshold = threshold

        checks = quality_gate(built, negatives, threshold=threshold,
                              phrase=phrase, progress=progress)
        result.checks = checks
        result.passed = all(c.passed for c in checks)
        result.summary = {
            "recordings_used": len(built["train_clips"]),
            "recordings_for_threshold": len(built.get("calib_clips") or []),
            "recordings_held_back": len(built["held_clips"]),
            "reference_rows": built["reference_rows"],
            "training_examples": built["training_examples"],
            "training_positives": built["training_positives"],
            "training_negatives": built["training_negatives"],
            "silence_negatives": built.get("silence_negatives", 0),
            "parameters": built["parameters"],
            "checkpoint_bytes": built["checkpoint_bytes"],
            "train_seconds": built["train_seconds"],
            "frames": built["frames"],
            "seed": built["seed"],
            "threshold_calibration": calibration,
            "distances_recorded": sorted({c.prompt.distance for c in clips if c.usable}),
        }
        result.provenance = {
            "positives": "recordings made by the user during enrollment",
            "architecture": "AARMA's own; trained from scratch, no pretrained weights",
            "stage1": ("EfficientWord-Net 1.0.5 resnet_50_arc, enrolled few-shot. "
                       "Apache-2.0 code and weights; its distilled training corpus "
                       "is UNVERIFIED and remains a redistribution concern."),
            "negative_sources": negatives.sources,
        }

        if not result.passed:
            failed = ", ".join(c.name for c in result.failures())
            result.message = (f"The wake word for {display_name} did not pass "
                              f"validation ({failed}). It has NOT been installed, "
                              "and whatever was working before is untouched.")
            logger.warning("Custom wake word failed validation", extra={
                "context": {"phrase": phrase, "failed": failed}})
            return result

        if not install:
            result.message = (f"\"{phrase}\" passed validation. It was not "
                              "installed because this was a dry run.")
            return result

        say("install", "saving the new wake word")
        try:
            model = store.install(
                phrase=phrase, display_name=display_name,
                verifier=built["verifier_path"], reference=built["reference_path"],
                stage1_threshold=STAGE1_THRESHOLD, stage2_threshold=threshold,
                enrollment_count=(len(built["train_clips"])
                                  + len(built.get("calib_clips") or [])
                                  + len(built["held_clips"])),
                validation={"checks": [c.as_dict() for c in checks],
                            "summary": result.summary},
                provenance=result.provenance,
                architecture={"parameters": built["parameters"],
                              "frames": built["frames"], "n_mels": 40,
                              "family": "log-mel 40 -> conv 16/32/64/64 -> "
                                        "mean+max pooling -> FC 128 -> 1 logit"})
        except CustomWakeWordError as exc:
            result.message = f"The wake word passed validation but could not be saved: {exc}"
            return result

        result.installed = True
        result.model = model
        result.message = (f"\"{phrase}\" is enrolled and active "
                          f"(version {model.model_version}).")
        return result
