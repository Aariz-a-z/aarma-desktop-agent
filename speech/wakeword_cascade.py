"""Two-stage wake-word detector: fast candidate generator plus learned verifier.

This is the first real :class:`~speech.wakeword.WakeWordDetector`; the module
docstring in :mod:`speech.wakeword` promised the engine would be replaceable
behind that interface, and this is the replacement.  Nothing outside this file
knows which libraries are involved.

    microphone -> rolling 1.5 s buffer
               -> stage 1  EfficientWord-Net, threshold 0.70
               -> candidate
               -> stage 2  learned far-field verifier, threshold 0.10
               -> wake event

Why two stages
--------------
Stage 1 alone hears "Hey Aariz" 38 times in 40 but also fires on 21 of 120
"Hey Asma"/"Hey Nova"/"Hey Zara" clips -- it cannot tell the names apart.  The
stage-2 verifier was trained specifically to, and on the held-out set it removed
every one of those 21 while rejecting **none** of the 39 true candidates.  It
also fixed the far-field problem that defeated four earlier designs: 2 m went
from 2/8 to 8/8 and 3 m from 1/8 to 7/8.

Both thresholds are **frozen** from that evaluation and must not be re-tuned
against live behaviour; doing so would invalidate the measurement they came
from.  See ``experiments/farfield_verifier/RESULTS.md``.

Availability
------------
Stage 1 needs the optional ``EfficientWord-Net`` package.  When it or the model
assets are missing this detector reports itself unavailable and explains why --
it never raises during start-up, so a machine without the extra dependency
simply runs without wake-word detection, exactly as before.

Known limitation, recorded rather than hidden: the false-activation rate is
**not yet established**.  The benchmark ambient corpus is 2.36 hours, in which
one false activation occurred; that is the resolution floor of the corpus, so
the target of <=0.125/hour can be neither confirmed nor refuted until a longer
recording exists.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from config.settings import normalise_wake_phrase
from speech.wakeword_store import validate_checkpoint
from speech.wakeword import WakeWordCallback, WakeWordDetector, WakeWordEvent
from utils.logger import get_logger

logger = get_logger("speech.wakeword_cascade")

#: The phrase this build is enrolled and trained for.
SUPPORTED_PHRASE = "hey aariz"

#: Frozen operating point. Do not tune these against live audio.
STAGE1_THRESHOLD = 0.70
STAGE2_THRESHOLD = 0.10

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 1.5
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)
TAIL_SECONDS = 1.0
HOP_SECONDS = 0.25

#: Audio kept behind the current instant, and collected after a candidate,
#: before the verifier runs.  The verifier was evaluated on a window centred on
#: the speech burst within [t-2.25 s, t+0.5 s]; giving it a narrower or
#: differently-aligned view measurably changes its answer -- without the 0.5 s
#: of trailing audio, two "Hey Asma" clips and two negatives slip through that
#: the benchmark rejected.  So the detector waits, at a cost of 0.5 s latency.
CONTEXT_SECONDS = 2.75
CONTEXT_SAMPLES = int(CONTEXT_SECONDS * SAMPLE_RATE)
LOOKAHEAD_SECONDS = 0.5
LOOKAHEAD_SAMPLES = int(LOOKAHEAD_SECONDS * SAMPLE_RATE)

#: Silence between accepted wakes.  One utterance produces several overlapping
#: windows above threshold; without this the assistant would wake three times
#: for one "Hey Aariz".
COOLDOWN_SECONDS = 2.0

#: How long :meth:`CascadeWakeWordDetector.start` waits for the worker to load
#: its models and open the microphone.  Generous: the first PyTorch import on a
#: cold machine is measured in seconds, not milliseconds.
LOAD_TIMEOUT_SECONDS = 60.0

#: Where the shipped assets live.
_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = _ROOT / "models" / "hey_aariz_verifier.pt"
DEFAULT_REFERENCE_PATH = _ROOT / "models" / "hey_aariz_reference.npz"


class CascadeWakeWordDetector(WakeWordDetector):
    """Continuous single-phrase detection on the local machine, fully offline.

    One instance is bound to one phrase, because both stages are
    phrase-specific: stage 1 needs an enrollment reference and stage 2 is a
    classifier trained on that phrase.  It therefore refuses any other
    phrase outright rather than answering for a name it was never shown.
    """

    name = "cascade"

    def __init__(
        self,
        *,
        phrase: str = SUPPORTED_PHRASE,
        model_path: Optional[Path] = None,
        reference_path: Optional[Path] = None,
        device_index: Optional[int] = None,
        speaking: Optional[Callable[[], bool]] = None,
        stage1_threshold: float = STAGE1_THRESHOLD,
        stage2_threshold: float = STAGE2_THRESHOLD,
    ) -> None:
        #: One instance detects one phrase.  The default is the phrase this
        #: build ships assets for; a future validated model registers itself
        #: with its own phrase and its own two files, and needs no new class.
        self._phrase = normalise_wake_phrase(phrase) or SUPPORTED_PHRASE
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._reference_path = Path(reference_path) if reference_path else DEFAULT_REFERENCE_PATH
        self._device_index = device_index
        #: Asked before every decision so the assistant cannot wake itself.
        self._speaking = speaking
        self._stage1_threshold = float(stage1_threshold)
        self._stage2_threshold = float(stage2_threshold)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._reason = ""
        self._last_wake = 0.0
        #: Set by the worker once the models have been loaded, or have failed
        #: to.  start() waits on it so it never reports success for a detector
        #: whose engines could not be built.
        self._ready = threading.Event()
        self._load_error = ""

    # ---------------------------------------------------------- availability

    def _missing(self) -> str:
        """Empty string when everything needed is present."""
        # Presence, not import.  Importing torch costs several seconds, and this
        # question is asked by the settings screen, which must stay responsive;
        # the real import happens on the detector's own thread in _load_engines,
        # and start() waits for it to succeed before claiming to be listening.
        for module, complaint in (
            ("numpy", "numpy is not installed"),
            ("torch", "PyTorch is not installed"),
            ("eff_word_net", "the optional EfficientWord-Net package is not installed "
                             "(pip install EfficientWord-Net)"),
        ):
            try:
                if importlib.util.find_spec(module) is None:
                    return complaint
            except (ImportError, ValueError):
                return complaint
        if not self._model_path.is_file():
            return f"verifier model missing: {self._model_path.name}"
        if not self._reference_path.is_file():
            return f"enrollment reference missing: {self._reference_path.name}"
        return ""

    def is_available(self) -> bool:
        self._reason = self._missing()
        return not self._reason

    def unavailable_reason(self) -> str:
        return self._reason or self._missing()

    def supports_phrase(self, phrase: str) -> bool:
        """Only the phrase this build was enrolled and trained for.

        The verifier is a model of one specific phrase, so it cannot honestly
        claim to detect anything the user types into settings.
        """
        return normalise_wake_phrase(phrase) == self._phrase

    # -------------------------------------------------------------- lifecycle

    def start(self, phrase: str, callback: WakeWordCallback) -> bool:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return True
            if not self.supports_phrase(phrase):
                self._reason = (f'this detector hears only "{self._phrase}", '
                                f'not "{phrase}"')
                logger.warning("Wake word not supported", extra={"context": {
                    "requested": phrase, "supported": self._phrase}})
                return False
            if not self.is_available():
                logger.warning("Wake-word engine unavailable", extra={
                    "context": {"reason": self._reason}})
                return False
            self._stop_event.clear()
            self._ready.clear()
            self._load_error = ""
            self._worker = threading.Thread(
                target=self._run, args=(phrase, callback),
                name="wakeword-cascade", daemon=True)
            self._worker.start()

        # Loading the models and opening the device happen on that thread, so
        # this waits for the verdict rather than reporting a success it has not
        # seen.  First start pays the PyTorch import; later ones are immediate.
        if not self._ready.wait(timeout=LOAD_TIMEOUT_SECONDS):
            self._reason = "the wake-word engine did not finish starting"
            logger.error("Wake-word engine start timed out")
            self.stop()
            return False
        if self._load_error:
            self._reason = self._load_error
            self.stop()
            return False
        logger.info("Wake-word listening started", extra={"context": {
            "phrase": phrase, "stage1": self._stage1_threshold,
            "stage2": self._stage2_threshold}})
        return True

    def stop(self) -> None:
        with self._lock:
            worker = self._worker
            self._worker = None
        self._stop_event.set()
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)
        logger.info("Wake-word listening stopped")

    # ------------------------------------------------------------------ loop

    def _run(self, phrase: str, callback: WakeWordCallback) -> None:
        import numpy as np

        from speech.capture import MicrophoneStream, resample_to_16k

        try:
            stage1, stage2 = self._load_engines()
        except Exception as exc:  # noqa: BLE001
            logger.error("Wake-word engines failed to load", extra={
                "context": {"error": str(exc)}})
            self._load_error = f"the wake-word models could not be loaded ({exc})"
            self._ready.set()
            return

        stream = MicrophoneStream(self._device_index)
        buffer = np.zeros(0, dtype=np.float32)
        hop_samples = int(HOP_SECONDS * SAMPLE_RATE)
        since_hop = 0

        try:
            stream.open()
        except Exception as exc:  # noqa: BLE001
            logger.error("Wake-word microphone could not open", extra={
                "context": {"error": str(exc)}})
            self._load_error = f"the microphone could not be opened ({exc})"
            self._ready.set()
            return
        self._load_error = ""
        self._ready.set()

        try:
            while not self._stop_event.is_set():
                try:
                    block, _overflowed = stream.read_block()
                except Exception as exc:  # noqa: BLE001
                    # A disconnected microphone must not crash the assistant.
                    logger.warning("Wake-word capture interrupted", extra={
                        "context": {"error": str(exc)}})
                    break

                if stream.sample_rate != SAMPLE_RATE:
                    block = resample_to_16k(block, stream.sample_rate)
                block = np.asarray(block, dtype=np.float32).reshape(-1)

                # The assistant's own voice must never reach the detector. The
                # buffer is dropped as well, so speech that straddles the end
                # of a reply cannot be stitched into a false wake.
                if self._speaking is not None:
                    try:
                        if self._speaking():
                            buffer = np.zeros(0, dtype=np.float32)
                            since_hop = 0
                            continue
                    except Exception:  # noqa: BLE001
                        pass

                buffer = np.concatenate([buffer, block])[-CONTEXT_SAMPLES:]
                since_hop += block.size
                if buffer.size < WINDOW_SAMPLES or since_hop < hop_samples:
                    continue
                since_hop = 0

                now = time.monotonic()
                if now - self._last_wake < COOLDOWN_SECONDS:
                    continue

                try:
                    score1 = stage1(buffer)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Stage 1 failed: %s", exc)
                    continue
                if score1 is None or score1 < self._stage1_threshold:
                    continue

                # Collect the trailing audio the verifier was evaluated with.
                collected = 0
                while collected < LOOKAHEAD_SAMPLES and not self._stop_event.is_set():
                    try:
                        extra, _ = stream.read_block()
                    except Exception:  # noqa: BLE001
                        break
                    if stream.sample_rate != SAMPLE_RATE:
                        extra = resample_to_16k(extra, stream.sample_rate)
                    extra = np.asarray(extra, dtype=np.float32).reshape(-1)
                    buffer = np.concatenate([buffer, extra])[-CONTEXT_SAMPLES:]
                    collected += extra.size

                try:
                    score2 = stage2(buffer)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Stage 2 failed: %s", exc)
                    continue
                if score2 < self._stage2_threshold:
                    logger.debug("Candidate rejected by verifier", extra={
                        "context": {"stage1": round(float(score1), 4),
                                    "stage2": round(float(score2), 4)}})
                    continue

                self._last_wake = now
                buffer = np.zeros(0, dtype=np.float32)   # no re-trigger on the same speech
                logger.info("Wake word detected", extra={"context": {
                    "stage1": round(float(score1), 4), "stage2": round(float(score2), 4)}})
                try:
                    callback(WakeWordEvent(phrase=phrase, confidence=float(score2)))
                except Exception as exc:  # noqa: BLE001
                    logger.error("Wake-word callback raised", extra={
                        "context": {"error": str(exc)}})
        finally:
            stream.close()

    # --------------------------------------------------------------- engines

    def _load_engines(self):
        """Return ``(stage1, stage2)`` scoring callables over a 1.5 s buffer."""
        import numpy as np
        import torch

        from eff_word_net.audio_processing import Resnet50_Arc_loss

        backend = Resnet50_Arc_loss()
        reference = np.load(self._reference_path)["embeddings"].astype(np.float32)

        def stage1(window: np.ndarray) -> Optional[float]:
            frame = window[-WINDOW_SAMPLES:]
            peak = float(np.max(frame)) if frame.size else 0.0
            if peak > 0:
                # The engine refuses windows that begin mid-speech; this is its
                # own false-positive guard and is reproduced faithfully.
                lead = frame[: SAMPLE_RATE // 10]
                if lead.size and float(np.max(lead / peak)) > 0.2:
                    return None
            vector = backend.audioToVector(frame.astype(np.float32))
            cosine = float(np.matmul(reference, np.asarray(vector).reshape(-1, 1)).max())
            return (cosine + 1.0) / 2.0

        # weights_only=True, and the shape checked first.  A verifier may be a
        # file the user's own enrollment wrote, and a checkpoint is data: it
        # must never be able to run code on load, however it got here.
        validate_checkpoint(self._model_path)
        blob = torch.load(self._model_path, map_location="cpu", weights_only=True)
        model = _FarFieldVerifier()
        model.load_state_dict(blob["state_dict"])
        model.eval()
        frames = blob.get("frames")
        tail = int(TAIL_SECONDS * SAMPLE_RATE)

        def stage2(window: np.ndarray) -> float:
            # `window` is the full context buffer, so the burst is located over
            # the same span the benchmark searched.
            centred = _energy_centred(window, tail)
            features = _log_mel(centred)
            if frames:
                features = (features[:, :frames] if features.shape[1] >= frames
                            else np.pad(features, ((0, 0), (0, frames - features.shape[1]))))
            tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                return float(torch.sigmoid(model(tensor))[0])

        return stage1, stage2


def _energy_centred(window, tail_samples: int):
    """Centre the verifier's view on the speech burst.

    The verifier was trained on windows centred on speech.  Feeding it a
    differently-aligned window costs a large amount of similarity -- during
    development that mistake alone took held-out recall from 25/40 to 0/38 --
    so the burst is located by frame energy, the same job the VAD does
    elsewhere, and the tail is cut around it.
    """
    import numpy as np

    audio = np.asarray(window, dtype=np.float32)
    frame_len, hop = 400, 160
    count = 1 + max(0, (audio.size - frame_len) // hop)
    if count <= 0:
        return audio[-tail_samples:]
    index = np.arange(frame_len)[None, :] + hop * np.arange(count)[:, None]
    energy = np.sqrt((audio[index] ** 2).mean(axis=1))
    if energy.max() <= 0:
        return audio[-tail_samples:]
    above = np.where(energy >= 0.25 * energy.max())[0]
    if not above.size:
        return audio[-tail_samples:]
    centre = int(((above[0] + above[-1]) / 2) * hop + frame_len // 2)
    start = max(0, min(centre - tail_samples // 2, max(0, audio.size - tail_samples)))
    out = audio[start:start + tail_samples]
    if out.size < tail_samples:
        out = np.pad(out, (0, tail_samples - out.size))
    return out


# --------------------------------------------------------------------- model
# The verifier's feature extraction and architecture live here rather than in a
# separate module so the wake-word engine remains a single replaceable file, as
# speech.wakeword intends.  Both must match the training code exactly or the
# checkpoint is meaningless; they are copied from
# experiments/farfield_verifier/scripts/model.py and must not drift.

_N_MELS = 40
_WIN = int(0.025 * SAMPLE_RATE)
_HOP = int(0.010 * SAMPLE_RATE)
_N_FFT = 512
_FMIN, _FMAX = 60.0, 7600.0
_FILTERBANK = None
_HANN = None


def _build_filterbank():
    import numpy as np

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(_FMIN), hz2mel(_FMAX), _N_MELS + 2)
    bins = np.floor((_N_FFT + 1) * mel2hz(mels) / SAMPLE_RATE).astype(int)
    fb = np.zeros((_N_MELS, _N_FFT // 2 + 1), dtype=np.float32)
    for i in range(1, _N_MELS + 1):
        left, centre, right = bins[i - 1], bins[i], bins[i + 1]
        centre = max(centre, left + 1)
        right = min(max(right, centre + 1), fb.shape[1] - 1)
        if left < centre:
            fb[i - 1, left:centre] = (np.arange(left, centre) - left) / max(1, centre - left)
        if centre < right:
            fb[i - 1, centre:right] = (right - np.arange(centre, right)) / max(1, right - centre)
    return fb


def _log_mel(audio):
    """(mels, frames) log-mel with per-utterance normalisation."""
    import numpy as np

    global _FILTERBANK, _HANN
    if _FILTERBANK is None:
        _FILTERBANK = _build_filterbank()
        _HANN = np.hanning(_WIN).astype(np.float32)

    a = np.asarray(audio, dtype=np.float32)
    if a.size < _WIN:
        a = np.pad(a, (0, _WIN - a.size))
    count = 1 + (a.size - _WIN) // _HOP
    index = np.arange(_WIN)[None, :] + _HOP * np.arange(count)[:, None]
    spectrum = np.abs(np.fft.rfft(a[index] * _HANN, n=_N_FFT)) ** 2
    mel = np.log(spectrum @ _FILTERBANK.T + 1e-10).astype(np.float32)
    # Removing the per-recording mean and scale is what lets one enrolment
    # match speech recorded at a different level and distance.
    mel = (mel - mel.mean(0, keepdims=True)) / (mel.std(0, keepdims=True) + 1e-8)
    return mel.T


def _FarFieldVerifier():
    """The trained classifier: ~77 k parameters, 313 KB on disk."""
    import torch.nn as nn

    def block(cin, cout, pool):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.MaxPool2d(pool))

    class _Verifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                block(1, 16, (2, 2)), block(16, 32, (2, 2)),
                block(32, 64, (2, 1)), block(64, 64, (2, 1)))
            self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True),
                                      nn.Dropout(0.3), nn.Linear(128, 1))

        def forward(self, x):
            import torch
            h = self.features(x).mean(dim=2)
            return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)).squeeze(1)

    return _Verifier()
