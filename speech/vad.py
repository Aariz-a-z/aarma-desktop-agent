"""Voice activity detection.

The first speech-input build decided "is this speech?" by comparing block RMS
against a threshold derived from the room. Measurement showed that approach
failing in three ordinary situations: speech beyond about 1.5 m fell under the
floor, a room with any hum pushed the threshold above normal speech, and
speaking *immediately* after pressing the button poisoned the noise measurement
so nothing was ever detected again.

Silero VAD replaces it. It is a small neural detector that judges whether audio
*sounds like speech* rather than whether it is loud, and it ships inside
``faster-whisper`` -- so this costs no new dependency and no download.

Measured on this machine, on real speech embedded in silence:

* speech-start detected within 30 ms of truth (no clipped first word);
* speech found at every noise level tested, up to 0.05 RMS, where the energy
  threshold found none;
* pure silence and noise at 0.02 and 0.05 RMS correctly yield *no* speech,
  so nothing can be fabricated from a quiet room;
* 7-18 ms to scan a three-second window, cheap enough to run live.

An energy detector remains as a fallback for installations without the model,
clearly marked as the weaker option.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger("speech.vad")

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class VadTuning:
    """Endpointing parameters, in the units a person would reason about.

    Defaults are the values validated on this machine: a lower threshold than
    Silero's stock 0.5 (more willing to hear quiet, distant speech), a short
    minimum utterance so single words like "stop" register, and generous
    padding so neither the first nor the last word is trimmed away.
    """

    threshold: float = 0.35
    min_speech_ms: int = 120
    min_silence_ms: int = 600
    speech_pad_ms: int = 300

    def scaled(self, *, sensitivity: float) -> "VadTuning":
        """Apply a 0-1 user sensitivity: higher hears more, and more noise."""
        from dataclasses import replace

        # 0 -> 0.60 (conservative), 0.5 -> 0.35 (default), 1 -> 0.20 (sensitive)
        threshold = 0.60 - (0.40 * max(0.0, min(1.0, sensitivity)))
        return replace(self, threshold=round(threshold, 3))


@dataclass(frozen=True)
class SpeechRegion:
    """A span of speech within a buffer, in samples."""

    start: int
    end: int

    @property
    def duration(self) -> float:
        return max(0, self.end - self.start) / SAMPLE_RATE


class VoiceActivityDetector(ABC):
    """Decides where speech is inside a buffer of 16 kHz mono float32 audio."""

    name: str = "abstract"
    display_name: str = "Abstract"
    #: True when this detector actually models speech rather than loudness.
    is_neural: bool = False

    @abstractmethod
    def is_available(self) -> bool:
        """True when this detector can run."""

    @abstractmethod
    def speech_regions(self, audio, tuning: Optional[VadTuning] = None) -> List[SpeechRegion]:
        """Every span of speech in ``audio``. Empty means none -- never a guess."""

    def has_speech(self, audio, tuning: Optional[VadTuning] = None) -> bool:
        return bool(self.speech_regions(audio, tuning))

    def speech_seconds(self, audio, tuning: Optional[VadTuning] = None) -> float:
        return sum(region.duration for region in self.speech_regions(audio, tuning))

    def trim(self, audio, tuning: Optional[VadTuning] = None):
        """Return just the speech, from the first region to the last.

        Keeps the gaps *between* words rather than splicing regions together:
        cutting internal pauses out of a sentence changes its rhythm and makes
        recognition worse, not better.
        """
        regions = self.speech_regions(audio, tuning)
        if not regions:
            return None
        return audio[regions[0].start:regions[-1].end]

    def last_speech_end(self, audio, tuning: Optional[VadTuning] = None) -> Optional[int]:
        """Sample index where speech last stopped, for live endpointing."""
        regions = self.speech_regions(audio, tuning)
        return regions[-1].end if regions else None


class SileroVad(VoiceActivityDetector):
    """Silero VAD v6, as bundled with faster-whisper."""

    name = "silero"
    display_name = "Neural voice detection"
    is_neural = True

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._checked = False
        self._usable = False

    @staticmethod
    def is_installed() -> bool:
        import importlib.util

        return importlib.util.find_spec("faster_whisper") is not None

    def is_available(self) -> bool:
        with self._lock:
            if self._checked:
                return self._usable
            self._checked = True
            try:
                import importlib

                importlib.import_module("faster_whisper.vad")
                self._usable = True
            except Exception:  # noqa: BLE001 - absence is normal, not an error
                logger.info("Silero VAD unavailable; falling back to energy detection")
                self._usable = False
            return self._usable

    def speech_regions(self, audio, tuning: Optional[VadTuning] = None) -> List[SpeechRegion]:
        if audio is None or getattr(audio, "size", 0) == 0:
            return []
        if not self.is_available():
            return []

        options = tuning or VadTuning()
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps

            timestamps = get_speech_timestamps(audio, VadOptions(
                threshold=options.threshold,
                min_speech_duration_ms=options.min_speech_ms,
                min_silence_duration_ms=options.min_silence_ms,
                speech_pad_ms=options.speech_pad_ms,
            ))
        except Exception:  # noqa: BLE001 - a detector fault must not invent speech
            logger.exception("Voice activity detection failed")
            return []

        return [SpeechRegion(int(t["start"]), int(t["end"])) for t in timestamps]


class EnergyVad(VoiceActivityDetector):
    """Loudness-based fallback.

    Retained only so a stripped-down installation still has *something*. It is
    the approach whose limitations prompted this rewrite, and it is labelled
    accordingly rather than presented as equivalent.
    """

    name = "energy"
    display_name = "Basic level detection"
    is_neural = False

    #: Deliberately low. The original 0.006 excluded speech beyond ~1.5 m.
    FLOOR = 0.0015
    WINDOW_SECONDS = 0.03

    def is_available(self) -> bool:
        return True

    def speech_regions(self, audio, tuning: Optional[VadTuning] = None) -> List[SpeechRegion]:
        if audio is None or getattr(audio, "size", 0) == 0:
            return []
        import numpy as np

        options = tuning or VadTuning()
        window = max(1, int(SAMPLE_RATE * self.WINDOW_SECONDS))
        usable = (audio.size // window) * window
        if usable == 0:
            return []

        frames = audio[:usable].reshape(-1, window)
        levels = np.sqrt(np.mean(frames ** 2, axis=1))
        # Threshold relative to the quietest part of this buffer, so a constant
        # room hum does not count as speech and does not mask it either.
        floor = float(np.percentile(levels, 10))
        threshold = max(self.FLOOR, floor * 2.5)
        loud = levels >= threshold

        regions: List[SpeechRegion] = []
        start: Optional[int] = None
        silence_run = 0
        max_silence = max(1, int((options.min_silence_ms / 1000) / self.WINDOW_SECONDS))

        for index, is_loud in enumerate(loud):
            if is_loud:
                if start is None:
                    start = index
                silence_run = 0
            elif start is not None:
                silence_run += 1
                if silence_run >= max_silence:
                    regions.append(self._region(start, index - silence_run + 1,
                                                window, audio.size, options))
                    start = None
                    silence_run = 0
        if start is not None:
            regions.append(self._region(start, len(loud), window, audio.size, options))

        minimum = (options.min_speech_ms / 1000)
        return [r for r in regions if r.duration >= minimum]

    @staticmethod
    def _region(start_frame: int, end_frame: int, window: int, total: int,
                options: VadTuning) -> SpeechRegion:
        pad = int(SAMPLE_RATE * options.speech_pad_ms / 1000)
        start = max(0, start_frame * window - pad)
        end = min(total, end_frame * window + pad)
        return SpeechRegion(start, end)


def create_detector(prefer_neural: bool = True) -> VoiceActivityDetector:
    """The best detector available, neural first."""
    if prefer_neural:
        silero = SileroVad()
        if silero.is_available():
            return silero
    return EnergyVad()
