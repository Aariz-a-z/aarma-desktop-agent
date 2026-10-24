"""Audio playback for synthesised speech.

Providers that render speech to a file (the online neural provider) need
something to play it. Windows' MCI interface in ``winmm.dll`` decodes MP3 and
WAV natively, so this costs no third-party dependency and no bundled codec --
which matters for a project that intends to ship as a single installer.

Playback blocks the calling thread by design: it runs on
:class:`~speech.engine.SpeechEngine`'s worker, never the GUI thread, and
blocking there is what makes "stop" and "don't overlap voices" simple to reason
about. :meth:`AudioPlayback.stop` is safe to call from any thread.
"""

from __future__ import annotations

import ctypes
import itertools
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger("speech.audio")

_POLL_INTERVAL = 0.05
_alias_counter = itertools.count(1)


class AudioPlayback(ABC):
    """Plays an audio file and can be interrupted."""

    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool:
        """True when this backend can actually produce sound on this machine."""

    @abstractmethod
    def play(self, path: Path) -> bool:
        """Play ``path``, returning when finished or stopped.

        Returns ``False`` if playback could not start or was interrupted.
        """

    @abstractmethod
    def stop(self) -> None:
        """Interrupt playback. Safe from any thread; safe when nothing is playing."""

    @property
    def is_playing(self) -> bool:
        return False


class NullAudioPlayback(AudioPlayback):
    """Records what would have been played. Used by tests and headless runs."""

    name = "null"

    def __init__(self, *, available: bool = True) -> None:
        self._available = available
        self.played: List[Path] = []
        self.stop_calls = 0
        self._playing = False

    def is_available(self) -> bool:
        return self._available

    def play(self, path: Path) -> bool:
        if not self._available:
            return False
        self.played.append(Path(path))
        return True

    def stop(self) -> None:
        self.stop_calls += 1

    @property
    def is_playing(self) -> bool:
        return self._playing


class MciAudioPlayback(AudioPlayback):
    """Windows MCI playback for MP3 and WAV."""

    name = "mci"

    #: MCI device types by file extension.
    _DEVICE_TYPES = {
        ".mp3": "mpegvideo",
        ".wav": "waveaudio",
    }

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._alias: Optional[str] = None
        self._stop_requested = threading.Event()
        self._winmm = None
        if sys.platform == "win32":
            try:
                self._winmm = ctypes.WinDLL("winmm.dll")
            except OSError as exc:  # pragma: no cover - depends on the host
                logger.error("winmm unavailable", extra={"context": {"error": str(exc)}})

    # ------------------------------------------------------------- mci plumbing

    def _send(self, command: str) -> tuple[int, str]:
        """Issue an MCI command, returning ``(error_code, response)``."""
        if self._winmm is None:
            return 1, ""
        buffer = ctypes.create_unicode_buffer(512)
        code = self._winmm.mciSendStringW(command, buffer, 511, None)
        return int(code), buffer.value

    def _error_text(self, code: int) -> str:
        if self._winmm is None:
            return "audio device unavailable"
        buffer = ctypes.create_unicode_buffer(256)
        self._winmm.mciGetErrorStringW(code, buffer, 255)
        return buffer.value or f"MCI error {code}"

    # ----------------------------------------------------------------- playback

    def is_available(self) -> bool:
        return self._winmm is not None

    def play(self, path: Path) -> bool:
        target = Path(path)
        if self._winmm is None:
            logger.warning("No audio backend available")
            return False
        if not target.is_file():
            logger.error("Audio file missing", extra={"context": {"path": str(target)}})
            return False

        device = self._DEVICE_TYPES.get(target.suffix.lower(), "mpegvideo")
        alias = f"aarma{next(_alias_counter)}"
        self._stop_requested.clear()

        code, _ = self._send(f'open "{target}" type {device} alias {alias}')
        if code:
            logger.error("Could not open audio device",
                         extra={"context": {"error": self._error_text(code),
                                            "suffix": target.suffix}})
            return False

        with self._lock:
            self._alias = alias

        try:
            code, _ = self._send(f"play {alias}")
            if code:
                logger.error("Playback failed",
                             extra={"context": {"error": self._error_text(code)}})
                return False
            return self._wait_for_finish(alias)
        finally:
            self._send(f"close {alias}")
            with self._lock:
                if self._alias == alias:
                    self._alias = None

    def _wait_for_finish(self, alias: str) -> bool:
        """Poll until playback ends or is stopped."""
        while not self._stop_requested.is_set():
            code, mode = self._send(f"status {alias} mode")
            if code:
                return False
            if mode.strip().lower() != "playing":
                return True
            time.sleep(_POLL_INTERVAL)
        return False

    def stop(self) -> None:
        self._stop_requested.set()
        with self._lock:
            alias = self._alias
        if alias:
            self._send(f"stop {alias}")

    @property
    def is_playing(self) -> bool:
        with self._lock:
            alias = self._alias
        if not alias:
            return False
        code, mode = self._send(f"status {alias} mode")
        return not code and mode.strip().lower() == "playing"


def create_playback() -> AudioPlayback:
    """The best playback backend for this machine."""
    if sys.platform == "win32":
        backend = MciAudioPlayback()
        if backend.is_available():
            return backend
    logger.info("No platform audio backend; speech will not be played")
    return NullAudioPlayback(available=False)
