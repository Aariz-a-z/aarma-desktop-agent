"""Whether AARMA is online, and whether it is allowed to be.

Specification section 19 requires the application to state its mode plainly and
never pretend an online capability works while offline.  Two things decide the
answer:

* the user's ``network_mode`` setting -- ``offline`` is a hard override, so a
  user can keep AARMA local even on a connected machine;
* an actual reachability probe, when the mode is ``auto``.

The probe opens a TCP socket rather than making an HTTP request: it is faster,
it sends nothing, and it does not depend on any third-party service being up.
Results are cached briefly so a burst of commands does not turn into a burst of
network checks.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

from config.settings import ConfigManager
from utils.logger import get_logger

logger = get_logger("core.connectivity")

#: Well-known resolvers that answer on port 53 almost everywhere.
DEFAULT_PROBES: Tuple[Tuple[str, int], ...] = (
    ("1.1.1.1", 53),
    ("8.8.8.8", 53),
)

PROBE_TIMEOUT_SECONDS = 1.5
CACHE_SECONDS = 20.0


class NetworkMode(str, Enum):
    AUTO = "auto"        # Probe the network.
    ONLINE = "online"    # Assume connectivity; skip the probe.
    OFFLINE = "offline"  # Refuse online actions regardless of connectivity.


@dataclass(frozen=True)
class ConnectivityState:
    online: bool
    mode: NetworkMode
    checked_at: float

    @property
    def forced(self) -> bool:
        return self.mode is not NetworkMode.AUTO


class ConnectivityMonitor:
    """Cached, setting-aware connectivity check."""

    def __init__(self, config: ConfigManager,
                 probes: Sequence[Tuple[str, int]] = DEFAULT_PROBES,
                 cache_seconds: float = CACHE_SECONDS) -> None:
        self._config = config
        self._probes = tuple(probes)
        self._cache_seconds = cache_seconds
        self._lock = threading.Lock()
        self._state: Optional[ConnectivityState] = None

    @property
    def mode(self) -> NetworkMode:
        raw = (self._config.config.runtime.network_mode or "auto").strip().lower()
        try:
            return NetworkMode(raw)
        except ValueError:
            logger.warning("Unknown network mode; treating as auto",
                           extra={"context": {"value": raw}})
            return NetworkMode.AUTO

    def is_online(self, *, force_refresh: bool = False) -> bool:
        return self.state(force_refresh=force_refresh).online

    def state(self, *, force_refresh: bool = False) -> ConnectivityState:
        mode = self.mode
        if mode is NetworkMode.OFFLINE:
            return ConnectivityState(online=False, mode=mode, checked_at=time.monotonic())
        if mode is NetworkMode.ONLINE:
            return ConnectivityState(online=True, mode=mode, checked_at=time.monotonic())

        now = time.monotonic()
        with self._lock:
            cached = self._state
            if (not force_refresh and cached is not None
                    and (now - cached.checked_at) < self._cache_seconds):
                return cached
            online = self._probe()
            state = ConnectivityState(online=online, mode=mode, checked_at=now)
            self._state = state
        return state

    def _probe(self) -> bool:
        for host, port in self._probes:
            try:
                with socket.create_connection((host, port), timeout=PROBE_TIMEOUT_SECONDS):
                    return True
            except OSError:
                continue
        logger.debug("Connectivity probe found no route to the internet")
        return False

    def invalidate(self) -> None:
        """Drop the cached result, e.g. after the user changes the mode."""
        with self._lock:
            self._state = None
