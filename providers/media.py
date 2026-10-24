"""Volume and playback, through the keys a keyboard already has.

Windows exposes volume and transport controls as virtual keys, and pressing one
does exactly what the physical key does: the shell routes it to whatever owns
audio. That is why this is done with key events rather than by reaching into a
specific application -- it works for Spotify, a browser tab and a video player
without knowing which of them is playing.

What it cannot do is *set* a level. Stepping is relative, and absolute control
needs the Core Audio API through ``pycaw``, which is not installed. So
:meth:`MediaProvider.set_volume` reports that honestly instead of pressing
volume-up an estimated number of times and claiming it landed on 30%.
"""

from __future__ import annotations

import platform
import time
from typing import Any, Optional

from providers import Availability, Provider, ProviderResult
from utils.logger import get_logger

logger = get_logger("providers.media")

#: Windows virtual key codes.  From the documented VK_ set; the shell treats
#: these identically to the keys on a keyboard's media row.
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3

KEYEVENTF_KEYUP = 0x0002

#: Each press moves Windows' volume by two points, so a "turn it down" that
#: moves a sixth of the range feels like one deliberate adjustment rather than
#: a nudge or a plunge.
DEFAULT_STEPS = 5


class MediaProvider(Provider):
    """Volume and transport control on Windows."""

    name = "media"
    description = "volume, mute and playback control"

    def __init__(self, user32: Any = None, *, system: Optional[str] = None) -> None:
        # Injected for tests: the real user32 presses keys on the machine
        # running the suite, which is not something a test should do.
        self._user32 = user32
        self._resolved = user32 is not None
        self._system = system if system is not None else platform.system()

    def _handle(self) -> Any:
        if not self._resolved:
            try:
                import ctypes  # noqa: PLC0415

                self._user32 = ctypes.windll.user32
            except Exception as exc:  # noqa: BLE001
                logger.warning("user32 unavailable", extra={
                    "context": {"error": str(exc)}})
                self._user32 = None
            self._resolved = True
        return self._user32

    def availability(self) -> Availability:
        if self._system != "Windows":
            return Availability.missing(
                self.name, f"Media control isn't implemented for {self._system}.")
        if self._handle() is None:
            return Availability.missing(
                self.name, "I can't reach the Windows media keys.")
        return Availability.ready(self.name, "volume, mute and playback")

    # ---------------------------------------------------------------- keys

    def _press(self, key: int, times: int = 1) -> bool:
        user32 = self._handle()
        if user32 is None:
            return False
        try:
            for index in range(max(1, times)):
                user32.keybd_event(key, 0, 0, 0)
                user32.keybd_event(key, 0, KEYEVENTF_KEYUP, 0)
                if index + 1 < times:
                    # Windows coalesces identical key events sent too fast.
                    time.sleep(0.01)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Media key failed", extra={
                "context": {"key": hex(key), "error": str(exc)}})
            return False

    # ------------------------------------------------------------- volume

    def volume_up(self, steps: int = DEFAULT_STEPS) -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        if self._press(VK_VOLUME_UP, steps):
            return ProviderResult.success("Volume up.", steps=steps,
                                          direction="up")
        return ProviderResult.failure("I couldn't change the volume.")

    def volume_down(self, steps: int = DEFAULT_STEPS) -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        if self._press(VK_VOLUME_DOWN, steps):
            return ProviderResult.success("Volume down.", steps=steps,
                                          direction="down")
        return ProviderResult.failure("I couldn't change the volume.")

    def mute(self) -> ProviderResult:
        """Toggle mute.  The key is a toggle; Windows exposes no 'set muted'."""
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        if self._press(VK_VOLUME_MUTE):
            return ProviderResult.success("Mute toggled.", toggled=True)
        return ProviderResult.failure("I couldn't toggle mute.")

    def set_volume(self, percent: int) -> ProviderResult:
        """Absolute level -- not supported without the Core Audio API.

        Deliberately a refusal rather than an approximation. Pressing volume-up
        fifteen times and announcing "volume set to 30%" would be a guess
        dressed as a measurement, and the user has no way to tell.
        """
        return ProviderResult.failure(
            f"I can't set the volume to exactly {percent}% -- I can only step it "
            "up or down. Setting a precise level needs the pycaw package, which "
            "isn't installed.",
            error_code="not_supported", requested_percent=percent)

    def current_volume(self) -> Optional[int]:
        """The current level, when it can be read.  ``None`` on this build."""
        return None

    # ------------------------------------------------------------ playback

    def play_pause(self) -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        if self._press(VK_MEDIA_PLAY_PAUSE):
            return ProviderResult.success("Play/pause sent.")
        return ProviderResult.failure("I couldn't reach the media controls.")

    def next_track(self) -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        if self._press(VK_MEDIA_NEXT_TRACK):
            return ProviderResult.success("Next track.")
        return ProviderResult.failure("I couldn't reach the media controls.")

    def previous_track(self) -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        if self._press(VK_MEDIA_PREV_TRACK):
            return ProviderResult.success("Previous track.")
        return ProviderResult.failure("I couldn't reach the media controls.")

    def stop(self) -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        if self._press(VK_MEDIA_STOP):
            return ProviderResult.success("Playback stopped.")
        return ProviderResult.failure("I couldn't reach the media controls.")


#: The process-wide provider.
MEDIA = MediaProvider()
