"""Windows: the windows themselves.

Every capability in this file answers one question first -- *which window?* --
and the answer is always a real handle obtained from the operating system, not
an assumption about what has focus.  That matters because the failure mode of
window automation is not an error message, it is the right action performed on
the wrong window: a "close this tab" that closes an unsaved document.

So three rules hold throughout:

* **The target is identified before anything is sent.**  Callers get a
  :class:`WindowInfo` carrying the process that owns the window, and can refuse
  when it is not what they expected.  :mod:`actions.browser` uses exactly this
  to avoid pressing Ctrl+W at a text editor.
* **Window state changes go through the Window API**, not through synthesised
  keystrokes.  ``ShowWindow`` minimises a specific handle; Win+Down minimises
  whatever happens to be in front at the moment the key lands.
* **Focus changes are verified.**  Windows may refuse ``SetForegroundWindow``
  depending on which process last received input, and it refuses *silently* by
  returning zero.  Every focus call here reads the foreground window back and
  reports what actually happened.

The win32 calls are reached through :class:`Win32Backend`, which exists so the
test-suite can drive this module with a fake desktop.  A test that really
minimised windows would be minimising the windows of whoever ran the suite.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from providers import Availability, Provider, ProviderResult
from utils.logger import get_logger

logger = get_logger("providers.desktop")

# ---------------------------------------------------------------- constants

SW_MAXIMIZE = 3
SW_MINIMIZE = 6
SW_RESTORE = 9

WM_CLOSE = 0x0010

GW_HWNDNEXT = 2

#: Process image name -> the browser a user would name it by.  Membership in
#: this map is what licenses a browser keyboard shortcut; anything absent is
#: treated as "not a browser" and the shortcut is withheld.
BROWSER_PROCESSES: Dict[str, str] = {
    "chrome.exe": "Chrome",
    "msedge.exe": "Edge",
    "firefox.exe": "Firefox",
    "brave.exe": "Brave",
    "opera.exe": "Opera",
    "vivaldi.exe": "Vivaldi",
    "chromium.exe": "Chromium",
}

#: Windows that exist but are not windows a person would switch to.  The shell
#: keeps several always-present top-level windows with real titles.
_SHELL_TITLES = frozenset({
    "Program Manager", "Windows Input Experience", "Windows Shell Experience Host",
    "Setup", "Microsoft Text Input Application",
})


@dataclass(frozen=True)
class WindowInfo:
    """One top-level window, as the OS describes it right now."""

    handle: int
    title: str
    process_name: str = ""
    pid: int = 0
    minimized: bool = False
    maximized: bool = False

    @property
    def browser(self) -> Optional[str]:
        """The browser this window belongs to, or ``None`` if it is not one."""
        return BROWSER_PROCESSES.get(self.process_name.lower())

    @property
    def is_browser(self) -> bool:
        return self.browser is not None

    def label(self) -> str:
        """How to name this window back to the user."""
        if self.title and self.process_name:
            return f"{self.title} ({self.process_name})"
        return self.title or self.process_name or f"window {self.handle}"

    def as_dict(self) -> Dict[str, Any]:
        return {"handle": self.handle, "title": self.title,
                "process_name": self.process_name, "pid": self.pid,
                "minimized": self.minimized, "maximized": self.maximized,
                "browser": self.browser}


# ----------------------------------------------------------------- backends

class Win32Backend:
    """The real desktop, reached through ``user32`` and :mod:`psutil`.

    ``ctypes`` rather than ``pywin32`` so this works on a stripped install --
    :mod:`providers.media` already reaches ``user32`` the same way, and adding
    a second mechanism for the same DLL would only be a second thing to break.
    """

    def __init__(self) -> None:
        self._user32: Any = None
        self._resolved = False

    # -- plumbing ---------------------------------------------------------

    @property
    def user32(self) -> Any:
        if not self._resolved:
            try:
                import ctypes  # noqa: PLC0415

                self._user32 = ctypes.windll.user32
            except Exception as exc:  # noqa: BLE001 - absent off Windows
                logger.warning("user32 unavailable",
                               extra={"context": {"error": str(exc)}})
                self._user32 = None
            self._resolved = True
        return self._user32

    def usable(self) -> bool:
        return self.user32 is not None

    # -- queries ----------------------------------------------------------

    def foreground_window(self) -> int:
        user32 = self.user32
        return int(user32.GetForegroundWindow()) if user32 else 0

    def window_title(self, handle: int) -> str:
        import ctypes  # noqa: PLC0415

        user32 = self.user32
        if user32 is None:
            return ""
        length = int(user32.GetWindowTextLengthW(handle))
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, buffer, length + 1)
        return buffer.value

    def is_visible(self, handle: int) -> bool:
        user32 = self.user32
        return bool(user32.IsWindowVisible(handle)) if user32 else False

    def is_minimized(self, handle: int) -> bool:
        user32 = self.user32
        return bool(user32.IsIconic(handle)) if user32 else False

    def is_maximized(self, handle: int) -> bool:
        user32 = self.user32
        return bool(user32.IsZoomed(handle)) if user32 else False

    def process_id(self, handle: int) -> int:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        user32 = self.user32
        if user32 is None:
            return 0
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        return int(pid.value)

    def process_name(self, pid: int) -> str:
        if not pid:
            return ""
        try:
            import psutil  # noqa: PLC0415

            return psutil.Process(pid).name()
        except Exception:  # noqa: BLE001 - the process may have exited
            return ""

    def top_level_windows(self) -> List[int]:
        """Visible top-level windows, in Z-order (front first).

        Z-order matters: "the previous window" means the one behind the
        foreground one, and only the OS knows that ordering.
        """
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        user32 = self.user32
        if user32 is None:
            return []
        handles: List[int] = []
        # GW_HWNDNEXT from the foreground window walks the Z-order directly,
        # but it misses anything in front of it, so enumerate and keep order:
        # EnumWindows is documented to visit windows in Z-order.
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND,
                                           wintypes.LPARAM)

        def _collect(handle, _param):  # pragma: no cover - driven by the OS
            handles.append(int(handle))
            return True

        user32.EnumWindows(callback_type(_collect), 0)
        return handles

    # -- commands ---------------------------------------------------------

    def show_window(self, handle: int, command: int) -> bool:
        user32 = self.user32
        if user32 is None:
            return False
        user32.ShowWindow(handle, command)
        # ShowWindow returns the *previous* visibility, not success, so its
        # return value says nothing useful here; the caller re-reads state.
        return True

    def set_foreground(self, handle: int) -> bool:
        user32 = self.user32
        if user32 is None:
            return False
        return bool(user32.SetForegroundWindow(handle))

    def close_window(self, handle: int) -> bool:
        user32 = self.user32
        if user32 is None:
            return False
        # WM_CLOSE is a request: the application may show "save changes?" and
        # stay open. That is the desired behaviour -- it is the user's document.
        return bool(user32.PostMessageW(handle, WM_CLOSE, 0, 0))

    def window_rect(self, handle: int) -> Optional[Tuple[int, int, int, int]]:
        """``(left, top, right, bottom)`` in screen pixels, or ``None``."""
        user32 = self.user32
        if user32 is None:
            return None
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        rect = wintypes.RECT()
        if not user32.GetWindowRect(handle, ctypes.byref(rect)):
            return None
        return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


# ---------------------------------------------------------------- provider

class WindowProvider(Provider):
    """Find, focus and re-state the windows on this desktop."""

    name = "windows"
    description = "find, focus, minimise, maximise and close windows"

    def __init__(self, backend: Optional[Win32Backend] = None, *,
                 system: Optional[str] = None,
                 settle: Callable[[], None] = lambda: time.sleep(0.05)) -> None:
        self._backend = backend if backend is not None else Win32Backend()
        self._system = system if system is not None else platform.system()
        # A short pause after a state change, because ShowWindow is asynchronous
        # and reading the state back immediately can observe the old value.
        self._settle = settle

    # -- availability ------------------------------------------------------

    def availability(self) -> Availability:
        if self._system != "Windows":
            return Availability.missing(
                self.name, f"Window control isn't implemented for {self._system}.")
        if not self._backend.usable():
            return Availability.missing(
                self.name, "I can't reach the Windows window manager.")
        return Availability.ready(self.name, "window control")

    # -- reading -----------------------------------------------------------

    def _describe(self, handle: int) -> Optional[WindowInfo]:
        if not handle:
            return None
        pid = self._backend.process_id(handle)
        return WindowInfo(
            handle=handle,
            title=self._backend.window_title(handle),
            process_name=self._backend.process_name(pid),
            pid=pid,
            minimized=self._backend.is_minimized(handle),
            maximized=self._backend.is_maximized(handle),
        )

    def active_window(self) -> Optional[WindowInfo]:
        """The window with focus right now, or ``None`` if there isn't one."""
        if not self.is_available():
            return None
        return self._describe(self._backend.foreground_window())

    def list_windows(self, *, include_untitled: bool = False) -> List[WindowInfo]:
        """Visible, titled top-level windows, front-most first."""
        if not self.is_available():
            return []
        windows: List[WindowInfo] = []
        for handle in self._backend.top_level_windows():
            if not self._backend.is_visible(handle):
                continue
            info = self._describe(handle)
            if info is None:
                continue
            if not include_untitled and not info.title.strip():
                continue
            if info.title in _SHELL_TITLES:
                continue
            windows.append(info)
        return windows

    def find_window(self, query: str) -> List[WindowInfo]:
        """Windows matching what the user called them, best match first.

        Matching is deliberately layered rather than fuzzy: an exact process or
        title match is a different kind of evidence from a substring hit, and
        collapsing them into one score is how "switch to Code" starts focusing
        a browser tab whose title mentions code.
        """
        needle = " ".join(str(query or "").lower().split())
        if not needle:
            return []
        for filler in ("the ", "my ", "to "):
            if needle.startswith(filler):
                needle = needle[len(filler):]
        exact_process: List[WindowInfo] = []
        exact_title: List[WindowInfo] = []
        partial: List[WindowInfo] = []
        for info in self.list_windows():
            process = info.process_name.lower()
            stem = process[:-4] if process.endswith(".exe") else process
            title = info.title.lower()
            if needle in (process, stem):
                exact_process.append(info)
            elif title == needle:
                exact_title.append(info)
            elif needle in title or needle in stem:
                partial.append(info)
        return exact_process + exact_title + partial

    def rect(self, window: WindowInfo) -> Optional[Tuple[int, int, int, int]]:
        """Where a window is on screen, or ``None`` when the OS will not say.

        ``None`` rather than a guessed box: a screenshot cropped to coordinates
        nobody verified is a picture of the wrong thing, and the caller cannot
        tell from the file that it went wrong.
        """
        if not self.is_available():
            return None
        box = self._backend.window_rect(window.handle)
        if box is None:
            return None
        left, top, right, bottom = box
        if right <= left or bottom <= top:
            return None
        return box

    def previous_window(self) -> Optional[WindowInfo]:
        """The window immediately behind the foreground one."""
        windows = self.list_windows()
        return windows[1] if len(windows) > 1 else None

    # -- writing -----------------------------------------------------------

    def _state_change(self, window: WindowInfo, command: int, *, verb: str,
                      expect: Optional[Callable[[WindowInfo], bool]] = None
                      ) -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        if not self._backend.show_window(window.handle, command):
            return ProviderResult.failure(f"I couldn't {verb} {window.label()}.",
                                          error_code="window_command_failed")
        self._settle()
        current = self._describe(window.handle)
        if expect is not None and current is not None and not expect(current):
            # Report what the OS actually did. Some windows refuse to maximise.
            return ProviderResult.failure(
                f"I asked Windows to {verb} {window.label()}, but it stayed as it was.",
                error_code="window_unchanged", window=window.as_dict())
        logger.info("Window state changed", extra={"context": {
            "verb": verb, "process": window.process_name}})
        return ProviderResult.success(f"{window.label()} {verb}d.",
                                      window=(current or window).as_dict())

    def minimize(self, window: WindowInfo) -> ProviderResult:
        return self._state_change(window, SW_MINIMIZE, verb="minimise",
                                  expect=lambda w: w.minimized)

    def maximize(self, window: WindowInfo) -> ProviderResult:
        return self._state_change(window, SW_MAXIMIZE, verb="maximise",
                                  expect=lambda w: w.maximized or not w.minimized)

    def restore(self, window: WindowInfo) -> ProviderResult:
        return self._state_change(window, SW_RESTORE, verb="restore",
                                  expect=lambda w: not w.minimized)

    def focus(self, window: WindowInfo) -> ProviderResult:
        """Bring a window to the front, and verify that it worked.

        Windows restricts which process may steal focus.  ``SetForegroundWindow``
        returning zero -- or returning non-zero while the foreground window does
        not change -- both mean the switch did not happen, and saying "switched
        to Chrome" in that case would be a plain falsehood.
        """
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        if window.minimized:
            self._backend.show_window(window.handle, SW_RESTORE)
            self._settle()
        self._backend.set_foreground(window.handle)
        self._settle()
        if self._backend.foreground_window() != window.handle:
            return ProviderResult.failure(
                f"Windows wouldn't let me bring {window.label()} to the front. "
                "That usually means another program is holding focus.",
                error_code="focus_refused", window=window.as_dict())
        logger.info("Window focused", extra={"context": {
            "process": window.process_name}})
        return ProviderResult.success(f"Switched to {window.label()}.",
                                      window=window.as_dict())

    def close(self, window: WindowInfo) -> ProviderResult:
        """Ask a window to close.  The application may still prompt to save."""
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        if not self._backend.close_window(window.handle):
            return ProviderResult.failure(f"I couldn't close {window.label()}.",
                                          error_code="close_failed")
        logger.info("Window close requested", extra={"context": {
            "process": window.process_name}})
        return ProviderResult.success(
            f"Asked {window.label()} to close.", window=window.as_dict())


#: The process-wide provider.
WINDOWS = WindowProvider()


def active_browser(provider: Optional[WindowProvider] = None
                   ) -> Tuple[Optional[WindowInfo], Optional[str]]:
    """The focused window if it is a browser, plus why not when it isn't.

    Returns ``(window, None)`` on success and ``(None, reason)`` otherwise, so
    a caller cannot accidentally treat "no window" as "a browser".
    """
    resolved = provider if provider is not None else WINDOWS
    if not resolved.is_available():
        return None, resolved.availability().detail
    window = resolved.active_window()
    if window is None:
        return None, "I can't tell which window is in front."
    if not window.is_browser:
        known = ", ".join(sorted({name for name in BROWSER_PROCESSES.values()}))
        return None, (
            f"The window in front is {window.label()}, which isn't a browser, "
            f"so I won't send browser shortcuts to it. I can do this in "
            f"{known} — switch to one and ask again.")
    return window, None


__all__ = ["BROWSER_PROCESSES", "WindowInfo", "WindowProvider", "Win32Backend",
           "WINDOWS", "active_browser", "SW_MAXIMIZE", "SW_MINIMIZE",
           "SW_RESTORE"]
