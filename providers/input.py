"""Synthesised keyboard and mouse input.

This is the most dangerous capability in AARMA, and the design says so out
loud.  Anything typed goes into whichever window has focus -- which could be a
password box, a banking session or a terminal -- and a click lands wherever the
pointer is told to go.  Three decisions follow from that, and they are the
reason this module is shaped the way it is:

**1. A key is parsed, never passed through.**  :func:`parse_chord` turns
``"ctrl+shift+t"`` into virtual key codes from a fixed table, and a name that
is not in the table is refused.  There is no path from free text to the
operating system; a sentence AARMA did not understand produces an error, not an
improvised keystroke.

**2. Destructive shortcuts are refused, not sent.**  ``Alt+F4`` closes whatever
is in front with no confirmation and no undo.  Rather than quietly executing
it, :data:`REFUSED_CHORDS` sends the user to the action that *does* identify
its target and ask first -- ``close_window``.  Refusing is stronger than
confirming, and it keeps the confirmation policy in one place instead of
letting a key name route around it.

**3. Clicks need coordinates the caller supplied.**  There is no screen
understanding in this build, so "click the Submit button" is a request this
module cannot honestly satisfy.  :meth:`InputProvider.click` takes a point or
uses where the pointer already is, and coordinates outside the desktop are
rejected rather than clamped -- clamping would turn a misheard "5000, 300" into
a real click somewhere the user never named.

Layout note: only keys whose virtual key code is layout-independent are in the
table.  Punctuation lives on different physical keys under different keyboard
layouts, so a ``ctrl+;`` that works here would silently do something else on a
machine set to another layout.  Typing punctuation still works -- that goes
through Unicode injection, which is layout-independent by construction.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from providers import Availability, Provider, ProviderResult
from utils.logger import get_logger

logger = get_logger("providers.input")

# --------------------------------------------------------------- key tables

#: Modifier name -> virtual key code.  These four are the only keys allowed to
#: be held down while another is pressed.
MODIFIERS: Dict[str, int] = {
    "ctrl": 0x11, "control": 0x11,
    "shift": 0x10,
    "alt": 0x12,
    "win": 0x5B, "windows": 0x5B, "super": 0x5B, "cmd": 0x5B,
}

#: Named key -> virtual key code.  Letters and digits are added below.
NAMED_KEYS: Dict[str, int] = {
    "backspace": 0x08, "back": 0x08,
    "tab": 0x09,
    "enter": 0x0D, "return": 0x0D,
    "pause": 0x13,
    "capslock": 0x14, "caps lock": 0x14,
    "escape": 0x1B, "esc": 0x1B,
    "space": 0x20, "spacebar": 0x20,
    "pageup": 0x21, "page up": 0x21,
    "pagedown": 0x22, "page down": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    "printscreen": 0x2C, "print screen": 0x2C,
    "insert": 0x2D,
    "delete": 0x2E, "del": 0x2E,
}
NAMED_KEYS.update({chr(code): code for code in range(0x41, 0x5B)})          # a-z
NAMED_KEYS.update({chr(code).lower(): code for code in range(0x41, 0x5B)})
NAMED_KEYS.update({str(digit): 0x30 + digit for digit in range(10)})        # 0-9
NAMED_KEYS.update({f"f{n}": 0x6F + n for n in range(1, 13)})                # F1-F12


@dataclass(frozen=True)
class Chord:
    """One keystroke: zero or more modifiers held while one key is pressed."""

    modifiers: Tuple[str, ...]
    key: str
    key_code: int

    def spelling(self) -> str:
        """Canonical form, for logs and for reading the action back."""
        parts = [m.capitalize() for m in self.modifiers]
        parts.append(self.key.upper() if len(self.key) == 1 else self.key.capitalize())
        return "+".join(parts)

    @property
    def modifier_codes(self) -> Tuple[int, ...]:
        return tuple(MODIFIERS[name] for name in self.modifiers)


class UnknownKeyError(ValueError):
    """Raised when a key name is not in the fixed table."""


#: Spellings of the same key that must resolve to one table entry, so that
#: "ctrl+alt+del" cannot slip past a rule written against "ctrl+alt+delete".
_KEY_ALIASES: Dict[str, str] = {
    "del": "delete", "esc": "escape", "return": "enter", "back": "backspace",
    "spacebar": "space", "page up": "pageup", "page down": "pagedown",
    "caps lock": "capslock", "print screen": "printscreen",
}

_MODIFIER_ALIASES: Dict[str, str] = {
    "control": "ctrl", "windows": "win", "super": "win", "cmd": "win",
}


def _canonical(modifiers: Sequence[str], key: str) -> str:
    """Order-independent spelling used to look a chord up in a table.

    Modifiers are sorted and every alias is folded, so every way of writing the
    same keystroke produces the same string.  A table keyed on raw input would
    catch "ctrl+alt+delete" and miss "alt+ctrl+del", which is the same
    keystroke and exactly what a refusal list must not miss.
    """
    canonical_modifiers = sorted({
        _MODIFIER_ALIASES.get(name, name) for name in modifiers})
    return "+".join([*canonical_modifiers, _KEY_ALIASES.get(key, key)])


#: Canonical chord -> why AARMA will not synthesise it, and what to do instead.
REFUSED_CHORDS: Dict[str, str] = {
    _canonical(("alt",), "f4"):
        ("Alt+F4 closes whatever is in front with no warning. Say "
         "\"close this window\" instead — that identifies the window first "
         "and asks before closing it."),
    _canonical(("ctrl", "alt"), "delete"):
        ("Windows reserves Ctrl+Alt+Delete for the operating system; no "
         "program is allowed to send it, so I'd be pretending if I said I had."),
    _canonical(("win",), "l"):
        "Say \"lock my PC\" and I'll use the proper lock command.",
    _canonical(("ctrl", "shift"), "escape"):
        ("Say \"open Task Manager\" and I'll launch it directly rather than "
         "guessing at a shortcut."),
}

#: How many characters :meth:`InputProvider.type_text` will send at once.
MAX_TYPED_CHARACTERS = 500


def parse_chord(text: str) -> Chord:
    """Parse ``"ctrl+shift+t"``.  Raises :class:`UnknownKeyError` otherwise.

    The parser is strict on purpose.  A lenient one that skipped names it did
    not recognise would turn "press control alt delete" into a bare Delete
    keypress, which is a different and destructive instruction.
    """
    cleaned = " ".join(str(text or "").strip().lower().split())
    if not cleaned:
        raise UnknownKeyError("I need a key to press.")
    # Accept "ctrl + shift + t", "ctrl-shift-t" and "control shift t".
    for separator in ("+", "-"):
        cleaned = cleaned.replace(f" {separator} ", separator)
    if "+" in cleaned:
        parts = [p.strip() for p in cleaned.split("+") if p.strip()]
    elif "-" in cleaned and cleaned not in NAMED_KEYS:
        parts = [p.strip() for p in cleaned.split("-") if p.strip()]
    elif cleaned in NAMED_KEYS or cleaned in MODIFIERS:
        parts = [cleaned]
    else:
        parts = cleaned.split(" ")

    if not parts:
        raise UnknownKeyError("I need a key to press.")

    modifiers: List[str] = []
    for part in parts[:-1]:
        if part not in MODIFIERS:
            raise UnknownKeyError(
                f"\"{part}\" isn't a modifier I know. I understand "
                "Ctrl, Shift, Alt and Win.")
        modifiers.append(part)

    key = parts[-1]
    if key in MODIFIERS and not modifiers:
        # "press ctrl" on its own is meaningless as a command and is far more
        # likely to be a truncated phrase than a real instruction.
        raise UnknownKeyError(
            f"\"{key}\" is a modifier — tell me what to press with it, "
            "like \"Ctrl+C\".")
    if key not in NAMED_KEYS:
        raise UnknownKeyError(
            f"I don't have a key called \"{key}\". I know letters, digits, "
            "F1-F12, Enter, Escape, Tab, Space, Backspace, Delete, the arrow "
            "keys, Home, End, Page Up and Page Down.")
    return Chord(modifiers=tuple(modifiers), key=key, key_code=NAMED_KEYS[key])


def refusal_for(chord: Chord) -> Optional[str]:
    """Why this chord will not be sent, or ``None`` when it is allowed."""
    reason = REFUSED_CHORDS.get(_canonical(chord.modifiers, chord.key))
    return reason or None


# ----------------------------------------------------------------- backends

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

SM_CXSCREEN = 0
SM_CYSCREEN = 1

_BUTTONS: Dict[str, Tuple[int, int]] = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}


class User32InputBackend:
    """Real input injection through ``user32``.

    Key chords use ``keybd_event`` with virtual key codes; typed text uses
    ``SendInput`` with ``KEYEVENTF_UNICODE``.  The split is not arbitrary:
    virtual key codes are what shortcuts are defined in terms of, while Unicode
    injection bypasses the keyboard layout entirely, which is the only way to
    type a character reliably on a machine whose layout is unknown.
    """

    def __init__(self, user32: Any = None) -> None:
        self._user32 = user32
        self._resolved = user32 is not None

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

    # -- keyboard ---------------------------------------------------------

    def press(self, modifiers: Sequence[int], key_code: int) -> bool:
        user32 = self.user32
        if user32 is None:
            return False
        pressed: List[int] = []
        try:
            for code in modifiers:
                user32.keybd_event(code, 0, 0, 0)
                pressed.append(code)
            user32.keybd_event(key_code, 0, 0, 0)
            user32.keybd_event(key_code, 0, KEYEVENTF_KEYUP, 0)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Key press failed",
                         extra={"context": {"error": str(exc)}})
            return False
        finally:
            # Releasing in reverse order, and in a finally, so a failure
            # halfway through never leaves Ctrl stuck down on the user's
            # machine -- which would make every subsequent keystroke a shortcut.
            for code in reversed(pressed):
                try:
                    user32.keybd_event(code, 0, KEYEVENTF_KEYUP, 0)
                except Exception:  # noqa: BLE001
                    pass

    def type_text(self, text: str) -> bool:
        user32 = self.user32
        if user32 is None:
            return False
        try:
            import ctypes  # noqa: PLC0415
            from ctypes import wintypes  # noqa: PLC0415

            class _KeyboardInput(ctypes.Structure):
                _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                            ("dwFlags", wintypes.DWORD),
                            ("time", wintypes.DWORD),
                            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

            class _InputUnion(ctypes.Union):
                _fields_ = [("ki", _KeyboardInput)]

            class _Input(ctypes.Structure):
                _fields_ = [("type", wintypes.DWORD), ("union", _InputUnion)]

            events = []
            for character in text:
                for flags in (KEYEVENTF_UNICODE,
                              KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
                    events.append(_Input(
                        type=1,
                        union=_InputUnion(ki=_KeyboardInput(
                            wVk=0, wScan=ord(character), dwFlags=flags,
                            time=0, dwExtraInfo=None))))
            if not events:
                return True
            array = (_Input * len(events))(*events)
            sent = user32.SendInput(len(events), array, ctypes.sizeof(_Input))
            return int(sent) == len(events)
        except Exception as exc:  # noqa: BLE001
            logger.error("Typing failed", extra={"context": {"error": str(exc)}})
            return False

    # -- mouse ------------------------------------------------------------

    def screen_size(self) -> Tuple[int, int]:
        user32 = self.user32
        if user32 is None:
            return (0, 0)
        return (int(user32.GetSystemMetrics(SM_CXSCREEN)),
                int(user32.GetSystemMetrics(SM_CYSCREEN)))

    def cursor_position(self) -> Tuple[int, int]:
        user32 = self.user32
        if user32 is None:
            return (0, 0)
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415

        point = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(point))
        return (int(point.x), int(point.y))

    def move_cursor(self, x: int, y: int) -> bool:
        user32 = self.user32
        if user32 is None:
            return False
        try:
            return bool(user32.SetCursorPos(int(x), int(y)))
        except Exception as exc:  # noqa: BLE001
            logger.error("Cursor move failed",
                         extra={"context": {"error": str(exc)}})
            return False

    def click(self, button: str, times: int) -> bool:
        user32 = self.user32
        if user32 is None:
            return False
        down, up = _BUTTONS[button]
        try:
            for index in range(times):
                user32.mouse_event(down, 0, 0, 0, 0)
                user32.mouse_event(up, 0, 0, 0, 0)
                if index + 1 < times:
                    # Inside the system double-click interval, or Windows reads
                    # two separate clicks instead of one double-click.
                    time.sleep(0.05)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Click failed", extra={"context": {"error": str(exc)}})
            return False

    def scroll(self, amount: int) -> bool:
        user32 = self.user32
        if user32 is None:
            return False
        try:
            user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(amount), 0)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Scroll failed", extra={"context": {"error": str(exc)}})
            return False


# ---------------------------------------------------------------- provider

class InputProvider(Provider):
    """Keyboard and mouse injection, with the target never assumed."""

    name = "input"
    description = "keyboard and mouse control"

    def __init__(self, backend: Optional[User32InputBackend] = None, *,
                 system: Optional[str] = None) -> None:
        self._backend = backend if backend is not None else User32InputBackend()
        self._system = system if system is not None else platform.system()

    def availability(self) -> Availability:
        if self._system != "Windows":
            return Availability.missing(
                self.name,
                f"Keyboard and mouse control aren't implemented for {self._system}.")
        if not self._backend.usable():
            return Availability.missing(
                self.name, "I can't reach the Windows input system.")
        return Availability.ready(self.name, "keyboard and mouse control")

    # -- keyboard ---------------------------------------------------------

    def press(self, chord: Chord) -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        reason = refusal_for(chord)
        if reason:
            return ProviderResult.failure(reason, error_code="chord_refused",
                                          chord=chord.spelling())
        if not self._backend.press(chord.modifier_codes, chord.key_code):
            return ProviderResult.failure(f"I couldn't send {chord.spelling()}.",
                                          error_code="press_failed")
        logger.info("Key sent", extra={"context": {"chord": chord.spelling()}})
        return ProviderResult.success(f"Pressed {chord.spelling()}.",
                                      chord=chord.spelling(),
                                      key=chord.key,
                                      modifiers=list(chord.modifiers))

    def type_text(self, text: str) -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        body = str(text or "")
        if not body:
            return ProviderResult.failure("There was nothing to type.",
                                          error_code="empty_text")
        if len(body) > MAX_TYPED_CHARACTERS:
            return ProviderResult.failure(
                f"That's {len(body)} characters; I'll type at most "
                f"{MAX_TYPED_CHARACTERS} at a time.",
                error_code="too_long")
        if not self._backend.type_text(body):
            return ProviderResult.failure("I couldn't type that.",
                                          error_code="type_failed")
        # The text itself is deliberately absent from the log: typed content is
        # whatever the user dictated, which may be a message or a password.
        logger.info("Text typed", extra={"context": {"characters": len(body)}})
        return ProviderResult.success(f"Typed {len(body)} characters.",
                                      characters=len(body))

    # -- mouse ------------------------------------------------------------

    def screen_size(self) -> Tuple[int, int]:
        return self._backend.screen_size()

    def _check_point(self, x: int, y: int) -> Optional[str]:
        width, height = self._backend.screen_size()
        if width <= 0 or height <= 0:
            return "I can't tell how big the screen is, so I won't guess at a point on it."
        if not (0 <= x < width and 0 <= y < height):
            return (f"({x}, {y}) is off this screen, which is {width} by "
                    f"{height}. I won't move the pointer somewhere you didn't name.")
        return None

    def move(self, x: int, y: int) -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        problem = self._check_point(x, y)
        if problem:
            return ProviderResult.failure(problem, error_code="off_screen",
                                          x=x, y=y)
        if not self._backend.move_cursor(x, y):
            return ProviderResult.failure("I couldn't move the pointer.",
                                          error_code="move_failed")
        return ProviderResult.success(f"Pointer moved to ({x}, {y}).", x=x, y=y)

    def click(self, *, x: Optional[int] = None, y: Optional[int] = None,
              button: str = "left", times: int = 1) -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        if button not in _BUTTONS:
            return ProviderResult.failure(
                f"I don't have a \"{button}\" mouse button.",
                error_code="unknown_button")
        if (x is None) != (y is None):
            return ProviderResult.failure(
                "I need both an x and a y to click at a point.",
                error_code="incomplete_point")
        if x is not None and y is not None:
            problem = self._check_point(x, y)
            if problem:
                return ProviderResult.failure(problem, error_code="off_screen",
                                              x=x, y=y)
            if not self._backend.move_cursor(x, y):
                return ProviderResult.failure("I couldn't move the pointer.",
                                              error_code="move_failed")
        where = (x, y) if x is not None else self._backend.cursor_position()
        if not self._backend.click(button, max(1, int(times))):
            return ProviderResult.failure("I couldn't click.",
                                          error_code="click_failed")
        label = {1: "Clicked", 2: "Double-clicked"}.get(times, f"Clicked {times} times")
        logger.info("Mouse click", extra={"context": {
            "button": button, "times": times}})
        return ProviderResult.success(f"{label} at ({where[0]}, {where[1]}).",
                                      x=where[0], y=where[1], button=button,
                                      times=times)

    def scroll(self, notches: int) -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        if not self._backend.scroll(int(notches) * 120):
            return ProviderResult.failure("I couldn't scroll.",
                                          error_code="scroll_failed")
        direction = "up" if notches > 0 else "down"
        return ProviderResult.success(f"Scrolled {direction}.", notches=notches)


#: The process-wide provider.
INPUT = InputProvider()


__all__ = ["Chord", "InputProvider", "User32InputBackend", "UnknownKeyError",
           "MODIFIERS", "NAMED_KEYS", "REFUSED_CHORDS", "MAX_TYPED_CHARACTERS",
           "INPUT", "parse_chord", "refusal_for"]
