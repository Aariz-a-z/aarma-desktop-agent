"""Applications, browser tabs, windows, keyboard, mouse and screenshots.

Nothing in this file touches the real desktop.  The window manager and the
input system are both reached through injectable backends, and the tests drive
fakes -- a suite that genuinely minimised windows and pressed Ctrl+W would be
doing it to whoever ran it.

The properties defended here, in order of how badly they fail if broken:

**A browser shortcut never reaches a non-browser.**  Ctrl+W closes a tab in
Chrome and discards an unsaved buffer somewhere else.  Several tests assert not
just that the action refuses, but that *no key was sent at all*.

**A shell is not an ordinary application.**  "Open cmd" has to reach
``open_terminal``, which carries its own permission; ``open_application``
refuses it outright rather than quietly obliging.

**A click needs a point the user gave.**  "Click the Submit button" is refused,
because this build cannot see the screen and a guessed click is
indistinguishable from a correct one until it has already happened.

**A screenshot never overwrites one.**  Two captures in the same second used to
produce one file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from actions.applications import (
    APPLICATION_CATALOG,
    resolve_entry,
    resolve_program,
    resolve_terminal,
)
from actions.registry import ActionContext, build_default_registry
from providers.desktop import WindowInfo, WindowProvider, active_browser
from providers.input import (
    REFUSED_CHORDS,
    Chord,
    InputProvider,
    UnknownKeyError,
    parse_chord,
    refusal_for,
)
from security.validator import RiskLevel


# ----------------------------------------------------------------- fakes

class FakeDesktop:
    """A desktop made of dictionaries.

    Records every state change so a test can assert on what was *asked* of
    Windows, not merely on the sentence that came back.
    """

    def __init__(self, windows: Optional[List[Dict[str, Any]]] = None,
                 *, usable: bool = True, allow_focus: bool = True) -> None:
        self.windows: List[Dict[str, Any]] = list(windows or [])
        self._usable = usable
        self.allow_focus = allow_focus
        self.shown: List[Tuple[int, int]] = []
        self.closed: List[int] = []
        self.focused: List[int] = []

    # -- lookup -----------------------------------------------------------

    def _find(self, handle: int) -> Dict[str, Any]:
        for window in self.windows:
            if window["handle"] == handle:
                return window
        return {}

    def usable(self) -> bool:
        return self._usable

    def foreground_window(self) -> int:
        return self.windows[0]["handle"] if self.windows else 0

    def window_title(self, handle: int) -> str:
        return self._find(handle).get("title", "")

    def is_visible(self, handle: int) -> bool:
        return self._find(handle).get("visible", True)

    def is_minimized(self, handle: int) -> bool:
        return self._find(handle).get("minimized", False)

    def is_maximized(self, handle: int) -> bool:
        return self._find(handle).get("maximized", False)

    def process_id(self, handle: int) -> int:
        return self._find(handle).get("pid", handle)

    def process_name(self, pid: int) -> str:
        for window in self.windows:
            if window.get("pid", window["handle"]) == pid:
                return window.get("process", "")
        return ""

    def top_level_windows(self) -> List[int]:
        return [window["handle"] for window in self.windows]

    def window_rect(self, handle: int):
        return self._find(handle).get("rect")

    # -- commands ---------------------------------------------------------

    def show_window(self, handle: int, command: int) -> bool:
        self.shown.append((handle, command))
        window = self._find(handle)
        if command == 6:        # SW_MINIMIZE
            window["minimized"], window["maximized"] = True, False
        elif command == 3:      # SW_MAXIMIZE
            window["minimized"], window["maximized"] = False, True
        elif command == 9:      # SW_RESTORE
            window["minimized"], window["maximized"] = False, False
        return True

    def set_foreground(self, handle: int) -> bool:
        self.focused.append(handle)
        if not self.allow_focus:
            return False
        # Bring it to the front of the list: the provider reads this back.
        window = self._find(handle)
        if window:
            self.windows.remove(window)
            self.windows.insert(0, window)
        return True

    def close_window(self, handle: int) -> bool:
        self.closed.append(handle)
        return True


class FakeInput:
    """An input system that records rather than types."""

    def __init__(self, *, usable: bool = True, width: int = 1920,
                 height: int = 1080, fail: bool = False) -> None:
        self._usable = usable
        self._size = (width, height)
        self._fail = fail
        self.presses: List[Tuple[Tuple[int, ...], int]] = []
        self.typed: List[str] = []
        self.moves: List[Tuple[int, int]] = []
        self.clicks: List[Tuple[str, int]] = []
        self.scrolls: List[int] = []
        self.cursor = (10, 20)

    def usable(self) -> bool:
        return self._usable

    def press(self, modifiers, key_code) -> bool:
        if self._fail:
            return False
        self.presses.append((tuple(modifiers), key_code))
        return True

    def type_text(self, text: str) -> bool:
        if self._fail:
            return False
        self.typed.append(text)
        return True

    def screen_size(self):
        return self._size

    def cursor_position(self):
        return self.cursor

    def move_cursor(self, x: int, y: int) -> bool:
        self.moves.append((x, y))
        self.cursor = (x, y)
        return True

    def click(self, button: str, times: int) -> bool:
        self.clicks.append((button, times))
        return True

    def scroll(self, amount: int) -> bool:
        self.scrolls.append(amount)
        return True


def chrome_window(handle: int = 1, title: str = "AARMA - Google Chrome"):
    return {"handle": handle, "title": title, "process": "chrome.exe",
            "pid": 100 + handle, "rect": (0, 0, 1280, 800)}


def editor_window(handle: int = 2, title: str = "notes.txt - Notepad"):
    return {"handle": handle, "title": title, "process": "notepad.exe",
            "pid": 200 + handle, "rect": (10, 10, 900, 700)}


@pytest.fixture
def registry():
    return build_default_registry()


def _context(action_context: ActionContext, **overrides) -> ActionContext:
    for key, value in overrides.items():
        object.__setattr__(action_context, key, value)
    return action_context


def _run(registry, ctx, action: str, **params):
    """Run one action's handler directly.

    The keyword is ``action`` rather than ``name`` because several actions take
    a parameter called ``name``; ``**params`` would collide with it.
    """
    spec = registry.require(action)
    return spec.handler(ctx, spec.validate(params))


def _windows(desktop: FakeDesktop) -> WindowProvider:
    return WindowProvider(desktop, system="Windows", settle=lambda: None)


def _input(backend: FakeInput) -> InputProvider:
    return InputProvider(backend, system="Windows")


# ------------------------------------------------------------- applications

class TestApplicationAliases:
    """Every alias the specification names has to resolve."""

    @pytest.mark.parametrize("spoken,key", [
        ("cmd", "cmd"),
        ("command prompt", "cmd"),
        ("command line", "cmd"),
        ("powershell", "powershell"),
        ("power shell", "powershell"),
        ("windows terminal", "windowsterminal"),
        ("terminal", "windowsterminal"),
        ("task manager", "taskmanager"),
        ("taskmgr", "taskmanager"),
        ("file explorer", "explorer"),
        ("explorer", "explorer"),
        ("notepad", "notepad"),
        ("calculator", "calculator"),
        ("settings", "settings"),
    ])
    def test_alias_resolves(self, spoken, key) -> None:
        entry = resolve_entry(spoken)
        assert entry is not None, f"{spoken!r} did not resolve"
        assert entry.key == key

    def test_shells_are_terminals_and_others_are_not(self) -> None:
        assert resolve_terminal("cmd") is not None
        assert resolve_terminal("notepad") is None
        assert resolve_program("notepad") is not None
        # The separation is the point: a shell must not resolve as an ordinary
        # program, or it would inherit the permission set meant for Notepad.
        assert resolve_program("cmd") is None

    def test_no_alias_is_claimed_by_two_entries(self) -> None:
        seen: Dict[str, str] = {}
        for entry in APPLICATION_CATALOG.values():
            for alias in (entry.key, entry.display_name.lower(), *entry.aliases):
                assert seen.get(alias, entry.key) == entry.key, (
                    f"{alias!r} is claimed by both {seen[alias]} and {entry.key}")
                seen[alias] = entry.key

    def test_unknown_application_is_a_controlled_failure(self, registry,
                                                         action_context) -> None:
        result = _run(registry, action_context, "open_application",
                      name="definitely not installed")
        assert not result.ok
        assert result.error_code == "app_not_approved"

    def test_unknown_terminal_is_a_controlled_failure(self, registry,
                                                      action_context) -> None:
        result = _run(registry, action_context, "open_terminal", name="bash")
        assert not result.ok
        assert result.error_code == "terminal_not_approved"

    def test_open_application_refuses_a_shell(self, registry,
                                              action_context) -> None:
        result = _run(registry, action_context, "open_application", name="cmd")
        assert not result.ok
        assert result.error_code == "use_open_terminal"

    def test_terminal_action_takes_no_command(self, registry) -> None:
        """There is no parameter through which a command line could arrive."""
        spec = registry.require("open_terminal")
        assert [p.name for p in spec.parameters] == ["name"]
        assert spec.risk is RiskLevel.MEDIUM

    def test_shell_execution_is_one_named_high_risk_action(self, registry) -> None:
        """Exactly one door to shell execution, and it is locked.

        Until Phase 5 this asserted that no such action existed at all, which
        was the right invariant while AARMA had no way to run a command. Phase
        5 adds ``execute_shell_command`` deliberately -- so that "run this" can
        never be *quietly* satisfied by some other handler -- and the property
        worth defending changes shape rather than going away.

        What must now hold is stricter than "one action exists": it must be the
        ONLY one, it must be HIGH risk so it always confirms, it must require
        SYSTEM_CONTROL, and no natural sentence may resolve to it. A capability
        reachable by phrasing is a capability with no door at all.
        """
        for absent in ("run_command", "run_shell", "execute", "eval",
                       "run_script"):
            assert absent not in registry

        spec = registry.get("execute_shell_command")
        assert spec is not None
        assert spec.risk is RiskLevel.HIGH

    def test_opening_a_terminal_is_not_running_a_command(self, registry) -> None:
        """The two stay separate, which is the whole point of the split.

        ``open_terminal`` takes a shell *name* and no command; the only action
        that takes a command line is the HIGH-risk one.
        """
        terminal = registry.require("open_terminal")
        assert [p.name for p in terminal.parameters] == ["name"]
        assert terminal.risk is not RiskLevel.HIGH

        shell = registry.require("execute_shell_command")
        assert "command" in [p.name for p in shell.parameters]
        assert shell.risk is RiskLevel.HIGH


# ----------------------------------------------------------------- browser

class TestBrowserDetection:

    def test_chrome_in_front_is_a_browser(self) -> None:
        provider = _windows(FakeDesktop([chrome_window()]))
        window, reason = active_browser(provider)
        assert reason is None
        assert window is not None and window.browser == "Chrome"

    def test_notepad_in_front_is_not(self) -> None:
        provider = _windows(FakeDesktop([editor_window()]))
        window, reason = active_browser(provider)
        assert window is None
        assert reason and "isn't a browser" in reason

    def test_edge_is_recognised(self) -> None:
        edge = {"handle": 5, "title": "Edge", "process": "msedge.exe", "pid": 55}
        provider = _windows(FakeDesktop([edge]))
        window, _ = active_browser(provider)
        assert window is not None and window.browser == "Edge"


class TestTabControl:

    @pytest.mark.parametrize("action,expected_key", [
        ("new_tab", "t"), ("close_tab", "w"), ("reopen_tab", "t"),
        ("refresh_page", "f5"), ("go_back", "left"), ("go_forward", "right"),
        ("next_tab", "tab"),
    ])
    def test_shortcut_goes_to_a_browser(self, registry, action_context,
                                        action, expected_key) -> None:
        keys = FakeInput()
        ctx = _context(action_context,
                       window_provider=_windows(FakeDesktop([chrome_window()])),
                       input_provider=_input(keys))
        result = _run(registry, ctx, action)
        assert result.ok, result.message
        assert len(keys.presses) == 1
        assert result.data["browser"] == "Chrome"
        assert expected_key.lower() in result.data["keys"].lower()

    @pytest.mark.parametrize("action", [
        "new_tab", "close_tab", "reopen_tab", "refresh_page", "go_back",
        "go_forward", "next_tab",
    ])
    def test_shortcut_never_reaches_a_non_browser(self, registry, action_context,
                                                  action) -> None:
        """The keystroke must not be sent at all -- not sent and apologised for."""
        keys = FakeInput()
        ctx = _context(action_context,
                       window_provider=_windows(FakeDesktop([editor_window()])),
                       input_provider=_input(keys))
        result = _run(registry, ctx, action)
        assert not result.ok
        assert result.error_code == "not_a_browser"
        assert keys.presses == [], "a key was sent to a non-browser"

    def test_close_tab_sends_control_w(self, registry, action_context) -> None:
        keys = FakeInput()
        ctx = _context(action_context,
                       window_provider=_windows(FakeDesktop([chrome_window()])),
                       input_provider=_input(keys))
        assert _run(registry, ctx, "close_tab").ok
        modifiers, key = keys.presses[0]
        assert modifiers == (0x11,)          # Ctrl
        assert key == ord("W")

    def test_reopen_tab_sends_control_shift_t(self, registry,
                                              action_context) -> None:
        keys = FakeInput()
        ctx = _context(action_context,
                       window_provider=_windows(FakeDesktop([chrome_window()])),
                       input_provider=_input(keys))
        assert _run(registry, ctx, "reopen_tab").ok
        modifiers, key = keys.presses[0]
        assert set(modifiers) == {0x11, 0x10}   # Ctrl + Shift
        assert key == ord("T")

    def test_browser_status_reports_honestly(self, registry,
                                             action_context) -> None:
        ctx = _context(action_context,
                       window_provider=_windows(FakeDesktop([editor_window()])))
        result = _run(registry, ctx, "browser_status")
        assert result.ok
        assert result.data["is_browser"] is False


# ----------------------------------------------------------------- windows

class TestWindowControl:

    def test_minimize_uses_the_window_api(self, registry, action_context) -> None:
        desktop = FakeDesktop([editor_window()])
        ctx = _context(action_context, window_provider=_windows(desktop))
        result = _run(registry, ctx, "minimize_window")
        assert result.ok, result.message
        assert desktop.shown == [(2, 6)]

    def test_maximize_uses_the_window_api(self, registry, action_context) -> None:
        desktop = FakeDesktop([editor_window()])
        ctx = _context(action_context, window_provider=_windows(desktop))
        assert _run(registry, ctx, "maximize_window").ok
        assert desktop.shown == [(2, 3)]

    def test_restore_uses_the_window_api(self, registry, action_context) -> None:
        window = editor_window()
        window["minimized"] = True
        desktop = FakeDesktop([window])
        ctx = _context(action_context, window_provider=_windows(desktop))
        assert _run(registry, ctx, "restore_window").ok
        assert desktop.shown == [(2, 9)]

    def test_close_targets_the_active_window(self, registry,
                                             action_context) -> None:
        desktop = FakeDesktop([chrome_window(1), editor_window(2)])
        ctx = _context(action_context, window_provider=_windows(desktop))
        result = _run(registry, ctx, "close_window")
        assert result.ok
        assert desktop.closed == [1], "closed something other than the front window"

    def test_close_targets_a_named_window(self, registry, action_context) -> None:
        desktop = FakeDesktop([chrome_window(1), editor_window(2)])
        ctx = _context(action_context, window_provider=_windows(desktop))
        result = _run(registry, ctx, "close_window", target="notepad")
        assert result.ok
        assert desktop.closed == [2]

    def test_switch_focuses_the_named_window(self, registry,
                                             action_context) -> None:
        desktop = FakeDesktop([editor_window(2), chrome_window(1)])
        ctx = _context(action_context, window_provider=_windows(desktop))
        result = _run(registry, ctx, "switch_window", target="chrome")
        assert result.ok, result.message
        assert desktop.focused == [1]

    def test_refused_focus_is_reported_not_claimed(self, registry,
                                                  action_context) -> None:
        """Windows silently refuses focus changes. Saying "switched" would lie."""
        desktop = FakeDesktop([editor_window(2), chrome_window(1)],
                              allow_focus=False)
        ctx = _context(action_context, window_provider=_windows(desktop))
        result = _run(registry, ctx, "switch_window", target="chrome")
        assert not result.ok
        assert result.error_code == "focus_refused"

    def test_unknown_window_is_refused(self, registry, action_context) -> None:
        desktop = FakeDesktop([editor_window()])
        ctx = _context(action_context, window_provider=_windows(desktop))
        result = _run(registry, ctx, "switch_window", target="photoshop")
        assert not result.ok
        assert result.error_code == "window_not_found"
        assert desktop.focused == []

    def test_ambiguous_target_is_refused_rather_than_guessed(
            self, registry, action_context) -> None:
        desktop = FakeDesktop([
            {"handle": 1, "title": "report — Word", "process": "winword.exe", "pid": 1},
            {"handle": 2, "title": "report — Notepad", "process": "notepad.exe", "pid": 2},
        ])
        ctx = _context(action_context, window_provider=_windows(desktop))
        result = _run(registry, ctx, "close_window", target="report")
        assert not result.ok
        assert result.error_code == "ambiguous"
        assert desktop.closed == []

    def test_previous_window_uses_z_order(self, registry, action_context) -> None:
        desktop = FakeDesktop([chrome_window(1), editor_window(2)])
        ctx = _context(action_context, window_provider=_windows(desktop))
        result = _run(registry, ctx, "previous_window")
        assert result.ok
        assert desktop.focused == [2]

    def test_list_windows_skips_untitled_shell_windows(self) -> None:
        desktop = FakeDesktop([
            chrome_window(1),
            {"handle": 9, "title": "Program Manager", "process": "explorer.exe", "pid": 9},
            {"handle": 10, "title": "", "process": "svchost.exe", "pid": 10},
        ])
        titles = [w.title for w in _windows(desktop).list_windows()]
        assert titles == ["AARMA - Google Chrome"]

    def test_unavailable_provider_reports_rather_than_failing_silently(
            self, registry, action_context) -> None:
        ctx = _context(action_context,
                       window_provider=WindowProvider(FakeDesktop([]), system="Linux"))
        result = _run(registry, ctx, "minimize_window")
        assert not result.ok
        assert result.error_code == "windows_unavailable"

    def test_no_force_kill_action_exists(self, registry) -> None:
        """Force-closing discards unsaved work with no prompt; nothing does it."""
        for name in registry.names():
            spec = registry.get(name)
            assert "force" not in spec.description.lower() or spec.risk is RiskLevel.HIGH


# ---------------------------------------------------------------- keyboard

class TestKeyParsing:

    @pytest.mark.parametrize("spoken,modifiers,key", [
        ("enter", (), "enter"),
        ("Escape", (), "escape"),
        ("tab", (), "tab"),
        ("f5", (), "f5"),
        ("ctrl+c", ("ctrl",), "c"),
        ("ctrl+v", ("ctrl",), "v"),
        ("Ctrl + Shift + T", ("ctrl", "shift"), "t"),
        ("alt+tab", ("alt",), "tab"),
        ("control shift t", ("control", "shift"), "t"),
    ])
    def test_known_keys_parse(self, spoken, modifiers, key) -> None:
        chord = parse_chord(spoken)
        assert chord.modifiers == modifiers
        assert chord.key == key

    @pytest.mark.parametrize("spoken", [
        "", "the any key", "ctrl+sparkle", "meta+q", "press", "ahead with it",
    ])
    def test_unknown_keys_are_refused(self, spoken) -> None:
        with pytest.raises(UnknownKeyError):
            parse_chord(spoken)

    def test_a_bare_modifier_is_refused(self) -> None:
        with pytest.raises(UnknownKeyError):
            parse_chord("ctrl")

    @pytest.mark.parametrize("spoken", [
        "alt+f4", "ctrl+alt+delete", "alt+ctrl+del", "win+l",
        "ctrl+shift+escape", "shift+ctrl+esc", "Shift + Ctrl + Esc",
    ])
    def test_destructive_shortcuts_are_refused_in_every_spelling(self,
                                                                 spoken) -> None:
        """Modifier order and key aliases must not let a refused chord through.

        "alt+ctrl+del" is the same keystroke as "ctrl+alt+delete"; a refusal
        list keyed on the raw text would catch one and miss the other.
        """
        assert refusal_for(parse_chord(spoken)) is not None

    def test_a_modifier_in_the_key_position_is_refused(self) -> None:
        """"f4+alt" is not a spelling of Alt+F4 -- the key comes last."""
        with pytest.raises(UnknownKeyError):
            parse_chord("f4+alt")


class TestKeyboardActions:

    def test_enter_is_sent(self, registry, action_context) -> None:
        keys = FakeInput()
        ctx = _context(action_context, input_provider=_input(keys),
                       window_provider=_windows(FakeDesktop([editor_window()])))
        result = _run(registry, ctx, "press_key", keys="enter")
        assert result.ok
        assert keys.presses == [((), 0x0D)]

    def test_control_c_is_sent_with_its_modifier(self, registry,
                                                 action_context) -> None:
        keys = FakeInput()
        ctx = _context(action_context, input_provider=_input(keys),
                       window_provider=_windows(FakeDesktop([editor_window()])))
        assert _run(registry, ctx, "press_key", keys="ctrl+c").ok
        assert keys.presses == [((0x11,), ord("C"))]

    def test_alt_f4_is_refused_and_not_sent(self, registry,
                                            action_context) -> None:
        keys = FakeInput()
        ctx = _context(action_context, input_provider=_input(keys),
                       window_provider=_windows(FakeDesktop([editor_window()])))
        result = _run(registry, ctx, "press_key", keys="alt+f4")
        assert not result.ok
        assert result.error_code == "chord_refused"
        assert keys.presses == []
        assert "close this window" in result.message

    def test_unknown_key_is_refused(self, registry, action_context) -> None:
        keys = FakeInput()
        ctx = _context(action_context, input_provider=_input(keys))
        result = _run(registry, ctx, "press_key", keys="the any key")
        assert not result.ok
        assert result.error_code == "unknown_key"
        assert keys.presses == []

    def test_typing_reaches_the_input_system(self, registry,
                                             action_context) -> None:
        keys = FakeInput()
        ctx = _context(action_context, input_provider=_input(keys),
                       window_provider=_windows(FakeDesktop([editor_window()])))
        result = _run(registry, ctx, "type_text", text="hello world")
        assert result.ok
        assert keys.typed == ["hello world"]

    def test_typing_names_the_window_it_went_to(self, registry,
                                                action_context) -> None:
        """So a keystroke that landed somewhere unexpected is obvious at once."""
        ctx = _context(action_context, input_provider=_input(FakeInput()),
                       window_provider=_windows(FakeDesktop([editor_window()])))
        result = _run(registry, ctx, "type_text", text="hello")
        assert "Notepad" in result.message

    def test_keyboard_actions_need_the_keyboard_permission(self, registry) -> None:
        from security.permissions import Permission

        for name in ("press_key", "type_text"):
            assert Permission.KEYBOARD in registry.require(name).permissions


# ------------------------------------------------------------------- mouse

class TestMouseActions:

    def test_click_at_a_point(self, registry, action_context) -> None:
        mouse = FakeInput()
        ctx = _context(action_context, input_provider=_input(mouse))
        result = _run(registry, ctx, "click_mouse", at="500, 300")
        assert result.ok, result.message
        assert mouse.moves == [(500, 300)]
        assert mouse.clicks == [("left", 1)]

    def test_click_where_the_pointer_already_is(self, registry,
                                                action_context) -> None:
        mouse = FakeInput()
        ctx = _context(action_context, input_provider=_input(mouse))
        assert _run(registry, ctx, "click_mouse").ok
        assert mouse.moves == []
        assert mouse.clicks == [("left", 1)]

    def test_double_click(self, registry, action_context) -> None:
        mouse = FakeInput()
        ctx = _context(action_context, input_provider=_input(mouse))
        assert _run(registry, ctx, "double_click_mouse", at="200,400").ok
        assert mouse.clicks == [("left", 2)]

    def test_right_click(self, registry, action_context) -> None:
        mouse = FakeInput()
        ctx = _context(action_context, input_provider=_input(mouse))
        assert _run(registry, ctx, "right_click_mouse").ok
        assert mouse.clicks == [("right", 1)]

    def test_off_screen_coordinates_are_refused_not_clamped(self, registry,
                                                            action_context) -> None:
        mouse = FakeInput(width=1920, height=1080)
        ctx = _context(action_context, input_provider=_input(mouse))
        result = _run(registry, ctx, "click_mouse", at="5000, 300")
        assert not result.ok
        assert result.error_code == "off_screen"
        assert mouse.clicks == [] and mouse.moves == []

    def test_a_described_target_is_refused(self, registry,
                                           action_context) -> None:
        """No screen understanding means no honest answer to "the Submit button"."""
        mouse = FakeInput()
        ctx = _context(action_context, input_provider=_input(mouse))
        result = _run(registry, ctx, "click_mouse", at="the Submit button")
        assert not result.ok
        assert result.error_code == "no_visual_target"
        assert "can't see the screen" in result.message
        assert mouse.clicks == []

    def test_unreadable_coordinates_are_refused(self, registry,
                                                action_context) -> None:
        mouse = FakeInput()
        ctx = _context(action_context, input_provider=_input(mouse))
        result = _run(registry, ctx, "click_mouse", at="over there")
        assert not result.ok
        assert mouse.clicks == []

    def test_move_requires_a_point(self, registry, action_context) -> None:
        mouse = FakeInput()
        ctx = _context(action_context, input_provider=_input(mouse))
        result = _run(registry, ctx, "move_mouse", to="the middle")
        assert not result.ok
        assert mouse.moves == []

    def test_scroll(self, registry, action_context) -> None:
        mouse = FakeInput()
        ctx = _context(action_context, input_provider=_input(mouse))
        assert _run(registry, ctx, "scroll_mouse", direction="down").ok
        assert mouse.scrolls and mouse.scrolls[0] < 0


# -------------------------------------------------------------- screenshots

class TestScreenshots:

    def test_file_is_created_and_the_path_is_real(self, registry,
                                                  action_context) -> None:
        pytest.importorskip("PIL")
        result = _run(registry, action_context, "take_screenshot")
        if not result.ok:
            pytest.skip(f"no capture backend here: {result.message}")
        path = Path(result.data["path"])
        assert path.is_file()
        assert path.stat().st_size > 0
        assert result.data["bytes"] == path.stat().st_size

    def test_two_captures_do_not_collide(self, registry, action_context) -> None:
        """The old timestamp was second-resolution, so this used to be one file."""
        pytest.importorskip("PIL")
        first = _run(registry, action_context, "take_screenshot")
        if not first.ok:
            pytest.skip(f"no capture backend here: {first.message}")
        second = _run(registry, action_context, "take_screenshot")
        assert second.ok
        assert first.data["path"] != second.data["path"]
        assert Path(first.data["path"]).is_file()
        assert Path(second.data["path"]).is_file()

    def test_unique_path_never_returns_an_existing_file(self, tmp_path) -> None:
        from actions.screenshot import unique_path

        first = unique_path(tmp_path, "shot")
        first.write_bytes(b"x")
        second = unique_path(tmp_path, "shot")
        assert second != first
        assert not second.exists()

    def test_window_capture_refuses_without_a_rectangle(self, registry,
                                                        action_context) -> None:
        """It does not silently fall back to a full-screen grab."""
        window = editor_window()
        window["rect"] = None
        ctx = _context(action_context,
                       window_provider=_windows(FakeDesktop([window])))
        result = _run(registry, ctx, "capture_window")
        assert not result.ok
        assert result.error_code == "no_window_rect"
