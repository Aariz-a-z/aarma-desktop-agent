"""The action registry: registration, lookup, refusal and parameter validation.

These are the tests that hold the central security claim in place -- that an
action which was never registered cannot run, and that a parameter which was
never declared cannot reach a handler.
"""

from __future__ import annotations

import pytest

from actions.registry import (
    ActionRegistry,
    ActionResult,
    ActionSpec,
    ParameterSpec,
    ParameterType,
    UnknownActionError,
    ValidationFailed,
    build_default_registry,
)
from security.permissions import Permission
from security.validator import RiskLevel


def _noop(ctx, params) -> ActionResult:
    return ActionResult.success("ok")


def _spec(name: str = "demo", **kwargs) -> ActionSpec:
    defaults = dict(description="A demo action.", category="Test",
                    risk=RiskLevel.LOW, handler=_noop)
    defaults.update(kwargs)
    return ActionSpec(name=name, **defaults)  # type: ignore[arg-type]


class TestRegistration:
    def test_registers_and_finds_an_action(self) -> None:
        registry = ActionRegistry()
        registry.register(_spec("demo"))
        assert "demo" in registry
        assert registry.require("demo").name == "demo"

    def test_duplicate_names_are_rejected(self) -> None:
        registry = ActionRegistry()
        registry.register(_spec("demo"))
        with pytest.raises(ValueError):
            registry.register(_spec("demo"))

    def test_invalid_action_names_are_rejected(self) -> None:
        registry = ActionRegistry()
        with pytest.raises(ValueError):
            registry.register(_spec("rm -rf /"))


class TestLookup:
    def test_unknown_action_raises(self) -> None:
        registry = ActionRegistry()
        with pytest.raises(UnknownActionError):
            registry.require("run_shell_command")

    def test_get_returns_none_for_unknown(self) -> None:
        assert ActionRegistry().get("definitely_not_registered") is None

    def test_a_model_style_hallucinated_name_is_refused(self) -> None:
        """The Phase 2 threat: a model proposing a tool that does not exist."""
        registry = build_default_registry()
        for invented in ("execute_shell", "run_powershell", "eval_python", "sudo"):
            assert invented not in registry
            with pytest.raises(UnknownActionError):
                registry.require(invented)


class TestParameterValidation:
    def test_missing_required_parameter_is_refused(self) -> None:
        spec = _spec(parameters=(ParameterSpec(name="name", description="a name"),))
        with pytest.raises(ValidationFailed):
            spec.validate({})

    def test_undeclared_parameters_are_refused(self) -> None:
        spec = _spec(parameters=(ParameterSpec(name="name", description="a name"),))
        with pytest.raises(ValidationFailed) as excinfo:
            spec.validate({"name": "ok", "shell": "cmd.exe"})
        assert "shell" in str(excinfo.value)

    def test_optional_parameter_falls_back_to_its_default(self) -> None:
        spec = _spec(parameters=(
            ParameterSpec(name="scope", description="scope", required=False, default="all"),
        ))
        assert spec.validate({}) == {"scope": "all"}

    def test_choices_are_enforced(self) -> None:
        spec = _spec(parameters=(
            ParameterSpec(name="mode", description="mode", choices=("fast", "slow")),
        ))
        assert spec.validate({"mode": "fast"})["mode"] == "fast"
        with pytest.raises(ValidationFailed):
            spec.validate({"mode": "reckless"})

    def test_length_limits_are_enforced(self) -> None:
        spec = _spec(parameters=(
            ParameterSpec(name="text", description="text", max_length=10),
        ))
        with pytest.raises(ValidationFailed):
            spec.validate({"text": "x" * 11})

    def test_control_characters_are_refused(self) -> None:
        spec = _spec(parameters=(ParameterSpec(name="text", description="text"),))
        with pytest.raises(ValidationFailed):
            spec.validate({"text": "hello\x00world"})

    def test_integers_are_coerced_and_bounded(self) -> None:
        spec = _spec(parameters=(
            ParameterSpec(name="count", type=ParameterType.INTEGER, description="count",
                          min_value=1, max_value=5),
        ))
        assert spec.validate({"count": "3"})["count"] == 3
        with pytest.raises(ValidationFailed):
            spec.validate({"count": "9"})
        with pytest.raises(ValidationFailed):
            spec.validate({"count": "not a number"})

    def test_booleans_accept_natural_words(self) -> None:
        spec = _spec(parameters=(
            ParameterSpec(name="flag", type=ParameterType.BOOLEAN, description="flag"),
        ))
        assert spec.validate({"flag": "yes"})["flag"] is True
        assert spec.validate({"flag": "off"})["flag"] is False
        with pytest.raises(ValidationFailed):
            spec.validate({"flag": "maybe"})

    def test_url_parameters_reject_dangerous_schemes(self) -> None:
        spec = _spec(parameters=(
            ParameterSpec(name="url", type=ParameterType.URL, description="a url"),
        ))
        assert spec.validate({"url": "example.com"})["url"].startswith("https://")
        for hostile in ("file:///C:/Windows/System32/cmd.exe",
                        "javascript:alert(1)",
                        "data:text/html,<script>"):
            with pytest.raises(ValidationFailed):
                spec.validate({"url": hostile})


class TestDefaultRegistry:
    def test_builds_without_error_and_is_not_empty(self, registry: ActionRegistry) -> None:
        assert len(registry) > 0

    def test_every_action_declares_a_risk_and_a_handler(self, registry: ActionRegistry) -> None:
        for spec in registry.all():
            assert isinstance(spec.risk, RiskLevel)
            assert callable(spec.handler)
            assert spec.description

    def test_phase_one_foundation_actions_are_present(self, registry: ActionRegistry) -> None:
        for name in ("get_time", "get_date", "open_application", "open_url",
                     "search_web", "create_folder", "list_files",
                     "set_assistant_name", "set_language"):
            assert name in registry

    def test_shell_execution_has_exactly_one_guarded_door(
        self, registry: ActionRegistry
    ) -> None:
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
        executors = [spec for spec in registry.all()
                     if any(word in spec.name.lower()
                            for word in ("shell", "exec", "eval", "script"))]
        assert [spec.name for spec in executors] == ["execute_shell_command"]

        spec = executors[0]
        assert spec.risk is RiskLevel.HIGH
        assert Permission.SYSTEM_CONTROL in spec.permissions

    def test_nothing_else_smuggles_a_command_parameter(
        self, registry: ActionRegistry
    ) -> None:
        """No other action takes a free-text command to run.

        The one door only helps if there is not a window beside it: an action
        that quietly accepted a ``command`` argument would be a second route
        with none of the grading.
        """
        for spec in registry.all():
            if spec.name == "execute_shell_command":
                continue
            for parameter in spec.parameters:
                assert parameter.name not in ("command", "cmd", "script",
                                              "shell", "argv")

    def test_keyboard_and_mouse_are_implemented_and_gated(self) -> None:
        """Synthetic input exists now, and carries the guards it was waiting for.

        This test used to assert that both modules registered nothing.  The
        reason given was that keystrokes go to whatever has focus, which is how
        an assistant types a password into a chat window -- and that reason has
        not gone away.  What changed is that the guards it was waiting for now
        exist, so the assertion moved from "absent" to "present and gated":
        every one of these needs a permission that is off by default.
        """
        from actions import keyboard, mouse

        keyboard_names = {spec.name for spec in keyboard.get_actions()}
        mouse_names = {spec.name for spec in mouse.get_actions()}
        assert {"press_key", "type_text"} <= keyboard_names
        assert {"click_mouse", "double_click_mouse", "move_mouse"} <= mouse_names

        for spec in keyboard.get_actions():
            assert Permission.KEYBOARD in spec.permissions
        for spec in mouse.get_actions():
            assert Permission.MOUSE in spec.permissions

    def test_synthetic_input_permissions_are_off_by_default(self) -> None:
        """The capability does not exist until the user switches it on."""
        from security.permissions import PERMISSION_CATALOG

        for permission in (Permission.KEYBOARD, Permission.MOUSE):
            assert PERMISSION_CATALOG[permission].default_granted is False

    def test_destructive_shortcuts_cannot_be_synthesised(self) -> None:
        """Alt+F4 and friends are refused, not sent.

        Otherwise a key name would be a way around the confirmation policy:
        "press alt+f4" would close a window that ``close_window`` would have
        identified and asked about first.
        """
        from providers.input import parse_chord, refusal_for

        for chord in ("alt+f4", "ctrl+alt+delete", "win+l"):
            assert refusal_for(parse_chord(chord)) is not None

    def test_reminders_are_implemented(self) -> None:
        from actions import reminders

        names = {spec.name for spec in reminders.get_actions()}
        assert {"create_reminder", "list_reminders", "cancel_reminder"} <= names

    def test_destructive_actions_are_graded_high(self, registry: ActionRegistry) -> None:
        assert registry.require("delete_item").risk is RiskLevel.HIGH

    def test_write_actions_are_graded_at_least_medium(self, registry: ActionRegistry) -> None:
        for name in ("create_folder", "rename_item", "move_item", "close_application"):
            assert registry.require(name).risk >= RiskLevel.MEDIUM

    def test_filesystem_actions_require_the_filesystem_permission(
        self, registry: ActionRegistry
    ) -> None:
        for name in ("create_folder", "list_files", "delete_item", "move_item"):
            assert Permission.FILESYSTEM in registry.require(name).permissions

    def test_online_actions_are_marked_as_needing_internet(
        self, registry: ActionRegistry
    ) -> None:
        assert registry.require("search_web").requires_internet
        assert not registry.require("get_time").requires_internet
