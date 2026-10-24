"""Pointing at a target instead of naming it.

"Open my AARMA repo." / "Now open it in VS Code." The second sentence names no
project. Resolving it needs memory of the first, and the whole risk of that
memory is acting on a target the user is no longer thinking about -- so the
tests here are as much about when a reference must *fail* as when it resolves.
"""

from __future__ import annotations

import pytest

from core.engine import AarmaEngine
from security.permissions import Permission


class TestTheEngineRemembersOneTarget:
    def test_nothing_is_remembered_before_anything_runs(
            self, engine: AarmaEngine) -> None:
        assert engine._last_action == ""
        assert engine._last_target == {}

    def test_a_successful_action_leaves_its_target_behind(
            self, engine: AarmaEngine) -> None:
        engine.handle_text("what time is it")
        assert engine._last_action == "get_time"

    def test_only_target_parameters_are_kept(self, engine: AarmaEngine) -> None:
        """A body of text or a flag is not something the next sentence points at."""
        engine._remember_target("draft_email", {
            "to": "someone@example.com", "body": "a long message",
            "name": "AARMA", "confirm": True})
        assert engine._last_target == {"name": "AARMA"}

    def test_a_blank_parameter_is_not_a_target(self, engine: AarmaEngine) -> None:
        engine._remember_target("open_repository", {"name": ""})
        assert engine._last_target == {}

    def test_the_context_carries_the_memory_to_the_resolver(
            self, engine: AarmaEngine) -> None:
        engine._remember_target("open_repository", {"name": "AARMA"})
        context = engine._resolution_context()
        assert context.last_action == "open_repository"
        assert context.last_entities == {"name": "AARMA"}

    def test_the_memory_is_one_turn_deep(self, engine: AarmaEngine) -> None:
        """A longer history is how "it" ends up meaning something stale."""
        engine._remember_target("open_repository", {"name": "first"})
        engine._remember_target("open_repository", {"name": "second"})
        assert engine._last_target == {"name": "second"}


class TestAPointerWithNothingToPointAt:
    """The half that matters: a reference with no referent must not act."""

    def test_a_terminal_somewhere_asks_which_project(
            self, engine: AarmaEngine) -> None:
        """Granted the permission, so the refusal under test is the real one.

        Without it the permission gate answers first, which would let this pass
        for the wrong reason even if the handler went back to picking a project
        off disk.
        """
        engine.permissions.grant(Permission.SYSTEM_CONTROL)
        response = engine.handle_text("open a terminal there")
        assert not response.ok
        assert "which project" in response.text.lower()

    def test_an_editor_reference_does_not_open_the_editor_instead(
            self, engine: AarmaEngine) -> None:
        """"Open it in VS Code" must not degrade into "open VS Code".

        The editor is the container, not the target. Opening the container and
        reporting success would look like the request was carried out.
        """
        response = engine.handle_text("open it in VS Code")
        assert not response.ok
        assert "visual studio code" not in response.text.lower()
