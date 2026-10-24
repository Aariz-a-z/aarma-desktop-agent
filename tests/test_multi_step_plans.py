"""Sentences that ask for more than one thing.

The unit under test is not really "can AARMA do two things" -- it is *what it
says it did*. A multi-step instruction has three ways to end badly and only one
of them looks like a failure:

* it runs both halves (fine),
* it runs neither (obvious, the user retries),
* it runs one half and reports success (invisible, and the user stops
  checking).

Most of what follows pins down the third case.
"""

from __future__ import annotations

import pytest

from ai.nlu import plans


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

class TestSplitting:
    @pytest.mark.parametrize(("sentence", "expected"), [
        ("open downloads and take a screenshot",
         ["open downloads", "take a screenshot"]),
        ("open downloads then take a screenshot",
         ["open downloads", "take a screenshot"]),
        ("open downloads, then take a screenshot",
         ["open downloads", "take a screenshot"]),
        ("open downloads and then take a screenshot",
         ["open downloads", "take a screenshot"]),
    ])
    def test_connectives_cut_a_sentence_in_two(self, sentence: str,
                                               expected: list) -> None:
        assert plans.split_sentence(sentence) == expected

    def test_a_sentence_without_a_connective_is_one_fragment(self) -> None:
        assert plans.split_sentence("take a screenshot") == ["take a screenshot"]

    def test_empty_input_produces_no_fragments(self) -> None:
        assert plans.split_sentence("   ") == []

    def test_a_one_word_fragment_survives_the_split(self) -> None:
        """"mute" is a real command, so length alone must not discard it.

        What stops "open chrome and github" from becoming two steps is that
        "github" does not resolve -- see :class:`TestBuilding` -- not that it
        is short. Rejecting by length threw away real one-word commands too.
        """
        assert plans.split_sentence("mute and take a screenshot") == [
            "mute", "take a screenshot"]

    def test_splitting_stops_at_the_step_ceiling(self) -> None:
        sentence = " and ".join(["open downloads"] * 6)
        assert len(plans.split_sentence(sentence)) == plans.MAX_STEPS


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------

class _Resolution:
    def __init__(self, action="", parameters=None, confidence=0.0, refused=""):
        self.resolved = bool(action)
        self.action = action
        self.parameters = parameters or {}
        self.confidence = confidence
        self.refused = refused


class _FakeResolver:
    """Resolves the fragments it was given a mapping for, and nothing else."""

    def __init__(self, known):
        self.known = known

    def resolve(self, fragment, context=None):
        entry = self.known.get(fragment)
        if entry is None:
            return _Resolution(refused="no capability")
        action, parameters = entry
        return _Resolution(action=action, parameters=parameters, confidence=0.9)


class TestBuilding:
    def test_both_halves_become_steps(self) -> None:
        resolver = _FakeResolver({
            "open downloads": ("open_folder", {"location": "downloads"}),
            "take a screenshot": ("take_screenshot", {}),
        })
        plan = plans.build("open downloads and take a screenshot", resolver)
        assert [s.action for s in plan.steps] == ["open_folder", "take_screenshot"]
        assert plan.is_multi
        assert plan.unmet == []

    def test_a_sentence_whose_first_half_fails_is_not_a_plan(self) -> None:
        """Nothing resolved yet, so this is one instruction containing "and"."""
        resolver = _FakeResolver({"take a screenshot": ("take_screenshot", {})})
        assert plans.build("buy milk and take a screenshot", resolver) is None

    def test_an_unresolvable_tail_is_recorded_rather_than_dropped(self) -> None:
        """The bug this exists for: success reported for half a request.

        "Open Downloads and open a nonexistent folder" used to open Downloads
        and answer "Opening Downloads." -- true, complete-sounding, and wrong.
        """
        resolver = _FakeResolver({
            "open downloads": ("open_folder", {"location": "downloads"})})
        plan = plans.build("open downloads and open a nonexistent folder", resolver)
        assert [s.action for s in plan.steps] == ["open_folder"]
        assert plan.unmet == ["open a nonexistent folder"]

    def test_a_gap_in_the_middle_is_rejected(self) -> None:
        """A later step may have depended on the one that went missing."""
        resolver = _FakeResolver({
            "open downloads": ("open_folder", {"location": "downloads"}),
            "take a screenshot": ("take_screenshot", {}),
        })
        assert plans.build(
            "open downloads and buy milk and take a screenshot", resolver) is None

    def test_two_identical_actions_are_a_bad_split(self) -> None:
        resolver = _FakeResolver({
            "close chrome": ("close_application", {"name": "chrome"}),
            "close firefox": ("close_application", {"name": "firefox"}),
        })
        assert plans.build("close chrome and close firefox", resolver) is None

    def test_a_single_instruction_is_not_a_plan(self) -> None:
        resolver = _FakeResolver({"take a screenshot": ("take_screenshot", {})})
        assert plans.build("take a screenshot", resolver) is None


# --------------------------------------------------------------------------
# Carrying context between steps
# --------------------------------------------------------------------------

class _Spec:
    def __init__(self, *names):
        self.parameters = tuple(type("P", (), {"name": n})() for n in names)


class _Registry:
    def __init__(self, specs):
        self.specs = specs

    def get(self, action):
        return self.specs.get(action)


class TestCarryingContext:
    def test_a_later_step_inherits_the_target_named_once(self) -> None:
        plan = plans.ActionPlan(steps=[
            plans.PlannedStep("open_repository_in_editor", {"name": "AARMA"}),
            plans.PlannedStep("open_terminal_in_repository", {}),
        ])
        registry = _Registry({
            "open_repository_in_editor": _Spec("name"),
            "open_terminal_in_repository": _Spec("name"),
        })
        plans.carry_context(plan, registry)
        assert plan.steps[1].parameters["name"] == "AARMA"

    def test_nothing_is_carried_into_a_step_that_does_not_accept_it(self) -> None:
        """Otherwise the step is rejected at execution time, after step one ran."""
        plan = plans.ActionPlan(steps=[
            plans.PlannedStep("open_folder", {"location": "downloads"}),
            plans.PlannedStep("take_screenshot", {}),
        ])
        registry = _Registry({
            "open_folder": _Spec("location"),
            "take_screenshot": _Spec(),
        })
        plans.carry_context(plan, registry)
        assert "location" not in plan.steps[1].parameters

    def test_a_step_that_names_its_own_target_keeps_it(self) -> None:
        plan = plans.ActionPlan(steps=[
            plans.PlannedStep("open_repository", {"name": "AARMA"}),
            plans.PlannedStep("open_repository_in_editor", {"name": "shopeasy"}),
        ])
        registry = _Registry({
            "open_repository": _Spec("name"),
            "open_repository_in_editor": _Spec("name"),
        })
        plans.carry_context(plan, registry)
        assert plan.steps[1].parameters["name"] == "shopeasy"


# --------------------------------------------------------------------------
# What the engine says it did
# --------------------------------------------------------------------------

from core.engine import AarmaEngine, ResponseKind  # noqa: E402


class TestTheEngineReportsPlansHonestly:
    """Every assertion here is about the *wording*, not the side effect.

    A plan that half-ran and said so is a good outcome. A plan that half-ran
    and said "Done" is the failure this module guards against, so these tests
    fail if the reply stops naming what was skipped.

    The actions used are the ones the autouse conftest guard permits. "mute"
    deliberately is not: under the guard the media keys are unreachable, which
    makes it a dependable way to fail a first step on purpose.
    """

    def test_both_halves_are_reported(self, engine: AarmaEngine) -> None:
        response = engine.handle_text("what time is it and what's the date")
        assert response.ok, response.text
        assert "current time" in response.text.lower()
        assert "today is" in response.text.lower()

    def test_a_failed_step_stops_the_plan_and_names_what_was_skipped(
            self, engine: AarmaEngine) -> None:
        """The case that must never read as success.

        The first step cannot reach the media keys. The screenshot after it is
        therefore never taken, and the reply has to say so -- otherwise the
        user is told about a failure and left to assume the rest went through.
        """
        response = engine.handle_text("mute and take a screenshot")
        assert not response.ok
        assert "haven't done" in response.text.lower()
        assert "screenshot" in response.text.lower()

    def test_an_unmet_half_is_named_and_the_answer_is_not_ok(
            self, engine: AarmaEngine) -> None:
        response = engine.handle_text(
            "what time is it and open a folder that does not exist anywhere")
        assert "time" in response.text.lower()
        assert not response.ok, "half a request must not report success"
        assert "didn't understand" in response.text.lower()

    def test_a_gated_step_never_slips_through_inside_a_plan(
            self, engine: AarmaEngine) -> None:
        """Being step two of a plan must not soften a safety gate.

        On the default profile a shutdown is refused for want of the
        SYSTEM_CONTROL permission, which is checked before confirmation is even
        offered; granting it would instead park the action for confirmation.
        Either way the invariant is the same and it is the one worth pinning:
        the plan does not report success, and the step that *did* run is still
        named. Power actions are stubbed by the autouse conftest guard, so no
        real shutdown is reachable from here.
        """
        response = engine.handle_text(
            "what time is it and shut down the computer")
        assert response.kind in (ResponseKind.ERROR,
                                 ResponseKind.CONFIRMATION_REQUIRED)
        assert "current time" in response.text.lower(),             "the step that ran must still be reported"
        if response.kind is ResponseKind.CONFIRMATION_REQUIRED:
            assert response.confirmation_token
        else:
            assert not response.ok

    def test_a_single_instruction_is_unaffected(self, engine: AarmaEngine) -> None:
        response = engine.handle_text("what time is it")
        assert response.ok
        assert "didn't understand" not in response.text.lower()

    def test_a_reminder_keeps_its_connective(self, engine: AarmaEngine) -> None:
        """"call mom and email dad" is one reminder, not two actions."""
        response = engine.handle_text(
            "remind me in 2 hours to call mom and email dad")
        assert "call mom and email dad" in response.text.lower()
