"""Power safety: proving a test can no longer switch this machine off.

This file exists because of an incident, and it is worth recording plainly.  A
mutation test in ``test_phase3_conversation.py`` disabled
``RiskPolicy.requires_confirmation`` -- a reasonable thing for a mutation test
to do -- and then sent "shut down the computer" through a live engine.  The
engine builds its own :class:`~actions.registry.ActionContext`, which carries
no injected ``process_runner``, so :func:`actions.power._runner` fell through to
the real :func:`subprocess.run` and issued ``shutdown /s /t 20``.  The assertion
passed.  The laptop powered off twenty seconds later, twice.

The lesson is not "that test was careless".  It is that per-test discipline is
a *convention*, and the next test always breaks the convention.  So the suite
now has three independent barriers, and this file tests all three separately,
because barriers that are only ever tested together can rot one at a time:

1. **Permission.**  Every power action requires ``SYSTEM_CONTROL``, which is
   off by default.
2. **Confirmation.**  Shutdown, restart and sign-out are HIGH risk, and the
   confirmation floor is not user-adjustable.
3. **A fail-closed runner.**  Under a test runner with no injected fake,
   :func:`actions.power._runner` refuses instead of returning ``subprocess.run``
   -- in production code, so it holds even if a conftest is bypassed -- and
   :func:`tests.conftest.block_real_os_side_effects` severs the same route
   again from outside.

Nothing in this file runs a power command.  Several tests assert that reaching
one *raises*, which is the only way to prove a guard is load-bearing without
relying on it.
"""

from __future__ import annotations

import subprocess
import types
from typing import Any, Dict, List

import pytest

import actions.power as power
from actions.registry import ActionContext, ActionRequest, build_default_registry
from core.engine import ResponseKind
from security.permissions import Permission
from security.validator import RiskLevel, RiskPolicy

# Imported as ``conftest``, not ``tests.conftest``.  pytest loads the conftest
# as a top-level module, and importing it by the dotted path would create a
# *second* module object with a second, unrelated ``RealOSCallBlocked`` class --
# so ``pytest.raises`` would not catch the one the fixture actually raises.
from conftest import RealOSCallBlocked  # noqa: E402

#: Everything that reaches the power subsystem.  Used for the checks that must
#: hold whatever an action is graded -- the fail-closed runner, and the "nothing
#: triggers this automatically" family.
POWER_ACTIONS = ("shutdown_computer", "restart_computer", "sign_out",
                 "sleep_computer", "lock_screen")

#: Actions behind ``SYSTEM_CONTROL``, which is off by default.  Each of these
#: can cost the user something: work in progress, or the session itself.
PERMISSION_GATED = ("shutdown_computer", "restart_computer", "sign_out",
                    "sleep_computer")

#: Deliberately ungated, and each for a stated reason.
#:
#: ``lock_screen`` interrupts nothing and loses nothing; a password undoes it.
#: Gating it would make "lock my PC" fail on a fresh install, which teaches the
#: user that AARMA refuses simple things.
#:
#: ``cancel_shutdown`` only ever *prevents*. The escape hatch must never be
#: harder to reach than the hazard.
UNGATED = ("lock_screen", "cancel_shutdown")

#: The subset that must never run on a single spoken sentence.
DESTRUCTIVE = ("shutdown_computer", "restart_computer", "sign_out")


class _RecordingRunner:
    """A runner that records argv and starts nothing."""

    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    def __call__(self, command, **_kwargs):
        self.calls.append(list(command))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")


@pytest.fixture
def registry():
    return build_default_registry()


def _context(action_context: ActionContext, **overrides) -> ActionContext:
    for key, value in overrides.items():
        object.__setattr__(action_context, key, value)
    return action_context


# ------------------------------------------------------- barrier 1: permission

class TestPermissionBarrier:
    """Off by default, so a fresh install cannot power *down* at all.

    The line is drawn at loss, not at "touches the power subsystem".  Shutting
    down, restarting, signing out and sleeping all end the session and can take
    unsaved work with them.  Locking and cancelling cannot cost the user
    anything, so they sit on the other side of it.
    """

    @pytest.mark.parametrize("name", PERMISSION_GATED)
    def test_actions_that_can_lose_work_require_system_control(self, registry,
                                                               name) -> None:
        assert Permission.SYSTEM_CONTROL in registry.require(name).permissions

    @pytest.mark.parametrize("name", UNGATED)
    def test_actions_that_cannot_lose_anything_are_not_gated(self, registry,
                                                             name) -> None:
        assert registry.require(name).permissions == ()

    def test_system_control_is_denied_by_default(self, permissions) -> None:
        assert permissions.is_granted(Permission.SYSTEM_CONTROL) is False

    @pytest.mark.parametrize("utterance", [
        "shut down the computer", "restart the computer", "sign me out",
        "put the computer to sleep",
    ])
    def test_without_the_permission_nothing_even_reaches_confirmation(
            self, engine, utterance) -> None:
        response = engine.handle_text(utterance)
        assert response.kind is ResponseKind.ERROR
        assert response.kind is not ResponseKind.CONFIRMATION_REQUIRED
        assert "permission" in response.text.lower() or "allow" in response.text.lower()

    def test_cancelling_a_shutdown_needs_no_permission(self, registry) -> None:
        """The escape hatch must never be harder to reach than the hazard."""
        assert registry.require("cancel_shutdown").permissions == ()


class TestLockWorksOutOfTheBox:
    """"Lock my PC" must work on a fresh configuration, with no setup.

    This is the usability half of the safety design, and it is worth a class of
    its own because it is the property most likely to be broken by a
    well-meaning tightening pass.  A locked screen costs nothing and a password
    undoes it; refusing it on a default install teaches the user that AARMA
    says no to simple things, and a user who expects refusals stops reading
    them -- which is exactly how a confirmation prompt stops protecting
    anything.
    """

    def test_lock_is_low_risk(self, registry) -> None:
        assert registry.require("lock_screen").risk is RiskLevel.LOW

    def test_lock_needs_no_permission(self, registry) -> None:
        assert registry.require("lock_screen").permissions == ()

    def test_lock_never_asks_for_confirmation(self, config, registry) -> None:
        policy = RiskPolicy.from_settings(config.config.security)
        assert policy.requires_confirmation(
            registry.require("lock_screen").risk) is False

    def test_lock_is_not_refused_on_a_default_configuration(self, engine) -> None:
        """No permission granted, nothing configured: it goes straight through.

        The assertion is that execution is *reached* -- the guard raises at the
        runner, which is the last step before the real ``rundll32`` call.
        Getting that far with a fresh config is the whole property: no
        permission check stopped it and no confirmation was demanded.
        """
        assert engine.permissions.is_granted(Permission.SYSTEM_CONTROL) is False

        with pytest.raises(RealOSCallBlocked):
            engine.handle_text("lock my PC")

    @pytest.mark.parametrize("utterance", [
        "lock my PC", "lock the computer", "lock my screen",
    ])
    def test_the_phrases_people_use_all_reach_it(self, engine, utterance) -> None:
        assert engine.intent_engine.parse(utterance).name == "lock_screen"

    def test_lock_issues_the_documented_command(self, registry,
                                                action_context) -> None:
        """With a fake runner: the right argv, and still nothing spawned."""
        runner = _RecordingRunner()
        ctx = _context(action_context, process_runner=runner,
                       platform_system="Windows")
        spec = registry.require("lock_screen")
        result = spec.handler(ctx, spec.validate({}))
        assert result.ok
        assert runner.calls == [["rundll32.exe", "user32.dll,LockWorkStation"]]

    def test_ungating_lock_did_not_ungate_anything_else(self, registry) -> None:
        """The blast radius of this exemption, asserted rather than assumed."""
        ungated = {name for name in registry.names()
                   if registry.require(name).category == "Power"
                   and not registry.require(name).permissions}
        assert ungated == set(UNGATED)


# ----------------------------------------------------- barrier 2: confirmation

class TestConfirmationBarrier:

    @pytest.mark.parametrize("name", DESTRUCTIVE)
    def test_destructive_power_actions_are_high_risk(self, registry, name) -> None:
        assert registry.require(name).risk is RiskLevel.HIGH

    def test_sleep_keeps_its_intended_protection(self, registry, config,
                                                 engine) -> None:
        """MEDIUM and permission-gated: disruptive, but nothing is lost.

        Sleep sits between the two groups on purpose. It ends the session's
        responsiveness, so a misheard command is genuinely disruptive and it
        stays behind ``SYSTEM_CONTROL`` -- but nothing is destroyed, so making
        the user confirm it as well would be a prompt with no decision in it.
        """
        spec = registry.require("sleep_computer")
        assert spec.risk is RiskLevel.MEDIUM
        assert Permission.SYSTEM_CONTROL in spec.permissions
        assert RiskPolicy.from_settings(
            config.config.security).requires_confirmation(spec.risk) is False

        # And on a default configuration it is refused outright.
        assert engine.permissions.is_granted(Permission.SYSTEM_CONTROL) is False
        response = engine.handle_text("put the computer to sleep")
        assert response.kind is ResponseKind.ERROR

    @pytest.mark.parametrize("name", DESTRUCTIVE)
    def test_high_risk_always_confirms_whatever_the_settings_say(self, config,
                                                                 registry,
                                                                 name) -> None:
        config.update_security(enabled=False, confirm_high_risk_actions=False)
        policy = RiskPolicy.from_settings(config.config.security)
        assert policy.requires_confirmation(registry.require(name).risk) is True

    @pytest.mark.parametrize("utterance,action", [
        ("shut down the computer", "shutdown_computer"),
        ("restart the computer", "restart_computer"),
        ("sign me out", "sign_out"),
    ])
    def test_both_barriers_hold_together(self, engine, utterance,
                                         action) -> None:
        """Permission *and* confirmation, in that order, for each of the three.

        Parametrised over all three rather than testing shutdown and trusting
        the rest: they are separate ``ActionSpec`` entries, and a spec is
        exactly the kind of thing that gets edited one line at a time.
        """
        denied = engine.handle_text(utterance)
        assert denied.kind is ResponseKind.ERROR
        assert denied.kind is not ResponseKind.CONFIRMATION_REQUIRED

        engine.permissions.grant(Permission.SYSTEM_CONTROL)
        asked = engine.handle_text(utterance)
        assert asked.kind is ResponseKind.CONFIRMATION_REQUIRED
        assert asked.action == action

    @pytest.mark.parametrize("utterance,action", [
        ("shut down the computer", "shutdown_computer"),
        ("restart the computer", "restart_computer"),
        ("sign me out", "sign_out"),
    ])
    def test_one_sentence_only_ever_asks(self, engine, utterance, action) -> None:
        """The full path, with the permission granted: it asks and stops.

        If it did anything else, the fail-closed runner would raise and this
        test would error rather than fail -- which is the outcome we want from
        a guard that is actually holding.
        """
        engine.permissions.grant(Permission.SYSTEM_CONTROL)
        response = engine.handle_text(utterance)
        assert response.kind is ResponseKind.CONFIRMATION_REQUIRED
        assert response.action == action
        assert response.confirmation_token

    def test_declining_runs_nothing(self, engine) -> None:
        engine.permissions.grant(Permission.SYSTEM_CONTROL)
        asked = engine.handle_text("shut down the computer")
        declined = engine.resolve_confirmation(asked.confirmation_token, False)
        assert declined.ok
        assert engine.confirmations.pending_count == 0

    def test_approval_is_the_only_route_to_execution(self, engine) -> None:
        """Approving *does* reach the runner -- which the guard then blocks.

        This is the test that proves the confirmation gate is real rather than
        decorative: everything before it returns without touching the runner,
        and the moment a human says yes, the call arrives.  In production that
        call is ``subprocess.run``; here it is a guard that refuses, so the
        assertion is that it was reached, not that it ran.
        """
        engine.permissions.grant(Permission.SYSTEM_CONTROL)
        asked = engine.handle_text("shut down the computer")
        assert asked.kind is ResponseKind.CONFIRMATION_REQUIRED

        with pytest.raises(RealOSCallBlocked):
            engine.resolve_confirmation(asked.confirmation_token, True)


# ----------------------------------------------------- barrier 3: fail closed

class TestFailClosedRunner:
    """The production guard, tested without the conftest fixture in the way."""

    def test_under_test_is_detected(self) -> None:
        assert power._under_test() is True

    @pytest.mark.parametrize("name", POWER_ACTIONS)
    def test_no_injected_runner_means_no_real_command(self, registry,
                                                      action_context,
                                                      monkeypatch, name) -> None:
        """With the conftest guard lifted, production still refuses.

        The fixture is undone here on purpose.  Two guards that are only ever
        tested together can rot one at a time, and this one is the one that
        survives a conftest being edited, replaced or bypassed.
        """
        monkeypatch.undo()
        spec = registry.require(name)
        assert getattr(action_context, "process_runner", None) is None

        with pytest.raises(power.UnsafePowerExecution):
            spec.handler(_context(action_context, platform_system="Windows"),
                         spec.validate({}))

    def test_the_refusal_is_not_swallowed_by_the_broad_except(self,
                                                              action_context,
                                                              monkeypatch) -> None:
        """``_execute`` catches Exception; the guard must travel past it.

        A guard that derived from ``Exception`` would be caught there and
        reported as a failed action -- and a failed action is a *passing* test
        for anyone asserting "it didn't shut down", which is exactly how this
        would go unnoticed a second time.
        """
        monkeypatch.undo()
        assert issubclass(power.UnsafePowerExecution, BaseException)
        assert not issubclass(power.UnsafePowerExecution, Exception)

        with pytest.raises(power.UnsafePowerExecution):
            power._execute(_context(action_context, platform_system="Windows"),
                           ["shutdown", "/s", "/t", "20"],
                           success="never", failure="never")

    def test_an_injected_runner_is_still_honoured(self, registry,
                                                 action_context) -> None:
        """Legitimate power tests keep working -- argv is asserted, nothing runs."""
        runner = _RecordingRunner()
        ctx = _context(action_context, process_runner=runner,
                       platform_system="Windows")
        spec = registry.require("shutdown_computer")
        result = spec.handler(ctx, spec.validate({}))
        assert result.ok
        assert runner.calls == [["shutdown", "/s", "/t", str(power.GRACE_SECONDS)]]

    def test_the_conftest_fixture_blocks_it_too(self, registry,
                                                action_context) -> None:
        """The outer lock, with the production guard's own path unused."""
        spec = registry.require("restart_computer")
        with pytest.raises((RealOSCallBlocked, power.UnsafePowerExecution)):
            spec.handler(_context(action_context, platform_system="Windows"),
                         spec.validate({}))


# ------------------------------------------------- no automatic route at all

class TestNothingTriggersPowerAutomatically:

    def test_the_scheduler_cannot_execute_any_action(self) -> None:
        """Read the module, not the behaviour: the capability must be absent.

        A behavioural test would only prove that the paths exercised today do
        not execute anything.  Asserting that the scheduler holds no reference
        to the registry, the engine or a subprocess proves it cannot, however
        it is called.
        """
        import inspect

        import core.scheduler as scheduler

        source = inspect.getsource(scheduler)
        for forbidden in ("subprocess", "os.system", "startfile", "registry",
                          "handle_text", "ActionRequest", "spec.handler"):
            assert forbidden not in source, (
                f"core.scheduler references {forbidden!r}; it must not be able "
                "to execute anything")

    def test_a_reminder_named_like_a_command_is_only_text(self, engine) -> None:
        """A reminder title is data. It is never parsed back into an action."""
        from datetime import timedelta

        from core.scheduler import ReminderNotifier, ReminderScheduler
        from utils.when import local_now

        engine.database.add_reminder(
            reminder_id="trap", title="shut down the computer",
            due_at=(local_now() - timedelta(minutes=1)).isoformat(timespec="seconds"))

        seen: List[str] = []
        scheduler = ReminderScheduler(
            engine.database,
            notifier=ReminderNotifier([lambda event: bool(seen.append(event.title)) or True]),
            clock=local_now)

        handled = scheduler.poll()
        assert [event.title for event in handled] == ["shut down the computer"]
        assert seen == ["shut down the computer"]
        # Reaching this line at all means no power command ran: the guards
        # raise BaseException, which no `except Exception` above would catch.

    def test_starting_the_engine_executes_nothing(self, config, database,
                                                  resolver) -> None:
        from core.engine import AarmaEngine

        engine = AarmaEngine(config, database=database, path_resolver=resolver,
                             session_id="startup-safety")
        try:
            engine.start()
            assert engine.reminders.running is False
            assert engine.confirmations.pending_count == 0
            actions_run = [row.action for row in engine.database.recent_actions(50)]
            assert not any(name in actions_run for name in POWER_ACTIONS)
        finally:
            engine.shutdown()

    def test_starting_the_reminder_scheduler_executes_nothing(self,
                                                              engine) -> None:
        delivered: List[Any] = []
        engine.start_reminders(lambda event: bool(delivered.append(event)) or True)
        try:
            assert engine.reminders.running is True
        finally:
            engine.stop_reminders()
        actions_run = [row.action for row in engine.database.recent_actions(50)]
        assert not any(name in actions_run for name in POWER_ACTIONS)

    @pytest.mark.parametrize("utterance", [
        "", "asdfghjkl", "do the thing", "make it go", "turn it up",
        "what can you do", "hello", "open notepad", "power",
    ])
    def test_no_utterance_falls_back_to_a_power_action(self, engine,
                                                       utterance) -> None:
        """There is no default action, so nothing can default to shutdown."""
        intent = engine.intent_engine.parse(utterance)
        assert intent.name not in POWER_ACTIONS

    def test_a_power_action_cannot_be_completed_across_turns(self) -> None:
        """Multi-turn filling is an allow-list, and power is not on it."""
        from core.pending import FILLABLE, PendingActionStore

        for name in POWER_ACTIONS:
            assert name not in FILLABLE

        store = PendingActionStore()
        assert store.open("shutdown_computer", {}, needs="when",
                          question="When?") is None


# --------------------------------------------- the smoke test, asked safely

class TestSmokeTestWithoutPoweringOff:
    """What the live smoke test is allowed to do instead.

    The brief asks for "what would happen if I asked you to shut down?" rather
    than the real thing.  Both forms are covered here so the answer is known
    without anyone typing the dangerous sentence on a live machine.
    """

    def test_asking_about_it_is_a_capability_question(self, engine) -> None:
        response = engine.handle_text("what would happen if I asked you to shut down")
        assert response.kind is not ResponseKind.CONFIRMATION_REQUIRED
        assert response.action != "shutdown_computer"

    def test_the_dry_run_reports_the_decision_without_making_it(self,
                                                               engine) -> None:
        engine.permissions.grant(Permission.SYSTEM_CONTROL)
        response = engine.execute(
            ActionRequest(action="shutdown_computer", parameters={}),
            dry_run=True)
        resolution = response.data["resolution"]
        assert resolution["selected"] == "shutdown_computer"
        assert resolution["risk_level"] == "High"
        assert resolution["would_require_confirmation"] is True
        assert resolution["executed"] is False
