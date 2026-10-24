"""The wiring that turns a stored reminder into one the user actually sees.

Phase 2 could store, list and cancel reminders.  Phase 3 added a scheduler that
rings them.  Between those two there was still a gap that no test noticed,
because both halves were correct on their own: **nothing ever called
``start_reminders``**, so a running AARMA had a scheduler that was never
started and reminders that therefore never fired.

That is the failure this file exists to catch, and it is a good example of why
unit tests alone were not enough: the scheduler had 59 passing tests while
being, in the shipped application, entirely inert.  So the tests here assert on
the *seam* -- that an interface starts the scheduler, that its sink reaches the
user, and that stopping is clean -- rather than on the scheduler's own logic,
which is covered in ``test_phase3_scheduler.py``.

Nothing here opens a window or a terminal.  The console interface is driven
with its own printing captured; the dashboard is checked by reading its source
for the wiring, because constructing a ``QMainWindow`` in a headless test run
is a different and much flakier problem than the one being tested.
"""

from __future__ import annotations

import inspect
from datetime import timedelta
from typing import Any, List

import pytest

from core.scheduler import ReminderEvent
from security.permissions import Permission
from utils.when import local_now


@pytest.fixture
def registry():
    from actions.registry import build_default_registry

    return build_default_registry()


def _event(title: str = "test the reminder", *, missed: bool = False,
           late_minutes: int = 0) -> ReminderEvent:
    now = local_now()
    return ReminderEvent(reminder_id="x", title=title,
                         due_at=now - timedelta(minutes=late_minutes), now=now,
                         missed=missed)


# ------------------------------------------------------- the console interface

class TestConsoleWiring:

    def test_the_console_starts_the_scheduler(self, engine, monkeypatch) -> None:
        """The seam that was missing: an interface has to start it.

        ``run()`` is not called -- it would block on ``input()``.  Instead the
        two things it does before the loop are checked directly, which is where
        the start call lives.
        """
        from ui.console import ConsoleInterface

        source = inspect.getsource(ConsoleInterface.run)
        assert "start_reminders" in source, (
            "the console never starts the scheduler, so reminders never fire")

    def test_the_console_sink_prints_and_reports_delivery(self, engine,
                                                          capsys) -> None:
        from ui.console import ConsoleInterface

        console = ConsoleInterface(engine)
        delivered = console.show_reminder(_event("submit my project"))

        assert delivered is True, (
            "the sink must report delivery, or the scheduler retries forever")
        printed = capsys.readouterr().out
        assert "submit my project" in printed

    def test_a_late_reminder_says_how_late(self, engine, capsys) -> None:
        from ui.console import ConsoleInterface

        ConsoleInterface(engine).show_reminder(_event("stand up", late_minutes=45))
        assert "45 minutes ago" in capsys.readouterr().out

    def test_a_missed_reminder_reads_as_history(self, engine, capsys) -> None:
        from ui.console import ConsoleInterface

        ConsoleInterface(engine).show_reminder(_event("the dentist", missed=True))
        printed = capsys.readouterr().out
        assert "You missed" in printed
        assert "Reminder —" not in printed

    def test_the_sink_survives_a_mute_speaker(self, engine, capsys,
                                              monkeypatch) -> None:
        """A speaker fault must not lose the reminder that was already shown."""
        from ui.console import ConsoleInterface

        class _Broken:
            enabled = True

            def speak(self, *_args, **_kwargs):
                raise RuntimeError("no audio device")

        monkeypatch.setattr(type(engine), "speech",
                            property(lambda self: _Broken()))
        assert ConsoleInterface(engine).show_reminder(_event()) is True
        assert "test the reminder" in capsys.readouterr().out


# ------------------------------------------------------------- the dashboard

class TestDashboardWiring:
    """Read the wiring rather than build a window.

    A headless ``QMainWindow`` is its own source of flakiness, and what needs
    asserting here is structural: the scheduler is started, its sink only
    emits, and it is stopped before the widgets go away.
    """

    def test_the_dashboard_exposes_a_start_and_something_calls_it(self) -> None:
        """Started after show(), not during construction.

        The first version of this wiring called ``start_reminders`` inside
        ``Dashboard.__init__``.  That pointed a live scheduler thread at a
        half-built QMainWindow and took the headless dashboard fixture down
        with a Windows access violation.  The start therefore has to be a
        separate call -- and something has to actually make it, which is the
        second half of this test and the whole reason the feature was inert
        before.
        """
        import inspect as _inspect

        import main
        import ui.dashboard as dashboard

        # The call form, not the bare word: the comment in __init__ explains
        # where the start moved to, and a word match would flag the comment.
        assert "engine.start_reminders(" not in _inspect.getsource(
            dashboard.Dashboard.__init__), (
            "constructing the window must not start background work")
        assert hasattr(dashboard.Dashboard, "start_reminders")

        run_gui = _inspect.getsource(main.run_gui)
        assert "start_reminders" in run_gui, (
            "nothing starts the scheduler, so reminders never fire")
        assert run_gui.index("window.show()") < run_gui.index("start_reminders"), (
            "the scheduler must start after the window is shown")

    def test_the_sink_only_emits_and_never_touches_a_widget(self) -> None:
        """Qt widgets may not be touched from the scheduler's thread."""
        import ui.dashboard as dashboard

        source = inspect.getsource(dashboard.Dashboard._deliver_reminder)
        assert "emit" in source
        for widget_call in ("add_assistant_message", "setText", "append",
                            "show(", "chat_page."):
            assert widget_call not in source, (
                f"_deliver_reminder touches {widget_call} on the scheduler "
                "thread; it must only emit")

    def test_the_scheduler_is_stopped_before_teardown(self) -> None:
        """Otherwise a reminder can arrive at a widget being destroyed."""
        import ui.dashboard as dashboard

        source = inspect.getsource(dashboard.Dashboard.closeEvent)
        assert "stop_reminders" in source
        assert source.index("stop_reminders") < source.index("_thread.quit"), (
            "the scheduler must stop before the worker thread is torn down")

    def test_the_bridge_exists_and_carries_the_event(self) -> None:
        import ui.dashboard as dashboard

        assert hasattr(dashboard, "ReminderBridge")
        assert hasattr(dashboard.ReminderBridge, "fired")


# --------------------------------------------------- end to end, on a fake clock

class TestSpokenReminderReachesASink:
    """From "remind me in one minute" to a sink being called, no real waiting."""

    def test_a_spoken_reminder_fires_through_the_engine_scheduler(self,
                                                                  engine) -> None:
        engine.permissions.grant(Permission.REMINDERS)
        delivered: List[ReminderEvent] = []

        engine.reminders.notifier.add_sink(
            lambda event: bool(delivered.append(event)) or True)

        response = engine.handle_text(
            "remind me in 1 minute to test the reminder")
        assert response.ok, response.text

        # The clock is moved, not waited on.
        engine.reminders.poll(now=local_now() + timedelta(minutes=2))

        assert [event.title for event in delivered] == ["test the reminder"]
        row = engine.database.reminder(delivered[0].reminder_id)
        assert row["state"] == "done"
        assert row["outcome"] in ("fired", "late")

    def test_a_reminder_the_sink_refuses_stays_pending(self, engine) -> None:
        """The assistant was mid-sentence: hold it, do not lose it."""
        engine.permissions.grant(Permission.REMINDERS)
        engine.reminders.notifier.add_sink(lambda event: False)

        engine.handle_text("remind me in 1 minute to call Mum")
        engine.reminders.poll(now=local_now() + timedelta(minutes=2))

        pending = engine.database.reminders(state="pending")
        assert [row["title"] for row in pending] == ["call Mum"]

    def test_the_engine_does_not_start_a_thread_by_itself(self, engine) -> None:
        """A headless session pays for no thread it did not ask for."""
        assert engine.reminders.running is False

    def test_starting_twice_is_harmless(self, engine) -> None:
        assert engine.start_reminders(lambda event: True) is True
        try:
            assert engine.start_reminders(lambda event: True) is False
        finally:
            engine.stop_reminders()
        assert engine.reminders.running is False

    def test_shutdown_stops_the_scheduler(self, config, database,
                                          resolver) -> None:
        from core.engine import AarmaEngine

        instance = AarmaEngine(config, database=database,
                               path_resolver=resolver, session_id="wiring")
        instance.start_reminders(lambda event: True)
        assert instance.reminders.running is True
        instance.shutdown()
        assert instance.reminders.running is False


# --------------------------------------------------------- volume phrasings

class TestVolumeRouting:
    """Stepping the volume, in the words people actually use.

    "Set volume lower" used to reach ``unknown``.  It reads like an absolute
    instruction and is not one -- there is no level named -- so it belongs with
    the stepping action rather than with ``set_volume``, which refuses outright
    because it cannot land on an exact figure without the Core Audio API.
    """

    @pytest.mark.parametrize("utterance", [
        "set volume lower", "set the volume lower", "lower the volume",
        "decrease volume", "decrease the volume", "turn the volume down",
        "turn it down", "volume down", "quieter", "reduce the sound",
    ])
    def test_these_all_step_the_volume_down(self, engine, utterance) -> None:
        assert engine.intent_engine.parse(utterance).name == "volume_down"

    @pytest.mark.parametrize("utterance", [
        "set volume higher", "set the volume louder", "raise the volume",
        "increase volume", "turn the volume up", "volume up", "louder",
    ])
    def test_these_all_step_the_volume_up(self, engine, utterance) -> None:
        assert engine.intent_engine.parse(utterance).name == "volume_up"

    @pytest.mark.parametrize("utterance", [
        "set volume to 30 percent", "set the volume to 50",
    ])
    def test_an_absolute_level_is_a_different_action(self, engine,
                                                     utterance) -> None:
        """And that action refuses honestly rather than approximating."""
        assert engine.intent_engine.parse(utterance).name == "set_volume"

    def test_setting_an_exact_level_is_refused_not_approximated(
            self, registry, action_context) -> None:
        spec = registry.require("set_volume")
        result = spec.handler(action_context, spec.validate({"percent": "30"}))
        assert result.ok is False
        assert "pycaw" in result.message or "can't set" in result.message


# --------------------------------------------- the scheduler's exact lifecycle

class TestSchedulerLifecycle:
    """The five properties asked for by name, each asserted on its own."""

    def test_it_starts_exactly_once(self, engine) -> None:
        assert engine.start_reminders(lambda event: True) is True
        try:
            assert engine.start_reminders(lambda event: True) is False
            assert engine.reminders.running is True
        finally:
            engine.stop_reminders()

    def test_it_is_not_running_before_an_interface_starts_it(self,
                                                             engine) -> None:
        assert engine.reminders.running is False

    def test_it_stops_on_teardown(self, engine) -> None:
        engine.start_reminders(lambda event: True)
        engine.shutdown()
        assert engine.reminders.running is False

    def test_it_emits_a_due_reminder(self, engine) -> None:
        from datetime import timedelta

        seen: List[ReminderEvent] = []
        engine.reminders.notifier.add_sink(
            lambda event: bool(seen.append(event)) or True)
        engine.database.add_reminder(
            reminder_id="due", title="ring me",
            due_at=(local_now() - timedelta(minutes=1)).isoformat(
                timespec="seconds"))

        engine.reminders.poll()
        assert [event.title for event in seen] == ["ring me"]

    def test_it_does_not_duplicate_a_notification(self, engine) -> None:
        """Polled repeatedly, a fired reminder is announced once."""
        from datetime import timedelta

        seen: List[ReminderEvent] = []
        engine.reminders.notifier.add_sink(
            lambda event: bool(seen.append(event)) or True)
        engine.database.add_reminder(
            reminder_id="once", title="only once",
            due_at=(local_now() - timedelta(minutes=1)).isoformat(
                timespec="seconds"))

        for _ in range(5):
            engine.reminders.poll()
        assert len(seen) == 1, f"announced {len(seen)} times"

    def test_two_sinks_both_hear_it_but_it_still_fires_once(self,
                                                            engine) -> None:
        from datetime import timedelta

        first: List[ReminderEvent] = []
        second: List[ReminderEvent] = []
        engine.reminders.notifier.add_sink(
            lambda event: bool(first.append(event)) or True)
        engine.reminders.notifier.add_sink(
            lambda event: bool(second.append(event)) or True)
        engine.database.add_reminder(
            reminder_id="fanout", title="both",
            due_at=(local_now() - timedelta(minutes=1)).isoformat(
                timespec="seconds"))

        for _ in range(3):
            engine.reminders.poll()
        assert len(first) == 1 and len(second) == 1


# ------------------------------------------------------ the weather dependency

class TestWeatherDependencyIsDeclared:

    def test_requests_is_in_requirements(self) -> None:
        """The weather provider needs it, so an install must supply it.

        Without the declaration the actions still behave honestly -- they
        report the component missing -- but "honest about being broken" is not
        the same as working.
        """
        from pathlib import Path

        text = Path("requirements.txt").read_text(encoding="utf-8")
        assert "requests" in text

    def test_the_provider_says_so_when_it_is_absent(self) -> None:
        """And if it really is missing, the message names the remedy."""
        from providers.weather import OpenMeteoProvider

        provider = OpenMeteoProvider(fetch=None)
        provider._fetch = None
        # Simulate the import failing without touching the real module table.
        object.__setattr__(provider, "_transport", lambda: None)
        availability = provider.availability()
        assert availability.available is False
        assert "requests" in availability.detail
        assert "pip install" in availability.remedy
