"""Multi-turn actions, driven through the engine exactly as a user would.

These are the tests that matter most for the feature, because the dangerous
failure is not "the dialogue didn't work" -- it is "the dialogue worked on the
wrong sentence".  A store that eagerly fills its slot turns

    "Schedule a meeting tomorrow."   "What time?"   "Check my email."

into a meeting called "Check my email", and the user finds out when it goes
off.  So most of this file is about what must *not* be captured.

The last class is a set of mutation tests: each one breaks a safety property on
purpose and asserts that something here notices.  A test that passes whether or
not the guard exists is not testing the guard.
"""

from __future__ import annotations

import dataclasses
import types
from datetime import timedelta
from typing import Any, Dict, List

import pytest

from actions.registry import ActionRequest
from core.engine import ResponseKind
from core.pending import PendingActionStore
from security.permissions import Permission
from security.validator import RiskLevel
from utils.when import local_now


@pytest.fixture
def ready(engine):
    """An engine with the permissions these flows need, and nothing more."""
    for permission in (Permission.CALENDAR, Permission.REMINDERS,
                       Permission.EMAIL):
        engine.permissions.grant(permission)
    return engine


def _say(engine, text: str):
    return engine.handle_text(text)


# --------------------------------------------------------------- the flow

class TestCalendarDialogue:

    def test_a_day_with_no_time_is_asked_about(self, ready) -> None:
        response = _say(ready, "schedule a meeting tomorrow")
        assert "what time" in response.text.lower()
        assert ready.pending.is_open
        assert ready.pending.current().needs == "when"

    def test_the_answer_completes_the_action(self, ready) -> None:
        _say(ready, "schedule a meeting tomorrow")
        second = _say(ready, "10 AM")
        # Now it wants a title -- the time was accepted and carried forward.
        assert ready.pending.current().needs == "title"
        assert "call" in second.text.lower()

        third = _say(ready, "Project review")
        assert third.kind is ResponseKind.RESULT, third.text
        assert "Project review" in third.text
        assert ready.pending.is_open is False

    def test_the_event_really_lands_in_the_local_calendar(self, ready) -> None:
        _say(ready, "schedule a meeting tomorrow")
        _say(ready, "10 AM")
        _say(ready, "Project review")

        events = ready.database.events()
        assert len(events) == 1
        assert events[0]["title"] == "Project review"
        assert events[0]["provider"] == "local"

    def test_the_named_day_survives_the_follow_up_answer(self, ready) -> None:
        """The meeting lands on the day the user said, at any hour of the day.

        Found live, not here: the answer "10 AM" used to replace the stored
        "tomorrow" outright, and ``parse_when("10 AM")`` then resolved against
        the current moment.  Run at 22:00 that yields tomorrow anyway -- 10 AM
        has passed, so the parser takes the next occurrence -- and the bug is
        invisible.  Run at 01:00 it yields *today*, and the meeting is silently
        booked on the wrong day.

        So this asserts the stored **date**, computed from the clock rather
        than hard-coded, which is what makes it fail at 01:00 as well as 22:00.
        """
        from datetime import timedelta

        _say(ready, "schedule a meeting tomorrow")
        _say(ready, "10 AM")
        _say(ready, "Project review")

        events = ready.database.events()
        assert len(events) == 1
        stored = events[0]["starts_at"][:10]
        expected = (local_now() + timedelta(days=1)).date().isoformat()
        assert stored == expected, (
            f"meeting stored for {stored}, expected tomorrow ({expected})")
        assert events[0]["starts_at"][11:16] == "10:00"

    def test_a_follow_up_that_names_its_own_day_replaces(self, ready) -> None:
        """Completion merges; correction replaces."""
        from core.pending import merge_time

        assert merge_time("tomorrow", "10 AM") == "tomorrow at 10 AM"
        assert merge_time("tomorrow", "friday at 9") == "friday at 9"
        assert merge_time("", "10 AM") == "10 AM"

    def test_the_reply_names_the_calendar_it_used(self, ready) -> None:
        """Not "scheduled" -- which invites the assumption that it synced."""
        _say(ready, "schedule a meeting tomorrow")
        _say(ready, "10 AM")
        response = _say(ready, "Project review")
        assert "AARMA's own calendar" in response.text

    def test_a_complete_sentence_needs_no_dialogue(self, ready) -> None:
        response = _say(ready,
                        "create an event called Project Review tomorrow at 2 PM")
        assert response.kind is ResponseKind.RESULT, response.text
        assert ready.pending.is_open is False


class TestReminderDialogue:

    def test_a_time_with_no_subject_is_asked_about(self, ready) -> None:
        response = _say(ready, "remind me at 6 PM")
        assert ready.pending.current().action == "create_reminder"
        assert ready.pending.current().needs == "title"
        assert "remind you about" in response.text

    def test_the_answer_creates_the_reminder(self, ready) -> None:
        _say(ready, "remind me at 6 PM")
        response = _say(ready, "submit my project")
        assert response.kind is ResponseKind.RESULT, response.text
        rows = ready.database.reminders(state="pending")
        assert [row["title"] for row in rows] == ["submit my project"]


class TestCancellation:

    @pytest.mark.parametrize("phrase", ["never mind", "cancel that", "forget it"])
    def test_cancelling_clears_the_pending_action(self, ready, phrase) -> None:
        _say(ready, "schedule a meeting tomorrow")
        assert ready.pending.is_open

        response = _say(ready, phrase)
        assert ready.pending.is_open is False
        assert "cancel" in response.text.lower()

    def test_nothing_is_created_after_a_cancellation(self, ready) -> None:
        _say(ready, "schedule a meeting tomorrow")
        _say(ready, "never mind")
        _say(ready, "10 AM")                 # would have completed it
        assert ready.database.events() == []

    def test_a_cancelled_dialogue_does_not_come_back(self, ready) -> None:
        _say(ready, "remind me at 6 PM")
        _say(ready, "forget it")
        _say(ready, "submit my project")
        assert ready.database.reminders(state="pending") == []


class TestUnrelatedSpeechIsNotCaptured:
    """The property the whole design is built around."""

    def test_a_new_command_wins_over_the_open_question(self, ready) -> None:
        _say(ready, "schedule a meeting tomorrow")
        response = _say(ready, "what reminders do I have")

        assert response.action == "list_reminders", response.text
        assert ready.pending.is_open is False
        assert ready.database.events() == []

    def test_a_command_does_not_become_an_event_title(self, ready) -> None:
        _say(ready, "schedule a meeting tomorrow")
        _say(ready, "10 AM")                 # now it wants a title
        _say(ready, "what reminders do I have")

        assert ready.database.events() == []
        assert ready.pending.is_open is False

    def test_an_answer_that_does_not_fit_the_slot_is_not_forced_in(self,
                                                                   ready) -> None:
        _say(ready, "schedule a meeting tomorrow")       # wants a time
        _say(ready, "the blue one")                      # not a time
        assert ready.pending.is_open is False
        assert ready.database.events() == []

    def test_a_stale_question_cannot_be_answered_later(self, ready) -> None:
        _say(ready, "schedule a meeting tomorrow")
        pending = ready.pending.current()
        pending.created_at -= 10_000          # asked a long time ago

        _say(ready, "10 AM")
        assert ready.database.events() == []
        assert ready.pending.is_open is False

    def test_only_one_dialogue_is_ever_open(self, ready) -> None:
        _say(ready, "schedule a meeting tomorrow")
        _say(ready, "remind me at 6 PM")
        current = ready.pending.current()
        assert current is not None
        assert current.action == "create_reminder"


class TestSchedulerWiring:

    def test_the_engine_exposes_a_scheduler_that_is_not_running(self,
                                                                engine) -> None:
        """Nothing starts a thread a headless session did not ask for."""
        assert engine.reminders.running is False

    def test_starting_and_stopping_is_clean(self, engine) -> None:
        delivered: List[Any] = []
        assert engine.start_reminders(lambda event: bool(delivered.append(event)) or True)
        assert engine.reminders.running is True
        engine.stop_reminders()
        assert engine.reminders.running is False

    def test_a_reminder_created_by_speech_is_visible_to_the_scheduler(self,
                                                                      ready) -> None:
        """The two halves meet in the database, which is the whole design."""
        _say(ready, "remind me in 1 minute to test the reminder")
        due = local_now() + timedelta(minutes=2)
        events = ready.reminders.due_now(now=due)
        assert [event.title for event in events] == ["test the reminder"]

    def test_a_missed_reminder_is_speakable_only_as_history(self, ready) -> None:
        from core.scheduler import ReminderEvent

        event = ReminderEvent(reminder_id="x", title="the dentist",
                              due_at=local_now() - timedelta(days=2),
                              now=local_now(), missed=True)
        # Never spoken: an alarm for something two days ago reads as one for now.
        assert ready.speak_reminder(event) is False


# ------------------------------------------------------------ mutation tests

class TestMutations:
    """Each of these breaks a guard and asserts that a test would catch it.

    Written because a passing suite proves nothing on its own: it has to be
    shown that the assertions are load-bearing.
    """

    def test_removing_the_cmd_alias_is_caught(self) -> None:
        from actions import applications

        original = dict(applications._ALIAS_INDEX)
        try:
            applications._ALIAS_INDEX.pop("cmd", None)
            assert applications.resolve_entry("cmd") is None, (
                "the alias index is not what resolve_entry consults")
        finally:
            applications._ALIAS_INDEX.clear()
            applications._ALIAS_INDEX.update(original)
        assert applications.resolve_entry("cmd") is not None

    def test_sending_a_browser_shortcut_to_a_non_browser_is_caught(self) -> None:
        from providers.desktop import BROWSER_PROCESSES, active_browser
        from tests.test_phase3_desktop import FakeDesktop, editor_window, _windows

        provider = _windows(FakeDesktop([editor_window()]))
        assert active_browser(provider)[0] is None

        # Mutation: pretend Notepad is a browser. The gate must be what decides.
        BROWSER_PROCESSES["notepad.exe"] = "Notepad"
        try:
            assert active_browser(provider)[0] is not None, (
                "the browser gate does not consult BROWSER_PROCESSES")
        finally:
            BROWSER_PROCESSES.pop("notepad.exe")

    def test_downgrading_send_email_risk_is_caught(self, engine) -> None:
        """Confirmation follows the declared risk, proved without sending.

        ``dry_run`` evaluates every gate -- registry, platform, parameters,
        permissions, connectivity, precheck, risk -- and then returns the
        decision instead of calling the handler.  ``AarmaEngine._invoke``
        raises if a dry run ever reaches it, so this cannot execute even if the
        dispatch path is changed later.
        """
        engine.permissions.grant(Permission.EMAIL)
        engine.permissions.grant(Permission.INTERNET)
        spec = engine.registry.require("send_email")
        assert spec.risk is RiskLevel.HIGH

        request = ActionRequest(action="send_email", parameters={}, source="test")
        honest = engine.execute(request, dry_run=True)
        assert honest.data["resolution"]["would_require_confirmation"] is True

        # dataclasses.replace, not SimpleNamespace: the mutated entry has to be
        # a real ActionSpec so it still validates parameters on the way through.
        # An earlier version of this test substituted a SimpleNamespace, which
        # raised AttributeError inside `execute` -- and `handle_text` caught it
        # and returned an error response, which "is not CONFIRMATION_REQUIRED".
        # The assertion passed while the mutation was never actually applied.
        mutated = dataclasses.replace(spec, risk=RiskLevel.LOW)
        object.__setattr__(engine.registry, "_actions",
                           {**engine.registry._actions, "send_email": mutated})
        lowered = engine.execute(request, dry_run=True)
        assert lowered.data["resolution"]["risk_level"] == "Low"
        assert lowered.data["resolution"]["would_require_confirmation"] is False, (
            "confirmation does not depend on the declared risk level")
        assert lowered.data["resolution"]["executed"] is False

    def test_removing_the_shutdown_confirmation_is_caught(
            self, engine, monkeypatch) -> None:
        """The confirmation gate is RiskPolicy -- proved without shutting down.

        This test is the reason the whole suite now has a hard guard around
        power commands.  Its first version disabled
        ``RiskPolicy.requires_confirmation`` and then sent "shut down the
        computer" through a live engine, which carries no injected process
        runner; ``actions.power`` fell through to the real ``subprocess.run``
        and scheduled ``shutdown /s /t 20``.  The assertion passed. The laptop
        did not.

        The property is the same and is worth keeping.  What changed is the
        instrument: ``dry_run`` evaluates the risk gate and reports the
        decision without calling the handler, and
        ``AarmaEngine._invoke`` raises ``BenchmarkExecutionError`` if a dry run
        ever reaches it -- so no future edit to the dispatch path can quietly
        turn this back into a real shutdown.
        """
        from security.validator import RiskPolicy

        engine.permissions.grant(Permission.SYSTEM_CONTROL)
        request = ActionRequest(action="shutdown_computer", parameters={},
                                source="test")

        honest = engine.execute(request, dry_run=True)
        assert honest.data["resolution"]["would_require_confirmation"] is True
        assert honest.data["resolution"]["executed"] is False

        monkeypatch.setattr(RiskPolicy, "requires_confirmation",
                            lambda self, risk: False)
        mutated = engine.execute(request, dry_run=True)
        assert mutated.data["resolution"]["would_require_confirmation"] is False, (
            "the confirmation gate is not RiskPolicy")
        assert mutated.data["resolution"]["executed"] is False

    def test_fabricating_weather_is_caught(self, registry, action_context) -> None:
        from providers.weather import WeatherReading

        honest = WeatherReading(place="Leeds")            # nothing measured
        assert "°C" not in honest.summary()

        invented = WeatherReading(place="Leeds", temperature_c=21.0)
        assert "21°C" in invented.summary(), (
            "the summary does not reflect the reading it was given")

    def test_a_pending_action_surviving_cancellation_is_caught(self) -> None:
        store = PendingActionStore()
        store.open("create_event", {}, needs="when", question="When?")
        store.clear("test")
        assert store.current() is None
        assert store.fill("10 AM") is None, (
            "fill works on a cleared store, so cancellation does not release it")

    def test_a_scheduler_that_forgets_across_restarts_is_caught(self,
                                                                database) -> None:
        from core.scheduler import ReminderNotifier, ReminderScheduler
        from tests.test_phase3_scheduler import FakeClock, RecordingSink

        clock = FakeClock()
        database.add_reminder(
            reminder_id="survivor", title="still here",
            due_at=(clock.now - timedelta(minutes=1)).isoformat(timespec="seconds"))

        sink = RecordingSink()
        fresh = ReminderScheduler(database, notifier=ReminderNotifier([sink]),
                                  clock=clock)
        assert [e.title for e in fresh.poll()] == ["still here"], (
            "a new scheduler does not read the store, so restarts lose reminders")

    def test_an_unavailable_provider_reporting_success_is_caught(self) -> None:
        from providers import Availability, ProviderResult

        result = ProviderResult.unavailable(
            Availability.missing("gmail", "Gmail isn't connected."))
        assert result.ok is False
        assert "isn't connected" in result.message

    def test_no_sentence_resolves_to_shell_execution(self, engine) -> None:
        """Phrasing must not be a route to running code.

        ``execute_shell_command`` exists from Phase 5 on, and this is what
        keeps it honest: it is reached by asking for it outright, never by a
        sentence that merely sounds like it might mean running something.
        A resolver that could reach it would make the HIGH-risk grading
        decorative, because the user never chose that door.
        """
        for utterance in ("run this", "run it", "execute this",
                          "run the project", "run my code", "just run it",
                          "start the server", "run npm install"):
            intent = engine.intent_engine.parse(utterance)
            assert intent.name != "execute_shell_command", (
                f"{utterance!r} resolved straight to shell execution")

    def test_shell_execution_still_demands_confirmation(self, engine) -> None:
        from actions.registry import ActionRequest
        from security.permissions import Permission

        engine.permissions.grant(Permission.SYSTEM_CONTROL)
        response = engine.execute(ActionRequest(
            action="execute_shell_command",
            parameters={"command": "echo hello"}))
        assert response.kind is ResponseKind.CONFIRMATION_REQUIRED
