"""The reminder scheduler and the multi-turn action context.

Every test here drives a fake clock.  Nothing waits: a reminder that is three
days late has to be testable, and a suite that slept for real seconds could
test neither that nor anything about the ordinary case in reasonable time.

The scheduler properties being defended:

**A reminder survives a restart.**  The store is the schedule.  A scheduler
built fresh over the same database finds what an earlier one left pending.

**A missed reminder is neither rung nor lost.**  Ringing Monday's 6 PM reminder
on Wednesday morning reads as something due now.  Dropping it silently makes
reminders untrustworthy.  So it is reported once, marked, and closed.

**Delivery is confirmed.**  A sink that declines -- the assistant is
mid-sentence -- leaves the reminder pending for the next poll.  Marking it done
because a notifier was *called* would be the same error as reporting an unsent
email as sent.

And for the pending-action store:

**Unrelated speech cannot fill a slot.**  This is the one that turns a helpful
dialogue into a wrong meeting, and it has the most tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

import pytest

from core.pending import (
    FILLABLE,
    MAX_TURNS,
    PendingAction,
    PendingActionStore,
    interpret_answer,
    is_cancellation,
)
from core.scheduler import (
    LATE_GRACE,
    MAX_DELIVERY_ATTEMPTS,
    Outcome,
    ReminderEvent,
    ReminderNotifier,
    ReminderScheduler,
)
from utils.when import local_now


# ---------------------------------------------------------------- fixtures

class FakeClock:
    """Time that only moves when a test says so."""

    def __init__(self, start: Optional[datetime] = None) -> None:
        self.now = start or local_now().replace(microsecond=0)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> datetime:
        self.now = self.now + timedelta(**kwargs)
        return self.now


class RecordingSink:
    """A notification sink that records, and can refuse delivery."""

    def __init__(self, *, accept: bool = True) -> None:
        self.accept = accept
        self.events: List[ReminderEvent] = []

    def __call__(self, event: ReminderEvent) -> bool:
        self.events.append(event)
        return self.accept

    @property
    def titles(self) -> List[str]:
        return [event.title for event in self.events]


def _add(database, clock: FakeClock, title: str, *, minutes: int,
         reminder_id: Optional[str] = None) -> str:
    identifier = reminder_id or title.replace(" ", "-")[:12]
    due = clock.now + timedelta(minutes=minutes)
    database.add_reminder(reminder_id=identifier, title=title,
                          due_at=due.isoformat(timespec="seconds"))
    return identifier


def _scheduler(database, clock: FakeClock, sink=None) -> ReminderScheduler:
    notifier = ReminderNotifier([sink] if sink is not None else [])
    return ReminderScheduler(database, notifier=notifier, clock=clock)


# --------------------------------------------------------------- scheduling

class TestDueDetection:

    def test_a_future_reminder_does_not_fire(self, database) -> None:
        clock = FakeClock()
        sink = RecordingSink()
        scheduler = _scheduler(database, clock, sink)
        _add(database, clock, "call Mum", minutes=30)

        assert scheduler.poll() == []
        assert sink.events == []
        assert database.reminder("call-Mum")["state"] == "pending"

    def test_a_reminder_fires_when_its_time_arrives(self, database) -> None:
        clock = FakeClock()
        sink = RecordingSink()
        scheduler = _scheduler(database, clock, sink)
        identifier = _add(database, clock, "submit project", minutes=1)

        assert scheduler.poll() == []          # not yet
        clock.advance(minutes=1, seconds=1)
        handled = scheduler.poll()

        assert [event.title for event in handled] == ["submit project"]
        assert sink.titles == ["submit project"]
        row = database.reminder(identifier)
        assert row["state"] == "done"
        assert row["outcome"] == Outcome.FIRED

    def test_a_fired_reminder_does_not_fire_twice(self, database) -> None:
        clock = FakeClock()
        sink = RecordingSink()
        scheduler = _scheduler(database, clock, sink)
        _add(database, clock, "take pills", minutes=-1)

        scheduler.poll()
        clock.advance(minutes=5)
        scheduler.poll()
        assert len(sink.events) == 1

    def test_a_cancelled_reminder_never_fires(self, database) -> None:
        clock = FakeClock()
        sink = RecordingSink()
        scheduler = _scheduler(database, clock, sink)
        identifier = _add(database, clock, "old plan", minutes=1)
        database.set_reminder_state(identifier, "cancelled")

        clock.advance(minutes=2)
        assert scheduler.poll() == []
        assert sink.events == []

    def test_due_now_reports_without_changing_anything(self, database) -> None:
        clock = FakeClock()
        scheduler = _scheduler(database, clock, RecordingSink())
        _add(database, clock, "read this", minutes=-1)

        assert [e.title for e in scheduler.due_now()] == ["read this"]
        assert database.reminder("read-this")["state"] == "pending"

    def test_an_unreadable_due_time_is_skipped_not_crashed(self, database) -> None:
        clock = FakeClock()
        sink = RecordingSink()
        scheduler = _scheduler(database, clock, sink)
        database.add_reminder(reminder_id="broken", title="nonsense",
                              due_at="not a date")
        assert scheduler.poll() == []
        assert sink.events == []


class TestRestart:

    def test_a_pending_reminder_is_found_by_a_new_scheduler(self,
                                                            database) -> None:
        """The store is the schedule -- nothing is held only in memory."""
        clock = FakeClock()
        _add(database, clock, "after restart", minutes=5)

        # The process ends here. A completely new scheduler, new notifier, new
        # everything, over the same database.
        clock.advance(minutes=6)
        sink = RecordingSink()
        revived = _scheduler(database, clock, sink)

        assert [e.title for e in revived.poll()] == ["after restart"]
        assert sink.titles == ["after restart"]

    def test_a_reminder_due_while_shut_down_fires_late_and_says_so(self,
                                                                   database) -> None:
        clock = FakeClock()
        _add(database, clock, "stand up", minutes=-45)
        sink = RecordingSink()
        scheduler = _scheduler(database, clock, sink)

        handled = scheduler.poll()
        assert len(handled) == 1
        assert handled[0].late is True
        assert handled[0].missed is False
        assert "45 minutes ago" in sink.events[0].spoken_text()
        assert database.reminder("stand-up")["outcome"] == Outcome.LATE


class TestMissedPolicy:

    def test_a_long_overdue_reminder_is_marked_missed(self, database) -> None:
        clock = FakeClock()
        identifier = _add(database, clock, "monday meeting", minutes=-(60 * 24 * 2))
        sink = RecordingSink()
        scheduler = _scheduler(database, clock, sink)

        handled = scheduler.poll()
        assert handled[0].missed is True
        row = database.reminder(identifier)
        assert row["state"] == "done"
        assert row["outcome"] == Outcome.MISSED

    def test_a_missed_reminder_is_reported_not_sounded_as_due(self,
                                                              database) -> None:
        clock = FakeClock()
        _add(database, clock, "the dentist", minutes=-(60 * 30))
        sink = RecordingSink()
        _scheduler(database, clock, sink).poll()

        text = sink.events[0].spoken_text()
        assert text.startswith("You missed")
        assert "Reminder —" not in text

    def test_the_grace_boundary_is_where_it_says_it_is(self, database) -> None:
        clock = FakeClock()
        inside = LATE_GRACE - timedelta(minutes=5)
        outside = LATE_GRACE + timedelta(minutes=5)
        _add(database, clock, "inside", minutes=-int(inside.total_seconds() // 60))
        _add(database, clock, "outside", minutes=-int(outside.total_seconds() // 60))

        sink = RecordingSink()
        handled = {e.title: e.missed for e in _scheduler(database, clock, sink).poll()}
        assert handled == {"inside": False, "outside": True}

    def test_a_missed_reminder_is_not_reported_again_next_time(self,
                                                               database) -> None:
        clock = FakeClock()
        _add(database, clock, "last week", minutes=-(60 * 24 * 7))
        sink = RecordingSink()
        scheduler = _scheduler(database, clock, sink)

        scheduler.poll()
        scheduler.poll()
        assert len(sink.events) == 1


class TestDelivery:

    def test_a_declined_delivery_leaves_the_reminder_pending(self,
                                                             database) -> None:
        """The assistant was mid-sentence; the reminder waits rather than dying."""
        clock = FakeClock()
        sink = RecordingSink(accept=False)
        scheduler = _scheduler(database, clock, sink)
        _add(database, clock, "speak later", minutes=-1)

        assert scheduler.poll() == []
        assert database.reminder("speak-later")["state"] == "pending"
        assert len(sink.events) == 1

    def test_a_retry_succeeds_once_the_sink_is_free(self, database) -> None:
        clock = FakeClock()
        sink = RecordingSink(accept=False)
        scheduler = _scheduler(database, clock, sink)
        _add(database, clock, "speak later", minutes=-1)

        scheduler.poll()
        sink.accept = True
        handled = scheduler.poll()

        assert [e.title for e in handled] == ["speak later"]
        assert database.reminder("speak-later")["state"] == "done"

    def test_delivery_gives_up_eventually_and_records_that(self,
                                                           database) -> None:
        clock = FakeClock()
        sink = RecordingSink(accept=False)
        scheduler = _scheduler(database, clock, sink)
        _add(database, clock, "nobody home", minutes=-1)

        for _ in range(MAX_DELIVERY_ATTEMPTS):
            scheduler.poll()
        row = database.reminder("nobody-home")
        assert row["state"] == "done"
        assert row["outcome"] == Outcome.UNDELIVERED

    def test_no_sink_at_all_is_not_a_silent_success(self, database) -> None:
        clock = FakeClock()
        scheduler = _scheduler(database, clock)          # no sinks
        identifier = _add(database, clock, "into the void", minutes=-1)

        assert scheduler.poll() == []
        assert database.reminder(identifier)["state"] == "pending"

    def test_one_broken_sink_does_not_stop_the_others(self, database) -> None:
        clock = FakeClock()
        good = RecordingSink()

        def _broken(event):
            raise RuntimeError("this front end is on fire")

        notifier = ReminderNotifier([_broken, good])
        scheduler = ReminderScheduler(database, notifier=notifier, clock=clock)
        _add(database, clock, "resilient", minutes=-1)

        assert [e.title for e in scheduler.poll()] == ["resilient"]
        assert good.titles == ["resilient"]


class TestTimezones:

    def test_a_stored_offset_is_respected(self, database) -> None:
        """6 PM means 6 PM on the clock the user was looking at."""
        from datetime import timezone

        clock = FakeClock(datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc))
        # Due at 12:30 UTC, written in a +05:30 zone as 18:00.
        plus_530 = timezone(timedelta(hours=5, minutes=30))
        due = datetime(2026, 6, 1, 18, 0, tzinfo=plus_530)
        database.add_reminder(reminder_id="tz", title="tz reminder",
                              due_at=due.isoformat(timespec="seconds"))
        sink = RecordingSink()
        scheduler = _scheduler(database, clock, sink)

        assert scheduler.poll() == [], "fired before its time in the stored zone"
        clock.advance(minutes=31)
        assert [e.title for e in scheduler.poll()] == ["tz reminder"]

    def test_next_due_is_measured_from_the_fake_clock(self, database) -> None:
        clock = FakeClock()
        scheduler = _scheduler(database, clock, RecordingSink())
        _add(database, clock, "soon", minutes=90)
        assert "1 hour" in (scheduler.next_due() or "")


class TestSchedulerThread:
    """The loop itself, rather than the policy it runs."""

    def test_start_and_stop_are_clean(self, database) -> None:
        scheduler = _scheduler(database, FakeClock(), RecordingSink())
        assert scheduler.start() is True
        assert scheduler.running is True
        assert scheduler.start() is False, "started twice"
        scheduler.stop()
        assert scheduler.running is False

    def test_stopping_one_that_never_started_is_harmless(self, database) -> None:
        _scheduler(database, FakeClock(), RecordingSink()).stop()


# ----------------------------------------------------------- pending actions

class TestCancellation:

    @pytest.mark.parametrize("phrase", [
        "never mind", "nevermind", "Never mind.", "cancel that", "cancel it",
        "forget it", "forget about it", "cancel", "leave it", "drop it",
    ])
    def test_cancellation_phrases_are_recognised(self, phrase) -> None:
        assert is_cancellation(phrase)

    @pytest.mark.parametrize("phrase", [
        "cancel my 6 PM reminder", "cancel the meeting", "10 AM",
        "project review", "never mind the gap",
    ])
    def test_commands_are_not_cancellations(self, phrase) -> None:
        assert not is_cancellation(phrase)


class TestPendingStore:

    def test_opening_and_filling_a_slot(self) -> None:
        store = PendingActionStore()
        opened = store.open("create_event", {"when": "tomorrow at 10 AM"},
                            needs="title", question="What should I call it?")
        assert opened is not None
        filled = store.fill("Project review")
        assert filled is not None
        assert filled.parameters["title"] == "Project review"
        assert filled.parameters["when"] == "tomorrow at 10 AM"

    def test_clearing_really_clears(self) -> None:
        store = PendingActionStore()
        store.open("create_reminder", {}, needs="when", question="When?")
        store.clear("test")
        assert store.current() is None
        assert store.is_open is False
        assert store.fill("6 PM") is None

    def test_an_expired_question_is_dropped(self) -> None:
        """An answer given long after the question must not complete it.

        The creation time is pushed into the past rather than slept through:
        the property is about elapsed time, not about this test taking any.
        """
        store = PendingActionStore(ttl_seconds=120.0)
        pending = store.open("create_event", {}, needs="when", question="When?")
        assert pending is not None
        pending.created_at -= 121.0
        assert store.current() is None

    def test_an_action_not_on_the_allow_list_opens_nothing(self) -> None:
        store = PendingActionStore()
        assert store.open("shutdown_computer", {}, needs="when",
                          question="When?") is None
        assert store.current() is None

    def test_a_slot_the_action_does_not_take_opens_nothing(self) -> None:
        store = PendingActionStore()
        assert store.open("create_reminder", {}, needs="recipient",
                          question="Who?") is None

    def test_send_email_is_not_conversationally_fillable(self) -> None:
        """A send is confirmed outright, never assembled out of fragments."""
        assert "send_email" not in FILLABLE

    def test_a_dialogue_gives_up_rather_than_interrogating(self) -> None:
        store = PendingActionStore()
        for turn in range(MAX_TURNS):
            assert store.open("create_event", {}, needs="title",
                              question="?") is not None
        assert store.open("create_event", {}, needs="title", question="?") is None


class TestAnswerInterpretation:

    def _pending(self, needs: str) -> PendingAction:
        return PendingAction(action="create_event", needs=needs, question="?")

    @pytest.mark.parametrize("answer", [
        "10 AM", "at 10 AM", "tomorrow at 3pm", "in 30 minutes", "noon",
        "next Monday at 2pm",
    ])
    def test_a_time_slot_takes_times(self, answer) -> None:
        assert interpret_answer(self._pending("when"), answer) == answer

    @pytest.mark.parametrize("answer", [
        "project review", "the weather", "sam@example.com", "",
    ])
    def test_a_time_slot_refuses_non_times(self, answer) -> None:
        assert interpret_answer(self._pending("when"), answer) is None

    @pytest.mark.parametrize("spoken,stored", [
        ("Project review", "Project review"),
        ("call it Project review", "Project review"),
        ("it's the standup", "standup"),
        ("\"Design sync\"", "Design sync"),
    ])
    def test_a_text_slot_strips_the_framing(self, spoken, stored) -> None:
        assert interpret_answer(self._pending("title"), spoken) == stored

    def test_a_cancellation_is_never_an_answer(self) -> None:
        assert interpret_answer(self._pending("title"), "never mind") is None

    def test_an_essay_is_not_a_slot_value(self) -> None:
        assert interpret_answer(self._pending("title"), "x" * 500) is None
