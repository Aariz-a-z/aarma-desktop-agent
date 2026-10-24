"""The reminder scheduler: the part that makes a reminder a promise.

Phase 2 could store a reminder and read it back.  That is worth something --
nothing was lost -- but it is not what the user asked for when they said
"remind me at 6 PM".  This module is the piece that rings.

Four properties the design is built around:

**It survives a restart.**  Nothing is held in memory that matters.  The
database is the schedule; the scheduler is a loop that reads it.  Closing AARMA
at 5:55 and reopening it at 5:58 still produces a 6 PM reminder, because
:meth:`ReminderScheduler.poll` looks at what is stored, not at what it was
told when it started.

**A missed reminder is handled by a stated policy, not by accident.**  If the
machine was off from Monday to Wednesday, Monday's 6 PM reminder must not go
off the instant AARMA starts, three days late, as though it were due now.  Nor
should it vanish silently.  :data:`LATE_GRACE` draws the line: inside it the
reminder fires and says how late it is, outside it the reminder is closed as
*missed* and reported once, without an alarm.

**Delivery is confirmed, not assumed.**  :meth:`ReminderNotifier.notify`
returns whether the reminder actually reached the user.  If nothing accepted it
-- the assistant was mid-sentence, the window was closed -- the reminder stays
pending and is tried again.  Marking it done on the strength of having *called*
a notifier would be the same class of error as reporting an unsent email as
sent.

**The clock is injected.**  Every test drives time explicitly.  A suite that
waited for real seconds would be slow, flaky, and unable to test a reminder
that is three days late at all.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_logger
from utils.when import describe, local_now, until_text

logger = get_logger("core.scheduler")

#: How often the loop wakes.  Reminders are minute-granular, so a shorter
#: interval would burn wake-ups for precision nobody asked for; a longer one
#: would make "remind me in one minute" land visibly late.
POLL_SECONDS = 15.0

#: How overdue a reminder may be and still be worth ringing.  Twelve hours
#: covers "the laptop was shut overnight" while stopping a reminder from last
#: week sounding like a live one.
LATE_GRACE = timedelta(hours=12)

#: Overdue by more than this and the reminder is *late* rather than on time,
#: which changes what the user is told.  A reminder half a minute behind is on
#: time by any standard a person applies.
LATE_AFTER = timedelta(minutes=2)

#: How many polls a reminder may fail to be delivered before it is closed out.
#: Without a bound, a machine with no notification sink would retry one
#: reminder for ever and never reach the next.
MAX_DELIVERY_ATTEMPTS = 20


class Outcome:
    """The values written to ``reminders.outcome``."""

    FIRED = "fired"
    LATE = "late"
    MISSED = "missed"
    UNDELIVERED = "undelivered"


@dataclass(frozen=True)
class ReminderEvent:
    """One reminder, at the moment it comes due."""

    reminder_id: str
    title: str
    due_at: datetime
    now: datetime
    #: True when the due time passed while AARMA was not running long enough
    #: ago that ringing would be misleading.  Missed events are *reported*,
    #: never sounded as though they were due now.
    missed: bool = False

    @property
    def late_by(self) -> timedelta:
        return max(timedelta(0), self.now - self.due_at)

    @property
    def late(self) -> bool:
        return self.late_by > LATE_AFTER

    def spoken_text(self) -> str:
        """What the user hears or reads.  Lateness is stated, never hidden."""
        if self.missed:
            return (f"You missed a reminder: {self.title} "
                    f"({describe(self.due_at, reference=self.now)}).")
        if self.late:
            minutes = int(self.late_by.total_seconds() // 60)
            unit = "minute" if minutes == 1 else "minutes"
            return (f"Reminder — {self.title}. This was due "
                    f"{describe(self.due_at, reference=self.now)}, "
                    f"{minutes} {unit} ago.")
        return f"Reminder — {self.title}."

    def as_dict(self) -> Dict[str, Any]:
        return {"reminder_id": self.reminder_id, "title": self.title,
                "due_at": self.due_at.isoformat(timespec="seconds"),
                "missed": self.missed, "late": self.late,
                "late_by_seconds": int(self.late_by.total_seconds()),
                "text": self.spoken_text()}


#: A sink returns True when it actually presented the reminder to the user.
Sink = Callable[[ReminderEvent], bool]


class ReminderNotifier:
    """Fans a due reminder out to every registered sink.

    Sinks are added by whichever interface is running -- the dashboard adds a
    banner, the voice loop adds speech, the CLI adds a printed line.  The
    scheduler knows about none of them, which is what keeps it testable and
    keeps a headless run from depending on a GUI.
    """

    def __init__(self, sinks: Optional[List[Sink]] = None) -> None:
        self._sinks: List[Sink] = list(sinks or [])
        self._lock = threading.RLock()

    def add_sink(self, sink: Sink) -> None:
        with self._lock:
            self._sinks.append(sink)

    def remove_sink(self, sink: Sink) -> None:
        with self._lock:
            if sink in self._sinks:
                self._sinks.remove(sink)

    @property
    def sink_count(self) -> int:
        with self._lock:
            return len(self._sinks)

    def notify(self, event: ReminderEvent) -> bool:
        """Return True if at least one sink accepted the reminder.

        A sink that raises is logged and treated as having declined: one broken
        front end must not stop the others from being told, and must not lose
        the reminder either.
        """
        with self._lock:
            sinks = list(self._sinks)
        delivered = False
        for sink in sinks:
            try:
                delivered = bool(sink(event)) or delivered
            except Exception:  # noqa: BLE001 - a sink fault is not a lost reminder
                logger.exception("Reminder sink raised",
                                 extra={"context": {"id": event.reminder_id}})
        return delivered


class ReminderScheduler:
    """Watches the store and rings what is due.

    The background thread is a thin wrapper around :meth:`poll`; everything
    that decides anything lives in ``poll``, which is synchronous and takes its
    time from the injected clock.  That is what lets the whole policy -- due,
    late, missed, undelivered -- be tested without a single real second
    passing.
    """

    def __init__(self, database: Any, *,
                 notifier: Optional[ReminderNotifier] = None,
                 clock: Callable[[], datetime] = local_now,
                 interval: float = POLL_SECONDS,
                 late_grace: timedelta = LATE_GRACE) -> None:
        self._database = database
        self.notifier = notifier if notifier is not None else ReminderNotifier()
        self._clock = clock
        self._interval = float(interval)
        self._late_grace = late_grace
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._attempts: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._started_at: Optional[datetime] = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> bool:
        """Begin polling on a daemon thread.  Idempotent.

        Daemon, because a scheduler must never be the reason the application
        refuses to exit; the loop holds no state that would be lost.
        """
        with self._lock:
            if self.running:
                return False
            self._stop.clear()
            self._started_at = self._clock()
            self._thread = threading.Thread(
                target=self._loop, name="aarma-reminders", daemon=True)
            self._thread.start()
        logger.info("Reminder scheduler started",
                    extra={"context": {"interval": self._interval}})
        return True

    def stop(self, timeout: float = 2.0) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
        self._stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        logger.info("Reminder scheduler stopped")

    def _loop(self) -> None:  # pragma: no cover - exercised via start/stop
        # Wait *first* so ``stop`` right after ``start`` costs nothing, then
        # poll; Event.wait returns immediately when the flag is set, which is
        # why this never sleeps through a shutdown.
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception:  # noqa: BLE001 - a bad poll must not kill the thread
                logger.exception("Reminder poll failed")
            self._stop.wait(self._interval)

    # -- the policy --------------------------------------------------------

    @staticmethod
    def _parse(value: str) -> Optional[datetime]:
        try:
            moment = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None
        # A stored time without an offset would be "some time, somewhere".
        # Attaching the local zone is the only reading that matches what the
        # user was looking at when they set it.
        return moment if moment.tzinfo is not None else moment.astimezone()

    def due_now(self, now: Optional[datetime] = None) -> List[ReminderEvent]:
        """Every pending reminder whose time has arrived.  Changes nothing."""
        moment = now or self._clock()
        events: List[ReminderEvent] = []
        try:
            rows = self._database.reminders(state="pending")
        except Exception:  # noqa: BLE001
            logger.exception("Could not read reminders")
            return []
        for row in rows:
            due = self._parse(row["due_at"])
            if due is None:
                logger.warning("Reminder has an unreadable due time",
                               extra={"context": {"id": row["id"]}})
                continue
            if due > moment:
                continue
            events.append(ReminderEvent(
                reminder_id=row["id"], title=row["title"], due_at=due,
                now=moment, missed=(moment - due) > self._late_grace))
        return events

    def poll(self, now: Optional[datetime] = None) -> List[ReminderEvent]:
        """One pass: fire what is due, close out what cannot be delivered.

        Returns the events that were *handled* this pass, so a caller (and the
        test-suite) can see exactly what happened without reading the database.
        """
        moment = now or self._clock()
        handled: List[ReminderEvent] = []
        for event in self.due_now(moment):
            if event.missed:
                # Deliberately not rung. Reported once so the user knows, then
                # closed so it cannot be reported again on the next startup.
                self.notifier.notify(event)
                self._close(event, Outcome.MISSED)
                handled.append(event)
                continue

            if self.notifier.notify(event):
                self._close(event, Outcome.LATE if event.late else Outcome.FIRED)
                self._attempts.pop(event.reminder_id, None)
                handled.append(event)
                continue

            # Nobody took it. Leave it pending and try again next pass -- the
            # assistant may be speaking, or no interface may be listening yet.
            attempts = self._attempts.get(event.reminder_id, 0) + 1
            self._attempts[event.reminder_id] = attempts
            if attempts >= MAX_DELIVERY_ATTEMPTS:
                logger.warning("Reminder could not be delivered",
                               extra={"context": {"id": event.reminder_id,
                                                  "attempts": attempts}})
                self._close(event, Outcome.UNDELIVERED)
                self._attempts.pop(event.reminder_id, None)
                handled.append(event)
        return handled

    def _close(self, event: ReminderEvent, outcome: str) -> None:
        try:
            self._database.finish_reminder(event.reminder_id, outcome=outcome)
        except Exception:  # noqa: BLE001
            logger.exception("Could not close out reminder",
                             extra={"context": {"id": event.reminder_id}})
            return
        logger.info("Reminder closed", extra={"context": {
            "id": event.reminder_id, "outcome": outcome}})

    # -- reporting ---------------------------------------------------------

    def next_due(self, now: Optional[datetime] = None) -> Optional[str]:
        """How long until the next pending reminder, in words."""
        moment = now or self._clock()
        try:
            rows = self._database.reminders(state="pending")
        except Exception:  # noqa: BLE001
            return None
        upcoming = []
        for row in rows:
            due = self._parse(row["due_at"])
            if due is not None and due > moment:
                upcoming.append(due)
        if not upcoming:
            return None
        return until_text(min(upcoming), reference=moment)


__all__ = ["LATE_AFTER", "LATE_GRACE", "MAX_DELIVERY_ATTEMPTS", "Outcome",
           "POLL_SECONDS", "ReminderEvent", "ReminderNotifier",
           "ReminderScheduler", "Sink"]
