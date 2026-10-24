"""Local reminders: create, list, cancel -- and, since Phase 3, actually ring.

Stored in AARMA's own database with a timezone offset, so a reminder set for
6 PM means 6 PM on the clock the user was looking at, and survives a restart.

These handlers still do not deliver anything themselves.  Ringing is
:class:`~core.scheduler.ReminderScheduler`, which polls the same table on a
background thread; the split is deliberate, because storage has to be correct
whether or not anything is listening, and a handler that also owned delivery
would make that impossible to test.  What changed in Phase 3 is that something
*is* listening -- so :func:`_create_reminder` can now promise to tell the user,
which it could not honestly do before.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from actions.registry import ActionContext, ActionResult, ActionSpec, ParameterSpec
from security.validator import RiskLevel, UnsafeInputError, ensure_text
from utils.logger import get_logger
from utils.when import describe, local_now, parse_when, until_text

logger = get_logger("actions.reminders")

MAX_TITLE = 120


def _parse_stored(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _clock_forms(moment: Optional[datetime]) -> set:
    """Every way someone might say a reminder's time.

    "6 PM", "6:00 pm" and "18:00" are the same instant, and a user cancelling
    "my 6 PM reminder" will not say the one the database happens to store.
    """
    if moment is None:
        return set()
    hour24 = moment.hour
    hour12 = hour24 % 12 or 12
    meridiem = "am" if hour24 < 12 else "pm"
    minute = moment.minute
    forms = {
        f"{hour12} {meridiem}", f"{hour12}{meridiem}",
        f"{hour12}:{minute:02d} {meridiem}", f"{hour12}:{minute:02d}{meridiem}",
        f"{hour24}:{minute:02d}",
    }
    if minute == 0:
        forms.add(str(hour12))
    return {f.lower() for f in forms}


def _matches_reminder(row, wanted: str) -> bool:
    """Whether `wanted` plausibly names this reminder."""
    if wanted == row["id"]:
        return True
    if wanted in (row["title"] or "").lower():
        return True
    moment = _parse_stored(row["due_at"])
    if moment is not None:
        spoken = describe(moment).lower()
        if wanted in spoken:
            return True
        # "6 pm" against a stored 18:00.
        for form in _clock_forms(moment):
            if form in wanted or wanted in form:
                return True
    return wanted in (row["due_at"] or "").lower()

def _create_reminder(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    raw_when = str(params.get("when") or "").strip()
    raw_title = str(params.get("title") or "").strip()

    parsed = parse_when(raw_when) if raw_when else None
    if parsed is None:
        return ActionResult.failure(
            "I couldn't work out when you meant. Try something like "
            "\"at 6 PM\", \"tomorrow at 9\" or \"in 30 minutes\".",
            error_code="unparsed_time")
    if parsed.when <= local_now():
        return ActionResult.failure(
            f"{describe(parsed.when)} has already passed.",
            error_code="time_in_past")

    if not raw_title:
        # The time is understood but not the subject. Say what is missing
        # rather than storing a reminder that says nothing.
        return ActionResult.failure(
            f"What should I remind you about {describe(parsed.when)}?",
            error_code="missing_title", needs="title",
            due_at=parsed.when.isoformat(timespec="seconds"))

    try:
        title = ensure_text(raw_title, field="reminder", max_length=MAX_TITLE)
    except UnsafeInputError as exc:
        return ActionResult.failure(str(exc), error_code="invalid_title")

    reminder_id = uuid.uuid4().hex[:12]
    try:
        ctx.database.add_reminder(reminder_id=reminder_id, title=title,
                                  due_at=parsed.when.isoformat(timespec="seconds"))
    except Exception as exc:  # noqa: BLE001 - a storage failure is not a crash
        logger.error("Reminder could not be stored", extra={
            "context": {"error": str(exc)}})
        return ActionResult.failure("I couldn't save that reminder.",
                                    error_code="storage_failed")

    logger.info("Reminder created", extra={"context": {
        "id": reminder_id, "due_at": parsed.when.isoformat(timespec="seconds")}})
    return ActionResult.success(
        f"I'll remind you to {title} {describe(parsed.when)} "
        f"({until_text(parsed.when)}).",
        reminder_id=reminder_id, title=title,
        due_at=parsed.when.isoformat(timespec="seconds"))


def _list_reminders(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    try:
        rows = ctx.database.reminders(state="pending")
    except Exception as exc:  # noqa: BLE001
        logger.error("Reminders could not be read", extra={
            "context": {"error": str(exc)}})
        return ActionResult.failure("I couldn't read your reminders.",
                                    error_code="storage_failed")
    if not rows:
        return ActionResult.success("You have no reminders set.",
                                    reminders=[], count=0)

    listed = []
    lines = []
    for row in rows:
        due = _parse_stored(row["due_at"])
        listed.append({"id": row["id"], "title": row["title"],
                       "due_at": row["due_at"]})
        lines.append(f"{row['title']} — {describe(due) if due else row['due_at']}")
    heading = ("You have one reminder: " if len(lines) == 1
               else f"You have {len(lines)} reminders: ")
    return ActionResult.success(heading + "; ".join(lines) + ".",
                                reminders=listed, count=len(listed))


def _cancel_reminder(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    wanted = str(params.get("which") or "").strip().lower()
    try:
        rows = ctx.database.reminders(state="pending")
    except Exception:  # noqa: BLE001
        return ActionResult.failure("I couldn't read your reminders.",
                                    error_code="storage_failed")
    if not rows:
        return ActionResult.failure("You have no reminders to cancel.",
                                    error_code="nothing_to_cancel")

    matches = list(rows)
    if wanted:
        matches = [r for r in rows if _matches_reminder(r, wanted)]
    if not matches:
        return ActionResult.failure(
            f"I couldn't find a reminder matching \"{wanted}\".",
            error_code="no_match")
    if len(matches) > 1:
        titles = ", ".join(r["title"] for r in matches[:4])
        return ActionResult.failure(
            f"I found {len(matches)} reminders matching that: {titles}. "
            "Which one?", error_code="ambiguous",
            candidates=[{"id": r["id"], "title": r["title"]} for r in matches])

    row = matches[0]
    if not ctx.database.set_reminder_state(row["id"], "cancelled"):
        return ActionResult.failure("That reminder was already gone.",
                                    error_code="nothing_to_cancel")
    return ActionResult.success(f"Cancelled the reminder to {row['title']}.",
                                reminder_id=row["id"], title=row["title"])


def _missed_reminders(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Reminders whose time passed while AARMA was not running.

    The scheduler closes these out without sounding an alarm -- a reminder from
    last Tuesday ringing now reads as something due now -- but it does not
    discard them, because "I quietly dropped it" is the behaviour that makes a
    reminder untrustworthy.  This is where they surface.
    """
    try:
        rows = ctx.database.reminders_by_outcome("missed")
    except Exception as exc:  # noqa: BLE001
        logger.error("Missed reminders could not be read", extra={
            "context": {"error": str(exc)}})
        return ActionResult.failure("I couldn't read your reminders.",
                                    error_code="storage_failed")
    if not rows:
        return ActionResult.success("You haven't missed any reminders.",
                                    reminders=[], count=0)
    listed = []
    lines = []
    for row in rows:
        due = _parse_stored(row["due_at"])
        listed.append({"id": row["id"], "title": row["title"],
                       "due_at": row["due_at"]})
        lines.append(f"{row['title']} — was due {describe(due) if due else row['due_at']}")
    heading = ("You missed one reminder: " if len(lines) == 1
               else f"You missed {len(lines)} reminders: ")
    return ActionResult.success(heading + "; ".join(lines) + ".",
                                reminders=listed, count=len(listed))


def _due_reminders(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Anything whose time has arrived, for a caller that wants to announce it."""
    now = local_now()
    try:
        rows = ctx.database.reminders(state="pending")
    except Exception:  # noqa: BLE001
        return ActionResult.failure("I couldn't read your reminders.",
                                    error_code="storage_failed")
    due = []
    for row in rows:
        moment = _parse_stored(row["due_at"])
        if moment is not None and moment <= now:
            due.append({"id": row["id"], "title": row["title"],
                        "due_at": row["due_at"]})
    if not due:
        return ActionResult.success("Nothing is due.", reminders=[], count=0)
    titles = "; ".join(item["title"] for item in due)
    return ActionResult.success(f"Due now: {titles}.", reminders=due,
                                count=len(due))


def get_actions() -> List[ActionSpec]:
    return [
        ActionSpec(
            name="create_reminder",
            description="Set a reminder for a time you name.",
            category="Reminders", risk=RiskLevel.LOW, handler=_create_reminder,
            parameters=(
                ParameterSpec(name="when", description="when to remind you",
                              max_length=80),
                ParameterSpec(name="title", description="what to remind you about",
                              required=False, default="", max_length=MAX_TITLE),
            ),
            examples=("remind me at 6 PM to submit my project",
                      "remind me tomorrow at 9 AM to call Mom",
                      "remind me in 30 minutes to check the oven"),
        ),
        ActionSpec(
            name="list_reminders", description="Show the reminders you've set.",
            category="Reminders", risk=RiskLevel.LOW, handler=_list_reminders,
            examples=("what reminders do I have", "show my reminders"),
        ),
        ActionSpec(
            name="cancel_reminder", description="Cancel a reminder.",
            category="Reminders", risk=RiskLevel.MEDIUM, handler=_cancel_reminder,
            parameters=(ParameterSpec(name="which", required=False, default="",
                                      description="which reminder", max_length=80),),
            examples=("cancel my 6 PM reminder", "cancel the reminder about the bank"),
        ),
        ActionSpec(
            name="due_reminders", description="Show reminders that are due now.",
            category="Reminders", risk=RiskLevel.LOW, handler=_due_reminders,
            examples=("is anything due", "any reminders due"),
        ),
        ActionSpec(
            name="missed_reminders",
            description="Show reminders that came due while I wasn't running.",
            category="Reminders", risk=RiskLevel.LOW, handler=_missed_reminders,
            examples=("did I miss any reminders", "show missed reminders"),
        ),
    ]
