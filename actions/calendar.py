"""Calendar actions, against whichever provider is connected.

Today that is AARMA's own local calendar and nothing else, so every reply says
so. When a Google or Outlook provider is added, these handlers do not change --
they ask :func:`active_provider` for a calendar and report what it tells them.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from actions.registry import ActionContext, ActionResult, ActionSpec, ParameterSpec
from providers.calendar import (
    DEFAULT_DURATION_MINUTES,
    CalendarProvider,
    LocalCalendarProvider,
)
from security.validator import RiskLevel, UnsafeInputError, ensure_text
from utils.logger import get_logger
from utils.when import describe, local_now, parse_when

logger = get_logger("actions.calendar")

MAX_TITLE = 120


def active_provider(ctx: ActionContext) -> CalendarProvider:
    """The calendar in use.

    A test may inject one through the context; otherwise it is the local
    calendar, which is the only one this build can actually write to.
    """
    injected = getattr(ctx, "calendar_provider", None)
    if injected is not None:
        return injected
    return LocalCalendarProvider(ctx.database)


def _create_event(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    raw_when = str(params.get("when") or "").strip()
    raw_title = str(params.get("title") or "").strip()

    parsed = parse_when(raw_when) if raw_when else None
    if parsed is None:
        return ActionResult.failure(
            "When should I schedule it? Try \"tomorrow at 10 AM\" or "
            "\"Friday at 3 PM\".",
            error_code="unparsed_time", needs="when")
    if parsed.assumed_time:
        # "Schedule a meeting tomorrow" names a day and no hour. The parser
        # falls back to 9 AM so *something* is returned, but booking a meeting
        # at a time the user never said is exactly the kind of confident guess
        # that gets someone to the wrong place. Ask instead.
        day = parsed.when.strftime("%A")
        return ActionResult.failure(
            f"What time on {day}?",
            error_code="missing_time", needs="when",
            day=day, assumed=parsed.when.isoformat(timespec="seconds"))
    if parsed.when <= local_now():
        return ActionResult.failure(
            f"{describe(parsed.when)} has already passed.",
            error_code="time_in_past", needs="when")
    if not raw_title:
        # The commonest half-said command: a time but no subject.
        return ActionResult.failure(
            f"What should I call the event {describe(parsed.when)}?",
            error_code="missing_title", needs="title",
            starts_at=parsed.when.isoformat(timespec="seconds"))

    try:
        title = ensure_text(raw_title, field="event", max_length=MAX_TITLE)
    except UnsafeInputError as exc:
        return ActionResult.failure(str(exc), error_code="invalid_title")

    duration = DEFAULT_DURATION_MINUTES
    raw_minutes = params.get("minutes")
    if raw_minutes:
        try:
            duration = max(5, min(24 * 60, int(raw_minutes)))
        except (TypeError, ValueError):
            duration = DEFAULT_DURATION_MINUTES

    provider = active_provider(ctx)
    if not provider.is_available():
        availability = provider.availability()
        return ActionResult.failure(availability.detail or "No calendar is available.",
                                    error_code="unavailable")

    result = provider.create_event(
        title=title, starts_at=parsed.when,
        ends_at=parsed.when + timedelta(minutes=duration),
        location=str(params.get("location") or ""))
    if not result.ok:
        return ActionResult.failure(result.message,
                                    error_code=result.error_code or "calendar_failed")

    # Name the calendar it actually went into. "Scheduled" on its own invites
    # the assumption that it reached whatever calendar the user normally uses.
    where = ("AARMA's own calendar" if provider.name == "local-calendar"
             else provider.name)
    return ActionResult.success(
        f"Scheduled \"{title}\" for {describe(parsed.when)} in {where}.",
        **result.data)


def _list_events(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = active_provider(ctx)
    if not provider.is_available():
        return ActionResult.failure(provider.availability().detail,
                                    error_code="unavailable")

    now = local_now()
    horizon = str(params.get("range") or "").strip().lower()
    if "tomorrow" in horizon:
        start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                  microsecond=0)
        finish = start + timedelta(days=1)
        label = "tomorrow"
    elif "week" in horizon:
        start, finish, label = now, now + timedelta(days=7), "this week"
    elif "today" in horizon:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        finish = start + timedelta(days=1)
        label = "today"
    else:
        start, finish, label = now, now + timedelta(days=7), "coming up"

    events = provider.list_events(since=start, until=finish)
    if not events:
        return ActionResult.success(f"Nothing {label}.", events=[], count=0)
    listed = "; ".join(f"{e.title} — {describe(e.starts_at, reference=now)}"
                       for e in events)
    return ActionResult.success(
        f"{label.capitalize()}: {listed}.",
        events=[e.as_dict() for e in events], count=len(events))


def _cancel_event(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    wanted = str(params.get("which") or "").strip().lower()
    provider = active_provider(ctx)
    if not provider.is_available():
        return ActionResult.failure(provider.availability().detail,
                                    error_code="unavailable")
    events = provider.list_events(since=local_now() - timedelta(days=1),
                                  until=local_now() + timedelta(days=365))
    matches = [e for e in events if not wanted or wanted in e.title.lower()
               or wanted == e.id]
    if not matches:
        return ActionResult.failure("I couldn't find that event.",
                                    error_code="no_match")
    if len(matches) > 1:
        titles = ", ".join(e.title for e in matches[:4])
        return ActionResult.failure(
            f"I found {len(matches)} events matching that: {titles}. Which one?",
            error_code="ambiguous",
            candidates=[e.as_dict() for e in matches])
    result = provider.cancel_event(matches[0].id)
    if not result.ok:
        return ActionResult.failure(result.message, error_code="calendar_failed")
    return ActionResult.success(f"Cancelled \"{matches[0].title}\".",
                                **result.data)


def get_actions() -> List[ActionSpec]:
    return [
        ActionSpec(
            name="create_event",
            description="Put a meeting or event in the calendar.",
            category="Calendar", risk=RiskLevel.MEDIUM, handler=_create_event,
            parameters=(
                ParameterSpec(name="when", description="when it starts",
                              max_length=80),
                ParameterSpec(name="title", description="what it's called",
                              required=False, default="", max_length=MAX_TITLE),
                ParameterSpec(name="minutes", description="how long, in minutes",
                              required=False, default=None),
                ParameterSpec(name="location", required=False, default="",
                              description="where", max_length=120),
            ),
            examples=("schedule a meeting tomorrow at 10 AM",
                      "create an event called project review tomorrow at 2 PM"),
        ),
        ActionSpec(
            name="list_events", description="Show what's in the calendar.",
            category="Calendar", risk=RiskLevel.LOW, handler=_list_events,
            parameters=(ParameterSpec(name="range", required=False, default="",
                                      description="today, tomorrow or this week",
                                      max_length=40),),
            examples=("show my schedule", "what do I have tomorrow",
                      "open my calendar"),
        ),
        ActionSpec(
            name="cancel_event", description="Cancel a calendar event.",
            category="Calendar", risk=RiskLevel.MEDIUM, handler=_cancel_event,
            parameters=(ParameterSpec(name="which", required=False, default="",
                                      description="which event", max_length=80),),
            examples=("cancel the project review", "cancel my 10 AM meeting"),
        ),
    ]
