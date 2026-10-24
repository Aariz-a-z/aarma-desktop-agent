"""Calendars, starting with the only one AARMA can honestly claim: its own.

The interface exists so that Google and Outlook can be added later without the
actions above them changing. They are *not* implemented, and the placeholders
below say so when asked rather than pretending.

That distinction is the reason this file is shaped this way. "Your meeting has
been added to Google Calendar" when no account is connected is the single most
damaging thing an assistant can say, because the user stops checking and misses
the meeting.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, List, Optional

from providers import Availability, Provider, ProviderResult, UnavailableProvider
from utils.logger import get_logger
from utils.when import describe

logger = get_logger("providers.calendar")

#: How long a meeting lasts when the user does not say.
DEFAULT_DURATION_MINUTES = 60


@dataclass(frozen=True)
class CalendarEvent:
    """One entry, as the calendar stores it."""

    id: str
    title: str
    starts_at: datetime
    ends_at: datetime
    location: str = ""
    notes: str = ""
    provider: str = "local"

    def describe(self) -> str:
        return f"{self.title} {describe(self.starts_at)}"

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title,
                "starts_at": self.starts_at.isoformat(timespec="seconds"),
                "ends_at": self.ends_at.isoformat(timespec="seconds"),
                "location": self.location, "notes": self.notes,
                "provider": self.provider}


class CalendarProvider(Provider):
    """What every calendar must be able to do."""

    name = "calendar"
    description = "create and list calendar events"

    def create_event(self, *, title: str, starts_at: datetime,
                     ends_at: Optional[datetime] = None, location: str = "",
                     notes: str = "") -> ProviderResult:
        raise NotImplementedError

    def list_events(self, *, since: Optional[datetime] = None,
                    until: Optional[datetime] = None,
                    limit: int = 50) -> List[CalendarEvent]:
        raise NotImplementedError

    def cancel_event(self, event_id: str) -> ProviderResult:
        raise NotImplementedError


class LocalCalendarProvider(CalendarProvider):
    """AARMA's own calendar, in AARMA's own database.

    Entirely offline and entirely honest: an event created here exists here,
    and the wording the actions use says "your AARMA calendar" rather than
    implying it reached anything else.
    """

    name = "local-calendar"
    description = "AARMA's own offline calendar"

    def __init__(self, database: Any) -> None:
        self._database = database

    def availability(self) -> Availability:
        if self._database is None:
            return Availability.missing(
                self.name, "AARMA's calendar store isn't open.")
        return Availability.ready(self.name, "AARMA's own offline calendar")

    @staticmethod
    def _parse(value: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(value)
        except (TypeError, ValueError):
            return None

    def create_event(self, *, title: str, starts_at: datetime,
                     ends_at: Optional[datetime] = None, location: str = "",
                     notes: str = "") -> ProviderResult:
        if self._database is None:
            return ProviderResult.unavailable(self.availability())
        finish = ends_at or (starts_at + timedelta(minutes=DEFAULT_DURATION_MINUTES))
        if finish <= starts_at:
            return ProviderResult.failure("An event has to end after it starts.",
                                          error_code="invalid_range")
        event_id = uuid.uuid4().hex[:12]
        try:
            self._database.add_event(
                event_id=event_id, title=title,
                starts_at=starts_at.isoformat(timespec="seconds"),
                ends_at=finish.isoformat(timespec="seconds"),
                location=location, notes=notes, provider="local")
        except Exception as exc:  # noqa: BLE001
            logger.error("Event could not be stored", extra={
                "context": {"error": str(exc)}})
            return ProviderResult.failure("I couldn't save that event.",
                                          error_code="storage_failed")
        logger.info("Calendar event created", extra={"context": {"id": event_id}})
        return ProviderResult.success(
            "Event created.", event_id=event_id, title=title,
            starts_at=starts_at.isoformat(timespec="seconds"),
            ends_at=finish.isoformat(timespec="seconds"))

    def list_events(self, *, since: Optional[datetime] = None,
                    until: Optional[datetime] = None,
                    limit: int = 50) -> List[CalendarEvent]:
        if self._database is None:
            return []
        try:
            rows = self._database.events(
                since=since.isoformat(timespec="seconds") if since else None,
                until=until.isoformat(timespec="seconds") if until else None,
                limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.error("Events could not be read", extra={
                "context": {"error": str(exc)}})
            return []
        events: List[CalendarEvent] = []
        for row in rows:
            starts, ends = self._parse(row["starts_at"]), self._parse(row["ends_at"])
            if starts is None or ends is None:
                continue
            events.append(CalendarEvent(
                id=row["id"], title=row["title"], starts_at=starts, ends_at=ends,
                location=row["location"] or "", notes=row["notes"] or "",
                provider=row["provider"] or "local"))
        return events

    def cancel_event(self, event_id: str) -> ProviderResult:
        if self._database is None:
            return ProviderResult.unavailable(self.availability())
        if self._database.cancel_event(event_id):
            return ProviderResult.success("Event cancelled.", event_id=event_id)
        return ProviderResult.failure("I couldn't find that event.",
                                      error_code="not_found")


def google_calendar() -> Provider:
    """Placeholder.  Google Calendar is not connected in this build."""
    return UnavailableProvider(
        "google-calendar",
        "Google Calendar isn't connected.",
        "I can put it in AARMA's own calendar instead.")


def outlook_calendar() -> Provider:
    """Placeholder.  Outlook is not connected in this build."""
    return UnavailableProvider(
        "outlook-calendar",
        "Outlook isn't connected.",
        "I can put it in AARMA's own calendar instead.")
