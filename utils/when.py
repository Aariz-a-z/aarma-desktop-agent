"""Turning "tomorrow at 9" into an actual moment.

Everything here is timezone-aware. A reminder is a promise about a wall clock,
and a naive ``datetime`` is a promise about nothing in particular -- it survives
being stored, compared and rescheduled while quietly meaning a different instant
than the one the user said. So the parser always attaches the system's local
zone, and callers store the result with its offset.

Deliberately small. It understands the handful of shapes people actually speak
to an assistant -- a clock time, a day plus a clock time, a relative delay --
and returns ``None`` for anything else rather than guessing. A reminder set for
the wrong time is worse than one that was never set, because the user believes
it exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Optional

#: Weekday names, Monday first, matching ``datetime.weekday()``.
_WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}

_UNITS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
    "day": 86400, "days": 86400,
    "week": 604800, "weeks": 604800,
}

#: "half past six" and friends are common in speech but ambiguous often enough
#: that they are left out; "6:30" and "half an hour" are both supported.
_CLOCK = re.compile(
    r"\b(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<meridiem>a\.?m\.?|p\.?m\.?)?\b",
    re.IGNORECASE)

_IN_DELAY = re.compile(
    r"\bin\s+(?P<count>\d+|a|an|half\s+an?)\s*(?P<unit>second|seconds|sec|secs|"
    r"minute|minutes|min|mins|hour|hours|hr|hrs|day|days|week|weeks)\b",
    re.IGNORECASE)

_AT_NOON = re.compile(r"\b(?P<word>noon|midday|midnight)\b", re.IGNORECASE)


def local_now() -> datetime:
    """The current moment, with the system's zone attached."""
    return datetime.now().astimezone()


@dataclass(frozen=True)
class ParsedTime:
    """A moment, and how confident the parse was.

    ``assumed_time`` records that the user named a day but no clock time, so
    the caller can ask rather than silently picking one.
    """

    when: datetime
    text: str = ""
    assumed_time: bool = False
    assumed_date: bool = False

    @property
    def is_past(self) -> bool:
        return self.when <= local_now()


def _meridiem_hour(hour: int, meridiem: Optional[str], *, reference: datetime,
                   explicit_day: bool) -> int:
    """Resolve a spoken hour to 24-hour form.

    With no am/pm, pick the reading a person would mean: the next occurrence.
    "Remind me at 6" said in the morning means this evening far more often than
    it means six minutes from now -- but only when the user did not name a day.
    If they said "tomorrow at 6", tomorrow morning is the natural reading.
    """
    if meridiem:
        pm = meridiem.lower().startswith("p")
        if hour == 12:
            return 12 if pm else 0
        return hour + 12 if pm else hour
    if hour == 12:
        return 12
    if hour > 12:
        return hour
    if explicit_day:
        return hour
    # Ambiguous and no day given: prefer the interpretation still ahead.
    candidate = reference.replace(hour=hour, minute=0, second=0, microsecond=0)
    if candidate <= reference and hour < 12:
        return hour + 12
    return hour


def parse_when(text: str, *, reference: Optional[datetime] = None
               ) -> Optional[ParsedTime]:
    """Best-effort parse of a spoken time.  ``None`` when nothing is recognised.

    Understood shapes::

        in 30 minutes / in an hour / in half an hour
        at 6 / at 6pm / at 18:30
        tomorrow / tomorrow at 9am
        tonight / this evening / this afternoon / this morning
        friday / on friday at 4pm / next monday
        noon / midday / midnight
    """
    if not text:
        return None
    now = (reference or local_now())
    if now.tzinfo is None:
        now = now.astimezone()
    lowered = " ".join(str(text).lower().split())

    # --- relative delay -------------------------------------------------
    delay = _IN_DELAY.search(lowered)
    if delay:
        raw = delay.group("count")
        unit = _UNITS[delay.group("unit").lower()]
        if raw in ("a", "an"):
            count: float = 1
        elif raw.startswith("half"):
            count = 0.5
        else:
            count = int(raw)
        return ParsedTime(when=now + timedelta(seconds=count * unit),
                          text=delay.group(0))

    # --- which day ------------------------------------------------------
    day = now
    explicit_day = False
    day_offset: Optional[int] = None
    named_weekday = False

    if re.search(r"\bday after tomorrow\b", lowered):
        day_offset, explicit_day = 2, True
    elif re.search(r"\btomorrow\b", lowered):
        day_offset, explicit_day = 1, True
    elif re.search(r"\btoday\b|\btonight\b|\bthis (?:morning|afternoon|evening)\b",
                   lowered):
        day_offset, explicit_day = 0, True
    else:
        wants_next = bool(re.search(r"\bnext\b", lowered))
        for name, index in _WEEKDAYS.items():
            if not re.search(rf"\b{name}\b", lowered):
                continue
            ahead = (index - now.weekday()) % 7
            if wants_next:
                # "next Friday" is always the coming one, never today.
                ahead = ahead or 7
                if ahead < 7 and index == now.weekday():
                    ahead = 7
            elif ahead == 0:
                # The day named is today. Whether that means today or a week
                # away depends on the hour, which is not known yet, so this
                # keeps today and the clock step rolls it forward if the time
                # has already passed.
                ahead = 0
            day_offset, explicit_day = ahead, True
            named_weekday = not wants_next
            break

    if day_offset is not None:
        day = now + timedelta(days=day_offset)

    # --- what time ------------------------------------------------------
    named = _AT_NOON.search(lowered)
    if named:
        word = named.group("word").lower()
        clock = time(0, 0) if word == "midnight" else time(12, 0)
        when = day.replace(hour=clock.hour, minute=clock.minute, second=0,
                           microsecond=0)
        if when <= now and day_offset is None:
            when += timedelta(days=1)
        return ParsedTime(when=when, text=named.group(0))

    # Ignore a clock match that is really part of a date or a duration.
    candidate = None
    for match in _CLOCK.finditer(lowered):
        hour = int(match.group("hour"))
        if hour > 23:
            continue
        # "the 25th" and bare years are not clock times.
        after = lowered[match.end():match.end() + 3]
        if after.startswith(("st", "nd", "rd", "th")):
            continue
        candidate = match
        break

    if candidate is None:
        if day_offset is None:
            return None
        # A day with no time: the caller should ask rather than assume.
        when = day.replace(hour=9, minute=0, second=0, microsecond=0)
        if re.search(r"\btonight\b|\bthis evening\b", lowered):
            when = day.replace(hour=19, minute=0, second=0, microsecond=0)
        elif re.search(r"\bthis afternoon\b", lowered):
            when = day.replace(hour=14, minute=0, second=0, microsecond=0)
        return ParsedTime(when=when, text=lowered, assumed_time=True)

    hour = _meridiem_hour(int(candidate.group("hour")),
                          candidate.group("meridiem"), reference=now,
                          explicit_day=explicit_day)
    minute = int(candidate.group("minute") or 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    # "tonight at 8" means 20:00 even though the user said 8.
    if (not candidate.group("meridiem")
            and re.search(r"\btonight\b|\bthis evening\b", lowered)
            and hour < 12):
        hour += 12

    when = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if when <= now and day_offset is None:
        when += timedelta(days=1)
    # "Friday at 4 PM" said on Friday morning means today, not next week: the
    # named day has already arrived and the hour has not. Rolling forward a
    # week here would silently miss the reminder the user actually wanted.
    if named_weekday and day_offset == 0 and when <= now:
        # They named today's weekday but the hour has gone: they meant next week.
        when += timedelta(days=7)
    return ParsedTime(when=when, text=candidate.group(0),
                      assumed_date=day_offset is None)


def describe(moment: datetime, *, reference: Optional[datetime] = None) -> str:
    """"tomorrow at 10:00 AM" -- how a person would say it back."""
    now = reference or local_now()
    if moment.tzinfo is None:
        moment = moment.astimezone()
    days = (moment.date() - now.date()).days
    clock = moment.strftime("%I:%M %p").lstrip("0")
    if days == 0:
        return f"today at {clock}"
    if days == 1:
        return f"tomorrow at {clock}"
    if 2 <= days <= 6:
        return f"{moment.strftime('%A')} at {clock}"
    return f"{moment.strftime('%d %B')} at {clock}"


def until_text(moment: datetime, *, reference: Optional[datetime] = None) -> str:
    """"in 2 hours 15 minutes", or "now" when it has already passed."""
    now = reference or local_now()
    if moment.tzinfo is None:
        moment = moment.astimezone()
    delta = moment - now
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "now"
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes = seconds // 60
    parts = []
    if days:
        parts.append(f"{days} day" + ("s" if days != 1 else ""))
    if hours:
        parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if minutes and not days:
        parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
    return "in " + " ".join(parts) if parts else "in under a minute"
