"""Weather actions.

Nothing here knows what Open-Meteo is.  The handlers ask a
:class:`~providers.weather.WeatherProvider` for structured data and put it into
a sentence; swapping the source is a change to :mod:`providers.weather` alone.
That separation is the point of the provider layer, and it is also what keeps
the parsing rules in :mod:`ai.intent` free of anything service-specific.

The one rule these handlers enforce for themselves: **a missing figure is
omitted, never filled in.**  :meth:`~providers.weather.WeatherReading.summary`
builds its sentence from the fields that came back, so a response with no
humidity produces a sentence with no humidity rather than a plausible number.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from actions.registry import (
    ActionContext,
    ActionResult,
    ActionSpec,
    ParameterSpec,
    ParameterType,
)
from providers.weather import WeatherProvider, build_weather_provider
from security.permissions import Permission
from security.validator import RiskLevel
from utils.logger import get_logger

logger = get_logger("actions.weather")

#: Words that mean "where I already am" rather than naming a place.  Stripped
#: so "the weather here" falls back to the configured location instead of
#: being geocoded as a town called "here".
_HERE = frozenset({"here", "outside", "today", "right now", "now", "my area",
                   "where i am", "local", "locally"})


def provider_for(ctx: ActionContext) -> WeatherProvider:
    """The weather provider in use, overridable by a test."""
    injected = getattr(ctx, "weather_provider", None)
    if injected is not None:
        return injected
    return build_weather_provider(ctx.config, online=bool(ctx.online))


def _clean_location(raw: object) -> str:
    """The place the user named, or "" meaning "wherever I'm configured to be".

    Returning "" for "here" matters: it lets the provider fall back to the
    configured location and, when there isn't one, say so -- rather than
    geocoding a town called "here" and answering about somewhere else entirely.
    """
    words = str(raw or "").split()
    while words and words[0].lower() in {"in", "at", "for", "the"}:
        words = words[1:]
    if not words:
        return ""
    return "" if " ".join(words).lower() in _HERE else " ".join(words)


def _unavailable(provider: WeatherProvider) -> ActionResult:
    availability = provider.availability()
    text = availability.detail or "I can't access weather data right now."
    if availability.remedy:
        text = f"{text} {availability.remedy}"
    return ActionResult.failure(text, error_code="weather_unavailable")


def _current(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = provider_for(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    location = _clean_location(params.get("location"))
    result = provider.current(location)
    if not result.ok:
        # The provider's own wording says *why* -- unknown place, no location
        # set, service unreachable -- and each needs a different response from
        # the user, so it is passed through rather than flattened.
        data = dict(result.data)
        if result.error_code == "no_location":
            # Names the slot, so the engine can ask for it and use the answer.
            data["needs"] = "location"
        return ActionResult.failure(result.message,
                                    error_code=result.error_code or "weather_failed",
                                    **data)
    return ActionResult.success(result.message, **result.data)


def _forecast(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = provider_for(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    location = _clean_location(params.get("location"))
    try:
        days = max(1, min(7, int(params.get("days") or 3)))
    except (TypeError, ValueError):
        days = 3
    result = provider.forecast(location, days=days)
    if not result.ok:
        return ActionResult.failure(result.message,
                                    error_code=result.error_code or "weather_failed",
                                    **result.data)

    entries = result.data.get("days") or []
    if not entries:
        return ActionResult.failure(
            "The forecast came back empty, so I've nothing to tell you.",
            error_code="empty_forecast")
    place = (result.data.get("place") or {}).get("label", "there")
    lines = []
    for entry in entries[:days]:
        bits = []
        if entry.get("condition"):
            bits.append(str(entry["condition"]))
        low, high = entry.get("low_c"), entry.get("high_c")
        if low is not None and high is not None:
            bits.append(f"{low:.0f}-{high:.0f}°C")
        chance = entry.get("precipitation_chance")
        if chance is not None:
            bits.append(f"{chance:.0f}% rain")
        lines.append(f"{entry['date']}: " + (", ".join(bits) or "no details"))
    return ActionResult.success(f"{place} — " + "; ".join(lines) + ".",
                                **result.data)


def _will_it_rain(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Answer the rain question from the forecast's own numbers.

    Kept as its own action because "will it rain tomorrow?" wants a yes or a
    no, and reading a probability out of a three-day table is work the user
    should not have to do.  The probability is always stated alongside the
    answer, so a 55% "probably" is visible as the near-coin-flip it is.
    """
    provider = provider_for(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    location = _clean_location(params.get("location"))
    when = str(params.get("when") or "tomorrow").strip().lower()
    index = 0 if "today" in when or "tonight" in when else 1
    result = provider.forecast(location, days=index + 1)
    if not result.ok:
        return ActionResult.failure(result.message,
                                    error_code=result.error_code or "weather_failed",
                                    **result.data)
    entries = result.data.get("days") or []
    if len(entries) <= index:
        return ActionResult.failure(
            f"The forecast didn't reach {when}.", error_code="out_of_range")

    entry = entries[index]
    place = (result.data.get("place") or {}).get("label", "there")
    chance = entry.get("precipitation_chance")
    label = "today" if index == 0 else "tomorrow"
    if chance is None:
        # No probability came back. Saying "no" here would be inventing one.
        condition = entry.get("condition") or "no conditions given"
        return ActionResult.success(
            f"The forecast for {place} {label} doesn't include a rain "
            f"probability — it says {condition}.",
            place=place, when=label, **entry)
    verdict = ("yes, probably" if chance >= 60
               else "possibly" if chance >= 30 else "probably not")
    return ActionResult.success(
        f"Rain in {place} {label}: {verdict} — {chance:.0f}% chance.",
        place=place, when=label, **entry)


def get_actions() -> List[ActionSpec]:
    location = ParameterSpec(
        name="location", type=ParameterType.STRING,
        description="the place to look up", required=False, default="",
        max_length=80)
    return [
        ActionSpec(
            name="get_weather",
            description="Report the current weather.",
            category="Weather", risk=RiskLevel.LOW, handler=_current,
            permissions=(Permission.INTERNET,), parameters=(location,),
            requires_internet=True,
            examples=("what's the weather", "what's the weather in Bengaluru",
                      "what's the weather today"),
        ),
        ActionSpec(
            name="get_forecast",
            description="Report the forecast for the next few days.",
            category="Weather", risk=RiskLevel.LOW, handler=_forecast,
            permissions=(Permission.INTERNET,),
            parameters=(location,
                        ParameterSpec(name="days", type=ParameterType.INTEGER,
                                      description="how many days", required=False,
                                      default=3, min_value=1, max_value=7)),
            requires_internet=True,
            examples=("what's the forecast", "what's the weather this week"),
        ),
        ActionSpec(
            name="will_it_rain",
            description="Say whether rain is forecast.",
            category="Weather", risk=RiskLevel.LOW, handler=_will_it_rain,
            permissions=(Permission.INTERNET,),
            parameters=(location,
                        ParameterSpec(name="when", required=False,
                                      default="tomorrow",
                                      description="today or tomorrow",
                                      max_length=20)),
            requires_internet=True,
            examples=("will it rain tomorrow", "is it going to rain today"),
        ),
    ]
