"""Answering "what can you do?" from the registry rather than from a list.

A hand-written capability list goes stale the first time an action is added or
removed, and a stale one makes the assistant claim things it cannot do. So this
reads the live registry: if an action is not registered, it is not mentioned,
and if it is registered it is genuinely callable.

Grouped and short by default. Someone asking what an assistant can do wants a
sense of its shape, not forty-odd action names; the detail is there for anyone
who asks for a category by name.
"""

from __future__ import annotations

from typing import Dict, List

from actions.registry import ActionContext, ActionResult, ActionSpec, ParameterSpec
from security.validator import RiskLevel
from utils.logger import get_logger

logger = get_logger("actions.capabilities")

#: Category -> the words someone would use when asking about it.
_TOPICS: Dict[str, tuple] = {
    "Desktop": ("app", "apps", "application", "applications", "program",
                "programs", "open", "computer"),
    "Web": ("web", "browser", "internet", "search", "website", "url"),
    "Files": ("file", "files", "folder", "folders", "documents", "downloads"),
    "Media": ("volume", "sound", "audio", "music", "media", "play", "mute"),
    "System": ("system", "cpu", "ram", "memory", "disk", "battery", "health",
               "performance"),
    "Power": ("power", "shutdown", "shut", "restart", "reboot", "lock", "sleep"),
    "Reminders": ("reminder", "reminders", "remind"),
    "Calendar": ("calendar", "meeting", "meetings", "event", "events",
                 "schedule"),
    "Information": ("time", "date", "day", "status"),
    "Assistant": ("name", "language", "voice", "wake"),
    # Keywords are matched as substrings of whatever the user asked about, so
    # every one here has to be *distinctive*. Generic verbs are the trap: an
    # earlier draft listed "send" under Email, which made "can you send a fax?"
    # answer yes and list the email actions. If a word could plausibly appear
    # in a question about something else, it does not belong in this table.
    "Windows": ("window", "windows", "minimize", "minimise", "maximize",
                "maximise"),
    "Input": ("keyboard", "mouse", "typing", "click", "cursor", "pointer",
              "scroll", "hotkey", "shortcut"),
    "Weather": ("weather", "forecast", "temperature", "umbrella", "raining"),
    "Email": ("email", "e-mail", "mail", "draft", "drafts", "inbox", "compose"),
}


def provider_report(ctx: ActionContext) -> List[Dict[str, object]]:
    """What every external integration can actually do, right now.

    Built by *asking each provider*, not from a written list.  A hand-kept
    table would say Google Calendar was connected long after it stopped being,
    and the whole value of this report is that it cannot.
    """
    from providers.calendar import LocalCalendarProvider, google_calendar, outlook_calendar
    from providers.desktop import WINDOWS
    from providers.email import configured_provider, gmail_email, outlook_email, smtp_email
    from providers.input import INPUT
    from providers.media import MEDIA
    from providers.system import SYSTEM
    from providers.weather import build_weather_provider

    def _row(label: str, provider) -> Dict[str, object]:
        try:
            availability = provider.availability()
        except Exception as exc:  # noqa: BLE001 - a broken provider is unavailable
            logger.warning("Provider could not report availability",
                           extra={"context": {"provider": label, "error": str(exc)}})
            return {"capability": label, "provider": getattr(provider, "name", label),
                    "available": False, "detail": "This one isn't answering."}
        return {"capability": label, "provider": availability.provider,
                "available": availability.available,
                "detail": availability.detail, "remedy": availability.remedy}

    return [
        _row("System", getattr(ctx, "system_provider", None) or SYSTEM),
        _row("Media", getattr(ctx, "media_provider", None) or MEDIA),
        _row("Windows", getattr(ctx, "window_provider", None) or WINDOWS),
        _row("Keyboard and mouse", getattr(ctx, "input_provider", None) or INPUT),
        _row("Calendar (local)",
             getattr(ctx, "calendar_provider", None) or LocalCalendarProvider(ctx.database)),
        _row("Calendar (Google)", google_calendar()),
        _row("Calendar (Outlook)", outlook_calendar()),
        _row("Weather",
             getattr(ctx, "weather_provider", None)
             or build_weather_provider(ctx.config, online=bool(ctx.online))),
        _row("Email (drafts)",
             getattr(ctx, "email_provider", None) or configured_provider(ctx.database)),
        _row("Email (Gmail)", gmail_email()),
        _row("Email (Outlook)", outlook_email()),
        _row("Email (SMTP)", smtp_email()),
    ]


def _integrations(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Answer "what are you actually connected to?" without flattering anything."""
    rows = provider_report(ctx)
    ready = [row for row in rows if row["available"]]
    missing = [row for row in rows if not row["available"]]
    lines = [f"Connected: {', '.join(str(r['capability']) for r in ready)}."
             if ready else "Nothing is connected."]
    if missing:
        lines.append("Not connected: " + "; ".join(
            f"{r['capability']} — {r['detail']}" for r in missing))
    return ActionResult.success(" ".join(lines), providers=rows,
                                connected=len(ready), total=len(rows))


def _capabilities(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """List what is actually registered, grouped by category."""
    registry = ctx.registry if hasattr(ctx, "registry") else None
    if registry is None:
        return ActionResult.failure("I can't reach my own capability list.",
                                    error_code="registry_unavailable")

    topic = str(params.get("topic") or "").strip().lower()
    grouped = registry.by_category()

    if topic:
        # Asked about one thing: answer about that thing, in detail.
        wanted = [name for name, words in _TOPICS.items()
                  if topic == name.lower() or any(w in topic for w in words)]
        matched = {name: grouped[name] for name in wanted if name in grouped}
        if not matched:
            available = ", ".join(sorted(grouped))
            return ActionResult.success(
                f"I don't have anything for \"{topic}\". I can help with: "
                f"{available}.", topic=topic, matched=[])
        lines = []
        for name, specs in sorted(matched.items()):
            examples = []
            for spec in specs:
                if spec.examples:
                    examples.append(spec.examples[0])
            lines.append(f"{name}: " + "; ".join(examples[:6]))
        return ActionResult.success(
            "Yes — " + " ".join(lines),
            topic=topic,
            matched=[s.name for specs in matched.values() for s in specs])

    # The general question: a shape, not a catalogue.
    summary = []
    for name in sorted(grouped):
        summary.append(f"{name} ({len(grouped[name])})")
    total = len(registry)
    return ActionResult.success(
        f"I can do {total} things, grouped as: " + ", ".join(summary)
        + ". Ask about any group — \"can you control the volume?\" — "
          "and I'll tell you exactly what I support.",
        total=total,
        categories={name: [s.name for s in specs]
                    for name, specs in grouped.items()})


def get_actions() -> List[ActionSpec]:
    return [
        ActionSpec(
            name="list_integrations",
            description="Say which outside services are actually connected.",
            category="Information", risk=RiskLevel.LOW, handler=_integrations,
            examples=("what are you connected to", "is my calendar connected",
                      "which integrations are set up"),
        ),
        ActionSpec(
            name="list_capabilities",
            description="Say what I can actually do.",
            category="Information", risk=RiskLevel.LOW, handler=_capabilities,
            parameters=(ParameterSpec(name="topic", required=False, default="",
                                      description="a category to ask about",
                                      max_length=40),),
            examples=("what can you do", "what commands do you support",
                      "can you control the volume", "can you schedule meetings"),
        ),
    ]
