"""Read-only information about the machine and the assistant itself.

These are the lowest-risk actions in the registry: nothing here changes state,
so none of them require a permission or a confirmation.  They exist partly to
be useful and partly because they exercise the full pipeline end to end
(intent -> plan -> permission -> validate -> execute -> respond) without
touching anything the user could regret.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime
from typing import Dict, List

from actions.registry import ActionContext, ActionResult, ActionSpec
from security.validator import RiskLevel

TIME_FORMAT = "%I:%M %p"
DATE_FORMAT = "%A, %d %B %Y"


def _get_time(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    now = datetime.now()
    formatted = now.strftime(TIME_FORMAT).lstrip("0")
    return ActionResult.success(
        ctx.language.t("response.time", time=formatted),
        time=formatted,
        iso=now.isoformat(timespec="seconds"),
    )


def _get_date(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    now = datetime.now()
    formatted = now.strftime(DATE_FORMAT)
    return ActionResult.success(
        ctx.language.t("response.date", date=formatted),
        date=formatted,
        iso=now.date().isoformat(),
    )


def _get_status(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Summarise identity, language, connectivity and granted permissions."""
    config = ctx.config.config
    granted = [name for name, value in
               ((p.value, v) for p, v in ctx.permissions.snapshot().items()) if value]
    connectivity = ctx.language.t("status.online" if ctx.online else "status.offline")
    language_info = ctx.language.info()

    lines = [
        f"Name: {config.assistant.name} ({config.assistant.gender} profile)",
        f"Voice profile: {config.assistant.voice_id}",
        f"Wake word: \"{config.assistant.wake_word}\"",
        f"Interface language: {language_info.english_name} ({language_info.status.value})",
        f"Connectivity: {connectivity}",
        f"Permissions granted: {len(granted)} of {len(ctx.permissions.snapshot())}",
        f"Security mode: {'on' if config.security.enabled else 'off'}",
        f"Platform: {platform.system()} {platform.release()} · Python {sys.version.split()[0]}",
    ]
    return ActionResult.success("\n".join(lines), granted_permissions=granted, online=ctx.online)


def get_actions() -> List[ActionSpec]:
    return [
        ActionSpec(
            name="get_time",
            description="Tell you the current time.",
            category="Information",
            risk=RiskLevel.LOW,
            handler=_get_time,
            examples=("what time is it", "tell me the time"),
        ),
        ActionSpec(
            name="get_date",
            description="Tell you today's date.",
            category="Information",
            risk=RiskLevel.LOW,
            handler=_get_date,
            examples=("what's the date", "what day is it today"),
        ),
        ActionSpec(
            name="get_status",
            description="Report my current name, language, connectivity and permissions.",
            category="Information",
            risk=RiskLevel.LOW,
            handler=_get_status,
            examples=("status", "are you online"),
        ),
    ]
