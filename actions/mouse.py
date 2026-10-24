"""Mouse automation, limited to what can be done honestly.

Phase 1 registered nothing here, on the grounds that clicking is only
meaningful once AARMA can *see* what it is clicking.  That is still true, and
this module does not pretend otherwise -- it ships the part that does not
require sight, and refuses the part that does.

**What works:** a click at coordinates the user named, a click where the
pointer already is, a double-click, a right-click, moving the pointer, and
scrolling.  Each of these is fully determined by its arguments.

**What is refused:** "click the Submit button".  There is no screen
understanding in this build, so the only way to answer that would be to guess
at a position, and a guessed click is indistinguishable from a correct one
until something has already been pressed.  :func:`_click` requires an explicit
point or none at all; it never infers one from a description.

Coordinates outside the desktop are rejected rather than clamped.  Clamping
would turn a misheard "5000, 300" into a real click at the screen edge -- a
plausible-looking action nobody asked for, which is the same failure as
inventing a sensor reading.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from actions.registry import (
    ActionContext,
    ActionResult,
    ActionSpec,
    ParameterSpec,
    ParameterType,
)
from providers.input import INPUT
from security.permissions import Permission
from security.validator import RiskLevel
from utils.logger import get_logger

logger = get_logger("actions.mouse")

#: "500,300", "500, 300" and "500 300" are all the same point spoken aloud.
_POINT = re.compile(r"^\s*(?P<x>-?\d{1,5})\s*(?:,|\s)\s*(?P<y>-?\d{1,5})\s*$")

#: Words that describe a target instead of naming a point.  Matched so the
#: refusal can explain *why*, rather than reporting a parse error for something
#: that is a capability gap and not a typo.
_DESCRIBED_TARGET = re.compile(
    r"\b(?:button|link|icon|menu|field|box|tab|checkbox|submit|ok|cancel)\b",
    re.IGNORECASE)


def _provider(ctx: ActionContext):
    return getattr(ctx, "input_provider", None) or INPUT


def _point(raw: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Parse a coordinate string.  Returns ``(x, y, refusal)``."""
    text = str(raw or "").strip()
    if not text:
        return None, None, None
    found = _POINT.match(text)
    if found:
        return int(found.group("x")), int(found.group("y")), None
    if _DESCRIBED_TARGET.search(text):
        return None, None, (
            f"I can't click \"{text}\" — I can't see the screen, so I don't "
            "know where that is. Give me a position like \"click at 500, 300\" "
            "and I'll click there.")
    return None, None, (
        f"I couldn't read \"{text}\" as a position. Say it like \"500, 300\".")


def _do_click(ctx: ActionContext, params: Dict[str, object], *,
              button: str, times: int) -> ActionResult:
    x, y, refusal = _point(str(params.get("at") or ""))
    if refusal:
        return ActionResult.failure(refusal, error_code="no_visual_target")
    provider = _provider(ctx)
    if not provider.is_available():
        return ActionResult.failure(provider.availability().detail,
                                    error_code="input_unavailable")
    result = provider.click(x=x, y=y, button=button, times=times)
    if not result.ok:
        return ActionResult.failure(result.message,
                                    error_code=result.error_code or "click_failed",
                                    **result.data)
    return ActionResult.success(result.message, **result.data)


def _click(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    return _do_click(ctx, params, button="left", times=1)


def _double_click(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    return _do_click(ctx, params, button="left", times=2)


def _right_click(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    return _do_click(ctx, params, button="right", times=1)


def _move(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    x, y, refusal = _point(str(params.get("to") or ""))
    if refusal:
        return ActionResult.failure(refusal, error_code="no_visual_target")
    if x is None or y is None:
        return ActionResult.failure(
            "Where should I move the pointer? Give me a position like \"500, 300\".",
            error_code="missing_point")
    provider = _provider(ctx)
    if not provider.is_available():
        return ActionResult.failure(provider.availability().detail,
                                    error_code="input_unavailable")
    result = provider.move(x, y)
    if not result.ok:
        return ActionResult.failure(result.message,
                                    error_code=result.error_code or "move_failed",
                                    **result.data)
    return ActionResult.success(result.message, **result.data)


def _scroll(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    direction = str(params.get("direction") or "down").strip().lower()
    if direction not in {"up", "down"}:
        return ActionResult.failure("I can scroll up or down.",
                                    error_code="unknown_direction")
    try:
        amount = max(1, min(20, int(params.get("amount") or 3)))
    except (TypeError, ValueError):
        amount = 3
    provider = _provider(ctx)
    if not provider.is_available():
        return ActionResult.failure(provider.availability().detail,
                                    error_code="input_unavailable")
    result = provider.scroll(amount if direction == "up" else -amount)
    if not result.ok:
        return ActionResult.failure(result.message,
                                    error_code=result.error_code or "scroll_failed")
    return ActionResult.success(result.message, direction=direction,
                                amount=amount)


def get_actions() -> List[ActionSpec]:
    at = ParameterSpec(
        name="at", type=ParameterType.STRING,
        description="where to click, as \"x, y\" — blank clicks where the pointer is",
        required=False, default="", max_length=40)
    return [
        ActionSpec(
            name="click_mouse", description="Click the mouse.",
            category="Input", risk=RiskLevel.MEDIUM, handler=_click,
            permissions=(Permission.MOUSE,), parameters=(at,),
            platforms=("win32",),
            examples=("click", "click at 500, 300"),
        ),
        ActionSpec(
            name="double_click_mouse", description="Double-click the mouse.",
            category="Input", risk=RiskLevel.MEDIUM, handler=_double_click,
            permissions=(Permission.MOUSE,), parameters=(at,),
            platforms=("win32",),
            examples=("double click", "double click at 200, 400"),
        ),
        ActionSpec(
            name="right_click_mouse", description="Right-click the mouse.",
            category="Input", risk=RiskLevel.MEDIUM, handler=_right_click,
            permissions=(Permission.MOUSE,), parameters=(at,),
            platforms=("win32",),
            examples=("right click", "right click at 640, 480"),
        ),
        ActionSpec(
            name="move_mouse", description="Move the pointer to a position.",
            category="Input", risk=RiskLevel.LOW, handler=_move,
            permissions=(Permission.MOUSE,),
            parameters=(ParameterSpec(
                name="to", type=ParameterType.STRING,
                description="the position, as \"x, y\"", max_length=40),),
            platforms=("win32",),
            examples=("move the mouse to 500, 300",),
        ),
        ActionSpec(
            name="scroll_mouse", description="Scroll up or down.",
            category="Input", risk=RiskLevel.LOW, handler=_scroll,
            permissions=(Permission.MOUSE,),
            parameters=(
                ParameterSpec(name="direction", required=False, default="down",
                              description="up or down", choices=("up", "down"),
                              max_length=10),
                ParameterSpec(name="amount", type=ParameterType.INTEGER,
                              required=False, default=3,
                              description="how far, in notches",
                              min_value=1, max_value=20),
            ),
            platforms=("win32",),
            examples=("scroll down", "scroll up"),
        ),
    ]
