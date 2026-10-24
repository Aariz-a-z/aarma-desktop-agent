"""Keyboard automation.

Phase 1 registered nothing here and said why: synthesising keystrokes means
AARMA can type into whatever window has focus, including a password box.  That
reasoning has not changed -- what has changed is that the guards it was waiting
for now exist:

* the ``keyboard`` permission is off by default, so the capability does not
  exist at all until the user switches it on;
* :func:`providers.input.parse_chord` turns a key name into a virtual key code
  from a fixed table, so no free text reaches the operating system;
* :data:`providers.input.REFUSED_CHORDS` refuses the shortcuts whose whole
  effect is destructive, naming the action that does the same job with a target
  and a confirmation.

What is deliberately still absent is any notion of *where* the keystroke lands.
AARMA does not read the focused control, so it cannot know a password box from
a search box.  ``type_text`` therefore says what it typed into -- the window's
title -- so a mistake is visible immediately rather than after the fact.
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
from providers.desktop import WINDOWS
from providers.input import INPUT, MAX_TYPED_CHARACTERS, UnknownKeyError, parse_chord
from security.permissions import Permission
from security.validator import RiskLevel
from utils.logger import get_logger

logger = get_logger("actions.keyboard")


def _input(ctx: ActionContext):
    return getattr(ctx, "input_provider", None) or INPUT


def _windows(ctx: ActionContext):
    return getattr(ctx, "window_provider", None) or WINDOWS


def _focus_label(ctx: ActionContext) -> str:
    """The window a keystroke is about to reach, for the reply.

    Best effort: if the window provider is unavailable the keystroke still
    goes, because the user asked for it -- but then the reply does not name a
    destination it could not establish.
    """
    provider = _windows(ctx)
    try:
        if not provider.is_available():
            return ""
        window = provider.active_window()
    except Exception:  # noqa: BLE001 - naming the target is a convenience
        return ""
    return window.label() if window is not None else ""


def _press_key(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    raw = str(params.get("keys") or "")
    try:
        chord = parse_chord(raw)
    except UnknownKeyError as exc:
        # The parser's message names what it does understand, which is more
        # use than "invalid input" and keeps the key table in one place.
        return ActionResult.failure(str(exc), error_code="unknown_key")

    provider = _input(ctx)
    if not provider.is_available():
        return ActionResult.failure(provider.availability().detail,
                                    error_code="input_unavailable")

    result = provider.press(chord)
    if not result.ok:
        return ActionResult.failure(result.message,
                                    error_code=result.error_code or "press_failed",
                                    **result.data)
    where = _focus_label(ctx)
    message = (f"Pressed {chord.spelling()} in {where}." if where
               else f"Pressed {chord.spelling()}.")
    return ActionResult.success(message, window=where, **result.data)


def _type_text(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    text = str(params.get("text") or "")
    provider = _input(ctx)
    if not provider.is_available():
        return ActionResult.failure(provider.availability().detail,
                                    error_code="input_unavailable")

    where = _focus_label(ctx)
    result = provider.type_text(text)
    if not result.ok:
        return ActionResult.failure(result.message,
                                    error_code=result.error_code or "type_failed",
                                    **result.data)
    # The window is named so a keystroke that went somewhere unexpected is
    # obvious at once. The text itself is echoed because the user dictated it
    # and needs to check the transcription, but it is never written to the log.
    message = (f"Typed \"{text}\" into {where}." if where
               else f"Typed \"{text}\".")
    return ActionResult.success(message, window=where, **result.data)


def get_actions() -> List[ActionSpec]:
    return [
        ActionSpec(
            name="press_key",
            description="Press a key or keyboard shortcut.",
            category="Input",
            # MEDIUM: a keystroke changes whatever is focused, but each one is
            # a single reversible action. The destructive shortcuts are not
            # reachable at all -- they are refused by the provider.
            risk=RiskLevel.MEDIUM,
            handler=_press_key,
            permissions=(Permission.KEYBOARD,),
            parameters=(
                ParameterSpec(
                    name="keys", type=ParameterType.STRING,
                    description="the key or combination, e.g. Enter or Ctrl+C",
                    max_length=40),
            ),
            platforms=("win32",),
            examples=("press enter", "press escape", "press ctrl+c",
                      "press ctrl+shift+t"),
        ),
        ActionSpec(
            name="type_text",
            description="Type text into whatever window is in front.",
            category="Input",
            risk=RiskLevel.MEDIUM,
            handler=_type_text,
            permissions=(Permission.KEYBOARD,),
            parameters=(
                ParameterSpec(
                    name="text", type=ParameterType.TEXT,
                    description="the text to type",
                    max_length=MAX_TYPED_CHARACTERS),
            ),
            platforms=("win32",),
            examples=("type hello world", "type my email address"),
        ),
    ]
