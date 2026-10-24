"""Window control: minimise, maximise, restore, switch and close.

The recurring hazard with window automation is not failure, it is *success on
the wrong window*.  "Close this window" said while a notification steals focus
for half a second closes something the user never looked at.  So every handler
here resolves a concrete :class:`~providers.desktop.WindowInfo` before acting,
names it in the reply, and refuses when the target is ambiguous rather than
picking the first match.

Two grading decisions worth stating:

* **Moving and re-stating a window is LOW.**  Minimising loses nothing and is
  undone by clicking the taskbar.  A confirmation here would be noise, and
  noisy confirmations are how people learn to click "yes" without reading.
* **Closing is MEDIUM, and graceful.**  ``WM_CLOSE`` is a *request*: an editor
  with unsaved work shows its own "save changes?" prompt and stays open.  That
  is the application's job and it does it better than AARMA could, so the reply
  says the window was *asked* to close rather than claiming it closed.

Nothing here force-kills a process.  ``taskkill /F`` discards unsaved work with
no prompt, and this build has no action that does it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from actions.registry import (
    ActionContext,
    ActionResult,
    ActionSpec,
    ParameterSpec,
    ParameterType,
)
from providers.desktop import WINDOWS, WindowInfo
from security.permissions import Permission
from security.validator import RiskLevel
from utils.logger import get_logger

logger = get_logger("actions.windows")

#: How many candidates to name back when a request is ambiguous.
MAX_LISTED = 6


def _provider(ctx: ActionContext):
    """The window provider, overridable by a test through the context."""
    return getattr(ctx, "window_provider", None) or WINDOWS


def _unavailable(provider) -> ActionResult:
    availability = provider.availability()
    text = availability.detail or "I can't reach the window manager."
    if availability.remedy:
        text = f"{text} {availability.remedy}"
    return ActionResult.failure(text, error_code="windows_unavailable")


def _resolve(ctx: ActionContext, params: Dict[str, object]
             ) -> Tuple[Optional[WindowInfo], Optional[ActionResult]]:
    """Work out which window the user meant.

    Returns ``(window, None)`` or ``(None, refusal)``.  An ambiguous request is
    a refusal, not a coin toss: two Chrome windows are two different things and
    only the user knows which one they meant.
    """
    provider = _provider(ctx)
    if not provider.is_available():
        return None, _unavailable(provider)

    wanted = str(params.get("target") or "").strip()
    if not wanted or wanted.lower() in {"this", "current", "the current one",
                                        "this one", "front", "active"}:
        window = provider.active_window()
        if window is None:
            return None, ActionResult.failure(
                "I can't tell which window is in front.",
                error_code="no_active_window")
        return window, None

    matches = provider.find_window(wanted)
    if not matches:
        open_now = ", ".join(sorted({w.title for w in provider.list_windows()
                                     if w.title})[:MAX_LISTED])
        detail = f" I can see: {open_now}." if open_now else ""
        return None, ActionResult.failure(
            f"I couldn't find a window for \"{wanted}\".{detail}",
            error_code="window_not_found")
    if len(matches) > 1 and matches[0].process_name != matches[1].process_name:
        listed = "; ".join(w.label() for w in matches[:MAX_LISTED])
        return None, ActionResult.failure(
            f"\"{wanted}\" matches {len(matches)} windows: {listed}. Which one?",
            error_code="ambiguous",
            candidates=[w.as_dict() for w in matches[:MAX_LISTED]])
    return matches[0], None


def _from_provider_result(result, *, fallback_error: str) -> ActionResult:
    """Translate a provider result, preserving its own honest wording."""
    if result.ok:
        return ActionResult.success(result.message, **result.data)
    return ActionResult.failure(result.message,
                                error_code=result.error_code or fallback_error,
                                **result.data)


# --------------------------------------------------------------------- handlers

def _minimize(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    window, refusal = _resolve(ctx, params)
    if refusal is not None:
        return refusal
    return _from_provider_result(_provider(ctx).minimize(window),
                                 fallback_error="minimize_failed")


def _maximize(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    window, refusal = _resolve(ctx, params)
    if refusal is not None:
        return refusal
    return _from_provider_result(_provider(ctx).maximize(window),
                                 fallback_error="maximize_failed")


def _restore(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    window, refusal = _resolve(ctx, params)
    if refusal is not None:
        return refusal
    return _from_provider_result(_provider(ctx).restore(window),
                                 fallback_error="restore_failed")


def _close(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    window, refusal = _resolve(ctx, params)
    if refusal is not None:
        return refusal
    return _from_provider_result(_provider(ctx).close(window),
                                 fallback_error="close_failed")


def _switch(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = _provider(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    wanted = str(params.get("target") or "").strip()
    if not wanted:
        return ActionResult.failure("Which window should I switch to?",
                                    error_code="missing_target")
    window, refusal = _resolve(ctx, params)
    if refusal is not None:
        return refusal
    return _from_provider_result(provider.focus(window),
                                 fallback_error="focus_failed")


def _previous(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Switch to the window behind the current one.

    Uses the Z-order the OS reports rather than sending Alt+Tab.  Alt+Tab is a
    held-key gesture whose result depends on timing and on how long the keys
    stay down; asking the window manager which window is second is a fact.
    """
    provider = _provider(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    window = provider.previous_window()
    if window is None:
        return ActionResult.failure("There's only one window open.",
                                    error_code="no_previous_window")
    return _from_provider_result(provider.focus(window),
                                 fallback_error="focus_failed")


def _list(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = _provider(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    windows = provider.list_windows()
    if not windows:
        return ActionResult.success("I can't see any open windows.",
                                    windows=[], count=0)
    shown = windows[:MAX_LISTED * 2]
    listed = "; ".join(w.label() for w in shown)
    header = (f"{len(windows)} windows open" if len(windows) != 1
              else "One window open")
    if len(shown) < len(windows):
        header += f" — the first {len(shown)}"
    return ActionResult.success(f"{header}: {listed}.",
                                windows=[w.as_dict() for w in shown],
                                count=len(windows))


def _active(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    provider = _provider(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    window = provider.active_window()
    if window is None:
        return ActionResult.failure("I can't tell which window is in front.",
                                    error_code="no_active_window")
    return ActionResult.success(f"{window.label()} is in front.",
                                **window.as_dict())


def get_actions() -> List[ActionSpec]:
    target = ParameterSpec(
        name="target",
        type=ParameterType.STRING,
        description="which window — a name, or blank for the one in front",
        required=False, default="", max_length=80,
    )
    state_permissions = (Permission.SYSTEM_CONTROL,)
    return [
        ActionSpec(
            name="minimize_window", description="Minimise a window.",
            category="Windows", risk=RiskLevel.LOW, handler=_minimize,
            permissions=state_permissions, parameters=(target,),
            platforms=("win32",),
            examples=("minimize this window", "minimise Chrome"),
        ),
        ActionSpec(
            name="maximize_window", description="Maximise a window.",
            category="Windows", risk=RiskLevel.LOW, handler=_maximize,
            permissions=state_permissions, parameters=(target,),
            platforms=("win32",),
            examples=("maximize this window", "maximise Notepad"),
        ),
        ActionSpec(
            name="restore_window",
            description="Restore a minimised or maximised window.",
            category="Windows", risk=RiskLevel.LOW, handler=_restore,
            permissions=state_permissions, parameters=(target,),
            platforms=("win32",),
            examples=("restore this window", "restore Chrome"),
        ),
        ActionSpec(
            name="switch_window", description="Switch to another window.",
            category="Windows", risk=RiskLevel.LOW, handler=_switch,
            permissions=state_permissions,
            parameters=(ParameterSpec(
                name="target", type=ParameterType.STRING,
                description="the window to switch to", max_length=80),),
            platforms=("win32",),
            examples=("switch to Chrome", "switch to VS Code"),
        ),
        ActionSpec(
            name="previous_window",
            description="Switch to the window you were in before.",
            category="Windows", risk=RiskLevel.LOW, handler=_previous,
            permissions=state_permissions, platforms=("win32",),
            examples=("switch to the previous window", "go back a window"),
        ),
        ActionSpec(
            name="close_window",
            description="Ask the window in front (or one you name) to close.",
            category="Windows",
            # MEDIUM, not HIGH: WM_CLOSE lets the application prompt about
            # unsaved work itself, so this is not the irreversible act that a
            # forced kill would be.
            risk=RiskLevel.MEDIUM, handler=_close,
            permissions=state_permissions, parameters=(target,),
            platforms=("win32",),
            examples=("close this window", "close Notepad"),
        ),
        ActionSpec(
            name="list_windows", description="List the windows that are open.",
            category="Windows", risk=RiskLevel.LOW, handler=_list,
            platforms=("win32",),
            examples=("what windows are open", "list my windows"),
        ),
        ActionSpec(
            name="active_window", description="Say which window is in front.",
            category="Windows", risk=RiskLevel.LOW, handler=_active,
            platforms=("win32",),
            examples=("what window is active", "which window is in front"),
        ),
    ]
