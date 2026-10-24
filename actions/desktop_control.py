"""Closing and switching whole applications, and the one action that runs code.

Phase 2 could close eleven catalogued programs by name.  This closes whatever
is actually open, found by enumerating windows -- which is what "close VS Code"
has to mean when VS Code was never in the catalogue.

Three decisions shape the file.

**Closing is graceful, always.**  ``WM_CLOSE`` asks a window to close, so an
editor with unsaved work shows its own "save changes?" prompt and stays open.
That prompt is better than anything AARMA could offer, because the application
knows what is unsaved and AARMA does not.  Nothing here calls ``taskkill /F``.

**Multiple windows is a question, not a detail.**  "Close VS Code" with three
windows open is ambiguous about scope, and silently closing one -- or all
three -- both guess.  The count is reported and the whole-application close is
graded so the user is asked.

**Running code is a separate, named, HIGH-risk action.**  ``execute_shell_command``
exists so that "run this" can never be *quietly* satisfied by some other
handler: there is one door, it is labelled, and it is locked.  It is not
reachable from the nearest-intent resolver at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from actions.registry import (
    ActionContext,
    ActionResult,
    ActionSpec,
    ParameterSpec,
    ParameterType,
)
from security.permissions import Permission
from security.validator import RiskLevel
from utils.logger import get_logger

logger = get_logger("actions.desktop_control")

#: Processes that own the desktop itself.  Closing these logs the user out or
#: leaves them with no shell, so they are never candidates however they are
#: named.  This is a floor, not a preference: it is not user-adjustable.
PROTECTED_PROCESSES = frozenset({
    "explorer.exe", "dwm.exe", "winlogon.exe", "csrss.exe", "services.exe",
    "lsass.exe", "smss.exe", "wininit.exe", "svchost.exe", "taskmgr.exe",
    "sihost.exe", "fontdrvhost.exe", "ctfmon.exe", "searchhost.exe",
    "startmenuexperiencehost.exe", "shellexperiencehost.exe",
    "applicationframehost.exe", "textinputhost.exe", "systemsettings.exe",
    "python.exe", "pythonw.exe", "windowsterminal.exe",
})

#: How many applications to name in a confirmation summary.
MAX_SUMMARY = 10


def _windows(ctx: ActionContext):
    injected = getattr(ctx, "window_provider", None)
    if injected is not None:
        return injected
    from providers.desktop import WINDOWS

    return WINDOWS


def _applications(ctx: ActionContext):
    injected = getattr(ctx, "application_index", None)
    if injected is not None:
        return injected
    from workspace.applications import ApplicationIndex

    return ApplicationIndex()


def _unavailable(provider) -> ActionResult:
    availability = provider.availability()
    text = availability.detail or "I can't reach the window manager."
    return ActionResult.failure(text, error_code="windows_unavailable")


def _is_protected(process_name: str) -> bool:
    return (process_name or "").lower() in PROTECTED_PROCESSES


def _aarma_process() -> str:
    return Path(sys.executable).name.lower()


def _windows_for(provider, wanted: str, application_index=None):
    """Every open window belonging to the named application.

    Matched on the process name first and the title second.  Process is the
    stronger signal: a browser tab whose title happens to contain "code" is not
    VS Code, and title-only matching is how "close VS Code" closes a web page.
    """
    needle = " ".join(str(wanted or "").lower().split())
    if not needle:
        return []

    process_names = set()
    if application_index is not None:
        for application in application_index.find(needle, limit=3):
            if application.process_name:
                process_names.add(application.process_name.lower())
    # The catalogue knows process names discovery cannot infer.
    from actions.applications import resolve_entry

    entry = resolve_entry(needle)
    if entry is not None and entry.process_name:
        process_names.add(entry.process_name.lower())

    matched = []
    for window in provider.list_windows():
        process = (window.process_name or "").lower()
        stem = process[:-4] if process.endswith(".exe") else process
        if process in process_names or stem == needle or process == f"{needle}.exe":
            matched.append(window)
        elif needle in stem:
            matched.append(window)
        elif needle in (window.title or "").lower() and not process_names:
            # Title matching only when nothing better identified the process.
            matched.append(window)
    return matched


# --------------------------------------------------------------- close an app

def _close_application_windows(ctx: ActionContext,
                               params: Dict[str, object]) -> ActionResult:
    """Close every window of one application, gracefully."""
    provider = _windows(ctx)
    if not provider.is_available():
        return _unavailable(provider)

    wanted = str(params.get("name") or "").strip()
    if not wanted:
        return ActionResult.failure("Which application should I close?",
                                    error_code="missing_target")

    matched = _windows_for(provider, wanted, _applications(ctx))
    if not matched:
        return ActionResult.failure(
            f"I couldn't find {wanted} running.", error_code="not_running")

    protected = [w for w in matched if _is_protected(w.process_name)]
    if protected:
        names = ", ".join(sorted({w.process_name for w in protected}))
        return ActionResult.failure(
            f"{names} is part of Windows itself — I won't close it.",
            error_code="protected")

    closed, refused = [], []
    for window in matched:
        result = provider.close(window)
        (closed if result.ok else refused).append(window.label())

    if not closed:
        return ActionResult.failure(
            f"I couldn't close {wanted}.", error_code="close_failed",
            attempted=[w.label() for w in matched])

    # "Asked to close", not "closed": WM_CLOSE is a request, and an editor with
    # unsaved work will still be sitting there with a prompt.
    count = f"{len(closed)} windows" if len(closed) != 1 else "1 window"
    tail = f" {len(refused)} wouldn't close." if refused else ""
    return ActionResult.success(
        f"Asked {wanted} to close ({count}). Anything with unsaved work will "
        f"ask you first.{tail}",
        closed=closed, refused=refused, count=len(closed))


def _count_application_windows(ctx: ActionContext,
                               params: Dict[str, object]) -> ActionResult:
    provider = _windows(ctx)
    if not provider.is_available():
        return _unavailable(provider)
    wanted = str(params.get("name") or "").strip()
    matched = _windows_for(provider, wanted, _applications(ctx))
    if not matched:
        return ActionResult.success(f"{wanted} doesn't appear to be running.",
                                    count=0, windows=[])
    listed = "; ".join(w.label() for w in matched[:MAX_SUMMARY])
    return ActionResult.success(
        f"{wanted} has {len(matched)} window(s) open: {listed}.",
        count=len(matched), windows=[w.as_dict() for w in matched])


# ------------------------------------------------------------------ bulk close

def _open_applications(provider, *, exceptions: Sequence[str] = ()) -> List:
    """Closable application windows, with the protected ones already removed."""
    excluded = {e.strip().lower() for e in exceptions if e.strip()}
    aarma = _aarma_process()
    candidates = []
    for window in provider.list_windows():
        process = (window.process_name or "").lower()
        if _is_protected(process) or process == aarma:
            continue
        stem = process[:-4] if process.endswith(".exe") else process
        title = (window.title or "").lower()
        if any(term == stem or term in stem or term in title for term in excluded):
            continue
        candidates.append(window)
    return candidates


def _parse_exceptions(raw: object) -> List[str]:
    text = str(raw or "")
    if not text.strip():
        return []
    import re

    parts = re.split(r",|\band\b|\bor\b", text)
    return [p.strip() for p in parts if p.strip()]


def _close_everything(ctx: ActionContext,
                      params: Dict[str, object]) -> ActionResult:
    """Close every open application, except any the user named.

    Graded HIGH, so the confirmation broker asks before this ever runs. The
    summary the user sees is built by the *precheck* below, so they are told
    how many and which before they answer rather than after.
    """
    provider = _windows(ctx)
    if not provider.is_available():
        return _unavailable(provider)

    exceptions = _parse_exceptions(params.get("except"))
    candidates = _open_applications(provider, exceptions=exceptions)
    if not candidates:
        return ActionResult.success("There's nothing open that I would close.",
                                    closed=[], count=0)

    closed, refused = [], []
    for window in candidates:
        result = provider.close(window)
        (closed if result.ok else refused).append(window.label())

    kept = f" I left {', '.join(exceptions)} alone." if exceptions else ""
    tail = f" {len(refused)} wouldn't close." if refused else ""
    logger.info("Bulk close performed", extra={"context": {
        "closed": len(closed), "refused": len(refused),
        "exceptions": exceptions}})
    return ActionResult.success(
        f"Asked {len(closed)} applications to close.{kept}{tail} Anything with "
        "unsaved work will ask you first.",
        closed=closed, refused=refused, count=len(closed),
        exceptions=exceptions)


def _precheck_close_everything(ctx: ActionContext,
                               params: Dict[str, object]) -> Optional[str]:
    """Refuse early when there is nothing to close.

    Runs before the confirmation, so the user is never asked to approve
    closing nothing.
    """
    provider = _windows(ctx)
    if not provider.is_available():
        return provider.availability().detail
    candidates = _open_applications(
        provider, exceptions=_parse_exceptions(params.get("except")))
    if not candidates:
        return "There's nothing open that I would close."
    return None


def _application_label(window) -> str:
    """A name a person would recognise: "Chrome", not "chrome.exe"."""
    process = (getattr(window, "process_name", "") or "").strip()
    stem = process[:-4] if process.lower().endswith(".exe") else process
    return stem.title() if stem else (getattr(window, "title", "") or "an application")


def _summarise_close_everything(ctx: ActionContext,
                                params: Dict[str, object]) -> str:
    """Name what would close, because that is what is being approved.

    Read at confirmation time rather than at parse time, so the list is the one
    the user is looking at. If the windows cannot be read the generic summary
    takes over -- an empty list here would read as "nothing would close", which
    is the opposite of the truth.
    """
    provider = _windows(ctx)
    if not provider.is_available():
        return ""
    exceptions = _parse_exceptions(params.get("except"))
    names = sorted({_application_label(window) for window
                    in _open_applications(provider, exceptions=exceptions)})
    if not names:
        return ""
    spared = f" Leaving open: {', '.join(sorted(exceptions))}." if exceptions else ""
    return (f"Close {len(names)} application(s): {', '.join(names)}.{spared} "
            "Unsaved work in them may be lost.")


# ------------------------------------------------------------ shell execution

def _execute_shell(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Run a command line the user explicitly asked to run.

    This is the only action in AARMA that executes an arbitrary command, and
    it is deliberately awkward to reach: HIGH risk so it always confirms, its
    own permission, and no route from the nearest-intent resolver -- "run this"
    does not resolve here, because a sentence that vaguely suggests execution
    must never *be* execution.

    ``shell=False`` with a parsed argument list, so the command is one program
    with arguments rather than a string a shell will re-interpret. No
    elevation, ever: a command needing administrator rights fails and says so.
    """
    command = str(params.get("command") or "").strip()
    if not command:
        return ActionResult.failure("What command should I run?",
                                    error_code="missing_command")

    working = None
    raw_directory = str(params.get("directory") or "").strip()
    if raw_directory:
        from workspace.policy import Operation, WorkspaceAccessPolicy

        policy = getattr(ctx, "workspace_policy", None) or \
            WorkspaceAccessPolicy(ctx.config)
        resolved = policy.resolve(raw_directory)
        if resolved is None:
            return ActionResult.failure(
                f"{raw_directory} is outside the folders you've approved.",
                error_code="outside_workspace")
        decision = policy.decide(Operation.EXECUTE, resolved)
        if decision.refused:
            return ActionResult.failure(decision.reason, error_code="refused")
        working = str(resolved)

    import shlex

    try:
        argv = shlex.split(command, posix=False)
    except ValueError as exc:
        return ActionResult.failure(f"I couldn't read that command: {exc}",
                                    error_code="unparsable")
    if not argv:
        return ActionResult.failure("That command was empty.",
                                    error_code="missing_command")

    runner = getattr(ctx, "process_runner", None) or subprocess.run
    # The command itself is logged; it was typed by the user and approved by
    # them, and without it an audit trail of "a command ran" says nothing.
    logger.info("Executing a confirmed shell command", extra={
        "context": {"argv": argv[:8], "cwd": working or ""}})
    try:
        completed = runner(argv, capture_output=True, text=True, timeout=60,
                           cwd=working, shell=False, check=False)
    except FileNotFoundError:
        return ActionResult.failure(f"{argv[0]} isn't something I can run.",
                                    error_code="not_found")
    except subprocess.TimeoutExpired:
        return ActionResult.failure("That command took too long and I stopped it.",
                                    error_code="timeout")
    except Exception as exc:  # noqa: BLE001 - one failed action, not a crash
        logger.error("Shell command failed", extra={"context": {"error": str(exc)}})
        return ActionResult.failure("I couldn't run that command.",
                                    error_code="execution_failed")

    code = getattr(completed, "returncode", 0)
    output = (getattr(completed, "stdout", "") or "").strip()
    errors = (getattr(completed, "stderr", "") or "").strip()
    shown = (output or errors)[:1200]
    if code not in (0, None):
        return ActionResult.failure(
            f"The command exited with code {code}." + (f"\n{shown}" if shown else ""),
            error_code="nonzero_exit", returncode=code, output=shown)
    return ActionResult.success(
        (shown or "The command finished with no output."),
        returncode=code, output=shown, command=argv[0])


def get_actions() -> List[ActionSpec]:
    return [
        ActionSpec(
            name="close_application_windows",
            description="Close an application and all of its windows.",
            category="Windows",
            # MEDIUM: WM_CLOSE lets each application prompt about unsaved work,
            # so this is not the irreversible act a forced kill would be.
            risk=RiskLevel.MEDIUM, handler=_close_application_windows,
            permissions=(Permission.SYSTEM_CONTROL,),
            parameters=(ParameterSpec(name="name", type=ParameterType.STRING,
                                      description="which application",
                                      max_length=80),),
            platforms=("win32",),
            examples=("close Chrome", "close VS Code", "quit Notepad"),
        ),
        ActionSpec(
            name="count_application_windows",
            description="Say how many windows an application has open.",
            category="Windows", risk=RiskLevel.LOW,
            handler=_count_application_windows,
            parameters=(ParameterSpec(name="name", type=ParameterType.STRING,
                                      description="which application",
                                      max_length=80),),
            platforms=("win32",),
            examples=("how many VS Code windows are open",),
        ),
        ActionSpec(
            name="close_all_applications",
            description="Close every open application.",
            category="Windows",
            # HIGH, and it must stay HIGH: this is the most far-reaching
            # desktop action AARMA has, and the confirmation summary is the
            # only chance the user gets to see what it would do.
            risk=RiskLevel.HIGH, handler=_close_everything,
            permissions=(Permission.SYSTEM_CONTROL,),
            precheck=_precheck_close_everything,
            summarise=_summarise_close_everything,
            parameters=(ParameterSpec(
                name="except", type=ParameterType.STRING, required=False,
                default="", max_length=200,
                description="applications to leave open"),),
            platforms=("win32",),
            examples=("close all applications", "close everything",
                      "close everything except Chrome"),
        ),
        ActionSpec(
            name="execute_shell_command",
            description="Run a command you give me, after confirming it.",
            category="System",
            risk=RiskLevel.HIGH, handler=_execute_shell,
            permissions=(Permission.SYSTEM_CONTROL,),
            parameters=(
                ParameterSpec(name="command", type=ParameterType.TEXT,
                              description="the exact command to run",
                              max_length=500),
                ParameterSpec(name="directory", type=ParameterType.STRING,
                              required=False, default="", max_length=400,
                              description="where to run it"),
            ),
            platforms=("win32",),
            # Deliberately no natural phrasings: this action is reached by
            # asking for it explicitly, never by a sentence that merely sounds
            # like it might mean running something.
            examples=("run the command git status",),
        ),
    ]
