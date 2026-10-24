"""Launching and closing applications from a fixed approved catalogue.

The catalogue is the point.  AARMA never receives a command line from the user
(or, in Phase 2, from a language model) and runs it -- it receives a *name*,
looks that name up in :data:`APPLICATION_CATALOG`, and executes only the
resolved path it found itself.  An application that is not in the catalogue
cannot be launched however the request is phrased.

Resolution tries, in order: a Windows shell URI (``calculator:``), the
executable on ``PATH``, then a short list of conventional install locations.
If none of those find the program, the action fails with a plain explanation
rather than guessing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

logger = get_logger("actions.applications")


@dataclass(frozen=True)
class ApplicationEntry:
    """One approved application and every way AARMA knows to find it."""

    key: str
    display_name: str
    aliases: Tuple[str, ...]
    executables: Tuple[str, ...] = ()
    uri: Optional[str] = None
    windows_paths: Tuple[str, ...] = ()
    process_name: Optional[str] = None
    #: A command shell.  Opening one is a window on screen; it is *not* running
    #: anything.  Flagged so the two can never be confused -- see
    #: :func:`_open_terminal` for why they are separate actions.
    terminal: bool = False
    #: Arguments AARMA supplies itself when launching.  Never user input.
    arguments: Tuple[str, ...] = ()


def _entry(key: str, display: str, aliases: Tuple[str, ...], **kwargs: object) -> ApplicationEntry:
    return ApplicationEntry(key=key, display_name=display, aliases=aliases, **kwargs)  # type: ignore[arg-type]


#: The allow-list.
#:
#: Terminals *are* in here, flagged ``terminal=True``, and that deserves an
#: explanation because an earlier build left them out entirely.  Launching
#: ``cmd.exe`` with no arguments opens an interactive prompt; it does not run
#: anything.  The dangerous capability is executing a command line the user
#: spoke, and AARMA has no action that does that -- there is no
#: ``execute_shell_command`` in the registry, so there is nothing for a
#: cleverly phrased sentence to reach.  Opening the window is graded higher
#: than opening Notepad all the same, and lives in its own action with its own
#: permission, so the two can never be conflated by a rule change.
APPLICATION_CATALOG: Dict[str, ApplicationEntry] = {
    entry.key: entry
    for entry in (
        _entry("chrome", "Google Chrome", ("chrome", "google chrome"),
               executables=("chrome.exe", "chrome", "google-chrome"),
               windows_paths=(
                   r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
                   r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe",
                   r"%LocalAppData%\Google\Chrome\Application\chrome.exe",
               ),
               process_name="chrome.exe"),
        _entry("edge", "Microsoft Edge", ("edge", "microsoft edge", "msedge"),
               executables=("msedge.exe",),
               windows_paths=(
                   r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
                   r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
               ),
               process_name="msedge.exe"),
        _entry("firefox", "Mozilla Firefox", ("firefox", "mozilla firefox"),
               executables=("firefox.exe", "firefox"),
               windows_paths=(
                   r"%ProgramFiles%\Mozilla Firefox\firefox.exe",
                   r"%ProgramFiles(x86)%\Mozilla Firefox\firefox.exe",
               ),
               process_name="firefox.exe"),
        _entry("notepad", "Notepad", ("notepad", "note pad", "text editor"),
               executables=("notepad.exe",), process_name="notepad.exe"),
        _entry("calculator", "Calculator", ("calculator", "calc"),
               uri="calculator:", executables=("calc.exe",),
               process_name="CalculatorApp.exe"),
        _entry("explorer", "File Explorer", ("file explorer", "explorer", "files", "my computer"),
               executables=("explorer.exe",)),
        _entry("paint", "Paint", ("paint", "ms paint", "mspaint"),
               executables=("mspaint.exe",), process_name="mspaint.exe"),
        _entry("settings", "Windows Settings", ("settings", "windows settings", "system settings"),
               uri="ms-settings:"),
        _entry("photos", "Photos", ("photos", "photo viewer"), uri="ms-photos:"),
        _entry("vscode", "Visual Studio Code", ("vs code", "vscode", "visual studio code"),
               executables=("code.cmd", "code.exe", "code"),
               windows_paths=(r"%LocalAppData%\Programs\Microsoft VS Code\Code.exe",),
               process_name="Code.exe"),
        _entry("spotify", "Spotify", ("spotify",),
               executables=("spotify.exe",),
               windows_paths=(r"%AppData%\Spotify\Spotify.exe",),
               process_name="Spotify.exe"),
        _entry("taskmanager", "Task Manager",
               ("task manager", "taskmgr", "task manger", "processes"),
               executables=("taskmgr.exe",), process_name="Taskmgr.exe"),
        _entry("controlpanel", "Control Panel", ("control panel", "control"),
               executables=("control.exe",)),
        _entry("snippingtool", "Snipping Tool",
               ("snipping tool", "snip", "snip and sketch"),
               uri="ms-screenclip:", executables=("SnippingTool.exe",)),

        # --- terminals ---------------------------------------------------
        _entry("cmd", "Command Prompt",
               ("cmd", "command prompt", "command line", "dos prompt",
                "cmd prompt", "the command prompt"),
               executables=("cmd.exe",), process_name="cmd.exe", terminal=True),
        _entry("powershell", "PowerShell",
               ("powershell", "power shell", "windows powershell", "ps"),
               executables=("powershell.exe", "pwsh.exe"),
               process_name="powershell.exe", terminal=True),
        _entry("windowsterminal", "Windows Terminal",
               ("windows terminal", "terminal", "wt", "the terminal"),
               executables=("wt.exe",), process_name="WindowsTerminal.exe",
               terminal=True),
    )
}

#: Alias -> catalogue key, built once so lookup is a dictionary hit.
_ALIAS_INDEX: Dict[str, str] = {
    alias: entry.key
    for entry in APPLICATION_CATALOG.values()
    for alias in (entry.key, entry.display_name.lower(), *entry.aliases)
}


def resolve_entry(name: str) -> Optional[ApplicationEntry]:
    """Map what the user said onto a catalogue entry, or ``None``."""
    needle = (name or "").strip().lower()
    if not needle:
        return None
    for filler in ("the ", "my ", "app ", "application "):
        if needle.startswith(filler):
            needle = needle[len(filler):]
    needle = needle.removesuffix(" app").removesuffix(" application").strip()
    key = _ALIAS_INDEX.get(needle)
    return APPLICATION_CATALOG.get(key) if key else None


def resolve_program(name: str) -> Optional[ApplicationEntry]:
    """A catalogue entry that is *not* a terminal, or ``None``.

    Used by the intent layer so that "open cmd" never resolves as an ordinary
    application: the terminal rule has to claim it, or nothing does.
    """
    entry = resolve_entry(name)
    return entry if entry is not None and not entry.terminal else None


def resolve_terminal(name: str) -> Optional[ApplicationEntry]:
    """A catalogue entry that *is* a terminal, or ``None``."""
    entry = resolve_entry(name)
    return entry if entry is not None and entry.terminal else None


def terminal_entries() -> List[ApplicationEntry]:
    """Every approved shell, in catalogue order."""
    return [entry for entry in APPLICATION_CATALOG.values() if entry.terminal]


def resolve_target(entry: ApplicationEntry) -> Optional[str]:
    """Find something concrete to launch for ``entry``, or ``None``."""
    if entry.uri and sys.platform == "win32":
        return entry.uri
    for executable in entry.executables:
        found = shutil.which(executable)
        if found:
            return found
    if sys.platform == "win32":
        for template in entry.windows_paths:
            candidate = Path(os.path.expandvars(template))
            if candidate.is_file():
                return str(candidate)
    return None


def _launch(target: str, arguments: Tuple[str, ...] = ()) -> None:
    """Start ``target`` without a shell.

    ``shell=False`` and an explicit argument list are what keep this from being
    a command-injection surface: ``target`` is a path this module resolved, not
    a string the user typed, and ``arguments`` comes from the catalogue rather
    than from anything spoken.  No registered action has a parameter that
    reaches either of them.
    """
    if sys.platform == "win32":
        if arguments:
            subprocess.Popen([target, *arguments],  # noqa: S603 - catalogue argv
                             close_fds=True)
            return
        # startfile handles both shell URIs and executables, and does not block.
        os.startfile(target)  # type: ignore[attr-defined]  # noqa: S606 - resolved path only
        return
    subprocess.Popen([target, *arguments], stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


def available_applications() -> List[ApplicationEntry]:
    """Catalogue entries that can actually be launched on this machine."""
    return [entry for entry in APPLICATION_CATALOG.values() if resolve_target(entry) is not None]


# --------------------------------------------------------------------- handlers

def _start(entry: ApplicationEntry) -> ActionResult:
    """Resolve and launch one catalogue entry, reporting honestly either way."""
    target = resolve_target(entry)
    if target is None:
        return ActionResult.failure(
            f"{entry.display_name} is approved, but I couldn't find it installed on this machine.",
            error_code="app_not_found",
        )
    try:
        _launch(target, entry.arguments)
    except OSError as exc:
        logger.error("Application launch failed",
                     extra={"context": {"app": entry.key, "error": str(exc)}})
        return ActionResult.failure(
            f"I couldn't start {entry.display_name}.", error_code="launch_failed"
        )
    logger.info("Application launched", extra={"context": {"app": entry.key}})
    return ActionResult.success(f"{entry.display_name} opened.",
                                application=entry.key,
                                display_name=entry.display_name,
                                terminal=entry.terminal)


def _open_application(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    name = str(params["name"])
    entry = resolve_entry(name)
    if entry is None:
        known = ", ".join(sorted(e.display_name for e in APPLICATION_CATALOG.values()
                                 if not e.terminal))
        return ActionResult.failure(
            f"'{name}' isn't in my approved application list. I can open: {known}.",
            error_code="app_not_approved",
        )
    if entry.terminal:
        # Refused here rather than quietly handled, so a caller that reached
        # this action with a shell name -- a future model proposing a tool
        # call, say -- is corrected instead of served.
        return ActionResult.failure(
            f"{entry.display_name} is a command shell, so it opens through my "
            "terminal action instead. Ask me to open the Command Prompt and "
            "I will use that.",
            error_code="use_open_terminal", application=entry.key,
        )
    return _start(entry)


def _open_terminal(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Open a command shell window.

    This opens a *window*.  It does not run a command, and there is
    deliberately no parameter through which one could be supplied: the only
    argument is a shell name that must resolve inside
    :data:`APPLICATION_CATALOG`.  Executing a command line the user spoke is a
    separate capability this build does not have, and keeping the two as
    different actions is what stops one quietly becoming the other.
    """
    name = str(params.get("name") or "").strip() or "command prompt"
    entry = resolve_terminal(name)
    if entry is None:
        known = ", ".join(e.display_name for e in terminal_entries())
        return ActionResult.failure(
            f"I don't have a terminal called '{name}'. I can open: {known}.",
            error_code="terminal_not_approved",
        )
    result = _start(entry)
    if not result.ok:
        return result
    return ActionResult.success(
        f"{entry.display_name} opened. It is an empty prompt -- I do not run "
        "commands in it.", **result.data)


def _close_application(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    name = str(params["name"])
    entry = resolve_entry(name)
    if entry is None or not entry.process_name:
        return ActionResult.failure(
            f"I can't close '{name}' — it isn't one of the applications I'm allowed to manage.",
            error_code="app_not_approved",
        )
    if sys.platform != "win32":
        return ActionResult.failure(
            "Closing applications is only implemented for Windows in this build.",
            error_code="unsupported_platform",
        )

    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, whitelisted image name
            ["taskkill", "/IM", entry.process_name],
            capture_output=True, text=True, timeout=15, shell=False, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("taskkill failed", extra={"context": {"app": entry.key, "error": str(exc)}})
        return ActionResult.failure(f"I couldn't close {entry.display_name}.",
                                    error_code="close_failed")

    if completed.returncode != 0:
        return ActionResult.failure(
            f"{entry.display_name} doesn't appear to be running.", error_code="not_running"
        )
    logger.info("Application closed", extra={"context": {"app": entry.key}})
    return ActionResult.success(f"{entry.display_name} closed.", application=entry.key)


def _list_applications(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    installed = {entry.key for entry in available_applications()}
    lines = [
        f"  {'*' if entry.key in installed else '-'} {entry.display_name}"
        + (" (terminal)" if entry.terminal else "")
        for entry in APPLICATION_CATALOG.values()
    ]
    body = "\n".join(lines)
    return ActionResult.success(
        f"Applications I'm approved to open (* = found on this machine):\n{body}",
        approved=sorted(APPLICATION_CATALOG), installed=sorted(installed),
    )


def get_actions() -> List[ActionSpec]:
    name_parameter = ParameterSpec(
        name="name",
        type=ParameterType.STRING,
        description="the name of the application",
        max_length=80,
    )
    return [
        ActionSpec(
            name="open_application",
            description="Open an application from my approved list.",
            category="Desktop",
            risk=RiskLevel.LOW,
            handler=_open_application,
            permissions=(Permission.APPLICATIONS,),
            parameters=(name_parameter,),
            examples=("open chrome", "launch notepad", "start the calculator"),
        ),
        ActionSpec(
            name="close_application",
            description="Close a running application from my approved list.",
            category="Desktop",
            risk=RiskLevel.MEDIUM,
            handler=_close_application,
            permissions=(Permission.APPLICATIONS, Permission.SYSTEM_CONTROL),
            parameters=(name_parameter,),
            platforms=("win32",),
            examples=("close notepad", "quit spotify"),
        ),
        ActionSpec(
            name="open_terminal",
            description="Open a command prompt or PowerShell window.",
            category="Desktop",
            # MEDIUM rather than LOW: a shell window is a doorway to the whole
            # machine even when AARMA itself never types in it, so it is
            # something the user should be able to withhold on its own.
            risk=RiskLevel.MEDIUM,
            handler=_open_terminal,
            permissions=(Permission.APPLICATIONS, Permission.SYSTEM_CONTROL),
            parameters=(
                ParameterSpec(
                    name="name",
                    type=ParameterType.STRING,
                    description="which shell: Command Prompt, PowerShell or Windows Terminal",
                    required=False,
                    default="command prompt",
                    max_length=40,
                ),
            ),
            examples=("open cmd", "open command prompt", "open powershell",
                      "open windows terminal"),
        ),
        ActionSpec(
            name="list_applications",
            description="Show which applications I'm approved to open.",
            category="Desktop",
            risk=RiskLevel.LOW,
            handler=_list_applications,
            examples=("what apps can you open", "list applications"),
        ),
    ]
