"""Workspace, repository and terminal-in-folder actions.

These are the actions that answer "open my AARMA repo" -- the ones where the
*target is discovered* rather than named in a catalogue.  That changes what
correctness means: a handler here can be wrong by finding the wrong project,
not just by failing, so every one of them either finds exactly one match or
asks.

The rule that keeps this safe is that discovery and permission are separate.
:mod:`workspace.repositories` finds candidates anywhere inside the approved
roots; :class:`~workspace.policy.WorkspaceAccessPolicy` decides whether the
chosen one may be touched; and the registry decides whether the *action* is
allowed at all.  A broader search does not widen any of the three.

Nothing here executes project code.  "Open a terminal in this repo" opens a
shell with a working directory -- it does not run the project, because what a
project's run command is cannot be inferred safely and guessing it would mean
executing whatever a directory happens to contain.
"""

from __future__ import annotations

import os
import subprocess
import sys
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

logger = get_logger("actions.workspace")

#: How many candidates to name when asking the user to choose.
MAX_CHOICES = 6

#: Terminals that accept a working directory, and how to give it to them.
#: ``cmd`` needs ``/K cd /d <path>`` because it has no working-directory flag
#: of its own; the others take one directly.
_TERMINAL_LAUNCH: Dict[str, Tuple[str, ...]] = {
    "cmd": ("/K", "cd", "/d"),
    "powershell": ("-NoExit", "-Command", "Set-Location", "-LiteralPath"),
    "windowsterminal": ("-d",),
}


def _policy(ctx: ActionContext):
    injected = getattr(ctx, "workspace_policy", None)
    if injected is not None:
        return injected
    from workspace.policy import WorkspaceAccessPolicy

    return WorkspaceAccessPolicy(ctx.config)


def _repositories(ctx: ActionContext):
    injected = getattr(ctx, "repository_index", None)
    if injected is not None:
        return injected
    from workspace.repositories import RepositoryIndex

    return RepositoryIndex(_policy(ctx))


def _applications(ctx: ActionContext):
    injected = getattr(ctx, "application_index", None)
    if injected is not None:
        return injected
    from workspace.applications import ApplicationIndex

    return ApplicationIndex()


def _choose(candidates, what: str, wanted: str) -> ActionResult:
    """The ambiguity reply, shared so every one of them reads the same."""
    listed = "; ".join(f"{c.name} ({c.path})" for c in candidates[:MAX_CHOICES])
    return ActionResult.failure(
        f"I found {len(candidates)} {what} matching \"{wanted}\": {listed}. "
        "Which one?",
        error_code="ambiguous",
        candidates=[c.as_dict() for c in candidates[:MAX_CHOICES]])


# ------------------------------------------------------------------ workspace

def _list_workspace(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    policy = _policy(ctx)
    roots = policy.roots()
    if not roots:
        return ActionResult.success(
            "No workspace folders are set up, so I can only use the folders "
            "Windows reports for your account.", roots=[], count=0)
    listed = "; ".join(str(root) for root in roots)
    return ActionResult.success(
        f"I can work in {len(roots)} folders: {listed}.",
        roots=[str(root) for root in roots], count=len(roots))


def _add_workspace_root(ctx: ActionContext,
                        params: Dict[str, object]) -> ActionResult:
    added, message = _policy(ctx).add_root(str(params.get("path") or ""))
    if not added:
        return ActionResult.failure(message, error_code="not_added")
    return ActionResult.success(message, path=str(params.get("path")))


def _remove_workspace_root(ctx: ActionContext,
                           params: Dict[str, object]) -> ActionResult:
    removed, message = _policy(ctx).remove_root(str(params.get("path") or ""))
    if not removed:
        return ActionResult.failure(message, error_code="not_removed")
    return ActionResult.success(message)


# ---------------------------------------------------------------- repositories

def _list_repositories(ctx: ActionContext,
                       params: Dict[str, object]) -> ActionResult:
    index = _repositories(ctx)
    repositories = index.all()
    if not repositories:
        return ActionResult.success(
            "I couldn't find any projects in your workspace folders.",
            repositories=[], count=0)
    shown = repositories[:MAX_CHOICES * 2]
    listed = "; ".join(f"{r.name} ({r.kind})" for r in shown)
    note = (" The scan hit its limit, so this may be incomplete."
            if getattr(index, "truncated", False) else "")
    return ActionResult.success(
        f"I found {len(repositories)} projects: {listed}.{note}",
        repositories=[r.as_dict() for r in shown], count=len(repositories),
        truncated=bool(getattr(index, "truncated", False)))


def _resolve_repository(ctx: ActionContext, wanted: str):
    """``(repository, refusal)`` -- never a guess between two candidates."""
    index = _repositories(ctx)
    cleaned = " ".join(str(wanted or "").split())

    if not cleaned or cleaned.lower() in {"this", "current", "the current one",
                                          "here", "it", "that"}:
        remembered = getattr(ctx, "last_repository", None)
        if remembered is not None:
            return remembered, None
        # No fallback to the most recently modified project. "The one you were
        # last in" has to mean one the user actually established in the
        # conversation; picking the newest directory on disk instead opened a
        # terminal in AARMA for a user who had named nothing at all, which is
        # an action on a target nobody chose.
        return None, ActionResult.failure(
            "I don't know which project you mean. Name it, or open one first.",
            error_code="no_repository")

    matches = index.find(cleaned)
    if not matches:
        return None, ActionResult.failure(
            f"I couldn't find a project called \"{cleaned}\" in your workspace.",
            error_code="not_found")
    if len(matches) > 1:
        return None, _choose(matches, "projects", cleaned)
    return matches[0], None


def _open_repository(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    repository, refusal = _resolve_repository(ctx, params.get("name"))
    if refusal is not None:
        return refusal

    from workspace.policy import Operation

    decision = _policy(ctx).decide(Operation.READ, repository.path)
    if decision.refused:
        return ActionResult.failure(decision.reason, error_code="refused")

    try:
        if sys.platform == "win32":
            os.startfile(str(repository.path))  # noqa: S606 - discovered, validated path
        else:
            subprocess.Popen(["xdg-open", str(repository.path)])  # noqa: S603
    except OSError as exc:
        logger.error("Could not open repository", extra={
            "context": {"error": str(exc)}})
        return ActionResult.failure(f"I couldn't open {repository.name}.",
                                    error_code="open_failed")
    logger.info("Repository opened", extra={"context": {
        "name": repository.name, "kind": repository.kind}})
    return ActionResult.success(f"Opening {repository.name}.",
                                **repository.as_dict())


def _open_repository_in_editor(ctx: ActionContext,
                               params: Dict[str, object]) -> ActionResult:
    """Open a project in an IDE, using that IDE's documented launcher."""
    repository, refusal = _resolve_repository(ctx, params.get("name"))
    if refusal is not None:
        return refusal

    wanted_editor = str(params.get("editor") or "").strip() or "vs code"
    from actions.applications import resolve_entry, resolve_target

    entry = resolve_entry(wanted_editor)
    target = resolve_target(entry) if entry is not None else None
    if target is None:
        matches = _applications(ctx).find(wanted_editor)
        if not matches:
            return ActionResult.failure(
                f"I couldn't find {wanted_editor} installed on this machine.",
                error_code="editor_not_found")
        target = str(matches[0].target)
        editor_name = matches[0].display_name
    else:
        editor_name = entry.display_name

    if not repository.path.is_dir():
        return ActionResult.failure(
            f"{repository.name} isn't there any more.", error_code="not_found")

    try:
        # An explicit argv with a path this module resolved -- never a command
        # string, and never anything the user typed.
        subprocess.Popen([target, str(repository.path)],  # noqa: S603
                         close_fds=True)
    except OSError as exc:
        logger.error("Could not launch editor", extra={
            "context": {"editor": editor_name, "error": str(exc)}})
        return ActionResult.failure(
            f"I couldn't start {editor_name}.", error_code="launch_failed")
    logger.info("Repository opened in editor", extra={"context": {
        "editor": editor_name, "repository": repository.name}})
    return ActionResult.success(
        f"Opening {repository.name} in {editor_name}.",
        editor=editor_name, **repository.as_dict())


def _open_terminal_here(ctx: ActionContext,
                        params: Dict[str, object]) -> ActionResult:
    """Open a shell with a project as its working directory.

    Opens a prompt; runs nothing.  The working directory is a path this module
    resolved and the policy approved, and the arguments come from
    :data:`_TERMINAL_LAUNCH` rather than from anything spoken -- so there is no
    route from a sentence to a command line.
    """
    repository, refusal = _resolve_repository(ctx, params.get("name"))
    if refusal is not None:
        return refusal

    from workspace.policy import Operation

    decision = _policy(ctx).decide(Operation.READ, repository.path)
    if decision.refused:
        return ActionResult.failure(decision.reason, error_code="refused")

    from actions.applications import resolve_target, resolve_terminal

    wanted = str(params.get("terminal") or "").strip() or "windows terminal"
    entry = resolve_terminal(wanted) or resolve_terminal("command prompt")
    if entry is None:
        return ActionResult.failure("I don't have a terminal I can open.",
                                    error_code="terminal_not_approved")
    target = resolve_target(entry)
    if target is None:
        return ActionResult.failure(
            f"{entry.display_name} isn't installed on this machine.",
            error_code="terminal_not_found")

    arguments = list(_TERMINAL_LAUNCH.get(entry.key, ()))
    argv = [target, *arguments, str(repository.path)]
    try:
        subprocess.Popen(argv, cwd=str(repository.path),  # noqa: S603
                         close_fds=True)
    except OSError as exc:
        logger.error("Could not open terminal", extra={
            "context": {"error": str(exc)}})
        return ActionResult.failure(
            f"I couldn't open {entry.display_name} there.",
            error_code="launch_failed")
    logger.info("Terminal opened in folder", extra={"context": {
        "terminal": entry.key, "repository": repository.name}})
    return ActionResult.success(
        f"{entry.display_name} opened in {repository.name}. It's an empty "
        "prompt — I don't run anything in it.",
        terminal=entry.key, **repository.as_dict())


# ------------------------------------------------------------ file discovery

def _files(ctx: ActionContext):
    injected = getattr(ctx, "file_index", None)
    if injected is not None:
        return injected
    from workspace.files import FileIndex

    return FileIndex(_policy(ctx))


def _describe_files(matches) -> str:
    return "; ".join(f"{m.name} ({m.path.parent})" for m in matches[:MAX_CHOICES])


def _find_file(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Find a file by what the user called it, and ask when unsure.

    Never opens anything on its own: "find my resume" with three plausible
    resumes is a question, and a function that picks one has answered a
    different question than the one asked.
    """
    from workspace.files import category_for

    wanted = str(params.get("name") or "").strip()
    category = category_for(str(params.get("kind") or ""))
    index = _files(ctx)

    if not wanted and category:
        matches = index.latest_of(category)
        if not matches:
            return ActionResult.failure(
                f"I couldn't find any {category} files in your workspace.",
                error_code="not_found")
        newest = matches[0]
        return ActionResult.success(
            f"The most recent {category} is {newest.name}, in {newest.path.parent}.",
            path=str(newest.path), name=newest.name,
            candidates=[m.as_dict() for m in matches[:MAX_CHOICES]],
            count=len(matches))

    if not wanted:
        return ActionResult.failure("What should I look for?",
                                    error_code="missing_target")

    matches = index.find_named(wanted)
    if not matches:
        note = (" I hit my search limit, so there may be more."
                if getattr(index, "truncated", False) else "")
        return ActionResult.failure(
            f"I couldn't find a file matching \"{wanted}\" in your workspace.{note}",
            error_code="not_found")
    if len(matches) > 1:
        return ActionResult.failure(
            f"I found {len(matches)} files matching \"{wanted}\": "
            f"{_describe_files(matches)}. Which one?",
            error_code="ambiguous",
            candidates=[m.as_dict() for m in matches[:MAX_CHOICES]])

    found = matches[0]
    return ActionResult.success(
        f"Found {found.name} in {found.path.parent}.",
        path=str(found.path), name=found.name, count=1)


def _open_found_file(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Open a file described rather than named, once it is unambiguous."""
    located = _find_file(ctx, params)
    if not located.ok:
        return located

    from workspace.policy import Operation

    path = Path(str(located.data.get("path") or ""))
    decision = _policy(ctx).decide(Operation.READ, path)
    if decision.refused:
        return ActionResult.failure(decision.reason, error_code="refused")

    from actions.filesystem import EXECUTABLE_EXTENSIONS, _open_with_default_app

    if path.suffix.lower() in EXECUTABLE_EXTENSIONS:
        return ActionResult.failure(
            f"I won't open {path.name} — running programs isn't something I do.",
            error_code="executable_blocked")
    try:
        _open_with_default_app(path)
    except OSError as exc:
        logger.error("Could not open file", extra={"context": {"error": str(exc)}})
        return ActionResult.failure(f"I couldn't open {path.name}.",
                                    error_code="open_failed")
    return ActionResult.success(f"Opening {path.name}.", path=str(path),
                                name=path.name)


def get_actions() -> List[ActionSpec]:
    name = ParameterSpec(
        name="name", type=ParameterType.STRING, required=False, default="",
        description="which project — blank means the one named earlier in this conversation",
        max_length=120)
    return [
        ActionSpec(
            name="list_workspace_roots",
            description="Show the folders I'm allowed to work in.",
            category="Workspace", risk=RiskLevel.LOW, handler=_list_workspace,
            examples=("what folders can you work in", "show my workspace"),
        ),
        ActionSpec(
            name="add_workspace_root",
            description="Approve another folder for me to work in.",
            category="Workspace", risk=RiskLevel.MEDIUM,
            handler=_add_workspace_root,
            permissions=(Permission.FILESYSTEM,),
            parameters=(ParameterSpec(name="path", type=ParameterType.STRING,
                                      description="the folder to add",
                                      max_length=400),),
            examples=("add C:/Projects to my workspace",),
        ),
        ActionSpec(
            name="remove_workspace_root",
            description="Stop working in one of your workspace folders.",
            category="Workspace", risk=RiskLevel.MEDIUM,
            handler=_remove_workspace_root,
            permissions=(Permission.FILESYSTEM,),
            parameters=(ParameterSpec(name="path", type=ParameterType.STRING,
                                      description="the folder to remove",
                                      max_length=400),),
            examples=("remove C:/Projects from my workspace",),
        ),
        ActionSpec(
            name="list_repositories",
            description="List the projects in your workspace.",
            category="Workspace", risk=RiskLevel.LOW,
            handler=_list_repositories,
            permissions=(Permission.FILESYSTEM,),
            examples=("show my repositories", "list my projects",
                      "what projects do I have"),
        ),
        ActionSpec(
            name="open_repository",
            description="Open one of your projects.",
            category="Workspace", risk=RiskLevel.LOW, handler=_open_repository,
            permissions=(Permission.FILESYSTEM,), parameters=(name,),
            examples=("open my AARMA repository", "open my Python project",
                      "open the project I was working on"),
        ),
        ActionSpec(
            name="open_repository_in_editor",
            description="Open one of your projects in a code editor.",
            category="Workspace", risk=RiskLevel.LOW,
            handler=_open_repository_in_editor,
            permissions=(Permission.FILESYSTEM, Permission.APPLICATIONS),
            parameters=(name,
                        ParameterSpec(name="editor", required=False,
                                      default="vs code", max_length=60,
                                      description="which editor")),
            examples=("open AARMA in VS Code", "open this repo in VS Code"),
        ),
        ActionSpec(
            name="find_file",
            description="Find a file in your workspace by name or kind.",
            category="Workspace", risk=RiskLevel.LOW, handler=_find_file,
            permissions=(Permission.FILESYSTEM,),
            parameters=(
                ParameterSpec(name="name", type=ParameterType.STRING,
                              required=False, default="", max_length=120,
                              description="what the file is called"),
                ParameterSpec(name="kind", type=ParameterType.STRING,
                              required=False, default="", max_length=40,
                              description="a kind, such as pdf or document"),
            ),
            examples=("find my resume", "find the latest PDF",
                      "where is my report"),
        ),
        ActionSpec(
            name="open_found_file",
            description="Open a file you describe rather than name exactly.",
            category="Workspace", risk=RiskLevel.LOW, handler=_open_found_file,
            permissions=(Permission.FILESYSTEM,),
            parameters=(
                ParameterSpec(name="name", type=ParameterType.STRING,
                              required=False, default="", max_length=120,
                              description="what the file is called"),
                ParameterSpec(name="kind", type=ParameterType.STRING,
                              required=False, default="", max_length=40,
                              description="a kind, such as pdf"),
            ),
            examples=("open my resume", "open the latest PDF"),
        ),
        ActionSpec(
            name="open_terminal_in_repository",
            description="Open a terminal inside one of your projects.",
            category="Workspace",
            # MEDIUM, matching open_terminal: a shell is a doorway to the
            # machine even when AARMA never types in it.
            risk=RiskLevel.MEDIUM, handler=_open_terminal_here,
            permissions=(Permission.FILESYSTEM, Permission.APPLICATIONS,
                         Permission.SYSTEM_CONTROL),
            parameters=(name,
                        ParameterSpec(name="terminal", required=False,
                                      default="windows terminal", max_length=40,
                                      description="which shell")),
            examples=("open a terminal in my AARMA repo",
                      "open a terminal here", "open cmd in this project"),
        ),
    ]
