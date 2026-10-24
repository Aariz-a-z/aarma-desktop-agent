"""File and folder operations, confined to a sandbox the user agreed to.

Every path argument is resolved by :class:`security.validator.SafePathResolver`
before it is used, so these handlers can only ever act inside the user's own
content folders (Documents, Desktop, Downloads, Pictures, Music, Videos).
Traversal attempts, absolute system paths and reserved device names are
rejected by the resolver rather than being cleaned up and quietly obeyed.

Risk grading follows specification section 11: listing is LOW, anything that
writes is MEDIUM, and deletion is HIGH and therefore always confirmed.
Deletion goes to the Recycle Bin, never a permanent unlink.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from actions.registry import (
    ActionContext,
    ActionResult,
    ActionSpec,
    ParameterSpec,
    ParameterType,
)
from security.permissions import Permission
from security.validator import RiskLevel, UnsafeInputError, sanitize_component
from utils.logger import get_logger

logger = get_logger("actions.filesystem")

#: Never handed to the shell's "open with default program" verb.  Opening any
#: of these is equivalent to executing it, which no MEDIUM-risk action may do.
EXECUTABLE_EXTENSIONS = frozenset({
    ".exe", ".bat", ".cmd", ".com", ".scr", ".pif", ".msi", ".msp",
    ".ps1", ".psm1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".wsh",
    ".hta", ".cpl", ".msc", ".jar", ".reg", ".lnk", ".inf", ".sh",
})

MAX_LISTED_ENTRIES = 40
MAX_SEARCH_DEPTH = 2
MAX_SEARCH_MATCHES = 8


def _location_parameter(description: str) -> ParameterSpec:
    return ParameterSpec(
        name="location",
        type=ParameterType.STRING,
        description=description,
        required=False,
        max_length=260,
    )


def _fmt_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _open_with_default_app(path: Path) -> None:
    """Ask the desktop environment to open ``path`` with its default handler."""
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606 - sandboxed path
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(path)],  # noqa: S603 - fixed argv
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _search_for_file(base: Path, needle: str) -> List[Path]:
    """Shallow, bounded search for files whose name contains ``needle``."""
    needle = needle.lower()
    matches: List[Path] = []
    base_depth = len(base.parts)
    for root, dirnames, filenames in os.walk(base):
        root_path = Path(root)
        if len(root_path.parts) - base_depth >= MAX_SEARCH_DEPTH:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            if needle in filename.lower():
                matches.append(root_path / filename)
                if len(matches) >= MAX_SEARCH_MATCHES:
                    return matches
    return matches


# --------------------------------------------------------------------- handlers

def _create_folder(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    name = str(params["name"])
    location = params.get("location")
    try:
        target = ctx.paths.resolve_child(str(location) if location else None, name)
    except UnsafeInputError as exc:
        return ActionResult.failure(str(exc), error_code="unsafe_path")

    if target.exists():
        return ActionResult.failure(f"'{target.name}' already exists in {target.parent}.",
                                    error_code="already_exists")
    try:
        target.mkdir(parents=False)
    except OSError as exc:
        logger.error("Folder creation failed", extra={"context": {"error": str(exc)}})
        return ActionResult.failure("I couldn't create that folder.", error_code="create_failed")

    logger.info("Folder created", extra={"context": {"parent": str(target.parent)}})
    return ActionResult.success(f"Created '{target.name}' in {target.parent}.", path=str(target))


def _list_files(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    location = params.get("location")
    try:
        base = ctx.paths.resolve_base(str(location) if location else None)
    except UnsafeInputError as exc:
        return ActionResult.failure(str(exc), error_code="unsafe_path")

    try:
        entries = sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError as exc:
        logger.error("Listing failed", extra={"context": {"error": str(exc)}})
        return ActionResult.failure("I couldn't read that folder.", error_code="list_failed")

    visible = [entry for entry in entries if not entry.name.startswith(".")]
    if not visible:
        return ActionResult.success(f"{base} is empty.", path=str(base), count=0)

    shown = visible[:MAX_LISTED_ENTRIES]
    lines: List[str] = []
    for entry in shown:
        if entry.is_dir():
            lines.append(f"  [folder] {entry.name}")
        else:
            try:
                lines.append(f"  {entry.name}  ({_fmt_size(entry.stat().st_size)})")
            except OSError:
                lines.append(f"  {entry.name}")

    header = f"{len(visible)} item(s) in {base}"
    if len(visible) > len(shown):
        header += f" — showing the first {len(shown)}"
    return ActionResult.success(header + ":\n" + "\n".join(lines),
                                path=str(base), count=len(visible))


def _open_file(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    name = str(params["name"])
    location = params.get("location")
    try:
        base = ctx.paths.resolve_base(str(location) if location else None)
    except UnsafeInputError as exc:
        return ActionResult.failure(str(exc), error_code="unsafe_path")

    direct = base / name
    matches = [direct] if direct.is_file() else _search_for_file(base, name)
    if not matches:
        return ActionResult.failure(f"I couldn't find a file matching '{name}' in {base}.",
                                    error_code="not_found")
    if len(matches) > 1:
        listing = "\n".join(f"  {m.relative_to(base)}" for m in matches)
        return ActionResult.failure(
            f"I found several files matching '{name}'. Tell me which one:\n{listing}",
            error_code="ambiguous",
        )

    target = matches[0]
    if not ctx.paths.is_within_roots(target):
        return ActionResult.failure("That file is outside the folders I'm allowed to use.",
                                    error_code="unsafe_path")
    if target.suffix.lower() in EXECUTABLE_EXTENSIONS:
        return ActionResult.failure(
            f"I won't open '{target.name}' — running programs isn't something I do.",
            error_code="executable_blocked",
        )

    try:
        _open_with_default_app(target)
    except OSError as exc:
        logger.error("Open file failed", extra={"context": {"error": str(exc)}})
        return ActionResult.failure("I couldn't open that file.", error_code="open_failed")

    logger.info("File opened", extra={"context": {"suffix": target.suffix}})
    return ActionResult.success(f"Opening {target.name}.", path=str(target))


def _open_folder(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Show a folder in the file manager.

    Separate from ``open_file`` because the two fail differently: a file can be
    ambiguous and has to be searched for, while a folder is named directly and
    either is or is not one of the roots AARMA may touch.  Routing "open
    Downloads" through the file search would look for a *file* called Downloads
    and report, wrongly, that there is no such thing.
    """
    location = params.get("location")
    try:
        base = ctx.paths.resolve_base(str(location) if location else None)
    except UnsafeInputError as exc:
        return ActionResult.failure(str(exc), error_code="unsafe_path")
    if not base.is_dir():
        return ActionResult.failure(f"{base} isn't a folder I can open.",
                                    error_code="not_found")
    try:
        _open_with_default_app(base)
    except OSError as exc:
        logger.error("Open folder failed", extra={"context": {"error": str(exc)}})
        return ActionResult.failure("I couldn't open that folder.",
                                    error_code="open_failed")
    logger.info("Folder opened", extra={"context": {"name": base.name}})
    return ActionResult.success(f"Opening {base.name}.", path=str(base),
                                name=base.name)


def _rename_item(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    try:
        source = ctx.paths.resolve_existing(str(params["path"]))
        new_name = sanitize_component(str(params["new_name"]))
    except UnsafeInputError as exc:
        return ActionResult.failure(str(exc), error_code="unsafe_path")

    destination = source.with_name(new_name)
    if not ctx.paths.is_within_roots(destination):
        return ActionResult.failure("That destination is outside the folders I'm allowed to use.",
                                    error_code="unsafe_path")
    if destination.exists():
        return ActionResult.failure(f"'{new_name}' already exists there.",
                                    error_code="already_exists")
    try:
        source.rename(destination)
    except OSError as exc:
        logger.error("Rename failed", extra={"context": {"error": str(exc)}})
        return ActionResult.failure("I couldn't rename that item.", error_code="rename_failed")

    logger.info("Item renamed")
    return ActionResult.success(f"Renamed to '{new_name}'.", path=str(destination))


def _move_item(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    try:
        source = ctx.paths.resolve_existing(str(params["path"]))
        target_dir = ctx.paths.resolve_base(str(params["destination"]))
    except UnsafeInputError as exc:
        return ActionResult.failure(str(exc), error_code="unsafe_path")

    destination = target_dir / source.name
    if destination.exists():
        return ActionResult.failure(f"'{source.name}' already exists in {target_dir}.",
                                    error_code="already_exists")
    try:
        source.rename(destination)
    except OSError as exc:
        logger.error("Move failed", extra={"context": {"error": str(exc)}})
        return ActionResult.failure("I couldn't move that item.", error_code="move_failed")

    logger.info("Item moved")
    return ActionResult.success(f"Moved '{source.name}' to {target_dir}.", path=str(destination))


def _delete_item(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Send a file or folder to the Recycle Bin.

    HIGH risk, so the engine has already obtained an explicit confirmation
    before this runs.  Even then the deletion is recoverable -- a permanent
    unlink is not something this build offers at all.
    """
    try:
        target = ctx.paths.resolve_existing(str(params["path"]))
    except UnsafeInputError as exc:
        return ActionResult.failure(str(exc), error_code="unsafe_path")

    try:
        from send2trash import send2trash
    except ImportError:
        return ActionResult.failure(
            "I can only delete to the Recycle Bin, and that component isn't installed. "
            "Nothing was deleted.",
            error_code="recycle_unavailable",
        )

    try:
        send2trash(str(target))
    except Exception as exc:  # noqa: BLE001 - send2trash raises platform-specific errors
        logger.error("Delete failed", extra={"context": {"error": str(exc)}})
        return ActionResult.failure("I couldn't delete that item.", error_code="delete_failed")

    logger.info("Item sent to Recycle Bin")
    return ActionResult.success(f"Sent '{target.name}' to the Recycle Bin.", path=str(target))


def _precheck_existing_path(ctx: ActionContext, params: Dict[str, object]) -> Optional[str]:
    """Resolve the target before the user is asked to approve a deletion.

    Confirming an action that the sandbox was always going to reject teaches
    people to click through prompts, so the refusal has to come first.
    """
    try:
        ctx.paths.resolve_existing(str(params.get("path", "")))
    except UnsafeInputError as exc:
        return str(exc)
    return None


def _list_allowed_folders(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    lines = "\n".join(f"  {root}" for root in ctx.paths.roots)
    return ActionResult.success(
        f"I'm allowed to work inside these folders:\n{lines}",
        roots=[str(root) for root in ctx.paths.roots],
    )


def get_actions() -> List[ActionSpec]:
    path_parameter = ParameterSpec(
        name="path",
        type=ParameterType.PATH,
        description="the file or folder to act on",
        max_length=400,
    )
    return [
        ActionSpec(
            name="list_files",
            description="List what's inside one of your folders.",
            category="Files",
            risk=RiskLevel.LOW,
            handler=_list_files,
            permissions=(Permission.FILESYSTEM,),
            parameters=(_location_parameter("which folder to list, e.g. Documents"),),
            examples=("list files in Documents", "what's on my desktop"),
        ),
        ActionSpec(
            name="open_folder",
            description="Open one of your folders in File Explorer.",
            category="Files",
            risk=RiskLevel.LOW,
            handler=_open_folder,
            permissions=(Permission.FILESYSTEM,),
            parameters=(_location_parameter("which folder to open, e.g. Downloads"),),
            examples=("open Downloads", "open my Documents folder",
                      "show me my Desktop"),
        ),
        ActionSpec(
            name="list_allowed_folders",
            description="Show which folders I'm allowed to touch.",
            category="Files",
            risk=RiskLevel.LOW,
            handler=_list_allowed_folders,
            examples=("which folders can you access",),
        ),
        ActionSpec(
            name="create_folder",
            description="Create a new folder inside one of your folders.",
            category="Files",
            risk=RiskLevel.MEDIUM,
            handler=_create_folder,
            permissions=(Permission.FILESYSTEM,),
            parameters=(
                ParameterSpec(name="name", description="the folder name", max_length=120),
                _location_parameter("where to create it, e.g. Documents"),
            ),
            examples=("create a folder called AI Project in Documents",),
        ),
        ActionSpec(
            name="open_file",
            description="Open a file with its default application.",
            category="Files",
            risk=RiskLevel.MEDIUM,
            handler=_open_file,
            permissions=(Permission.FILESYSTEM,),
            parameters=(
                ParameterSpec(name="name", description="the file name or part of it",
                              max_length=200),
                _location_parameter("which folder to look in"),
            ),
            examples=("open resume.pdf", "open my notes in Documents"),
        ),
        ActionSpec(
            name="rename_item",
            description="Rename a file or folder.",
            category="Files",
            risk=RiskLevel.MEDIUM,
            handler=_rename_item,
            permissions=(Permission.FILESYSTEM,),
            parameters=(
                path_parameter,
                ParameterSpec(name="new_name", description="the new name", max_length=120),
            ),
        ),
        ActionSpec(
            name="move_item",
            description="Move a file or folder into another of your folders.",
            category="Files",
            risk=RiskLevel.MEDIUM,
            handler=_move_item,
            permissions=(Permission.FILESYSTEM,),
            parameters=(
                path_parameter,
                ParameterSpec(name="destination", description="the destination folder",
                              max_length=260),
            ),
        ),
        ActionSpec(
            name="delete_item",
            description="Send a file or folder to the Recycle Bin.",
            category="Files",
            risk=RiskLevel.HIGH,
            handler=_delete_item,
            permissions=(Permission.FILESYSTEM,),
            parameters=(path_parameter,),
            precheck=_precheck_existing_path,
            examples=("delete old draft.txt from Documents",),
        ),
    ]
