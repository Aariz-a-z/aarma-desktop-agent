"""Where AARMA may look, and what it may do there.

Phase 5 widens AARMA from a fixed catalogue to "find the real thing on this
computer".  That is only safe if *where it may look* is a decision the user
made, held in one place, and independent of how a sentence was phrased.  So
this policy is deliberately not reachable from the language layer: the resolver
can propose any target it likes, and this decides whether that target is
allowed.

Three separate questions, kept separate because conflating them is how a read
becomes a write:

**Is this path inside the workspace?**  :meth:`WorkspaceAccessPolicy.contains`.
Roots come from the user's configuration plus, optionally, the content folders
Windows itself reports.  Nothing is assumed about the shape of a machine --
``C:\\Projects`` is not a default, because it is a guess about someone else's
computer.

**Is this path forbidden regardless?**  :data:`DENIED_PREFIXES`.  The Windows
directory and Program Files stay out even if a user adds them as a root, which
they should not be able to do by accident.  A root that swallows the operating
system is not a workspace.

**Is this *operation* allowed here?**  :meth:`WorkspaceAccessPolicy.decide`.
Reading inside the workspace is free; writing is allowed where the user asked;
deleting and executing are never decided here at all -- they are handed to the
existing confirmation machinery, because this module must not become a second
place where destructive operations are approved.

The last point is the important one.  This policy can only ever *narrow* what
happens.  It has no way to grant anything, and nothing downstream treats its
approval as permission.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from utils.logger import get_logger

logger = get_logger("workspace.policy")


class Operation(str, Enum):
    """What is being attempted, for :meth:`WorkspaceAccessPolicy.decide`."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    LAUNCH = "launch"          # start an application, no path involved


class Verdict(str, Enum):
    ALLOWED = "allowed"
    #: Permitted, but the existing confirmation machinery must run first.
    NEEDS_CONFIRMATION = "needs_confirmation"
    REFUSED = "refused"


#: Directory prefixes that are never part of a workspace, whatever the
#: configuration says.  Compared case-insensitively against the resolved path.
#:
#: These are not "dangerous to read" -- they are dangerous to treat as a place
#: the assistant may work, because a fuzzy filename match inside Windows\\System32
#: is a very different thing from one inside Documents.
DENIED_PREFIXES: Tuple[str, ...] = (
    r"C:\Windows",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\ProgramData",
    r"C:\$Recycle.Bin",
    r"C:\System Volume Information",
)

#: Path shapes that are refused before any root is consulted.  Device paths and
#: UNC shares reach outside the machine's ordinary filesystem, and a workspace
#: is by definition local and named.
_REFUSED_SHAPES = ("\\\\.\\", "\\\\?\\")


@dataclass(frozen=True)
class AccessDecision:
    """Why an operation on a path was allowed, questioned or refused."""

    verdict: Verdict
    operation: Operation
    path: Optional[Path] = None
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOWED

    @property
    def refused(self) -> bool:
        return self.verdict is Verdict.REFUSED

    def as_dict(self) -> Dict[str, object]:
        return {"verdict": self.verdict.value, "operation": self.operation.value,
                "path": str(self.path) if self.path else "", "reason": self.reason}


def _normalise(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return Path(os.path.abspath(os.path.expanduser(str(path))))


def _is_denied(path: Path) -> str:
    """The prefix that forbids ``path``, or "" when none does."""
    text = str(path)
    lowered = text.lower()
    for shape in _REFUSED_SHAPES:
        if shape.lower() in lowered:
            return shape
    for prefix in DENIED_PREFIXES:
        normalised = prefix.lower().rstrip("\\")
        if lowered == normalised or lowered.startswith(normalised + "\\"):
            return prefix
    return ""


class WorkspaceAccessPolicy:
    """The one place that decides where AARMA may work.

    Reads its roots from configuration on every access rather than caching
    them: the settings screen can add a root at any moment and the very next
    command must see it, which is the same reason
    :class:`~security.permissions.PermissionManager` does not cache either.
    """

    def __init__(self, config, *, content_folders=None) -> None:
        self._config = config
        # Injected for tests, which must not depend on the machine's own
        # Documents folder existing or being where Windows usually puts it.
        self._content_folders = content_folders

    # -- roots -------------------------------------------------------------

    def _discovered_folders(self) -> Dict[str, Path]:
        if self._content_folders is not None:
            return dict(self._content_folders)
        from utils.paths import user_content_folders

        return user_content_folders()

    def roots(self) -> List[Path]:
        """Every directory AARMA may work inside, deduplicated and existing.

        A configured root that has been deleted or renamed simply drops out;
        reporting it as available and failing later would be worse than a
        shorter list.
        """
        settings = getattr(self._config.config, "workspace", None)
        found: List[Path] = []

        for raw in list(getattr(settings, "roots", []) or []):
            candidate = _normalise(Path(str(raw)))
            if _is_denied(candidate):
                logger.warning("Configured root is not permitted", extra={
                    "context": {"root": str(candidate)}})
                continue
            if candidate.is_dir():
                found.append(candidate)

        if getattr(settings, "include_content_folders", True):
            for path in self._discovered_folders().values():
                candidate = _normalise(Path(path))
                if candidate.is_dir() and not _is_denied(candidate):
                    found.append(candidate)

        unique: List[Path] = []
        seen = set()
        for path in found:
            key = str(path).lower()
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    def named_roots(self) -> Dict[str, Path]:
        """Roots keyed by a name a person would use."""
        named = {name: _normalise(Path(path))
                 for name, path in self._discovered_folders().items()
                 if _normalise(Path(path)).is_dir()}
        settings = getattr(self._config.config, "workspace", None)
        for raw in list(getattr(settings, "roots", []) or []):
            candidate = _normalise(Path(str(raw)))
            if candidate.is_dir() and not _is_denied(candidate):
                named.setdefault(candidate.name, candidate)
        return named

    # -- containment -------------------------------------------------------

    def contains(self, path: Path) -> bool:
        target = _normalise(Path(path))
        if _is_denied(target):
            return False
        for root in self.roots():
            try:
                target.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def resolve(self, path_like: str) -> Optional[Path]:
        """A real path inside the workspace, or ``None``.

        ``None`` rather than an exception because callers routinely ask about
        paths that turn out to be elsewhere, and that is an ordinary answer.
        """
        raw = str(path_like or "").strip().strip('"')
        if not raw:
            return None
        candidate = Path(raw)
        if not candidate.is_absolute():
            roots = self.roots()
            if not roots:
                return None
            candidate = roots[0] / raw
        resolved = _normalise(candidate)
        return resolved if self.contains(resolved) else None

    # -- the decision ------------------------------------------------------

    def decide(self, operation: Operation,
               path: Optional[Path] = None) -> AccessDecision:
        """Whether ``operation`` may proceed on ``path``.

        Deleting and executing return ``NEEDS_CONFIRMATION`` rather than
        ``ALLOWED``: this module deliberately cannot approve them.  The
        confirmation itself belongs to
        :class:`~security.confirmation.ConfirmationBroker`, and having two
        places that can approve a delete is how one of them gets forgotten.
        """
        if operation is Operation.LAUNCH:
            return AccessDecision(Verdict.ALLOWED, operation,
                                  reason="launching needs no workspace path")

        if path is None:
            return AccessDecision(Verdict.REFUSED, operation,
                                  reason="no path was given")

        target = _normalise(Path(path))
        denied = _is_denied(target)
        if denied:
            return AccessDecision(
                Verdict.REFUSED, operation, target,
                reason=f"{denied} is a system location AARMA does not work in")

        if not self.contains(target):
            return AccessDecision(
                Verdict.REFUSED, operation, target,
                reason="that path is outside the folders you've approved")

        if operation in (Operation.DELETE, Operation.EXECUTE):
            return AccessDecision(
                Verdict.NEEDS_CONFIRMATION, operation, target,
                reason=f"{operation.value} always asks first")

        return AccessDecision(Verdict.ALLOWED, operation, target,
                              reason="inside the workspace")

    # -- mutation ----------------------------------------------------------

    def add_root(self, path_like: str) -> Tuple[bool, str]:
        """Approve a new directory.  Returns ``(added, message)``."""
        candidate = _normalise(Path(str(path_like or "").strip().strip('"')))
        if not str(candidate).strip():
            return False, "I need a folder to add."
        denied = _is_denied(candidate)
        if denied:
            return False, (f"I won't work inside {denied} — that's a Windows "
                           "system location, not a workspace.")
        if not candidate.is_dir():
            return False, f"I couldn't find a folder at {candidate}."

        settings = self._config.config.workspace
        existing = [str(r) for r in settings.roots]
        if any(str(candidate).lower() == str(_normalise(Path(r))).lower()
               for r in existing):
            return False, f"{candidate} is already in your workspace."
        self._config.update_workspace(roots=existing + [str(candidate)])
        logger.info("Workspace root added", extra={"context": {
            "root": str(candidate)}})
        return True, f"Added {candidate} to your workspace."

    def remove_root(self, path_like: str) -> Tuple[bool, str]:
        raw = str(path_like or "").strip().strip('"')
        candidate = _normalise(Path(raw))
        settings = self._config.config.workspace
        remaining = [r for r in settings.roots
                     if str(_normalise(Path(r))).lower() != str(candidate).lower()
                     and r.lower() != raw.lower()]
        if len(remaining) == len(settings.roots):
            return False, f"{raw} isn't one of your workspace folders."
        self._config.update_workspace(roots=remaining)
        logger.info("Workspace root removed", extra={"context": {"root": raw}})
        return True, f"Removed {candidate} from your workspace."

    def describe(self) -> str:
        roots = self.roots()
        if not roots:
            return "No workspace folders are available."
        return ", ".join(str(root) for root in roots)


__all__ = ["AccessDecision", "DENIED_PREFIXES", "Operation", "Verdict",
           "WorkspaceAccessPolicy"]
