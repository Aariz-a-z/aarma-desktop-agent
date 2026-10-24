"""Finding the user's projects, by looking for what makes a folder a project.

"Open my AARMA repo" cannot be answered from a catalogue, because the catalogue
would have to be written by the person asking.  So it is answered by looking:
walk the approved workspace roots, and treat a directory as a repository when
it carries one of the markers that only projects carry.

Two costs govern the design.

**Scanning is expensive**, so it is bounded and cached.  A depth limit stops the
walk before ``node_modules`` swallows it, and the cache is invalidated by the
roots' modification times rather than by a timer -- a project list that is
rebuilt every command would make every command slow, and one that never rebuilds
would never see a new project.

**Guessing is worse than scanning**, so ambiguity is returned rather than
resolved.  Two folders matching "portfolio" is a question for the user;
:func:`RepositoryIndex.find` hands back both and lets the caller ask.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from utils.logger import get_logger

logger = get_logger("workspace.repositories")

#: Marker file (or directory) -> the project type it indicates.  A folder needs
#: only one; requiring several would miss every project that has not grown one
#: yet, which is most new ones.
MARKERS: Dict[str, str] = {
    ".git": "git",
    "package.json": "node",
    "pyproject.toml": "python",
    "requirements.txt": "python",
    "setup.py": "python",
    "Cargo.toml": "rust",
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "java",
    "CMakeLists.txt": "cmake",
    "go.mod": "go",
    "Gemfile": "ruby",
    "composer.json": "php",
    ".sln": "dotnet",
    "Makefile": "make",
}

#: Directories never worth descending into.  Without this the walk spends its
#: entire budget inside dependency trees and finds nothing the user owns.
SKIP_DIRECTORIES = frozenset({
    "node_modules", ".git", ".venv", "venv", "env", "__pycache__", ".idea",
    ".vscode", "dist", "build", "target", ".tox", ".mypy_cache", ".pytest_cache",
    "site-packages", "AppData", "Library", ".gradle", ".next", ".cache",
    "OneDriveTemp", "$RECYCLE.BIN", "System Volume Information",
})

#: How deep below a root to look.  Projects live a couple of levels down
#: ("Documents/Code/aarma"); anything deeper is usually inside a project rather
#: than being one.
MAX_DEPTH = 4

#: A ceiling on directories visited per scan, so a pathological tree cannot
#: make a command hang. Reaching it is logged, because a truncated index is a
#: different thing from an empty one and the user deserves to know which.
MAX_DIRECTORIES = 6000

#: How long an index may be reused before its roots are re-checked.
CACHE_SECONDS = 120.0


@dataclass(frozen=True)
class Repository:
    """One project directory, as found on disk."""

    name: str
    path: Path
    kind: str = ""
    markers: Tuple[str, ...] = ()
    modified_at: float = 0.0

    @property
    def is_git(self) -> bool:
        return "git" in self.markers or (self.path / ".git").exists()

    def remote_url(self) -> str:
        """The git remote, read from the config file rather than by running git.

        Reading the file avoids launching a subprocess on a path the user has
        not confirmed, which matters because repository discovery runs over
        directories AARMA found rather than ones it was given.
        """
        config = self.path / ".git" / "config"
        try:
            text = config.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            return ""
        inside = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                inside = stripped.replace(" ", "").startswith('[remote"origin"]')
                continue
            if inside and stripped.startswith("url"):
                _, _, value = stripped.partition("=")
                return value.strip()
        return ""

    def as_dict(self) -> Dict[str, object]:
        return {"name": self.name, "path": str(self.path), "kind": self.kind,
                "markers": list(self.markers), "modified_at": self.modified_at}


def _score_name(needle: str, repository: Repository) -> float:
    """How well a spoken name matches a repository.  Higher is better.

    Layered rather than fuzzy: an exact folder name is different evidence from
    a substring, and collapsing them into one number is how "open my portfolio"
    lands on ``portfolio-old-backup``.
    """
    name = repository.name.lower()
    wanted = needle.lower().strip()
    if not wanted:
        return 0.0
    if name == wanted:
        return 1.0
    if name.replace("-", " ").replace("_", " ") == wanted:
        return 0.95
    if name.startswith(wanted) or wanted.startswith(name):
        return 0.8
    if wanted in name:
        return 0.6
    # All the words asked for appear somewhere in the name.
    parts = [p for p in wanted.replace("-", " ").replace("_", " ").split() if p]
    if parts and all(part in name for part in parts):
        return 0.55
    return 0.0


class RepositoryIndex:
    """A cached list of the projects inside the workspace."""

    def __init__(self, policy, *, max_depth: int = MAX_DEPTH,
                 cache_seconds: float = CACHE_SECONDS) -> None:
        self._policy = policy
        self._max_depth = max_depth
        self._cache_seconds = cache_seconds
        self._cache: List[Repository] = []
        self._built_at: float = 0.0
        self._signature: Tuple = ()
        self._truncated = False

    # -- cache -------------------------------------------------------------

    def _roots_signature(self) -> Tuple:
        """Cheap fingerprint of the roots, to notice a new project quickly."""
        signature = []
        for root in self._policy.roots():
            try:
                signature.append((str(root), root.stat().st_mtime))
            except OSError:
                signature.append((str(root), 0.0))
        return tuple(signature)

    @property
    def truncated(self) -> bool:
        """Whether the last scan hit :data:`MAX_DIRECTORIES`."""
        return self._truncated

    def invalidate(self) -> None:
        self._built_at = 0.0

    def all(self, *, refresh: bool = False) -> List[Repository]:
        signature = self._roots_signature()
        fresh = (time.monotonic() - self._built_at) < self._cache_seconds
        if not refresh and fresh and signature == self._signature:
            return list(self._cache)
        self._cache = self._scan()
        self._built_at = time.monotonic()
        self._signature = signature
        return list(self._cache)

    # -- scanning ----------------------------------------------------------

    def _scan(self) -> List[Repository]:
        started = time.perf_counter()
        found: Dict[str, Repository] = {}
        visited = 0
        self._truncated = False

        for root in self._policy.roots():
            for directory, subdirectories, filenames in os.walk(root):
                visited += 1
                if visited > MAX_DIRECTORIES:
                    self._truncated = True
                    logger.warning("Repository scan truncated", extra={
                        "context": {"limit": MAX_DIRECTORIES}})
                    break

                path = Path(directory)
                depth = len(path.relative_to(root).parts)
                if depth >= self._max_depth:
                    subdirectories[:] = []
                    continue
                subdirectories[:] = [d for d in subdirectories
                                     if d not in SKIP_DIRECTORIES
                                     and not d.startswith(".")
                                     or d == ".git"]

                if path == root:
                    # A workspace root is a place projects live, not a project.
                    # Downloads happens to contain a requirements.txt on this
                    # machine, and calling the whole folder a Python project
                    # would put it in every "which project?" list for ever.
                    continue

                entries = set(filenames) | set(subdirectories)
                markers = tuple(sorted(
                    marker for marker in MARKERS
                    if marker in entries
                    or (marker == ".sln" and any(f.endswith(".sln")
                                                 for f in filenames))))
                if not markers:
                    continue

                kinds = {MARKERS[m] for m in markers}
                # A project directory is not searched for nested projects: the
                # thing inside a repository is part of it, not another one.
                subdirectories[:] = []
                try:
                    modified = path.stat().st_mtime
                except OSError:
                    modified = 0.0
                found[str(path).lower()] = Repository(
                    name=path.name, path=path,
                    kind="git" if "git" in kinds and len(kinds) == 1
                         else next((k for k in sorted(kinds) if k != "git"), "git"),
                    markers=markers, modified_at=modified)
            if self._truncated:
                break

        repositories = sorted(found.values(), key=lambda r: -r.modified_at)
        logger.info("Repository scan complete", extra={"context": {
            "found": len(repositories), "directories": visited,
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "truncated": self._truncated}})
        return repositories

    # -- lookup ------------------------------------------------------------

    def find(self, needle: str, *, limit: int = 5) -> List[Repository]:
        """Repositories matching a spoken name, best first.

        Returns *every* plausible match rather than the best one.  Choosing
        between two projects called "portfolio" is the user's decision, and a
        function that returns one of them gives the caller no way to know a
        choice existed.
        """
        scored = [(score, repository)
                  for repository in self.all()
                  if (score := _score_name(needle, repository)) > 0]
        scored.sort(key=lambda item: (-item[0], -item[1].modified_at))
        best = scored[0][0] if scored else 0.0
        # Keep only matches of comparable quality: an exact hit should not be
        # presented alongside a weak substring as though they were rivals.
        close = [repository for score, repository in scored
                 if score >= best - 0.2]
        return close[:limit]

    def most_recent(self) -> Optional[Repository]:
        """The project touched most recently, for "the one I was working on"."""
        repositories = self.all()
        return repositories[0] if repositories else None


__all__ = ["CACHE_SECONDS", "MARKERS", "MAX_DEPTH", "MAX_DIRECTORIES",
           "Repository", "RepositoryIndex", "SKIP_DIRECTORIES"]
