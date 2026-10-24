"""Finding a file the user described rather than named.

"Find my resume" and "open the latest PDF" are searches, and a search that
returns one answer when several were plausible is the dangerous kind.  So every
function here returns *candidates*, and the caller asks when there is more than
one worth asking about.

Two limits keep it honest and quick.  The walk stays inside the approved
workspace roots and skips the same dependency directories the repository index
skips, so a search never wanders into somewhere the user did not approve.  And
it is bounded by a file budget, so a large Documents folder degrades into a
partial answer rather than a hang -- with ``truncated`` set, because a partial
answer presented as complete is a lie about what was searched.

Nothing here opens, moves or deletes anything.  Discovery answers "which file";
the action layer and the safety policy decide what happens to it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from utils.logger import get_logger
from workspace.repositories import SKIP_DIRECTORIES

logger = get_logger("workspace.files")

#: Spoken category -> the extensions it means.  Categories exist because
#: "the latest PDF" is a common request and "*.pdf" is not something a person
#: says out loud.
CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "pdf": (".pdf",),
    "document": (".doc", ".docx", ".odt", ".rtf", ".txt", ".md", ".pdf"),
    "spreadsheet": (".xls", ".xlsx", ".csv", ".ods"),
    "presentation": (".ppt", ".pptx", ".odp"),
    "image": (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".heic"),
    "video": (".mp4", ".mov", ".avi", ".mkv", ".webm"),
    "audio": (".mp3", ".wav", ".flac", ".m4a", ".ogg"),
    "code": (".py", ".js", ".ts", ".java", ".c", ".cpp", ".cs", ".go", ".rs"),
    "archive": (".zip", ".7z", ".rar", ".tar", ".gz"),
}

#: How many files to visit before giving up and saying so.
MAX_FILES = 40000

#: How deep to walk. Deeper than the repository index because a file can be
#: several folders inside a project.
MAX_DEPTH = 6

MAX_RESULTS = 8


@dataclass(frozen=True)
class FoundFile:
    """One file, and why it was offered."""

    path: Path
    modified_at: float
    size: int = 0
    score: float = 0.0

    @property
    def name(self) -> str:
        return self.path.name

    def as_dict(self) -> Dict[str, object]:
        return {"path": str(self.path), "name": self.name,
                "modified_at": self.modified_at, "size": self.size}


def _score(needle: str, filename: str) -> float:
    """How well a filename answers what was asked for.

    Layered, like every other matcher here: an exact stem is different evidence
    from a substring, and flattening them is how "find my resume" returns
    ``resume-template-do-not-use.docx`` ahead of ``Resume.pdf``.
    """
    stem = Path(filename).stem.lower()
    wanted = needle.lower().strip()
    if not wanted:
        return 0.0
    if stem == wanted:
        return 1.0
    normalised = stem.replace("-", " ").replace("_", " ")
    if normalised == wanted:
        return 0.95
    if stem.startswith(wanted) or normalised.startswith(wanted):
        return 0.8
    if wanted in normalised:
        return 0.6
    parts = [p for p in wanted.replace("-", " ").replace("_", " ").split() if p]
    if parts and all(part in normalised for part in parts):
        return 0.5
    return 0.0


class FileIndex:
    """Searches the approved workspace for a file the user described."""

    def __init__(self, policy, *, max_depth: int = MAX_DEPTH) -> None:
        self._policy = policy
        self._max_depth = max_depth
        self.truncated = False

    def _walk(self) -> Iterable[Tuple[Path, os.stat_result]]:
        visited = 0
        self.truncated = False
        for root in self._policy.roots():
            for directory, subdirectories, filenames in os.walk(root):
                path = Path(directory)
                try:
                    depth = len(path.relative_to(root).parts)
                except ValueError:
                    continue
                if depth >= self._max_depth:
                    subdirectories[:] = []
                    continue
                subdirectories[:] = [d for d in subdirectories
                                     if d not in SKIP_DIRECTORIES
                                     and not d.startswith(".")]
                for filename in filenames:
                    visited += 1
                    if visited > MAX_FILES:
                        self.truncated = True
                        logger.warning("File search truncated", extra={
                            "context": {"limit": MAX_FILES}})
                        return
                    candidate = path / filename
                    try:
                        yield candidate, candidate.stat()
                    except OSError:
                        continue

    def find_named(self, needle: str, *,
                   limit: int = MAX_RESULTS) -> List[FoundFile]:
        """Files whose name matches what was described, best first."""
        wanted = " ".join(str(needle or "").split())
        if not wanted:
            return []
        started = time.perf_counter()
        scored: List[FoundFile] = []
        for path, info in self._walk():
            score = _score(wanted, path.name)
            if score > 0:
                scored.append(FoundFile(path=path, modified_at=info.st_mtime,
                                        size=info.st_size, score=score))
        scored.sort(key=lambda f: (-f.score, -f.modified_at))
        best = scored[0].score if scored else 0.0
        close = [f for f in scored if f.score >= best - 0.2]
        logger.info("File search complete", extra={"context": {
            "needle": wanted, "matches": len(close),
            "ms": round((time.perf_counter() - started) * 1000, 1)}})
        return close[:limit]

    def latest_of(self, category: str, *,
                  limit: int = MAX_RESULTS) -> List[FoundFile]:
        """The most recently modified files of a kind.

        Returns a list rather than one file even though "the latest" sounds
        singular: two files modified seconds apart are not meaningfully
        ordered, and the caller should be able to see that.
        """
        extensions = CATEGORIES.get(str(category or "").lower().strip())
        if not extensions:
            return []
        found: List[FoundFile] = []
        for path, info in self._walk():
            if path.suffix.lower() in extensions:
                found.append(FoundFile(path=path, modified_at=info.st_mtime,
                                       size=info.st_size, score=1.0))
        found.sort(key=lambda f: -f.modified_at)
        return found[:limit]


def category_for(word: str) -> str:
    """The file category a spoken word names, or ""."""
    lowered = " ".join(str(word or "").lower().split()).rstrip("s")
    if lowered in CATEGORIES:
        return lowered
    for name in CATEGORIES:
        if lowered == name or lowered == name.rstrip("s"):
            return name
    return ""


__all__ = ["CATEGORIES", "FileIndex", "FoundFile", "MAX_DEPTH", "MAX_FILES",
           "MAX_RESULTS", "category_for"]
