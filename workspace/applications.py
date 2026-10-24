"""What is actually installed, found from the places Windows already lists it.

The Phase-2 catalogue was a hand-written list of eleven programs.  It is still
here and still authoritative for the things it names -- a curated entry knows
the process name, the shell URI and the fallback install paths, which discovery
cannot infer -- but it can no longer be the *only* way to name an application,
because the user has software the list has never heard of.

Discovery is deliberately narrow about where it looks:

* **Start Menu shortcuts**, for the current user and all users.  This is the
  list Windows itself shows when someone types into Start, which makes it the
  closest thing to "applications this person has".
* **Executables on PATH**, which is how command-line tools are reachable.

It does **not** crawl Program Files or the disk at large.  A scan like that is
slow, finds thousands of uninstallers and helper binaries, and would turn "open
my editor" into a list of things nobody would call an editor.  Section 24 of the
brief asks for exactly this restraint.

Discovery finds things to *launch*.  It does not decide whether launching is
allowed, and it never runs anything itself.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from utils.logger import get_logger

logger = get_logger("workspace.applications")

#: Shortcut names that are not applications a person opens.  Uninstallers and
#: documentation links dominate the Start Menu otherwise.
_NOISE_WORDS = ("uninstall", "readme", "release notes", "documentation",
                "help", "website", "web site", "manual", "licence", "license",
                "changelog", "report a", "feedback", "support", "repair",
                "modify", "remove")

#: Extensions worth offering. ``.lnk`` is the Start Menu's own format; ``.exe``
#: covers PATH entries.
_LAUNCHABLE = (".lnk", ".exe")

#: How long a discovery pass may be reused.  Applications are installed rarely,
#: so this is long; :meth:`ApplicationIndex.refresh` forces a rebuild.
CACHE_SECONDS = 900.0

MAX_SHORTCUTS = 1500


@dataclass(frozen=True)
class DiscoveredApplication:
    """One launchable application, and where the knowledge came from."""

    display_name: str
    target: Path
    source: str                  # "start-menu" | "path" | "catalogue"
    process_name: str = ""
    aliases: Tuple[str, ...] = ()
    catalogue_key: str = ""

    @property
    def key(self) -> str:
        return self.catalogue_key or self.display_name.lower()

    def as_dict(self) -> Dict[str, object]:
        return {"display_name": self.display_name, "target": str(self.target),
                "source": self.source, "process_name": self.process_name,
                "aliases": list(self.aliases),
                "catalogue_key": self.catalogue_key}


def _start_menu_directories() -> List[Path]:
    directories = []
    for variable, tail in (("APPDATA", r"Microsoft\Windows\Start Menu\Programs"),
                           ("ProgramData", r"Microsoft\Windows\Start Menu\Programs")):
        base = os.environ.get(variable)
        if base:
            candidate = Path(base) / tail
            if candidate.is_dir():
                directories.append(candidate)
    return directories


def _is_noise(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in _NOISE_WORDS)


class ApplicationIndex:
    """Installed applications, from Start Menu shortcuts and PATH.

    The curated catalogue is merged in first and wins on name collisions: it
    carries process names and shell URIs that a shortcut cannot tell us, and
    those are what make closing and launching reliable.
    """

    def __init__(self, *, start_menu=None, path_dirs=None,
                 cache_seconds: float = CACHE_SECONDS) -> None:
        # Injected for tests: a real Start Menu differs on every machine, and a
        # test that depended on one would assert something about the tester.
        self._start_menu = start_menu
        self._path_dirs = path_dirs
        self._cache_seconds = cache_seconds
        self._cache: List[DiscoveredApplication] = []
        self._built_at = 0.0

    # -- building ----------------------------------------------------------

    def _from_catalogue(self) -> List[DiscoveredApplication]:
        from actions.applications import APPLICATION_CATALOG, resolve_target

        found = []
        for entry in APPLICATION_CATALOG.values():
            target = resolve_target(entry)
            if target is None:
                continue
            found.append(DiscoveredApplication(
                display_name=entry.display_name, target=Path(target),
                source="catalogue", process_name=entry.process_name or "",
                aliases=tuple(entry.aliases), catalogue_key=entry.key))
        return found

    def _from_start_menu(self) -> List[DiscoveredApplication]:
        directories = (self._start_menu if self._start_menu is not None
                       else _start_menu_directories())
        found: List[DiscoveredApplication] = []
        seen = 0
        for directory in directories:
            for root, _subdirs, filenames in os.walk(directory):
                for filename in filenames:
                    if not filename.lower().endswith(_LAUNCHABLE):
                        continue
                    seen += 1
                    if seen > MAX_SHORTCUTS:
                        logger.warning("Start Menu scan truncated")
                        return found
                    name = Path(filename).stem
                    if _is_noise(name):
                        continue
                    found.append(DiscoveredApplication(
                        display_name=name, target=Path(root) / filename,
                        source="start-menu"))
        return found

    def _from_path(self) -> List[DiscoveredApplication]:
        directories = (self._path_dirs if self._path_dirs is not None
                       else [Path(p) for p in
                             os.environ.get("PATH", "").split(os.pathsep) if p])
        found: List[DiscoveredApplication] = []
        for directory in directories:
            try:
                if not Path(directory).is_dir():
                    continue
                for entry in os.scandir(directory):
                    if not entry.is_file():
                        continue
                    if not entry.name.lower().endswith(".exe"):
                        continue
                    name = Path(entry.name).stem
                    if _is_noise(name):
                        continue
                    found.append(DiscoveredApplication(
                        display_name=name, target=Path(entry.path),
                        source="path", process_name=entry.name))
            except OSError:
                continue
        return found

    def all(self, *, refresh: bool = False) -> List[DiscoveredApplication]:
        if not refresh and self._cache and \
                (time.monotonic() - self._built_at) < self._cache_seconds:
            return list(self._cache)

        started = time.perf_counter()
        merged: Dict[str, DiscoveredApplication] = {}
        # Catalogue first so it wins collisions.
        for application in self._from_catalogue():
            merged[application.display_name.lower()] = application
        for application in self._from_start_menu() + self._from_path():
            merged.setdefault(application.display_name.lower(), application)

        self._cache = sorted(merged.values(),
                             key=lambda a: (a.source != "catalogue",
                                            a.display_name.lower()))
        self._built_at = time.monotonic()
        logger.info("Application discovery complete", extra={"context": {
            "found": len(self._cache),
            "ms": round((time.perf_counter() - started) * 1000, 1)}})
        return list(self._cache)

    # -- lookup ------------------------------------------------------------

    def find(self, needle: str, *, limit: int = 5
             ) -> List[DiscoveredApplication]:
        """Applications matching a spoken name, best first.

        Layered like repository matching, and for the same reason: an exact
        name and a substring are different kinds of evidence, and merging them
        is how "open code" opens something called "Bandicode".
        """
        wanted = " ".join(str(needle or "").lower().split())
        if not wanted:
            return []

        exact, prefix, contains = [], [], []
        for application in self.all():
            name = application.display_name.lower()
            aliases = {a.lower() for a in application.aliases}
            if wanted == name or wanted in aliases:
                exact.append(application)
            elif name.startswith(wanted):
                prefix.append(application)
            elif wanted in name:
                contains.append(application)

        # Within a tier, a curated entry outranks a discovered one. "Code"
        # prefix-matches ``code-tunnel`` and only substring-matches "Visual
        # Studio Code", so tier order alone would offer the wrong one first --
        # and the curated entry is the one that knows the process name and the
        # install path.
        # Exactness first, then curation, then tier. Curation outranks tier
        # because "code" prefix-matches ``code-tunnel`` and only
        # substring-matches "Visual Studio Code" -- and the curated entry is
        # both what the user meant and the one that knows the process name.
        ranked = ([(0, a) for a in exact]
                  + [(1, a) for a in prefix]
                  + [(2, a) for a in contains])
        ranked.sort(key=lambda item: (
            item[0] if item[0] == 0 else 1,          # exact always first
            item[1].source != "catalogue",           # then curated
            item[0],                                 # then prefix over contains
            len(item[1].display_name)))
        return [application for _tier, application in ranked][:limit]

    def refresh(self) -> int:
        return len(self.all(refresh=True))


__all__ = ["ApplicationIndex", "CACHE_SECONDS", "DiscoveredApplication",
           "MAX_SHORTCUTS"]
