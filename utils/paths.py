"""Filesystem locations used by AARMA.

AARMA keeps *code* and *user data* strictly separate so that the future
Windows installer can drop the application into ``Program Files`` while the
per-user configuration, database and logs live in a writable profile
directory.

Resolution order for the data root:

1. ``AARMA_HOME`` environment variable (absolute or relative path).
2. ``%APPDATA%\\AARMA`` on Windows.
3. ``~/.local/share/aarma`` elsewhere (development on non-Windows hosts).

Set ``AARMA_HOME=./data`` to run a fully portable checkout, which is what the
bundled ``.env.example`` demonstrates.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict

#: Directory containing ``main.py`` -- the root of the source checkout.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

_ENV_HOME = "AARMA_HOME"


def _default_data_root() -> Path:
    """Return the platform-appropriate per-user data directory."""
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "AARMA"
        return Path.home() / "AppData" / "Roaming" / "AARMA"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "aarma"
    return Path.home() / ".local" / "share" / "aarma"


def resolve_data_root() -> Path:
    """Resolve the data root, honouring the ``AARMA_HOME`` override."""
    override = os.environ.get(_ENV_HOME, "").strip()
    if override:
        return (PROJECT_ROOT / override).resolve() if not Path(override).is_absolute() else Path(override).resolve()
    return _default_data_root()


@dataclass(frozen=True)
class AppPaths:
    """Concrete, absolute locations for every file AARMA writes."""

    root: Path
    config_file: Path
    database_file: Path
    log_dir: Path
    log_file: Path
    screenshot_dir: Path
    #: Where "Save diagnostic audio" writes, when the user switches it on.
    #: Deliberately absent from :meth:`ensure` -- a folder for microphone
    #: recordings should not exist on a machine that never asked for one.
    diagnostic_dir: Path

    @classmethod
    def build(cls, root: Path) -> "AppPaths":
        log_dir = root / "logs"
        return cls(
            root=root,
            config_file=root / "config.json",
            database_file=root / "aarma.db",
            log_dir=log_dir,
            log_file=log_dir / "aarma.log",
            screenshot_dir=root / "screenshots",
            diagnostic_dir=root / "diagnostics",
        )

    def ensure(self) -> "AppPaths":
        """Create every directory AARMA needs.  Safe to call repeatedly."""
        for directory in (self.root, self.log_dir, self.screenshot_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def get_paths() -> AppPaths:
    """Return the process-wide :class:`AppPaths`, creating directories once."""
    return AppPaths.build(resolve_data_root()).ensure()


def reset_paths_cache() -> None:
    """Forget the cached paths.  Used by the test-suite after changing ``AARMA_HOME``."""
    get_paths.cache_clear()
    user_content_folders.cache_clear()


# ------------------------------------------------------- user content folders

#: Canonical folder name -> Windows ``FOLDERID`` GUID string.  These are the
#: folders AARMA is allowed to work in, and the ones users name out loud.
_KNOWN_FOLDER_IDS: Dict[str, str] = {
    "Documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "Desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "Downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "Pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "Music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
    "Videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
}


def _windows_known_folder(guid: str) -> Path | None:
    """Ask Windows where a known folder actually is.

    Guessing ``~/Documents`` is wrong on any profile where the folder has been
    redirected -- which is the default once OneDrive Backup is enabled, and is
    therefore the common case rather than an edge case.  ``SHGetKnownFolderPath``
    returns the real location.
    """
    import ctypes
    from ctypes import wintypes

    class _GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    try:
        shell32 = ctypes.windll.shell32  # type: ignore[attr-defined]
        ole32 = ctypes.windll.ole32      # type: ignore[attr-defined]
        folder_id = _GUID()
        if ole32.CLSIDFromString(ctypes.c_wchar_p(guid), ctypes.byref(folder_id)) != 0:
            return None
        buffer = ctypes.c_wchar_p()
        if shell32.SHGetKnownFolderPath(ctypes.byref(folder_id), 0, None,
                                        ctypes.byref(buffer)) != 0:
            return None
        try:
            return Path(buffer.value) if buffer.value else None
        finally:
            ole32.CoTaskMemFree(buffer)
    except (OSError, AttributeError, ValueError):
        return None


@lru_cache(maxsize=1)
def user_content_folders() -> Dict[str, Path]:
    """The user's real content folders, keyed by canonical name.

    Only folders that exist are returned, so a stripped-down profile simply has
    fewer entries rather than paths that cannot be used.
    """
    home = Path.home()
    resolved: Dict[str, Path] = {}
    for name, guid in _KNOWN_FOLDER_IDS.items():
        candidate: Path | None = None
        if sys.platform == "win32":
            candidate = _windows_known_folder(guid)
        if candidate is None or not candidate.is_dir():
            fallback = home / name
            candidate = fallback if fallback.is_dir() else None
        if candidate is not None:
            resolved[name] = candidate
    return resolved
