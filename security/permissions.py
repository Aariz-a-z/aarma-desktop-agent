"""User-controlled capability permissions (specification section 12).

AARMA never asks the operating system for broad rights and never grants itself
a capability implicitly.  Each registered action declares the permissions it
needs; :class:`PermissionManager` is the only component allowed to answer
whether those permissions are currently held.

Defaults follow least privilege in the direction that matters most: anything
that can *observe* the user (microphone, camera, screen, keyboard, mouse) or
*control the machine* is denied until the user turns it on.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Sequence

from config.settings import ConfigManager
from utils.logger import get_logger

logger = get_logger("security.permissions")


class Permission(str, Enum):
    """Every capability the user can switch on or off."""

    MICROPHONE = "microphone"
    SPEAKER = "speaker"
    BROWSER = "browser"
    APPLICATIONS = "applications"
    FILESYSTEM = "filesystem"
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    SCREENSHOT = "screenshot"
    CALENDAR = "calendar"
    REMINDERS = "reminders"
    EMAIL = "email"
    SYSTEM_CONTROL = "system_control"
    INTERNET = "internet"
    MEMORY = "memory"
    CAMERA = "camera"

    def __str__(self) -> str:  # pragma: no cover - display helper
        return self.value


@dataclass(frozen=True)
class PermissionInfo:
    """Presentation metadata for a permission, used by the settings UI."""

    permission: Permission
    display_name: str
    description: str
    group: str
    default_granted: bool


def _perm(permission: Permission, display: str, description: str, group: str,
          default: bool) -> PermissionInfo:
    return PermissionInfo(permission=permission, display_name=display,
                          description=description, group=group, default_granted=default)


#: Ordered so the settings screen can render it top to bottom without sorting.
PERMISSION_CATALOG: Dict[Permission, PermissionInfo] = {
    info.permission: info
    for info in (
        _perm(Permission.SPEAKER, "Speaker",
              "Let AARMA speak responses out loud.", "Voice", True),
        _perm(Permission.MICROPHONE, "Microphone",
              "Let AARMA listen for the wake word and your voice commands.", "Voice", False),
        _perm(Permission.BROWSER, "Browser",
              "Let AARMA open websites in your default browser.", "Web", True),
        _perm(Permission.INTERNET, "Internet",
              "Let AARMA use online services such as web search.", "Web", True),
        _perm(Permission.APPLICATIONS, "Launch applications",
              "Let AARMA start applications from its approved list.", "Desktop", True),
        _perm(Permission.FILESYSTEM, "Files and folders",
              "Let AARMA list, open and create items inside your personal folders.", "Desktop", True),
        _perm(Permission.SCREENSHOT, "Screenshots",
              "Let AARMA capture an image of your screen when you ask.", "Desktop", False),
        _perm(Permission.KEYBOARD, "Keyboard control",
              "Let AARMA type on your behalf.", "Desktop", False),
        _perm(Permission.MOUSE, "Mouse control",
              "Let AARMA move and click the mouse on your behalf.", "Desktop", False),
        _perm(Permission.SYSTEM_CONTROL, "System control",
              "Let AARMA close applications and change system state.", "Desktop", False),
        _perm(Permission.REMINDERS, "Reminders",
              "Let AARMA store and read your local reminders.", "Productivity", True),
        _perm(Permission.CALENDAR, "Calendar",
              "Let AARMA read and create calendar events.", "Productivity", False),
        _perm(Permission.EMAIL, "Email",
              "Let AARMA read and send email on your behalf.", "Productivity", False),
        _perm(Permission.MEMORY, "Memory",
              "Let AARMA remember facts and preferences you tell it.", "Personalisation", True),
        _perm(Permission.CAMERA, "Camera",
              "Reserved for a future release. Not used by this build.", "Personalisation", False),
    )
}

DEFAULT_PERMISSIONS: Dict[str, bool] = {
    info.permission.value: info.default_granted for info in PERMISSION_CATALOG.values()
}


@dataclass(frozen=True)
class PermissionDecision:
    """Result of checking a set of required permissions."""

    allowed: bool
    missing: tuple[Permission, ...] = ()

    @property
    def first_missing(self) -> Permission | None:
        return self.missing[0] if self.missing else None

    def describe(self) -> str:
        if self.allowed:
            return "allowed"
        return "missing: " + ", ".join(p.value for p in self.missing)


class PermissionManager:
    """Reads and writes permission state through the shared configuration.

    Permissions are intentionally *not* cached: the GUI can toggle one at any
    moment and the very next action must see the new value.
    """

    def __init__(self, config: ConfigManager) -> None:
        self._config = config

    # ------------------------------------------------------------------ read

    def is_granted(self, permission: Permission) -> bool:
        granted = self._config.config.permissions.granted
        if permission.value in granted:
            return bool(granted[permission.value])
        return PERMISSION_CATALOG[permission].default_granted

    def check(self, required: Sequence[Permission]) -> PermissionDecision:
        """Check every required permission at once and report *all* that are missing."""
        missing = tuple(p for p in required if not self.is_granted(p))
        decision = PermissionDecision(allowed=not missing, missing=missing)
        if missing:
            logger.info("Permission check failed",
                        extra={"context": {"missing": [p.value for p in missing]}})
        return decision

    def snapshot(self) -> Dict[Permission, bool]:
        """Current state of every permission, including unset ones."""
        return {permission: self.is_granted(permission) for permission in PERMISSION_CATALOG}

    def grouped(self) -> Dict[str, List[PermissionInfo]]:
        """Catalogue entries bucketed by UI group, preserving catalogue order."""
        groups: Dict[str, List[PermissionInfo]] = {}
        for info in PERMISSION_CATALOG.values():
            groups.setdefault(info.group, []).append(info)
        return groups

    # ----------------------------------------------------------------- write

    def set(self, permission: Permission, granted: bool) -> None:
        self._config.set_permission(permission.value, granted)
        logger.info("Permission changed",
                    extra={"context": {"permission": permission.value, "granted": bool(granted)}})

    def grant(self, permission: Permission) -> None:
        self.set(permission, True)

    def revoke(self, permission: Permission) -> None:
        self.set(permission, False)

    def apply_defaults(self, *, overwrite: bool = False) -> None:
        """Materialise the default permission set into the config file.

        Called at the end of first-run setup so the stored config is explicit
        about every capability rather than relying on implicit fallbacks.
        """
        current = dict(self._config.config.permissions.granted)
        for name, default in DEFAULT_PERMISSIONS.items():
            if overwrite or name not in current:
                current[name] = default
        from config.settings import PermissionSettings  # local import: avoids a cycle at module load

        self._config.update(permissions=PermissionSettings(granted=current))

    def apply_selection(self, selection: Mapping[Permission, bool]) -> None:
        """Write a complete permission selection, e.g. from the setup wizard."""
        current = dict(self._config.config.permissions.granted)
        for permission, granted in selection.items():
            current[permission.value] = bool(granted)
        from config.settings import PermissionSettings

        self._config.update(permissions=PermissionSettings(granted=current))


def describe(permissions: Iterable[Permission]) -> str:
    """Human-readable list of permission display names."""
    names = [PERMISSION_CATALOG[p].display_name for p in permissions]
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]
