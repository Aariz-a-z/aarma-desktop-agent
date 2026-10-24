"""Risk classification and input safety.

Two responsibilities live here:

* :class:`RiskLevel` / :class:`RiskPolicy` -- how dangerous an action is and
  what extra proof the user must supply before it runs (section 11).
* :class:`SafePathResolver`, :func:`is_safe_url`, :func:`sanitize_component` --
  turning loosely-worded user input into arguments that cannot escape the
  boundaries the user agreed to.

The filesystem sandbox is the important one.  A language model (Phase 2) will
eventually be proposing these arguments, so "Documents" must resolve to the
real Documents folder and ``../../Windows/System32`` must not resolve at all.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence
from urllib.parse import urlparse, urlunparse

from config.settings import SecuritySettings
from utils.logger import get_logger
from utils.paths import user_content_folders

logger = get_logger("security.validator")


class UnsafeInputError(ValueError):
    """Raised when user (or model) supplied input fails a safety check."""


class RiskLevel(IntEnum):
    """How much damage an action could do if it were wrong.

    Ordered so policies can be written as ``risk >= RiskLevel.HIGH``.
    """

    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @property
    def label(self) -> str:
        return self.name.capitalize()


#: Short explanations shown next to actions in the GUI and in ``help``.
RISK_DESCRIPTIONS: Dict[RiskLevel, str] = {
    RiskLevel.LOW: "Read-only or trivially reversible.",
    RiskLevel.MEDIUM: "Changes something on your machine, but not destructively.",
    RiskLevel.HIGH: "Destructive, irreversible or security-sensitive. Always confirmed.",
}


@dataclass(frozen=True)
class RiskPolicy:
    """Maps a risk level onto the proof required before execution.

    Reading the policy from :class:`~config.settings.SecuritySettings` means the
    user can tighten it (require the passphrase for high-risk actions) but the
    floor -- always confirm ``HIGH`` -- is not user-adjustable.
    """

    confirm_from: RiskLevel = RiskLevel.HIGH
    passphrase_from: Optional[RiskLevel] = None

    @classmethod
    def from_settings(cls, settings: SecuritySettings) -> "RiskPolicy":
        passphrase_from: Optional[RiskLevel] = None
        if settings.enabled and settings.require_passphrase_for_high_risk:
            passphrase_from = RiskLevel.HIGH
        return cls(confirm_from=RiskLevel.HIGH, passphrase_from=passphrase_from)

    def requires_confirmation(self, risk: RiskLevel) -> bool:
        """High-risk actions always require a yes/no, regardless of settings."""
        return risk >= self.confirm_from

    def requires_passphrase(self, risk: RiskLevel) -> bool:
        return self.passphrase_from is not None and risk >= self.passphrase_from


# --------------------------------------------------------------------- paths

#: Folder words users actually say, mapped onto canonical folder names.  The
#: canonical name is then resolved to a real directory by
#: :func:`utils.paths.user_content_folders`, which asks Windows rather than
#: assuming ``~/Documents`` -- that assumption breaks on any redirected profile.
FOLDER_ALIASES: Dict[str, str] = {
    "documents": "Documents",
    "document": "Documents",
    "docs": "Documents",
    "my documents": "Documents",
    "desktop": "Desktop",
    "downloads": "Downloads",
    "download": "Downloads",
    "pictures": "Pictures",
    "photos": "Pictures",
    "music": "Music",
    "videos": "Videos",
    "video": "Videos",
}

#: Characters Windows forbids in a file name, plus path separators.
_ILLEGAL_COMPONENT_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

#: Reserved device names on Windows -- creating these causes very odd failures.
_RESERVED_NAMES = {
    "con", "prn", "aux", "nul",
    *{f"com{i}" for i in range(1, 10)},
    *{f"lpt{i}" for i in range(1, 10)},
}

MAX_COMPONENT_LENGTH = 120


def sanitize_component(name: str) -> str:
    """Validate a single file or folder name.

    Rejects (rather than silently rewrites) anything containing a separator or
    a traversal sequence, because quietly turning ``../secrets`` into
    ``secrets`` would hide an attempted escape instead of surfacing it.
    """
    candidate = (name or "").strip().strip(".")
    if not candidate:
        raise UnsafeInputError("The name is empty.")
    if len(candidate) > MAX_COMPONENT_LENGTH:
        raise UnsafeInputError(f"The name is longer than {MAX_COMPONENT_LENGTH} characters.")
    if candidate in (".", "..") or "/" in name or "\\" in name:
        raise UnsafeInputError("The name may not contain a path separator.")
    if _ILLEGAL_COMPONENT_CHARS.search(candidate):
        raise UnsafeInputError("The name contains characters that aren't allowed in a file name.")
    if candidate.split(".", 1)[0].lower() in _RESERVED_NAMES:
        raise UnsafeInputError(f"'{candidate}' is a reserved system name.")
    return candidate


class SafePathResolver:
    """Confines every filesystem action to an explicit set of allowed roots.

    By default the roots are the user's own content folders.  System
    directories, other user profiles and the AARMA installation itself are all
    outside the sandbox, so no registered action can reach them however the
    arguments are phrased.
    """

    def __init__(self, roots: Optional[Sequence[Path]] = None) -> None:
        self._named_roots: Dict[str, Path] = {}
        if roots is None:
            folders = user_content_folders()
            # Fall back to the home directory so the resolver is never empty on
            # a stripped-down profile.
            self._named_roots = {name: self._normalise(path) for name, path in folders.items()}
            resolved = list(self._named_roots.values()) or [self._normalise(Path.home())]
        else:
            resolved = [self._normalise(Path(root)) for root in roots]
        self._roots: tuple[Path, ...] = tuple(resolved)

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._roots

    @property
    def named_roots(self) -> Dict[str, Path]:
        """Canonical folder name -> path, for the folders this resolver allows."""
        return dict(self._named_roots)

    @staticmethod
    def _normalise(path: Path) -> Path:
        """Absolute, symlink-free, case-normalised form used for comparisons."""
        try:
            resolved = path.expanduser().resolve(strict=False)
        except (OSError, RuntimeError):
            resolved = Path(os.path.abspath(os.path.expanduser(str(path))))
        return resolved

    def is_within_roots(self, path: Path) -> bool:
        target = self._normalise(path)
        for root in self._roots:
            try:
                target.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def resolve_base(self, hint: Optional[str]) -> Path:
        """Turn a spoken folder word ("Documents") into a real allowed directory."""
        if not hint:
            return self._roots[0]
        key = hint.strip().strip('"').lower()

        canonical = FOLDER_ALIASES.get(key)
        if canonical:
            known = self._named_roots.get(canonical)
            if known is not None:
                return known
            raise UnsafeInputError(
                f"I don't have a '{canonical}' folder I'm allowed to use on this machine."
            )

        # A subfolder of an allowed root, named relative to the first root.
        candidate = Path(key if Path(key).is_absolute() else hint.strip().strip('"'))
        if not candidate.is_absolute():
            candidate = self._roots[0] / candidate
        resolved = self._normalise(candidate)
        if not self.is_within_roots(resolved):
            raise UnsafeInputError(f"'{hint}' is outside the folders I'm allowed to use.")
        if not resolved.is_dir():
            raise UnsafeInputError(f"I couldn't find a folder called '{hint}'.")
        return resolved

    def resolve_child(self, base_hint: Optional[str], name: str) -> Path:
        """Resolve ``base_hint / name`` after validating both halves."""
        base = self.resolve_base(base_hint)
        child = self._normalise(base / sanitize_component(name))
        if not self.is_within_roots(child):
            raise UnsafeInputError("That location is outside the folders I'm allowed to use.")
        return child

    def resolve_existing(self, path_like: str) -> Path:
        """Resolve an existing file or folder, refusing anything outside the sandbox."""
        raw = (path_like or "").strip().strip('"')
        if not raw:
            raise UnsafeInputError("No path was given.")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self._roots[0] / raw
        resolved = self._normalise(candidate)
        if not self.is_within_roots(resolved):
            raise UnsafeInputError("That path is outside the folders I'm allowed to use.")
        if not resolved.exists():
            raise UnsafeInputError("I couldn't find that file or folder.")
        return resolved

    def describe_roots(self) -> str:
        return ", ".join(str(root) for root in self._roots)


# ---------------------------------------------------------------------- urls

SAFE_URL_SCHEMES = ("http", "https")
_HOSTNAME_RE = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


def is_safe_url(url: str) -> bool:
    """True when ``url`` is an ordinary http(s) web address.

    ``file:``, ``javascript:``, ``data:`` and custom application schemes are all
    rejected: opening those through the default handler is how a "just open a
    link" action turns into arbitrary local execution.
    """
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False
    return parsed.scheme.lower() in SAFE_URL_SCHEMES and bool(parsed.netloc)


def normalize_url(value: str) -> str:
    """Turn what a user says into a safe absolute URL, or raise.

    ``"youtube.com"`` becomes ``https://youtube.com``; anything that is not a
    plausible web address is rejected rather than guessed at.
    """
    raw = (value or "").strip().strip('"').strip("<>")
    if not raw:
        raise UnsafeInputError("No web address was given.")
    if "://" not in raw:
        host = raw.split("/", 1)[0].lower()
        if not _HOSTNAME_RE.match(host) and host not in ("localhost",):
            raise UnsafeInputError(f"'{value}' doesn't look like a web address.")
        raw = f"https://{raw}"
    if not is_safe_url(raw):
        raise UnsafeInputError("I can only open http and https web addresses.")
    parsed = urlparse(raw)
    if any(ch in raw for ch in ("\n", "\r", "\t")):
        raise UnsafeInputError("That web address contains invalid characters.")
    return urlunparse(parsed)


# --------------------------------------------------------------- misc guards

def ensure_text(value: str, *, field: str, max_length: int = 500) -> str:
    """Validate a free-text argument: non-empty, bounded, no control characters."""
    text = (value or "").strip()
    if not text:
        raise UnsafeInputError(f"'{field}' can't be empty.")
    if len(text) > max_length:
        raise UnsafeInputError(f"'{field}' is longer than {max_length} characters.")
    if any(ord(ch) < 32 and ch not in "\t" for ch in text):
        raise UnsafeInputError(f"'{field}' contains control characters.")
    return text


def platform_supported(required: Iterable[str]) -> bool:
    """True when the current platform is in ``required`` (e.g. ``("win32",)``)."""
    return sys.platform in tuple(required)
