"""Secure storage and verification of the optional security passphrase.

Specification section 13 requires that the passphrase is never stored in
plaintext and never logged.  This module is the *only* place a passphrase
string is allowed to exist, and it exists only long enough to be hashed:

* derivation is PBKDF2-HMAC-SHA256 with a per-user random salt;
* comparison uses :func:`hmac.compare_digest` so verification time does not
  leak how much of the passphrase was correct;
* nothing here writes the passphrase to a logger, and the logging filter in
  :mod:`utils.logger` redacts it as a second line of defence.

The KDF parameters are stored alongside the digest so the iteration count can
be raised in a later release without invalidating existing passphrases.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import Optional

from config.settings import ConfigManager
from utils.logger import get_logger

logger = get_logger("security.passphrase")

#: OWASP-aligned floor for PBKDF2-HMAC-SHA256 at the time of writing.
DEFAULT_ITERATIONS = 210_000
SALT_BYTES = 16
MIN_PASSPHRASE_LENGTH = 8


class PassphraseError(ValueError):
    """Raised when a passphrase fails policy or verification preconditions."""


@dataclass(frozen=True)
class PassphraseRecord:
    """What actually gets persisted -- never the passphrase itself."""

    digest_hex: str
    salt_hex: str
    iterations: int

    @property
    def algorithm(self) -> str:
        return "pbkdf2-hmac-sha256"


def _derive(passphrase: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, iterations)


def hash_passphrase(passphrase: str, *, iterations: int = DEFAULT_ITERATIONS) -> PassphraseRecord:
    """Derive a storable record from a plaintext passphrase."""
    if len(passphrase or "") < MIN_PASSPHRASE_LENGTH:
        raise PassphraseError(
            f"The passphrase must be at least {MIN_PASSPHRASE_LENGTH} characters long."
        )
    salt = secrets.token_bytes(SALT_BYTES)
    digest = _derive(passphrase, salt, iterations)
    return PassphraseRecord(digest_hex=digest.hex(), salt_hex=salt.hex(), iterations=iterations)


def verify_passphrase(passphrase: str, record: PassphraseRecord) -> bool:
    """Constant-time check of ``passphrase`` against a stored record."""
    try:
        salt = bytes.fromhex(record.salt_hex)
        expected = bytes.fromhex(record.digest_hex)
    except ValueError:
        logger.error("Stored passphrase record is malformed")
        return False
    candidate = _derive(passphrase or "", salt, record.iterations)
    return hmac.compare_digest(candidate, expected)


class PassphraseManager:
    """Enable, change, disable, lock and unlock the security passphrase.

    The lock state is deliberately in-memory only: restarting AARMA with
    security enabled starts locked, which is the safe default.
    """

    def __init__(self, config: ConfigManager) -> None:
        self._config = config
        self._unlocked = not self.is_enabled

    # ----------------------------------------------------------------- state

    @property
    def is_enabled(self) -> bool:
        settings = self._config.config.security
        return bool(settings.enabled and settings.passphrase_hash and settings.passphrase_salt)

    @property
    def is_unlocked(self) -> bool:
        return (not self.is_enabled) or self._unlocked

    def stored_record(self) -> Optional[PassphraseRecord]:
        settings = self._config.config.security
        if not (settings.passphrase_hash and settings.passphrase_salt):
            return None
        return PassphraseRecord(
            digest_hex=settings.passphrase_hash,
            salt_hex=settings.passphrase_salt,
            iterations=settings.kdf_iterations or DEFAULT_ITERATIONS,
        )

    # -------------------------------------------------------------- lifecycle

    def enable(self, passphrase: str, *, require_for_high_risk: bool = True) -> None:
        """Turn on security mode with a new passphrase."""
        record = hash_passphrase(passphrase)
        self._config.update_security(
            enabled=True,
            passphrase_hash=record.digest_hex,
            passphrase_salt=record.salt_hex,
            kdf_iterations=record.iterations,
            require_passphrase_for_high_risk=require_for_high_risk,
        )
        self._unlocked = True
        logger.info("Security mode enabled")  # No passphrase material is logged.

    def change(self, current: str, new: str) -> None:
        """Replace the passphrase, proving knowledge of the current one first."""
        if not self.is_enabled:
            raise PassphraseError("Security mode is not enabled.")
        if not self.verify(current):
            raise PassphraseError("The current passphrase is incorrect.")
        record = hash_passphrase(new)
        self._config.update_security(
            passphrase_hash=record.digest_hex,
            passphrase_salt=record.salt_hex,
            kdf_iterations=record.iterations,
        )
        self._unlocked = True
        logger.info("Passphrase changed")

    def disable(self, current: str) -> None:
        """Turn security mode off, again only with the correct passphrase."""
        if not self.is_enabled:
            return
        if not self.verify(current):
            raise PassphraseError("The passphrase is incorrect.")
        self._config.update_security(
            enabled=False,
            passphrase_hash=None,
            passphrase_salt=None,
            kdf_iterations=0,
            require_passphrase_for_high_risk=False,
        )
        self._unlocked = True
        logger.info("Security mode disabled")

    # ------------------------------------------------------------ lock/unlock

    def verify(self, passphrase: str) -> bool:
        record = self.stored_record()
        if record is None:
            return False
        return verify_passphrase(passphrase, record)

    def unlock(self, passphrase: str) -> bool:
        if not self.is_enabled:
            return True
        ok = self.verify(passphrase)
        self._unlocked = ok
        logger.info("Unlock attempt", extra={"context": {"success": ok}})
        return ok

    def lock(self) -> None:
        if self.is_enabled:
            self._unlocked = False
            logger.info("Assistant locked")
