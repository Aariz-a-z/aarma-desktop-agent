"""Structured logging for AARMA.

Two sinks are configured:

* a rotating file handler under ``<data root>/logs/aarma.log`` holding the full
  developer-facing detail;
* a console handler that stays quiet by default (``WARNING``) so the text-mode
  interface is not polluted by diagnostics.

Every record passes through :class:`SecretRedactingFilter`, which scrubs values
that must never reach disk: passphrases, API keys and tokens (specification
section 29).  Redaction is a defence in depth measure -- callers are still
expected not to pass secrets to the logger in the first place.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
from pathlib import Path
from typing import Any, Mapping, Optional

from utils.paths import get_paths

LOGGER_NAMESPACE = "aarma"

_MAX_BYTES = 2 * 1024 * 1024
_BACKUP_COUNT = 5

#: Substrings that mark a mapping key as sensitive.
SENSITIVE_KEY_HINTS: tuple[str, ...] = (
    "passphrase",
    "password",
    "passcode",
    "secret",
    "api_key",
    "apikey",
    "token",
    "credential",
    "authorization",
    "salt",
    "hash",
)

REDACTED = "***REDACTED***"

# Catches "password=hunter2", "api_key: sk-123", "token" : "abc" in free text.
_INLINE_SECRET_RE = re.compile(
    r"(?i)\b(" + "|".join(re.escape(hint) for hint in SENSITIVE_KEY_HINTS) + r")"
    r"(\"?\s*[:=]\s*\"?)([^\s,;\"'}\]]+)"
)

_configured = False


def redact(value: Any) -> Any:
    """Recursively replace sensitive values inside ``value``.

    Mappings are filtered by key name; strings are filtered by pattern.  The
    structure of the input is preserved so redacted payloads stay readable.
    """
    if isinstance(value, Mapping):
        return {
            key: (REDACTED if _is_sensitive_key(str(key)) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return type(value)(redact(item) for item in value)  # type: ignore[call-arg]
    if isinstance(value, str):
        return _INLINE_SECRET_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", value)
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in SENSITIVE_KEY_HINTS)


class SecretRedactingFilter(logging.Filter):
    """Scrub secrets from both the message and the structured ``extra`` payload."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003 - logging API
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = redact(record.args)  # type: ignore[assignment]
        context = getattr(record, "context", None)
        if context is not None:
            record.context = redact(context)
        return True


class ContextFormatter(logging.Formatter):
    """Append the JSON-encoded ``context`` payload when a record carries one."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = getattr(record, "context", None)
        if not context:
            return base
        try:
            encoded = json.dumps(context, default=str, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            encoded = str(context)
        return f"{base} | {encoded}"


def configure_logging(
    *,
    level: int = logging.INFO,
    console_level: int = logging.WARNING,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Install AARMA's handlers on the ``aarma`` logger.  Idempotent."""
    global _configured
    root = logging.getLogger(LOGGER_NAMESPACE)
    if _configured:
        return root

    target = log_file or get_paths().log_file
    target.parent.mkdir(parents=True, exist_ok=True)

    root.setLevel(min(level, console_level))
    root.propagate = False

    redactor = SecretRedactingFilter()

    file_handler = logging.handlers.RotatingFileHandler(
        target, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        ContextFormatter("%(asctime)s %(levelname)-8s %(name)-22s %(message)s")
    )
    file_handler.addFilter(redactor)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ContextFormatter("[%(levelname)s] %(name)s: %(message)s"))
    console_handler.addFilter(redactor)

    root.addHandler(file_handler)
    root.addHandler(console_handler)
    _configured = True
    root.debug("Logging configured", extra={"context": {"file": str(target)}})
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, e.g. ``get_logger("actions.browser")``."""
    if name.startswith(LOGGER_NAMESPACE):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAMESPACE}.{name}")


def reset_logging() -> None:
    """Detach all handlers.  Used by the test-suite between cases."""
    global _configured
    root = logging.getLogger(LOGGER_NAMESPACE)
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)
    _configured = False
