"""The language abstraction layer.

All human-facing strings and all language decisions go through
:class:`language.manager.LanguageManager`.  Specification section 6 is explicit
that language logic must *not* be scattered across modules, so no other package
may hardcode user-visible English text.
"""

from language.manager import (
    LANGUAGE_CATALOG,
    LanguageInfo,
    LanguageManager,
    LanguageStatus,
    resolve_language_code,
)

__all__ = [
    "LANGUAGE_CATALOG",
    "LanguageInfo",
    "LanguageManager",
    "LanguageStatus",
    "resolve_language_code",
]
