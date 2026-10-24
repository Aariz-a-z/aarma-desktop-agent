"""Everything that stands between an intent and a real system change.

The layering is deliberate and must not be short-circuited:

``permissions`` (may AARMA touch this capability at all?)
    -> ``validator`` (are the arguments safe, and how risky is this?)
    -> ``confirmation`` (does the human have to say yes first?)
    -> ``passphrase`` (does the human have to prove who they are?)

No module outside this package may decide on its own that an action is
allowed.
"""

from security.confirmation import ConfirmationBroker, ConfirmationRequest, ConfirmationOutcome
from security.passphrase import PassphraseManager, PassphraseRecord
from security.permissions import (
    PERMISSION_CATALOG,
    Permission,
    PermissionDecision,
    PermissionInfo,
    PermissionManager,
)
from security.validator import (
    RiskLevel,
    RiskPolicy,
    SafePathResolver,
    UnsafeInputError,
    is_safe_url,
    sanitize_component,
)

__all__ = [
    "ConfirmationBroker",
    "ConfirmationOutcome",
    "ConfirmationRequest",
    "PERMISSION_CATALOG",
    "PassphraseManager",
    "PassphraseRecord",
    "Permission",
    "PermissionDecision",
    "PermissionInfo",
    "PermissionManager",
    "RiskLevel",
    "RiskPolicy",
    "SafePathResolver",
    "UnsafeInputError",
    "is_safe_url",
    "sanitize_component",
]
