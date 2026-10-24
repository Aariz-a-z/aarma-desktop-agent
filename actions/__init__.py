"""The capability layer: every real thing AARMA can do to the machine.

Nothing outside this package touches the operating system.  Capabilities are
declared as :class:`~actions.registry.ActionSpec` objects and executed only
through :class:`~actions.registry.ActionRegistry`, which is what makes the
"an LLM may never run arbitrary commands" rule enforceable rather than
aspirational (specification sections 9 and 10).
"""

from actions.registry import (
    ActionContext,
    ActionRegistry,
    ActionRequest,
    ActionResult,
    ActionSpec,
    ParameterSpec,
    ParameterType,
    UnknownActionError,
    ValidationFailed,
    build_default_registry,
)

__all__ = [
    "ActionContext",
    "ActionRegistry",
    "ActionRequest",
    "ActionResult",
    "ActionSpec",
    "ParameterSpec",
    "ParameterType",
    "UnknownActionError",
    "ValidationFailed",
    "build_default_registry",
]
