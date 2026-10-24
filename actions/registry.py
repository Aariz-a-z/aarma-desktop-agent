"""The action registry: AARMA's allow-list of executable capabilities.

Every capability must be declared here as an :class:`ActionSpec` carrying its
own permission requirements, risk level and typed parameter schema.  The
registry then guarantees three properties that the rest of the architecture
depends on:

1. **Unknown actions cannot run.**  Looking up a name that was never
   registered raises :class:`UnknownActionError`.  When Phase 2 introduces a
   language model that proposes tool calls, a hallucinated function name hits
   this wall instead of a shell.
2. **Unknown parameters cannot be smuggled in.**  Validation rejects keys the
   spec does not declare, so extra fields never reach a handler.
3. **Risk and permissions travel with the capability**, not with the caller,
   so no code path can execute an action while forgetting to check them.

Handlers receive a validated ``dict`` and an :class:`ActionContext`, and return
an :class:`ActionResult`.  They never format user-facing prose themselves --
that is the language layer's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from config.settings import ConfigManager
from database.db import Database
from language.manager import LanguageManager
from security.permissions import Permission, PermissionManager
from security.validator import RiskLevel, SafePathResolver, UnsafeInputError
from utils.logger import get_logger

logger = get_logger("actions.registry")


class UnknownActionError(LookupError):
    """Raised when a name that is not in the registry is requested."""


class ValidationFailed(ValueError):
    """Raised when supplied parameters do not satisfy an action's schema."""


class ParameterType(str, Enum):
    STRING = "string"
    TEXT = "text"        # Longer free text; same checks, larger bound.
    INTEGER = "integer"
    BOOLEAN = "boolean"
    URL = "url"
    PATH = "path"


_TRUE_WORDS = {"true", "yes", "y", "1", "on"}
_FALSE_WORDS = {"false", "no", "n", "0", "off"}


@dataclass(frozen=True)
class ParameterSpec:
    """Typed declaration of one action argument."""

    name: str
    type: ParameterType = ParameterType.STRING
    description: str = ""
    required: bool = True
    default: Any = None
    choices: Optional[Tuple[str, ...]] = None
    max_length: int = 500
    min_value: Optional[int] = None
    max_value: Optional[int] = None

    def coerce(self, value: Any) -> Any:
        """Validate and normalise a single supplied value.

        Raises :class:`ValidationFailed` with a message written for the user,
        not for a developer -- it is surfaced directly in the conversation.
        """
        if self.type is ParameterType.BOOLEAN:
            return self._coerce_boolean(value)
        if self.type is ParameterType.INTEGER:
            return self._coerce_integer(value)

        text = str(value).strip()
        if not text:
            raise ValidationFailed(f"'{self.name}' can't be empty.")
        limit = 2000 if self.type is ParameterType.TEXT else self.max_length
        if len(text) > limit:
            raise ValidationFailed(f"'{self.name}' is longer than {limit} characters.")
        if any(ord(ch) < 32 and ch != "\t" for ch in text):
            raise ValidationFailed(f"'{self.name}' contains characters I can't accept.")
        if self.choices and text.lower() not in {c.lower() for c in self.choices}:
            raise ValidationFailed(
                f"'{self.name}' must be one of: {', '.join(self.choices)}."
            )
        if self.type is ParameterType.URL:
            from security.validator import normalize_url  # local import keeps the cycle out

            try:
                return normalize_url(text)
            except UnsafeInputError as exc:
                raise ValidationFailed(str(exc)) from exc
        return text

    def _coerce_boolean(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in _TRUE_WORDS:
            return True
        if lowered in _FALSE_WORDS:
            return False
        raise ValidationFailed(f"'{self.name}' must be yes or no.")

    def _coerce_integer(self, value: Any) -> int:
        try:
            number = int(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise ValidationFailed(f"'{self.name}' must be a whole number.") from exc
        if self.min_value is not None and number < self.min_value:
            raise ValidationFailed(f"'{self.name}' must be at least {self.min_value}.")
        if self.max_value is not None and number > self.max_value:
            raise ValidationFailed(f"'{self.name}' must be at most {self.max_value}.")
        return number


@dataclass
class ActionResult:
    """What a handler reports back to the engine."""

    ok: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None

    @classmethod
    def success(cls, message: str = "", **data: Any) -> "ActionResult":
        return cls(ok=True, message=message, data=dict(data))

    @classmethod
    def failure(cls, message: str, *, error_code: str = "action_failed", **data: Any) -> "ActionResult":
        return cls(ok=False, message=message, data=dict(data), error_code=error_code)


@dataclass
class ActionContext:
    """Everything a handler is allowed to reach.

    Handlers get this object instead of module-level globals, which is what
    makes them testable: the suite passes a context backed by a temporary
    config, a temporary database and a sandbox rooted in a temp directory.
    """

    config: ConfigManager
    language: LanguageManager
    database: Database
    permissions: PermissionManager
    paths: SafePathResolver
    session_id: str = "default"
    online: bool = False
    #: The registry itself, so an action can answer questions *about* the
    #: available actions without a hand-written list going stale.
    registry: Optional["ActionRegistry"] = None

    # --- provider overrides ------------------------------------------------
    #
    # Left ``None`` in normal use, when each handler reaches for its own
    # process-wide provider.  A test supplies a fake here instead of patching
    # module globals, which matters because the real ones minimise the windows
    # and press the keys of whoever is running the suite.
    window_provider: Optional[Any] = None
    input_provider: Optional[Any] = None
    weather_provider: Optional[Any] = None
    email_provider: Optional[Any] = None
    calendar_provider: Optional[Any] = None
    #: Phase 5 discovery services, injected by tests so a scan never depends on
    #: the machine running the suite having particular projects installed.
    workspace_policy: Optional[Any] = None
    repository_index: Optional[Any] = None
    application_index: Optional[Any] = None
    #: The repository referred to by "it"/"there" on the previous turn.
    last_repository: Optional[Any] = None
    system_provider: Optional[Any] = None
    media_provider: Optional[Any] = None
    #: Injected by :mod:`actions.power` so no test can really shut a machine down.
    process_runner: Optional[Any] = None
    #: Overrides ``platform.system()`` for the platform-specific handlers.
    platform_system: Optional[str] = None
    #: Ends the voice session, returning True when there was one to end.
    #: A callable rather than the engine itself: an action that could reach
    #: the engine could reach everything, and the registry would depend on
    #: the thing that owns it.
    session_control: Optional[Any] = None


@dataclass(frozen=True)
class ActionRequest:
    """A concrete, validated intent to run one action."""

    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    source: str = "text"          # text | voice | gui
    raw_utterance: str = ""


ActionHandler = Callable[[ActionContext, Dict[str, Any]], ActionResult]

#: Optional cheap check run *before* the user is asked to confirm.  Returns an
#: explanation to refuse, or ``None`` to proceed.  It must not change anything:
#: its only job is to avoid asking someone to approve an action that was always
#: going to be rejected.
ActionPrecheck = Callable[[ActionContext, Dict[str, Any]], Optional[str]]

#: Optional plain-language summary for the confirmation prompt, replacing the
#: generic key/value rendering. Worth writing for any action whose *scope* is
#: the thing being approved -- "close everything" is a different decision
#: depending on what "everything" currently is. It must not change anything,
#: and a failure must cost detail rather than blocking the confirmation.
ActionSummary = Callable[[ActionContext, Dict[str, Any]], str]


@dataclass(frozen=True)
class ActionSpec:
    """The complete declaration of one capability."""

    name: str
    description: str
    category: str
    risk: RiskLevel
    handler: ActionHandler
    permissions: Tuple[Permission, ...] = ()
    parameters: Tuple[ParameterSpec, ...] = ()
    requires_internet: bool = False
    platforms: Optional[Tuple[str, ...]] = None   # ``None`` means every platform.
    examples: Tuple[str, ...] = ()
    precheck: Optional[ActionPrecheck] = None
    summarise: Optional[ActionSummary] = None

    # ---------------------------------------------------------------- schema

    def validate(self, raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        """Validate supplied parameters against this spec.

        Unknown keys are rejected outright rather than ignored: silently
        dropping an argument a caller believed was meaningful is how a
        "create folder in Documents" becomes "create folder somewhere else".
        """
        supplied = dict(raw or {})
        declared = {spec.name: spec for spec in self.parameters}

        unknown = sorted(set(supplied) - set(declared))
        if unknown:
            raise ValidationFailed(
                f"'{self.name}' doesn't accept: {', '.join(unknown)}."
            )

        validated: Dict[str, Any] = {}
        for name, spec in declared.items():
            if name in supplied and supplied[name] is not None:
                validated[name] = spec.coerce(supplied[name])
            elif spec.required:
                raise ValidationFailed(f"I still need '{name}' — {spec.description or 'no value given'}.")
            elif spec.default is not None:
                validated[name] = spec.default
        return validated

    def summary(self) -> str:
        """One-line description used in confirmation prompts and ``help``."""
        return f"{self.name} — {self.description}"

    @property
    def risk_label(self) -> str:
        return self.risk.label


class ActionRegistry:
    """The allow-list.  Lookup is the only way to reach a handler."""

    def __init__(self) -> None:
        self._actions: Dict[str, ActionSpec] = {}

    # ------------------------------------------------------------- mutation

    def register(self, spec: ActionSpec) -> ActionSpec:
        """Add a capability.  Re-registering a name is a programming error."""
        if spec.name in self._actions:
            raise ValueError(f"Action '{spec.name}' is already registered.")
        if not spec.name.replace("_", "").isalnum():
            raise ValueError(f"Action name '{spec.name}' must be alphanumeric with underscores.")
        self._actions[spec.name] = spec
        return spec

    def register_all(self, specs: Iterable[ActionSpec]) -> None:
        for spec in specs:
            self.register(spec)

    # ---------------------------------------------------------------- lookup

    def get(self, name: str) -> Optional[ActionSpec]:
        return self._actions.get(name)

    def require(self, name: str) -> ActionSpec:
        """Fetch a spec or refuse.  This is the hard boundary for tool calls."""
        spec = self._actions.get(name)
        if spec is None:
            logger.warning("Rejected unregistered action", extra={"context": {"action": name}})
            raise UnknownActionError(f"'{name}' is not a registered action.")
        return spec

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._actions

    def __len__(self) -> int:
        return len(self._actions)

    def names(self) -> List[str]:
        return sorted(self._actions)

    def all(self) -> List[ActionSpec]:
        return [self._actions[name] for name in self.names()]

    def by_category(self) -> Dict[str, List[ActionSpec]]:
        grouped: Dict[str, List[ActionSpec]] = {}
        for spec in self.all():
            grouped.setdefault(spec.category, []).append(spec)
        return grouped

    def by_risk(self, risk: RiskLevel) -> List[ActionSpec]:
        return [spec for spec in self.all() if spec.risk is risk]

    def required_permissions(self) -> Dict[Permission, List[str]]:
        """Which actions each permission unlocks -- shown in the settings UI."""
        mapping: Dict[Permission, List[str]] = {}
        for spec in self.all():
            for permission in spec.permissions:
                mapping.setdefault(permission, []).append(spec.name)
        return mapping


def build_default_registry() -> ActionRegistry:
    """Assemble the registry from every action module shipped in this build.

    Imports are local so the action modules can import the types above without
    a circular import at module load time.
    """
    from actions import (
        applications,
        assistant,
        browser,
        calendar,
        capabilities,
        desktop_control,
        email,
        filesystem,
        health,
        keyboard,
        media,
        mouse,
        power,
        reminders,
        screenshot,
        session,
        system,
        weather,
        windows,
        workspace_actions,
    )

    registry = ActionRegistry()
    modules: Sequence[Any] = (
        system, applications, browser, filesystem, screenshot, assistant,
        health, media, power, reminders, calendar, capabilities,
        keyboard, mouse, windows, weather, email,
        workspace_actions, desktop_control, session,
    )
    for module in modules:
        registry.register_all(module.get_actions())

    logger.info("Action registry built",
                extra={"context": {"count": len(registry), "actions": registry.names()}})
    return registry
