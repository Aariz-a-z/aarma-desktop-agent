"""The language-model boundary.

Phase 1 ships **no** model.  What it ships is the interface a model will have
to fit behind, and a provider that says so honestly instead of pretending.

The shape of :class:`LLMProvider` encodes the central security rule from
specification section 9: a provider may *propose* a tool call, but it returns
a name and a dictionary of arguments — never a command, a script or a path to
execute.  The proposal is then handed to the ordinary pipeline, where the
action registry rejects unknown names and the permission and risk layers apply
exactly as they do for a typed command.  A model that hallucinates
``run_shell_command`` gets a refusal, not a shell.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from utils.logger import get_logger

logger = get_logger("ai.llm")


@dataclass(frozen=True)
class ToolProposal:
    """A model's suggestion that an action be run.  Never trusted as-is."""

    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass(frozen=True)
class LLMResponse:
    """What a provider returns for one turn."""

    text: str = ""
    proposals: Sequence[ToolProposal] = ()
    available: bool = True
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.available and self.error is None


class LLMProvider(ABC):
    """Interface every language-model backend must implement."""

    #: Short identifier used in configuration, e.g. ``"anthropic"``.
    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool:
        """True when the provider is configured and reachable."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        history: Sequence[Mapping[str, str]] = (),
        tools: Sequence[Mapping[str, Any]] = (),
        language: str = "en",
    ) -> LLMResponse:
        """Produce a reply and, optionally, tool proposals."""

    def describe(self) -> str:
        return f"{self.name} ({'available' if self.is_available() else 'unavailable'})"


class NullLLMProvider(LLMProvider):
    """The provider used when no model is configured.

    It does not fabricate answers.  Every call reports that no model is
    available, which is what lets the engine fall back to the deterministic
    rule-based path and tell the user the truth about what it can do.
    """

    name = "none"

    def is_available(self) -> bool:
        return False

    def complete(
        self,
        prompt: str,
        *,
        history: Sequence[Mapping[str, str]] = (),
        tools: Sequence[Mapping[str, Any]] = (),
        language: str = "en",
    ) -> LLMResponse:
        return LLMResponse(
            text="",
            available=False,
            error="No language model is configured in this build.",
        )


#: Providers this build knows how to construct.  Phase 2 registers real
#: backends here; the key is what ``runtime.llm_provider`` stores.
PROVIDERS: Dict[str, type[LLMProvider]] = {
    "none": NullLLMProvider,
}


def get_provider(name: Optional[str] = None) -> LLMProvider:
    """Construct the configured provider, falling back to the null provider."""
    key = (name or "none").strip().lower()
    provider_class = PROVIDERS.get(key)
    if provider_class is None:
        logger.warning("Unknown LLM provider requested; using none",
                       extra={"context": {"requested": key}})
        provider_class = NullLLMProvider
    return provider_class()


def available_providers() -> List[str]:
    return sorted(PROVIDERS)


def tool_schema_from_registry(registry: Any) -> List[Dict[str, Any]]:
    """Describe the action registry in the shape a tool-calling model expects.

    Phase 2 passes this to the model so it proposes only actions that exist.
    The registry remains the enforcement point regardless -- this function is a
    hint to the model, never a substitute for validation.
    """
    schema: List[Dict[str, Any]] = []
    for spec in registry.all():
        schema.append({
            "name": spec.name,
            "description": spec.description,
            "risk": spec.risk_label,
            "requires_internet": spec.requires_internet,
            "parameters": {
                parameter.name: {
                    "type": parameter.type.value,
                    "description": parameter.description,
                    "required": parameter.required,
                }
                for parameter in spec.parameters
            },
        })
    return schema
