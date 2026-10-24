"""Understanding layer: language in, structured action requests out.

The separation this package enforces is the one specification section 8 asks
for -- raw user language never reaches an action handler.  It is turned into an
:class:`~ai.intent.Intent` first, and only a planner may promote an intent into
an :class:`~actions.registry.ActionRequest`.

Phase 1 understands language with deterministic rules
(:mod:`ai.intent`).  Phase 2 adds a model behind the
:class:`~ai.llm.LLMProvider` interface; because the planner's *output* type
does not change, that swap does not ripple into the security or action layers.
"""

from ai.intent import Intent, IntentEngine, META_INTENTS
from ai.llm import LLMProvider, LLMResponse, NullLLMProvider, get_provider
from ai.memory import ConversationContext, ConversationTurn
from ai.planner import Plan, PlanKind, TaskPlanner

__all__ = [
    "ConversationContext",
    "ConversationTurn",
    "Intent",
    "IntentEngine",
    "LLMProvider",
    "LLMResponse",
    "META_INTENTS",
    "NullLLMProvider",
    "Plan",
    "PlanKind",
    "TaskPlanner",
    "get_provider",
]
