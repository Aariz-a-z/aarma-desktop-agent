"""Turning an intent into a plan the security layer can inspect.

The planner is the only component allowed to promote an :class:`~ai.intent.Intent`
into an :class:`~actions.registry.ActionRequest`.  Keeping that promotion in one
place is what lets the engine assume that *everything* it executes has already
been checked against the registry.

Phase 1 plans are single-step, because the rule-based parser produces one
intent per utterance.  The type is a *sequence* of steps from the start so that
Phase 2's multi-step reasoning ("find my resume, then open it") slots in
without changing the engine, the permission layer or the action layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from actions.registry import ActionRegistry, ActionRequest
from ai.intent import Intent
from language.manager import LanguageManager
from utils.logger import get_logger

logger = get_logger("ai.planner")


#: Refusal reasons that mean "you did not ask me to do anything", as opposed
#: to "I could not work out what you wanted". The first is AARMA behaving
#: correctly; the second is a request it failed to serve, and the interface
#: colours them differently for exactly that reason.
_NOT_A_REQUEST = frozenset({
    "negated",
    "a statement about someone else",
    "reported speech",
    "a question about how, not a request",
    "hypothetical",
    "nothing was asked for",
})


class PlanKind(str, Enum):
    """What the engine should do with a plan."""

    ACTIONS = "actions"     # Execute the steps.
    REPLY = "reply"         # Say something; touch nothing.
    EXIT = "exit"           # Say goodbye and shut down.
    UNKNOWN = "unknown"     # Nothing was understood.


@dataclass(frozen=True)
class Plan:
    """The planner's output: zero or more steps plus an optional spoken reply."""

    kind: PlanKind
    steps: Tuple[ActionRequest, ...] = ()
    reply: Optional[str] = None
    intent: Optional[Intent] = None

    @property
    def has_steps(self) -> bool:
        return bool(self.steps)


class TaskPlanner:
    """Maps intents onto executable, registry-backed plans."""

    def __init__(self, registry: ActionRegistry, language: LanguageManager) -> None:
        self._registry = registry
        self._language = language

    def plan(self, intent: Intent, *, source: str = "text") -> Plan:
        if intent.name == "exit":
            return Plan(kind=PlanKind.EXIT,
                        reply=self._language.t("response.goodbye"), intent=intent)

        if intent.name == "greeting":
            return Plan(kind=PlanKind.REPLY, reply=self._greeting(), intent=intent)

        if intent.name == "thanks":
            return Plan(kind=PlanKind.REPLY, reply="Any time.", intent=intent)

        if intent.name == "help":
            return Plan(kind=PlanKind.REPLY, reply=self.help_text(), intent=intent)

        if intent.name == "clarify":
            # Understood enough to name a capability, not enough to choose.
            # A reply, never a step: asking is the whole outcome.
            question = str(intent.parameters.get("question") or
                           "Which did you mean?")
            return Plan(kind=PlanKind.REPLY, reply=question, intent=intent)

        if intent.name == "declined":
            # Two different situations wear the same label, and they deserve
            # different answers.
            #
            # "I don't want you to open Chrome" is a request *satisfied* by
            # doing nothing: AARMA understood and complied, so it is a REPLY
            # and ok.
            #
            # "Please reticulate the splines" is a request AARMA could not act
            # on. That is the long-standing meaning of UNKNOWN, it renders in
            # the error colour, and flattening the two would have quietly
            # turned every unrecognised command into a success.
            reason = str(intent.parameters.get("reason") or "")
            kind = (PlanKind.REPLY if reason in _NOT_A_REQUEST
                    else PlanKind.UNKNOWN)
            return Plan(kind=kind, reply=self._decline_text(intent),
                        intent=intent)

        if intent.name == "unknown":
            return Plan(kind=PlanKind.UNKNOWN,
                        reply=self._language.t("response.unknown_command"), intent=intent)

        # Any remaining intent must name a registered action.  If the parser
        # ever emits a name the registry does not know -- a bug today, a model
        # hallucination in Phase 2 -- it is refused here rather than executed.
        if intent.name not in self._registry:
            logger.warning("Planner rejected unregistered intent",
                           extra={"context": {"intent": intent.name}})
            return Plan(kind=PlanKind.UNKNOWN,
                        reply=self._language.t("response.unknown_command"), intent=intent)

        request = ActionRequest(
            action=intent.name,
            parameters=dict(intent.parameters),
            source=source,
            raw_utterance=intent.raw_text,
        )
        return Plan(kind=PlanKind.ACTIONS, steps=(request,), intent=intent)

    # ------------------------------------------------------------------ text

    @staticmethod
    def _decline_text(intent: Intent) -> str:
        """Why nothing happened, in words that fit the reason.

        Distinguished per reason because they are genuinely different
        situations: "I won't, because you told me not to" and "I can't do that
        at all" should not read the same.
        """
        reason = str(intent.parameters.get("reason") or "")
        wording = {
            "negated": "Understood — I won't.",
            "a statement about someone else": "Noted. I'll leave that alone.",
            "reported speech": "Noted — I'll take that as something you were "
                               "telling me, not an instruction.",
            "a question about how, not a request":
                "That's a how-to question rather than something to run, so I "
                "haven't done anything.",
            "hypothetical": "I'll wait until you actually want that done.",
            "nothing was asked for": "Nothing to do there, then.",
            "no capability": "I don't have a capability for that.",
            "not confident enough":
                "I'm not sure what you'd like me to do there.",
            "spelling too uncertain for this action":
                "I wasn't sure enough of that to act on it — could you say it "
                "again?",
        }
        return wording.get(reason, "I haven't done anything with that.")

    def _greeting(self) -> str:
        name = self._language.config.config.assistant.name
        return self._language.t("response.greeting", assistant=name)

    def help_text(self) -> str:
        """A grouped, risk-annotated summary of everything this build can do."""
        lines: List[str] = ["Here's what I can do in this build:"]
        for category, specs in sorted(self._registry.by_category().items()):
            lines.append(f"\n{category}")
            for spec in specs:
                marker = {"Low": " ", "Medium": "~", "High": "!"}[spec.risk_label]
                lines.append(f"  {marker} {spec.description}")
                if spec.examples:
                    lines.append(f"      e.g. \"{spec.examples[0]}\"")
        lines.append("\n  ~ changes something   ! high risk, I'll ask first")
        lines.append("Type 'exit' to quit.")
        return "\n".join(lines)
