"""Deciding what was meant -- and whether that is enough to act on.

This file holds the judgements.  Everything before it gathers evidence; this is
where evidence becomes an action, a question, or a refusal.

Four refusals come *before* any scoring, because each describes a sentence that
mentions a capability without requesting it, and a scorer that has already
matched "close the tab" will happily act on all of them:

**Negation.**  "Don't open Chrome" names Chrome and the open verb and must not
open Chrome.

**Reported speech.**  "My friend said close the tab" is a fact about a friend.

**Hypothetical and instructional.**  "I'm thinking about shutting down", "how
do I delete a file" -- describing an action is not requesting it.

**Satisfied state.**  "My volume is already fine" mentions volume and wants
nothing done.

These are checked first and short-circuit, because the cost of the checks
running late is that the sentence has already been resolved into something
plausible and the refusal has to argue with a confident score.

Then confidence decides between acting, asking, and declining.  The thresholds
in :data:`Confidence` were read off the corpus's measured score distribution
rather than chosen in advance -- the evaluation report records the separation
they achieve.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ai.nlu.candidates import Candidate, generate
from ai.nlu.context import ResolutionContext
from ai.nlu.entities import Entities, extract
from ai.nlu.normalize import normalise
from utils.logger import get_logger

logger = get_logger("ai.nlu.resolver")


class Confidence:
    """Where the decision changes.

    Calibrated, not assumed: the paraphrase corpus and the false-positive
    corpus produce two score distributions, and these sit in the gap between
    them.  They are class attributes so a test can move one and prove the
    behaviour depends on it.
    """

    #: Act, subject to every existing permission and confirmation gate.
    ACT = 0.80
    #: Enough to name the action, not enough to pick between readings.
    CLARIFY = 0.55
    #: Below this, say so plainly rather than guessing.
    UNKNOWN = 0.55
    #: Two candidates closer than this are a genuine ambiguity, not a ranking.
    AMBIGUITY_MARGIN = 0.12


#: Risk level above which a *fuzzy* match may not resolve on its own.
#: Spelling repair can reach the volume; it cannot reach anything destructive.
FUZZY_RISK_CEILING = "Low"

# ------------------------------------------------------------- pre-refusals

#: Unambiguous negation only.
#:
#: "Stop" and "cancel" are deliberately absent. They are *commands* in AARMA --
#: "cancel shutdown", "stop the restart", "cancel my 6 PM reminder" -- and
#: reading them as negation would disable the escape hatches the safety design
#: leans on. Cancellation of a pending question is a separate mechanism in
#: :func:`core.pending.is_cancellation`, where it belongs.
_NEGATION = re.compile(
    r"\b(?:do not|don'?t|never|no need to|without|rather not|"
    r"i do not want you to|i don'?t want you to|please do not|please don'?t|"
    r"instead of|not to)\b",
    re.IGNORECASE)

_REPORTED = re.compile(
    r"\b(?:said|says|told me|asked|mentioned|thinks|believes|reckons|"
    r"was saying|were saying|wrote|texted)\b", re.IGNORECASE)

_THIRD_PARTY = re.compile(
    r"\b(?:my (?:friend|brother|sister|mum|mom|dad|boss|colleague|teacher)|"
    r"he|she|they|someone|somebody|everyone)\b", re.IGNORECASE)

_HYPOTHETICAL = re.compile(
    r"\b(?:thinking about|considering|might|maybe i should|wondering (?:if|whether)|"
    r"what if|suppose|imagine|would it|should i|planning to|about to|"
    r"i may|perhaps i)\b", re.IGNORECASE)

_INSTRUCTIONAL = re.compile(
    r"\b(?:how (?:do|does|did|can|would) (?:i|you|we|one)|how to|"
    r"what is the way to|explain how|teach me|tell me how|show me how|"
    r"is it possible to|can one)\b", re.IGNORECASE)

#: A sentence that *opens* with somebody else as its subject is describing the
#: world, not instructing AARMA.  "My friend uses GitHub" names a site and a
#: verb and must still do nothing -- and no reporting verb appears in it, so
#: the reported-speech rule above cannot catch it.
#:
#: Anchored at the start on purpose: "open the file my friend sent" is a
#: perfectly good command that happens to mention a friend later on.
_DECLARATIVE_SUBJECT = re.compile(
    r"^(?:my (?:friend|brother|sister|mum|mom|dad|boss|colleague|teacher|wife|"
    r"husband|partner|son|daughter|neighbou?r)|he|she|they|someone|somebody|"
    r"everyone|nobody|people|everybody)\b",
    re.IGNORECASE)

_SATISFIED = re.compile(
    r"\b(?:already (?:fine|good|ok|okay|set|done|open|closed|loud enough|"
    r"quiet enough)|is fine|are fine|no problem with|happy with|"
    r"do not need|don'?t need)\b", re.IGNORECASE)


@dataclass
class Resolution:
    """What the resolver concluded, and why.

    ``action`` being set does **not** mean anything will run.  It means this is
    the nearest registered capability; permissions, risk and confirmation are
    decided afterwards by machinery this layer does not touch.
    """

    action: str = ""
    confidence: float = 0.0
    entities: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_clarification: bool = False
    clarification: str = ""
    options: Tuple[str, ...] = ()
    refused: str = ""
    reasons: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def resolved(self) -> bool:
        """Whether an action should be attempted at all."""
        return bool(self.action) and not self.requires_clarification \
            and not self.refused

    def as_dict(self) -> Dict[str, Any]:
        return {"intent": self.action, "confidence": round(self.confidence, 3),
                "entities": dict(self.entities),
                "requires_clarification": self.requires_clarification,
                "refused": self.refused}


def _risk_label(registry, action: str) -> str:
    spec = registry.get(action) if registry is not None else None
    return spec.risk_label if spec is not None else "High"


def declines(text: str) -> str:
    """Why this sentence is not a request at all, or "" if it is one.

    Exported because it must run **before** pattern matching, not only before
    scoring.  "I don't want you to open Chrome" matches the ``open_application``
    rule perfectly -- its pattern is unanchored, so "open chrome" is found
    inside the sentence -- and the rule has no way to see the negation in front
    of it.  Negation is a property of the sentence, not of whichever matcher
    happens to claim it, so the check belongs ahead of both.

    Deliberately conservative: it answers "is this clearly *not* an
    instruction", and anything it is unsure about it passes through to the
    matchers unchanged.
    """
    raw = " ".join(str(text or "").lower().split())
    if not raw:
        return ""
    from ai.nlu.normalize import normalise as _normalise

    return NaturalLanguageResolver._pre_refusal(raw, _normalise(raw))


class NaturalLanguageResolver:
    """Nearest registered intent, or an honest answer that there isn't one."""

    def __init__(self, registry, *, context_provider=None) -> None:
        self._registry = registry
        self._context_provider = context_provider

    # -- context ----------------------------------------------------------

    def _context(self) -> ResolutionContext:
        if self._context_provider is None:
            return ResolutionContext()
        try:
            return self._context_provider() or ResolutionContext()
        except Exception:  # noqa: BLE001 - context is a convenience, never a gate
            logger.debug("Context provider failed")
            return ResolutionContext()

    # -- the pre-refusals -------------------------------------------------

    @staticmethod
    def _pre_refusal(raw: str, normalised: str) -> str:
        """Why this sentence is not a request, or "" if it is one.

        Runs on the *raw* text as well as the normalised form: filler stripping
        removes "i want you to", and "i don't want you to open chrome" must not
        become "open chrome" on the way in.
        """
        for text in (raw, normalised):
            if _NEGATION.search(text):
                return "negated"
            if _DECLARATIVE_SUBJECT.match(text.strip()):
                return "a statement about someone else"
            if _REPORTED.search(text) and _THIRD_PARTY.search(text):
                return "reported speech"
            if _INSTRUCTIONAL.search(text):
                return "a question about how, not a request"
            if _HYPOTHETICAL.search(text):
                return "hypothetical"
            if _SATISFIED.search(text):
                return "nothing was asked for"
        return ""

    # -- resolution -------------------------------------------------------

    def resolve(self, text: str,
                context: Optional[ResolutionContext] = None) -> Resolution:
        """Interpret one sentence.  Never raises, never invents an action."""
        started = time.perf_counter()
        raw = str(text or "").strip()
        if not raw:
            return Resolution(refused="empty",
                              latency_ms=(time.perf_counter() - started) * 1000)

        normalised = normalise(raw)
        refusal = self._pre_refusal(raw.lower(), normalised)
        if refusal:
            logger.info("Sentence declined before scoring",
                        extra={"context": {"reason": refusal}})
            return Resolution(refused=refusal,
                              latency_ms=(time.perf_counter() - started) * 1000)

        resolution_context = context if context is not None else self._context()
        entities = extract(normalised)
        candidates = generate(normalised, entities,
                              registry=self._registry,
                              context=resolution_context)

        if not candidates:
            return Resolution(
                entities=entities.as_dict(), refused="no capability",
                latency_ms=(time.perf_counter() - started) * 1000)

        best = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None

        # A fuzzy match must not be the thing that reaches a risky action.
        if best.parameters.get("_fuzzy") and \
                _risk_label(self._registry, best.action) != FUZZY_RISK_CEILING:
            logger.info("Refused a fuzzy match to a non-trivial action",
                        extra={"context": {"action": best.action}})
            return Resolution(
                refused="spelling too uncertain for this action",
                entities=entities.as_dict(),
                latency_ms=(time.perf_counter() - started) * 1000)
        best.parameters.pop("_fuzzy", None)

        result = Resolution(
            action=best.action, confidence=best.confidence,
            entities=entities.as_dict(), parameters=dict(best.parameters),
            reasons=list(best.reasons),
            alternatives=[c.action for c in candidates[1:4]])

        # The candidate itself may know it left a choice open.
        if best.clarification:
            result.requires_clarification = True
            result.clarification = best.clarification
            result.options = best.options

        # Two readings too close to separate is a question, not a ranking.
        elif runner_up is not None and \
                (best.confidence - runner_up.confidence) < Confidence.AMBIGUITY_MARGIN \
                and runner_up.confidence >= Confidence.CLARIFY:
            result.requires_clarification = True
            result.clarification = self._ask_between(best, runner_up)
            result.options = (best.action, runner_up.action)

        elif best.confidence < Confidence.UNKNOWN:
            result.action = ""
            result.refused = "not confident enough"

        elif best.confidence < Confidence.ACT:
            # Named the capability, not sure enough to run it unasked.
            result.requires_clarification = True
            result.clarification = self._ask_to_confirm(best)
            result.options = (best.action,)

        result.latency_ms = (time.perf_counter() - started) * 1000
        logger.info("Resolved by nearest intent", extra={"context": {
            "action": result.action or "(none)",
            "confidence": round(result.confidence, 3),
            "clarify": result.requires_clarification}})
        return result

    # -- wording ----------------------------------------------------------

    def _describe(self, action: str) -> str:
        spec = self._registry.get(action) if self._registry else None
        if spec is None:
            return action.replace("_", " ")
        return spec.description.rstrip(".").lower()

    def _ask_between(self, first: Candidate, second: Candidate) -> str:
        return (f"I could {self._describe(first.action)} or "
                f"{self._describe(second.action)}. Which did you mean?")

    def _ask_to_confirm(self, candidate: Candidate) -> str:
        return f"Did you want me to {self._describe(candidate.action)}?"


__all__ = ["Confidence", "FUZZY_RISK_CEILING", "NaturalLanguageResolver",
           "Resolution", "declines"]
