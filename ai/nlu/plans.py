"""Sentences that ask for more than one thing.

"Open my AARMA repo in VS Code and open a terminal there" is two actions and
one instruction.  Phase 4 resolved a single nearest intent and reported the
rest as unmet; this splits the sentence and resolves each half.

The design is mostly about what happens when a step fails, because that is
where a multi-step plan can quietly mislead:

**A later step never runs on a failed earlier one.**  If the repository cannot
be found, opening a terminal "there" has no *there*, and running it against
whatever the previous context held would be worse than stopping.

**Partial success is reported as partial.**  The result says which steps ran
and which did not.  "Done" after one of two steps is the kind of summary that
makes someone believe their terminal is open when it is not.

**Splitting is conservative.**  Only a small set of connectives splits a
sentence, and only when both halves resolve to something.  "Open the file I
downloaded and edited yesterday" is one request, and a splitter that cut it at
"and" would invent a second action out of an adjective.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.logger import get_logger

logger = get_logger("ai.nlu.plans")

#: Connectives that can join two whole instructions.  Deliberately short: each
#: one is a place a sentence might be cut in two, and a wrong cut invents an
#: action nobody asked for.
_SPLIT = re.compile(
    r"\s*(?:,\s*(?:and\s+)?then\s+|\s+and\s+then\s+|\s*,\s*then\s+|"
    r"\s+then\s+|\s+and\s+also\s+|\s+and\s+)",
    re.IGNORECASE)

#: More than this many steps in one sentence is far more likely to be a
#: mis-split than a genuine request.
MAX_STEPS = 3


@dataclass
class PlannedStep:
    """One resolved action within a larger instruction."""

    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    text: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "parameters": dict(self.parameters),
                "confidence": round(self.confidence, 3), "text": self.text}


@dataclass
class ActionPlan:
    """An ordered set of steps, with the sentence each came from."""

    steps: List[PlannedStep] = field(default_factory=list)
    source: str = ""
    #: Fragments that were asked for but could not be resolved. Present only
    #: when earlier fragments *did* resolve, so the caller can run what it
    #: understood and still say plainly what it did not.
    unmet: List[str] = field(default_factory=list)

    @property
    def is_multi(self) -> bool:
        return len(self.steps) > 1

    def as_dict(self) -> Dict[str, Any]:
        return {"steps": [step.as_dict() for step in self.steps],
                "source": self.source, "unmet": list(self.unmet)}


def split_sentence(text: str) -> List[str]:
    """Cut an instruction into fragments at its connectives.

    Returns a single-element list when the sentence does not split, so callers
    have one shape to handle.
    """
    working = " ".join(str(text or "").split())
    if not working:
        return []
    fragments = [f.strip(" ,.;!?") for f in _SPLIT.split(working)]
    fragments = [f for f in fragments if f]
    if len(fragments) < 2:
        return [working]
    return fragments[:MAX_STEPS]


def build(text: str, resolver, context=None) -> Optional[ActionPlan]:
    """Resolve a multi-step sentence, or ``None`` when it is not one.

    ``None`` is returned whenever the sentence is a single instruction *or*
    when any fragment fails to resolve -- in which case the caller falls back
    to resolving the whole sentence, which is nearly always what a sentence
    containing "and" actually is.

    That resolution rule is also what keeps a noun-tail from becoming a step,
    and it replaced an earlier minimum-word-count rule that tried to do the
    same job by length. Measured on this project's own resolver, every tail the
    length rule existed to catch is declined anyway -- "github" and "calculator"
    and "terminal" as "not confident enough", "eggs" as "no capability" -- while
    the length rule additionally threw away "mute", a real one-word command.
    The cost of that was not a missing feature but a misleading answer:
    "mute and shut down the computer" refused to split, matched the unanchored
    mute rule on the whole sentence, and reported success for half a request.
    """
    fragments = split_sentence(text)
    if len(fragments) < 2:
        return None

    steps: List[PlannedStep] = []
    unmet: List[str] = []
    for fragment in fragments:
        resolution = resolver.resolve(fragment, context)
        if not resolution.resolved:
            logger.info("Fragment did not resolve", extra={"context": {
                "fragment": fragment,
                "reason": resolution.refused or "needs clarification"}})
            # Nothing has resolved yet, so this is not a multi-step sentence at
            # all -- it is one instruction that contains a connective, and the
            # caller should resolve it whole.
            if not steps:
                return None
            # Something before it did resolve. Record the rest as unmet rather
            # than discarding it: running only the understood half and calling
            # that success is the report this module exists to avoid.
            unmet.append(fragment)
            continue
        if unmet:
            # A gap in the middle means the split is not trustworthy as an
            # ordered plan -- a later step may have depended on the one that
            # went missing. Fall back to the whole sentence.
            logger.info("Rejected a plan with a gap in the middle")
            return None
        steps.append(PlannedStep(action=resolution.action,
                                 parameters=dict(resolution.parameters),
                                 confidence=resolution.confidence,
                                 text=fragment))

    # Two fragments resolving to the *same* action is a split that went wrong:
    # "close chrome and firefox" should not become two identical closes.
    if len({step.action for step in steps}) == 1 and len(steps) > 1:
        logger.info("Rejected a plan whose steps were all the same action")
        return None

    logger.info("Multi-step plan built", extra={"context": {
        "steps": [step.action for step in steps], "unmet": unmet}})
    return ActionPlan(steps=steps, source=" ".join(fragments), unmet=unmet)


def carry_context(plan: ActionPlan, registry=None) -> ActionPlan:
    """Let a later step inherit the target an earlier one established.

    "Open my AARMA repo in VS Code and open a terminal there" names the project
    once.  The second step says "there", which only means something because of
    the first -- so the repository name is copied forward when the later step
    did not name one of its own.

    ``registry`` is how this stays correct rather than merely helpful.  Without
    it, "open Downloads and take a screenshot" copied ``location`` into
    ``take_screenshot``, which does not declare that parameter -- the step
    would have been rejected by schema validation at execution time, after the
    first step had already run.  A value is only carried into a step whose
    action actually accepts it.
    """
    carried: Dict[str, Any] = {}
    for step in plan.steps:
        accepted = None
        if registry is not None:
            spec = registry.get(step.action)
            accepted = {p.name for p in spec.parameters} if spec else set()

        for key in ("name", "site", "location"):
            value = step.parameters.get(key)
            if value:
                carried[key] = value
                continue
            if key not in carried or key in step.parameters:
                continue
            if accepted is not None and key not in accepted:
                continue
            step.parameters[key] = carried[key]
    return plan


__all__ = ["ActionPlan", "MAX_STEPS", "PlannedStep",
           "build", "carry_context", "split_sentence"]
