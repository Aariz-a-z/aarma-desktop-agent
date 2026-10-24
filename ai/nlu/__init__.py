"""Nearest-intent understanding: flexible language, deterministic actions.

AARMA's rule engine answers "does this sentence match a pattern I know?".  That
is the right question for a command line and the wrong one for speech, where
the same instruction arrives twenty different ways and the user has no way to
discover the exact wording.  This package answers a different question: **which
registered capability is this person most plausibly asking for, and am I sure
enough to act?**

The order of the pipeline is the whole design:

    text
      -> normalize   strip filler, repair ASR spellings, fold synonyms
      -> candidates  score every *registered* action against the sentence
      -> entities    pull out site, query, folder, application, amount
      -> resolve     confidence, ambiguity, negation, reported speech
      -> Intent      handed to the existing planner and registry unchanged

Four properties hold throughout, and each exists because its opposite is a way
for a helpful guess to become a harmful one:

**Nothing is invented.**  Candidates are generated *from the action registry*,
so the resolver cannot propose a capability that does not exist.  "Fly a drone"
has no nearest intent; it has no intent at all.

**Understanding is not permission.**  Resolving "could you maybe turn the
computer off?" to ``shutdown_computer`` with high confidence changes nothing
about permissions, risk grading or confirmation.  The resolver produces an
:class:`~ai.intent.Intent`; every gate downstream is untouched.

**Guessing is bounded by risk.**  Fuzzy and semantic matching can reach a
volume change.  They cannot, alone, reach anything destructive -- see
:data:`ai.nlu.resolver.FUZZY_RISK_CEILING`.

**Uncertainty is spoken, not resolved.**  When two capabilities are genuinely
plausible the resolver asks.  Choosing the first one and hoping is how an
assistant opens the wrong thing.

The deterministic rules in :mod:`ai.intent` remain the fast path and remain
authoritative: this layer only ever sees sentences they declined.
"""

from ai.nlu.context import ResolutionContext
from ai.nlu.entities import Entities, extract
from ai.nlu.normalize import normalise, repair_asr
from ai.nlu.resolver import (
    Resolution,
    NaturalLanguageResolver,
    Confidence,
    declines,
)

__all__ = [
    "Confidence",
    "Entities",
    "NaturalLanguageResolver",
    "Resolution",
    "ResolutionContext",
    "declines",
    "extract",
    "normalise",
    "repair_asr",
]
