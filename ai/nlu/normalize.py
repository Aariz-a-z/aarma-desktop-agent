"""Turning what was heard into something matchable, without changing it.

Everything here is reversible in meaning: filler is removed, spellings are
repaired, spacing is folded.  Nothing is *interpreted* -- that is the resolver's
job, and keeping the two apart is what makes a wrong answer debuggable.

The ASR repairs are deliberately narrow.  A speech recogniser produces "you
tube" for YouTube and "power shell" for PowerShell reliably enough to be worth
a table; it does not produce arbitrary words that should be bent into commands.
So repairs are applied to a **fixed list of known targets** and never as
general-purpose spell-correction -- the difference between fixing "gitub" and
deciding that "delegate" was probably "delete".
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

from ai.nlu.lexicon import APPLICATION_ASR, SITE_ALIASES

#: Filler that carries no instruction.  Stripped from the front only: "please
#: close this" is a command, "how do I please a crowd" is not a close request,
#: and anchoring keeps the second one intact.
_LEAD_FILLER = re.compile(
    r"^(?:"
    r"hey|hi|hello|ok|okay|so|well|um+|uh+|err?|right|now|just|"
    r"could you|can you|would you|will you|please|kindly|"
    r"i want you to|i'?d like you to|i would like you to|i need you to|"
    r"i want to|i'?d like to|i would like to|i need to|i wanna|"
    r"let'?s|lets|help me|do me a favou?r and|be a dear and"
    r")\b[,\s]*",
    re.IGNORECASE)

_TRAIL_FILLER = re.compile(
    r"[,\s]*\b(?:please|for me|thanks|thank you|if you (?:can|could|would)|"
    r"would you|ok|okay|alright|right now|real quick|quickly)\b[.!?]*$",
    re.IGNORECASE)

#: Contractions expanded so one spelling reaches the matcher.
_CONTRACTIONS: Dict[str, str] = {
    "what's": "what is", "how's": "how is", "that's": "that is",
    "it's": "it is", "i'm": "i am", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "can't": "can not", "won't": "will not",
    "shouldn't": "should not", "wouldn't": "would not", "couldn't": "could not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "i'll": "i will", "let's": "let us", "i've": "i have",
}

#: Multi-word forms folded to one token so they survive word splitting.
_PHRASE_FOLD: List[Tuple[re.Pattern, str]] = [
    (re.compile(rf"\b{re.escape(spoken)}\b", re.IGNORECASE), canonical)
    for spoken, canonical in sorted(
        {**SITE_ALIASES, **APPLICATION_ASR}.items(),
        key=lambda item: -len(item[0]))
    if " " in spoken or spoken != canonical
]


def _strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(ch))


def repair_asr(text: str) -> str:
    """Fold known mis-spacings and mishearings onto their canonical spelling.

    Applied only to the fixed tables in :mod:`ai.nlu.lexicon`.  This is a
    lookup, not a corrector: a word that is not in the tables is left exactly
    as heard, because bending unknown words toward known commands is how
    "delegate this" becomes "delete this".
    """
    working = str(text or "")
    for pattern, canonical in _PHRASE_FOLD:
        working = pattern.sub(canonical, working)
    return working


def expand_contractions(text: str) -> str:
    working = str(text or "")
    for short, long in _CONTRACTIONS.items():
        working = re.sub(rf"\b{re.escape(short)}\b", long, working,
                         flags=re.IGNORECASE)
    return working


def strip_filler(text: str) -> str:
    """Remove leading politeness and trailing pleasantries, repeatedly.

    Looped because they stack: "hey, could you please just open chrome" has
    four layers, and one pass would leave three of them in front of the verb.
    """
    working = str(text or "").strip()
    for _ in range(6):
        reduced = _LEAD_FILLER.sub("", working, count=1).strip()
        reduced = _TRAIL_FILLER.sub("", reduced, count=1).strip()
        if reduced == working or not reduced:
            break
        working = reduced
    return working


def normalise(text: str) -> str:
    """The matchable form: lower case, no accents, no filler, one spacing."""
    working = _strip_accents(str(text or ""))
    working = working.lower()
    working = expand_contractions(working)
    working = repair_asr(working)
    working = re.sub(r"[‘’]", "'", working)
    working = strip_filler(working)
    working = re.sub(r"\s+", " ", working).strip()
    return working.strip(" .!?,;")


# ------------------------------------------------------------ fuzzy matching

def edit_distance(left: str, right: str, *, ceiling: int = 3) -> int:
    """Levenshtein distance, abandoned once it passes ``ceiling``.

    Bounded because the only question asked of it is "is this within one or two
    typos", and computing an exact large distance to answer "no" is wasted work
    on a path that must stay under a hundred milliseconds.
    """
    if left == right:
        return 0
    if abs(len(left) - len(right)) > ceiling:
        return ceiling + 1
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        best = current[0]
        for j, b in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (a != b)))
            best = min(best, current[-1])
        if best > ceiling:
            return ceiling + 1
        previous = current
    return previous[-1]


#: Below this length a single edit changes too much of the word to be a typo:
#: "cat" and "cut" are both words, and treating one as the other is a guess.
MIN_FUZZY_LENGTH = 5


#: The floor when some *other* part of the sentence is already certain.  "Opn
#: chrome" names Chrome exactly, so the only thing in doubt is a three-letter
#: verb -- one uncertain token beside one certain one, rather than a guess
#: resting on a guess.  Callers must supply the certainty; this only relaxes
#: the length rule.
CORROBORATED_FUZZY_LENGTH = 3


def close_enough(heard: str, target: str, *, corroborated: bool = False) -> bool:
    """Whether ``heard`` is plausibly a mistyping of ``target``.

    The length floor matters more than the distance: at four characters almost
    everything is within one edit of everything else, so short words must match
    exactly or not at all.  ``corroborated`` lowers that floor, and callers
    pass it only when another element of the sentence is known exactly --
    never to make a lone word match harder.
    """
    heard, target = heard.lower(), target.lower()
    if heard == target:
        return True
    floor = CORROBORATED_FUZZY_LENGTH if corroborated else MIN_FUZZY_LENGTH
    if len(target) < floor or len(heard) < floor:
        return False
    allowed = 1 if len(target) < 8 else 2
    return edit_distance(heard, target, ceiling=allowed) <= allowed


__all__ = ["CORROBORATED_FUZZY_LENGTH", "MIN_FUZZY_LENGTH",
           "close_enough", "edit_distance",
           "expand_contractions", "normalise", "repair_asr", "strip_filler"]
