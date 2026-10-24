"""Turning what the user said into a structured intent.

Phase 1 uses ordered rules rather than a model.  That is a deliberate choice,
not a shortcut: it makes the whole pipeline deterministic and testable *before*
a language model is allowed anywhere near it, and it forces the interface
between "language" and "action" to be designed properly up front.  Phase 2
replaces the body of :meth:`IntentEngine.parse` with a model call that emits
the same :class:`Intent` objects — nothing downstream changes.

Two properties matter more than coverage:

* **Rules validate before they claim a match.**  "open chrome" only becomes
  ``open_application`` if *chrome* actually resolves in the approved
  catalogue; otherwise the rule declines and the next one gets a turn.  This
  keeps the parser from confidently producing arguments that the action layer
  will then reject.
* **An unrecognised sentence produces ``unknown``, never a guess.**  Guessing
  is how an assistant opens the wrong thing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Pattern, Sequence, Tuple

from config.settings import ConfigManager
from utils.logger import get_logger

logger = get_logger("ai.intent")

#: Intents handled by the engine itself rather than by a registered action.
META_INTENTS: frozenset[str] = frozenset({
    "exit", "help", "greeting", "thanks", "confirm", "decline", "unknown",
    # Phase 4. The resolver understood the sentence well enough to name a
    # capability but not well enough to pick one reading, so the assistant
    # asks. Meta rather than an action: nothing executes.
    "clarify",
    # The sentence mentioned a capability without requesting it -- negation,
    # reported speech, a question about how. Distinct from "unknown", which
    # means nothing was recognised at all, because the two deserve different
    # replies.
    "declined",
})


@dataclass(frozen=True)
class _FragmentMatch:
    """One fragment's outcome, in the shape :mod:`ai.nlu.plans` expects."""

    resolved: bool = False
    action: str = ""
    parameters: Dict[str, object] = field(default_factory=dict)
    confidence: float = 0.0
    refused: str = ""
    requires_clarification: bool = False
    clarification: str = ""
    options: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Intent:
    """A structured interpretation of one utterance."""

    name: str
    parameters: Dict[str, object] = field(default_factory=dict)
    #: The matching rule's declared confidence. **A constant, not a score** --
    #: every rule states how sure it is when it fires, and no ranking happens.
    #: Recorded under this name so the pipeline trace does not imply a
    #: similarity measurement that the engine never computed.
    confidence: float = 0.0
    #: How the match was made. ``pattern`` is the only method today; the field
    #: exists so that adding synonym or fuzzy matching later shows up in the
    #: trace data rather than being invisible.
    match_method: str = "none"
    raw_text: str = ""
    normalised_text: str = ""

    @property
    def is_meta(self) -> bool:
        return self.name in META_INTENTS

    @property
    def is_action(self) -> bool:
        return not self.is_meta


#: A validator receives the regex groups and returns the action parameters, or
#: ``None`` to decline the match and let a later rule try.
Validator = Callable[[Dict[str, str]], Optional[Dict[str, object]]]


@dataclass(frozen=True)
class IntentRule:
    intent: str
    patterns: Tuple[Pattern[str], ...]
    confidence: float = 0.9
    validate: Optional[Validator] = None

    def match(self, text: str) -> Optional[Dict[str, object]]:
        for pattern in self.patterns:
            found = pattern.search(text)
            if not found:
                continue
            groups = {k: (v or "").strip() for k, v in found.groupdict().items()}
            if self.validate is None:
                return {k: v for k, v in groups.items() if v}
            resolved = self.validate(groups)
            if resolved is not None:
                return resolved
        return None


def _compile(*expressions: str) -> Tuple[Pattern[str], ...]:
    return tuple(re.compile(expression, re.IGNORECASE) for expression in expressions)


# ------------------------------------------------------------------ validators

def _validate_application(groups: Dict[str, str]) -> Optional[Dict[str, object]]:
    """Only claim an application match if the name is in the approved catalogue.

    Terminals are excluded deliberately.  "Open cmd" has to be claimed by the
    ``open_terminal`` rule or by nothing: routing it here would let a shell
    window through the permission set meant for Notepad.
    """
    from actions.applications import resolve_program

    name = groups.get("name", "")
    return {"name": name} if resolve_program(name) else None


def _validate_terminal(groups: Dict[str, str]) -> Optional[Dict[str, object]]:
    """Only claim a terminal match for a shell that is actually approved."""
    from actions.applications import resolve_terminal

    name = groups.get("name", "")
    return {"name": name} if resolve_terminal(name) else None


def _validate_folder(groups: Dict[str, str]) -> Optional[Dict[str, object]]:
    """Only claim a folder match for one of the user's real content folders."""
    from security.validator import FOLDER_ALIASES

    location = " ".join(groups.get("location", "").lower().split())
    location = location.removesuffix(" folder").strip()
    return {"location": location} if location in FOLDER_ALIASES else None


def _validate_site_search(groups: Dict[str, str]) -> Optional[Dict[str, object]]:
    """Only claim the match when the named site is one AARMA actually knows.

    Declining is what keeps this rule in its lane: "search my notes for the
    address" names no known site, so the rule steps aside and the ordinary web
    search -- or the resolver -- takes the sentence instead of turning "my
    notes" into a site nobody can open.
    """
    from ai.nlu.lexicon import canonical_site

    site = canonical_site(groups.get("site", ""))
    query = " ".join(str(groups.get("query", "")).split())
    if not site or not query:
        return None
    return {"site": site, "query": query}


def _validate_keys(groups: Dict[str, str]) -> Optional[Dict[str, object]]:
    """Only claim a keyboard match if the key name really parses.

    This is what keeps "press ahead with it" from becoming a keystroke: the
    rule declines and a later rule -- or ``unknown`` -- gets the sentence,
    rather than the action layer receiving a confident nonsense argument.
    """
    from providers.input import UnknownKeyError, parse_chord

    keys = groups.get("keys", "")
    try:
        parse_chord(keys)
    except UnknownKeyError:
        return None
    return {"keys": keys}


def _validate_language(groups: Dict[str, str]) -> Optional[Dict[str, object]]:
    from language.manager import resolve_language_code

    value = groups.get("language", "")
    code = resolve_language_code(value)
    if code is None:
        return None
    parameters: Dict[str, object] = {"language": code}
    if groups.get("scope"):
        parameters["scope"] = groups["scope"]
    return parameters


def _validate_url(groups: Dict[str, str]) -> Optional[Dict[str, object]]:
    """Accept a bare site shortcut or anything that normalises to http(s)."""
    from actions.browser import SITE_SHORTCUTS
    from security.validator import UnsafeInputError, normalize_url

    raw = groups.get("url", "").rstrip(".,!?")
    if not raw:
        return None
    if raw.lower() in SITE_SHORTCUTS:
        return {"url": raw.lower()}
    try:
        normalize_url(raw)
    except UnsafeInputError:
        return None
    return {"url": raw}


def _validate_non_empty(*names: str) -> Validator:
    def _validator(groups: Dict[str, str]) -> Optional[Dict[str, object]]:
        values = {name: groups.get(name, "").strip() for name in names}
        if not all(values[name] for name in names):
            return None
        optional = {k: v for k, v in groups.items() if k not in names and v}
        return {**values, **optional}
    return _validator



#: Words that follow "call you" in a sentence that is not about a name.
#: "I'll call you back" and "I'll call you tomorrow" are about telephoning, and
#: renaming the assistant to "back" would be an unrecoverable kind of silly.
_NOT_A_NAME = frozenset({
    "back", "later", "tomorrow", "tonight", "soon", "then", "again", "now",
    "that", "this", "it", "him", "her", "them", "me", "you", "us", "something",
    "anything", "whatever", "sometime", "someday",
})

#: A captured name starting with one of these is a phrase, not a name:
#: "call you on Friday" is a day, not a rename.
_NOT_A_NAME_FIRST = frozenset({
    "on", "at", "in", "when", "if", "about", "after", "before", "around",
    "by", "during", "next", "last", "this",
})

#: Trailing decoration people type or say around a name.
_NAME_EDGES = " \t\"'\u201c\u201d\u2018\u2019.!?,;:"


def _validate_assistant_name(groups: Dict[str, str]) -> Optional[Dict[str, object]]:
    """Accept a rename only when what was captured is plausibly a name.

    The rule has to be generous about phrasing and strict about meaning.  People
    say "from now on I'll call you Friday" and mean it as an instruction; they
    also say "I'll call you tomorrow" and "my friend's name is Friday" and mean
    nothing of the sort.  The patterns handle the first kind by requiring the
    sentence to be addressed to the assistant; this handles the rest.
    """
    raw = (groups.get("name") or "").strip().strip(_NAME_EDGES)
    # "I'll call you Friday from now on" -- the instruction can trail the name.
    raw = re.sub(r"\s+from now on$", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"^(?:the\s+name\s+)", "", raw, flags=re.IGNORECASE).strip()
    # "Your name is Friday now." -- the trailing adverb is not part of the name.
    raw = re.sub(r"\s+(?:now|from\s+now\s+on|please|ok(?:ay)?)$", "", raw,
                 flags=re.IGNORECASE).strip()
    raw = raw.strip(_NAME_EDGES)
    if not raw or len(raw) > 32:
        return None
    if not any(character.isalpha() for character in raw):
        return None
    words = raw.lower().split()
    if not words:
        return None
    # Check the first word as well as the whole string: the guard used to
    # compare only the whole capture, so "back" was rejected but
    # "back tomorrow" was accepted and renamed the assistant to "back tomorrow".
    if (raw.lower() in _NOT_A_NAME or words[0] in _NOT_A_NAME
            or words[0] in _NOT_A_NAME_FIRST):
        return None
    if len(words) > 1 and words[-1] in _NOT_A_NAME:
        return None
    if len(words) > 4:            # a sentence, not a name
        return None
    return {"name": raw}


# ---------------------------------------------------------------------- rules

#: Order is significant: the first rule that both matches *and* validates wins.
#: More specific phrasings therefore come before broader catch-alls.
INTENT_RULES: Sequence[IntentRule] = (
    IntentRule(
        intent="exit",
        # Anchored at both ends on purpose: "quit" ends the session, but
        # "quit spotify" must still reach close_application further down.
        patterns=_compile(
            r"^(?:exit|quit|bye|goodbye|good bye|see you|log ?out|"
            r"close aarma|shut down aarma)$"
        ),
        confidence=0.99,
    ),
    IntentRule(
        intent="help",
        patterns=_compile(
            r"^(help|commands?|what can you do|show (me )?(the )?commands?|"
            r"what are your (capabilities|abilities))\b"
        ),
        confidence=0.98,
    ),
    IntentRule(
        intent="greeting",
        patterns=_compile(r"^(hi|hello|hey|yo|good (morning|afternoon|evening))\b"),
        confidence=0.9,
    ),
    IntentRule(
        intent="thanks",
        patterns=_compile(r"^(thanks|thank you|thx|cheers)\b"),
        confidence=0.9,
    ),

    # --- machine health -----------------------------------------------------
    # Ordered before the broader information rules: "how much memory" is a
    # specific question and must not be answered by a generic status report.
    IntentRule(
        intent="cpu_temperature",
        patterns=_compile(
            r"\b(?:cpu |processor )?temperature\b",
            r"\b(?:is|are) (?:my |the )?(?:laptop|pc|computer|cpu)\b.*\bover ?heat",
            r"\bover ?heating\b",
            r"\bhow hot\b",
        ),
    ),
    IntentRule(
        intent="cpu_usage",
        patterns=_compile(
            r"\bcpu (?:usage|load|use)\b",
            r"\bhow much cpu\b",
            r"\b(?:show|check|what'?s) (?:my |the )?cpu\b",
        ),
    ),
    IntentRule(
        intent="memory_usage",
        patterns=_compile(
            r"\b(?:ram|memory) (?:usage|use|used)\b",
            r"\bhow much (?:ram|memory)\b",
            r"\b(?:show|check|what'?s) (?:my |the )?(?:ram|memory)\b",
        ),
    ),
    IntentRule(
        intent="disk_usage",
        patterns=_compile(
            r"\b(?:disk|drive|storage) (?:space|usage|use)\b",
            r"\bhow much (?:disk|drive|storage|space)\b",
            r"\bfree space\b",
        ),
    ),
    IntentRule(
        intent="battery_status",
        patterns=_compile(
            r"\bbattery\b",
            r"\b(?:am i|is it) charging\b",
        ),
    ),
    IntentRule(
        intent="top_processes",
        patterns=_compile(
            r"\bwhat(?:'?s| is) using (?:the )?most (?P<resource>cpu|memory|ram)\b",
            r"\bwhat(?:'?s| is) using (?:up )?(?:my |the )?(?P<resource>cpu|memory|ram)\b",
            r"\b(?:top|heaviest) (?:processes|programs|apps)\b",
            r"\bshow (?:me )?running (?:processes|applications|programs|apps)\b",
        ),
    ),
    IntentRule(
        intent="system_info",
        patterns=_compile(
            r"\bsystem info(?:rmation)?\b",
            r"\b(?:what are|show) (?:my |the )?specs?\b",
            r"\bmachine spec",
        ),
    ),
    IntentRule(
        intent="system_health",
        patterns=_compile(
            r"\bsystem health\b",
            r"\bhow(?:'?s| is) (?:my |the )?(?:pc|computer|laptop|machine|system)\b",
            r"\bcheck (?:my |the )?(?:pc|computer|laptop|machine|system)\b",
            r"\bhealth check\b",
        ),
    ),

    # --- media --------------------------------------------------------------
    IntentRule(
        intent="set_volume",
        patterns=_compile(
            r"\bset (?:the )?volume to (?P<percent>\d{1,3})\s*(?:percent|%)?\b",
            r"\bvolume (?:to )?(?P<percent>\d{1,3})\s*(?:percent|%)\b",
        ),
    ),
    IntentRule(
        intent="volume_up",
        patterns=_compile(
            r"\b(?:turn|put) (?:the |up the )?(?:volume|sound|audio)(?: up)?\b(?=.*\bup\b)",
            r"\b(?:increase|raise) (?:the )?(?:volume|sound|audio)\b",
            r"\bset (?:the )?(?:volume|sound|audio) (?:higher|up|louder)\b",
            r"\bvolume up\b", r"\blouder\b",
        ),
    ),
    IntentRule(
        intent="volume_down",
        patterns=_compile(
            r"\b(?:turn|put) (?:the )?(?:volume|sound|audio) down\b",
            r"\b(?:decrease|lower|reduce) (?:the )?(?:volume|sound|audio)\b",
            # "set volume lower" -- this is stepping, not an absolute
            # level, so it belongs here rather than with set_volume, which
            # refuses outright because it cannot land on an exact figure.
            r"\bset (?:the )?(?:volume|sound|audio) (?:lower|down|quieter)\b",
            r"\bvolume down\b", r"\bquieter\b", r"\bturn it down\b",
        ),
    ),
    IntentRule(
        intent="mute",
        patterns=_compile(r"\b(?:un)?mute\b"),
    ),
    IntentRule(
        intent="next_track",
        patterns=_compile(
            r"\b(?:next|skip) (?:the )?(?:track|song)\b",
            r"\bplay the next (?:song|track)\b",
            r"\bskip this (?:song|track)\b",
        ),
    ),
    IntentRule(
        intent="previous_track",
        patterns=_compile(
            r"\b(?:previous|last) (?:track|song)\b",
            r"\b(?:go back|back) a (?:track|song)\b",
        ),
    ),
    IntentRule(
        intent="stop_playback",
        patterns=_compile(r"\bstop (?:the )?(?:music|playback|song|audio)\b"),
    ),
    IntentRule(
        intent="play_pause",
        patterns=_compile(
            r"\bpause (?:the )?(?:music|song|video|playback)?\b",
            r"\bresume (?:the )?(?:music|song|playback)?\b",
            r"^\s*(?:play|pause)\s*$",
            r"\bplay the music\b",
        ),
    ),

    # --- power --------------------------------------------------------------
    # "cancel" comes first: it must win over the shutdown patterns it contains.
    IntentRule(
        intent="cancel_shutdown",
        patterns=_compile(
            r"\bcancel (?:the )?(?:shut ?down|restart|reboot)\b",
            r"\bstop (?:the )?(?:shut ?down|restart|reboot)\b",
            r"\babort (?:the )?(?:shut ?down|restart)\b",
        ),
    ),
    IntentRule(
        intent="lock_screen",
        patterns=_compile(
            r"\block (?:my |the )?(?:pc|computer|screen|laptop|machine)\b",
            r"^\s*lock\s*$",
        ),
    ),
    IntentRule(
        intent="sleep_computer",
        # Every phrasing names the machine. A bare "go to sleep" is addressed
        # to the assistant and belongs to `end_voice_session`: suspending
        # someone's computer because they told their assistant to stop
        # listening is not a recoverable misunderstanding.
        patterns=_compile(
            r"\b(?:put|send) (?:my |the )?(?:pc|computer|laptop|machine) to sleep\b",
            r"\bsleep (?:my |the )(?:pc|computer|laptop|machine)\b",
            r"\bsuspend (?:my |the )?(?:pc|computer|laptop|machine)\b",
            r"\b(?:pc|computer|laptop|machine) (?:should )?go to sleep\b",
        ),
    ),
    IntentRule(
        intent="sign_out",
        patterns=_compile(
            r"\bsign (?:me )?out\b", r"\blog (?:me )?(?:out|off)\b",
        ),
    ),
    IntentRule(
        intent="restart_computer",
        patterns=_compile(
            r"\brestart (?:my |the )?(?:pc|computer|laptop|machine|system)\b",
            r"\breboot (?:my |the )?(?:pc|computer|laptop|machine)?\b",
        ),
    ),
    IntentRule(
        intent="shutdown_computer",
        patterns=_compile(
            r"\bshut ?down (?:my |the )?(?:pc|computer|laptop|machine|system)?\b",
            r"\bturn off (?:my |the )(?:pc|computer|laptop|machine)\b",
            r"\bpower off\b",
        ),
    ),

    # --- reminders ----------------------------------------------------------
    IntentRule(
        intent="cancel_reminder",
        patterns=_compile(
            r"\bcancel (?:my |the )?(?P<which>.+?) reminder\b",
            r"\bcancel (?:my |the )?reminder(?: (?:about|for|to) (?P<which>.+))?$",
            r"\bdelete (?:my |the )?reminder(?: (?:about|for|to) (?P<which>.+))?$",
        ),
    ),
    IntentRule(
        intent="list_reminders",
        patterns=_compile(
            r"\b(?:show|list|what) (?:are )?(?:my |the )?reminders\b",
            r"\bwhat reminders\b",
            r"\bdo i have any reminders\b",
        ),
    ),
    IntentRule(
        intent="due_reminders",
        patterns=_compile(
            r"\b(?:is )?anything due\b", r"\bany reminders due\b",
        ),
    ),
    IntentRule(
        intent="create_reminder",
        patterns=_compile(
            # "remind me at 6 to submit" and "remind me to submit at 6" both occur.
            r"\bremind me (?P<when>(?:at|on|in|tomorrow|tonight|today|next|this)[^,]*?)"
            r"\s+(?:to|about|that)\s+(?P<title>.+)$",
            r"\bremind me (?:to|about|that)\s+(?P<title>.+?)\s+"
            r"(?P<when>(?:at|on|in|tomorrow|tonight|today|next|this)\s+.+)$",
            r"\bset a reminder (?:for|at)\s+(?P<when>.+?)\s+(?:to|about)\s+(?P<title>.+)$",
            r"\bset a reminder (?:for|at)\s+(?P<when>.+)$",
            r"\bremind me (?P<when>(?:at|on|in|tomorrow|tonight|today|next|this)\s+.+)$",
        ),
        validate=_validate_non_empty("when"),
    ),

    IntentRule(
        intent="missed_reminders",
        patterns=_compile(
            r"\bdid i miss any reminders?\b",
            r"\b(?:show|list) (?:me )?(?:my )?missed reminders\b",
        ),
    ),

    # --- calendar -----------------------------------------------------------
    IntentRule(
        intent="cancel_event",
        patterns=_compile(
            r"\bcancel (?:my |the )?(?:meeting|event|appointment)"
            r"(?: (?:called|named|about|for) (?P<which>.+))?$",
        ),
    ),
    IntentRule(
        intent="list_events",
        patterns=_compile(
            r"\b(?:show|what(?:'?s| is)) (?:my |on my )?(?P<range>today'?s|tomorrow'?s)?"
            r"\s*(?:schedule|calendar|agenda)\b",
            r"\bwhat do i have (?P<range>today|tomorrow|this week)\b",
            r"\bopen (?:my )?calendar\b",
            r"\bwhat(?:'?s| is) (?:on )?(?P<range>today|tomorrow)\b",
        ),
    ),
    IntentRule(
        intent="create_event",
        patterns=_compile(
            r"\b(?:schedule|book|set up) (?:a |an )?(?:meeting|event|appointment|call)"
            r"(?: (?:called|named|about|for) (?P<title>.+?))?"
            r"\s+(?P<when>(?:at|on|in|tomorrow|today|next|this)\s+.+)$",
            r"\bcreate an? event called (?P<title>.+?)\s+"
            r"(?P<when>(?:at|on|in|tomorrow|today|next|this)\s+.+)$",
            r"\b(?:schedule|book) (?:a |an )?(?:meeting|event|appointment)"
            r"\s*(?P<when>.*)$",
        ),
    ),

    # --- weather ------------------------------------------------------------
    # Before the generic information rules: "what's the weather today" must not
    # be answered by the date.
    IntentRule(
        intent="will_it_rain",
        patterns=_compile(
            r"\b(?:will|is) it (?:going to )?rain(?:ing)?"
            r"(?: (?P<when>today|tonight|tomorrow))?"
            r"(?: in (?P<location>[\w ,.'-]+))?$",
            r"\bdo i need an umbrella(?: (?P<when>today|tomorrow))?$",
        ),
    ),
    IntentRule(
        intent="get_forecast",
        patterns=_compile(
            r"\bwhat(?:'?s| is) the (?:weather )?forecast"
            r"(?: (?:for|in) (?P<location>[\w ,.'-]+))?$",
            r"\bweather (?:for|this) (?:the )?(?:week|next few days)"
            r"(?: in (?P<location>[\w ,.'-]+))?$",
        ),
    ),
    IntentRule(
        intent="get_weather",
        patterns=_compile(
            r"\b(?:what|how)(?:'?s| is) (?:the )?weather"
            r"(?: (?:like|today|right now|now|outside|here))?"
            r"(?: in (?P<location>[\w ,.'-]+))?$",
            r"^weather(?: in (?P<location>[\w ,.'-]+))?$",
            r"\bwhat(?:'?s| is) (?:the )?temperature(?: outside)?"
            r"(?: in (?P<location>[\w ,.'-]+))?$",
        ),
    ),

    # --- email --------------------------------------------------------------
    # Ahead of the file rules on purpose: "delete the draft about the report"
    # is about a message, and delete_item would read it as a path.
    IntentRule(
        intent="email_status",
        patterns=_compile(
            r"\bis (?:my )?e-?mail (?:set ?up|connected|configured|working)\b",
            r"\bdo you have (?:my )?e-?mail\b",
        ),
    ),
    IntentRule(
        intent="list_drafts",
        patterns=_compile(
            r"\b(?:show|list) (?:me )?(?:my )?(?:e-?mail )?drafts\b",
            r"\bwhat drafts do i have\b",
        ),
    ),
    IntentRule(
        intent="discard_draft",
        patterns=_compile(
            r"\b(?:discard|delete|throw away|bin) (?:the |that |this |my )?"
            r"(?:e-?mail )?draft(?: (?:about|to|for) (?P<which>.+))?$",
        ),
    ),
    IntentRule(
        intent="send_email",
        patterns=_compile(
            r"\bsend (?:the |this |that |my )?(?:e-?mail|draft)"
            r"(?: to (?P<which>.+))?$",
        ),
    ),
    IntentRule(
        intent="draft_email",
        patterns=_compile(
            r"\b(?:draft|write|compose|start) (?:an? |the )?(?:e-?mail|message)"
            r"(?: to (?P<to>[\w.@+-]+))?(?: (?:about|saying|re) (?P<subject>.+))?$",
            r"\be-?mail (?P<to>[\w.@+-]+@[\w.-]+)(?: about (?P<subject>.+))?$",
        ),
    ),

    # --- workspace and integrations ----------------------------------------
    # Deliberately ahead of capability discovery: its "can you X" pattern
    # matches anything, so these questions never reached their own actions.
    IntentRule(
        intent="list_workspace_roots",
        patterns=_compile(
            r"\bwh(?:at|ich) folders? can you (?:work|use|access)\b",
            r"\bwhat(?:'s| is)? (?:in )?my workspace\b",
            r"\b(?:show|list) (?:me )?(?:my |the )?workspace"
            r"(?: folders| roots)?$",
            r"\bwhere can you work\b",
        ),
    ),
    IntentRule(
        intent="list_integrations",
        patterns=_compile(
            r"\bwhat (?:are you|'re you) connected to\b",
            r"\bwhat(?:'s| is) connected\b",
            r"\b(?:show|list) (?:me )?(?:my |the )?"
            r"(?:integrations|connections)$",
            r"\bis (?:my )?(?:e-?mail|calendar|weather) connected\b",
            r"\bwhich (?:integrations|services) are (?:set up|connected)\b",
        ),
    ),

    # --- capability discovery ----------------------------------------------
    IntentRule(
        intent="list_capabilities",
        patterns=_compile(
            r"\bwhat can you do\b",
            r"\bwhat (?:commands|actions) (?:do you|can you) (?:support|do)\b",
            r"\bwhat are your (?:commands|actions|capabilities)\b",
            r"\bcan you (?P<topic>.+?)\??$",
            r"\bwhat are you (?:able to do|capable of)\b",
            # The intent engine strips polite prefixes, so "can you X" arrives
            # as a bare "X" and is indistinguishable from a command. These
            # forms survive stripping and are unambiguously questions.
            r"\bdo you support (?P<topic>.+?)\??$",
            r"\bare you able to (?P<topic>.+?)\??$",
            r"\bwhat (?:else )?can you help (?:me )?with\b",
            r"\b(?:list|show) (?:your |the )?(?:capabilities|commands|actions)\b",
        ),
    ),

    IntentRule(
        intent="end_voice_session",
        # "Go to sleep" addressed to the assistant, never to the machine.
        # The power rule owns the phrasings that name the computer, and this
        # one deliberately does not match them.
        patterns=_compile(
            r"^(?:go to sleep|goodbye for now|that'?s all for now)$",
            r"^stop listening(?: to me)?$",
            r"\b(?:end|stop|close) the (?:voice )?session\b",
            r"^(?:you can )?(?:go to sleep|rest) now$",
        ),
    ),

    # --- assistant configuration -------------------------------------------
    IntentRule(
        intent="set_voice_presentation",
        patterns=_compile(
            r"\b(?:use|switch to|change to|give me|i want)\s+"
            r"(?:a |the |your )?(?P<presentation>male|female)\s+"
            r"(?:assistant )?voice\b",
            r"\bchange your voice to (?:a |the )?(?P<presentation>male|female)\b",
            r"\b(?:speak|talk) (?:in|with) a (?P<presentation>male|female) voice\b",
            r"\b(?:sound|be) (?:more )?(?P<presentation>male|female)\b",
        ),
    ),

    IntentRule(
        intent="set_assistant_name",
        # Every pattern requires the sentence to be *addressed to the
        # assistant* -- "your name", "call you", "yourself".  That is what
        # separates an instruction from a remark: "my friend's name is Friday"
        # and "I met someone called Friday" match none of these, because
        # neither is about what to call the assistant.
        patterns=_compile(
            r"\bchange your name to (?P<name>.+)$",
            r"\bcall yourself (?P<name>.+)$",
            r"\byour (?:new )?name (?:is|will be) (?:now )?(?P<name>.+)$",
            r"\brename yourself (?:to )?(?P<name>.+)$",
            # "from now on I'll call you Friday" and its many spoken variants.
            r"\b(?:i'?ll|i will|i'?m going to|im going to|i want to|"
            r"i'?d like to|let'?s|we'?ll) call you (?!on\b)(?P<name>.+)$",
            r"\bfrom now on,?\s+(?:i'?ll|i will|i'?m going to|im going to)?\s*"
            r"call you (?!on\b)(?P<name>.+)$",
            r"\bfrom now on,?\s+your (?:new )?name is (?P<name>.+)$",
            r"\bi'?m going to start calling you (?P<name>.+)$",
            r"\byou'?re (?:called|named) (?P<name>.+)$",
            # "From now on, you're Friday."  The "from now on" is required:
            # a bare "you're X" would rename the assistant to "right".
            r"\bfrom now on,?\s+you'?re (?!called\b|named\b)(?P<name>.+)$",
            # Asking is still instructing, when it is about what to call it.
            r"\bcan i call you (?!on\b)(?P<name>.+)$",
            r"\bmay i call you (?!on\b)(?P<name>.+)$",
            r"\bwould you (?:mind )?(?:if i call(?:ed)? you )(?!on\b)(?P<name>.+)$",
        ),
        validate=_validate_assistant_name,
    ),
    IntentRule(
        intent="set_assistant_profile",
        patterns=_compile(
            r"\bswitch to (?:the )?(?P<profile>\w+)(?: profile| voice)\b",
            r"\buse the (?P<profile>\w+) profile\b",
        ),
        validate=_validate_non_empty("profile"),
    ),
    IntentRule(
        intent="set_wake_word",
        patterns=_compile(r"\bset (?:the |your )?wake ?word to (?P<phrase>.+)$"),
        validate=_validate_non_empty("phrase"),
    ),
    IntentRule(
        intent="list_languages",
        patterns=_compile(
            r"\bwhat languages\b", r"\blist (?:the )?languages\b",
            r"\bwhich languages\b",
        ),
    ),
    IntentRule(
        intent="set_language",
        patterns=_compile(
            r"\b(?:change|set|switch) (?:your |the )?(?P<scope>interface|input|output)? ?"
            r"language to (?P<language>[\w\- ]+)$",
            r"\b(?:speak|talk in|reply in|respond in) (?P<language>[\w\- ]+)$",
            r"^use (?P<language>[\w\- ]+)$",
        ),
        validate=_validate_language,
    ),

    # --- information --------------------------------------------------------
    IntentRule(
        intent="get_time",
        patterns=_compile(
            r"\bwhat(?:'s| is)? the time\b", r"\bwhat time is it\b",
            r"\bcurrent time\b", r"\btell me the time\b", r"^time$",
        ),
    ),
    IntentRule(
        intent="get_date",
        patterns=_compile(
            r"\bwhat(?:'s| is)? (?:the |today'?s )?date\b",
            r"\bwhat day is it\b", r"\btoday'?s date\b", r"^date$",
        ),
    ),
    IntentRule(
        intent="get_status",
        patterns=_compile(
            r"^status$", r"\bare you (online|offline)\b", r"\bsystem status\b",
            r"\bwhat(?:'s| is) your status\b",
        ),
    ),
    IntentRule(
        intent="list_applications",
        patterns=_compile(
            r"\bwhat (?:apps|applications) can you (?:open|launch)\b",
            r"\blist (?:the )?(?:approved )?(?:apps|applications)\b",
        ),
    ),
    IntentRule(
        intent="list_allowed_folders",
        patterns=_compile(
            r"\bwh(?:ich|at) folders? can you (?:access|use|touch)\b",
            r"\blist (?:your )?allowed folders\b",
        ),
    ),

    # --- browser control ----------------------------------------------------
    # These come before the generic web rules so "open a new tab" is a tab and
    # not an attempt to browse to a site called "tab".
    IntentRule(
        intent="new_tab",
        patterns=_compile(
            r"\b(?:open|make|create|start|give me) (?:a |an )?new tab\b",
            r"^new tab$",
        ),
    ),
    IntentRule(
        intent="reopen_tab",
        patterns=_compile(
            r"\b(?:reopen|re-open|bring back|restore) (?:the |my )?"
            r"(?:last |previous |closed )*tab\b",
            r"\bundo (?:that )?close(?: tab)?\b",
        ),
    ),
    IntentRule(
        intent="close_tab",
        patterns=_compile(
            r"\bclose (?:this|the|my|current)? ?(?:current )?tab\b",
            r"^close tab$",
        ),
    ),
    IntentRule(
        intent="next_tab",
        patterns=_compile(
            r"^(?:next tab|switch tabs?)$",
            r"\b(?:go|switch) to the next tab\b",
        ),
    ),
    IntentRule(
        intent="refresh_page",
        patterns=_compile(
            r"\b(?:refresh|reload) (?:this |the |current )*(?:page|tab)\b",
            r"^(?:refresh|reload)$",
        ),
    ),
    IntentRule(
        intent="go_back",
        patterns=_compile(r"^(?:go back|back)$",
                          r"\bgo back (?:a |one )?page\b"),
    ),
    IntentRule(
        intent="go_forward",
        patterns=_compile(r"^(?:go forward|forward)$",
                          r"\bgo forward (?:a |one )?page\b"),
    ),
    IntentRule(
        intent="browser_status",
        patterns=_compile(
            r"\bis (?:a |the )?browser (?:open|in front|focused|active)\b",
            r"\bwhich browser is (?:open|in front)\b",
        ),
    ),

    # --- web ----------------------------------------------------------------
    #
    # Site searches come first. The web rule's "^search (?:for )?(.+)$" is
    # greedy by design, so it would otherwise claim "search YouTube for Python
    # tutorials" and make the site part of the query.
    IntentRule(
        intent="search_site",
        patterns=_compile(
            r"^search (?:on |in )?(?P<site>.+?) for (?P<query>.+)$",
            r"^search (?:for )?(?P<query>.+?) (?:on|in|at) (?P<site>[\w .]+)$",
            r"^(?:find|look for) (?P<query>.+?) (?:on|in|at) (?P<site>[\w .]+)$",
        ),
        validate=_validate_site_search,
    ),
    IntentRule(
        intent="search_web",
        # Anchored at the start: the search verb has to be the whole point of
        # the sentence. Matching "google" mid-sentence turned "run google
        # chrome" into a web search for "chrome".
        patterns=_compile(
            r"^search (?:the )?(?:web|internet|online) (?:for )?(?P<query>.+)$",
            r"^search (?:for )?(?P<query>.+)$",
            r"^google (?P<query>.+)$",
            r"^look up (?P<query>.+)$",
            r"^find (?:information|info|details) (?:about|on) (?P<query>.+)$",
        ),
        validate=_validate_non_empty("query"),
    ),
    IntentRule(
        intent="open_url",
        patterns=_compile(
            r"\b(?:open|go to|visit|navigate to|browse to|browse) (?P<url>\S+)$",
        ),
        validate=_validate_url,
    ),

    # --- keyboard and mouse -------------------------------------------------
    IntentRule(
        intent="press_key",
        patterns=_compile(
            r"\b(?:press|hit|tap|send) (?:the |a )?(?P<keys>[\w+ -]+?)(?: key)?$",
        ),
        validate=_validate_keys,
    ),
    IntentRule(
        intent="type_text",
        patterns=_compile(
            r"^type (?:out |in )?(?P<text>.+)$",
            r"^enter the text (?P<text>.+)$",
        ),
        validate=_validate_non_empty("text"),
    ),
    IntentRule(
        intent="double_click_mouse",
        patterns=_compile(r"^double.?click(?: (?:at|on) (?P<at>.+))?$"),
    ),
    IntentRule(
        intent="right_click_mouse",
        patterns=_compile(r"^right.?click(?: (?:at|on) (?P<at>.+))?$"),
    ),
    IntentRule(
        intent="click_mouse",
        patterns=_compile(r"^(?:left )?click(?: (?:at|on) (?P<at>.+))?$"),
    ),
    IntentRule(
        intent="move_mouse",
        patterns=_compile(
            r"\bmove (?:the )?(?:mouse|cursor|pointer) (?:to )?(?P<to>.+)$"),
        validate=_validate_non_empty("to"),
    ),
    IntentRule(
        intent="scroll_mouse",
        patterns=_compile(
            r"^scroll (?P<direction>up|down)$",
            r"^scroll$",
        ),
    ),

    # --- files --------------------------------------------------------------
    IntentRule(
        intent="create_folder",
        patterns=_compile(
            r"\b(?:create|make|add|new) (?:a |an )?(?:new )?(?:folder|directory) "
            r"(?:called |named )?(?P<name>.+?)(?: (?:in|inside|under|on) (?:my |the )?"
            r"(?P<location>[\w ]+))?$",
        ),
        validate=_validate_non_empty("name"),
    ),
    IntentRule(
        intent="list_files",
        patterns=_compile(
            r"\b(?:list|show me|show) (?:the )?(?:files|contents|items) "
            r"(?:in|of|inside|on) (?:my |the )?(?P<location>[\w ]+)$",
            r"\bwhat(?:'s| is) (?:in|on) (?:my |the )?(?P<location>[\w ]+)$",
        ),
        validate=_validate_non_empty("location"),
    ),
    IntentRule(
        intent="rename_item",
        patterns=_compile(r"\brename (?P<path>.+?) to (?P<new_name>.+)$"),
        validate=_validate_non_empty("path", "new_name"),
    ),
    IntentRule(
        intent="move_item",
        patterns=_compile(r"\bmove (?P<path>.+?) (?:to|into) (?:my |the )?(?P<destination>.+)$"),
        validate=_validate_non_empty("path", "destination"),
    ),
    IntentRule(
        intent="delete_item",
        patterns=_compile(
            r"\b(?:delete|remove|trash) (?:the )?(?:file |folder )?(?P<path>.+)$",
        ),
        validate=_validate_non_empty("path"),
    ),
    IntentRule(
        intent="open_file",
        patterns=_compile(
            r"\bopen (?:the )?file (?:called |named )?(?P<name>.+?)"
            r"(?: (?:in|from|inside) (?:my |the )?(?P<location>[\w ]+))?$",
            r"\bopen (?:my |the )?(?P<name>[\w .\-()]+\."
            r"(?:pdf|docx?|txt|md|rtf|xlsx?|csv|pptx?|png|jpe?g|gif|mp[34]))"
            r"(?: (?:in|from) (?:my |the )?(?P<location>[\w ]+))?$",
        ),
        validate=_validate_non_empty("name"),
    ),

    # --- terminals ----------------------------------------------------------
    # Ahead of open_application, which refuses shells on purpose. Opening a
    # prompt is its own action with its own permission; nothing here runs a
    # command, and no rule below turns spoken text into one.
    IntentRule(
        intent="open_terminal",
        patterns=_compile(
            r"\b(?:open|launch|start|run|fire up|bring up) (?:the |a |my )?"
            r"(?P<name>.+?)(?: window)?$",
        ),
        validate=_validate_terminal,
    ),

    # --- windows --------------------------------------------------------
    # After the assistant-configuration rules, so "switch to the calm profile"
    # is still a profile change rather than a hunt for a window called "calm".
    IntentRule(
        intent="previous_window",
        patterns=_compile(
            r"\b(?:switch|go|flip) (?:to |back )?(?:the )?(?:previous|last) window\b",
            r"^(?:previous|last) window$",
        ),
    ),
    IntentRule(
        intent="list_windows",
        patterns=_compile(
            r"\bwhat windows are open\b",
            r"\blist (?:my |the )?(?:open )?windows\b",
        ),
    ),
    IntentRule(
        intent="active_window",
        patterns=_compile(
            r"\bwh(?:at|ich) window is (?:in front|active|focused|open)\b",
            r"\bwhat(?:'?s| is) in front\b",
        ),
    ),
    IntentRule(
        intent="minimize_window",
        patterns=_compile(
            r"^minimi[sz]e (?:this|the current|current)(?: window)?$",
            r"\bminimi[sz]e (?:the |my )?(?P<target>.+?)(?:\s+window)?$",
        ),
    ),
    IntentRule(
        intent="maximize_window",
        patterns=_compile(
            r"^maximi[sz]e (?:this|the current|current)(?: window)?$",
            r"\bmaximi[sz]e (?:the |my )?(?P<target>.+?)(?:\s+window)?$",
        ),
    ),
    IntentRule(
        intent="restore_window",
        patterns=_compile(
            r"^restore (?:this|the current|current)(?: window)?$",
            r"\brestore (?:the |my )?(?P<target>.+?)(?:\s+window)?$",
        ),
    ),
    IntentRule(
        intent="close_window",
        # The word "window" is required. Without it, "close Notepad" would stop
        # reaching close_application, which shuts the whole program down rather
        # than one of its windows -- a different instruction.
        patterns=_compile(
            r"^close (?:the |this |that |current )?window$",
            r"\bclose (?:the |my )?(?P<target>.+?)\s+window$",
        ),
    ),
    IntentRule(
        intent="switch_window",
        patterns=_compile(
            r"\b(?:switch|go|flip) to (?:the |my )?(?P<target>.+?)(?:\s+window)?$",
            r"\bbring (?:up |me )?(?P<target>.+?) to the front$",
        ),
        validate=_validate_non_empty("target"),
    ),

    # --- desktop ------------------------------------------------------------
    IntentRule(
        intent="capture_window",
        patterns=_compile(
            r"\b(?:capture|screen ?shot|photograph) (?:this|the (?:active|current|front))"
            r"(?: window)?$",
            r"\btake a (?:screen ?shot|picture) of (?:this|the current) window\b",
        ),
    ),
    IntentRule(
        intent="take_screenshot",
        patterns=_compile(
            r"\b(?:take|capture|grab|get) (?:a |the )?screen ?shot\b",
            r"\bscreen ?shot (?:my |the )?screen\b", r"^screen ?shot$",
        ),
    ),
    IntentRule(
        intent="close_application",
        patterns=_compile(
            r"\b(?:close|quit|kill|shut) (?:down )?(?:the |my )?(?P<name>.+?)"
            r"(?: (?:app|application|window))?$",
        ),
        validate=_validate_application,
    ),
    IntentRule(
        intent="open_application",
        patterns=_compile(
            r"\b(?:open|launch|start|run|fire up) (?:the |my )?(?P<name>.+?)"
            r"(?: (?:app|application|for me|please))?$",
            r"\bi want to use (?:the |my )?(?P<name>.+)$",
        ),
        validate=_validate_application,
    ),

    # Last of all. "Open Photos" should open the Photos app rather than the
    # Pictures folder, so the application rule gets first refusal and this
    # catches what it declined.
    IntentRule(
        intent="open_folder",
        patterns=_compile(
            r"\bopen (?:my |the )?(?P<location>[\w ]+?)(?: folder)?$",
            r"\bshow (?:me )?(?:my |the )?(?P<location>[\w ]+?)(?: folder)?$",
            r"\b(?:go|take me) to (?:my |the )?(?P<location>[\w ]+?) folder$",
        ),
        validate=_validate_folder,
    ),
)


#: Filler that adds nothing to the meaning and confuses anchored patterns.
_POLITE_PREFIXES = _compile(
    r"^(?:could you|can you|would you|will you|please|kindly|i want you to|"
    r"i'd like you to|i would like you to)\s+",
)
_POLITE_SUFFIXES = _compile(r"\s+(?:please|for me|thanks|thank you)$")


class IntentEngine:
    """Deterministic rules first, nearest-intent resolution second.

    The rules below have not changed and remain authoritative: an exact
    phrasing still matches in microseconds and never reaches the resolver.
    What Phase 4 added is what happens *after* they all decline -- instead of
    "I didn't understand that", the sentence is scored against the action
    registry and either resolved, questioned, or honestly refused.

    The ordering is the safety property. A sentence that matches a rule cannot
    be re-interpreted by the resolver, so nearest-intent understanding can
    only ever turn an ``unknown`` into something, never turn one known command
    into a different one.
    """

    def __init__(self, config: ConfigManager,
                 rules: Optional[Sequence[IntentRule]] = None,
                 *, registry=None, context_provider=None) -> None:
        self._config = config
        self._rules: Sequence[IntentRule] = rules if rules is not None else INTENT_RULES
        self._registry = registry
        self._resolver = None
        if registry is not None:
            from ai.nlu import NaturalLanguageResolver

            self._resolver = NaturalLanguageResolver(
                registry, context_provider=context_provider)

    # ------------------------------------------------------------ normalising

    #: Words that can precede the assistant's name when addressing it. Each
    #: one is optional and only ever stripped as part of an address, so
    #: "okay, open Chrome" keeps its "okay" -- there is no name after it.
    _ADDRESS_PREFIXES = ("hey", "okay", "ok", "hi", "hello", "yo")

    def strip_wake_word(self, text: str) -> str:
        """Remove a leading address -- "Aariz,", "Hey Aariz", "Okay Friday".

        Only at the start, and only the current identity's name, so the name
        keeps its ordinary meaning everywhere else: "open the Aariz folder" and
        "search for Aariz on YouTube" are left exactly as spoken. That
        restraint is what stops the assistant's own name leaking into a search
        query as though the user had asked for it.

        The name is read from configuration on every call rather than baked in,
        so a rename takes effect on the next sentence with nothing to reload.
        """
        assistant = self._config.config.assistant
        name = assistant.name.strip().lower()
        names = {assistant.wake_word.strip().lower(), name}
        candidates = {n for n in names if n}
        # Each name, optionally preceded by a greeting. Built from the current
        # name rather than listed, so a rename cannot leave a stale spelling
        # behind and no identity is addressable that is not the current one.
        for prefix in self._ADDRESS_PREFIXES:
            candidates |= {f"{prefix} {n}" for n in names if n}

        stripped = text.strip()
        lowered = stripped.lower()
        for candidate in sorted(candidates, key=len, reverse=True):
            if not lowered.startswith(candidate):
                continue
            # The name has to end here, or "Aariz" would match the start of
            # "Aarizabad" and eat half a word.
            tail = stripped[len(candidate):]
            if tail and tail[0].isalnum():
                continue
            remainder = tail.lstrip(" ,.:;!-").strip()
            # "Aariz." on its own is an address with no command. Returning the
            # original keeps it a recognisable utterance rather than an empty
            # one the parser would have to invent a meaning for.
            return remainder or stripped
        return stripped

    def normalise(self, text: str) -> str:
        """Lower-noise form used for matching: wake word, filler and punctuation removed."""
        working = self.strip_wake_word(text)
        working = re.sub(r"\s+", " ", working).strip()
        for pattern in _POLITE_PREFIXES:
            working = pattern.sub("", working)
        for pattern in _POLITE_SUFFIXES:
            working = pattern.sub("", working)
        return working.strip().rstrip("?.!,")

    # ---------------------------------------------------------------- parsing

    def parse(self, text: str) -> Intent:
        """Interpret ``text``.  Always returns an :class:`Intent`, never raises."""
        raw = (text or "").strip()
        if not raw:
            return Intent(name="unknown", raw_text=text or "", normalised_text="")

        # The sentence with the address removed and nothing else -- what the
        # user asked for, minus who they asked. `normalised` goes further and
        # strips polite filler too, which the rules want and the resolver does
        # not.
        addressed = self.strip_wake_word(raw)
        normalised = self.normalise(raw)

        # Before any matching. A negated or reported sentence is not an
        # instruction however well it matches a pattern, and the action rules
        # are unanchored by design -- "open chrome" is found inside "I don't
        # want you to open Chrome" -- so a check that ran only on the
        # resolver's path would never see the sentences that need it most.
        if self._resolver is not None:
            from ai.nlu import declines

            refusal = declines(raw)
            if refusal:
                logger.info("Sentence declined before matching",
                            extra={"context": {"reason": refusal}})
                return Intent(name="declined", parameters={"reason": refusal},
                              match_method="declined", raw_text=raw,
                              normalised_text=normalised)

        for rule in self._rules:
            parameters = rule.match(normalised)
            if parameters is None:
                continue
            intent = Intent(
                name=rule.intent,
                parameters=parameters,
                confidence=rule.confidence,
                match_method="pattern",
                raw_text=raw,
                normalised_text=normalised,
            )
            logger.debug("Intent matched",
                         extra={"context": {"intent": intent.name,
                                            "confidence": intent.confidence}})
            return intent

        # No rule matched. Ask the resolver for the nearest registered
        # capability before giving up -- this is the whole of Phase 4, and it
        # is reached only here, on sentences the rules already declined.
        #
        # Addressed, not normalised: the address is removed because it is not
        # part of the command, and nothing else is, because Phase 4 gives the
        # resolver the user's own words on purpose. Handing it `raw` put the
        # assistant's name inside the query it searched for.
        if self._resolver is not None:
            resolved = self._resolver.resolve(addressed)
            if resolved.requires_clarification:
                return Intent(
                    name="clarify",
                    parameters={"question": resolved.clarification,
                                "options": list(resolved.options),
                                "nearest": resolved.action},
                    confidence=resolved.confidence, match_method="nearest",
                    raw_text=raw, normalised_text=normalised)
            if resolved.refused:
                return Intent(
                    name="declined",
                    parameters={"reason": resolved.refused},
                    match_method="nearest", raw_text=raw,
                    normalised_text=normalised)
            if resolved.resolved:
                logger.info("Resolved by nearest intent", extra={"context": {
                    "action": resolved.action,
                    "confidence": round(resolved.confidence, 3)}})
                return Intent(
                    name=resolved.action, parameters=dict(resolved.parameters),
                    confidence=resolved.confidence, match_method="nearest",
                    raw_text=raw, normalised_text=normalised)

        logger.info("No intent matched", extra={"context": {"normalised": normalised}})
        return Intent(name="unknown", raw_text=raw, normalised_text=normalised)

    def _fragment_resolver(self):
        """Resolve one fragment the way a whole sentence is resolved.

        :func:`ai.nlu.plans.build` originally asked the nearest-intent resolver
        directly. That made a fragment which only a *deterministic rule* can
        match -- "mute", "help", "next song" -- unresolvable, so any sentence
        containing one refused to split and was run as a single instruction
        with the other half silently discarded.
        """
        outer = self

        class _RuleThenResolver:
            @staticmethod
            def resolve(fragment: str, context=None):
                intent = outer.parse(fragment)
                if intent.is_action and intent.name != "unknown":
                    return _FragmentMatch(
                        resolved=True, action=intent.name,
                        parameters=dict(intent.parameters),
                        confidence=intent.confidence)
                if outer._resolver is None:
                    return _FragmentMatch(refused="no capability")
                return outer._resolver.resolve(fragment, context)

        return _RuleThenResolver()

    #: A connective inside a captured parameter means the rule swallowed the
    #: join, so the sentence was one instruction that merely contains "and".
    _CONSUMED_CONNECTIVE = re.compile(r"\s(?:and|then)\s", re.IGNORECASE)

    def parse_plan(self, text: str, context=None):
        """Read one utterance as several instructions, or ``None`` for one.

        The parse is consulted first, but only to find out *how much of the
        sentence a rule claimed*. The rules are unanchored, so "take a
        screenshot and open Chrome" matches the screenshot rule on its own and
        would otherwise drop the second half without ever saying so. The
        opposite case is "remind me to call mom and email dad": there the rule
        captured "call mom and email dad" as the reminder text, connective
        included, which is the signal that the sentence really is one request.

        Splitting stays conservative beyond that -- see :mod:`ai.nlu.plans`,
        which returns ``None`` unless every fragment resolves to a distinct
        registered action.
        """
        if self._resolver is None:
            return None

        from ai.nlu import plans

        if len(plans.split_sentence(text)) < 2:
            return None

        intent = self.parse(text)
        if intent.match_method == "pattern" and any(
                self._CONSUMED_CONNECTIVE.search(value)
                for value in intent.parameters.values()
                if isinstance(value, str)):
            return None

        plan = plans.build(text, self._fragment_resolver(), context)
        return None if plan is None else plans.carry_context(plan, self._registry)

    def known_intents(self) -> List[str]:
        return sorted({rule.intent for rule in self._rules})
