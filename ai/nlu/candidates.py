"""Scoring registered actions against a sentence.

Candidates come *from the action registry*.  That is the single most important
property in this package: the resolver physically cannot propose a capability
AARMA does not have, because the list it scores is the list of things it can
do.  "Fly a drone" produces no candidates rather than a near miss.

Scoring is additive and explainable.  Every point has a named reason attached,
so a wrong answer can be read back rather than guessed at:

    site named on a site that supports search   +0.35
    a search verb                               +0.30
    a query survived extraction                 +0.20
    ...

That matters more than raw accuracy.  A black-box similarity score that is
wrong gives nobody anything to fix; a list of reasons points straight at the
rule that misfired.

Confidence is the score squashed into 0..1 and is *calibrated against the
corpus*, not asserted -- see the evaluation report. The thresholds in
:mod:`ai.nlu.resolver` were chosen from measured distributions, not picked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ai.nlu.context import ResolutionContext
from ai.nlu.entities import Entities
import re

from ai.nlu.lexicon import (
    SELF_EVIDENT_AUDIO,
    SITE_SEARCH,
    VERBS,
    VOLUME_PATTERNS,
)
from ai.nlu.normalize import close_enough

#: Score above which a candidate is treated as certain.  Reached only by
#: several independent signals agreeing, never by one.
FULL_SCORE = 1.0


@dataclass
class Candidate:
    """One registered action, scored, with its reasons."""

    action: str
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    #: Set when the sentence names the action but leaves a choice open.
    clarification: str = ""
    options: Tuple[str, ...] = ()

    def add(self, points: float, reason: str) -> "Candidate":
        self.score += points
        self.reasons.append(f"{reason} (+{points:.2f})")
        return self

    def penalise(self, points: float, reason: str) -> "Candidate":
        self.score -= points
        self.reasons.append(f"{reason} (-{points:.2f})")
        return self

    @property
    def confidence(self) -> float:
        """Score mapped into 0..1, saturating rather than clipping hard."""
        if self.score <= 0:
            return 0.0
        return min(0.99, round(self.score, 4))

    def explain(self) -> str:
        return f"{self.action}: " + ", ".join(self.reasons)


def _has_verb(text: str, sense: str) -> Tuple[bool, str]:
    """Whether ``text`` uses a verb from ``sense``, and which one."""
    for phrase in sorted(VERBS.get(sense, ()), key=len, reverse=True):
        if f" {phrase} " in f" {text} " or text.startswith(f"{phrase} "):
            return True, phrase
    return False, ""


def _fuzzy_verb(text: str, sense: str, *,
                corroborated: bool = False) -> Tuple[bool, str]:
    """A verb reached only through spelling repair, reported as such.

    Kept separate from :func:`_has_verb` so the resolver can refuse to let a
    guessed word reach a destructive action.  ``corroborated`` is passed when
    the sentence's *target* was matched exactly, which is what makes a
    three-letter repair like "opn" acceptable.
    """
    words = text.split()
    for phrase in VERBS.get(sense, ()):
        if " " in phrase:
            continue
        for word in words:
            if word != phrase and close_enough(word, phrase,
                                               corroborated=corroborated):
                return True, word
    return False, ""


#: "volum down", "sond up" -- the direction word is exact and only the noun is
#: mangled, so the sentence still says which way and about what.
_DIRECTION = {"volume_down": ("down", "lower", "quieter"),
              "volume_up": ("up", "higher", "louder")}


def _fuzzy_audio_noun(text: str, sense: str) -> str:
    """A mangled 'volume'/'sound' next to an exact direction word."""
    words = text.split()
    if not any(direction in words for direction in _DIRECTION.get(sense, ())):
        return ""
    for word in words:
        for noun in ("volume", "sound", "audio"):
            if word != noun and close_enough(word, noun, corroborated=True):
                return word
    return ""


def _registered(registry, name: str) -> bool:
    return registry is not None and name in registry


#: Words that point at something already established rather than naming it.
#: Kept short: every addition is a way for a sentence to acquire a target its
#: author never mentioned.
_POINTERS = ("it", "that one", "there", "this project", "the project",
             "that project", "the repo", "that repo")

#: "open **it in** vs code" -- a pointer followed by a container. What follows
#: "in" is where the thing goes, not the thing itself.
_POINTS_INTO = re.compile(
    r"\b(?:it|that|this|the repo(?:sitory)?|the project|that project)\s+in(?:to)?\b",
    re.IGNORECASE)


def generate(text: str, entities: Entities, *, registry,
             context: Optional[ResolutionContext] = None) -> List[Candidate]:
    """Every plausible registered action for this sentence, best first.

    ``registry`` is required rather than optional: without it there is no
    authoritative list of capabilities, and a resolver that will guess without
    one is a resolver that can invent.
    """
    context = context or ResolutionContext()
    found: List[Candidate] = []

    def offer(action: str) -> Optional[Candidate]:
        if not _registered(registry, action):
            return None
        candidate = Candidate(action=action)
        found.append(candidate)
        return candidate

    # ---------------------------------------------------------- web search
    search_verb, verb_word = _has_verb(text, "search")
    if entities.site and entities.site in SITE_SEARCH and entities.query:
        candidate = offer("search_site")
        if candidate is not None:
            candidate.add(0.40, f"'{entities.site}' supports a direct search")
            candidate.add(0.30, "a search query was extracted")
            if search_verb:
                candidate.add(0.25, f"search verb '{verb_word}'")
            else:
                # "Open python tutorials on youtube" has no search verb but is
                # unmistakably a search: a site plus a subject is the pattern.
                candidate.add(0.15, "site plus subject reads as a search")
            candidate.parameters = {"site": entities.site,
                                    "query": entities.query}

    if entities.query and not (entities.site and entities.site in SITE_SEARCH):
        candidate = offer("search_web")
        if candidate is not None:
            if search_verb:
                candidate.add(0.55, f"search verb '{verb_word}'")
            else:
                candidate.add(0.20, "a subject with no destination")
            candidate.add(0.25, "a search query was extracted")
            candidate.parameters = {"query": entities.query}

    # ------------------------------------------------------------ open site
    open_verb, open_word = _has_verb(text, "open")
    if entities.site and not entities.query:
        candidate = offer("open_url")
        if candidate is not None:
            candidate.add(0.45, f"site '{entities.site}' named")
            if open_verb:
                candidate.add(0.40, f"open verb '{open_word}'")
            candidate.parameters = {"url": entities.site}

    # ----------------------------------------------------------- open folder
    if entities.folder:
        candidate = offer("open_folder")
        if candidate is not None:
            candidate.add(0.50, f"folder '{entities.folder}' named")
            if open_verb:
                candidate.add(0.40, f"open verb '{open_word}'")
            candidate.parameters = {"location": entities.folder}

    if not open_verb:
        # Corroborated: an exactly-matched application, site or folder is the
        # certain half of the sentence, so only the verb is being repaired.
        certain_target = bool(entities.application or entities.site
                              or entities.folder)
        fuzzy_open, fuzzy_word = _fuzzy_verb(text, "open",
                                             corroborated=certain_target)
        if fuzzy_open and certain_target:
            # A repaired verb only counts when the *target* was spelled
            # correctly; two guesses in one sentence is not a resolution.
            open_verb, open_word = True, f"{fuzzy_word} (spelling repaired)"

    # ------------------------------------------------------ open application
    #
    # A terminal asked for *somewhere* is handled by the repository branch
    # above; offering a bare terminal as well would put two near-identical
    # scores inside the ambiguity margin and turn a clear instruction into a
    # question.
    refers_elsewhere = (
        any(word in text for word in ("terminal", "console", "shell"))
        and any(word in text for word in ("there", "here", "in this",
                                          "in the project", "in it")))
    # The same reasoning for editors. "Open it in VS Code" names VS Code as the
    # container; the thing being opened is whatever "it" refers to. Offering
    # VS Code as the target as well put two near-identical scores inside the
    # ambiguity margin and turned a clear instruction into a question.
    if _POINTS_INTO.search(text) and entities.ide:
        refers_elsewhere = True
    if (entities.application or entities.application_choices) and not refers_elsewhere:
        terminal_keys = {"cmd", "powershell", "windowsterminal"}
        if entities.application_choices:
            candidate = offer("open_application")
            if candidate is not None:
                candidate.add(0.35, "an application description was used")
                if open_verb:
                    candidate.add(0.25, f"open verb '{open_word}'")
                candidate.clarification = (
                    "Which one should I open: "
                    + ", ".join(entities.application_choices) + "?")
                candidate.options = entities.application_choices
        else:
            target = ("open_terminal" if entities.application in terminal_keys
                      else "open_application")
            candidate = offer(target)
            if candidate is not None:
                candidate.add(0.50, f"application '{entities.application}' named")
                if open_verb:
                    candidate.add(0.40, f"open verb '{open_word}'")
                candidate.parameters = {"name": entities.application}

    # ----------------------------------------------------------------- volume
    #
    # Three ways in, strongest first. The pattern list exists because the
    # common phrasings are discontinuous -- "turn the sound down" puts two
    # words between the verb and its particle -- and a table of contiguous
    # phrases cannot express that however many entries it is given.
    for sense, action in (("volume_down", "volume_down"),
                          ("volume_up", "volume_up")):
        pattern_hit = next(
            (p for p in VOLUME_PATTERNS.get(sense, ())
             if re.search(p, text, re.IGNORECASE)), "")
        present, word = _has_verb(text, sense)
        fuzzy = False
        if not pattern_hit and not present:
            present, word = _fuzzy_verb(text, sense)
            fuzzy = present
        if not pattern_hit and not present:
            mangled = _fuzzy_audio_noun(text, sense)
            if mangled:
                present, word, fuzzy = True, mangled, True
        if not pattern_hit and not present:
            continue

        names_audio = any(term in text for term in
                          ("volume", "sound", "audio", "music"))
        self_evident = word in SELF_EVIDENT_AUDIO
        if fuzzy and word not in SELF_EVIDENT_AUDIO:
            # A repaired audio noun beside an exact direction word is enough
            # to name the audio; the direction is what was never in doubt.
            names_audio = names_audio or bool(_fuzzy_audio_noun(text, sense))
        if not (pattern_hit or names_audio or self_evident):
            # "lower it" could be a window; "lower the volume" could not.
            if word in {"lower", "raise", "drop", "reduce", "increase"}:
                continue
        candidate = offer(action)
        if candidate is None:
            continue
        if pattern_hit:
            candidate.add(0.65, "a volume phrase was matched")
        else:
            candidate.add(0.55, f"volume verb '{word}'"
                                + (" (spelling repaired)" if fuzzy else ""))
        if names_audio:
            candidate.add(0.30, "audio named explicitly")
        elif self_evident:
            # "Quieter" names the audio by itself: nothing else in AARMA is.
            candidate.add(0.30, f"'{word}' can only mean the audio")
        if fuzzy:
            candidate.parameters["_fuzzy"] = True
        if entities.amount:
            candidate.parameters["amount"] = entities.amount

    # -------------------------------------------------------- browser / window
    close_verb, close_word = _has_verb(text, "close")
    if close_verb or "close" in text:
        referent = entities.referent or ""
        kind = referent if referent not in ("", "unspecified") else None
        if kind is None and referent == "unspecified":
            kind = context.referent_kind()   # may be None -> ask
        if kind == "tab":
            candidate = offer("close_tab")
            if candidate is not None:
                candidate.add(0.50, f"close verb '{close_word}'")
                candidate.add(0.35, "a browser tab is the referent")
        elif kind == "window":
            candidate = offer("close_window")
            if candidate is not None:
                candidate.add(0.50, f"close verb '{close_word}'")
                candidate.add(0.35, "a window is the referent")
        elif referent == "unspecified":
            candidate = offer("close_window")
            if candidate is not None:
                candidate.add(0.30, f"close verb '{close_word}'")
                candidate.clarification = (
                    "I can't tell what's in front. Do you mean the browser tab "
                    "or the window?")

    if any(phrase in text for phrase in ("new tab", "another tab", "another one")):
        candidate = offer("new_tab")
        if candidate is not None:
            candidate.add(0.65, "a new tab was named")
            if context.browser:
                candidate.add(0.25, f"{context.browser} is in front")

    for phrase, action, label in (
            ("go back", "go_back", "back navigation"),
            ("go forward", "go_forward", "forward navigation"),
            ("refresh", "refresh_page", "refresh"),
            ("reload", "refresh_page", "reload"),
            ("reopen", "reopen_tab", "reopen a closed tab")):
        if phrase not in text:
            continue
        candidate = offer(action)
        if candidate is not None:
            candidate.add(0.45, label)
            if context.browser:
                candidate.add(0.40, f"{context.browser} is in front")
            elif context.has_window_context:
                candidate.penalise(0.25, "the window in front is not a browser")

    # ------------------------------------------------------- repositories
    #
    # A repository target outranks an application one: "open my AARMA repo"
    # names a project, and the entity layer has already cleared `application`
    # when it found a repository, so these do not compete.
    if entities.repository:
        if entities.ide:
            candidate = offer("open_repository_in_editor")
            if candidate is not None:
                candidate.add(0.55, f"project '{entities.repository}' named")
                candidate.add(0.35, f"editor '{entities.ide}' named")
                candidate.parameters = {"name": entities.repository,
                                        "editor": entities.ide}
        elif any(word in text for word in ("terminal", "cmd", "powershell",
                                           "shell", "console")):
            candidate = offer("open_terminal_in_repository")
            if candidate is not None:
                candidate.add(0.55, f"project '{entities.repository}' named")
                candidate.add(0.35, "a terminal was asked for")
                candidate.parameters = {"name": entities.repository}
        else:
            candidate = offer("open_repository")
            if candidate is not None:
                candidate.add(0.55, f"project '{entities.repository}' named")
                if open_verb:
                    candidate.add(0.35, f"open verb '{open_word}'")
                candidate.parameters = {"name": entities.repository}

    # ------------------------------------------------- referring to a target
    #
    # "Open my AARMA repo." / "Now open it in VS Code." The second sentence
    # names no project; the first one did, and the engine kept it. Only a
    # pronoun triggers this -- a sentence that names nothing *and* points at
    # nothing is a sentence with no target, and the resolver should ask.
    remembered = str((context.last_entities or {}).get("name") or "")
    if (not entities.repository and remembered
            and any(word in text for word in _POINTERS)):
        if entities.ide:
            candidate = offer("open_repository_in_editor")
            if candidate is not None:
                candidate.add(0.50, f"'{remembered}' was the last thing opened")
                candidate.add(0.35, f"editor '{entities.ide}' named")
                candidate.parameters = {"name": remembered,
                                        "editor": entities.ide}

    # "Open a terminal there/here" -- a terminal *somewhere*, where the
    # somewhere came from an earlier step or an earlier turn. When the earlier
    # turn is known the project is filled in here; otherwise the candidate goes
    # out without a name, the plan layer may carry one in, and failing that the
    # handler asks rather than picking a project nobody named.
    if (not entities.repository
            and any(word in text for word in ("terminal", "console", "shell",
                                              "cmd", "powershell"))
            and any(word in text for word in ("there", "here", "in this",
                                              "in the project", "in it"))):
        candidate = offer("open_terminal_in_repository")
        if candidate is not None:
            candidate.add(0.55, "a terminal was asked for")
            candidate.add(0.30, "somewhere was referred to rather than named")
            if remembered:
                candidate.add(0.15, f"'{remembered}' was the last thing opened")
                candidate.parameters = {"name": remembered}

    if any(phrase in text for phrase in
           ("my repositories", "my repos", "my projects", "list projects",
            "list repositories", "what projects", "show repositories",
            "show projects")):
        candidate = offer("list_repositories")
        if candidate is not None:
            candidate.add(0.85, "a list of projects was asked for")

    if any(phrase in text for phrase in
           ("workspace", "folders can you work", "work in")):
        candidate = offer("list_workspace_roots")
        if candidate is not None:
            candidate.add(0.85, "the workspace was asked about")

    # --------------------------------------------------- whole applications
    #
    # "Close Chrome" is a different instruction from "close this tab" and
    # "close this window": it names an application, so it addresses all of
    # that application's windows.
    if close_verb and entities.everything:
        candidate = offer("close_all_applications")
        if candidate is not None:
            candidate.add(0.85, "everything was named")
            # Resolved confidently on purpose. This is HIGH risk, so the
            # confirmation broker shows the user what would close and asks.
            # A clarification here would only be a prompt before that prompt.
            if entities.exceptions:
                candidate.add(0.20, "an exception list was given")
                candidate.parameters = {"except": ", ".join(entities.exceptions)}
    elif close_verb and entities.application and not entities.referent:
        candidate = offer("close_application_windows")
        if candidate is not None:
            candidate.add(0.55, f"application '{entities.application}' named")
            candidate.add(0.25, f"close verb '{close_word}'")
            candidate.parameters = {"name": entities.application}

    # ------------------------------------------------------- window state
    #
    # The deterministic rules already cover the exact phrasings; these exist so
    # the paraphrases reach them too ("shrink this window", "make this window
    # full screen"), and so the referent still decides which window.
    window_states = {
        "minimize_window": ("minimize", "minimise", "shrink", "hide this",
                            "put this away"),
        "maximize_window": ("maximize", "maximise", "full screen",
                            "fullscreen", "make this bigger"),
        "restore_window": ("restore", "un-maximize", "unmaximise",
                           "bring this back"),
    }
    for action, terms in window_states.items():
        hit = next((term for term in terms if term in text), "")
        if not hit:
            continue
        candidate = offer(action)
        if candidate is not None:
            candidate.add(0.55, f"'{hit}' named")
            if "window" in text or entities.referent == "window":
                candidate.add(0.35, "a window is the referent")
            elif context.has_window_context:
                candidate.add(0.25, "something is in front to act on")

    # --------------------------------------------------------------- capture
    capture_verb, capture_word = _has_verb(text, "capture")
    capture_fuzzy = False
    if not capture_verb:
        # "screnshot" is a typo worth forgiving: the action is LOW risk and
        # nothing else in the vocabulary is one edit away from it.
        capture_verb, capture_word = _fuzzy_verb(text, "capture")
        capture_fuzzy = capture_verb
    if capture_verb or "screenshot" in text or "screen shot" in text:
        wants_window = entities.referent == "window"
        candidate = offer("capture_window" if wants_window else "take_screenshot")
        if candidate is not None:
            candidate.add(0.55, f"capture verb '{capture_word or 'screenshot'}'"
                                + (" (spelling repaired)" if capture_fuzzy else ""))
            if "screen" in text or "screenshot" in text:
                candidate.add(0.35, "the screen was named")
            elif capture_fuzzy:
                candidate.add(0.30, "no other command is that close a spelling")
            if capture_fuzzy:
                candidate.parameters["_fuzzy"] = True

    # -------------------------------------------------------------- reminders
    remind_verb, remind_word = _has_verb(text, "remind")
    if remind_verb:
        candidate = offer("create_reminder")
        if candidate is not None:
            candidate.add(0.60, f"reminder verb '{remind_word}'")
            if entities.raw_time:
                candidate.add(0.30, "a time was given")
                candidate.parameters["when"] = entities.raw_time

    # --------------------------------------------------------------- calendar
    schedule_verb, schedule_word = _has_verb(text, "schedule")
    if schedule_verb:
        candidate = offer("create_event")
        if candidate is not None:
            candidate.add(0.60, f"calendar verb '{schedule_word}'")
            if entities.raw_time:
                candidate.add(0.30, "a time was given")
                candidate.parameters["when"] = entities.raw_time

    # ---------------------------------------------------------- system health
    health_terms = {
        "system_health": ("how is my pc", "how is my computer", "how is the pc",
                          "system health", "how is my laptop", "how is it doing",
                          "how my pc is", "pc doing", "computer doing"),
        "cpu_usage": ("cpu",),
        "memory_usage": ("ram", "memory"),
        "disk_usage": ("disk space", "storage", "free space", "disk"),
        "battery_status": ("battery", "charge", "charging"),
    }
    for action, terms in health_terms.items():
        hit = next((term for term in terms if term in text), "")
        if not hit:
            continue
        candidate = offer(action)
        if candidate is not None:
            candidate.add(0.55, f"'{hit}' named")
            check_verb, check_word = _has_verb(text, "check")
            if check_verb:
                candidate.add(0.30, f"enquiry verb '{check_word}'")

    # ----------------------------------------------------------------- power
    # Resolved like anything else. Understanding is not permission: every one
    # of these still meets SYSTEM_CONTROL and a confirmation downstream.
    power_terms = {
        "shutdown_computer": ("shut down", "shutdown", "turn off the computer",
                              "turn the computer off", "power off", "power down"),
        "restart_computer": ("restart", "reboot"),
        "lock_screen": ("lock",),
        "sleep_computer": ("sleep", "hibernate"),
        "sign_out": ("sign out", "sign me out", "log out", "log me out",
                     "log off", "logout"),
    }
    names_machine = any(word in text for word in
                        ("computer", "pc", "laptop", "machine", "system",
                         "windows", "it"))
    for action, terms in power_terms.items():
        hit = next((term for term in terms if term in text), "")
        if not hit:
            continue
        candidate = offer(action)
        if candidate is not None:
            candidate.add(0.65, f"'{hit}' named")
            if names_machine:
                candidate.add(0.25, "the machine was named")
            # Resolving confidently is the point: the sentence is clear, and
            # every gate that makes this safe -- SYSTEM_CONTROL, HIGH risk,
            # confirmation -- is downstream and untouched. Asking "did you
            # mean shut down?" here would only add a prompt before the prompt
            # that actually protects anything.

    # --------------------------------------------------------------- weather
    wants_rain = "rain" in text or "umbrella" in text
    if "weather" in text or wants_rain or "temperature outside" in text:
        action = "will_it_rain" if wants_rain else "get_weather"
        candidate = offer(action)
        if candidate is not None:
            candidate.add(0.65, "weather named")

    # Highest first; ties broken by name so the order is reproducible.
    found.sort(key=lambda item: (-item.score, item.action))
    return [item for item in found if item.score > 0]


__all__ = ["Candidate", "FULL_SCORE", "generate"]
