"""Pulling the arguments out of a sentence.

The rule this file exists to enforce: **the query is not the sentence**.
"Open Python tutorials on YouTube" carries a site and a query, and searching
YouTube for the string "open python tutorials on youtube" is the kind of
failure that looks like it worked.

Extraction is ordered most-specific first, because the specific readings are
the ones that can be wrong quietly.  A site named with "on youtube" is certain;
a site guessed from a bare mention of YouTube anywhere in the sentence is not,
and the second only runs when the first found nothing.

Nothing here decides *which action* to run -- that is
:mod:`ai.nlu.candidates`.  These functions answer "if this were a search, what
would it search for?", and the resolver decides whether it is a search at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ai.nlu.lexicon import (
    APPLICATION_DESCRIPTIONS,
    EVERYTHING_WORDS,
    IDE_NAMES,
    REFERENT_NOUNS,
    REPOSITORY_WORDS,
    SETTLED_DESCRIPTIONS,
    SITE_ALIASES,
    THIS_REFERENCES,
    amount_of,
    canonical_folder,
    canonical_site,
)

#: "... on youtube", "... in github", "... at stack overflow" -- a site named
#: as the place to look.  The strongest site signal there is.
_SITE_TAIL = re.compile(
    r"\b(?:on|in|at|from|via|using|through)\s+(?P<site>[a-z][a-z ]{1,18}?)\s*$",
    re.IGNORECASE)

#: "on youtube search for X", "in github find X" -- site named up front.
_SITE_LEAD = re.compile(
    r"^(?:on|in|at)\s+(?P<site>[a-z][a-z ]{1,18}?)\s+(?=\w)",
    re.IGNORECASE)

#: "search youtube for X" / "search X on youtube"
_SEARCH_SITE_FOR = re.compile(
    r"\b(?:search|look|browse)\s+(?P<site>[a-z][a-z ]{1,18}?)\s+for\s+(?P<query>.+)$",
    re.IGNORECASE)

#: Verbs that introduce the thing being looked for.
_QUERY_LEAD = re.compile(
    r"\b(?:search(?:\s+for)?|look\s+for|look\s+up|find(?:\s+me)?|"
    r"show\s+me|watch|browse\s+for|google)\s+(?P<query>.+)$",
    re.IGNORECASE)

#: Noise that survives verb stripping and must not reach a search box.
_QUERY_NOISE = re.compile(
    r"^(?:a|an|the|some|any|me|us|for|about|on|regarding)\s+", re.IGNORECASE)

_TRAILING_NOISE = re.compile(
    r"\s+(?:for me|please|now|today|quickly|instead)$", re.IGNORECASE)


#: "open my aarma repo", "open the portfolio project" -- a name followed by a
#: repository word, or a repository word followed by a name.
#: Built from REPOSITORY_WORDS rather than spelled out again.  These were
#: two separate lists once and they drifted: "code" was added to the
#: vocabulary and "bring up my AARMA code" still failed, because the pattern
#: carried its own hard-coded alternation that nobody thought to update.
_REPO_WORDS_PATTERN = "|".join(
    re.escape(word) for word in sorted(REPOSITORY_WORDS, key=len, reverse=True))

_REPO_BEFORE = re.compile(
    r"(?P<name>(?:[\w.+#-]+\s+){0,2}[\w.+#-]+)\s+"
    rf"(?:{_REPO_WORDS_PATTERN})\b",
    re.IGNORECASE)

#: Words that are never part of a project's name.  Without this, "open a
#: terminal in my AARMA repo" yields a project called "terminal in my aarma",
#: and "show my repositories" yields one called "my".
_NOT_A_REPO_NAME = frozenset({
    "my", "the", "this", "that", "a", "an", "all", "your", "our", "its",
    "in", "on", "at", "to", "for", "with", "from", "into", "inside",
    "open", "close", "show", "find", "list", "launch", "start", "run",
    "terminal", "cmd", "powershell", "shell", "window", "folder", "file",
    "and", "or", "me", "please", "here", "there", "it",
    "up", "down", "out", "back", "over", "bring", "take", "go", "get",
})
_REPO_AFTER = re.compile(
    rf"\b(?:{_REPO_WORDS_PATTERN})\s+"
    r"(?:called |named )?(?P<name>[\w.+#-]+(?:\s+[\w.+#-]+){0,2})\b",
    re.IGNORECASE)

#: "my AARMA code" -- a possessive, one name, then "code".
#:
#: Handled by its own pattern rather than by adding "code" to the repository
#: vocabulary, because that turned "open vs code" into a request for a project
#: called "vs".  The possessive is what disambiguates: nobody says "open my vs
#: code" meaning the editor.
_REPO_POSSESSIVE_CODE = re.compile(
    r"\bmy\s+(?P<name>[\w.+#-]+)\s+code\b", re.IGNORECASE)

#: "... in vs code", "... with pycharm"
_IN_IDE = re.compile(
    r"\b(?:in|with|using)\s+(?P<ide>vs ?code|visual studio code|pycharm|"
    r"intellij|webstorm|sublime|atom|notepad\+\+|the code editor|my editor)\b",
    re.IGNORECASE)

#: "close everything except chrome and vs code"
_EXCEPT = re.compile(
    r"\b(?:except|apart from|but not|other than|besides|excluding)\s+"
    r"(?P<list>.+)$", re.IGNORECASE)


@dataclass
class Entities:
    """Everything the sentence supplied, and nothing it did not.

    A field left empty means *not stated*.  No field is ever filled with a
    default that the user did not say -- an assumed site or an assumed folder
    is an action performed on something nobody named.
    """

    site: str = ""
    query: str = ""
    folder: str = ""
    application: str = ""
    #: Catalogue keys when a description matched several, e.g. "my editor".
    application_choices: Tuple[str, ...] = ()
    amount: str = ""
    #: Phase 5 targets. A repository is discovered rather than catalogued, so
    #: this holds what the user *called* it, never a path.
    repository: str = ""
    ide: str = ""
    exceptions: Tuple[str, ...] = ()
    everything: bool = False
    referent: str = ""          # "tab" | "window" | "application"
    url: str = ""
    raw_time: str = ""
    title: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Only the fields that were actually found."""
        found = {"site": self.site, "query": self.query, "folder": self.folder,
                 "application": self.application, "amount": self.amount,
                 "referent": self.referent, "url": self.url,
                 "when": self.raw_time, "title": self.title,
                 "repository": self.repository, "ide": self.ide,
                 "exceptions": list(self.exceptions) if self.exceptions else "",
                 "everything": self.everything or ""}
        return {key: value for key, value in found.items() if value}


def _clean_query(text: str) -> str:
    """Strip the leftovers of the verb phrase from a search query."""
    working = " ".join(str(text or "").split())
    working = _TRAILING_NOISE.sub("", working)
    for _ in range(3):
        reduced = _QUERY_NOISE.sub("", working).strip()
        if reduced == working:
            break
        working = reduced
    # "tutorial on youtube" -> "tutorial": the site is an entity, not a term.
    tail = _SITE_TAIL.search(working)
    if tail and canonical_site(tail.group("site")):
        working = working[:tail.start()].strip()
    return working.strip(" .,!?")


def find_site(text: str) -> Tuple[str, str]:
    """The site named, and the text with that mention removed.

    Returns ``("", text)`` when no site was named.  Removing the mention
    matters: it is what stops "youtube" ending up inside the search query.
    """
    working = " ".join(str(text or "").split())

    explicit = _SEARCH_SITE_FOR.search(working)
    if explicit:
        site = canonical_site(explicit.group("site"))
        if site:
            return site, explicit.group("query").strip()

    tail = _SITE_TAIL.search(working)
    if tail:
        site = canonical_site(tail.group("site"))
        if site:
            return site, (working[:tail.start()] + " "
                          + working[tail.end():]).strip()

    lead = _SITE_LEAD.search(working)
    if lead:
        site = canonical_site(lead.group("site"))
        if site:
            return site, working[lead.end():].strip()

    # Weakest signal last: a bare mention anywhere. Longest alias first so
    # "stack overflow" is not shadowed by a shorter entry.
    for alias in sorted(SITE_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", working):
            site = SITE_ALIASES[alias]
            stripped = re.sub(rf"\b{re.escape(alias)}\b", " ", working, count=1)
            return site, " ".join(stripped.split())
    return "", working


def find_query(text: str) -> str:
    """What the user wants looked up, with the verb phrase removed."""
    working = " ".join(str(text or "").split())
    lead = _QUERY_LEAD.search(working)
    if lead:
        return _clean_query(lead.group("query"))
    return ""


def find_folder(text: str) -> str:
    """A named content folder, or "" -- never a default."""
    lowered = " ".join(str(text or "").lower().split())
    for phrase in sorted(
            ("my downloads", "download folder", "my documents", "my desktop"),
            key=len, reverse=True):
        if phrase in lowered:
            return canonical_folder(phrase)
    for word in re.findall(r"[a-z]+", lowered):
        folder = canonical_folder(word)
        if folder:
            return folder
    return ""


def find_application(text: str) -> Tuple[str, Tuple[str, ...]]:
    """An application, or the candidates a vague description could mean.

    Returns ``(resolved_key, choices)``.  A description matching several
    installed applications yields ``("", choices)`` so the resolver can ask
    instead of picking -- "open my editor" with two editors installed is a
    question, and answering it by guessing is how the wrong program opens.
    """
    from actions.applications import resolve_entry, resolve_target

    lowered = " ".join(str(text or "").lower().split())

    # A description first: these are the ambiguous ones.
    for description in sorted(APPLICATION_DESCRIPTIONS, key=len, reverse=True):
        if not re.search(rf"\b{re.escape(description)}\b", lowered):
            continue
        settled = SETTLED_DESCRIPTIONS.get(description)
        if settled:
            # A convention, not a guess: "the command line" means the Command
            # Prompt. Asking would be pedantry rather than caution.
            return settled, ()
        candidates = APPLICATION_DESCRIPTIONS[description]
        installed = []
        for key in candidates:
            entry = resolve_entry(key)
            if entry is not None and resolve_target(entry) is not None:
                installed.append(key)
        if len(installed) == 1:
            return installed[0], ()
        if len(installed) > 1:
            return "", tuple(installed)
        # Described something real that is not installed here: name the first
        # so the action layer can report it honestly rather than "unknown".
        return (candidates[0], ()) if candidates else ("", ())

    # Then a direct name, longest first so "windows terminal" beats "terminal".
    words = lowered.split()
    for size in range(min(3, len(words)), 0, -1):
        for start in range(len(words) - size + 1):
            phrase = " ".join(words[start:start + size])
            entry = resolve_entry(phrase)
            if entry is not None:
                return entry.key, ()
    return "", ()


def find_referent(text: str) -> str:
    """What "this"/"that" is pointing at, when the sentence says.

    ``"tab"``/``"window"``/``"application"`` when a noun names the kind,
    ``"unspecified"`` when a reference word appears with no noun, and ``""``
    when the sentence does not refer to anything contextual at all.
    """
    lowered = " ".join(str(text or "").lower().split())
    for noun in sorted(REFERENT_NOUNS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(noun)}\b", lowered):
            return REFERENT_NOUNS[noun]
    for reference in THIS_REFERENCES:
        if re.search(rf"\b{re.escape(reference)}\b", lowered):
            return "unspecified"
    return ""


def find_repository(text: str) -> str:
    """What the user called their project, or "".

    Returns the *name as spoken*, never a path: resolving it against the disk
    belongs to :mod:`workspace.repositories`, which can also report that two
    projects match. An extractor that returned one path would hide that.
    """
    working = " ".join(str(text or "").split())

    possessive = _REPO_POSSESSIVE_CODE.search(working)
    if possessive:
        name = possessive.group("name").strip()
        if name.lower() not in _NOT_A_REPO_NAME:
            return name

    for pattern in (_REPO_BEFORE, _REPO_AFTER):
        found = pattern.search(working)
        if not found:
            continue
        words = [w for w in found.group("name").split() if w]
        # Drop leading words that cannot be part of a name, from the left, so
        # "a terminal in my AARMA" reduces to "AARMA" rather than being
        # rejected outright.
        while words and words[0].lower() in _NOT_A_REPO_NAME:
            words.pop(0)
        name = " ".join(words).strip(" .,!?")
        if not name:
            continue
        if name.lower() in REPOSITORY_WORDS or name.lower() in _NOT_A_REPO_NAME:
            continue
        return name
    return ""


def find_ide(text: str) -> str:
    found = _IN_IDE.search(" ".join(str(text or "").split()))
    if not found:
        return ""
    return IDE_NAMES.get(" ".join(found.group("ide").lower().split()), "")


def find_exceptions(text: str) -> Tuple[str, ...]:
    """Applications the user wants left alone, from "except X and Y"."""
    found = _EXCEPT.search(" ".join(str(text or "").split()))
    if not found:
        return ()
    raw = found.group("list")
    parts = re.split(r",|\band\b|\bor\b", raw)
    cleaned = []
    for part in parts:
        name = re.sub(r"^(?:my|the|this)\s+", "", part.strip(),
                      flags=re.IGNORECASE).strip(" .,!?")
        if name:
            cleaned.append(name)
    return tuple(cleaned)


def mentions_everything(text: str) -> bool:
    lowered = " ".join(str(text or "").lower().split())
    return any(phrase in lowered for phrase in EVERYTHING_WORDS)


def find_time(text: str) -> str:
    """The time phrase, delegated wholesale to the existing parser.

    Deliberately no new time grammar: :mod:`utils.when` already handles every
    shape AARMA supports, and a second parser would drift from it.
    """
    from utils.when import parse_when

    parsed = parse_when(text)
    return parsed.text if parsed is not None else ""


#: Verbs that introduce a destination rather than a subject.  Stripped when
#: deciding whether a sentence carries a subject at all.
_OPEN_LEAD = re.compile(
    r"^(?:open|launch|start|run|load|pull up|bring up|fire up|go to|"
    r"take me to|navigate to|visit|watch|show me|get me|find me|find|"
    r"search(?:\s+for)?|look(?:\s+for| up)?)\s+",
    re.IGNORECASE)

#: Words left over after a verb and a site that carry no subject of their own.
_EMPTY_SUBJECT = re.compile(
    r"^(?:and|then|a|an|the|some|me|for|on|it|please|up|to)?$", re.IGNORECASE)

#: Tokens that can only be part of a verb phrase.  A "subject" made entirely of
#: these is the remains of the instruction, not the thing being asked for:
#: "go to github" left "go to" behind as a search query once the site was
#: removed, and AARMA searched GitHub for the words "go to".
_VERB_TOKENS = frozenset({
    "go", "to", "open", "take", "me", "bring", "up", "get", "show", "launch",
    "start", "run", "load", "visit", "navigate", "browse", "pull", "fire",
    "and", "then", "please", "the", "a", "an", "my", "it", "there", "here",
    "watch", "find", "search", "for", "look", "at", "on", "in",
})


def find_subject(text: str) -> str:
    """What is left once the verb and connectives are removed.

    This is what makes "open python tutorials on youtube" a search.  There is
    no search verb in it -- "open" is a destination verb -- but a site plus a
    leftover subject is a search in every reading a person would give it.
    """
    working = " ".join(str(text or "").split())
    for _ in range(3):
        reduced = _OPEN_LEAD.sub("", working).strip()
        if reduced == working:
            break
        working = reduced
    working = re.sub(r"^(?:and|then)\s+", "", working).strip()
    working = _TRAILING_NOISE.sub("", working)
    for _ in range(3):
        reduced = _QUERY_NOISE.sub("", working).strip()
        if reduced == working:
            break
        working = reduced
    working = working.strip(" .,!?")
    if _EMPTY_SUBJECT.match(working):
        return ""
    tokens = [w for w in working.lower().split() if w]
    if tokens and all(token in _VERB_TOKENS for token in tokens):
        return ""
    return working


def extract(text: str) -> Entities:
    """Everything extractable from one sentence.

    Site resolution runs before query extraction so the site mention is gone by
    the time the query is read -- the ordering *is* the fix for "search youtube
    for open python tutorials on youtube".
    """
    working = " ".join(str(text or "").split())

    # The explicit "search <site> for <query>" form settles both halves at
    # once; treating its second half as leftover text and re-parsing it was
    # how "search youtube for a python tutorial" came back with a query of
    # "youtube for a python tutorial".
    explicit = _SEARCH_SITE_FOR.search(working)
    if explicit and canonical_site(explicit.group("site")):
        return Entities(
            site=canonical_site(explicit.group("site")),
            query=_clean_query(explicit.group("query")),
            folder=find_folder(working),
            amount=amount_of(working),
            referent=find_referent(working),
            raw_time=find_time(working))

    site, without_site = find_site(working)
    query = find_query(without_site) or find_query(working)
    if query:
        query = _clean_query(query)
    if not query and site:
        # A site was named and something else remains: that remainder is what
        # they want looked up.
        query = find_subject(without_site)

    # "Close everything except Chrome" names Chrome as something to spare, so
    # the target search must not see it. Strip the clause before looking.
    exceptions = find_exceptions(working)
    target_text = _EXCEPT.sub("", working).strip() if exceptions else working

    application, choices = find_application(target_text)
    repository = find_repository(target_text)
    if repository:
        # A project name is not also an application name: "open my aarma repo"
        # must not simultaneously look like a request to launch something.
        application, choices = "", ()
    return Entities(
        site=site,
        query=query,
        folder=find_folder(target_text),
        application=application,
        application_choices=choices,
        amount=amount_of(working),
        repository=repository,
        ide=find_ide(working),
        exceptions=exceptions,
        everything=mentions_everything(working),
        referent=find_referent(working),
        raw_time=find_time(working),
    )


__all__ = ["Entities", "extract", "find_application", "find_exceptions",
           "find_folder", "find_ide", "find_query", "find_referent",
           "find_repository", "find_site", "find_subject", "find_time",
           "mentions_everything"]
