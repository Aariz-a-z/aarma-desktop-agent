"""The words people actually use, grouped by what they mean.

This is data, not logic, and keeping it separate from the matching code is the
point: adding a way of saying "quieter" should be a one-line edit to a table,
not a new regular expression appended to a file of nine hundred.

Two rules govern what may go in here:

**A synonym must be unambiguous in the imperative.**  "Show me" means *open* in
"show me my downloads" and *search* in "show me python tutorials", so it
appears in both groups and the entity layer decides between them from what
follows.  A word that could mean anything is worse than no entry at all.

**Nothing here grants capability.**  These tables map language onto the names
of actions that already exist.  If an action is removed from the registry, the
words that pointed at it simply stop resolving -- which is the correct outcome
and is asserted by a test.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# --------------------------------------------------------------- verb groups
#
# Keyed by a canonical verb sense.  The resolver scores a sentence against
# these rather than against action names, because "launch", "bring up" and
# "fire up" are the same request and should not each need their own rule.

VERBS: Dict[str, FrozenSet[str]] = {
    "open": frozenset({
        "open", "launch", "start", "run", "bring up", "fire up", "load",
        "pull up", "get me", "take me to", "go to", "navigate to", "visit",
        "show me", "boot", "boot up", "spin up",
    }),
    "close": frozenset({
        "close", "shut", "exit", "dismiss", "get rid of", "kill", "quit",
        "hide", "close down", "shut down",
    }),
    "search": frozenset({
        "search", "find", "look for", "look up", "find me", "show me",
        "search for", "google", "look", "browse for", "watch",
    }),
    "volume_down": frozenset({
        "lower", "decrease", "reduce", "turn down", "quieter", "softer",
        "turn it down", "lower the", "drop", "bring down",
        "volume down", "sound down", "mute a bit",
    }),
    "volume_up": frozenset({
        "raise", "increase", "turn up", "louder", "boost", "bring up the",
        "crank", "pump up", "volume up", "sound up",
    }),
    "remind": frozenset({
        "remind", "remind me", "set a reminder", "alert me", "notify me",
        "remember to tell me", "ping me", "nudge me", "wake me",
    }),
    "schedule": frozenset({
        "schedule", "book", "create an event", "put on my calendar",
        "add an event", "set up a meeting", "arrange", "pencil in",
        "put in my calendar", "block out", "on my calendar", "in my calendar",
        "in my diary", "on my diary",
    }),
    "capture": frozenset({
        "screenshot", "screen shot", "capture", "grab", "snap", "take a picture of",
    }),
    "delete": frozenset({
        "delete", "remove", "erase", "trash", "bin", "throw away", "get rid of",
        "wipe",
    }),
    "check": frozenset({
        "check", "how is", "how's", "what's", "tell me", "show", "report",
        "give me", "read", "look at", "how much", "how many", "what is",
        "how long", "am i", "do i have",
    }),
}

#: Words that name the audio on their own. "Make it quieter" needs no noun --
#: nothing else in AARMA is quieter -- whereas "lower it" could be a window.
SELF_EVIDENT_AUDIO: FrozenSet[str] = frozenset({
    "quieter", "louder", "softer", "volume down", "volume up", "sound down",
    "sound up", "crank", "boost", "pump up",
})

#: Discontinuous volume phrasings: "turn the sound down", "turn my volume up".
#: A contiguous-phrase table cannot express these, so they are patterns.
VOLUME_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "volume_down": (
        r"\bturn\s+(?:the\s+|my\s+|it\s+)?(?:volume|sound|audio|music)?\s*down\b",
        r"\b(?:lower|reduce|decrease|drop)\s+(?:the\s+|my\s+)?(?:volume|sound|audio|music)\b",
        r"\bmake\s+(?:it|the\s+(?:volume|sound|audio|music))\s+(?:quieter|lower|softer)\b",
    ),
    "volume_up": (
        r"\bturn\s+(?:the\s+|my\s+|it\s+)?(?:volume|sound|audio|music)?\s*up\b",
        r"\b(?:raise|increase|boost)\s+(?:the\s+|my\s+)?(?:volume|sound|audio|music)\b",
        r"\bmake\s+(?:it|the\s+(?:volume|sound|audio|music))\s+(?:louder|higher)\b",
    ),
}

#: Words that name a project rather than a folder. "Repo" is unambiguous;
#: "project" and "code" are not, so they only count alongside a name.
REPOSITORY_WORDS: FrozenSet[str] = frozenset({
    "repo", "repos", "repository", "repositories", "project", "projects",
    "codebase", "code base", "solution",
})

#: Editors a project can be opened in, spoken -> catalogue/discovery name.
IDE_NAMES: Dict[str, str] = {
    "vs code": "vs code", "vscode": "vs code", "code": "vs code",
    "visual studio code": "vs code",
    "pycharm": "pycharm", "intellij": "intellij", "webstorm": "webstorm",
    "sublime": "sublime text", "atom": "atom", "notepad++": "notepad++",
    "the code editor": "vs code", "my editor": "vs code",
}

#: Phrases that mean "everything that is open".
EVERYTHING_WORDS: FrozenSet[str] = frozenset({
    "everything", "all applications", "all apps", "all open apps",
    "all my apps", "all windows", "all programs", "all open applications",
})

#: Verb senses that must never be reached by fuzzy spelling repair alone.
#: Getting "delete" out of a misheard word is not a risk worth taking.
GUARDED_VERBS: FrozenSet[str] = frozenset({"delete", "close"})


# ------------------------------------------------------------- site aliases
#
# Spoken forms on the left, the key used by actions.browser on the right.
# Includes the spacings a speech recogniser produces: "you tube", "git hub".

SITE_ALIASES: Dict[str, str] = {
    "youtube": "youtube", "you tube": "youtube", "utube": "youtube",
    "yt": "youtube", "you toob": "youtube", "youtub": "youtube",
    "google": "google", "gogle": "google",
    "gmail": "gmail", "g mail": "gmail", "google mail": "gmail",
    "my email": "gmail", "my mail": "gmail", "my inbox": "gmail",
    "github": "github", "git hub": "github", "gitub": "github",
    "gethub": "github", "git": "github",
    "wikipedia": "wikipedia", "wiki": "wikipedia",
    "stack overflow": "stackoverflow", "stackoverflow": "stackoverflow",
    "stackoverflo": "stackoverflow",
    "linkedin": "linkedin", "linked in": "linkedin",
    "chatgpt": "chatgpt", "chat gpt": "chatgpt", "open ai": "chatgpt",
    "openai": "chatgpt",
    "claude": "claude",
}

#: Sites whose own search AARMA can address directly with a URL.  Anything not
#: here falls back to a web search, which is honest: AARMA is not pretending to
#: drive a site it has no search endpoint for.
SITE_SEARCH: Dict[str, str] = {
    "youtube": "https://www.youtube.com/results?search_query={query}",
    "github": "https://github.com/search?q={query}",
    "wikipedia": "https://en.wikipedia.org/w/index.php?search={query}",
    "stackoverflow": "https://stackoverflow.com/search?q={query}",
    "google": "https://www.google.com/search?q={query}",
}


# ------------------------------------------------------ application aliases
#
# Natural descriptions that are not the application's name.  Resolved against
# the real catalogue, so an entry here for something not installed still
# produces an honest "I couldn't find it" rather than a silent failure.

APPLICATION_DESCRIPTIONS: Dict[str, Tuple[str, ...]] = {
    # description -> candidate catalogue keys, most likely first
    "browser": ("chrome", "edge", "firefox"),
    "web browser": ("chrome", "edge", "firefox"),
    "code editor": ("vscode",),
    "editor": ("vscode", "notepad"),
    "text editor": ("notepad", "vscode"),
    "terminal": ("windowsterminal", "cmd", "powershell"),
    "command line": ("cmd", "powershell", "windowsterminal"),
    "command prompt": ("cmd",),
    "shell": ("powershell", "cmd"),
    "calculator": ("calculator",),
    "file manager": ("explorer",),
    "file browser": ("explorer",),
    "music player": ("spotify",),
    "task manager": ("taskmanager",),
}

#: Descriptions that have one settled answer even though several applications
#: could technically serve.  "The command line" means the Command Prompt on
#: Windows; "the terminal" means Windows Terminal.  These are conventions, not
#: guesses, and asking about them is pedantry rather than caution.
#:
#: "My editor" and "my browser" are deliberately NOT here: which editor a
#: person means is a real question, and opening the wrong one wastes their time
#: in a way that opening the conventional terminal does not.
SETTLED_DESCRIPTIONS: Dict[str, str] = {
    "command line": "cmd",
    "command prompt": "cmd",
    "terminal": "windowsterminal",
    "shell": "powershell",
    "file manager": "explorer",
    "file browser": "explorer",
    "code editor": "vscode",
    "music player": "spotify",
    "task manager": "taskmanager",
    "calculator": "calculator",
}

#: ASR spellings for application names, folded before catalogue lookup.
APPLICATION_ASR: Dict[str, str] = {
    "power shell": "powershell", "powershel": "powershell",
    "vs code": "vs code", "vscode": "vs code", "v s code": "vs code",
    "visual studio code": "vs code",
    "note pad": "notepad", "notpad": "notepad",
    "task manger": "task manager", "taskmanager": "task manager",
    "windows terminal": "windows terminal", "win terminal": "windows terminal",
    "comand prompt": "command prompt", "command promt": "command prompt",
    "chrom": "chrome", "crome": "chrome", "google chrome": "chrome",
    "explorar": "explorer", "file explorer": "file explorer",
}


# ---------------------------------------------------------- folder language

FOLDER_WORDS: Dict[str, str] = {
    "downloads": "downloads", "download": "downloads",
    "my downloads": "downloads", "download folder": "downloads",
    "documents": "documents", "docs": "documents", "my documents": "documents",
    "desktop": "desktop", "my desktop": "desktop",
    "pictures": "pictures", "photos": "pictures", "images": "pictures",
    "music": "music", "videos": "videos", "movies": "videos",
}


# ------------------------------------------------------------- amount words

AMOUNT_WORDS: Dict[str, str] = {
    "a little": "small", "a bit": "small", "slightly": "small",
    "a touch": "small", "a tad": "small", "a little bit": "small",
    "a lot": "large", "much": "large", "way": "large", "a load": "large",
    "right": "large", "all the way": "maximum", "completely": "maximum",
    "fully": "maximum",
}


# ------------------------------------------------ context reference phrases
#
# Words that point at whatever is in front rather than naming it.  The
# resolver refuses to act on these without an active-window context, because
# "close this" with nothing in front is not a resolvable instruction.

THIS_REFERENCES: FrozenSet[str] = frozenset({
    "this", "that", "it", "current", "this one", "that one", "the current one",
    "here", "there", "the current", "active",
})

#: Nouns that, combined with a reference word, say which *kind* of thing.
REFERENT_NOUNS: Dict[str, str] = {
    "tab": "tab", "page": "tab", "browser tab": "tab", "web page": "tab",
    "window": "window", "app": "application", "application": "application",
    "program": "application",
}


def canonical_site(text: str) -> str:
    """Fold a spoken site name onto its registry key, or "" if unknown."""
    return SITE_ALIASES.get(" ".join(str(text or "").lower().split()), "")


def canonical_folder(text: str) -> str:
    return FOLDER_WORDS.get(" ".join(str(text or "").lower().split()), "")


def amount_of(text: str) -> str:
    """The size of a requested change, or "" when none was expressed."""
    lowered = " ".join(str(text or "").lower().split())
    for phrase, size in AMOUNT_WORDS.items():
        if phrase in lowered:
            return size
    return ""


__all__ = [
    "AMOUNT_WORDS", "APPLICATION_ASR", "APPLICATION_DESCRIPTIONS",
    "FOLDER_WORDS", "GUARDED_VERBS", "REFERENT_NOUNS", "SELF_EVIDENT_AUDIO",
    "SETTLED_DESCRIPTIONS", "SITE_ALIASES", "SITE_SEARCH", "THIS_REFERENCES",
    "EVERYTHING_WORDS", "IDE_NAMES", "REPOSITORY_WORDS",
    "VERBS", "VOLUME_PATTERNS",
    "amount_of", "canonical_folder", "canonical_site",
]
