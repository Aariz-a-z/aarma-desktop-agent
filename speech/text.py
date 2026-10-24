"""Turning assistant text into something worth listening to -- and safe to say.

This module is the last boundary before audio, and it has two jobs.

**Safety** (specification section 12). Speech leaves the machine through a
speaker, where anyone in the room hears it and no redaction is possible after
the fact. Anything resembling a credential is removed *before* synthesis, and
raw tool payloads -- JSON, code, log lines -- are never spoken. This mirrors
the redaction already applied to logs and the audit trail, so a secret would
have to defeat three independent filters to be disclosed.

**Naturalness** (section 5). Screen text and speech want different things. A
40-line directory listing reads fine and is unbearable to listen to; a Windows
path spoken character by character is noise. The condensing here keeps spoken
replies short and human while the full text stays on screen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

from utils.logger import SENSITIVE_KEY_HINTS

#: Spoken replies stay short. Anything longer is a summary on the screen's terms,
#: not a transcript read aloud.
DEFAULT_MAX_SENTENCES = 3
DEFAULT_MAX_CHARS = 320

_SAFE_REPLACEMENT = "I've left that on screen rather than reading it out."

# --------------------------------------------------------------- secret rules

_HINTS = "|".join(re.escape(h) for h in SENSITIVE_KEY_HINTS)

#: ``password: hunter2``, ``api_key=sk-123`` and friends.
_LABELLED_SECRET = re.compile(rf"(?i)\b({_HINTS})\s*[:=]\s*\S+")

#: The same thing said in prose -- "the password is hunter2". Assistant replies
#: are sentences, not key-value pairs, so the separator is usually a verb.
_SPOKEN_SECRET = re.compile(rf"(?i)\b({_HINTS})\s+(?:is|was|will be)\s+\S+")

#: Credential shapes that are recognisable without a label.
_SECRET_SHAPES: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),          # OpenAI-style keys
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),      # GitHub tokens
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),               # AWS access key ids
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWTs
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}"),   # Slack tokens
    re.compile(r"\b[A-Fa-f0-9]{40,}\b"),              # Long hex digests
)

# ----------------------------------------------------------- payload patterns

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_JSON_BLOB = re.compile(r"[{\[][^{}\[\]]*[:,][^{}\[\]]*[}\]]")
_WINDOWS_PATH = re.compile(r"[A-Za-z]:\\(?:[^\\\s\"']+\\)*([^\\\s\"']*)")
_POSIX_PATH = re.compile(r"(?:/[^/\s\"']+){2,}/?")
_URL = re.compile(r"https?://(?:www\.)?([^\s/]+)\S*")
_BULLET = re.compile(r"^[\s]*[-*•>]\s+", re.MULTILINE)
_MARKDOWN_EMPHASIS = re.compile(r"[*_]{1,3}(?=\S)(.+?)(?<=\S)[*_]{1,3}")
_WHITESPACE = re.compile(r"[ \t]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: Symbols that would otherwise be spoken literally, and what to say instead.
_SPOKEN_SUBSTITUTIONS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\s*[→>]{1,2}\s*"), ", then "),
    (re.compile(r"\s*·\s*"), ", "),
    (re.compile(r"\s*—\s*"), ", "),
    (re.compile(r"\s*--\s*"), ", "),
    (re.compile(r"\be\.g\.\s*"), "for example, "),
    (re.compile(r"\bi\.e\.\s*"), "that is, "),
    (re.compile(r"\betc\.(?=\s|$)"), "and so on"),
    (re.compile(r"\bvs\.?(?=\s)"), "versus"),
    (re.compile(r"\s*\|\s*"), ", "),
    (re.compile(r"[\"“”]"), ""),
    # "item(s)" is one word to a reader and two to a synthesiser; fold the
    # optional plural in before parentheses become pauses.
    (re.compile(r"\((e?s)\)"), r"\1"),
    (re.compile(r"\s*\(\s*"), ", "),
    (re.compile(r"\s*\)\s*"), " "),
)


@dataclass(frozen=True)
class SpeechText:
    """Text cleared for synthesis, with a record of what was changed."""

    text: str
    redacted: bool = False
    condensed: bool = False
    original_length: int = 0

    @property
    def speakable(self) -> bool:
        return bool(self.text.strip())


def contains_sensitive(text: str) -> bool:
    """Whether ``text`` looks like it carries a credential."""
    probe = text or ""
    if _LABELLED_SECRET.search(probe) or _SPOKEN_SECRET.search(probe):
        return True
    return any(pattern.search(probe) for pattern in _SECRET_SHAPES)


def strip_sensitive(text: str) -> Tuple[str, bool]:
    """Remove credential-shaped content. Returns the text and whether it changed.

    Removed rather than masked: speaking "password redacted" still announces
    that a password was in the reply, which is more than the listener needs.
    """
    cleaned = _LABELLED_SECRET.sub("", text or "")
    cleaned = _SPOKEN_SECRET.sub("", cleaned)
    for pattern in _SECRET_SHAPES:
        cleaned = pattern.sub("", cleaned)
    return cleaned, cleaned != (text or "")


def strip_payloads(text: str) -> str:
    """Remove machine-facing content that has no business being read aloud."""
    cleaned = _CODE_FENCE.sub(" ", text or "")
    cleaned = _JSON_BLOB.sub(" ", cleaned)
    cleaned = _INLINE_CODE.sub(r"\1", cleaned)
    # A full path is unlistenable; its last component is the useful part.
    cleaned = _WINDOWS_PATH.sub(lambda m: m.group(1) or "that folder", cleaned)
    cleaned = _POSIX_PATH.sub(" that folder ", cleaned)
    cleaned = _URL.sub(lambda m: m.group(1), cleaned)
    return cleaned


def naturalise(text: str) -> str:
    """Make punctuation and shorthand sound like speech rather than markup."""
    cleaned = _BULLET.sub("", text or "")
    cleaned = _MARKDOWN_EMPHASIS.sub(r"\1", cleaned)
    for pattern, replacement in _SPOKEN_SUBSTITUTIONS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned)
    # Sentence-ending newlines become real sentence breaks, so the synthesiser
    # pauses instead of running two thoughts together.
    cleaned = re.sub(r"\s*\n+\s*", ". ", cleaned)
    cleaned = re.sub(r"\.\s*\.(\s*\.)*", ". ", cleaned)
    cleaned = re.sub(r"\s*,\s*,+", ", ", cleaned)
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    return cleaned.strip(" ,.").strip()


def summarise_lines(text: str) -> Tuple[str, bool]:
    """Reduce multi-line output to its first line.

    A directory listing or the help text puts its summary on line one and the
    detail below; the detail is for the screen. Must run *before*
    :func:`naturalise`, which deliberately flattens newlines into sentence
    breaks and would leave nothing to recognise here.
    """
    original = (text or "").strip()
    lines = [line.strip() for line in original.splitlines() if line.strip()]
    if len(lines) > 2:
        return lines[0], True
    return original, False


def condense(text: str, *, max_sentences: int = DEFAULT_MAX_SENTENCES,
             max_chars: int = DEFAULT_MAX_CHARS) -> Tuple[str, bool]:
    """Cap a reply at a comfortable number of sentences and characters."""
    working = (text or "").strip()
    if not working:
        return "", False
    condensed = False

    sentences = _SENTENCE_SPLIT.split(working)
    if len(sentences) > max_sentences:
        working = " ".join(sentences[:max_sentences])
        condensed = True

    if len(working) > max_chars:
        clipped = working[:max_chars].rsplit(" ", 1)[0]
        working = clipped.rstrip(" ,;:") + "."
        condensed = True

    return working, condensed


def prepare_for_speech(text: str, *, max_sentences: int = DEFAULT_MAX_SENTENCES,
                       max_chars: int = DEFAULT_MAX_CHARS) -> SpeechText:
    """Full pipeline: make it safe, then make it listenable.

    Order matters. Secrets are removed from the *raw* text first, before any
    transformation could reshape a credential into something the filters no
    longer recognise.
    """
    raw = text or ""
    if not raw.strip():
        return SpeechText("", original_length=0)

    _safe, redacted = strip_sensitive(raw)
    if redacted:
        # Removing a credential mid-sentence leaves a fragment ("The for the
        # account.") that is safe but meaningless to hear. Say one clear thing
        # instead and leave the real text on screen, where it can be read
        # deliberately rather than announced to the room.
        return SpeechText(_SAFE_REPLACEMENT, redacted=True, original_length=len(raw))
    safe = raw

    # Line structure first, because naturalise() flattens it into sentences.
    summarised, dropped_lines = summarise_lines(strip_payloads(safe))
    cleaned = naturalise(summarised)
    shortened, capped = condense(cleaned, max_sentences=max_sentences,
                                 max_chars=max_chars)
    condensed = dropped_lines or capped

    shortened = shortened.rstrip(" :;,-")
    if shortened and not shortened.endswith((".", "!", "?")):
        shortened += "."

    return SpeechText(text=shortened, redacted=redacted, condensed=condensed,
                      original_length=len(raw))


def preview_line(assistant_name: str) -> str:
    """The Test Voice sample (section 10): short, natural, representative.

    Uses the configured assistant name, so a renamed assistant introduces
    itself correctly (section 15).
    """
    return f"Hello, I'm {assistant_name}. I'm ready to help."


def notification_line(message: str, *, greeting: str = "") -> str:
    """Phrase a reminder or notification for speech (section 6).

    A notification arrives unprompted, so it opens with a short greeting to give
    the listener a moment before the content. Phrasing only -- the delivery
    difference is prosody, applied by the engine.
    """
    body = (message or "").strip()
    if not body:
        return ""
    if greeting:
        return f"{greeting.rstrip('.,')}. {body}"
    return body
