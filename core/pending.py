"""One half-finished action, and nothing else.

"Schedule a meeting tomorrow" is a complete sentence and an incomplete
instruction.  The natural repair is a question -- *what time?* -- and then the
answer, "10 AM", has to be understood as belonging to that question rather than
as a new command.  This module is the whole of that mechanism.

It is deliberately the smallest thing that works, because the failure mode of
conversational state is not forgetting, it is **remembering the wrong thing**.
A store that kept everything the user said would eventually attach a stray
sentence to a stale action and create a meeting nobody asked for.  So:

* **One slot at a time.**  A pending action names exactly one missing field and
  the question that was asked about it.  There is no queue of half-heard
  instructions.
* **An answer must fit the slot.**  A time slot only accepts something
  :func:`utils.when.parse_when` can read.  "Actually, what's the weather?" is
  not a time, so it does not become one.
* **A new command wins.**  The engine parses first; if the utterance is a
  recognised instruction, the pending action is abandoned rather than fed.
  This is the rule that stops unrelated speech filling a slot.
* **It expires.**  An unanswered question is dropped after
  :data:`DEFAULT_TTL_SECONDS`, so an answer given ten minutes later to a
  different question cannot complete it.
* **Only listed actions can be completed this way.**  :data:`FILLABLE` is an
  allow-list, so a failure in some unrelated handler cannot open a dialogue.
  Sending email is not on it: a send is confirmed explicitly, never assembled
  out of fragments.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from utils.logger import get_logger
from utils.when import parse_when

logger = get_logger("core.pending")

#: How long a question stays open.  Two minutes is long enough to think and
#: short enough that the answer is still about the question.
DEFAULT_TTL_SECONDS = 120.0

#: How many follow-up questions one action may ask before giving up.  Without a
#: bound, a handler that always reports something missing would interrogate the
#: user indefinitely.
MAX_TURNS = 4

#: What kind of answer each slot name takes.  Slot names come from the
#: ``needs`` field a handler already returns, so this table is the only place
#: the two layers have to agree.
SLOT_KINDS: Dict[str, str] = {
    "when": "time",
    "title": "text",
    "location": "text",
    "subject": "text",
    "body": "text",
    "to": "text",
}

#: Actions that may be completed across turns, and the slots each will accept.
#: An allow-list rather than "anything that reports a missing field", so a new
#: handler cannot start a conversation by accident.
FILLABLE: Dict[str, Tuple[str, ...]] = {
    "create_event": ("when", "title", "location"),
    "create_reminder": ("when", "title"),
    "draft_email": ("to", "subject", "body"),
    "get_weather": ("location",),
}

#: Phrases that abandon whatever is pending.  Matched against the whole
#: utterance, so "cancel my 6 PM reminder" is a command and not a cancellation.
_CANCEL = re.compile(
    r"^(?:never ?mind|nevermind|cancel(?: that| it)?|forget (?:it|that|about it)|"
    r"stop|drop it|don'?t bother|no thanks?|leave it)[.!]?$",
    re.IGNORECASE)

#: An answer longer than this is a sentence, not a slot value.
MAX_ANSWER_LENGTH = 200


def is_cancellation(text: str) -> bool:
    """Whether this utterance abandons the pending action."""
    return bool(_CANCEL.match(" ".join(str(text or "").strip().split())))


@dataclass
class PendingAction:
    """One action waiting on one answer."""

    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    needs: str = ""
    question: str = ""
    turns: int = 0
    created_at: float = field(default_factory=time.monotonic)
    ttl_seconds: float = DEFAULT_TTL_SECONDS
    source: str = "text"

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds

    @property
    def exhausted(self) -> bool:
        return self.turns >= MAX_TURNS

    def as_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "needs": self.needs,
                "question": self.question, "turns": self.turns,
                "parameters": dict(self.parameters)}


def merge_time(previous: str, answer: str) -> str:
    """Combine a day already given with a clock time supplied later.

    This exists because of a bug found in live testing, and the bug is worth
    recording because every unit test passed through it.

    "Schedule a meeting tomorrow" stores ``when="tomorrow"`` and asks for the
    time.  The answer "10 AM" used to *replace* that value outright, so the day
    the user actually said was discarded and ``parse_when("10 AM")`` resolved
    against the current moment instead.  At 22:00 that returns tomorrow at
    10:00 -- because 10 AM has passed, the parser picks the next occurrence --
    and the right answer appears by accident.  At 01:00 it returns *today* at
    10:00, and the meeting is booked on the wrong day while the assistant says
    it is done.

    So the merge is conditional on exactly the case that loses information: the
    earlier phrase named a day and no clock time, and the answer names a clock
    time and no day.  Anything else replaces, because an answer that carries
    its own day is a correction rather than a completion.
    """
    if not previous:
        return answer
    earlier, later = parse_when(previous), parse_when(answer)
    if earlier is None or later is None:
        return answer
    if earlier.assumed_time and later.assumed_date:
        # The day came from the first phrase, the clock from the second.
        return f"{previous} at {answer}"
    return answer


def interpret_answer(pending: PendingAction, text: str) -> Optional[Any]:
    """Read ``text`` as the answer to ``pending``'s question.

    ``None`` means "this is not an answer to that question" -- which the engine
    treats as a reason to drop the pending action rather than to guess.  That
    is the whole safety property: a value that does not fit the slot never
    reaches the slot.
    """
    answer = " ".join(str(text or "").strip().split())
    if not answer or len(answer) > MAX_ANSWER_LENGTH:
        return None
    if is_cancellation(answer):
        return None

    kind = SLOT_KINDS.get(pending.needs, "text")
    if kind == "time":
        # The slot holds the phrase, not a datetime: the handler re-parses it
        # with its own reference clock, so a slow answer does not shift the
        # meaning of "in ten minutes" to when the question was asked.
        return answer if parse_when(answer) is not None else None

    # Text slots: strip the framing people put round a short answer.
    cleaned = re.sub(
        r"^(?:it'?s |call it |name it |title it |the |to |just )+", "",
        answer, flags=re.IGNORECASE).strip(" \"'.,")
    return cleaned or None


class PendingActionStore:
    """Holds at most one pending action.

    Single-slot by construction: opening a new one replaces whatever was there.
    Two simultaneous half-finished actions would need the user to track which
    question they were answering, which is precisely the confusion this exists
    to avoid.
    """

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._pending: Optional[PendingAction] = None

    # -- reading -----------------------------------------------------------

    def current(self) -> Optional[PendingAction]:
        """The live pending action, dropping it if it has expired."""
        pending = self._pending
        if pending is None:
            return None
        if pending.expired:
            logger.info("Pending action expired",
                        extra={"context": {"action": pending.action}})
            self._pending = None
            return None
        return pending

    @property
    def is_open(self) -> bool:
        return self.current() is not None

    # -- writing -----------------------------------------------------------

    def open(self, action: str, parameters: Mapping[str, Any], *, needs: str,
             question: str, source: str = "text") -> Optional[PendingAction]:
        """Start (or continue) a dialogue for ``action``.

        Returns ``None`` -- and opens nothing -- when the action is not on the
        allow-list or the slot is not one it accepts.  A caller that ignores
        the ``None`` simply gets the handler's own error message, which is the
        safe outcome.
        """
        slots = FILLABLE.get(action)
        if slots is None or needs not in slots:
            logger.info("Refused to open a dialogue",
                        extra={"context": {"action": action, "needs": needs}})
            self.clear()
            return None

        previous = self.current()
        turns = previous.turns + 1 if previous is not None and previous.action == action else 1
        if turns > MAX_TURNS:
            logger.info("Pending action gave up",
                        extra={"context": {"action": action, "turns": turns}})
            self.clear()
            return None

        pending = PendingAction(
            action=action, parameters=dict(parameters), needs=needs,
            question=question, turns=turns, ttl_seconds=self._ttl, source=source)
        self._pending = pending
        logger.info("Pending action opened", extra={"context": {
            "action": action, "needs": needs, "turn": turns}})
        return pending

    def fill(self, value: Any) -> Optional[PendingAction]:
        """Write ``value`` into the slot being asked about.

        Returns the updated pending action, or ``None`` if nothing is pending.
        The action is *not* cleared: the engine runs it and clears on the
        outcome, so a second missing field can reopen from the same parameters.
        """
        pending = self.current()
        if pending is None:
            return None
        if SLOT_KINDS.get(pending.needs) == "time":
            # Never a blind overwrite: see :func:`merge_time`.
            value = merge_time(str(pending.parameters.get(pending.needs) or ""),
                               str(value))
        pending.parameters[pending.needs] = value
        return pending

    def clear(self, reason: str = "") -> Optional[PendingAction]:
        """Drop whatever is pending.  Returns what was dropped, if anything."""
        pending = self._pending
        self._pending = None
        if pending is not None and reason:
            logger.info("Pending action cleared", extra={"context": {
                "action": pending.action, "reason": reason}})
        return pending


__all__ = ["DEFAULT_TTL_SECONDS", "FILLABLE", "MAX_ANSWER_LENGTH", "MAX_TURNS",
           "PendingAction", "PendingActionStore", "SLOT_KINDS",
           "interpret_answer", "is_cancellation", "merge_time"]
