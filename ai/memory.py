"""Short-term conversation context.

Specification section 14 describes five memory categories.  Phase 1 implements
exactly one of them — the short-term conversation buffer — because that is the
one the engine needs today, and because the other four (explicit memories,
preferences, task history, persona data) all require the user-facing controls
to view, edit and delete them before it would be right to start storing
anything.  Building the store before the controls is how an assistant ends up
quietly retaining things nobody can get back out.

The buffer here is bounded and in-memory.  The transcript is separately
persisted by :class:`database.db.Database` so the dashboard can restore it, and
the user can clear that at any time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque, Dict, Iterator, List, Literal, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("ai.memory")

Role = Literal["user", "assistant", "system"]

DEFAULT_TURN_LIMIT = 20


@dataclass(frozen=True)
class ConversationTurn:
    """One line of the conversation."""

    role: Role
    text: str
    language: str = "en"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_message(self) -> Dict[str, str]:
        """The shape a language model expects in Phase 2."""
        return {"role": self.role, "content": self.text}


class ConversationContext:
    """A bounded rolling window of recent turns.

    Bounded on purpose: an unbounded transcript would grow without limit and,
    once a model is in the loop, would be resent on every request.
    """

    def __init__(self, limit: int = DEFAULT_TURN_LIMIT) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self._turns: Deque[ConversationTurn] = deque(maxlen=limit)
        self._limit = limit

    @property
    def limit(self) -> int:
        return self._limit

    def add(self, role: Role, text: str, language: str = "en") -> ConversationTurn:
        turn = ConversationTurn(role=role, text=text, language=language)
        self._turns.append(turn)
        return turn

    def add_user(self, text: str, language: str = "en") -> ConversationTurn:
        return self.add("user", text, language)

    def add_assistant(self, text: str, language: str = "en") -> ConversationTurn:
        return self.add("assistant", text, language)

    def turns(self) -> List[ConversationTurn]:
        return list(self._turns)

    def last_user_message(self) -> Optional[ConversationTurn]:
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn
        return None

    def as_messages(self) -> List[Dict[str, str]]:
        """History in provider format, ready for :meth:`ai.llm.LLMProvider.complete`."""
        return [turn.as_message() for turn in self._turns]

    def clear(self) -> None:
        count = len(self._turns)
        self._turns.clear()
        logger.info("Conversation context cleared", extra={"context": {"turns": count}})

    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self) -> Iterator[ConversationTurn]:
        return iter(self._turns)


#: Memory categories from specification section 14 and the phase each is
#: scheduled for.  Kept here so the roadmap lives next to the code that will
#: implement it rather than only in the README.
PLANNED_MEMORY_CATEGORIES: Tuple[Tuple[str, int, str], ...] = (
    ("short_term_context", 1, "Implemented: the rolling window in this module."),
    ("user_preferences", 4, "Preferences AARMA infers or is told, with a settings UI."),
    ("explicit_memories", 4, "\"Remember that ...\", with view/edit/delete controls."),
    ("task_history", 4, "Completed task summaries, opt-in."),
    ("persona_data", 6, "User-authored persona profiles, deletable."),
)
