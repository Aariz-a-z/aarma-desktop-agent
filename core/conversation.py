"""One place every visible turn is announced from.

The bug this exists for: a command spoken to the wake word ran, spoke its
reply, and never appeared in the conversation view. Typing worked. Pressing the
microphone button worked. Saying "Hey Aariz" did not -- because
:meth:`AarmaEngine._run_voice_turn` called ``handle_text`` on its own thread and
the desktop window was started with no callbacks to hear about it. Three entry
routes, two of which the interface knew about.

So the interface no longer learns about turns by being on the right end of a
call. Every route publishes here, and anything that displays a conversation
subscribes. Adding a fourth route cannot silently skip the transcript, because
publishing is not something a route does -- it is something ``handle_text``
does, once, for all of them.

Two rules keep this from becoming a liability of its own:

**A subscriber cannot break a turn.** Delivery is wrapped per subscriber. A
window that throws while rendering must cost a rendered message, never the
action the user asked for.

**Publishing is not storage.** :class:`ai.memory.ConversationContext` remains
the record of what was said, and the database remains the persistent one. This
is a notification that a turn happened, and it deliberately holds no history:
a bus that remembers is a second source of truth about the conversation, and
the two would eventually disagree.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("core.conversation")

#: Who said it. ``system`` covers things that are neither: a reminder ringing,
#: a wake phrase being heard, a microphone that could not be opened.
ROLES = ("user", "assistant", "system")


@dataclass(frozen=True)
class ConversationEvent:
    """One thing worth showing in the transcript."""

    role: str
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    #: How it arrived: ``text``, ``gui``, ``voice``, ``wake``, ``reminder``.
    #: Interfaces use it to avoid echoing a message they drew themselves.
    source: str = ""
    intent: str = ""
    action: str = ""
    #: ``ok``, ``error``, ``confirmation``, or empty for a plain line.
    status: str = ""
    confirmation_token: str = ""
    #: Recognition confidence for a spoken turn, when the recogniser gave one.
    confidence: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def spoken(self) -> bool:
        return self.source in ("voice", "wake")

    def as_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "text": self.text,
                "timestamp": self.timestamp.isoformat(timespec="seconds"),
                "source": self.source, "intent": self.intent,
                "action": self.action, "status": self.status,
                "confidence": self.confidence}


Subscriber = Callable[[ConversationEvent], None]


class ConversationBus:
    """Publishes turns to whatever is displaying them.

    Subscribers are called on whichever thread published -- the GUI thread for
    typing, a wake-turn thread for speech -- so a subscriber that touches
    widgets is responsible for marshalling. That is deliberately the
    subscriber's problem: the bus has no way to know what a given interface
    considers its own thread, and a bus that guessed would be wrong for the
    console.
    """

    def __init__(self) -> None:
        self._subscribers: List[Subscriber] = []
        self._lock = threading.RLock()

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        """Register a listener; returns a callable that removes it again."""
        with self._lock:
            self._subscribers.append(subscriber)

        def _unsubscribe() -> None:
            with self._lock:
                if subscriber in self._subscribers:
                    self._subscribers.remove(subscriber)

        return _unsubscribe

    def publish(self, event: ConversationEvent) -> int:
        """Deliver to every subscriber. Returns how many received it.

        One subscriber's failure does not stop the next one, and none of them
        can stop the turn: an interface that throws while rendering has cost
        the user a line on screen, and must not also cost them the action.
        """
        with self._lock:
            subscribers = list(self._subscribers)
        delivered = 0
        for subscriber in subscribers:
            try:
                subscriber(event)
                delivered += 1
            except Exception:  # noqa: BLE001 - display must not break execution
                logger.exception("Conversation subscriber raised",
                                 extra={"context": {"role": event.role}})
        return delivered

    # -- convenience, so callers do not assemble events by hand -------------

    def say_user(self, text: str, *, source: str = "", intent: str = "",
                 confidence: Optional[float] = None) -> None:
        self.publish(ConversationEvent(role="user", text=text, source=source,
                                       intent=intent, confidence=confidence))

    def say_assistant(self, text: str, *, source: str = "", action: str = "",
                      status: str = "", confirmation_token: str = "",
                      data: Optional[Dict[str, Any]] = None) -> None:
        self.publish(ConversationEvent(
            role="assistant", text=text, source=source, action=action,
            status=status, confirmation_token=confirmation_token,
            data=dict(data or {})))

    def say_system(self, text: str, *, source: str = "",
                   status: str = "") -> None:
        self.publish(ConversationEvent(role="system", text=text, source=source,
                                       status=status))

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


__all__ = ["ROLES", "ConversationBus", "ConversationEvent", "Subscriber"]
