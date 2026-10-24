"""Human-in-the-loop confirmation for high-risk actions.

The engine never blocks waiting for an answer.  Instead it *parks* the request
here, returns a "needs confirmation" response, and the interface (CLI prompt,
GUI dialog, and later a spoken question) resolves it by token.  That keeps the
same confirmation flow working across every front end and means a user can
always walk away without approving anything.

Pending requests expire, so an unanswered confirmation can never be approved
much later by an unrelated "yes".
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("security.confirmation")

DEFAULT_TTL_SECONDS = 120.0


class ConfirmationOutcome(str, Enum):
    APPROVED = "approved"
    DECLINED = "declined"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


@dataclass
class ConfirmationRequest:
    """A high-risk action waiting for the user to say yes or no."""

    token: str
    action_name: str
    summary: str
    parameters: Dict[str, object] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)
    ttl_seconds: float = DEFAULT_TTL_SECONDS

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl_seconds

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.ttl_seconds - (time.monotonic() - self.created_at))


#: Words that count as approval / refusal in the text interface.  Kept here so
#: the CLI and the GUI agree on what "yes" means.
AFFIRMATIVE = frozenset({"yes", "y", "yeah", "yep", "confirm", "confirmed", "do it",
                         "go ahead", "ok", "okay", "proceed", "sure"})
NEGATIVE = frozenset({"no", "n", "nope", "cancel", "stop", "don't", "dont",
                      "abort", "never mind", "nevermind"})


def interpret_answer(text: str) -> Optional[bool]:
    """Map a free-text reply onto approve/decline, or ``None`` if it is neither."""
    normalised = (text or "").strip().lower().rstrip(".!")
    if normalised in AFFIRMATIVE:
        return True
    if normalised in NEGATIVE:
        return False
    return None


class ConfirmationBroker:
    """Holds pending confirmations, keyed by an unguessable token."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._pending: Dict[str, ConfirmationRequest] = {}

    def create(self, action_name: str, summary: str,
               parameters: Optional[Dict[str, object]] = None) -> ConfirmationRequest:
        """Park an action and return the request the interface should present."""
        self.purge_expired()
        request = ConfirmationRequest(
            token=secrets.token_urlsafe(12),
            action_name=action_name,
            summary=summary,
            parameters=dict(parameters or {}),
            ttl_seconds=self._ttl,
        )
        with self._lock:
            self._pending[request.token] = request
        logger.info("Confirmation requested",
                    extra={"context": {"action": action_name, "token": request.token}})
        return request

    def get(self, token: str) -> Optional[ConfirmationRequest]:
        with self._lock:
            return self._pending.get(token)

    def latest(self) -> Optional[ConfirmationRequest]:
        """Most recent non-expired request -- what a bare "yes" refers to."""
        with self._lock:
            live = [r for r in self._pending.values() if not r.expired]
        return max(live, key=lambda r: r.created_at) if live else None

    def resolve(self, token: str, approved: bool) -> tuple[ConfirmationOutcome, Optional[ConfirmationRequest]]:
        """Consume a pending request.  A token can only be used once."""
        with self._lock:
            request = self._pending.pop(token, None)
        if request is None:
            logger.info("Confirmation token not found", extra={"context": {"token": token}})
            return ConfirmationOutcome.UNKNOWN, None
        if request.expired:
            logger.info("Confirmation expired",
                        extra={"context": {"action": request.action_name, "token": token}})
            return ConfirmationOutcome.EXPIRED, request
        outcome = ConfirmationOutcome.APPROVED if approved else ConfirmationOutcome.DECLINED
        logger.info("Confirmation resolved",
                    extra={"context": {"action": request.action_name, "outcome": outcome.value}})
        return outcome, request

    def cancel_all(self) -> int:
        """Drop every pending request, e.g. when the user locks the assistant."""
        with self._lock:
            count = len(self._pending)
            self._pending.clear()
        if count:
            logger.info("All pending confirmations cancelled", extra={"context": {"count": count}})
        return count

    def purge_expired(self) -> List[ConfirmationRequest]:
        with self._lock:
            expired = [r for r in self._pending.values() if r.expired]
            for request in expired:
                self._pending.pop(request.token, None)
        return expired

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)
