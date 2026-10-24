"""Which phrase is armed, and why it changes during a session.

A fresh session starts locked to one phrase -- "hey aariz" -- because that is
the only phrase in this build with a validated shipped model behind it. Hearing
it unlocks the session, and from then on the engine arms the current identity's
own phrase, which the rename system already maintains in
``assistant.wake_word``. Deriving that phrase a second time here would be a
second answer to a question that already has one.

The distinction this module exists to keep is the one in the product brief:

    bootstrap phrase  vs  casual wake phrase

They are not the same string and they are not backed by the same model. The
shipped model was trained on "hey aariz" and hears "hey aariz". It does not
hear a bare "aariz", and nothing here is allowed to imply that it does -- so a
bare name is only ever armed when a detector was actually enrolled for it.

What follows from that, and is deliberately not softened:

**No substitution across identities.** If Friday has no detector, the answer is
that Friday has no detector -- the registry returns a ``MissingWakeWordDetector``
and the loop refuses to start. Arming Aariz's model because it happens to exist
would mean the user says "Friday" and nothing happens, or says "Hey Aariz" and
an assistant called Friday answers -- both of which teach the wrong thing about
what the machine can hear.

**Bootstrap does not survive a restart.** It is a session-level activation
step, so the state lives on the object and a new process starts locked again.
Persisting it would quietly turn an unlock into a setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger

logger = get_logger("core.wake_session")

#: The one phrase with a validated shipped model in this build. Kept here as a
#: constant rather than read from configuration: the whole point of a bootstrap
#: is that it cannot be pointed at a phrase nothing can actually detect.
BOOTSTRAP_PHRASE = "hey aariz"


class SessionState(str, Enum):
    """Whether this session has been activated for ambient voice."""

    BOOTSTRAP_REQUIRED = "bootstrap_required"
    ACTIVE_SESSION = "active_session"


#: Where an utterance came from, and whether the session gates it.
#:
#: ``wake`` is ambient: the room was being listened to and something in it
#: triggered a detector. That is the only source the bootstrap gates, because
#: it is the only one that happens without the user deciding to speak *to* the
#: assistant at that moment.
#:
#: ``voice`` is the microphone button -- a deliberate press, and therefore an
#: activation in its own right. ``text`` and ``gui`` are typing. Gating those
#: would leave someone with a broken microphone unable to use the assistant at
#: all, and would gate the one path that cannot be triggered by the room.
GATED_SOURCES = frozenset({"wake"})


@dataclass
class WakeSession:
    """Whether this session has been unlocked, and what that means it hears."""

    bootstrap_required: bool = True
    #: The phrase that completed the bootstrap, for the record.
    unlocked_by: str = ""

    @property
    def bootstrap_complete(self) -> bool:
        return not self.bootstrap_required

    @property
    def state(self) -> SessionState:
        return (SessionState.BOOTSTRAP_REQUIRED if self.bootstrap_required
                else SessionState.ACTIVE_SESSION)

    @property
    def is_active(self) -> bool:
        return self.state is SessionState.ACTIVE_SESSION

    def allows(self, source: str) -> bool:
        """Whether an utterance from ``source`` may run in this state."""
        return self.is_active or source not in GATED_SOURCES

    def complete(self, phrase: str = BOOTSTRAP_PHRASE) -> bool:
        """Unlock the session. Returns False if the phrase was not the right one.

        A wrong phrase leaves the session locked rather than unlocking it
        quietly, which is what makes the state testable: "the user said
        something else" and "the user said the bootstrap phrase" have to be
        distinguishable outcomes.
        """
        if normalise(phrase) != BOOTSTRAP_PHRASE:
            logger.info("Bootstrap not completed by an unrelated phrase",
                        extra={"context": {"heard": phrase}})
            return False
        if self.bootstrap_required:
            logger.info("Wake bootstrap complete",
                        extra={"context": {"phrase": BOOTSTRAP_PHRASE}})
        self.bootstrap_required = False
        self.unlocked_by = BOOTSTRAP_PHRASE
        return True

    def reset(self) -> None:
        """Lock the session again. A new process gets this for free."""
        if not self.bootstrap_required:
            logger.info("Voice session deactivated")
        self.bootstrap_required = True
        self.unlocked_by = ""

    def as_dict(self) -> dict:
        return {"state": self.state.value,
                "bootstrap_required": self.bootstrap_required,
                "bootstrap_complete": self.bootstrap_complete,
                "bootstrap_phrase": BOOTSTRAP_PHRASE,
                "unlocked_by": self.unlocked_by}


def normalise(phrase: str) -> str:
    """Exactly the registry's normalisation: lower-cased and single-spaced.

    Deliberately the same function rather than a similar one. If this were any
    stricter or any looser, a phrase could unlock the session without arming a
    detector, or arm one without unlocking -- and the two layers would disagree
    about what the user just said.
    """
    from config.settings import normalise_wake_phrase

    return normalise_wake_phrase(phrase)


__all__ = ["BOOTSTRAP_PHRASE", "GATED_SOURCES", "SessionState",
           "WakeSession", "normalise"]
