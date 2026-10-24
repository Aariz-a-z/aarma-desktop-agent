"""Ending the voice session, which is not the computer going to sleep.

"Go to sleep" is ambiguous in exactly the way that matters: it can mean *stop
listening to me* or *suspend this machine*. One is free and instantly
reversible; the other closes the user's work and needs a password to undo. So
the ambiguity is resolved towards the harmless reading, and the destructive one
keeps the phrasings that name the machine -- "sleep the computer", "put the PC
to sleep" -- which nobody says when they mean "stop listening".

This is the same reasoning the project already applies to "close this": when a
phrase could address two things and one of them is expensive to get wrong, the
cheap reading wins and the expensive one has to be asked for explicitly.
"""

from __future__ import annotations

from typing import Dict, List

from actions.registry import ActionContext, ActionResult, ActionSpec
from security.validator import RiskLevel
from utils.logger import get_logger

logger = get_logger("actions.session")


def _end_session(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Stop responding to ambient voice until the bootstrap phrase is said.

    Reaches the engine through ``ctx.session_control`` rather than importing
    it: the action layer describes *what* may happen, and an action that
    reached back into the engine would make the registry depend on the thing
    that owns it.
    """
    control = getattr(ctx, "session_control", None)
    if control is None:
        return ActionResult.failure(
            "I can't end the voice session from here.",
            error_code="no_session_control")

    if not control():
        return ActionResult.success(
            ctx.language.t("wake.session_already_ended"))

    logger.info("Voice session ended by command")
    return ActionResult.success(ctx.language.t("wake.session_ended"))


def get_actions() -> List[ActionSpec]:
    return [
        ActionSpec(
            name="end_voice_session",
            description="Stop listening until you say the wake phrase again.",
            category="Assistant",
            # Free and instantly reversible: the user says the bootstrap
            # phrase and the session is back. Nothing is closed or lost.
            risk=RiskLevel.LOW,
            handler=_end_session,
            examples=("go to sleep", "stop listening", "goodbye for now"),
        ),
    ]


__all__ = ["get_actions"]
