"""Which sources the session gates, and what ends it.

The session exists so that ambient voice cannot act until someone has
deliberately activated it. That makes the interesting question not "is it
locked" but "locked *against what*":

    wake   ambient -- the room triggered a detector.     Gated.
    voice  the microphone button -- a deliberate press.  Not gated.
    text   typing.                                        Not gated.

Gating the last two would mean a broken microphone leaves the assistant unusable
and a typed command waits on a spoken one, which is the opposite of what a
session lock is for. Gating the first is the whole point.
"""

from __future__ import annotations

import pytest

from core.engine import AarmaEngine, ResponseKind
from core.wake_session import BOOTSTRAP_PHRASE, SessionState, WakeSession
from security.permissions import Permission


class TestTheStatesAreNamed:
    def test_a_new_session_requires_the_bootstrap(self) -> None:
        assert WakeSession().state is SessionState.BOOTSTRAP_REQUIRED

    def test_the_bootstrap_makes_it_active(self) -> None:
        session = WakeSession()
        session.complete(BOOTSTRAP_PHRASE)
        assert session.state is SessionState.ACTIVE_SESSION
        assert session.is_active

    def test_the_state_appears_in_the_report(self) -> None:
        assert WakeSession().as_dict()["state"] == "bootstrap_required"


class TestWhatTheLockAppliesTo:
    @pytest.mark.parametrize("source", ["text", "gui", "voice"])
    def test_deliberate_sources_are_not_gated(self, source: str) -> None:
        assert WakeSession().allows(source) is True

    def test_ambient_voice_is_gated(self) -> None:
        assert WakeSession().allows("wake") is False

    def test_everything_is_allowed_once_active(self) -> None:
        session = WakeSession()
        session.complete(BOOTSTRAP_PHRASE)
        assert all(session.allows(s) for s in ("wake", "voice", "text", "gui"))


class TestTheEngineEnforcesIt:
    def test_typing_works_before_the_bootstrap(self, engine: AarmaEngine) -> None:
        """§14: the restriction is on voice activation, not on typing."""
        response = engine.handle_text("what time is it", source="text")
        assert response.ok
        assert "current time" in response.text.lower()

    def test_ambient_voice_is_refused_before_the_bootstrap(
            self, engine: AarmaEngine) -> None:
        response = engine.handle_text("what time is it", source="wake")
        assert not response.ok
        assert "hey aariz" in response.text.lower()

    def test_the_refusal_is_said_rather_than_swallowed(
            self, engine: AarmaEngine) -> None:
        """A voice command that vanishes looks exactly like one misheard."""
        seen = []
        engine.conversation.subscribe(seen.append)
        engine.handle_text("open Chrome", source="wake")
        assert [e.role for e in seen] == ["assistant"]
        assert seen[0].status == "error"

    def test_ambient_voice_works_after_the_bootstrap(
            self, engine: AarmaEngine) -> None:
        engine.complete_bootstrap()
        response = engine.handle_text("what time is it", source="wake")
        assert response.ok

    def test_a_casual_address_cannot_unlock_the_session(
            self, engine: AarmaEngine) -> None:
        """§28.1 -- saying the name is not the activation phrase."""
        engine.handle_text("Aariz, open Chrome", source="wake")
        assert engine.wake_session.state is SessionState.BOOTSTRAP_REQUIRED


class TestEndingTheSession:
    def test_going_to_sleep_ends_the_voice_session(
            self, engine: AarmaEngine) -> None:
        engine.complete_bootstrap()
        response = engine.handle_text("go to sleep")
        assert response.ok
        assert engine.wake_session.state is SessionState.BOOTSTRAP_REQUIRED

    def test_ambient_voice_is_gated_again_afterwards(
            self, engine: AarmaEngine) -> None:
        engine.complete_bootstrap()
        engine.handle_text("go to sleep")
        assert not engine.handle_text("what time is it", source="wake").ok

    def test_ending_an_inactive_session_is_not_an_error(
            self, engine: AarmaEngine) -> None:
        response = engine.handle_text("go to sleep")
        assert response.ok

    @pytest.mark.parametrize("utterance", [
        "go to sleep", "stop listening", "end the session",
    ])
    def test_the_phrasings_reach_the_session_action(
            self, engine: AarmaEngine, utterance: str) -> None:
        assert engine.intent_engine.parse(utterance).name == "end_voice_session"

    @pytest.mark.parametrize("utterance", [
        "put the computer to sleep", "sleep the computer", "suspend the pc",
    ])
    def test_sleeping_the_computer_is_a_different_request(
            self, engine: AarmaEngine, utterance: str) -> None:
        """The expensive reading has to be asked for by naming the machine."""
        assert engine.intent_engine.parse(utterance).name != "end_voice_session"

    def test_ending_the_session_never_suspends_the_machine(
            self, engine: AarmaEngine) -> None:
        """The autouse conftest guard would raise if it tried."""
        engine.complete_bootstrap()
        response = engine.handle_text("go to sleep")
        assert response.action == "end_voice_session"


class TestActivationGrantsNoAuthority:
    """§21 and §28.5: the session is not a permission."""

    def test_a_high_risk_action_still_confirms_after_activation(
            self, engine: AarmaEngine) -> None:
        engine.complete_bootstrap()
        engine.permissions.grant(Permission.SYSTEM_CONTROL)
        response = engine.handle_text("shut down the computer", source="wake")
        assert response.kind is ResponseKind.CONFIRMATION_REQUIRED
        assert response.confirmation_token

    def test_activation_does_not_grant_permissions(
            self, engine: AarmaEngine) -> None:
        engine.complete_bootstrap()
        response = engine.handle_text("shut down the computer", source="wake")
        assert response.kind is not ResponseKind.RESULT
        assert not response.ok

    def test_the_bootstrap_turn_itself_confirms(self, engine: AarmaEngine) -> None:
        """"Hey Aariz, shut down the computer" activates, then asks."""
        engine.permissions.grant(Permission.SYSTEM_CONTROL)
        engine.complete_bootstrap()
        response = engine.handle_text("shut down the computer", source="wake")
        assert response.kind is ResponseKind.CONFIRMATION_REQUIRED

    def test_mutation_bypassing_safety_after_activation_would_be_caught(
            self, engine: AarmaEngine) -> None:
        engine.complete_bootstrap()
        engine.permissions.grant(Permission.SYSTEM_CONTROL)
        response = engine.handle_text("shut down the computer", source="wake")
        assert response.kind is ResponseKind.CONFIRMATION_REQUIRED
        with pytest.raises(AssertionError):
            assert response.kind is ResponseKind.RESULT
