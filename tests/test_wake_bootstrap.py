"""A session is locked until "Hey Aariz" is heard, then follows the identity.

Two phrases, two different things:

    bootstrap phrase   "hey aariz"  -- the one phrase in this build with a
                                      validated shipped model behind it
    casual wake phrase "asma"       -- what the user actually calls the
                                      assistant, audible only if enrolled

The temptation the product brief warns against is collapsing them: arming the
shipped "hey aariz" model and telling the user it hears a bare "aariz". It does
not. It was trained, evaluated and frozen on "hey aariz". So the rule these
tests defend is that a name becomes audible by being *enrolled*, never by being
configured, and that an identity with no detector is reported as having none
rather than borrowing somebody else's.
"""

from __future__ import annotations

import pytest

from config.settings import ConfigManager
from core.engine import AarmaEngine
from core.wake_session import BOOTSTRAP_PHRASE, WakeSession


# --------------------------------------------------------------------------
# The state machine on its own
# --------------------------------------------------------------------------

class TestTheSessionState:
    def test_a_new_session_is_locked(self) -> None:
        session = WakeSession()
        assert session.bootstrap_required
        assert not session.bootstrap_complete

    def test_the_bootstrap_phrase_unlocks(self) -> None:
        session = WakeSession()
        assert session.complete(BOOTSTRAP_PHRASE) is True
        assert session.bootstrap_complete
        assert session.unlocked_by == BOOTSTRAP_PHRASE

    @pytest.mark.parametrize("heard", ["hey asma", "asma", "friday",
                                       "hey computer", ""])
    def test_anything_else_leaves_it_locked(self, heard: str) -> None:
        session = WakeSession()
        assert session.complete(heard) is False
        assert session.bootstrap_required, \
            "a phrase that is not the bootstrap phrase must not unlock a session"

    @pytest.mark.parametrize("heard", ["HEY AARIZ", "Hey Aariz", "hey  aariz",
                                      "  hey aariz  "])
    def test_case_and_spacing_do_not_matter(self, heard: str) -> None:
        """The same normalisation the registry uses, so the two cannot drift.

        Punctuation is not stripped by either, and does not need to be: the
        phrase compared here is the one the detector was armed with, not
        something a person typed.
        """
        assert WakeSession().complete(heard) is True

    def test_reset_locks_it_again(self) -> None:
        session = WakeSession()
        session.complete(BOOTSTRAP_PHRASE)
        session.reset()
        assert session.bootstrap_required


# --------------------------------------------------------------------------
# The engine follows it
# --------------------------------------------------------------------------

class TestTheEngineArmsWhatTheSessionSays:
    def test_startup_waits_for_the_bootstrap_phrase(
            self, engine: AarmaEngine) -> None:
        assert engine.wake_state["bootstrap_required"] is True
        assert engine.armed_phrase == BOOTSTRAP_PHRASE

    def test_startup_waits_even_for_a_renamed_assistant(
            self, engine: AarmaEngine, config: ConfigManager) -> None:
        """A rename does not let the user skip the unlock.

        The identity's own phrase follows the rename immediately -- that part
        never depended on the session -- but nothing is armed for it yet.
        """
        config.update_assistant(name="Friday", wake_word="hey friday")
        assert engine.wake_phrase == "hey friday"
        assert engine.armed_phrase == BOOTSTRAP_PHRASE

    def test_the_bootstrap_phrase_unlocks_the_engine(
            self, engine: AarmaEngine) -> None:
        assert engine.complete_bootstrap(BOOTSTRAP_PHRASE) is True
        assert engine.wake_state["bootstrap_complete"] is True

    def test_a_wrong_phrase_does_not_unlock_the_engine(
            self, engine: AarmaEngine) -> None:
        assert engine.complete_bootstrap("hey computer") is False
        assert engine.wake_state["bootstrap_required"] is True

    def test_after_bootstrap_the_default_identity_keeps_its_own_model(
            self, engine: AarmaEngine) -> None:
        """Aariz's model is "hey aariz"; it stays armed because it is Aariz's."""
        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        engine.complete_bootstrap()
        assert engine.armed_phrase == "hey aariz"

    def test_a_rename_moves_the_casual_wake_to_the_new_identity(
            self, engine: AarmaEngine, config: ConfigManager) -> None:
        engine.complete_bootstrap()
        config.update_assistant(name="Friday", wake_word="hey friday")
        assert engine.armed_phrase == "hey friday"
        assert engine.wake_state["identity_detectable"] is False, \
            "nothing is enrolled for Friday, and that must be reported"
        assert engine.start_wake_loop() is False, \
            "Friday has no detector, and Aariz's must not stand in for it"

    def test_the_old_identitys_wake_stops_after_a_rename(
            self, engine: AarmaEngine, config: ConfigManager) -> None:
        engine.complete_bootstrap()
        config.update_assistant(name="Aariz", wake_word="hey aariz")
        assert engine.armed_phrase == "hey aariz"
        config.update_assistant(name="Friday", wake_word="hey friday")
        assert engine.armed_phrase != "hey aariz"

    def test_an_identity_with_no_detector_is_reported_as_unavailable(
            self, engine: AarmaEngine, config: ConfigManager) -> None:
        """Reported, not substituted: Aariz's model must not answer for Friday."""
        engine.complete_bootstrap()
        config.update_assistant(name="Friday", wake_word="hey friday")
        state = engine.wake_identity()
        assert state["available"] is False
        assert state["reason"], "an unavailable identity must say why"
        assert engine.wake_state["identity_detectable"] is False

    def test_a_new_engine_is_locked_again(self, config: ConfigManager,
                                          database, resolver) -> None:
        """Restarting the application resets the unlock, by construction."""
        first = AarmaEngine(config, database=database, path_resolver=resolver)
        first.complete_bootstrap()
        assert first.wake_state["bootstrap_complete"]
        first.shutdown()

        second = AarmaEngine(config, database=database, path_resolver=resolver)
        try:
            assert second.wake_state["bootstrap_required"], \
                "a restart must require the bootstrap again"
        finally:
            second.shutdown()

    def test_the_shipped_model_is_untouched(self) -> None:
        """The frozen Aariz model must still be the only shipped provider."""
        from speech.wakeword import REGISTRY

        assert REGISTRY.is_builtin("hey aariz")
        assert [p for p in REGISTRY.phrases() if REGISTRY.is_builtin(p)] == \
            ["hey aariz"]


# --------------------------------------------------------------------------
# Mutation: prove the rules above are load-bearing
# --------------------------------------------------------------------------

class TestTheseTestsWouldCatchTheRegression:
    def test_accepting_a_bare_name_before_bootstrap_would_be_caught(
            self, engine: AarmaEngine, config: ConfigManager) -> None:
        """Arming "asma" while locked is the failure this gate exists for."""
        config.update_assistant(name="Asma", wake_word="asma")
        armed = engine.armed_phrase
        assert armed == BOOTSTRAP_PHRASE
        with pytest.raises(AssertionError):
            assert armed == "asma"

    def test_keeping_the_old_detector_after_a_rename_would_be_caught(
            self, engine: AarmaEngine, config: ConfigManager) -> None:
        engine.complete_bootstrap()
        config.update_assistant(name="Aariz", wake_word="hey aariz")
        before = engine.armed_phrase
        config.update_assistant(name="Friday", wake_word="hey friday")
        after = engine.armed_phrase
        assert after != before
        with pytest.raises(AssertionError):
            assert after == before

    def test_falling_back_to_another_identity_would_be_caught(
            self, engine: AarmaEngine, config: ConfigManager) -> None:
        """On a machine where only Aariz is enrolled, Friday is still silent."""
        engine.complete_bootstrap()
        config.update_assistant(name="Friday", wake_word="hey friday")
        armed = engine.armed_phrase
        assert armed == "hey friday"
        assert engine.start_wake_loop() is False, \
            "no detector for Friday means no wake, not somebody else's model"
        with pytest.raises(AssertionError):
            assert engine.start_wake_loop() is True
