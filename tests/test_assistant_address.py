"""Saying the assistant's name, and what that name is allowed to mean.

Two layers that are easy to confuse and must not be:

    acoustic wake     a model hears a phrase in a room
    assistant address a sentence opens by naming who it is for

Only the first one activates a session. The second is text handling: given a
sentence that reached the pipeline somehow, decide whether it begins with a
form of address and remove it before the command is interpreted.

Keeping them apart is what stops the tempting shortcut -- run recognition
continuously and treat every "Aariz" as a wake -- which would make the assistant
act on its own name being mentioned in a conversation it was never part of.

The failure these tests were written from: "Aariz, open a Python tutorial on
YouTube" searched YouTube for "aariz, open a python tutorial". The address was
removed for the deterministic rules and not for the nearest-intent resolver, so
the assistant's own name became part of what the user was searching for.
"""

from __future__ import annotations

import pytest

from ai.intent import IntentEngine
from config.settings import ConfigManager
from core.engine import AarmaEngine


@pytest.fixture
def parser(config: ConfigManager) -> IntentEngine:
    config.update_assistant(name="Aariz", wake_word="hey aariz")
    return IntentEngine(config)


@pytest.fixture
def active(engine: AarmaEngine) -> AarmaEngine:
    """An engine past the bootstrap, which is where addressing applies."""
    engine.complete_bootstrap()
    engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
    return engine


# --------------------------------------------------------------------------
# Stripping the address
# --------------------------------------------------------------------------

class TestTheAddressIsRemoved:
    @pytest.mark.parametrize("utterance", [
        "Aariz, open Chrome",
        "Aariz open Chrome",
        "Hey Aariz, open Chrome",
        "hey aariz open Chrome",
        "Okay Aariz, open Chrome",
        "OK Aariz, open Chrome",
        "Hi Aariz, open Chrome",
        "AARIZ, OPEN CHROME",
        "Aariz.  open Chrome",
    ])
    def test_the_command_survives_the_address(self, parser: IntentEngine,
                                              utterance: str) -> None:
        assert parser.strip_wake_word(utterance).lower() == "open chrome"

    def test_an_address_with_no_command_is_kept_whole(
            self, parser: IntentEngine) -> None:
        """"Aariz." is a recognisable utterance; an empty string is not."""
        assert parser.strip_wake_word("Aariz.") == "Aariz."

    def test_a_polite_request_still_reaches_the_command(
            self, parser: IntentEngine) -> None:
        assert parser.normalise("Aariz, could you check my system health?") == \
            "check my system health"


class TestTheNameIsStillAWord:
    """§12: strip an address, not every occurrence of the name."""

    @pytest.mark.parametrize("utterance", [
        "open the aariz folder",
        "search for aariz on youtube",
        "tell me about Aariz",
        "what does aariz mean",
    ])
    def test_a_mention_is_left_alone(self, parser: IntentEngine,
                                     utterance: str) -> None:
        assert parser.strip_wake_word(utterance) == utterance

    def test_a_longer_word_starting_with_the_name_is_not_an_address(
            self, parser: IntentEngine) -> None:
        """Otherwise "Aariz" eats the front of any word beginning with it."""
        assert parser.strip_wake_word("Aarizabad weather") == "Aarizabad weather"


# --------------------------------------------------------------------------
# The name must not become part of the command
# --------------------------------------------------------------------------

class TestTheNameNeverReachesTheAction:
    @pytest.mark.parametrize("utterance", [
        "Aariz, open a Python tutorial on YouTube",
        "Aariz, search YouTube for Python tutorials",
        "Aariz, take me to YouTube and find Python tutorials",
        "Hey Aariz, look up the weather in Delhi",
    ])
    def test_no_parameter_contains_the_assistants_name(
            self, active: AarmaEngine, utterance: str) -> None:
        intent = active.intent_engine.parse(utterance)
        rendered = str(dict(intent.parameters)).lower()
        assert "aariz" not in rendered, \
            f"the assistant's own name reached the action: {rendered}"

    def test_the_addressed_and_bare_forms_resolve_alike(
            self, active: AarmaEngine) -> None:
        """Addressing must change who the sentence is for, not what it says."""
        bare = active.intent_engine.parse("open a Python tutorial on YouTube")
        addressed = active.intent_engine.parse(
            "Aariz, open a Python tutorial on YouTube")
        assert addressed.name == bare.name
        assert dict(addressed.parameters) == dict(bare.parameters)


# --------------------------------------------------------------------------
# The address follows the identity
# --------------------------------------------------------------------------

class TestTheAddressIsDynamic:
    @pytest.mark.parametrize("name", ["Friday", "Jarvis", "Nova", "Asma"])
    def test_the_current_name_addresses_the_assistant(
            self, parser: IntentEngine, config: ConfigManager, name: str) -> None:
        config.update_assistant(name=name, wake_word=name.lower())
        assert parser.strip_wake_word(f"{name}, open Chrome").lower() == "open chrome"

    def test_a_name_that_is_not_the_current_one_is_not_an_address(
            self, parser: IntentEngine, config: ConfigManager) -> None:
        config.update_assistant(name="Friday", wake_word="friday")
        assert parser.strip_wake_word("Jarvis, open Chrome") == "Jarvis, open Chrome"

    def test_a_rename_takes_effect_on_the_next_sentence(
            self, parser: IntentEngine, config: ConfigManager) -> None:
        """No reload, no restart -- the name is read per call."""
        config.update_assistant(name="Friday", wake_word="friday")
        assert parser.strip_wake_word("Friday, open Chrome").lower() == "open chrome"
        config.update_assistant(name="Jarvis", wake_word="jarvis")
        assert parser.strip_wake_word("Jarvis, open Chrome").lower() == "open chrome"
        assert parser.strip_wake_word("Friday, open Chrome") == "Friday, open Chrome"


class TestSpeakingToTheOldName:
    """§7: the previous name stops addressing the assistant.

    Not merely ignored -- answered. The action rules are unanchored, so "Aariz,
    open Chrome" matches "open Chrome" inside it and would otherwise be carried
    out with the address silently discarded, leaving the user believing the old
    name still works.
    """

    def test_speaking_to_the_old_name_is_corrected(
            self, active: AarmaEngine) -> None:
        active.handle_text("change your name to Friday")
        response = active.handle_text("Aariz, what time is it", source="voice")
        assert "friday" in response.text.lower()
        assert "current time" not in response.text.lower()

    def test_the_new_name_works_immediately(self, active: AarmaEngine) -> None:
        active.handle_text("change your name to Friday")
        response = active.handle_text("Friday, what time is it", source="voice")
        assert "current time" in response.text.lower()

    def test_typing_is_not_policed_for_addressing(
            self, active: AarmaEngine) -> None:
        """§14: someone typing has already addressed the assistant by typing."""
        active.handle_text("change your name to Friday")
        response = active.handle_text("Aariz, what time is it", source="text")
        assert "current time" in response.text.lower()

    def test_a_mention_of_the_old_name_is_not_an_address(
            self, active: AarmaEngine) -> None:
        active.handle_text("change your name to Friday")
        response = active.handle_text("tell me about Aariz", source="voice")
        assert "i'm friday now" not in response.text.lower()


# --------------------------------------------------------------------------
# Mutation: §28
# --------------------------------------------------------------------------

class TestTheseTestsWouldCatchTheRegression:
    def test_passing_the_name_into_the_query_would_be_caught(
            self, active: AarmaEngine) -> None:
        """§28.4 -- the defect that was actually present."""
        intent = active.intent_engine.parse(
            "Aariz, open a Python tutorial on YouTube")
        leaked = "aariz" in str(dict(intent.parameters)).lower()
        assert leaked is False
        with pytest.raises(AssertionError):
            assert leaked is True

    def test_keeping_the_old_name_would_be_caught(
            self, parser: IntentEngine, config: ConfigManager) -> None:
        """§28.2"""
        config.update_assistant(name="Friday", wake_word="friday")
        stripped = parser.strip_wake_word("Aariz, open Chrome")
        assert stripped == "Aariz, open Chrome"
        with pytest.raises(AssertionError):
            assert stripped.lower() == "open chrome"

    def test_hard_coding_aariz_would_be_caught(
            self, parser: IntentEngine, config: ConfigManager) -> None:
        """§28.3 -- nothing may recognise "aariz" that does not read the name."""
        config.update_assistant(name="Jarvis", wake_word="jarvis")
        assert parser.strip_wake_word("Jarvis, open Chrome").lower() == "open chrome"
        assert parser.strip_wake_word("Aariz, open Chrome") == "Aariz, open Chrome"

    def test_treating_every_mention_as_an_address_would_be_caught(
            self, parser: IntentEngine) -> None:
        """§28.6 -- a name in the middle of a sentence is not an activation."""
        untouched = parser.strip_wake_word("search for aariz on youtube")
        assert untouched == "search for aariz on youtube"
        with pytest.raises(AssertionError):
            assert untouched == "on youtube"
