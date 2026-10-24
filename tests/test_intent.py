"""Natural-language understanding.

Specification section 8 requires that users are not forced into exact command
strings, so these tests check *phrasing variety* rather than a fixed grammar.
They also pin down the two behaviours that matter more than coverage: a rule
must not claim a match it cannot fulfil, and an unrecognised sentence must
produce ``unknown`` rather than a guess.
"""

from __future__ import annotations

import pytest

from ai.intent import META_INTENTS, IntentEngine
from config.settings import ConfigManager


@pytest.fixture
def parser(config: ConfigManager) -> IntentEngine:
    return IntentEngine(config)


class TestNormalisation:
    @pytest.mark.parametrize("utterance", [
        "hey asma, what time is it",
        "Hey Asma what time is it",
        "asma, what time is it",
    ])
    def test_wake_word_prefix_is_stripped(self, parser: IntentEngine, utterance: str) -> None:
        assert parser.parse(utterance).name == "get_time"

    def test_wake_word_follows_a_rename(self, parser: IntentEngine,
                                        config: ConfigManager) -> None:
        config.update_assistant(name="Nova", wake_word="hey nova")
        assert parser.parse("hey nova, what time is it").name == "get_time"

    @pytest.mark.parametrize("utterance", [
        "could you open notepad",
        "can you open notepad please",
        "please open notepad",
        "I would like you to open notepad for me",
    ])
    def test_politeness_is_stripped(self, parser: IntentEngine, utterance: str) -> None:
        assert parser.parse(utterance).name == "open_application"


class TestPhrasingVariety:
    @pytest.mark.parametrize("utterance", [
        "what time is it", "what's the time", "tell me the time", "time",
        "What time is it?",
    ])
    def test_time_phrasings(self, parser: IntentEngine, utterance: str) -> None:
        assert parser.parse(utterance).name == "get_time"

    @pytest.mark.parametrize("utterance", [
        "what's the date", "what day is it", "today's date", "date",
    ])
    def test_date_phrasings(self, parser: IntentEngine, utterance: str) -> None:
        assert parser.parse(utterance).name == "get_date"

    @pytest.mark.parametrize("utterance", [
        "open chrome", "launch chrome", "start chrome", "run google chrome",
        "open the chrome app",
    ])
    def test_application_phrasings(self, parser: IntentEngine, utterance: str) -> None:
        intent = parser.parse(utterance)
        assert intent.name == "open_application"
        assert "chrome" in str(intent.parameters["name"]).lower()

    @pytest.mark.parametrize("utterance", [
        "search for python tutorials",
        "search the web for python tutorials",
        "search the internet for python tutorials",
        "google python tutorials",
        "look up python tutorials",
    ])
    def test_search_phrasings(self, parser: IntentEngine, utterance: str) -> None:
        intent = parser.parse(utterance)
        assert intent.name == "search_web"
        assert intent.parameters["query"] == "python tutorials"


class TestParameterExtraction:
    def test_folder_name_and_location(self, parser: IntentEngine) -> None:
        intent = parser.parse("create a folder called AI Project in Documents")
        assert intent.name == "create_folder"
        assert intent.parameters["name"] == "AI Project"
        assert intent.parameters["location"] == "Documents"

    def test_folder_without_a_location(self, parser: IntentEngine) -> None:
        intent = parser.parse("create a folder called Notes")
        assert intent.name == "create_folder"
        assert intent.parameters["name"] == "Notes"
        assert "location" not in intent.parameters

    def test_rename_extracts_both_halves(self, parser: IntentEngine) -> None:
        intent = parser.parse("rename draft.txt to final.txt")
        assert intent.name == "rename_item"
        assert intent.parameters == {"path": "draft.txt", "new_name": "final.txt"}

    def test_assistant_rename(self, parser: IntentEngine) -> None:
        intent = parser.parse("change your name to Nova")
        assert intent.name == "set_assistant_name"
        assert intent.parameters["name"] == "Nova"

    def test_language_change_resolves_to_a_code(self, parser: IntentEngine) -> None:
        intent = parser.parse("change language to Hindi")
        assert intent.name == "set_language"
        assert intent.parameters["language"] == "hi"

    def test_scoped_language_change(self, parser: IntentEngine) -> None:
        intent = parser.parse("change the output language to Tamil")
        assert intent.name == "set_language"
        assert intent.parameters["language"] == "ta"
        assert intent.parameters["scope"] == "output"

    def test_url_is_recognised(self, parser: IntentEngine) -> None:
        intent = parser.parse("open youtube.com")
        assert intent.name == "open_url"
        assert intent.parameters["url"] == "youtube.com"


class TestRulesDeclineRatherThanGuess:
    def test_an_unapproved_application_is_not_claimed(self, parser: IntentEngine) -> None:
        """"open blender" must not become open_application with a bogus argument."""
        assert parser.parse("open blender").name != "open_application"

    def test_an_unknown_language_is_not_claimed(self, parser: IntentEngine) -> None:
        assert parser.parse("speak Klingon").name != "set_language"

    def test_a_non_url_is_not_claimed_as_a_url(self, parser: IntentEngine) -> None:
        assert parser.parse("open chrome").name == "open_application"


class TestOrdering:
    def test_bare_quit_exits_but_quitting_an_app_does_not(self, parser: IntentEngine) -> None:
        assert parser.parse("quit").name == "exit"
        assert parser.parse("quit spotify").name == "close_application"

    @pytest.mark.parametrize("utterance", ["exit", "bye", "goodbye", "see you"])
    def test_exit_phrasings(self, parser: IntentEngine, utterance: str) -> None:
        assert parser.parse(utterance).name == "exit"

    def test_google_as_a_verb_beats_google_as_a_site(self, parser: IntentEngine) -> None:
        assert parser.parse("google quantum computing").name == "search_web"


class TestUnknownInput:
    @pytest.mark.parametrize("utterance", [
        "blah blah nonsense",
        "asdkjhaskjdh",
        "tell me a story about a dragon",
        "what is the capital of Peru",
        "",
        "   ",
    ])
    def test_unrecognised_input_is_unknown(self, parser: IntentEngine, utterance: str) -> None:
        intent = parser.parse(utterance)
        assert intent.name == "unknown"
        assert intent.confidence == 0.0

    def test_parsing_never_raises(self, parser: IntentEngine) -> None:
        for hostile in ("()[]{}", "\\", "*" * 500, "\x00\x01", "%s%s%s", "' OR 1=1 --"):
            assert parser.parse(hostile) is not None


class TestMetaIntents:
    @pytest.mark.parametrize(("utterance", "expected"), [
        ("hello", "greeting"), ("hi there", "greeting"), ("good morning", "greeting"),
        ("thanks", "thanks"), ("help", "help"), ("what can you do", "help"),
    ])
    def test_conversational_intents(self, parser: IntentEngine, utterance: str,
                                    expected: str) -> None:
        intent = parser.parse(utterance)
        assert intent.name == expected
        assert intent.is_meta

    def test_action_intents_are_not_meta(self, parser: IntentEngine) -> None:
        assert parser.parse("what time is it").is_action

    def test_every_non_meta_intent_names_a_registered_action(self, parser: IntentEngine,
                                                             registry) -> None:
        """A rule that emits an unregistered name would be dead code at best."""
        for name in parser.known_intents():
            assert name in META_INTENTS or name in registry, name


class TestQuestionsThatLookLikeCapabilityQuestions:
    """Three phrasings that a live command sweep found going to the wrong place.

    All three are about the *assistant's own* state -- what it is connected to,
    which folders it may touch -- and all three lost to a neighbouring rule
    rather than to a missing one. Capability discovery matches ``can you X``
    anywhere in a sentence, so it claimed the workspace question and answered
    with a list of categories; ``show me my X`` demanded the word "me" and so
    missed the shorter phrasing entirely.
    """

    @pytest.mark.parametrize("utterance", [
        "what folders can you work in",
        "which folders can you access",
        "show my workspace",
        "where can you work",
    ])
    def test_workspace_questions_beat_capability_discovery(
            self, parser: IntentEngine, utterance: str) -> None:
        assert parser.parse(utterance).name == "list_workspace_roots"

    @pytest.mark.parametrize("utterance", [
        "what are you connected to",
        "what's connected",
        "is my calendar connected",
        "which integrations are set up",
    ])
    def test_integration_questions_beat_capability_discovery(
            self, parser: IntentEngine, utterance: str) -> None:
        assert parser.parse(utterance).name == "list_integrations"

    @pytest.mark.parametrize("utterance", ["show my documents", "show my downloads"])
    def test_show_my_folder_does_not_need_the_word_me(
            self, parser: IntentEngine, utterance: str) -> None:
        assert parser.parse(utterance).name == "open_folder"

    @pytest.mark.parametrize("utterance", [
        "what can you do",
        "what commands do you support",
        "list your capabilities",
    ])
    def test_capability_discovery_still_wins_its_own_questions(
            self, parser: IntentEngine, utterance: str) -> None:
        """The new rules sit in front of it, so prove they did not shadow it."""
        assert parser.parse(utterance).name in {"list_capabilities", "help"}

    @pytest.mark.parametrize("utterance", ["show my repositories", "show my drafts"])
    def test_show_my_x_does_not_swallow_things_that_are_not_folders(
            self, parser: IntentEngine, utterance: str) -> None:
        """Widening the folder pattern must stay contained by its validator."""
        assert parser.parse(utterance).name != "open_folder"
