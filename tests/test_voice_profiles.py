"""Male and female are complete voice profiles, not two loose fields.

The bug was visible in the live configuration: ``assistant.gender`` said
"female" while ``assistant.voice_id`` said "male_calm". Two stored answers to
one question, with nothing reconciling them, so the assistant reported a female
profile in its status while a male voice actually spoke -- the synthesiser
resolves gender from the *voice*, and the gender field was read in exactly one
place, a status line.

What follows from that is the shape of these tests. Most of them assert not
that a setting changed, but that two settings agree: the label and the voice,
the UI and the synthesiser, the command and the settings screen. A profile
switch that updated only half of itself is the regression being guarded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import ConfigManager
from core.engine import AarmaEngine
from speech.voices import VOICE_PROFILES, get_profile, profiles_for_gender


def resolved_gender(config: ConfigManager) -> str:
    """What the synthesiser will actually use, as opposed to what is claimed."""
    return get_profile(config.config.assistant.voice_id).gender


@pytest.fixture
def voice_config(tmp_path: Path) -> ConfigManager:
    manager = ConfigManager(tmp_path / "config.json")
    manager.load()
    return manager


# --------------------------------------------------------------------------
# The label and the voice must agree
# --------------------------------------------------------------------------

class TestTheProfileNeverLiesAboutTheVoice:
    def test_the_default_configuration_agrees_with_itself(
            self, voice_config: ConfigManager) -> None:
        assert voice_config.config.assistant.gender == resolved_gender(voice_config)

    @pytest.mark.parametrize("voice_id", sorted(VOICE_PROFILES))
    def test_choosing_a_voice_moves_the_profile_with_it(
            self, voice_config: ConfigManager, voice_id: str) -> None:
        """The exact bug: a male voice selected under a female profile."""
        voice_config.select_voice(voice_id)
        assert voice_config.config.assistant.gender == resolved_gender(voice_config)
        assert voice_config.config.assistant.gender == VOICE_PROFILES[voice_id].gender

    def test_the_stale_combination_cannot_survive_a_selection(
            self, voice_config: ConfigManager) -> None:
        voice_config.update_assistant(gender="female")
        voice_config.select_voice("male_calm")
        assert voice_config.config.assistant.gender == "male"

    def test_an_unknown_voice_is_refused_rather_than_stored(
            self, voice_config: ConfigManager) -> None:
        with pytest.raises(ValueError):
            voice_config.select_voice("not_a_voice")

    @pytest.mark.parametrize("presentation", ["male", "female"])
    def test_switching_presentation_agrees_with_itself(
            self, voice_config: ConfigManager, presentation: str) -> None:
        voice_config.set_voice_presentation(presentation)
        assert voice_config.config.assistant.gender == presentation
        assert resolved_gender(voice_config) == presentation


# --------------------------------------------------------------------------
# A profile carries its own delivery settings
# --------------------------------------------------------------------------

class TestEachProfileRemembersItsOwnSettings:
    def test_switching_back_restores_what_was_set(
            self, voice_config: ConfigManager) -> None:
        voice_config.set_voice_presentation("female")
        voice_config.update_voice(rate=1.3, pitch=2.0)
        voice_config.set_voice_presentation("male")
        voice_config.update_voice(rate=0.8, pitch=-3.0)

        voice_config.set_voice_presentation("female")
        assert voice_config.config.voice.rate == pytest.approx(1.3)
        assert voice_config.config.voice.pitch == pytest.approx(2.0)

        voice_config.set_voice_presentation("male")
        assert voice_config.config.voice.rate == pytest.approx(0.8)
        assert voice_config.config.voice.pitch == pytest.approx(-3.0)

    def test_the_other_profiles_settings_do_not_leak_in(
            self, voice_config: ConfigManager) -> None:
        """Stale values from the previous profile are the visible symptom."""
        voice_config.set_voice_presentation("female")
        voice_config.update_voice(rate=1.5)
        voice_config.set_voice_presentation("male")
        assert voice_config.config.voice.rate != pytest.approx(1.5)

    def test_shared_settings_survive_a_switch(
            self, voice_config: ConfigManager) -> None:
        """Volume and provider mode are not traits of a voice."""
        voice_config.update_voice(volume=0.42, provider_mode="offline",
                                  speak_notifications=False)
        voice_config.set_voice_presentation("male")
        voice = voice_config.config.voice
        assert voice.volume == pytest.approx(0.42)
        assert voice.provider_mode == "offline"
        assert voice.speak_notifications is False

    def test_everything_survives_a_restart(self, tmp_path: Path) -> None:
        first = ConfigManager(tmp_path / "config.json")
        first.load()
        first.set_voice_presentation("female")
        first.update_voice(rate=1.25, pitch=1.0)
        first.set_voice_presentation("male")
        first.update_voice(rate=0.75)

        second = ConfigManager(tmp_path / "config.json")
        second.load()
        assert second.config.assistant.gender == "male"
        assert second.config.voice.rate == pytest.approx(0.75)
        assert second.config.voice.female_rate == pytest.approx(1.25)
        assert second.config.voice.female_pitch == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------

class TestAvailability:
    def test_an_unknown_presentation_is_refused(
            self, voice_config: ConfigManager) -> None:
        changed, explanation = voice_config.set_voice_presentation("robot")
        assert changed is False
        assert "robot" in explanation

    def test_a_refusal_changes_nothing(self, voice_config: ConfigManager) -> None:
        before = voice_config.config.assistant.voice_id
        voice_config.set_voice_presentation("robot")
        assert voice_config.config.assistant.voice_id == before

    def test_asking_for_the_current_profile_is_not_a_failure(
            self, voice_config: ConfigManager) -> None:
        voice_config.set_voice_presentation("male")
        changed, explanation = voice_config.set_voice_presentation("male")
        assert changed is False
        assert "already" in explanation.lower()

    def test_no_candidate_is_ever_the_other_presentation(
            self, voice_config: ConfigManager) -> None:
        """The rule that matters: never answer "male" with a female voice.

        Driven by pointing the remembered male voice at a female one, which is
        the state a corrupted or hand-edited config could produce.
        """
        voice_config.update_voice(male_voice_id="female_warm")
        voice_config.set_voice_presentation("male")
        assert resolved_gender(voice_config) == "male"
        assert voice_config.config.assistant.voice_id in {
            p.voice_id for p in profiles_for_gender("male")}


# --------------------------------------------------------------------------
# By command, through the registry
# --------------------------------------------------------------------------

class TestSpokenAndTypedCommands:
    @pytest.mark.parametrize(("utterance", "expected"), [
        ("use a male voice", "male"),
        ("switch to a female voice", "female"),
        ("change your voice to male", "male"),
        ("i want a female voice", "female"),
        ("use the male assistant voice", "male"),
    ])
    def test_a_command_switches_the_profile(
            self, engine: AarmaEngine, utterance: str, expected: str) -> None:
        response = engine.handle_text(utterance)
        assert response.ok, response.text
        assert engine.config.config.assistant.gender == expected
        assert resolved_gender(engine.config) == expected

    def test_a_command_and_the_settings_screen_reach_the_same_state(
            self, engine: AarmaEngine) -> None:
        """One state, not a command-only copy of it."""
        engine.handle_text("use a male voice")
        by_command = (engine.config.config.assistant.voice_id,
                      engine.config.config.voice.style)

        engine.config.set_voice_presentation("female")
        engine.config.set_voice_presentation("male")
        by_settings = (engine.config.config.assistant.voice_id,
                       engine.config.config.voice.style)

        assert by_command == by_settings

    def test_the_reply_does_not_claim_a_change_that_did_not_happen(
            self, engine: AarmaEngine) -> None:
        engine.handle_text("use a male voice")
        response = engine.handle_text("use a male voice")
        assert "already" in response.text.lower()


# --------------------------------------------------------------------------
# Mutation: prove the agreement assertions are load-bearing
# --------------------------------------------------------------------------

class TestTheseTestsWouldCatchTheRegression:
    def test_changing_gender_without_the_voice_would_be_caught(
            self, voice_config: ConfigManager) -> None:
        """Precisely the live configuration that started this."""
        voice_config.select_voice("male_calm")
        voice_config.update_assistant(gender="female")   # the old, partial write
        assert voice_config.config.assistant.gender != resolved_gender(voice_config)
        with pytest.raises(AssertionError):
            assert voice_config.config.assistant.gender == resolved_gender(voice_config)

    def test_falling_back_to_the_opposite_profile_would_be_caught(
            self, voice_config: ConfigManager) -> None:
        voice_config.set_voice_presentation("male")
        pretend_fallback = "female_warm"
        assert get_profile(pretend_fallback).gender != \
            voice_config.config.assistant.gender
        with pytest.raises(AssertionError):
            assert get_profile(pretend_fallback).gender == \
                voice_config.config.assistant.gender

    def test_a_restart_reverting_the_profile_would_be_caught(
            self, tmp_path: Path) -> None:
        first = ConfigManager(tmp_path / "config.json")
        first.load()
        first.set_voice_presentation("male")

        fresh = ConfigManager(tmp_path / "unwritten.json")
        fresh.load()
        assert fresh.config.assistant.gender != first.config.assistant.gender
        with pytest.raises(AssertionError):
            assert fresh.config.assistant.gender == first.config.assistant.gender
