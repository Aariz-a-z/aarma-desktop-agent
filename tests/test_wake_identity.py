"""The wake word follows the configured identity, and says so honestly.

Two ideas are being defended here.

The first is that "which detector is listening" is derived from settings and
from nothing else, so renaming the assistant in the GUI, in the setup wizard or
by speaking to it all have the same effect.

The second is harder and matters more: a wake-word model is trained for one
phrase, so an identity with no model must fail *visibly*.  Quietly arming the
"Hey Aariz" detector for someone who renamed the assistant to Asma would mean
waking at another person's name, which is worse than not waking at all.  Most
of the tests below exist to pin that down.

None of these tests need a microphone, and none of them start audio.
"""

from __future__ import annotations

import threading
from typing import Iterator

import pytest

from actions.assistant import get_actions
from actions.registry import ActionContext
from config.settings import (
    ASSISTANT_PROFILES,
    ConfigManager,
    normalise_wake_phrase,
    wake_phrase_for_name,
)
from core.engine import AarmaEngine
from speech.wakeword import (
    REGISTRY,
    MissingWakeWordDetector,
    WakeWordDetector,
    WakeWordProvider,
    WakeWordRegistry,
)
from speech.wakeword_cascade import SUPPORTED_PHRASE, CascadeWakeWordDetector


def _action(name: str):
    return next(spec for spec in get_actions() if spec.name == name)


class _FakeDetector(WakeWordDetector):
    """Records what the engine asks of it, without touching the microphone."""

    name = "fake"

    def __init__(self, phrase: str, **kwargs) -> None:
        self.phrase = normalise_wake_phrase(phrase)
        self.kwargs = kwargs
        self.started_with: list = []
        self.stops = 0
        self.running = False
        self.callback = None

    def is_available(self) -> bool:
        return True

    def supports_phrase(self, phrase: str) -> bool:
        return normalise_wake_phrase(phrase) == self.phrase

    def start(self, phrase: str, callback) -> bool:
        self.started_with.append(phrase)
        self.callback = callback
        self.running = True
        return True

    def stop(self) -> None:
        self.stops += 1
        self.running = False


@pytest.fixture
def fake_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict]:
    """Replace the installed providers with two fakes: "aariz" and "iris".

    The engine is exercised against the registry *interface* rather than
    against torch and a microphone, which is the whole point of having a
    registry.  Phrase-to-model behaviour of the real cascade is covered
    separately below.
    """
    built: dict = {}
    registry = WakeWordRegistry()
    for phrase in ("hey aariz", "hey iris"):
        def factory(_phrase=phrase, **kwargs):
            built[_phrase] = _FakeDetector(_phrase, **kwargs)
            return built[_phrase]

        registry.register(WakeWordProvider(phrase=phrase, factory=factory))

    monkeypatch.setattr("core.engine._wake_registry", lambda: registry)
    yield built


@pytest.fixture(autouse=True)
def _unlocked(engine):
    """Phase 6 gates the wake loop behind a one-time "hey aariz" per session.

    Every test in this module asks which detector an *identity* gets once the
    assistant is live, so the session is unlocked before they run. The gate
    itself -- including that a fresh engine arms nothing else -- is covered in
    tests/test_wake_bootstrap.py.
    """
    engine.complete_bootstrap()
    return engine


class TestPhraseDerivation:
    def test_name_derives_the_phrase(self) -> None:
        # The phrase is the name, said on its own.
        assert wake_phrase_for_name("Nova") == "nova"
        assert wake_phrase_for_name("Asma") == "asma"
        assert wake_phrase_for_name("Friday") == "friday"

    def test_aariz_keeps_the_phrase_its_model_was_trained_on(self) -> None:
        """The one exception, and it is a fact about the model, not a taste.

        The shipped Phase-15 detector was frozen on "hey aariz". A bare-name
        phrase for that identity would leave the only validated model in the
        project unreachable.
        """
        from speech.wakeword import REGISTRY

        assert wake_phrase_for_name("Aariz") == "hey aariz"
        assert REGISTRY.provider_for(wake_phrase_for_name("Aariz")) is not None

    def test_case_and_spacing_fold_away(self) -> None:
        for written in ("Hey Aariz", "hey aariz", "HEY AARIZ", "  Hey   Aariz  "):
            assert normalise_wake_phrase(written) == "hey aariz"

    def test_an_empty_name_yields_no_phrase(self) -> None:
        # Better than "hey ", which would be a phrase nothing can detect and
        # everything might match.
        assert wake_phrase_for_name("   ") == ""
        assert normalise_wake_phrase(None) == ""

    def test_shipped_profiles_agree_with_the_rule(self) -> None:
        for profile in ASSISTANT_PROFILES.values():
            assert profile.wake_word == wake_phrase_for_name(profile.display_name)


class TestRegistry:
    def test_only_hey_aariz_ships_with_a_model(self) -> None:
        assert REGISTRY.phrases() == ("hey aariz",)

    def test_the_registered_phrase_matches_what_the_cascade_detects(self) -> None:
        # Guards against the registry and the detector drifting apart.
        assert REGISTRY.provider_for("hey aariz") is not None
        assert SUPPORTED_PHRASE == "hey aariz"
        assert CascadeWakeWordDetector().supports_phrase(SUPPORTED_PHRASE)

    def test_lookup_is_case_insensitive(self) -> None:
        assert REGISTRY.provider_for("HEY AARIZ") is REGISTRY.provider_for("hey aariz")

    def test_an_unregistered_phrase_yields_a_missing_detector(self) -> None:
        detector = REGISTRY.create("hey asma", display_name="Asma")
        assert isinstance(detector, MissingWakeWordDetector)
        assert detector.is_available() is False
        assert "Asma" in detector.unavailable_reason()
        assert "not installed" in detector.unavailable_reason()

    def test_a_missing_detector_never_fires(self) -> None:
        detector = REGISTRY.create("hey nova", display_name="Nova")
        fired: list = []
        assert detector.start("hey nova", fired.append) is False
        detector.stop()                      # must be safe before any start
        assert fired == []

    def test_a_missing_detector_supports_nothing(self) -> None:
        detector = REGISTRY.create("hey nova", display_name="Nova")
        assert detector.supports_phrase("hey nova") is False
        assert detector.supports_phrase("hey aariz") is False

    def test_a_provider_that_cannot_be_built_is_unavailable_not_fatal(self) -> None:
        registry = WakeWordRegistry()

        def explode(**kwargs):
            raise RuntimeError("model file corrupt")

        registry.register(WakeWordProvider(phrase="hey iris", factory=explode))
        detector = registry.create("hey iris", display_name="Iris")
        assert isinstance(detector, MissingWakeWordDetector)


class TestCascadeIsPhraseBound:
    def test_it_answers_only_for_its_own_phrase(self) -> None:
        detector = CascadeWakeWordDetector()
        assert detector.supports_phrase("Hey Aariz") is True
        assert detector.supports_phrase("hey asma") is False
        assert detector.supports_phrase("hey nova") is False

    def test_starting_it_on_another_name_refuses(self) -> None:
        detector = CascadeWakeWordDetector()
        fired: list = []
        assert detector.start("hey asma", fired.append) is False
        assert fired == []

    def test_a_future_model_can_bind_a_different_phrase(self) -> None:
        # The extension point: registering a validated "Hey Nova" model later
        # needs no new class and no change to the engine.
        detector = CascadeWakeWordDetector(phrase="hey nova")
        assert detector.supports_phrase("hey nova") is True
        assert detector.supports_phrase("hey aariz") is False


class TestEngineFollowsConfiguration:
    def test_the_active_phrase_comes_from_settings(self, engine: AarmaEngine) -> None:
        engine.config.update_assistant(name="Aariz", wake_word="Hey Aariz")
        assert engine.wake_phrase == "hey aariz"

    def test_the_detector_is_chosen_by_the_configured_phrase(
            self, engine: AarmaEngine, fake_registry: dict) -> None:
        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        assert engine.wake_detector is fake_registry["hey aariz"]

    def test_switching_identity_swaps_the_detector(
            self, engine: AarmaEngine, fake_registry: dict) -> None:
        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        first = engine.wake_detector
        engine.config.update_assistant(name="Iris", wake_word="hey iris")
        second = engine.wake_detector
        assert second is not first
        assert second is fake_registry["hey iris"]

    def test_the_previous_detector_is_stopped_on_a_switch(
            self, engine: AarmaEngine, fake_registry: dict) -> None:
        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        assert engine.start_wake_loop() is True
        aariz = fake_registry["hey aariz"]
        assert aariz.running is True

        engine.config.update_assistant(name="Iris", wake_word="hey iris")
        assert aariz.running is False            # not left holding the microphone
        assert aariz.stops >= 1
        assert fake_registry["hey iris"].running is True

    def test_the_loop_starts_with_the_configured_phrase(
            self, engine: AarmaEngine, fake_registry: dict) -> None:
        engine.config.update_assistant(name="Iris", wake_word="hey iris")
        assert engine.start_wake_loop() is True
        assert fake_registry["hey iris"].started_with == ["hey iris"]

    def test_an_identity_without_a_model_reports_instead_of_falling_back(
            self, engine: AarmaEngine, fake_registry: dict) -> None:
        engine.config.update_assistant(name="Nova", wake_word="hey nova")
        state = engine.wake_identity()
        assert state["available"] is False
        assert state["detector"] == "missing"
        assert "Nova" in state["reason"]
        assert engine.start_wake_loop() is False
        # The crucial part: no Aariz detector was built or armed.
        assert "hey aariz" not in fake_registry

    def test_switching_to_an_unsupported_identity_disarms_the_old_one(
            self, engine: AarmaEngine, fake_registry: dict) -> None:
        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        assert engine.start_wake_loop() is True
        engine.config.update_assistant(name="Nova", wake_word="hey nova")

        assert fake_registry["hey aariz"].running is False
        assert engine.wake_identity()["available"] is False
        # And it must not answer to the old phrase any more.
        assert engine.wake_detector.supports_phrase("hey aariz") is False

    def test_availability_survives_a_missing_phrase(self, engine: AarmaEngine) -> None:
        engine.config.update_assistant(name="Aariz", wake_word="   ")
        available, reason = engine.wake_availability()
        assert available is False
        assert reason
        assert engine.start_wake_loop() is False

    def test_stopping_before_starting_is_safe(self, engine: AarmaEngine) -> None:
        engine.stop_wake_loop()
        engine.stop_wake_loop()

    def test_shutdown_detaches_the_config_listener(
            self, engine: AarmaEngine, fake_registry: dict) -> None:
        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        engine.start_wake_loop()
        engine.shutdown()
        assert fake_registry["hey aariz"].running is False
        # A later settings change must not resurrect anything.
        engine.config.update_assistant(name="Iris", wake_word="hey iris")
        assert "hey iris" not in fake_registry


class TestDuplicateWakePrevention:
    def test_a_second_wake_during_a_turn_is_ignored(
            self, engine: AarmaEngine, fake_registry: dict,
            monkeypatch: pytest.MonkeyPatch) -> None:
        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        turns: list = []
        release = threading.Event()

        def slow_turn(event) -> None:
            turns.append(event)
            release.wait(timeout=5.0)
            with engine._wake_lock:
                engine._wake_session = None

        monkeypatch.setattr(engine, "_run_voice_turn", slow_turn)
        assert engine.start_wake_loop() is True
        callback = fake_registry["hey aariz"].callback

        callback("first")
        # Wait for the turn thread to actually be running before the second.
        for _ in range(200):
            if turns:
                break
            threading.Event().wait(0.01)
        callback("second")
        release.set()
        session = engine._wake_session
        if session is not None:
            session.join(timeout=5.0)

        assert turns == ["first"]              # the second wake was dropped


class TestConversationalRename:
    def test_rename_persists_through_the_settings_layer(
            self, action_context: ActionContext, tmp_path) -> None:
        spec = _action("set_assistant_name")
        result = spec.handler(action_context, {"name": "Nova"})
        assert result.ok
        assert action_context.config.config.assistant.name == "Nova"
        assert action_context.config.config.assistant.wake_word == "nova"

        # Reload from disk: an in-memory-only rename would not survive this.
        reloaded = ConfigManager(action_context.config.path)
        reloaded.load()
        assert reloaded.config.assistant.name == "Nova"
        assert reloaded.config.assistant.wake_word == "nova"

    def test_rename_says_plainly_that_no_model_exists(
            self, action_context: ActionContext) -> None:
        result = _action("set_assistant_name").handler(action_context, {"name": "Nova"})
        assert result.ok
        message = result.message.lower()
        # The wording may change; what must not is that it says the model is
        # missing and says what to do about it.
        assert "wake word" in message
        assert "enrol" in message or "teach" in message

    def test_rename_to_aariz_reports_that_it_will_wake(
            self, action_context: ActionContext) -> None:
        result = _action("set_assistant_name").handler(action_context, {"name": "Aariz"})
        assert result.ok
        assert "hey aariz" in result.message.lower()
        assert "wake me" in result.message.lower()

    def test_a_custom_wake_word_is_not_overwritten_by_a_rename(
            self, action_context: ActionContext) -> None:
        action_context.config.update_assistant(wake_word="computer wake up")
        _action("set_assistant_name").handler(action_context, {"name": "Nova"})
        assert action_context.config.config.assistant.wake_word == "computer wake up"

    def test_setting_the_wake_word_directly_is_normalised(
            self, action_context: ActionContext) -> None:
        result = _action("set_wake_word").handler(action_context, {"phrase": "  Hey   Aariz "})
        assert result.ok
        assert action_context.config.config.assistant.wake_word == "hey aariz"

    def test_the_wake_word_action_no_longer_claims_phase_two(
            self, action_context: ActionContext) -> None:
        result = _action("set_wake_word").handler(action_context, {"phrase": "hey aariz"})
        assert "phase 2" not in result.message.lower()

    def test_an_empty_rename_is_rejected(self, action_context: ActionContext) -> None:
        before = action_context.config.config.assistant.name
        result = _action("set_assistant_name").handler(action_context, {"name": "  "})
        assert not result.ok
        assert action_context.config.config.assistant.name == before


class TestEngineIdentityReport:
    def test_it_names_the_installed_phrases(self, engine: AarmaEngine) -> None:
        state = engine.wake_identity()
        assert state["installed_phrases"] == ("hey aariz",)

    def test_the_default_identity_is_reported_as_unavailable(
            self, engine: AarmaEngine) -> None:
        # AARMA ships as "Asma", and there is no Asma model.  This is a real
        # out-of-the-box state, not a contrived one.
        assert engine.config.config.assistant.name == "Asma"
        state = engine.wake_identity()
        assert state["phrase"] == "asma"
        assert state["available"] is False
        assert "Asma" in state["reason"]
