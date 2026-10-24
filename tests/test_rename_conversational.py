"""Renaming the assistant by talking to it.

The assistant's name is not permanent and is not special. "Asma, from now on
I'll call you Friday" has to change who it is, immediately and durably, through
the same settings layer the GUI writes to.

The hard part is not the rename. It is that the *identity* and the *wake-word
model* are separate things, and after a rename they are usually out of step:
Friday is the assistant, and nothing has been trained to hear "Friday" yet.
That is a valid state. What must never happen is the old identity's detector
staying armed — an assistant called Friday that still wakes to "Asma" is
answering to a name it no longer has.

The other half is restraint: "my friend's name is Friday" is a remark, not an
instruction, and must leave the assistant alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import pytest

from config.settings import ConfigManager, normalise_wake_phrase, wake_phrase_for_name
from core.engine import AarmaEngine
from speech.wakeword import (
    MissingWakeWordDetector,
    WakeWordProvider,
    WakeWordRegistry,
)
from speech.wakeword_store import CustomWakeWordStore


def _write_model(store: CustomWakeWordStore, tmp_path: Path, *, phrase: str,
                 name: str, stage2: float = 0.4):
    """Install a real, loadable custom model for `phrase`."""
    import numpy as np
    import torch

    from speech.wakeword_cascade import _FarFieldVerifier

    build = tmp_path / f"build-{normalise_wake_phrase(phrase).replace(' ', '_')}"
    build.mkdir(parents=True, exist_ok=True)
    verifier, reference = build / "verifier.pt", build / "reference.npz"
    model = _FarFieldVerifier()
    torch.save({"state_dict": model.state_dict(), "seed": 1, "n_mels": 40,
                "frames": 98,
                "params": sum(p.numel() for p in model.parameters())}, verifier)
    np.savez_compressed(reference, embeddings=np.zeros((10, 2048), dtype=np.float32))
    return store.install(phrase=phrase, display_name=name, verifier=verifier,
                         reference=reference, stage1_threshold=0.70,
                         stage2_threshold=stage2, enrollment_count=25,
                         validation={"checks": []}, provenance={"positives": "test"})



@pytest.fixture(autouse=True)
def _unlocked(engine):
    """Unlock the Phase 6 session gate; these tests are about renaming.

    Renaming moves the casual wake phrase, which only matters once the session
    is live. The gate is covered in tests/test_wake_bootstrap.py.
    """
    engine.complete_bootstrap()
    return engine

class _Detector(MissingWakeWordDetector):
    """A shipped detector that records whether it is holding the microphone."""

    name = "shipped-fake"

    def __init__(self, phrase: str, **kwargs) -> None:
        super().__init__(phrase)
        self.phrase = normalise_wake_phrase(phrase)
        self.running = False

    def is_available(self) -> bool:
        return True

    def supports_phrase(self, phrase: str) -> bool:
        return normalise_wake_phrase(phrase) == self.phrase

    def start(self, phrase, callback) -> bool:
        self.running = True
        return True

    def stop(self) -> None:
        self.running = False


@pytest.fixture
def renamable(engine: AarmaEngine, tmp_path: Path,
              monkeypatch: pytest.MonkeyPatch) -> Iterator[dict]:
    """An engine starting as Asma, with one shipped detector and a real store.

    "hey aariz" stands in for the shipped, validated model; the store is empty
    until a test enrols something.
    """
    store = CustomWakeWordStore(tmp_path / "custom")
    registry = WakeWordRegistry(custom_store=store)
    built: List[_Detector] = []

    def factory(**kwargs):
        built.append(_Detector("hey aariz", **kwargs))
        return built[-1]

    registry.register(WakeWordProvider(phrase="hey aariz", factory=factory))
    monkeypatch.setattr("core.engine._wake_registry", lambda: registry)
    monkeypatch.setattr("speech.wakeword.REGISTRY", registry)
    monkeypatch.setattr("speech.wakeword_store.STORE", store)

    engine.config.update_assistant(name="Asma", wake_word=wake_phrase_for_name("Asma"),
                                   profile_key="asma")
    yield {"engine": engine, "store": store, "registry": registry, "built": built,
           "tmp_path": tmp_path}


def _say(engine: AarmaEngine, text: str):
    return engine.handle_text(text, source="text")


# ------------------------------------------------------------------ 1, 5, 9

class TestRenamingByTalking:
    def test_from_now_on_ill_call_you_friday(self, renamable: dict) -> None:
        engine = renamable["engine"]
        assert engine.config.config.assistant.name == "Asma"

        response = _say(engine, "Asma, from now on I'll call you Friday.")

        assert response.ok
        assert engine.config.config.assistant.name == "Friday"
        assert engine.config.config.assistant.wake_word == "friday"
        assert engine.wake_phrase == "friday"

    def test_the_reply_says_who_it_is_now(self, renamable: dict) -> None:
        response = _say(renamable["engine"], "from now on I'll call you Friday")
        assert "Friday" in response.text

    def test_change_your_name_to_athena(self, renamable: dict) -> None:
        engine = renamable["engine"]
        assert _say(engine, "Change your name to Athena.").ok
        assert engine.config.config.assistant.name == "Athena"
        assert engine.wake_phrase == "athena"

    def test_the_phrase_is_the_name_not_hey_the_name(self, renamable: dict) -> None:
        engine = renamable["engine"]
        _say(engine, "I want to call you Boss")
        assert engine.wake_phrase == "boss"
        assert not engine.wake_phrase.startswith("hey ")

    def test_a_two_word_name_survives(self, renamable: dict) -> None:
        engine = renamable["engine"]
        _say(engine, "from now on your name is My Assistant")
        assert engine.config.config.assistant.name == "My Assistant"
        assert engine.wake_phrase == "my assistant"

    def test_it_persists_across_a_restart(self, renamable: dict) -> None:
        engine = renamable["engine"]
        _say(engine, "Asma, from now on I'll call you Friday.")

        # Everything the next launch would do: read the same file again.
        reloaded = ConfigManager(engine.config.path)
        reloaded.load()
        assert reloaded.config.assistant.name == "Friday"
        assert reloaded.config.assistant.wake_word == "friday"


# --------------------------------------------------------------------- 7, 8

class TestRemarksAreNotInstructions:
    @pytest.mark.parametrize("remark", [
        "My friend's name is Friday.",
        "I met someone called Friday.",
        "I met someone called Nova.",
        "Her name is Nova.",
        "I'll call you back.",
        "I'll call you tomorrow",
        "I'll call you on Friday",
        "What is your name?",
    ])
    def test_it_does_not_rename(self, renamable: dict, remark: str) -> None:
        engine = renamable["engine"]
        before = engine.config.config.assistant.name
        _say(engine, remark)
        assert engine.config.config.assistant.name == before
        assert engine.wake_phrase == wake_phrase_for_name(before)


# ------------------------------------------------------------------------ 2

class TestRenamingToAnIdentityWithNoModel:
    def test_the_detector_is_missing_not_the_old_one(self, renamable: dict) -> None:
        engine = renamable["engine"]
        _say(engine, "from now on I'll call you Friday")

        state = engine.wake_identity()
        assert state["phrase"] == "friday"
        assert state["available"] is False
        assert state["enrolled"] is False
        assert state["can_enrol"] is True
        assert engine.wake_detector.name == "missing"
        assert engine.start_wake_loop() is False

    def test_the_old_identitys_detector_is_never_armed(self, renamable: dict) -> None:
        engine = renamable["engine"]
        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        assert engine.start_wake_loop() is True
        aariz = renamable["built"][0]
        assert aariz.running is True

        _say(engine, "from now on I'll call you Friday")

        assert aariz.running is False
        assert engine.wake_detector.supports_phrase("hey aariz") is False
        assert engine.wake_detector.supports_phrase("friday") is False  # nothing yet

    def test_the_reply_does_not_claim_voice_wake_works(self, renamable: dict) -> None:
        response = _say(renamable["engine"], "from now on I'll call you Friday")
        lowered = response.text.lower()
        assert "haven't learned" in lowered
        assert "voice wake-up is off" in lowered

    def test_typing_still_works_afterwards(self, renamable: dict) -> None:
        engine = renamable["engine"]
        _say(engine, "from now on I'll call you Friday")
        assert _say(engine, "what can you do").ok


# ------------------------------------------------------------------------ 3

class TestRenamingToAnIdentityThatHasAModel:
    def test_the_new_detector_becomes_active(self, renamable: dict) -> None:
        engine, store = renamable["engine"], renamable["store"]
        _write_model(store, renamable["tmp_path"], phrase="friday", name="Friday")

        _say(engine, "from now on I'll call you Friday")

        state = engine.wake_identity()
        assert state["phrase"] == "friday"
        assert state["enrolled"] is True
        assert engine.wake_detector.supports_phrase("friday") is True

    def test_the_reply_says_how_to_wake_it(self, renamable: dict) -> None:
        store = renamable["store"]
        _write_model(store, renamable["tmp_path"], phrase="friday", name="Friday")
        response = _say(renamable["engine"], "from now on I'll call you Friday")
        assert "friday" in response.text.lower()
        assert "wake me" in response.text.lower()

    def test_it_uses_the_stored_threshold(self, renamable: dict) -> None:
        store = renamable["store"]
        _write_model(store, renamable["tmp_path"], phrase="friday", name="Friday",
                     stage2=0.37)
        _say(renamable["engine"], "from now on I'll call you Friday")
        detector = renamable["engine"].wake_detector
        assert detector._stage2_threshold == pytest.approx(0.37)


# ------------------------------------------------------------------------ 4

class TestRenamingAgain:
    def test_friday_stops_and_nova_takes_over(self, renamable: dict) -> None:
        engine, store = renamable["engine"], renamable["store"]
        _write_model(store, renamable["tmp_path"], phrase="friday", name="Friday")
        _write_model(store, renamable["tmp_path"], phrase="nova", name="Nova")

        _say(engine, "from now on I'll call you Friday")
        friday = engine.wake_detector
        assert friday.supports_phrase("friday") is True

        _say(engine, "Friday, from now on I'll call you Nova.")

        nova = engine.wake_detector
        assert engine.config.config.assistant.name == "Nova"
        assert nova is not friday
        assert nova.supports_phrase("nova") is True
        assert nova.supports_phrase("friday") is False

    def test_renaming_to_something_untrained_disarms_the_working_one(
            self, renamable: dict) -> None:
        engine, store = renamable["engine"], renamable["store"]
        _write_model(store, renamable["tmp_path"], phrase="friday", name="Friday")
        _say(engine, "from now on I'll call you Friday")
        assert engine.wake_identity()["available"] is True

        _say(engine, "Friday, from now on I'll call you Nova.")

        assert engine.wake_identity()["available"] is False
        assert engine.wake_detector.supports_phrase("friday") is False


# ------------------------------------------------------------------------ 6

class TestRenamingToABuiltInIdentity:
    def test_a_shipped_model_is_used_when_one_exists(self, renamable: dict) -> None:
        engine = renamable["engine"]
        _say(engine, "change your name to Aariz")

        assert engine.config.config.assistant.name == "Aariz"
        # Aariz keeps the phrase its frozen model was trained on.
        assert engine.wake_phrase == "hey aariz"
        assert engine.wake_identity()["available"] is True
        assert engine.start_wake_loop() is True

    def test_a_built_in_name_without_a_model_is_still_unavailable(
            self, renamable: dict) -> None:
        engine = renamable["engine"]
        _say(engine, "change your name to Zara")
        assert engine.wake_phrase == "zara"
        assert engine.wake_identity()["available"] is False
        assert engine.wake_detector.name == "missing"


# ----------------------------------------------------------------------- 10

class TestEnrollmentFollowsTheIdentity:
    def test_it_asks_for_the_bare_name(self, renamable: dict) -> None:
        from speech.wakeword_enrollment import enrollment_plan

        engine = renamable["engine"]
        _say(engine, "from now on I'll call you Friday")

        prompts = enrollment_plan(engine.wake_phrase)
        assert all(p.phrase == "friday" for p in prompts)
        assert not any(p.phrase.startswith("hey ") for p in prompts)

    def test_the_store_directory_follows_the_bare_name(self, renamable: dict) -> None:
        from speech.wakeword_store import slug_for_phrase

        engine = renamable["engine"]
        _say(engine, "from now on I'll call you Friday")
        assert slug_for_phrase(engine.wake_phrase) == "friday"


class TestNoIdentityIsHardCoded:
    @pytest.mark.parametrize("name", ["Friday", "Boss", "Jarvis", "Athena",
                                      "Computer", "Alex"])
    def test_any_name_can_become_the_assistant(self, renamable: dict,
                                               name: str) -> None:
        engine = renamable["engine"]
        assert _say(engine, f"change your name to {name}").ok
        assert engine.config.config.assistant.name == name
        assert engine.wake_phrase == name.lower()

    def test_the_engine_never_names_a_detector_class(self) -> None:
        """The engine resolves detectors through the registry and nothing else."""
        import inspect

        import core.engine as engine_module

        source = inspect.getsource(engine_module)
        for hard_coded in ("CascadeWakeWordDetector", '"aariz"', "'aariz'",
                           '"asma"', "'asma'"):
            assert hard_coded not in source, f"{hard_coded} is hard-coded in the engine"


class TestEveryRequiredPhrasing:
    """The full set of phrasings the product requires, end to end."""

    @pytest.mark.parametrize("said", [
        "Change your name to Friday.",
        "From now on I'll call you Friday.",
        "I'll call you Friday from now on.",
        "I'm going to call you Friday.",
        "Your name is Friday now.",
        "I want to call you Friday.",
        "Let's call you Friday.",
        "From now on, you're Friday.",
        "Can I call you Friday?",
        "Asma, from now on I'll call you Friday.",
    ])
    def test_it_renames(self, renamable: dict, said: str) -> None:
        engine = renamable["engine"]
        assert _say(engine, said).ok
        assert engine.config.config.assistant.name == "Friday"
        assert engine.wake_phrase == "friday"

    @pytest.mark.parametrize("said", [
        "My friend's name is Athena.",
        "I'll call you back tomorrow.",
        "I'll call you on Friday.",
        "I am meeting Friday tomorrow.",
        "Friday is my friend's name.",
        "Call you later.",
        "I'll call you when I reach home.",
    ])
    def test_it_does_not_rename(self, renamable: dict, said: str) -> None:
        engine = renamable["engine"]
        before = engine.config.config.assistant.name
        _say(engine, said)
        assert engine.config.config.assistant.name == before


class TestTheBareNameRule:
    """The phrase is the name. Nothing prepends "hey" to a custom identity."""

    @pytest.mark.parametrize("name,phrase", [
        ("Friday", "friday"), ("Jarvis", "jarvis"), ("Boss", "boss"),
        ("Athena", "athena"), ("Computer", "computer"),
        ("My Assistant", "my assistant"),
    ])
    def test_custom_names_get_the_bare_phrase(self, renamable: dict,
                                              name: str, phrase: str) -> None:
        engine = renamable["engine"]
        _say(engine, f"change your name to {name}")
        assert engine.wake_phrase == phrase
        assert not engine.wake_phrase.startswith("hey ")

    def test_aariz_keeps_its_legacy_phrase(self, renamable: dict) -> None:
        # Its frozen model hears "hey aariz" and nothing else.
        engine = renamable["engine"]
        _say(engine, "change your name to Aariz")
        assert engine.wake_phrase == "hey aariz"

    def test_only_aariz_is_an_exception(self) -> None:
        from config.settings import LEGACY_WAKE_PHRASES

        assert set(LEGACY_WAKE_PHRASES) == {"aariz"}


class TestAnOldPrefixedModelIsNeverReused:
    def test_a_hey_boss_model_does_not_serve_bare_boss(
            self, renamable: dict) -> None:
        """models/custom/hey_boss/ must not answer for the identity "Boss".

        They are different phrases and a detector trained on one does not
        detect the other; silently substituting would wake the assistant at
        something the user never asked for.
        """
        engine, store = renamable["engine"], renamable["store"]
        _write_model(store, renamable["tmp_path"], phrase="hey boss", name="Boss")

        _say(engine, "change your name to Boss")

        assert engine.wake_phrase == "boss"
        state = engine.wake_identity()
        assert state["available"] is False
        assert state["enrolled"] is False
        assert engine.wake_detector.name == "missing"
        assert engine.wake_detector.supports_phrase("hey boss") is False

    def test_the_matching_bare_model_is_used(self, renamable: dict) -> None:
        engine, store = renamable["engine"], renamable["store"]
        _write_model(store, renamable["tmp_path"], phrase="boss", name="Boss")
        _say(engine, "change your name to Boss")
        assert engine.wake_identity()["available"] is True
        assert engine.wake_detector.supports_phrase("boss") is True


class TestTheFullRenameChain:
    def test_asma_friday_jarvis_nova_friday(self, renamable: dict) -> None:
        """Repeated renaming, with the previous identity retired each time."""
        engine, store = renamable["engine"], renamable["store"]
        _write_model(store, renamable["tmp_path"], phrase="friday", name="Friday")

        seen = []
        for said, expected_name, expected_phrase in [
                ("Asma, from now on I'll call you Friday.", "Friday", "friday"),
                ("Change your name to Jarvis.", "Jarvis", "jarvis"),
                ("Change your name to Nova.", "Nova", "nova"),
                ("From now on, you're Friday.", "Friday", "friday"),
        ]:
            _say(engine, said)
            assert engine.config.config.assistant.name == expected_name
            assert engine.wake_phrase == expected_phrase
            detector = engine.wake_detector
            # Whatever it is now, it answers only for the current phrase.
            for other in ("asma", "friday", "jarvis", "nova"):
                if other != expected_phrase:
                    assert detector.supports_phrase(other) is False
            seen.append((expected_name, detector.name))

        # Friday has a model at both ends of the chain; the middle two do not.
        assert seen[0][1] == "cascade"
        assert seen[1][1] == "missing"
        assert seen[2][1] == "missing"
        assert seen[3][1] == "cascade"
