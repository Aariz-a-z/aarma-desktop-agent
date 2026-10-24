"""Custom wake words: storage, resolution, enrollment and the quality gate.

The thing being defended here is a refusal.  Enrolling "Hey Friday" must either
produce a detector that was measured on recordings it never trained on, or
produce nothing at all — and in the second case the assistant has to say so
rather than quietly leaving the Aariz model armed under a new name.

None of these tests open a microphone or train a real model.  Training is
exercised through its own seams: the store is given genuine checkpoints (cheap
to build, since an untrained model is still the right shape), the gate is given
a scoring function whose answers the test chooses, and the workflow is given a
stub builder.  What that leaves untested is whether training *works*, which is
a measurement and not something a unit test can assert.
"""

from __future__ import annotations

import json
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
from speech.wakeword_store import (
    CustomWakeWordError,
    CustomWakeWordStore,
    is_safe_slug,
    slug_for_phrase,
    validate_checkpoint,
    validate_reference,
)


# --------------------------------------------------------------- fixtures

def _write_checkpoint(path: Path, *, frames: int = 98, n_mels: int = 40) -> Path:
    """A real, loadable checkpoint of the right architecture.

    Untrained: these tests are about storage and resolution, and an untrained
    model has exactly the same shape as a trained one.
    """
    import torch

    from speech.wakeword_cascade import _FarFieldVerifier

    model = _FarFieldVerifier()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seed": 1, "n_mels": n_mels,
                "frames": frames,
                "params": sum(p.numel() for p in model.parameters())}, path)
    return path


def _write_reference(path: Path, rows: int = 12) -> Path:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=np.zeros((rows, 2048), dtype=np.float32))
    return path


@pytest.fixture
def store(tmp_path: Path) -> CustomWakeWordStore:
    return CustomWakeWordStore(tmp_path / "custom")


@pytest.fixture
def artefacts(tmp_path: Path) -> dict:
    return {"verifier": _write_checkpoint(tmp_path / "build" / "verifier.pt"),
            "reference": _write_reference(tmp_path / "build" / "reference.npz")}


def _install(store: CustomWakeWordStore, artefacts: dict, *, phrase="hey friday",
             name="Friday", stage2=0.4, count=15):
    return store.install(
        phrase=phrase, display_name=name,
        verifier=artefacts["verifier"], reference=artefacts["reference"],
        stage1_threshold=0.70, stage2_threshold=stage2, enrollment_count=count,
        validation={"checks": []}, provenance={"positives": "test"})


# ------------------------------------------------------------------- naming


@pytest.fixture(autouse=True)
def _unlocked(engine):
    """Unlock the Phase 6 session gate; these tests are about enrolled models.

    A fresh engine arms only "hey aariz" until it hears it, which would
    otherwise make every assertion here about the bootstrap instead of about
    the custom detector under test. The gate is covered in
    tests/test_wake_bootstrap.py.
    """
    engine.complete_bootstrap()
    return engine

class TestNaming:
    def test_a_custom_name_makes_the_expected_phrase(self) -> None:
        assert wake_phrase_for_name("Friday") == "friday"
        assert wake_phrase_for_name("JARVIS") == "jarvis"
        assert wake_phrase_for_name("Nova") == "nova"
        assert wake_phrase_for_name("My Assistant") == "my assistant"

    def test_the_phrase_makes_one_folder_however_it_is_typed(self) -> None:
        for written in ("Hey Friday", "hey friday", "HEY  FRIDAY", "  Hey Friday "):
            assert slug_for_phrase(written) == "hey_friday"

    def test_unusable_names_do_not_make_folders(self) -> None:
        assert slug_for_phrase("   ") == ""
        assert slug_for_phrase("...") == ""
        assert is_safe_slug("") is False
        assert is_safe_slug("../escape") is False
        assert is_safe_slug("hey_friday") is True

    def test_a_hostile_name_lands_inside_the_store_or_nowhere(self, store) -> None:
        """Separators are folded away, so traversal cannot address a parent.

        The guarantee is not that these raise -- "../../etc/passwd" is a
        perfectly ordinary thing for someone to type -- but that whatever comes
        back is inside the store.
        """
        root = store.root.resolve()
        for hostile in ("../../etc/passwd", "/absolute", r"..\..\windows"):
            resolved = store.directory_for(hostile)
            assert root in resolved.parents
            assert ".." not in resolved.parts

    def test_a_name_that_folds_away_to_nothing_is_refused(self, store) -> None:
        for empty in ("  ", "...", "", "///"):
            with pytest.raises(CustomWakeWordError):
                store.directory_for(empty)


# -------------------------------------------------------------------- store

class TestStore:
    def test_an_empty_store_has_nothing_in_it(self, store) -> None:
        assert store.list() == ()
        assert store.get("hey friday") is None

    def test_install_then_read_back(self, store, artefacts) -> None:
        model = _install(store, artefacts)
        assert model.phrase == "hey friday"
        assert model.display_name == "Friday"
        assert model.model_version == 1
        assert model.is_intact()
        assert store.get("HEY FRIDAY").slug == "hey_friday"
        assert [m.phrase for m in store.list()] == ["hey friday"]

    def test_the_threshold_is_stored_and_read_back(self, store, artefacts) -> None:
        _install(store, artefacts, stage2=0.37)
        model = store.get("hey friday")
        assert model.stage2_threshold == pytest.approx(0.37)
        assert model.stage1_threshold == pytest.approx(0.70)

    def test_metadata_is_json_and_carries_no_paths(self, store, artefacts) -> None:
        model = _install(store, artefacts)
        raw = json.loads(model.metadata_path.read_text(encoding="utf-8"))
        assert raw["wake_phrase"] == "hey friday"
        assert raw["schema_version"] == 1
        # Paths are derived from the slug, never trusted from the file.
        assert not any(isinstance(v, str) and ("\\\\" in v or "/" in v)
                       for k, v in raw.items() if k != "detector_type")

    def test_re_enrollment_bumps_the_version(self, store, artefacts) -> None:
        assert _install(store, artefacts).model_version == 1
        assert _install(store, artefacts, stage2=0.5).model_version == 2
        assert store.get("hey friday").stage2_threshold == pytest.approx(0.5)

    def test_a_failed_install_leaves_the_working_model_alone(
            self, store, artefacts, tmp_path: Path) -> None:
        _install(store, artefacts, stage2=0.42)
        broken = tmp_path / "broken" / "verifier.pt"
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_bytes(b"not a checkpoint")

        with pytest.raises(CustomWakeWordError):
            store.install(phrase="hey friday", display_name="Friday",
                          verifier=broken, reference=artefacts["reference"],
                          stage1_threshold=0.7, stage2_threshold=0.9,
                          enrollment_count=15, validation={}, provenance={})

        survivor = store.get("hey friday")
        assert survivor is not None
        assert survivor.model_version == 1
        assert survivor.stage2_threshold == pytest.approx(0.42)

    def test_metadata_that_disagrees_with_its_folder_is_ignored(
            self, store, artefacts) -> None:
        model = _install(store, artefacts)
        raw = json.loads(model.metadata_path.read_text(encoding="utf-8"))
        raw["wake_phrase"] = "hey aariz"          # claim to be the shipped one
        model.metadata_path.write_text(json.dumps(raw), encoding="utf-8")
        assert store.get("hey friday") is None
        assert store.list() == ()

    def test_thresholds_out_of_range_are_ignored(self, store, artefacts) -> None:
        model = _install(store, artefacts)
        raw = json.loads(model.metadata_path.read_text(encoding="utf-8"))
        raw["stage2_threshold"] = 0.0
        model.metadata_path.write_text(json.dumps(raw), encoding="utf-8")
        assert store.get("hey friday") is None

    def test_a_missing_file_makes_the_model_not_intact(self, store, artefacts) -> None:
        model = _install(store, artefacts)
        model.verifier_path.unlink()
        assert store.get("hey friday").is_intact() is False
        assert store.list() == ()

    def test_remove(self, store, artefacts) -> None:
        _install(store, artefacts)
        assert store.remove("hey friday") is True
        assert store.get("hey friday") is None
        assert store.remove("hey friday") is False


class TestCheckpointSafety:
    def test_a_valid_checkpoint_reports_its_shape(self, artefacts) -> None:
        info = validate_checkpoint(artefacts["verifier"])
        assert info["n_mels"] == 40
        assert info["frames"] == 98
        assert info["parameters"] > 0

    def test_rubbish_is_refused(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"\\x00\\x01\\x02not-a-model")
        with pytest.raises(CustomWakeWordError):
            validate_checkpoint(bad)

    def test_a_checkpoint_for_another_front_end_is_refused(
            self, tmp_path: Path) -> None:
        wrong = _write_checkpoint(tmp_path / "wrong.pt", n_mels=64)
        with pytest.raises(CustomWakeWordError):
            validate_checkpoint(wrong)

    def test_a_nonsense_frame_count_is_refused(self, tmp_path: Path) -> None:
        wrong = _write_checkpoint(tmp_path / "frames.pt", frames=0)
        with pytest.raises(CustomWakeWordError):
            validate_checkpoint(wrong)

    def test_a_pickled_reference_is_refused(self, tmp_path: Path) -> None:
        import numpy as np

        path = tmp_path / "pickled.npz"
        np.savez(path, embeddings=np.array([{"not": "an array"}], dtype=object))
        with pytest.raises(CustomWakeWordError):
            validate_reference(path)


# ----------------------------------------------------------------- registry

class TestRegistryResolvesCustomModels:
    def test_an_enrolled_phrase_becomes_available(self, store, artefacts) -> None:
        registry = WakeWordRegistry(custom_store=store)
        assert registry.provider_for("hey friday") is None

        _install(store, artefacts)
        provider = registry.provider_for("hey friday")
        assert provider is not None
        assert provider.phrase == "hey friday"
        assert "hey friday" in registry.phrases()

    def test_the_detector_it_builds_carries_the_stored_threshold(
            self, store, artefacts) -> None:
        _install(store, artefacts, stage2=0.33)
        registry = WakeWordRegistry(custom_store=store)
        detector = registry.create("hey friday", display_name="Friday")
        assert detector.name == "cascade"
        assert detector._stage2_threshold == pytest.approx(0.33)
        assert detector._model_path == store.get("hey friday").verifier_path

    def test_it_answers_only_for_its_own_phrase(self, store, artefacts) -> None:
        _install(store, artefacts)
        registry = WakeWordRegistry(custom_store=store)
        detector = registry.create("hey friday")
        assert detector.supports_phrase("hey friday") is True
        assert detector.supports_phrase("hey aariz") is False

    def test_an_unenrolled_phrase_still_reports_missing(self, store) -> None:
        registry = WakeWordRegistry(custom_store=store)
        detector = registry.create("hey jarvis", display_name="Jarvis")
        assert isinstance(detector, MissingWakeWordDetector)
        assert "Jarvis" in detector.unavailable_reason()

    def test_a_shipped_model_wins_over_an_enrolled_one(self, store, artefacts) -> None:
        # The validated model is the one whose behaviour is known.
        _install(store, artefacts, phrase="hey aariz", name="Aariz")
        registry = WakeWordRegistry(custom_store=store)
        shipped = WakeWordProvider(phrase="hey aariz",
                                   factory=lambda **kw: MissingWakeWordDetector("x"),
                                   label="shipped")
        registry.register(shipped)
        assert registry.provider_for("hey aariz") is shipped
        assert registry.is_builtin("hey aariz") is True
        assert registry.is_builtin("hey friday") is False

    def test_a_damaged_model_is_not_offered(self, store, artefacts) -> None:
        model = _install(store, artefacts)
        model.reference_path.unlink()
        registry = WakeWordRegistry(custom_store=store)
        assert registry.provider_for("hey friday") is None
        assert registry.custom_model("hey friday") is None


# ------------------------------------------------------------------- engine

@pytest.fixture
def custom_engine(engine: AarmaEngine, tmp_path: Path,
                  monkeypatch: pytest.MonkeyPatch) -> Iterator[dict]:
    """An engine whose registry has one shipped fake and a real custom store."""
    store = CustomWakeWordStore(tmp_path / "custom")
    registry = WakeWordRegistry(custom_store=store)
    built: List = []

    class _Fake(MissingWakeWordDetector):
        name = "shipped-fake"

        def __init__(self, phrase="hey aariz", **kwargs) -> None:
            super().__init__(phrase)
            self.kwargs = kwargs
            self.running = False
            built.append(self)

        def is_available(self) -> bool:
            return True

        def supports_phrase(self, phrase: str) -> bool:
            return normalise_wake_phrase(phrase) == "hey aariz"

        def start(self, phrase, callback) -> bool:
            self.running = True
            return True

        def stop(self) -> None:
            self.running = False

    registry.register(WakeWordProvider(phrase="hey aariz",
                                       factory=lambda **kw: _Fake(**kw)))
    monkeypatch.setattr("core.engine._wake_registry", lambda: registry)
    yield {"engine": engine, "store": store, "registry": registry, "built": built}


class TestEngineWithCustomModels:
    def test_an_unenrolled_identity_is_unavailable_and_offers_enrollment(
            self, custom_engine: dict) -> None:
        engine = custom_engine["engine"]
        engine.config.update_assistant(name="Friday", wake_word="hey friday")
        state = engine.wake_identity()
        assert state["available"] is False
        assert state["enrolled"] is False
        assert state["can_enrol"] is True
        assert "Friday" in state["reason"]
        assert engine.start_wake_loop() is False
        # Nothing built the shipped detector under a different name.
        assert custom_engine["built"] == []

    def test_a_shipped_identity_cannot_be_re_enrolled(self, custom_engine: dict) -> None:
        engine = custom_engine["engine"]
        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        assert engine.wake_identity()["can_enrol"] is False

    def test_enrolling_makes_the_identity_available(
            self, custom_engine: dict, artefacts: dict) -> None:
        engine, store = custom_engine["engine"], custom_engine["store"]
        engine.config.update_assistant(name="Friday", wake_word="hey friday")
        assert engine.wake_identity()["available"] is False

        _install(store, artefacts, stage2=0.31)
        engine._invalidate_wake_detector()
        state = engine.wake_identity()
        assert state["enrolled"] is True
        assert state["enrolled_version"] == 1
        assert state["enrollment_count"] == 15
        assert state["can_enrol"] is True             # re-enrollment stays open

    def test_switching_to_a_custom_identity_stops_the_shipped_detector(
            self, custom_engine: dict, artefacts: dict) -> None:
        engine, store = custom_engine["engine"], custom_engine["store"]
        _install(store, artefacts)
        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        assert engine.start_wake_loop() is True
        shipped = custom_engine["built"][0]
        assert shipped.running is True

        engine.config.update_assistant(name="Friday", wake_word="hey friday")
        assert shipped.running is False
        assert engine.wake_detector.supports_phrase("hey aariz") is False
        assert engine.wake_detector.supports_phrase("hey friday") is True

    def test_the_wrong_phrase_never_wakes_a_custom_detector(
            self, custom_engine: dict, artefacts: dict) -> None:
        engine, store = custom_engine["engine"], custom_engine["store"]
        _install(store, artefacts)
        engine.config.update_assistant(name="Friday", wake_word="hey friday")
        detector = engine.wake_detector
        fired: List = []
        assert detector.start("hey aariz", fired.append) is False
        assert fired == []

    def test_removing_an_enrolled_model_returns_to_unavailable(
            self, custom_engine: dict, artefacts: dict,
            monkeypatch: pytest.MonkeyPatch) -> None:
        engine, store = custom_engine["engine"], custom_engine["store"]
        monkeypatch.setattr("speech.wakeword_store.STORE", store)
        _install(store, artefacts)
        engine.config.update_assistant(name="Friday", wake_word="hey friday")
        assert engine.wake_identity()["enrolled"] is True

        assert engine.remove_wake_word() is True
        assert engine.wake_identity()["enrolled"] is False
        assert engine.wake_identity()["available"] is False


class TestMicrophoneIsGivenBackAndTakenBack:
    def test_pausing_restores_a_running_loop(self, custom_engine: dict) -> None:
        engine = custom_engine["engine"]
        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        assert engine.start_wake_loop() is True
        with engine.wake_paused():
            assert engine._wake_running is False
            assert all(not d.running for d in custom_engine["built"])
        assert engine._wake_running is True

    def test_pausing_restores_even_when_the_body_raises(
            self, custom_engine: dict) -> None:
        engine = custom_engine["engine"]
        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        engine.start_wake_loop()
        with pytest.raises(RuntimeError):
            with engine.wake_paused():
                raise RuntimeError("enrollment blew up")
        assert engine._wake_running is True

    def test_pausing_a_stopped_loop_does_not_start_one(
            self, custom_engine: dict) -> None:
        engine = custom_engine["engine"]
        engine.config.update_assistant(name="Aariz", wake_word="hey aariz")
        with engine.wake_paused():
            pass
        assert engine._wake_running is False


# --------------------------------------------------------------- the gate

def _window():
    """A silent 1.5 s window: the augmentation is real, so this must be too."""
    import numpy as np

    from speech.wakeword_cascade import WINDOW_SAMPLES

    return np.zeros(WINDOW_SAMPLES, dtype=np.float32)


class TestThresholdCalibration:
    def test_it_picks_the_highest_threshold_that_keeps_the_positives(self) -> None:
        from speech.wakeword_enrollment import NegativePool, calibrate_threshold

        # A model that scores every training positive at 0.62: the rule should
        # settle on 0.60, the highest grid point below it.
        built = {"score": lambda window: 0.62, "calib_windows": [_window()] * 4}
        threshold, detail = calibrate_threshold(built, NegativePool())
        assert threshold == pytest.approx(0.60)
        assert "held-out" in detail["method"]

    def test_a_weak_model_gets_a_low_threshold_not_a_pass(self) -> None:
        from speech.wakeword_enrollment import NegativePool, calibrate_threshold

        built = {"score": lambda window: 0.07, "calib_windows": [_window()] * 4}
        threshold, _ = calibrate_threshold(built, NegativePool())
        assert threshold == pytest.approx(0.05)

    def test_it_never_reuses_the_aariz_threshold_by_default(self) -> None:
        from speech.wakeword_enrollment import NegativePool, calibrate_threshold

        built = {"score": lambda window: 0.95, "calib_windows": [_window()] * 4}
        threshold, _ = calibrate_threshold(built, NegativePool())
        assert threshold == pytest.approx(0.95)       # calibrated, not 0.10


class TestQualityGate:
    def _pool(self):
        from speech.wakeword_enrollment import NegativePool

        return NegativePool()

    def test_a_detector_that_hears_nothing_fails(self) -> None:
        from speech.wakeword_enrollment import quality_gate

        built = {"score": lambda w: 0.01, "held_windows": [object()] * 4}
        checks = quality_gate(built, self._pool(), threshold=0.4, phrase="hey friday")
        recall = next(c for c in checks if c.name == "held_out_recall")
        assert recall.passed is False
        assert recall.numerator == 0 and recall.denominator == 4

    def test_a_detector_that_hears_its_own_phrase_passes_recall(self) -> None:
        from speech.wakeword_enrollment import quality_gate

        built = {"score": lambda w: 0.9, "held_windows": [object()] * 4}
        checks = quality_gate(built, self._pool(), threshold=0.4, phrase="hey friday")
        recall = next(c for c in checks if c.name == "held_out_recall")
        assert recall.passed is True
        assert recall.numerator == 4

    def test_no_held_out_recordings_means_failure_not_a_free_pass(self) -> None:
        from speech.wakeword_enrollment import quality_gate

        built = {"score": lambda w: 0.9, "held_windows": []}
        checks = quality_gate(built, self._pool(), threshold=0.4, phrase="hey friday")
        recall = next(c for c in checks if c.name == "held_out_recall")
        assert recall.passed is False

    def test_the_margin_check_rejects_a_detector_sitting_on_the_line(self) -> None:
        from speech.wakeword_enrollment import quality_gate

        built = {"score": lambda w: 0.41, "held_windows": [object()] * 4}
        checks = quality_gate(built, self._pool(), threshold=0.40, phrase="hey friday")
        margin = next(c for c in checks if c.name == "margin")
        assert margin.passed is False


# ---------------------------------------------------------------- workflow

class _StubBuild(dict):
    """What ``build_detector`` would have returned, without training anything."""


@pytest.fixture
def stub_training(monkeypatch: pytest.MonkeyPatch, artefacts: dict):
    """Replace training with a stub whose score is chosen by the test."""

    def make(train_score: float, held_score: float = 0.95):
        """Two scores, because a real model does not give one.

        Calibration only ever sees training windows and the gate only ever sees
        held-out ones, so the stub scores them separately -- a single constant
        would collapse the margin check into an artefact of the grid spacing.
        """
        def fake_build(clips, *, phrase, negatives, work_dir, seed=0, epochs=0,
                       progress=None):
            import shutil

            verifier = Path(work_dir) / "verifier.pt"
            reference = Path(work_dir) / "reference.npz"
            shutil.copyfile(artefacts["verifier"], verifier)
            shutil.copyfile(artefacts["reference"], reference)
            held = [_window() for _ in range(4)]
            calib = [_window() for _ in range(3)]
            held_ids = {id(w) for w in held}
            calib_ids = {id(w) for w in calib}
            return _StubBuild(
                verifier_path=verifier, reference_path=reference,
                # Calibration windows generalise like the held-out ones do;
                # the training windows are the memorised set.
                score=lambda window: (held_score
                                      if id(window) in held_ids | calib_ids
                                      else train_score),
                frames=98, seed=seed, train_seconds=0.1, parameters=77041,
                checkpoint_bytes=verifier.stat().st_size,
                train_clips=[object()] * 11, held_clips=[object()] * 4,
                train_windows=[_window() for _ in range(11)],
                calib_windows=calib, calib_clips=[object()] * 3,
                held_windows=held,
                training_examples=1000, training_positives=300,
                training_negatives=700, reference_rows=11)

        monkeypatch.setattr("speech.wakeword_enrollment.build_detector", fake_build)

    return make


def _sufficient_pool():
    from speech.wakeword_enrollment import NegativePool

    rows = tuple({"path": f"/nonexistent/{i}.wav"} for i in range(40))
    return NegativePool(impostor_clips=rows, negative_clips=rows,
                        speech_files=("a.flac",), noise_files=("b.wav",),
                        sources={"test": True})


class TestEnrolmentWorkflow:
    def test_a_good_detector_is_installed(self, store, stub_training) -> None:
        from speech.wakeword_enrollment import enrol

        stub_training(0.62)
        result = enrol(display_name="Friday", clips=[], store=store,
                       phrase="hey friday", negatives=_sufficient_pool())
        assert result.passed is True
        assert result.installed is True
        assert store.get("hey friday") is not None
        assert result.stage2_threshold > 0.1

    def test_a_failing_detector_is_not_installed(self, store, stub_training) -> None:
        from speech.wakeword_enrollment import enrol

        # Trains fine, then does not recognise the recordings it never saw --
        # the failure mode enrollment exists to catch.
        stub_training(0.62, held_score=0.02)
        result = enrol(display_name="Friday", clips=[], store=store,
                       phrase="hey friday", negatives=_sufficient_pool())
        assert result.passed is False
        assert result.installed is False
        assert store.get("hey friday") is None
        assert "did not pass validation" in result.message

    def test_a_failed_re_enrollment_leaves_the_old_detector_working(
            self, store, artefacts, stub_training) -> None:
        from speech.wakeword_enrollment import enrol

        _install(store, artefacts, stage2=0.44)
        stub_training(0.62, held_score=0.02)
        result = enrol(display_name="Friday", clips=[], store=store,
                       phrase="hey friday", negatives=_sufficient_pool())
        assert result.installed is False
        survivor = store.get("hey friday")
        assert survivor.model_version == 1
        assert survivor.stage2_threshold == pytest.approx(0.44)

    def test_a_dry_run_installs_nothing(self, store, stub_training) -> None:
        from speech.wakeword_enrollment import enrol

        stub_training(0.62)
        result = enrol(display_name="Friday", clips=[], store=store,
                       phrase="hey friday", negatives=_sufficient_pool(),
                       install=False)
        assert result.passed is True
        assert result.installed is False
        assert store.get("hey friday") is None

    def test_too_little_negative_data_refuses_to_train(self, store) -> None:
        from speech.wakeword_enrollment import NegativePool, enrol

        result = enrol(display_name="Friday", clips=[], store=store,
                       phrase="hey friday", negatives=NegativePool())
        assert result.passed is False
        assert result.installed is False
        assert "can't train a wake word safely" in result.message

    def test_a_nameless_identity_is_refused(self, store) -> None:
        from speech.wakeword_enrollment import enrol

        result = enrol(display_name="   ", clips=[], store=store,
                       negatives=_sufficient_pool())
        assert result.passed is False
        assert result.installed is False

    def test_the_installed_threshold_is_the_calibrated_one(
            self, store, stub_training) -> None:
        from speech.wakeword_enrollment import enrol

        stub_training(0.72)
        result = enrol(display_name="Friday", clips=[], store=store,
                       phrase="hey friday", negatives=_sufficient_pool())
        assert result.installed is True
        assert store.get("hey friday").stage2_threshold == pytest.approx(
            result.stage2_threshold)
        assert result.stage2_threshold == pytest.approx(0.70)


class TestEnrolmentPlan:
    def test_it_asks_for_varied_recordings(self) -> None:
        from speech.wakeword_enrollment import TARGET_RECORDINGS, enrollment_plan

        prompts = enrollment_plan("hey friday")
        assert len(prompts) == TARGET_RECORDINGS
        assert len({p.distance for p in prompts}) >= 3
        assert len({p.condition for p in prompts}) >= 3
        assert all(p.phrase == "hey friday" for p in prompts)

    def test_every_prompt_asks_for_something_different(self) -> None:
        """Variety is the point; recall was measured to rise with genuinely
        different examples, not with more takes of the same delivery."""
        from speech.wakeword_enrollment import enrollment_plan

        prompts = enrollment_plan("hey friday")
        assert len({p.hint for p in prompts}) == len(prompts)

    def test_the_default_reflects_what_was_measured(self) -> None:
        # 15 recordings measured 0.50-0.60 recall on unseen audio; the default
        # must not drift back below what the study showed was needed.
        from speech.wakeword_enrollment import MIN_RECORDINGS, TARGET_RECORDINGS

        assert TARGET_RECORDINGS >= 25
        assert MIN_RECORDINGS >= 15

    def test_it_refuses_to_ask_for_too_few(self) -> None:
        from speech.wakeword_enrollment import MIN_RECORDINGS, enrollment_plan

        assert len(enrollment_plan("hey friday", 1)) == MIN_RECORDINGS


class TestRenameOffersEnrollment:
    def test_renaming_to_an_unenrolled_name_says_what_to_do(
            self, action_context) -> None:
        from actions.assistant import get_actions

        spec = next(s for s in get_actions() if s.name == "set_assistant_name")
        result = spec.handler(action_context, {"name": "Friday"})
        assert result.ok
        assert action_context.config.config.assistant.wake_word == "friday"
        message = result.message.lower()
        assert "haven't learned" in message
        assert "voice wake-up is off" in message
        assert "enrol" in message or "teach" in message

    def test_the_rename_still_persists(self, action_context) -> None:
        from actions.assistant import get_actions

        spec = next(s for s in get_actions() if s.name == "set_assistant_name")
        spec.handler(action_context, {"name": "Friday"})
        reloaded = ConfigManager(action_context.config.path)
        reloaded.load()
        assert reloaded.config.assistant.name == "Friday"
        assert reloaded.config.assistant.wake_word == "friday"
