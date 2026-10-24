"""Training data must actually reach training, and failures must be audible.

These exist because of a bug that ran undetected: `_load_noise_pool` was changed
to return decoded arrays for speed, the loop below it still called `_read_wav`
on each entry, every call raised, and a bare `except: continue` swallowed all of
them. Every enrolled detector trained with **zero** MUSAN noise negatives and
nothing anywhere said so. It surfaced only weeks later, as an unexplained
difference between two false-activation measurements.

So the tests here are not about noise. They are about the two properties that
would have caught it: data that is meant to be used is used, and data that
cannot be read is counted and logged rather than skipped in silence.
"""

from __future__ import annotations

import inspect
import logging

import numpy as np
import pytest

import speech.wakeword_enrollment as enrollment
from speech.wakeword_cascade import WINDOW_SAMPLES
from speech.wakeword_enrollment import (
    NOISE_NEGATIVE_WINDOWS,
    _load_noise_pool,
    _read_wav,
    _Skipped,
)


class TestTheNoisePoolIsUsable:
    """The exact shape mismatch that caused the bug."""

    def test_the_pool_yields_arrays_not_paths(self, tmp_path) -> None:
        import soundfile as sf

        paths = []
        for n in range(3):
            path = tmp_path / f"noise-{n}.wav"
            sf.write(path, np.zeros(WINDOW_SAMPLES * 2, dtype=np.float32), 16000)
            paths.append(str(path))

        pool = _load_noise_pool(tuple(paths), np.random.default_rng(0))
        assert pool, "the pool decoded nothing"
        assert all(isinstance(entry, np.ndarray) for entry in pool)

    def test_reading_a_pool_entry_as_a_file_fails(self, tmp_path) -> None:
        """Documents why the bug was silent: this raises, it does not return junk."""
        import soundfile as sf

        path = tmp_path / "noise.wav"
        sf.write(path, np.zeros(WINDOW_SAMPLES * 2, dtype=np.float32), 16000)
        pool = _load_noise_pool((str(path),), np.random.default_rng(0))

        with pytest.raises(Exception):
            _read_wav(pool[0])

    def test_the_trainer_does_not_read_pool_entries_as_files(self) -> None:
        """The regression guard.

        Source inspection rather than behaviour, because the failure mode is
        someone reintroducing `_read_wav(noise_pool[i])` — which would once
        again raise on every entry and once again be invisible.
        """
        source = inspect.getsource(enrollment.build_detector)
        assert "_read_wav(noise_pool[" not in source
        assert "usable_noise" in source

    def test_noise_windows_are_actually_requested(self) -> None:
        assert NOISE_NEGATIVE_WINDOWS > 0
        source = inspect.getsource(enrollment.build_detector)
        assert "NOISE_NEGATIVE_WINDOWS" in source
        assert "noise_added" in source


class TestDatasetFailuresAreAudible:
    def test_no_data_loader_swallows_errors_silently(self) -> None:
        """Every audio loader must record what it dropped.

        A bare `except: continue` around a read is the pattern that hid the
        bug; this fails if one comes back.
        """
        source = inspect.getsource(enrollment)
        assert "except Exception:  # noqa: BLE001\n            continue" not in source

    def test_the_skip_counter_records_the_first_failure(self) -> None:
        skipped = _Skipped("test source")
        skipped.note("a.wav", ValueError("boom"))
        skipped.note("b.wav", ValueError("second"))
        assert skipped.count == 2
        assert "ValueError" in skipped.first
        assert "boom" in skipped.first

    def test_losing_every_entry_is_logged_loudly(self, caplog) -> None:
        skipped = _Skipped("MUSAN noise")
        for n in range(5):
            skipped.note(f"{n}.wav", OSError("cannot decode"))
        with caplog.at_level(logging.WARNING):
            skipped.report(total=5)
        assert any(record.levelno >= logging.WARNING for record in caplog.records)
        assert "every" in caplog.text.lower()
        assert "musan noise" in caplog.text.lower()

    def test_losing_some_entries_is_still_reported(self, caplog) -> None:
        skipped = _Skipped("ambient")
        skipped.note("x.wav", OSError("bad"))
        with caplog.at_level(logging.WARNING):
            skipped.report(total=10)
        assert "1 of 10" in caplog.text

    def test_nothing_is_logged_when_everything_loads(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            _Skipped("ambient").report(total=10)
        assert caplog.text == ""


class TestTrainingReportsWhatItUsed:
    def test_the_summary_names_both_negative_sources(self) -> None:
        """A run that lost a whole source should be visible in its own report."""
        source = inspect.getsource(enrollment.build_detector)
        assert '"silence_negatives"' in source
        assert '"noise_negatives"' in source

    def test_an_empty_source_warns(self) -> None:
        source = inspect.getsource(enrollment.build_detector)
        assert "No %s negatives reached training" in source


class TestSilenceNegativesRemain:
    """The fix that took room-noise false activations from 296/h to single digits."""

    def test_the_constant_is_still_set(self) -> None:
        assert enrollment.SILENCE_NEGATIVE_WINDOWS >= 100

    def test_only_the_declared_dev_ambient_may_be_trained_on(self) -> None:
        from speech.wakeword_enrollment import AMBIENT_DEV_FILES, NegativePool, \
            split_ambient

        rows = tuple({"path": f"/ambient-{n:03d}.wav"} for n in range(37))
        dev, held = split_ambient(NegativePool(ambient_clips=rows))
        assert len(dev) == AMBIENT_DEV_FILES
        assert len(held) == 37 - AMBIENT_DEV_FILES
        # Disjoint: the gate must never score what the trainer saw.
        assert not ({r["path"] for r in dev} & {r["path"] for r in held})

    def test_the_gate_scores_only_held_out_ambient(self) -> None:
        source = inspect.getsource(enrollment.quality_gate)
        assert "split_ambient" in source
        assert "held_ambient" in source
