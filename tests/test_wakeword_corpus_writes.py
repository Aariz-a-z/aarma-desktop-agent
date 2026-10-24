"""Recorded audio is never overwritten, and never was allowed to be.

A second ``tts-echo`` session wrote over the first because both runs built the
same filenames from phrase and rep alone. Fifteen recordings were destroyed and
survived only because a backup had been taken for an unrelated reason. Audio is
the one artefact in this project that cannot be regenerated -- a re-record is a
different room, a different voice, a different day -- so the writer now refuses
to replace a file rather than trusting its caller to have chosen a fresh name.

These tests use the recorder's own path allocation, not a re-implementation of
it, so they cannot drift away from the code they protect. No microphone, no
speakers and no network are involved: the fault was in naming, and naming is
what is asserted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.wakeword_corpus import (
    TTS_ECHO_PHRASES,
    ClipOverwriteError,
    ClipRecord,
    ClipWriteError,
    Corpus,
    echo_stem,
)

REPS = 3


def _audio(seconds: float = 0.2):
    return np.zeros(int(16_000 * seconds), dtype="float32")


def _echo_run(root: Path, run_id: str) -> list:
    """One full tts-echo session's writes, exactly as ``run_tts_echo`` does."""
    corpus = Corpus(root=root, run_id=run_id)
    written = []
    for phrase in TTS_ECHO_PHRASES:
        for rep in range(1, REPS + 1):
            path, take = corpus.reserve("tts_echo", echo_stem(phrase, rep))
            actual = corpus.write_clip(_audio(), path)
            corpus.append(ClipRecord(
                kind="tts_echo", path=actual, run_id=corpus.run_id, take=take,
                phrase=phrase, rep=rep, group="acoustic_loop"))
            written.append(Path(actual))
    return written


class TestTwoConsecutiveEchoRuns:
    """The exact scenario that destroyed the first fifteen clips."""

    def test_two_runs_produce_thirty_distinct_files(self, tmp_path) -> None:
        first = _echo_run(tmp_path, "20260816-090000")
        second = _echo_run(tmp_path, "20260816-100000")

        assert len(first) == 15 and len(second) == 15
        assert len(set(first) | set(second)) == 30, "paths collided across runs"
        assert all(p.is_file() for p in first + second)
        assert len(list(tmp_path.rglob("*.wav"))) == 30

    def test_the_first_run_is_still_readable_after_the_second(self,
                                                              tmp_path) -> None:
        """Preservation is the point; distinct names alone would not prove it."""
        first = _echo_run(tmp_path, "20260816-090000")
        sizes = {p: p.stat().st_size for p in first}
        _echo_run(tmp_path, "20260816-100000")
        for path, size in sizes.items():
            assert path.is_file(), f"{path.name} disappeared"
            assert path.stat().st_size == size, f"{path.name} was rewritten"

    def test_each_run_lands_in_its_own_directory(self, tmp_path) -> None:
        _echo_run(tmp_path, "20260816-090000")
        _echo_run(tmp_path, "20260816-100000")
        runs = sorted(p.name for p in (tmp_path / "tts_echo").iterdir()
                      if p.is_dir())
        assert runs == ["20260816-090000", "20260816-100000"]
        for run in runs:
            assert len(list((tmp_path / "tts_echo" / run).glob("*.wav"))) == 15

    def test_the_manifest_points_at_the_exact_file_for_every_record(
            self, tmp_path) -> None:
        _echo_run(tmp_path, "20260816-090000")
        _echo_run(tmp_path, "20260816-100000")
        records = [r for r in Corpus(root=tmp_path).records
                   if r.kind == "tts_echo"]
        assert len(records) == 30
        paths = [r.path for r in records]
        assert len(set(paths)) == 30, "two records share one file"
        for record in records:
            assert Path(record.path).is_file(), record.path
            assert record.run_id, "record carries no run id"
            assert record.run_id in Path(record.path).parts

    def test_records_are_attributable_to_their_run(self, tmp_path) -> None:
        _echo_run(tmp_path, "20260816-090000")
        _echo_run(tmp_path, "20260816-100000")
        records = [r for r in Corpus(root=tmp_path).records
                   if r.kind == "tts_echo"]
        by_run: dict = {}
        for record in records:
            by_run.setdefault(record.run_id, []).append(record)
        assert sorted(by_run) == ["20260816-090000", "20260816-100000"]
        assert all(len(v) == 15 for v in by_run.values())


class TestOverwriteIsRefused:
    def test_writing_over_an_existing_clip_raises(self, tmp_path) -> None:
        corpus = Corpus(root=tmp_path, run_id="20260816-090000")
        path, _ = corpus.reserve("tts_echo", "clip")
        corpus.write_clip(_audio(), path)
        with pytest.raises(ClipOverwriteError, match="never replaced"):
            corpus.write_clip(_audio(), path)

    def test_a_retry_inside_one_run_adds_a_take(self, tmp_path) -> None:
        """The latent half of the bug: a retry reused its own filename."""
        corpus = Corpus(root=tmp_path, run_id="20260816-090000")
        paths = []
        for _ in range(3):
            path, take = corpus.reserve("tts_echo", echo_stem("Hey Asma.", 1))
            corpus.write_clip(_audio(), path)
            paths.append((path.name, take))
        assert [t for _, t in paths] == [1, 2, 3]
        assert len({n for n, _ in paths}) == 3
        assert "take2" in paths[1][0] and "take3" in paths[2][0]

    def test_reserve_never_returns_a_path_that_exists(self, tmp_path) -> None:
        corpus = Corpus(root=tmp_path, run_id="20260816-090000")
        for _ in range(5):
            path, _take = corpus.reserve("positives", "asma_1m_normal_01")
            assert not path.exists()
            corpus.write_clip(_audio(), path)


class TestEveryModeIsProtected:
    """The bug was a class, not one filename. All five kinds are checked."""

    @pytest.mark.parametrize("kind,stem", [
        ("distance", "distance-1m"),
        ("positives", "asma_1m_normal_01"),
        ("negatives", "near_miss_Hey_Alma_01"),
        ("ambient", "ambient-001"),
        ("tts_echo", "echo_Hey_Asma_is_ready__01"),
    ])
    def test_a_second_run_cannot_collide(self, tmp_path, kind, stem) -> None:
        first = Corpus(root=tmp_path, run_id="20260816-090000")
        path_a, _ = first.reserve(kind, stem)
        first.write_clip(_audio(), path_a)

        second = Corpus(root=tmp_path, run_id="20260816-100000")
        path_b, take = second.reserve(kind, stem)

        assert path_a != path_b
        assert take == 1, "a fresh run should start at take 1, not inherit one"
        second.write_clip(_audio(), path_b)
        assert path_a.is_file() and path_b.is_file()


class TestBackwardCompatibility:
    def test_records_written_before_run_ids_still_load(self, tmp_path) -> None:
        """The existing 456-line manifest must keep parsing."""
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text(
            '{"kind": "tts_echo", "path": "tts_echo/old.wav", '
            '"phrase": "Hey Asma is ready.", "rep": 1}\n', encoding="utf-8")
        records = Corpus(root=tmp_path).records
        assert len(records) == 1
        assert records[0].run_id == ""      # Absent, not invented.
        assert records[0].take == 1


class TestAFailedWriteIsNeverRecorded:
    """A clip that was not written must not appear as though it was."""

    def test_a_write_failure_raises_instead_of_returning_an_empty_path(
            self, tmp_path, monkeypatch) -> None:
        corpus = Corpus(root=tmp_path, run_id="20260816-090000")
        path, _ = corpus.reserve("tts_echo", "clip")
        # write_wav reports failure by returning "" -- the recorder used to
        # store that as the record's path.
        monkeypatch.setattr("speech.input_engine.write_wav",
                            lambda audio, target: "")
        with pytest.raises(ClipWriteError, match="was NOT saved"):
            corpus.write_clip(_audio(), path)

    def test_the_error_names_the_windows_path_limit(self, tmp_path,
                                                    monkeypatch) -> None:
        """The failure that actually happened, so the message is useful."""
        corpus = Corpus(root=tmp_path, run_id="20260816-090000")
        path, _ = corpus.reserve("tts_echo", "clip")
        monkeypatch.setattr("speech.input_engine.write_wav",
                            lambda audio, target: "")
        with pytest.raises(ClipWriteError, match="260"):
            corpus.write_clip(_audio(), path)


class TestPathsFitWindowsLimit:
    def test_every_echo_clip_path_stays_well_inside_max_path(self) -> None:
        """MAX_PATH is 260. The nesting added by run ids must not breach it."""
        from tools.wakeword_corpus import corpus_root

        root = corpus_root()
        for phrase in TTS_ECHO_PHRASES:
            for rep in range(1, 4):
                candidate = (root / "tts_echo" / "20260816-090000" /
                             f"{echo_stem(phrase, rep)}-take9.wav")
                assert len(str(candidate)) < 240, (
                    f"{len(str(candidate))} chars: {candidate}")
