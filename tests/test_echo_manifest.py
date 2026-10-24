"""The echo manifest guard: no clip is counted or skipped without a verdict.

A near-silent clip that reaches the echo denominator produces "zero false
wakes" for the wrong reason. The guard exists so that failure cannot happen
quietly, and these tests pin the two ways it could: an unclassified clip being
silently included, and an unclassified clip being silently dropped. Both are
made loud.
"""

from __future__ import annotations

import pytest

from benchmark.echo_manifest import (
    BORDERLINE,
    INVALID,
    MINIMUM_VALID_CLIPS,
    VALID,
    EchoManifest,
    EchoManifestError,
    assert_measurable,
)


def _clip(name: str, label: str = VALID, batch: str = "new_15", **extra):
    entry = {"file": f"tts_echo/{name}", "batch": batch, "label": label,
             "reason": "test fixture", "rms_dbfs": -40.0, "vad_triggered": True}
    entry.update(extra)
    return entry


def _manifest(*clips) -> EchoManifest:
    return EchoManifest({"thresholds": {"min_rms_dbfs": -48.4},
                         "clips": list(clips)})


class TestGuard:
    def test_an_unknown_clip_raises_rather_than_counting(self) -> None:
        manifest = _manifest(_clip("known.wav"))
        with pytest.raises(EchoManifestError, match="not in the TTS-echo manifest"):
            manifest.label_for("tts_echo/stranger.wav")

    def test_an_unknown_clip_is_not_silently_skipped_either(self) -> None:
        """The failure mode is symmetric; both directions must be loud."""
        manifest = _manifest(_clip("known.wav"))
        try:
            manifest.label_for("stranger.wav")
        except EchoManifestError as exc:
            assert "may never be counted, or skipped" in str(exc)
        else:
            pytest.fail("an unclassified clip returned a label")

    def test_a_missing_label_is_rejected_at_load(self) -> None:
        with pytest.raises(EchoManifestError, match="not one of"):
            EchoManifest({"clips": [{"file": "tts_echo/a.wav", "reason": "x"}]})

    def test_an_unrecognised_label_is_rejected(self) -> None:
        with pytest.raises(EchoManifestError, match="not one of"):
            _manifest(_clip("a.wav", label="PROBABLY_FINE"))

    def test_every_entry_must_carry_a_reason(self) -> None:
        with pytest.raises(EchoManifestError, match="needs a 'reason'"):
            EchoManifest({"clips": [{"file": "tts_echo/a.wav", "label": VALID}]})

    def test_a_missing_manifest_names_the_tool_that_writes_it(self, tmp_path) -> None:
        with pytest.raises(EchoManifestError, match="audit_tts_echo"):
            EchoManifest.load(tmp_path / "absent.json")

    def test_malformed_json_is_reported_not_swallowed(self, tmp_path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(EchoManifestError, match="not valid JSON"):
            EchoManifest.load(path)


class TestOnlyValidClipsCount:
    def test_borderline_and_invalid_are_excluded_from_the_valid_set(self) -> None:
        manifest = _manifest(_clip("a.wav", VALID), _clip("b.wav", BORDERLINE),
                             _clip("c.wav", INVALID))
        assert [c["file"] for c in manifest.valid_clips()] == ["tts_echo/a.wav"]
        assert manifest.counts() == {VALID: 1, BORDERLINE: 1, INVALID: 1}

    def test_borderline_is_kept_visible_rather_than_folded_away(self) -> None:
        """Three labels exist so the middle group cannot be quietly binned."""
        manifest = _manifest(_clip("b.wav", BORDERLINE))
        assert manifest.entries(BORDERLINE)
        assert not manifest.valid_clips()


class TestMeasurability:
    def test_a_thin_set_is_refused_rather_than_reported(self) -> None:
        manifest = _manifest(_clip("a.wav"))
        assert not manifest.is_measurable()
        with pytest.raises(EchoManifestError, match="NOT measurable"):
            assert_measurable(manifest)

    def test_a_sufficient_set_is_allowed(self) -> None:
        manifest = _manifest(*[_clip(f"c{i}.wav") for i in range(MINIMUM_VALID_CLIPS)])
        assert manifest.is_measurable()
        assert assert_measurable(manifest) is manifest

    def test_readiness_states_the_shortfall_plainly(self) -> None:
        assert "Only 1 valid clip" in _manifest(_clip("a.wav")).readiness()


class TestTheRealManifest:
    """The manifest actually written for this corpus."""

    @pytest.fixture
    def manifest(self) -> EchoManifest:
        return EchoManifest.load()

    def test_it_loads_and_classifies_all_thirty_clips(self, manifest) -> None:
        assert sum(manifest.counts().values()) == 30

    def test_every_clip_has_an_explicit_reason(self, manifest) -> None:
        for clip in manifest.entries():
            assert clip["reason"], clip["file"]

    def test_the_thresholds_are_recorded_for_audit(self, manifest) -> None:
        thresholds = manifest.thresholds
        for key in ("min_rms_dbfs", "min_speech_sec", "require_vad",
                    "positive_rms_p10_dbfs"):
            assert key in thresholds, key

    def test_this_corpus_is_not_measurable_for_echo(self, manifest) -> None:
        """Recorded as a fact so a later run cannot quietly assume otherwise."""
        assert not manifest.is_measurable()
        assert "NOT measurable" in manifest.readiness()
