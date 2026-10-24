"""Microphone diagnostics, device identity, recovery and reset.

Written after a user reported "No speech heard — 0.0000 RMS" while their
microphone worked in other applications. That message conflated several
distinct problems; these tests pin down the distinctions so it cannot happen
again.

No physical hardware is required: devices are mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import ConfigManager, SpeechInputSettings, load_config
from speech.capture import MicrophoneDevice, resolve_device
from speech.diagnostics import (
    CaptureStatus,
    DeviceProbe,
    MicrophoneDiagnosis,
    MicrophonePermission,
    check_windows_permission,
    classify,
    format_level,
)


def device(index: int, name: str, *, default: bool = False,
           api: str = "MME") -> MicrophoneDevice:
    return MicrophoneDevice(index=index, name=name, channels=2,
                            sample_rate=48000.0, is_default=default, host_api=api)


DEVICES = [
    device(0, "Sound Mapper - Input"),
    device(1, "Microphone (Realtek Audio)", default=True),
    device(5, "USB Headset Microphone", api="Windows WASAPI"),
]


# ------------------------------------------------------------ level display

class TestLevelFormatting:
    """Four decimal places is what made a working microphone look dead."""

    def test_true_zero_is_stated_explicitly(self) -> None:
        assert format_level(0.0) == "0 (exactly)"

    def test_tiny_values_do_not_collapse_to_zero(self) -> None:
        """0.000017 RMS is 'quiet', not 'nothing' — and used to print 0.0000."""
        rendered = format_level(0.000017)
        assert rendered != "0.0000"
        assert "e-" in rendered

    def test_normal_values_stay_readable(self) -> None:
        assert format_level(0.0312) == "0.0312"

    def test_a_quiet_reading_is_distinguishable_from_silence(self) -> None:
        assert format_level(0.0) != format_level(0.000017)


# ------------------------------------------------------------ classification

class TestClassification:
    """Every failure gets its own verdict (specification section 1)."""

    def test_no_frames_is_an_open_failure(self) -> None:
        assert classify(frames=0, peak=0.0, nonzero=0,
                        speech=False) is CaptureStatus.OPEN_FAILED

    def test_all_zero_samples_is_not_the_same_as_quiet(self) -> None:
        assert classify(frames=16000, peak=0.0, nonzero=0,
                        speech=False) is CaptureStatus.SILENT_SAMPLES

    def test_noise_floor_signal_is_near_silent(self) -> None:
        assert classify(frames=16000, peak=0.00012, nonzero=5000,
                        speech=False) is CaptureStatus.NEAR_SILENT

    def test_real_audio_without_speech_is_its_own_state(self) -> None:
        """"Audio arriving but no speech" must not read as "no microphone"."""
        assert classify(frames=16000, peak=0.05, nonzero=16000,
                        speech=False) is CaptureStatus.SIGNAL_NO_SPEECH

    def test_speech_is_detected(self) -> None:
        assert classify(frames=16000, peak=0.08, nonzero=16000,
                        speech=True) is CaptureStatus.SPEECH_DETECTED

    def test_only_real_audio_counts_as_ok(self) -> None:
        assert CaptureStatus.SPEECH_DETECTED.ok
        assert CaptureStatus.SIGNAL_NO_SPEECH.ok
        assert not CaptureStatus.NEAR_SILENT.ok
        assert not CaptureStatus.SILENT_SAMPLES.ok
        assert not CaptureStatus.PERMISSION_DENIED.ok

    def test_every_status_has_a_distinct_label(self) -> None:
        labels = {status.label for status in CaptureStatus}
        assert len(labels) == len(list(CaptureStatus))


# ---------------------------------------------------------------- permission

class TestWindowsPermission:
    def test_reading_consent_never_raises(self) -> None:
        permission = check_windows_permission()
        assert isinstance(permission.allowed, bool)

    def test_blocked_permission_gives_actionable_advice(self) -> None:
        blocked = MicrophonePermission(allowed=False, desktop_value="Deny")
        assert "Windows Settings" in blocked.advice()
        assert blocked.label == "Blocked by Windows"

    def test_allowed_permission_offers_no_advice(self) -> None:
        assert MicrophonePermission(allowed=True).advice() == ""

    def test_unchecked_platform_is_not_reported_as_blocked(self) -> None:
        unknown = MicrophonePermission(allowed=True, checked=False)
        assert unknown.label == "Not checked"


# ----------------------------------------------------------- device identity

class TestDeviceResolution:
    """Devices are stored by name because indexes move between runs."""

    @pytest.fixture(autouse=True)
    def devices(self, monkeypatch):
        monkeypatch.setattr("speech.capture.list_devices", lambda: DEVICES)
        return DEVICES

    def test_no_name_uses_the_windows_default(self) -> None:
        resolution = resolve_device(None)
        assert resolution.device.name == "Microphone (Realtek Audio)"
        assert not resolution.fell_back

    def test_a_saved_name_is_found(self) -> None:
        resolution = resolve_device("USB Headset Microphone")
        assert resolution.device.index == 5
        assert not resolution.fell_back

    def test_matching_ignores_case_and_padding(self) -> None:
        assert resolve_device("  usb headset microphone ").device.index == 5

    def test_partial_names_still_match(self) -> None:
        """Windows reports the same hardware with small variations between runs."""
        assert resolve_device("USB Headset").device.index == 5

    def test_a_missing_device_falls_back_to_the_default(self) -> None:
        resolution = resolve_device("Studio Condenser That Was Unplugged")
        assert resolution.fell_back
        assert resolution.device.name == "Microphone (Realtek Audio)"

    def test_the_fallback_is_announced(self) -> None:
        """Silently switching devices is how a user loses track of reality."""
        notice = resolve_device("Vanished Microphone").notice()
        assert "Vanished Microphone" in notice
        assert "Windows default" in notice

    def test_no_notice_when_nothing_changed(self) -> None:
        assert resolve_device("USB Headset Microphone").notice() == ""

    def test_no_devices_resolves_to_nothing(self, monkeypatch) -> None:
        monkeypatch.setattr("speech.capture.list_devices", lambda: [])
        resolution = resolve_device("anything")
        assert resolution.device is None
        assert resolution.index is None


class TestListeningEngineRecovery:
    @pytest.fixture(autouse=True)
    def devices(self, monkeypatch):
        monkeypatch.setattr("speech.capture.list_devices", lambda: DEVICES)

    def test_engine_resolves_the_saved_device(self, config: ConfigManager) -> None:
        from speech.input_engine import ListeningEngine

        config.update_speech_input(device_name="USB Headset Microphone")
        engine = ListeningEngine(config)
        assert engine.resolve_device().device.index == 5

    def test_engine_recovers_from_a_vanished_device(self, config: ConfigManager) -> None:
        from speech.input_engine import ListeningEngine

        config.update_speech_input(device_name="Gone Microphone")
        engine = ListeningEngine(config)
        resolution = engine.resolve_device()
        assert resolution.fell_back
        assert resolution.device.is_default

    def test_the_user_is_told_once_about_the_switch(self, config: ConfigManager) -> None:
        from speech.input_engine import ListeningEngine

        config.update_speech_input(device_name="Gone Microphone")
        engine = ListeningEngine(config)
        engine._capture_backend()
        notice = engine.take_device_notice()
        assert "Gone Microphone" in notice
        # Consumed: it should not nag on every subsequent call.
        assert engine.take_device_notice() == ""


# ----------------------------------------------------------------- diagnosis

def probe(name: str, status: CaptureStatus, *, peak: float = 0.0,
          selected: bool = False) -> DeviceProbe:
    return DeviceProbe(index=0, name=name, status=status, peak=peak,
                       frames=16000, is_selected=selected)


class TestDiagnosisRecommendations:
    def test_permission_problem_takes_priority(self) -> None:
        diagnosis = MicrophoneDiagnosis(
            permission=MicrophonePermission(allowed=False, desktop_value="Deny"),
            probes=[probe("Mic", CaptureStatus.SILENT_SAMPLES, selected=True)])
        assert "Windows Settings" in diagnosis.recommendation()

    def test_no_devices_is_reported_plainly(self) -> None:
        diagnosis = MicrophoneDiagnosis(
            permission=MicrophonePermission(allowed=True))
        assert "No input devices" in diagnosis.recommendation()
        assert diagnosis.status is CaptureStatus.NO_DEVICE

    def test_a_working_microphone_says_so(self) -> None:
        diagnosis = MicrophoneDiagnosis(
            permission=MicrophonePermission(allowed=True),
            probes=[probe("Mic", CaptureStatus.SPEECH_DETECTED, peak=0.1,
                          selected=True)])
        diagnosis.selected = diagnosis.probes[0]
        assert "working" in diagnosis.recommendation().lower()

    def test_a_silent_selection_points_at_a_working_device(self) -> None:
        """The single most useful thing the old message could not do."""
        silent = probe("Dead Mic", CaptureStatus.SILENT_SAMPLES, selected=True)
        good = probe("USB Headset", CaptureStatus.SPEECH_DETECTED, peak=0.09)
        diagnosis = MicrophoneDiagnosis(
            permission=MicrophonePermission(allowed=True),
            probes=[silent, good], selected=silent)
        recommendation = diagnosis.recommendation()
        assert "USB Headset" in recommendation

    def test_a_quiet_room_is_not_called_a_fault(self) -> None:
        """Passive probing must not accuse a microphone that was never spoken to."""
        quiet = probe("Mic", CaptureStatus.NEAR_SILENT, peak=0.0001, selected=True)
        diagnosis = MicrophoneDiagnosis(
            permission=MicrophonePermission(allowed=True),
            probes=[quiet], selected=quiet, expected_speech=False)
        assert "quiet during the check" in diagnosis.recommendation()

    def test_a_quiet_reading_when_speech_was_expected_is_a_fault(self) -> None:
        quiet = probe("Mic", CaptureStatus.NEAR_SILENT, peak=0.0001, selected=True)
        diagnosis = MicrophoneDiagnosis(
            permission=MicrophonePermission(allowed=True),
            probes=[quiet], selected=quiet, expected_speech=True)
        advice = diagnosis.recommendation()
        assert "input volume" in advice or "almost zero" in advice

    def test_unusable_devices_are_excluded_from_suggestions(self) -> None:
        diagnosis = MicrophoneDiagnosis(
            permission=MicrophonePermission(allowed=True),
            probes=[probe("Broken", CaptureStatus.OPEN_FAILED),
                    probe("Silent", CaptureStatus.SILENT_SAMPLES),
                    probe("Good", CaptureStatus.SIGNAL_NO_SPEECH, peak=0.03)])
        assert [p.name for p in diagnosis.usable] == ["Good"]

    def test_probe_explanations_are_distinct_and_actionable(self) -> None:
        seen = set()
        for status in CaptureStatus:
            text = probe("Mic", status).explain()
            assert text and text not in seen
            seen.add(text)


class TestProbeReporting:
    def test_summary_reports_frames_and_levels(self) -> None:
        p = DeviceProbe(index=1, name="Mic", frames=19200, rms=0.031, peak=0.18,
                        status=CaptureStatus.SPEECH_DETECTED)
        assert "19200 frames" in p.summary
        assert "0.0310" in p.summary
        assert "0.1800" in p.summary

    def test_open_failure_surfaces_the_error(self) -> None:
        p = DeviceProbe(index=1, name="Mic", status=CaptureStatus.OPEN_FAILED,
                        error="PortAudioError: device in use")
        assert "device in use" in p.summary

    def test_default_device_is_marked(self) -> None:
        p = DeviceProbe(index=1, name="Mic", is_default=True)
        assert "Windows default" in p.display_name


# --------------------------------------------------------------------- reset

class TestReset:
    def test_speech_reset_restores_defaults(self, config: ConfigManager) -> None:
        config.update_speech_input(device_name="Some Mic", model_size="tiny",
                                   vad_sensitivity=0.1, max_seconds=30.0,
                                   confidence_threshold=0.9, enabled=False)
        config.reset_speech_input()
        assert config.config.speech_input == SpeechInputSettings()

    def test_speech_reset_leaves_everything_else_alone(self, config: ConfigManager) -> None:
        """Fixing a microphone must not cost you your assistant's name."""
        config.update_assistant(name="Nova")
        config.update_language(interface="hi")
        config.update_voice(rate=1.3)
        config.set_permission("microphone", True)
        config.update_speech_input(device_name="Some Mic")

        config.reset_speech_input()

        assert config.config.assistant.name == "Nova"
        assert config.config.language.interface == "hi"
        assert config.config.voice.rate == pytest.approx(1.3)
        assert config.config.permissions.granted["microphone"] is True
        assert config.config.speech_input.device_name is None

    def test_speech_reset_persists(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        manager = ConfigManager(path)
        manager.load()
        manager.update_speech_input(device_name="Some Mic", model_size="tiny")
        manager.reset_speech_input()
        assert load_config(path).config.speech_input.model_size == "small"

    def test_full_reset_restores_every_section(self, config: ConfigManager) -> None:
        config.update_assistant(name="Nova")
        config.update_voice(rate=1.4)
        config.update_speech_input(model_size="tiny")
        config.mark_first_run_complete()

        config.reset_all_settings()

        assert config.config.assistant.name == "Asma"
        assert config.config.voice.rate == pytest.approx(1.0)
        assert config.config.speech_input.model_size == "small"
        assert config.config.first_run_completed is False

    def test_full_reset_does_not_touch_stored_data(self, config: ConfigManager,
                                                   database) -> None:
        """Resetting settings is not the same as deleting a user's history."""
        database.add_message("s1", "user", "remember this")
        database.record_action(session_id="s1", action="get_time", parameters={},
                               risk="Low", source="text", allowed=True,
                               succeeded=True)

        config.reset_all_settings()

        assert len(database.recent_messages()) == 1
        assert len(database.recent_actions()) == 1

    def test_recommended_defaults_after_reset(self, config: ConfigManager) -> None:
        config.reset_speech_input()
        settings = config.config.speech_input
        assert settings.device_name is None          # Windows default.
        assert settings.model_size == "small"        # Benchmarked recommendation.
        assert settings.enabled is True
        assert 0.0 <= settings.vad_sensitivity <= 1.0
        assert settings.no_speech_timeout < settings.max_seconds


class TestConfigMigration:
    def test_a_config_saved_with_a_device_index_still_loads(self,
                                                            tmp_path: Path) -> None:
        """Older builds stored an index; it must not break loading."""
        path = tmp_path / "config.json"
        path.write_text('{"speech_input": {"device_index": 7, "model_size": "base"}}',
                        encoding="utf-8")
        manager = load_config(path)
        assert manager.config.speech_input.device_index == 7
        assert manager.config.speech_input.device_name is None
        assert manager.config.speech_input.model_size == "base"
