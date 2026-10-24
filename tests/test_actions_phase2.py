"""The command system: system health, media, power, reminders, calendar.

Two properties matter more than the rest and most of this file defends them.

**Nothing destructive happens without a yes.** Restart and shutdown close every
open application; a speech recogniser must never be trusted to have heard those
correctly on its own. The tests assert both that confirmation is demanded *and*
that no command reached the operating system.

**Nothing is claimed that did not happen.** A temperature with no sensor, a
volume level that cannot be set, a calendar that is not connected -- each has to
report the gap rather than produce a plausible number. That failure mode is
invisible to the user, which is exactly why it is tested.

No test here touches the real machine: the process runner, the media keys and
psutil are all injected.
"""

from __future__ import annotations

import types
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

from actions.registry import ActionContext, build_default_registry
from providers.media import MediaProvider
from providers.system import SystemProvider
from security.validator import RiskLevel
from utils.when import describe, local_now, parse_when


# ----------------------------------------------------------------- fixtures

class _RecordingRunner:
    """Stands in for subprocess.run and remembers what it was asked to do."""

    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.calls: List[List[str]] = []
        self._result = types.SimpleNamespace(returncode=returncode, stderr=stderr,
                                             stdout="")

    def __call__(self, command, **kwargs):
        self.calls.append(list(command))
        return self._result


class _FakeKeyboard:
    def __init__(self) -> None:
        self.presses: List[int] = []

    def keybd_event(self, key, scan, flags, extra):
        if not flags:                       # count key-down only
            self.presses.append(key)


def _context(action_context: ActionContext, **overrides) -> ActionContext:
    """A context with extra provider slots the handlers look for."""
    for key, value in overrides.items():
        object.__setattr__(action_context, key, value)
    return action_context


@pytest.fixture
def registry():
    return build_default_registry()


def _run(registry, action_context, name: str, **params):
    spec = registry.require(name)
    return spec.handler(action_context, spec.validate(params))


# ------------------------------------------------------------ system health

class _FakePsutil:
    """Just enough psutil to drive the provider, with controllable gaps."""

    def __init__(self, *, battery=True, temps=None, raise_memory=False) -> None:
        self._battery = battery
        self._temps = temps
        self._raise_memory = raise_memory

    def cpu_percent(self, interval=None):
        return 17.5

    def cpu_count(self, logical=True):
        return 8

    def virtual_memory(self):
        if self._raise_memory:
            raise OSError("no memory info")
        return types.SimpleNamespace(total=16_000_000_000, used=8_000_000_000,
                                     percent=50.0)

    def disk_partitions(self, all=False):
        return [types.SimpleNamespace(mountpoint="C:\\")]

    def disk_usage(self, mount):
        return types.SimpleNamespace(total=500_000_000_000, used=358_000_000_000,
                                     free=142_000_000_000)

    def sensors_battery(self):
        if not self._battery:
            return None
        return types.SimpleNamespace(percent=74.0, power_plugged=True, secsleft=-1)

    def sensors_temperatures(self):
        if self._temps is None:
            raise AttributeError("not on this platform")
        return self._temps

    def boot_time(self):
        return 0.0

    def pids(self):
        return list(range(120))

    def net_if_stats(self):
        return {"Wi-Fi": types.SimpleNamespace(isup=True)}

    def process_iter(self, attrs=None):
        return []


class TestSystemHealth:
    def test_it_reports_what_it_measured(self, registry, action_context) -> None:
        ctx = _context(action_context, system_provider=SystemProvider(_FakePsutil()))
        result = _run(registry, ctx, "system_health")
        assert result.ok
        assert result.data["cpu_percent"] == 17.5
        assert result.data["battery_percent"] == 74.0
        assert "142 GB free" in result.message

    def test_temperature_is_refused_when_no_sensor_exists(self, registry,
                                                          action_context) -> None:
        """The claim this whole module exists to prevent."""
        ctx = _context(action_context, system_provider=SystemProvider(_FakePsutil()))
        result = _run(registry, ctx, "cpu_temperature")
        assert not result.ok
        assert result.error_code == "reading_unavailable"
        # No number is offered, and the refusal says why.
        assert "can't read" in result.message
        assert "°C" not in result.message

    def test_temperature_is_given_when_a_sensor_exists(self, registry,
                                                       action_context) -> None:
        temps = {"coretemp": [types.SimpleNamespace(current=61.0)]}
        ctx = _context(action_context,
                       system_provider=SystemProvider(_FakePsutil(temps=temps)))
        result = _run(registry, ctx, "cpu_temperature")
        assert result.ok
        assert result.data["cpu_temperature_c"] == 61.0

    def test_a_machine_with_no_battery_is_not_an_error(self, registry,
                                                       action_context) -> None:
        ctx = _context(action_context,
                       system_provider=SystemProvider(_FakePsutil(battery=False)))
        result = _run(registry, ctx, "battery_status")
        assert result.ok                       # a desktop, not a failure
        assert result.data["battery_percent"] is None

    def test_one_unreadable_field_does_not_lose_the_others(self, registry,
                                                           action_context) -> None:
        provider = SystemProvider(_FakePsutil(raise_memory=True))
        ctx = _context(action_context, system_provider=provider)
        result = _run(registry, ctx, "system_health")
        assert result.ok
        assert result.data["cpu_percent"] == 17.5      # still measured
        assert result.data["memory_percent"] is None   # honestly absent
        assert "memory" in result.data["unavailable"]

    def test_no_psutil_is_reported_not_faked(self, registry, action_context) -> None:
        provider = SystemProvider()
        provider._psutil, provider._resolved = None, True
        ctx = _context(action_context, system_provider=provider)
        result = _run(registry, ctx, "system_health")
        assert not result.ok
        assert result.error_code == "system_unavailable"


# -------------------------------------------------------------------- media

class TestMedia:
    def test_volume_down_presses_the_volume_key(self, registry,
                                                action_context) -> None:
        keyboard = _FakeKeyboard()
        ctx = _context(action_context,
                       media_provider=MediaProvider(keyboard, system="Windows"))
        result = _run(registry, ctx, "volume_down")
        assert result.ok
        assert keyboard.presses and all(k == 0xAE for k in keyboard.presses)

    def test_setting_an_exact_level_is_refused_not_approximated(
            self, registry, action_context) -> None:
        """Stepping to an estimate and calling it 30% would be a fabrication."""
        keyboard = _FakeKeyboard()
        ctx = _context(action_context,
                       media_provider=MediaProvider(keyboard, system="Windows"))
        result = _run(registry, ctx, "set_volume", percent=30)
        assert not result.ok
        assert result.error_code == "not_supported"
        assert keyboard.presses == []          # nothing was pressed
        assert "pycaw" in result.message

    def test_an_out_of_range_level_is_rejected(self, registry,
                                               action_context) -> None:
        ctx = _context(action_context,
                       media_provider=MediaProvider(_FakeKeyboard(),
                                                    system="Windows"))
        result = _run(registry, ctx, "set_volume", percent=150)
        assert not result.ok
        assert result.error_code == "invalid_volume"

    def test_mute_does_not_claim_a_direction(self, registry,
                                             action_context) -> None:
        # The key toggles and Windows does not report which way it went.
        ctx = _context(action_context,
                       media_provider=MediaProvider(_FakeKeyboard(),
                                                    system="Windows"))
        result = _run(registry, ctx, "mute")
        assert result.ok
        assert "muted" not in result.message.lower()

    def test_media_is_unavailable_off_windows(self, registry,
                                              action_context) -> None:
        ctx = _context(action_context,
                       media_provider=MediaProvider(_FakeKeyboard(),
                                                    system="Linux"))
        result = _run(registry, ctx, "play_pause")
        assert not result.ok

    @pytest.mark.parametrize("action,key", [
        ("volume_up", 0xAF), ("volume_down", 0xAE), ("mute", 0xAD),
        ("play_pause", 0xB3), ("next_track", 0xB0), ("previous_track", 0xB1),
    ])
    def test_each_control_sends_its_own_key(self, registry, action_context,
                                            action: str, key: int) -> None:
        keyboard = _FakeKeyboard()
        ctx = _context(action_context,
                       media_provider=MediaProvider(keyboard, system="Windows"))
        assert _run(registry, ctx, action).ok
        assert key in keyboard.presses


# -------------------------------------------------------------------- power

class TestPowerRiskGrading:
    """The grading is the safety mechanism, so it is asserted directly."""

    @pytest.mark.parametrize("name", ["restart_computer", "shutdown_computer",
                                      "sign_out"])
    def test_destructive_power_actions_are_high_risk(self, registry,
                                                     name: str) -> None:
        assert registry.require(name).risk is RiskLevel.HIGH

    def test_locking_is_not_high_risk(self, registry) -> None:
        # Confirming a lock would teach the user to dismiss prompts unread.
        assert registry.require("lock_screen").risk is RiskLevel.LOW

    def test_high_risk_always_confirms_whatever_the_settings_say(
            self, config) -> None:
        from security.validator import RiskPolicy

        config.update_security(enabled=False)
        policy = RiskPolicy.from_settings(config.config.security)
        assert policy.requires_confirmation(RiskLevel.HIGH) is True


class TestPowerExecution:
    def test_lock_runs_the_documented_command(self, registry,
                                              action_context) -> None:
        runner = _RecordingRunner()
        ctx = _context(action_context, process_runner=runner,
                       platform_system="Windows")
        result = _run(registry, ctx, "lock_screen")
        assert result.ok
        assert runner.calls == [["rundll32.exe", "user32.dll,LockWorkStation"]]

    def test_shutdown_uses_a_grace_period(self, registry, action_context) -> None:
        """The delay is what makes `shutdown /a` a usable escape hatch."""
        from actions.power import GRACE_SECONDS

        runner = _RecordingRunner()
        ctx = _context(action_context, process_runner=runner,
                       platform_system="Windows")
        result = _run(registry, ctx, "shutdown_computer")
        assert result.ok
        assert runner.calls == [["shutdown", "/s", "/t", str(GRACE_SECONDS)]]
        assert GRACE_SECONDS > 0

    def test_a_failing_command_is_reported_not_raised(self, registry,
                                                      action_context) -> None:
        runner = _RecordingRunner(returncode=1, stderr="Access is denied.")
        ctx = _context(action_context, process_runner=runner,
                       platform_system="Windows")
        result = _run(registry, ctx, "restart_computer")
        assert not result.ok
        assert result.error_code == "command_refused"

    def test_a_missing_command_is_reported_not_raised(self, registry,
                                                      action_context) -> None:
        def missing(command, **kwargs):
            raise FileNotFoundError(command[0])

        ctx = _context(action_context, process_runner=missing,
                       platform_system="Windows")
        result = _run(registry, ctx, "shutdown_computer")
        assert not result.ok
        assert result.error_code == "command_missing"

    def test_power_actions_refuse_off_windows(self, registry,
                                              action_context) -> None:
        runner = _RecordingRunner()
        ctx = _context(action_context, process_runner=runner,
                       platform_system="Linux")
        result = _run(registry, ctx, "shutdown_computer")
        assert not result.ok
        assert runner.calls == []


# ---------------------------------------------------------------- reminders

class TestReminders:
    def test_creating_and_listing(self, registry, action_context) -> None:
        result = _run(registry, action_context, "create_reminder",
                      when="in 2 hours", title="submit my project")
        assert result.ok
        assert result.data["reminder_id"]

        listed = _run(registry, action_context, "list_reminders")
        assert listed.data["count"] == 1
        assert "submit my project" in listed.message

    def test_a_reminder_survives_a_restart(self, registry, action_context,
                                           database) -> None:
        _run(registry, action_context, "create_reminder",
             when="tomorrow at 9 AM", title="call Mom")
        # Everything a restart does: read the same database again.
        rows = database.reminders(state="pending")
        assert [r["title"] for r in rows] == ["call Mom"]
        assert "+" in rows[0]["due_at"] or "Z" in rows[0]["due_at"], \
            "stored time must carry its offset"

    def test_a_time_in_the_past_is_refused(self, registry,
                                           action_context) -> None:
        result = _run(registry, action_context, "create_reminder",
                      when="in 0 minutes", title="too late")
        assert not result.ok
        assert result.error_code in ("time_in_past", "unparsed_time")

    def test_an_unparseable_time_is_refused(self, registry,
                                            action_context) -> None:
        result = _run(registry, action_context, "create_reminder",
                      when="sometime later", title="vague")
        assert not result.ok
        assert result.error_code == "unparsed_time"

    def test_a_missing_subject_asks_rather_than_storing_nothing(
            self, registry, action_context) -> None:
        # The live path omits the key entirely; an explicit "" is rejected
        # by the registry before the handler ever sees it.
        result = _run(registry, action_context, "create_reminder",
                      when="tomorrow at 10 AM")
        assert not result.ok
        assert result.data.get("needs") == "title"
        assert "What should I remind you about" in result.message

    def test_cancelling_by_spoken_time(self, registry, action_context) -> None:
        """"my 6 PM reminder" against a stored 18:00."""
        target = (local_now() + timedelta(days=1)).replace(
            hour=18, minute=0, second=0, microsecond=0)
        _run(registry, action_context, "create_reminder",
             when=f"tomorrow at 6 PM", title="submit my project")
        result = _run(registry, action_context, "cancel_reminder", which="6 pm")
        assert result.ok, result.message
        assert _run(registry, action_context, "list_reminders").data["count"] == 0

    def test_cancelling_by_subject(self, registry, action_context) -> None:
        _run(registry, action_context, "create_reminder",
             when="tomorrow at 9 AM", title="call the bank")
        result = _run(registry, action_context, "cancel_reminder", which="bank")
        assert result.ok

    def test_an_ambiguous_cancel_asks_instead_of_guessing(
            self, registry, action_context) -> None:
        _run(registry, action_context, "create_reminder",
             when="tomorrow at 9 AM", title="call the bank")
        _run(registry, action_context, "create_reminder",
             when="tomorrow at 11 AM", title="call the dentist")
        result = _run(registry, action_context, "cancel_reminder", which="call")
        assert not result.ok
        assert result.error_code == "ambiguous"

    def test_due_reminders_reports_only_what_has_arrived(
            self, registry, action_context, database) -> None:
        past = (local_now() - timedelta(minutes=5)).isoformat(timespec="seconds")
        database.add_reminder(reminder_id="old", title="already due", due_at=past)
        _run(registry, action_context, "create_reminder",
             when="in 3 hours", title="not yet")
        result = _run(registry, action_context, "due_reminders")
        assert result.data["count"] == 1
        assert "already due" in result.message


# ----------------------------------------------------------------- calendar

class TestCalendar:
    def test_creating_an_event_names_the_calendar_it_went_into(
            self, registry, action_context) -> None:
        """Never let the user assume it reached a calendar they actually use."""
        result = _run(registry, action_context, "create_event",
                      when="tomorrow at 10 AM", title="Project review")
        assert result.ok
        assert "AARMA's own calendar" in result.message
        assert "Google" not in result.message

    def test_a_missing_title_asks(self, registry, action_context) -> None:
        result = _run(registry, action_context, "create_event",
                      when="tomorrow at 10 AM")
        assert not result.ok
        assert result.data.get("needs") == "title"

    def test_a_missing_time_asks(self, registry, action_context) -> None:
        result = _run(registry, action_context, "create_event",
                      title="Project review", when="sometime")
        assert not result.ok
        assert result.data.get("needs") == "when"

    def test_listing_what_is_scheduled(self, registry, action_context) -> None:
        _run(registry, action_context, "create_event",
             when="tomorrow at 2 PM", title="Project review")
        result = _run(registry, action_context, "list_events", range="tomorrow")
        assert result.data["count"] == 1
        assert "Project review" in result.message

    def test_an_empty_calendar_says_so(self, registry, action_context) -> None:
        result = _run(registry, action_context, "list_events", range="tomorrow")
        assert result.ok
        assert result.data["count"] == 0

    def test_cancelling_an_event(self, registry, action_context) -> None:
        _run(registry, action_context, "create_event",
             when="tomorrow at 2 PM", title="Project review")
        result = _run(registry, action_context, "cancel_event", which="project")
        assert result.ok
        assert _run(registry, action_context, "list_events").data["count"] == 0

    def test_google_calendar_is_not_pretended_to_exist(self) -> None:
        from providers.calendar import google_calendar

        provider = google_calendar()
        assert provider.is_available() is False
        result = provider.create_event(title="x", starts_at=local_now())
        assert result.ok is False
        assert "isn't connected" in result.message


# ------------------------------------------------------------- capabilities

class TestCapabilityDiscovery:
    def test_it_answers_from_the_registry(self, registry, action_context) -> None:
        ctx = _context(action_context, registry=registry)
        result = _run(registry, ctx, "list_capabilities")
        assert result.ok
        assert result.data["total"] == len(registry)
        assert "Reminders" in result.data["categories"]

    def test_asking_about_a_topic_lists_real_examples(self, registry,
                                                      action_context) -> None:
        ctx = _context(action_context, registry=registry)
        result = _run(registry, ctx, "list_capabilities", topic="volume")
        assert result.ok
        assert "Media" in result.message

    def test_it_does_not_claim_a_capability_it_lacks(self, registry,
                                                     action_context) -> None:
        ctx = _context(action_context, registry=registry)
        result = _run(registry, ctx, "list_capabilities", topic="send a fax")
        assert result.ok
        assert "don't have anything" in result.message


# ------------------------------------------------------------- time parsing

class TestTimeParsing:
    def _ref(self, hour: int = 10) -> datetime:
        # A Friday, so weekday handling is exercised.
        return datetime(2026, 9, 11, hour, 30,
                        tzinfo=timezone(timedelta(hours=5, minutes=30)))

    @pytest.mark.parametrize("text,expected", [
        ("at 6 PM", "2026-09-11 18:00"),
        ("tomorrow at 9 AM", "2026-09-12 09:00"),
        ("in 30 minutes", "2026-09-11 11:00"),
        ("in an hour", "2026-09-11 11:30"),
        ("next Monday at 2pm", "2026-09-14 14:00"),
        ("noon", "2026-09-11 12:00"),
        ("at 18:30", "2026-09-11 18:30"),
    ])
    def test_known_shapes(self, text: str, expected: str) -> None:
        parsed = parse_when(text, reference=self._ref())
        assert parsed is not None, text
        assert parsed.when.strftime("%Y-%m-%d %H:%M") == expected

    def test_every_result_carries_a_timezone(self) -> None:
        """A naive datetime is a promise about no particular instant."""
        parsed = parse_when("tomorrow at 9 AM", reference=self._ref())
        assert parsed.when.tzinfo is not None

    def test_a_named_day_still_ahead_means_today(self) -> None:
        parsed = parse_when("Friday at 4 PM", reference=self._ref(10))
        assert parsed.when.strftime("%Y-%m-%d") == "2026-09-11"

    def test_a_named_day_already_past_means_next_week(self) -> None:
        parsed = parse_when("Friday at 4 PM", reference=self._ref(17))
        assert parsed.when.strftime("%Y-%m-%d") == "2026-09-18"

    def test_next_always_means_the_coming_one(self) -> None:
        parsed = parse_when("next Friday at 4 PM", reference=self._ref(10))
        assert parsed.when.strftime("%Y-%m-%d") == "2026-09-18"

    @pytest.mark.parametrize("text", ["submit my project", "the 25th", "", "hello"])
    def test_nonsense_returns_nothing_rather_than_a_guess(self, text: str) -> None:
        assert parse_when(text, reference=self._ref()) is None

    def test_a_day_without_a_time_is_flagged_as_assumed(self) -> None:
        parsed = parse_when("tomorrow", reference=self._ref())
        assert parsed is not None
        assert parsed.assumed_time is True
