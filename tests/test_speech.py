"""Voice output: providers, selection, filtering, delivery and persistence.

No test needs a sound card, a network connection or a real voice. Providers are
replaced with fakes that record what they were asked to say, which is also the
only way to assert the thing that matters most -- that a credential never
reaches a provider at all.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

import pytest

from config.settings import ConfigManager, VoiceSettings, load_config
from speech.audio import NullAudioPlayback
from speech.engine import SpeechEngine, SpeechOutcome, SpeechPurpose
from speech.speaker import (
    MAX_RATE,
    MIN_RATE,
    NullSpeechSpeaker,
    SpeechRequest,
    SpeechResult,
    SpeechSpeaker,
)
from speech.text import (
    contains_sensitive,
    naturalise,
    prepare_for_speech,
    preview_line,
    strip_payloads,
    strip_sensitive,
    summarise_lines,
)
from speech.voices import (
    ONLINE_VOICES,
    STYLE_PROSODY,
    VOICE_PROFILES,
    VoiceAvailability,
    VoiceQuality,
    VoiceStyle,
    get_profile,
    language_display_name,
    resolve_online_voice,
)


# ------------------------------------------------------------- fake providers

class FakeSpeaker(SpeechSpeaker):
    """A provider that records requests instead of making noise."""

    def __init__(self, name: str, *, requires_internet: bool = False,
                 quality: VoiceQuality = VoiceQuality.STANDARD,
                 available: bool = True, languages: Optional[tuple] = None,
                 fail: bool = False, duration: float = 0.0) -> None:
        self.name = name
        self.display_name = name
        self.requires_internet = requires_internet
        self.quality = quality
        self._available = available
        self._languages = languages if languages is not None else ("en",)
        self._fail = fail
        self._duration = duration
        self.requests: List[SpeechRequest] = []
        self.stop_calls = 0
        self._stop = threading.Event()

    def is_available(self) -> bool:
        return self._available

    def describe_voice(self, language, voice_id, style=None) -> VoiceAvailability:
        base = (language or "en").split("-")[0]
        if not self._available:
            return VoiceAvailability(False, self.name, VoiceQuality.UNAVAILABLE,
                                     language=language, reason=f"{self.name} is unavailable.")
        if base not in self._languages:
            return VoiceAvailability(
                False, self.name, VoiceQuality.UNAVAILABLE, language=language,
                reason=f"There's no {language_display_name(base)} voice in {self.name}.",
                suggestion="Try another voice source.")
        return VoiceAvailability(True, self.name, self.quality,
                                 voice_name=f"{self.name}-{base}", language=language,
                                 online=self.requires_internet)

    def speak(self, request: SpeechRequest) -> SpeechResult:
        self.requests.append(request)
        self._stop.clear()
        if self._fail:
            return SpeechResult.failure(self.name, "Synthesis failed.",
                                        error_code="synthesis_failed")
        if self._duration:
            deadline = time.time() + self._duration
            while time.time() < deadline:
                if self._stop.is_set():
                    return SpeechResult.cancelled(self.name)
                time.sleep(0.01)
        return SpeechResult.success(self.name)

    def stop(self) -> None:
        self.stop_calls += 1
        self._stop.set()


class FakeConnectivity:
    def __init__(self, online: bool = True) -> None:
        self.online = online

    def is_online(self) -> bool:
        return self.online


class FakePermissions:
    def __init__(self, granted: bool = True) -> None:
        self.granted = granted

    def is_granted(self, _permission) -> bool:
        return self.granted


@pytest.fixture
def offline_provider() -> FakeSpeaker:
    return FakeSpeaker("windows", requires_internet=False,
                       quality=VoiceQuality.STANDARD, languages=("en",))


@pytest.fixture
def online_provider() -> FakeSpeaker:
    return FakeSpeaker("neural", requires_internet=True,
                       quality=VoiceQuality.NEURAL,
                       languages=tuple(ONLINE_VOICES))


@pytest.fixture
def speech(config: ConfigManager, offline_provider, online_provider):
    engine = SpeechEngine(config, providers=[online_provider, offline_provider],
                          connectivity=FakeConnectivity(True),
                          permissions=FakePermissions(True))
    yield engine
    engine.shutdown()


def wait_for(outcomes: List[SpeechOutcome], timeout: float = 4.0) -> Optional[SpeechOutcome]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if outcomes:
            return outcomes.pop(0)
        time.sleep(0.01)
    return None


@pytest.fixture
def outcomes(speech: SpeechEngine) -> List[SpeechOutcome]:
    collected: List[SpeechOutcome] = []
    speech.on_finished(collected.append)
    return collected


# --------------------------------------------------------------- voice catalog

class TestVoiceProfiles:
    def test_asma_and_aariz_defaults_are_the_shipped_profiles(self) -> None:
        from config.settings import ASSISTANT_PROFILES

        assert ASSISTANT_PROFILES["asma"].voice_id == "female_warm"
        assert ASSISTANT_PROFILES["aariz"].voice_id == "male_calm"
        assert get_profile("female_warm").gender == "female"
        assert get_profile("male_calm").gender == "male"

    def test_asma_default_matches_the_requested_character(self) -> None:
        profile = get_profile("female_warm")
        assert profile.default_style is VoiceStyle.WARM
        assert profile.display_name == "Warm"

    def test_aariz_default_matches_the_requested_character(self) -> None:
        profile = get_profile("male_calm")
        assert profile.default_style is VoiceStyle.CALM
        assert profile.display_name == "Calm"

    def test_every_profile_has_both_genders_represented(self) -> None:
        genders = {p.gender for p in VOICE_PROFILES.values()}
        assert genders == {"female", "male"}

    def test_unknown_voice_id_falls_back_rather_than_raising(self) -> None:
        assert get_profile("no_such_voice").voice_id == "female_warm"

    def test_quality_tiers_are_described_honestly(self) -> None:
        assert VoiceQuality.NEURAL.label == "Natural"
        assert VoiceQuality.STANDARD.label == "Standard"
        # The fallback must not be described as equivalent to the preferred voice.
        assert "synthetic" in VoiceQuality.STANDARD.description.lower()

    def test_styles_stay_within_comfortable_prosody(self) -> None:
        for style, prosody in STYLE_PROSODY.items():
            assert 0.85 <= prosody.rate_scale <= 1.15, style
            assert -3.0 <= prosody.pitch_offset <= 3.0, style


class TestOnlineVoiceResolution:
    @pytest.mark.parametrize("language",
                             ["en", "hi", "kn", "te", "ta", "ml", "ur", "ar", "es", "fr"])
    def test_every_catalogue_language_has_both_genders(self, language: str) -> None:
        female = resolve_online_voice(language, "female_warm")
        male = resolve_online_voice(language, "male_calm")
        assert female is not None and male is not None, language
        assert female.name != male.name

    def test_language_variants_resolve_to_the_base_language(self) -> None:
        assert resolve_online_voice("en-IN", "female_warm") is not None
        assert resolve_online_voice("hi-IN", "male_calm") is not None

    def test_unknown_language_resolves_to_nothing(self) -> None:
        assert resolve_online_voice("klingon", "female_warm") is None

    def test_english_defaults_to_an_indian_english_voice(self) -> None:
        voice = resolve_online_voice("en", "female_warm", VoiceStyle.WARM)
        assert voice is not None and voice.name.startswith("en-IN")

    def test_style_picks_a_different_voice_only_where_one_exists(self) -> None:
        warm = resolve_online_voice("en", "female_warm", VoiceStyle.WARM)
        clear = resolve_online_voice("en", "female_warm", VoiceStyle.CLEAR)
        assert warm.name != clear.name          # English has several.

        # Hindi ships one voice per gender: the style is prosody only, and the
        # resolver says so rather than implying a choice it did not make.
        hindi = resolve_online_voice("hi", "female_warm", VoiceStyle.CLEAR)
        assert hindi.style_matched_voice is False

    def test_resolved_voices_are_marked_neural(self) -> None:
        assert resolve_online_voice("ta", "male_calm").quality is VoiceQuality.NEURAL

    def test_language_names_are_spoken_not_coded(self) -> None:
        assert language_display_name("kn") == "Kannada"
        assert language_display_name("hi-IN") == "Hindi"


# ---------------------------------------------------------------- text safety

class TestSpeechSafety:
    @pytest.mark.parametrize("text", [
        "api_key=sk-abcdefghijklmnopqrst",
        "password: hunter2",
        "The passphrase is correct-horse-battery",
        "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345",
        "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9",
        "AKIAIOSFODNN7EXAMPLE",
        "d41d8cd98f00b204e9800998ecf8427e0123456789abcdef",
    ])
    def test_credentials_are_detected(self, text: str) -> None:
        assert contains_sensitive(text)

    @pytest.mark.parametrize("text", [
        "The current time is 9:10 AM.",
        "Created 'AI Project' in Documents.",
        "I opened Chrome for you.",
        "109 items in Documents.",
    ])
    def test_ordinary_replies_are_not_flagged(self, text: str) -> None:
        assert not contains_sensitive(text)

    @pytest.mark.parametrize("secret", ["hunter2", "sk-abcdefghijklmnopqrst"])
    def test_secrets_never_survive_preparation(self, secret: str) -> None:
        prepared = prepare_for_speech(f"Your password: {secret} is set.")
        assert secret not in prepared.text
        assert prepared.redacted

    def test_a_redacted_reply_says_one_clear_thing(self) -> None:
        prepared = prepare_for_speech("The password is hunter2 for the account.")
        assert "hunter2" not in prepared.text
        assert "on screen" in prepared.text

    def test_strip_sensitive_reports_whether_it_changed_anything(self) -> None:
        _clean, changed = strip_sensitive("nothing to see here")
        assert changed is False
        _clean, changed = strip_sensitive("password=hunter2")
        assert changed is True

    def test_json_payloads_are_not_spoken(self) -> None:
        prepared = prepare_for_speech('Result: {"action": "get_time", "ok": true}')
        assert "action" not in prepared.text
        assert "{" not in prepared.text

    def test_code_blocks_are_not_spoken(self) -> None:
        assert "import os" not in strip_payloads("Here:\n```\nimport os\n```\ndone")

    def test_engine_never_hands_a_secret_to_a_provider(
        self, speech: SpeechEngine, online_provider: FakeSpeaker,
        outcomes: List[SpeechOutcome]
    ) -> None:
        """The assertion that matters: filtering happens before synthesis."""
        speech.speak("Your api_key=sk-abcdefghijklmnopqrst is ready.")
        assert wait_for(outcomes) is not None
        assert online_provider.requests
        assert "sk-abcdefghijklmnopqrst" not in online_provider.requests[0].text


class TestNaturalSpeech:
    def test_windows_paths_become_folder_names(self) -> None:
        spoken = prepare_for_speech(
            r"Created 'AI Project' in C:\Users\me\OneDrive\Documents.").text
        assert "C:" not in spoken
        assert "Documents" in spoken

    def test_optional_plurals_are_folded(self) -> None:
        assert "item, s" not in naturalise("109 item(s) in Documents")
        assert "items" in naturalise("109 item(s) in Documents")

    def test_arrows_are_spoken_as_words(self) -> None:
        spoken = naturalise("Enable it in Settings → Permissions")
        assert "→" not in spoken
        assert "then" in spoken

    def test_bullets_and_emphasis_are_removed(self) -> None:
        assert "*" not in naturalise("* **bold** item")

    def test_long_listings_are_summarised_to_their_first_line(self) -> None:
        listing = "109 items in Documents:\n  a.txt\n  b.txt\n  c.txt"
        summarised, condensed = summarise_lines(listing)
        assert summarised == "109 items in Documents:"
        assert condensed

    def test_short_replies_are_left_alone(self) -> None:
        prepared = prepare_for_speech("The current time is 9:10 AM.")
        assert prepared.text == "The current time is 9:10 AM."
        assert not prepared.condensed

    def test_replies_are_capped_at_a_listenable_length(self) -> None:
        prepared = prepare_for_speech(" ".join(["word"] * 400))
        assert len(prepared.text) <= 340
        assert prepared.condensed

    def test_empty_input_produces_nothing_speakable(self) -> None:
        assert not prepare_for_speech("   ").speakable

    def test_preview_uses_the_configured_assistant_name(self) -> None:
        assert preview_line("Nova") == "Hello, I'm Nova. I'm ready to help."
        assert "Asma" not in preview_line("Nova")


# ------------------------------------------------------------ provider choice

class TestProviderSelection:
    def test_online_neural_is_preferred_when_connected(self, speech: SpeechEngine) -> None:
        provider, availability, notice = speech.resolve()
        assert provider.name == "neural"
        assert availability.quality is VoiceQuality.NEURAL
        assert notice == ""

    def test_offline_is_used_when_disconnected(self, speech: SpeechEngine) -> None:
        speech._connectivity.online = False
        provider, availability, _notice = speech.resolve()
        assert provider.name == "windows"
        assert availability.quality is VoiceQuality.STANDARD

    def test_offline_mode_never_reaches_for_the_network(self, speech: SpeechEngine,
                                                        config: ConfigManager) -> None:
        config.update_voice(provider_mode="offline")
        provider, _availability, _notice = speech.resolve()
        assert provider.name == "windows"

    def test_online_mode_does_not_fall_back_to_offline(self, speech: SpeechEngine,
                                                       config: ConfigManager,
                                                       online_provider) -> None:
        config.update_voice(provider_mode="online")
        online_provider._available = False
        provider, availability, _notice = speech.resolve()
        assert provider is None
        assert not availability.available

    def test_fallback_is_announced_not_silent(self, speech: SpeechEngine,
                                              online_provider) -> None:
        """A silent downgrade reads as 'this product sounds bad'."""
        online_provider._available = False
        provider, availability, notice = speech.resolve()
        assert provider.name == "windows"
        assert availability.available
        assert notice, "falling back to a worse voice must be explained"

    def test_a_missing_language_is_refused_not_substituted(
        self, speech: SpeechEngine, config: ConfigManager
    ) -> None:
        """An English voice reading Kannada is a different failure, not a fallback."""
        config.update_voice(provider_mode="offline")
        provider, availability, _notice = speech.resolve(language="kn")
        assert provider is None
        assert not availability.available
        assert "Kannada" in availability.reason
        assert availability.suggestion

    def test_online_covers_a_language_offline_cannot(self, speech: SpeechEngine) -> None:
        provider, availability, _notice = speech.resolve(language="kn")
        assert provider.name == "neural"
        assert availability.available

    def test_availability_reports_the_permission_when_denied(
        self, config: ConfigManager, offline_provider, online_provider
    ) -> None:
        engine = SpeechEngine(config, providers=[online_provider, offline_provider],
                              connectivity=FakeConnectivity(True),
                              permissions=FakePermissions(False), autostart=False)
        availability = engine.availability()
        assert not availability.available
        assert "Speaker" in availability.reason

    def test_no_providers_at_all_is_reported_gracefully(
        self, config: ConfigManager
    ) -> None:
        engine = SpeechEngine(config, providers=[NullSpeechSpeaker()],
                              connectivity=FakeConnectivity(False), autostart=False)
        availability = engine.availability()
        assert not availability.available
        assert availability.reason


# ---------------------------------------------------------------- the engine

class TestSpeechEngine:
    def test_speaking_reaches_the_chosen_provider(
        self, speech: SpeechEngine, online_provider, outcomes
    ) -> None:
        assert speech.speak("Hello there.")
        outcome = wait_for(outcomes)
        assert outcome is not None and outcome.ok
        assert online_provider.requests[0].text == "Hello there."

    def test_preview_speaks_the_current_assistant_name(
        self, speech: SpeechEngine, config: ConfigManager, online_provider, outcomes
    ) -> None:
        config.update_assistant(name="Nova")
        assert speech.speak_preview()
        assert wait_for(outcomes) is not None
        assert "Nova" in online_provider.requests[0].text

    def test_disabled_voice_says_nothing(self, speech: SpeechEngine,
                                         config: ConfigManager, online_provider) -> None:
        config.update_voice(enabled=False)
        assert speech.speak("Hello.") is False
        assert online_provider.requests == []

    def test_preview_still_works_while_replies_are_muted(
        self, speech: SpeechEngine, config: ConfigManager, online_provider, outcomes
    ) -> None:
        """Otherwise Test voice is unusable exactly when you want to try a voice."""
        config.update_voice(enabled=False)
        assert speech.speak_preview()
        assert wait_for(outcomes) is not None
        assert online_provider.requests

    def test_denied_speaker_permission_blocks_audio(
        self, config: ConfigManager, offline_provider, online_provider
    ) -> None:
        engine = SpeechEngine(config, providers=[online_provider, offline_provider],
                              connectivity=FakeConnectivity(True),
                              permissions=FakePermissions(False))
        try:
            assert engine.speak("Hello.") is False
            assert engine.speak_preview() is False
            assert online_provider.requests == []
        finally:
            engine.shutdown()

    def test_notifications_can_be_muted_separately(
        self, speech: SpeechEngine, config: ConfigManager
    ) -> None:
        config.update_voice(speak_notifications=False)
        assert speech.speak("Reminder.", purpose=SpeechPurpose.NOTIFICATION) is False
        assert speech.speak("A reply.") is not False

    def test_notifications_use_a_calmer_delivery(
        self, speech: SpeechEngine, online_provider, outcomes
    ) -> None:
        speech.speak("Time to submit your assignment.",
                     purpose=SpeechPurpose.NOTIFICATION)
        assert wait_for(outcomes) is not None
        assert online_provider.requests[0].style is VoiceStyle.CALM

    def test_delivery_settings_reach_the_provider(
        self, speech: SpeechEngine, config: ConfigManager, online_provider, outcomes
    ) -> None:
        config.update_voice(rate=1.2, volume=0.5, pitch=3.0)
        speech.speak("Testing.")
        assert wait_for(outcomes) is not None
        request = online_provider.requests[0]
        assert request.rate == pytest.approx(1.2)
        assert request.volume == pytest.approx(0.5)
        assert request.pitch == pytest.approx(3.0)

    def test_out_of_range_delivery_values_are_clamped(self) -> None:
        wild = SpeechRequest(text="hi", rate=99.0, volume=-4.0, pitch=500.0).normalised()
        assert MIN_RATE <= wild.rate <= MAX_RATE
        assert 0.0 <= wild.volume <= 1.0
        assert -10.0 <= wild.pitch <= 10.0

    def test_failures_are_reported_not_raised(
        self, config: ConfigManager, outcomes=None
    ) -> None:
        failing = FakeSpeaker("broken", fail=True)
        engine = SpeechEngine(config, providers=[failing],
                              connectivity=FakeConnectivity(False))
        collected: List[SpeechOutcome] = []
        engine.on_finished(collected.append)
        try:
            engine.speak("Hello.")
            outcome = wait_for(collected)
            assert outcome is not None
            assert not outcome.ok
            assert outcome.error_code == "synthesis_failed"
        finally:
            engine.shutdown()

    def test_a_listener_that_raises_does_not_break_speech(
        self, speech: SpeechEngine, outcomes
    ) -> None:
        def explode(_outcome):
            raise RuntimeError("listener fault")

        speech.on_finished(explode)
        speech.speak("Hello.")
        assert wait_for(outcomes) is not None

    def test_stop_cancels_the_current_utterance(self, config: ConfigManager) -> None:
        slow = FakeSpeaker("slow", duration=3.0)
        engine = SpeechEngine(config, providers=[slow],
                              connectivity=FakeConnectivity(False))
        collected: List[SpeechOutcome] = []
        engine.on_finished(collected.append)
        try:
            engine.speak("A long sentence.")
            time.sleep(0.3)
            engine.stop()
            outcome = wait_for(collected)
            assert outcome is not None
            assert outcome.interrupted
            assert slow.stop_calls >= 1
        finally:
            engine.shutdown()

    def test_speaking_replaces_rather_than_overlaps_by_default(
        self, config: ConfigManager
    ) -> None:
        """Two replies talking over each other helps nobody."""
        slow = FakeSpeaker("slow", duration=1.0)
        engine = SpeechEngine(config, providers=[slow],
                              connectivity=FakeConnectivity(False))
        try:
            engine.speak("First utterance.")
            time.sleep(0.2)
            engine.speak("Second utterance.")
            time.sleep(0.5)
            assert slow.stop_calls >= 1
        finally:
            engine.shutdown()

    def test_speak_returns_immediately(self, speech: SpeechEngine,
                                       config: ConfigManager) -> None:
        """The interface must never wait for audio."""
        speech._providers = [FakeSpeaker("slow", duration=2.0)]
        config.update_voice(provider_mode="offline")
        started = time.time()
        speech.speak("Hello.")
        assert time.time() - started < 0.5

    def test_is_speaking_tracks_state(self, config: ConfigManager) -> None:
        slow = FakeSpeaker("slow", duration=1.0)
        engine = SpeechEngine(config, providers=[slow],
                              connectivity=FakeConnectivity(False))
        try:
            assert not engine.is_speaking
            engine.speak("Hello.")
            time.sleep(0.3)
            assert engine.is_speaking
            engine.stop()
            time.sleep(0.3)
            assert not engine.is_speaking
        finally:
            engine.shutdown()

    def test_shutdown_is_idempotent(self, config: ConfigManager) -> None:
        engine = SpeechEngine(config, providers=[FakeSpeaker("x")])
        engine.shutdown()
        engine.shutdown()

    def test_outcome_records_that_text_was_shortened(
        self, speech: SpeechEngine, outcomes
    ) -> None:
        speech.speak("Line one.\nLine two.\nLine three.\nLine four.")
        outcome = wait_for(outcomes)
        assert outcome is not None and outcome.condensed


# ------------------------------------------------------------- configuration

class TestVoiceConfiguration:
    def test_defaults_are_sensible(self, config: ConfigManager) -> None:
        voice = config.config.voice
        assert voice.enabled is True
        assert voice.provider_mode == "auto"
        assert voice.style == "warm"
        assert voice.rate == pytest.approx(1.0)
        assert 0.0 <= voice.volume <= 1.0

    def test_settings_persist_across_a_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        first = ConfigManager(path)
        first.load()
        first.update_voice(provider_mode="offline", rate=1.25, volume=0.4,
                           pitch=-3.0, style="calm", enabled=False)

        reloaded = load_config(path)
        voice = reloaded.config.voice
        assert voice.provider_mode == "offline"
        assert voice.rate == pytest.approx(1.25)
        assert voice.volume == pytest.approx(0.4)
        assert voice.pitch == pytest.approx(-3.0)
        assert voice.style == "calm"
        assert voice.enabled is False

    def test_a_config_written_before_voice_existed_still_loads(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "config.json"
        path.write_text('{"assistant": {"name": "Nova"}}', encoding="utf-8")
        manager = load_config(path)
        assert manager.config.assistant.name == "Nova"
        assert manager.config.voice == VoiceSettings()

    def test_unknown_style_falls_back_to_the_profile_default(
        self, speech: SpeechEngine, config: ConfigManager
    ) -> None:
        config.update_voice(style="interpretive-dance")
        assert speech.current_style() is VoiceStyle.WARM


# -------------------------------------------------------------------- audio

class TestAudioPlayback:
    def test_null_playback_records_without_making_sound(self) -> None:
        playback = NullAudioPlayback()
        assert playback.play(Path("nowhere.mp3"))
        assert playback.played == [Path("nowhere.mp3")]

    def test_unavailable_playback_refuses(self) -> None:
        playback = NullAudioPlayback(available=False)
        assert not playback.is_available()
        assert not playback.play(Path("nowhere.mp3"))

    def test_stop_is_safe_when_nothing_is_playing(self) -> None:
        playback = NullAudioPlayback()
        playback.stop()
        assert playback.stop_calls == 1


class TestNullSpeaker:
    def test_never_claims_success(self) -> None:
        speaker = NullSpeechSpeaker()
        assert not speaker.is_available()
        result = speaker.speak(SpeechRequest(text="hello"))
        assert not result.ok
        assert result.error_code == "no_provider"

    def test_describes_itself_as_unavailable_with_a_suggestion(self) -> None:
        availability = NullSpeechSpeaker().describe_voice("en", "female_warm")
        assert not availability.available
        assert availability.suggestion


# ------------------------------------------------- real providers, no hardware

class TestRealProvidersDegradeGracefully:
    """Constructing and interrogating the shipped providers must never raise."""

    def test_edge_provider_reports_availability_without_network_access(self) -> None:
        from speech.providers.edge import EdgeNeuralSpeaker

        provider = EdgeNeuralSpeaker(playback=NullAudioPlayback(available=False))
        assert provider.is_available() is False
        availability = provider.describe_voice("en", "female_warm")
        assert not availability.available
        assert availability.reason

    def test_edge_provider_maps_every_catalogue_language(self) -> None:
        from speech.providers.edge import EdgeNeuralSpeaker

        provider = EdgeNeuralSpeaker(playback=NullAudioPlayback())
        for language in ONLINE_VOICES:
            assert provider.voices(language), language

    def test_edge_provider_refuses_an_unmapped_language(self) -> None:
        from speech.providers.edge import EdgeNeuralSpeaker

        provider = EdgeNeuralSpeaker(playback=NullAudioPlayback())
        result = provider.speak(SpeechRequest(text="hello", language="klingon"))
        assert not result.ok
        assert result.error_code in {"voice_unavailable", "provider_unavailable"}

    def test_sapi_provider_constructs_on_any_platform(self) -> None:
        from speech.providers.sapi import SapiSpeaker, friendly_voice_name

        provider = SapiSpeaker()
        assert provider.requires_internet is False
        assert provider.quality is VoiceQuality.STANDARD
        # Availability depends on the host; the call must simply not raise.
        provider.is_available()
        assert friendly_voice_name(
            "Microsoft Zira Desktop - English (United States)") == "Zira"

    def test_default_providers_always_returns_something(self) -> None:
        from speech.engine import default_providers

        providers = default_providers()
        assert providers
        for provider in providers:
            assert isinstance(provider.name, str)
