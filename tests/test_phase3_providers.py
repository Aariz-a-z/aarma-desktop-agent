"""Weather, email, provider availability, and the multi-turn dialogue end to end.

No network is touched.  The weather provider's transport is injected, so the
tests cover the paths that matter most and are hardest to arrange for real: an
unreachable service, a malformed payload, a place that does not exist, and a
reading with fields missing.

The property every one of these defends is the same, and it is the one the
whole provider layer exists for:

    **A provider that cannot do something says so.  It never pretends.**

That is why there are as many tests for the unavailable paths as for the happy
one.  A fabricated temperature or a false "email sent" is invisible to the
user -- they dress for it, or they stop checking -- so the failure modes are
where the testing effort belongs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from actions.registry import ActionContext, build_default_registry
from core.engine import ResponseKind
from providers import UnavailableProvider
from providers.email import (
    EmailDraft,
    LocalDraftEmailProvider,
    gmail_email,
    invalid_recipients,
    outlook_email,
    smtp_email,
    split_recipients,
)
from providers.weather import (
    FORECAST_URL,
    GEOCODE_URL,
    OpenMeteoProvider,
    WeatherReading,
    condition_text,
)
from security.validator import RiskLevel


# ----------------------------------------------------------------- fixtures

class FakeTransport:
    """Serves canned JSON and remembers every URL it was given."""

    def __init__(self, geocode: Any = None, forecast: Any = None, *,
                 raises: Optional[Exception] = None) -> None:
        self.geocode = geocode
        self.forecast = forecast
        self.raises = raises
        self.urls: List[str] = []

    def __call__(self, url: str) -> Any:
        self.urls.append(url)
        if self.raises is not None:
            raise self.raises
        if url.startswith(GEOCODE_URL):
            return self.geocode
        return self.forecast


BENGALURU = {"results": [{"name": "Bengaluru", "latitude": 12.97,
                          "longitude": 77.59, "country": "India",
                          "admin1": "Karnataka", "timezone": "Asia/Kolkata"}]}

CURRENT = {"current": {"time": "2026-09-11T10:00", "temperature_2m": 27.4,
                       "apparent_temperature": 29.1, "weather_code": 61,
                       "relative_humidity_2m": 78, "wind_speed_10m": 11.2}}

FORECAST = {"daily": {"time": ["2026-09-11", "2026-09-12"],
                      "weather_code": [3, 61],
                      "temperature_2m_max": [29.0, 27.5],
                      "temperature_2m_min": [21.0, 20.5],
                      "precipitation_probability_max": [10, 80]}}


@pytest.fixture
def registry():
    return build_default_registry()


def _context(action_context: ActionContext, **overrides) -> ActionContext:
    for key, value in overrides.items():
        object.__setattr__(action_context, key, value)
    return action_context


def _run(registry, ctx, action: str, **params):
    spec = registry.require(action)
    return spec.handler(ctx, spec.validate(params))


def _weather(**kwargs) -> OpenMeteoProvider:
    transport = kwargs.pop("transport", FakeTransport(BENGALURU, CURRENT))
    return OpenMeteoProvider(fetch=transport, **kwargs)


# ----------------------------------------------------------------- weather

class TestWeatherProvider:

    def test_a_real_reading_is_reported(self) -> None:
        provider = _weather()
        result = provider.current("Bengaluru")
        assert result.ok
        reading = result.data["reading"]
        assert reading["temperature_c"] == 27.4
        assert reading["condition"] == "light rain"
        assert reading["humidity_percent"] == 78

    def test_only_the_place_name_leaves_the_machine(self) -> None:
        """Nothing identifying is sent -- no account, no address, no utterance."""
        transport = FakeTransport(BENGALURU, CURRENT)
        _weather(transport=transport).current("Bengaluru")
        joined = " ".join(transport.urls).lower()
        assert "bengaluru" in joined
        for leak in ("aarma", "@", "token", "key=", "apikey"):
            assert leak not in joined

    def test_an_unreachable_service_is_a_failure_not_a_forecast(self) -> None:
        provider = _weather(transport=FakeTransport(raises=OSError("no route")))
        result = provider.current("Bengaluru")
        assert not result.ok
        assert result.error_code in {"fetch_failed", "no_location"}
        assert "temperature" not in result.message.lower()

    def test_a_malformed_payload_is_a_failure(self) -> None:
        provider = _weather(transport=FakeTransport(BENGALURU, {"nonsense": 1}))
        result = provider.current("Bengaluru")
        assert not result.ok
        assert result.error_code == "bad_payload"

    def test_an_unknown_place_is_named_as_unknown(self) -> None:
        provider = _weather(transport=FakeTransport({"results": []}, CURRENT))
        result = provider.current("Nowherefordshire")
        assert not result.ok
        assert "Nowherefordshire" in result.message

    def test_no_location_asks_rather_than_guessing(self) -> None:
        """No IP geolocation: it would answer about somewhere never mentioned."""
        provider = _weather()
        result = provider.current("")
        assert not result.ok
        assert result.error_code == "no_location"

    def test_a_configured_location_is_used_when_none_is_named(self) -> None:
        provider = _weather(default_location="Bengaluru")
        assert provider.current("").ok

    def test_offline_is_unavailable_not_empty(self) -> None:
        provider = _weather(online=False)
        assert not provider.is_available()
        result = provider.current("Bengaluru")
        assert not result.ok
        assert "offline" in result.message.lower()

    def test_missing_fields_are_none_not_zero(self) -> None:
        sparse = {"current": {"time": "2026-09-11T10:00", "temperature_2m": 18.0}}
        provider = _weather(transport=FakeTransport(BENGALURU, sparse))
        reading = provider.current("Bengaluru").data["reading"]
        assert reading["temperature_c"] == 18.0
        assert reading["humidity_percent"] is None
        assert reading["wind_kph"] is None

    def test_an_unknown_weather_code_is_not_rounded_to_a_familiar_word(self) -> None:
        odd = {"current": {"temperature_2m": 18.0, "weather_code": 4242}}
        provider = _weather(transport=FakeTransport(BENGALURU, odd))
        assert provider.current("Bengaluru").data["reading"]["condition"] is None

    def test_clear_is_code_zero_and_not_treated_as_missing(self) -> None:
        assert condition_text(0) == "clear"

    def test_forecast_days_are_parsed(self) -> None:
        provider = _weather(transport=FakeTransport(BENGALURU, FORECAST))
        result = provider.forecast("Bengaluru", days=2)
        assert result.ok
        days = result.data["days"]
        assert [d["date"] for d in days] == ["2026-09-11", "2026-09-12"]
        assert days[1]["precipitation_chance"] == 80


class TestWeatherSummaries:

    def test_a_summary_omits_what_was_not_measured(self) -> None:
        reading = WeatherReading(place="Leeds", temperature_c=11.0)
        text = reading.summary()
        assert "11°C" in text
        for absent in ("humidity", "wind", "feels like"):
            assert absent not in text

    def test_a_reading_with_nothing_in_it_says_so(self) -> None:
        text = WeatherReading(place="Leeds").summary()
        assert "no temperature or conditions" in text


class TestWeatherActions:

    def test_current_weather_reads_the_provider(self, registry,
                                                action_context) -> None:
        ctx = _context(action_context, weather_provider=_weather(), online=True)
        result = _run(registry, ctx, "get_weather", location="Bengaluru")
        assert result.ok
        assert "27°C" in result.message

    def test_unavailable_weather_is_honest(self, registry,
                                           action_context) -> None:
        ctx = _context(action_context, weather_provider=_weather(online=False))
        result = _run(registry, ctx, "get_weather")
        assert not result.ok
        assert result.error_code == "weather_unavailable"

    def test_no_figure_is_invented_when_the_service_fails(self, registry,
                                                          action_context) -> None:
        broken = _weather(transport=FakeTransport(raises=OSError("down")),
                          default_location="Leeds")
        ctx = _context(action_context, weather_provider=broken, online=True)
        result = _run(registry, ctx, "get_weather")
        assert not result.ok
        assert not any(ch.isdigit() for ch in result.message), result.message

    def test_rain_question_quotes_the_probability(self, registry,
                                                  action_context) -> None:
        provider = _weather(transport=FakeTransport(BENGALURU, FORECAST))
        ctx = _context(action_context, weather_provider=provider, online=True)
        result = _run(registry, ctx, "will_it_rain", location="Bengaluru",
                      when="tomorrow")
        assert result.ok
        assert "80%" in result.message
        assert "probably" in result.message

    def test_a_missing_probability_is_not_answered_as_no(self, registry,
                                                         action_context) -> None:
        no_chance = {"daily": {"time": ["2026-09-11", "2026-09-12"],
                               "weather_code": [3, 3],
                               "temperature_2m_max": [20, 20],
                               "temperature_2m_min": [10, 10]}}
        provider = _weather(transport=FakeTransport(BENGALURU, no_chance))
        ctx = _context(action_context, weather_provider=provider, online=True)
        result = _run(registry, ctx, "will_it_rain", location="Bengaluru")
        assert result.ok
        assert "doesn't include a rain probability" in result.message
        assert result.data["precipitation_chance"] is None

    def test_weather_needs_the_internet_flag(self, registry) -> None:
        assert registry.require("get_weather").requires_internet is True


# ------------------------------------------------------------------- email

class TestEmailDrafting:

    def test_a_draft_is_stored_and_listed(self, registry, database,
                                          action_context) -> None:
        result = _run(registry, action_context, "draft_email",
                      to="sam@example.com", subject="Notes",
                      body="Here are the notes.")
        assert result.ok, result.message
        listing = _run(registry, action_context, "list_drafts")
        assert listing.data["count"] == 1
        assert listing.data["drafts"][0]["subject"] == "Notes"

    def test_a_draft_says_it_is_only_a_draft(self, registry,
                                             action_context) -> None:
        result = _run(registry, action_context, "draft_email",
                      to="sam@example.com", subject="Notes", body="Body.")
        assert "AARMA's own drafts" in result.message
        assert "can't send it" in result.message

    def test_a_stored_draft_is_never_marked_sent(self, registry, database,
                                                 action_context) -> None:
        _run(registry, action_context, "draft_email", to="sam@example.com",
             subject="Notes", body="Body.")
        row = database.email_drafts()[0]
        assert row["sent_at"] == ""
        assert row["state"] == "draft"

    @pytest.mark.parametrize("address", ["not an address", "sam@", "@example.com",
                                         "sam example.com"])
    def test_a_malformed_recipient_is_refused(self, registry, action_context,
                                              address) -> None:
        result = _run(registry, action_context, "draft_email", to=address,
                      subject="Hi", body="Body.")
        assert not result.ok

    def test_missing_pieces_name_the_slot_they_need(self, registry,
                                                    action_context) -> None:
        no_subject = _run(registry, action_context, "draft_email",
                          to="sam@example.com")
        assert not no_subject.ok
        assert no_subject.data["needs"] == "subject"

    def test_recipient_splitting(self) -> None:
        assert split_recipients("a@b.com, c@d.com and e@f.com") == [
            "a@b.com", "c@d.com", "e@f.com"]

    def test_address_shape_check(self) -> None:
        assert invalid_recipients(["a@b.com"]) == []
        assert invalid_recipients(["nope"]) == ["nope"]

    def test_a_draft_can_be_discarded(self, registry, action_context) -> None:
        _run(registry, action_context, "draft_email", to="sam@example.com",
             subject="Bin me", body="Body.")
        result = _run(registry, action_context, "discard_draft", which="bin me")
        assert result.ok
        assert _run(registry, action_context, "list_drafts").data["count"] == 0


class TestEmailSending:

    def test_send_is_refused_with_a_reason(self, registry,
                                           action_context) -> None:
        _run(registry, action_context, "draft_email", to="sam@example.com",
             subject="Notes", body="Body.")
        result = _run(registry, action_context, "send_email")
        assert not result.ok
        assert result.error_code == "no_transport"
        assert result.data["sent"] is False
        assert "no mail account is connected" in result.message.lower()

    def test_nothing_ever_says_sent_without_a_provider(self, registry,
                                                       action_context) -> None:
        _run(registry, action_context, "draft_email", to="sam@example.com",
             subject="Notes", body="Body.")
        result = _run(registry, action_context, "send_email")
        assert "sent" not in result.message.lower().replace("no mail", "")[:40]
        assert result.ok is False

    def test_a_provider_that_confirms_is_reported_as_sent(self, registry,
                                                          action_context) -> None:
        """The success path exists and is reached only on a real confirmation."""
        from providers import ProviderResult

        class ConnectedProvider(LocalDraftEmailProvider):
            name = "test-transport"
            can_send = True

            def send(self, draft_id: str) -> ProviderResult:
                return ProviderResult.success("Delivered.", message_id="abc123")

        ctx = _context(action_context,
                       email_provider=ConnectedProvider(action_context.database))
        _run(registry, ctx, "draft_email", to="sam@example.com",
             subject="Notes", body="Body.")
        result = _run(registry, ctx, "send_email")
        assert result.ok
        assert result.data["sent"] is True
        assert "test-transport" in result.message

    def test_send_email_is_high_risk(self, registry) -> None:
        """A sent message cannot be recalled, so it always needs a yes."""
        assert registry.require("send_email").risk is RiskLevel.HIGH

    def test_the_engine_demands_confirmation_before_sending(self, engine) -> None:
        from security.permissions import Permission

        engine.permissions.grant(Permission.EMAIL)
        engine.permissions.grant(Permission.INTERNET)
        response = engine.handle_text("send this email")
        assert response.kind is ResponseKind.CONFIRMATION_REQUIRED

    def test_email_status_states_the_gap(self, registry, action_context) -> None:
        result = _run(registry, action_context, "email_status")
        assert result.ok
        assert result.data["can_send"] is False
        assert "can't send" in result.message


class TestEmailPlaceholders:

    @pytest.mark.parametrize("factory", [gmail_email, outlook_email, smtp_email])
    def test_unconnected_accounts_refuse_everything(self, factory) -> None:
        provider = factory()
        assert isinstance(provider, UnavailableProvider)
        assert provider.is_available() is False
        # Any call at all answers the same way rather than raising.
        result = provider.send("anything")
        assert result.ok is False


# -------------------------------------------------------- provider registry

class TestProviderAvailability:

    def test_the_report_is_built_from_live_providers(self, registry,
                                                     action_context) -> None:
        result = _run(registry, action_context, "list_integrations")
        assert result.ok
        by_name = {row["capability"]: row for row in result.data["providers"]}
        assert by_name["Calendar (local)"]["available"] is True
        assert by_name["Calendar (Google)"]["available"] is False
        assert by_name["Email (Gmail)"]["available"] is False

    def test_unconnected_integrations_are_named_with_a_reason(
            self, registry, action_context) -> None:
        result = _run(registry, action_context, "list_integrations")
        for row in result.data["providers"]:
            if not row["available"]:
                assert row["detail"], f"{row['capability']} gave no reason"

    def test_capability_listing_comes_from_the_registry(self, registry,
                                                        action_context) -> None:
        ctx = _context(action_context, registry=registry)
        result = _run(registry, ctx, "list_capabilities")
        assert result.data["total"] == len(registry)

    def test_a_capability_that_does_not_exist_is_not_claimed(self, registry,
                                                             action_context) -> None:
        ctx = _context(action_context, registry=registry)
        for absent in ("send a fax", "order a taxi", "print a document"):
            result = _run(registry, ctx, "list_capabilities", topic=absent)
            assert "don't have anything" in result.message, absent
