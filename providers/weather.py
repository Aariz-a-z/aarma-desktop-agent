"""Weather, from a real forecast service or not at all.

There is no offline weather.  A machine cannot know the temperature outside
without asking something that does, so the only two honest outcomes here are a
reading that came from an API and an explicit "I can't reach weather data right
now".  A plausible-sounding number in the third case would be undetectable --
the user would dress for it.

The implementation talks to **Open-Meteo**, which needs no account and no API
key, so nothing about the user is registered anywhere to make it work.  Two
privacy decisions follow from that and are deliberate:

* **No IP geolocation.**  Looking the user's position up from their address
  would send their network identity to a third party to answer a question they
  did not ask that way.  A location has to be named in the request or set in
  settings, and when neither exists the provider says so.
* **Nothing but a place name leaves the machine.**  Not the utterance, not the
  assistant's name, not the user's account.

``fetch`` is injected, so the test-suite exercises every path -- success,
unknown place, transport failure, malformed payload -- without a network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode

from providers import Availability, Provider, ProviderResult
from utils.logger import get_logger

logger = get_logger("providers.weather")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

#: Seconds before a weather call is abandoned.  Short, because this runs on a
#: worker thread that a spoken answer is waiting on: a forecast that arrives
#: after the user has walked away is worse than "I couldn't reach it".
REQUEST_TIMEOUT = 8.0

#: WMO present-weather code -> how a person would say it.  Codes with no entry
#: are reported as an unknown condition rather than rounded to the nearest
#: familiar word.
WMO_CONDITIONS: Dict[int, str] = {
    0: "clear", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "light snow showers", 86: "snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
    99: "thunderstorms with heavy hail",
}


def condition_text(code: Optional[int]) -> Optional[str]:
    """Plain words for a WMO code, or ``None`` when the code is unknown."""
    if code is None:
        return None
    return WMO_CONDITIONS.get(int(code))


@dataclass(frozen=True)
class Place:
    """A resolved location.  ``label`` is what gets said back to the user."""

    label: str
    latitude: float
    longitude: float
    country: str = ""
    timezone: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "latitude": self.latitude,
                "longitude": self.longitude, "country": self.country,
                "timezone": self.timezone}


@dataclass(frozen=True)
class WeatherReading:
    """Current conditions.  Every field is optional, and ``None`` means
    the service did not supply it -- never a stand-in value."""

    place: str
    temperature_c: Optional[float] = None
    feels_like_c: Optional[float] = None
    condition: Optional[str] = None
    humidity_percent: Optional[float] = None
    wind_kph: Optional[float] = None
    observed_at: Optional[str] = None
    provider: str = "open-meteo"

    def summary(self) -> str:
        """A sentence built only from what was actually measured."""
        bits: List[str] = []
        if self.temperature_c is not None:
            bits.append(f"{self.temperature_c:.0f}°C")
        if self.condition:
            bits.append(f"and {self.condition}")
        head = f"It's {' '.join(bits)} in {self.place}" if bits else \
            f"I got a reading for {self.place}, but it had no temperature or conditions in it"
        tail: List[str] = []
        if self.feels_like_c is not None and self.temperature_c is not None \
                and abs(self.feels_like_c - self.temperature_c) >= 2:
            tail.append(f"feels like {self.feels_like_c:.0f}°C")
        if self.humidity_percent is not None:
            tail.append(f"{self.humidity_percent:.0f}% humidity")
        if self.wind_kph is not None:
            tail.append(f"wind {self.wind_kph:.0f} km/h")
        if tail:
            return f"{head} — {', '.join(tail)}."
        return f"{head}."

    def as_dict(self) -> Dict[str, Any]:
        return {"place": self.place, "temperature_c": self.temperature_c,
                "feels_like_c": self.feels_like_c, "condition": self.condition,
                "humidity_percent": self.humidity_percent,
                "wind_kph": self.wind_kph, "observed_at": self.observed_at,
                "provider": self.provider}


@dataclass(frozen=True)
class ForecastDay:
    """One day ahead."""

    date: str
    high_c: Optional[float] = None
    low_c: Optional[float] = None
    condition: Optional[str] = None
    precipitation_chance: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"date": self.date, "high_c": self.high_c, "low_c": self.low_c,
                "condition": self.condition,
                "precipitation_chance": self.precipitation_chance}

    def summary(self) -> str:
        bits: List[str] = []
        if self.condition:
            bits.append(self.condition)
        if self.high_c is not None and self.low_c is not None:
            bits.append(f"{self.low_c:.0f}–{self.high_c:.0f}°C")
        if self.precipitation_chance is not None:
            bits.append(f"{self.precipitation_chance:.0f}% chance of rain")
        return ", ".join(bits) if bits else "no details supplied"


class WeatherProvider(Provider):
    """What any weather source must offer."""

    name = "weather"
    description = "current conditions and a short forecast"

    def current(self, location: str = "") -> ProviderResult:
        raise NotImplementedError

    def forecast(self, location: str = "", *, days: int = 3) -> ProviderResult:
        raise NotImplementedError


Fetcher = Callable[[str], Any]


def _requests_fetch(url: str) -> Any:
    """Default transport.  Kept tiny so the injected test double matches it."""
    import requests  # noqa: PLC0415

    response = requests.get(url, timeout=REQUEST_TIMEOUT,
                            headers={"Accept": "application/json"})
    response.raise_for_status()
    return response.json()


class OpenMeteoProvider(WeatherProvider):
    """Open-Meteo: free, key-less, and therefore account-less."""

    name = "open-meteo"
    description = "weather from the Open-Meteo public API"

    def __init__(self, *, fetch: Optional[Fetcher] = None,
                 default_location: str = "", online: bool = True) -> None:
        self._fetch = fetch
        self._default_location = (default_location or "").strip()
        self._online = online

    # -- availability ------------------------------------------------------

    def _transport(self) -> Optional[Fetcher]:
        if self._fetch is not None:
            return self._fetch
        try:
            import requests  # noqa: F401,PLC0415
        except ImportError:
            return None
        return _requests_fetch

    def availability(self) -> Availability:
        if self._transport() is None:
            return Availability.missing(
                self.name, "Weather needs the requests component, which isn't installed.",
                "Run: pip install requests")
        if not self._online:
            return Availability.missing(
                self.name, "I'm offline, so I can't fetch the weather.",
                "Reconnect and ask again.")
        return Availability.ready(self.name, "weather from Open-Meteo")

    # -- lookup ------------------------------------------------------------

    def _call(self, url: str, params: Dict[str, Any]) -> Any:
        transport = self._transport()
        if transport is None:
            raise RuntimeError("no transport")
        return transport(f"{url}?{urlencode(params)}")

    def resolve_place(self, location: str) -> Optional[Place]:
        """Turn a place name into coordinates, or ``None`` if it is unknown."""
        wanted = " ".join(str(location or "").split())
        if not wanted:
            return None
        try:
            payload = self._call(GEOCODE_URL, {
                "name": wanted, "count": 1, "language": "en", "format": "json"})
        except Exception as exc:  # noqa: BLE001 - network and parsing alike
            logger.warning("Geocoding failed", extra={"context": {"error": str(exc)}})
            return None
        results = (payload or {}).get("results") or []
        if not results:
            return None
        first = results[0]
        try:
            latitude = float(first["latitude"])
            longitude = float(first["longitude"])
        except (KeyError, TypeError, ValueError):
            return None
        label = str(first.get("name") or wanted)
        admin = str(first.get("admin1") or "")
        country = str(first.get("country") or "")
        if admin and admin.lower() != label.lower():
            label = f"{label}, {admin}"
        return Place(label=label, latitude=latitude, longitude=longitude,
                     country=country, timezone=str(first.get("timezone") or ""))

    def _place_for(self, location: str) -> Tuple[Optional[Place], Optional[str]]:
        wanted = (location or "").strip() or self._default_location
        if not wanted:
            return None, ("I don't know where you are. Tell me a place — "
                          "\"what's the weather in Bengaluru?\" — or set your "
                          "location in Settings and I'll use it by default.")
        place = self.resolve_place(wanted)
        if place is None:
            return None, f"I couldn't find a place called \"{wanted}\"."
        return place, None

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        """A float, or ``None``.  Anything unparseable is unknown, not zero."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _code(value: Any) -> Optional[int]:
        """A WMO code as an int, or ``None``.  Code 0 is "clear", not missing."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # -- capabilities ------------------------------------------------------

    def current(self, location: str = "") -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        place, problem = self._place_for(location)
        if place is None:
            return ProviderResult.failure(problem or "I don't know where to look.",
                                          error_code="no_location")
        try:
            payload = self._call(FORECAST_URL, {
                "latitude": place.latitude, "longitude": place.longitude,
                "current": ("temperature_2m,relative_humidity_2m,"
                            "apparent_temperature,weather_code,wind_speed_10m"),
                "timezone": "auto"})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Weather fetch failed",
                           extra={"context": {"error": str(exc)}})
            return ProviderResult.failure(
                "I couldn't reach the weather service just now.",
                error_code="fetch_failed")
        current = (payload or {}).get("current")
        if not isinstance(current, dict):
            # A response we cannot read is a failure, not an empty forecast.
            return ProviderResult.failure(
                "The weather service sent something I couldn't read.",
                error_code="bad_payload")
        reading = WeatherReading(
            place=place.label,
            temperature_c=self._number(current.get("temperature_2m")),
            feels_like_c=self._number(current.get("apparent_temperature")),
            condition=condition_text(self._code(current.get("weather_code"))),
            humidity_percent=self._number(current.get("relative_humidity_2m")),
            wind_kph=self._number(current.get("wind_speed_10m")),
            observed_at=str(current.get("time") or "") or None,
            provider=self.name)
        logger.info("Weather read", extra={"context": {"place": place.label}})
        return ProviderResult.success(reading.summary(), reading=reading.as_dict(),
                                      place=place.as_dict())

    def forecast(self, location: str = "", *, days: int = 3) -> ProviderResult:
        if not self.is_available():
            return ProviderResult.unavailable(self.availability())
        place, problem = self._place_for(location)
        if place is None:
            return ProviderResult.failure(problem or "I don't know where to look.",
                                          error_code="no_location")
        span = max(1, min(7, int(days)))
        try:
            payload = self._call(FORECAST_URL, {
                "latitude": place.latitude, "longitude": place.longitude,
                "daily": ("weather_code,temperature_2m_max,temperature_2m_min,"
                          "precipitation_probability_max"),
                "forecast_days": span, "timezone": "auto"})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Forecast fetch failed",
                           extra={"context": {"error": str(exc)}})
            return ProviderResult.failure(
                "I couldn't reach the weather service just now.",
                error_code="fetch_failed")
        daily = (payload or {}).get("daily")
        if not isinstance(daily, dict) or not daily.get("time"):
            return ProviderResult.failure(
                "The weather service sent something I couldn't read.",
                error_code="bad_payload")

        def _at(key: str, index: int) -> Any:
            values = daily.get(key)
            if isinstance(values, list) and index < len(values):
                return values[index]
            return None

        entries: List[ForecastDay] = []
        for index, date in enumerate(daily["time"]):
            entries.append(ForecastDay(
                date=str(date),
                high_c=self._number(_at("temperature_2m_max", index)),
                low_c=self._number(_at("temperature_2m_min", index)),
                condition=condition_text(self._code(_at("weather_code", index))),
                precipitation_chance=self._number(
                    _at("precipitation_probability_max", index))))
        return ProviderResult.success(
            f"Forecast for {place.label}.",
            days=[day.as_dict() for day in entries],
            place=place.as_dict())


def build_weather_provider(config: Any = None, *, online: bool = True,
                           fetch: Optional[Fetcher] = None) -> WeatherProvider:
    """The weather source for this build, configured from settings."""
    location = ""
    try:
        location = str(getattr(config.config.runtime, "weather_location", "") or "")
    except AttributeError:
        location = ""
    return OpenMeteoProvider(fetch=fetch, default_location=location, online=online)


__all__ = ["ForecastDay", "OpenMeteoProvider", "Place", "WeatherProvider",
           "WeatherReading", "WMO_CONDITIONS", "build_weather_provider",
           "condition_text", "GEOCODE_URL", "FORECAST_URL"]
