"""Interfaces between AARMA and the things it cannot do by itself.

The assistant talks to a *provider* whenever a capability depends on something
outside its own process: the machine's sensors, a calendar, a mail account, the
weather. Each one is an interface first and an implementation second, so that a
local stand-in can ship today and a real integration can replace it later
without the actions that use it changing at all.

The rule every provider follows, and the reason this package exists:

    **A provider that cannot do something says so. It never pretends.**

That is not a style preference. "Your meeting has been added to Google
Calendar" when Google Calendar was never connected is worse than an error,
because the user stops checking. So availability is part of the interface --
:meth:`Provider.availability` -- and every result carries whether it actually
happened rather than whether it was attempted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Availability:
    """Whether a capability can be used right now, and why not if it cannot.

    ``detail`` is written for the user, not the log: it is the sentence the
    assistant says when someone asks for something that is not connected.
    """

    available: bool
    provider: str
    detail: str = ""
    #: Something the user could do about it, when there is something.
    remedy: str = ""

    @classmethod
    def ready(cls, provider: str, detail: str = "") -> "Availability":
        return cls(available=True, provider=provider, detail=detail)

    @classmethod
    def missing(cls, provider: str, detail: str, remedy: str = "") -> "Availability":
        return cls(available=False, provider=provider, detail=detail, remedy=remedy)

    def as_dict(self) -> Dict[str, Any]:
        return {"available": self.available, "provider": self.provider,
                "detail": self.detail, "remedy": self.remedy}


@dataclass
class ProviderResult:
    """What a provider did, and whether it really did it.

    ``ok`` means the underlying system confirmed the operation. A provider that
    queued, drafted or simulated something sets ``ok`` and says so in
    ``message``; it never reports a draft as a send.
    """

    ok: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error_code: str = ""

    @classmethod
    def success(cls, message: str = "", **data: Any) -> "ProviderResult":
        return cls(ok=True, message=message, data=data)

    @classmethod
    def failure(cls, message: str, *, error_code: str = "provider_failed",
                **data: Any) -> "ProviderResult":
        return cls(ok=False, message=message, error_code=error_code, data=data)

    @classmethod
    def unavailable(cls, availability: Availability) -> "ProviderResult":
        text = availability.detail
        if availability.remedy:
            text = f"{text} {availability.remedy}"
        return cls(ok=False, message=text, error_code="unavailable")


class Provider:
    """Base class: a name, and an honest answer about whether it works."""

    #: Short identifier used in logs and in the capability report.
    name: str = "provider"

    #: What the provider does, for the capability listing.
    description: str = ""

    def availability(self) -> Availability:
        raise NotImplementedError

    def is_available(self) -> bool:
        try:
            return self.availability().available
        except Exception:  # noqa: BLE001 - a broken provider is an unavailable one
            return False


class UnavailableProvider(Provider):
    """Stands in for an integration that is not connected.

    Returned instead of ``None`` so callers never branch on nullness; asking it
    to do anything yields a :class:`ProviderResult` explaining what is missing.
    """

    def __init__(self, name: str, detail: str, remedy: str = "") -> None:
        self.name = name
        self._detail = detail
        self._remedy = remedy

    def availability(self) -> Availability:
        return Availability.missing(self.name, self._detail, self._remedy)

    def __getattr__(self, item: str):
        # Any capability call on an unconnected provider answers the same way.
        def _unavailable(*_args: Any, **_kwargs: Any) -> ProviderResult:
            return ProviderResult.unavailable(self.availability())

        return _unavailable


#: Sentinel for a measurement the hardware or API genuinely cannot supply.
#: Distinct from zero, and from a missing key: "I don't know" is information,
#: and inventing a plausible number in its place is the failure this guards.
UNKNOWN: Optional[float] = None


__all__ = ["Availability", "Provider", "ProviderResult", "UnavailableProvider",
           "UNKNOWN"]
