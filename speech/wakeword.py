"""Wake-word detection interface.

Specification section 5 requires the wake-word implementation to be
replaceable, so the application depends only on this abstraction and never on a
particular detection library.  Different engines have very different
constraints -- some need a trained model file per phrase, some run offline,
some need a licence key -- and :meth:`WakeWordDetector.supports_phrase` exists
so the settings UI can tell the user *before* they type a phrase whether the
active engine can actually detect it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Tuple

from config.settings import normalise_wake_phrase
from speech.wakeword_store import STORE as CUSTOM_STORE
from utils.logger import get_logger

logger = get_logger("speech.wakeword")

#: Called with the detection event when the wake phrase is heard.
WakeWordCallback = Callable[["WakeWordEvent"], None]


@dataclass(frozen=True)
class WakeWordEvent:
    """A single detection."""

    phrase: str
    confidence: float = 0.0
    detected_at: datetime = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.detected_at is None:
            object.__setattr__(self, "detected_at", datetime.now(timezone.utc))


class WakeWordDetector(ABC):
    """Listens continuously for the configured wake phrase."""

    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool:
        """True when this engine can run on the current machine."""

    @abstractmethod
    def start(self, phrase: str, callback: WakeWordCallback) -> bool:
        """Begin listening.  Returns False if the engine could not start."""

    @abstractmethod
    def stop(self) -> None:
        """Stop listening and release the microphone."""

    def supports_phrase(self, phrase: str) -> bool:
        """Whether this engine can detect ``phrase`` without extra training."""
        return False

    def unavailable_reason(self) -> str:
        return "Wake-word detection is not implemented in this build."


class NullWakeWordDetector(WakeWordDetector):
    """Stand-in used while Phase 1 is text-only.

    It never invokes the callback, so nothing downstream can mistake it for a
    working detector.
    """

    name = "none"

    def is_available(self) -> bool:
        return False

    def start(self, phrase: str, callback: WakeWordCallback) -> bool:
        logger.debug("Wake-word detection requested but no engine is installed")
        return False

    def stop(self) -> None:
        return None


class MissingWakeWordDetector(WakeWordDetector):
    """Placeholder for an identity that has no validated model installed.

    Selecting "Asma" when only the "Hey Aariz" model is present must not
    silently arm the Aariz detector -- that would wake the assistant at another
    person's name, which is worse than not waking at all.  So the registry
    hands back one of these instead: it reports itself unavailable, explains
    which model is missing, and never invokes the callback.
    """

    name = "missing"

    def __init__(self, phrase: str, *, display_name: str = "") -> None:
        self._phrase = normalise_wake_phrase(phrase)
        self._display = display_name or _name_from_phrase(self._phrase)

    def is_available(self) -> bool:
        return False

    def start(self, phrase: str, callback: WakeWordCallback) -> bool:
        logger.info("Wake-word model not installed", extra={
            "context": {"phrase": self._phrase}})
        return False

    def stop(self) -> None:
        return None

    def supports_phrase(self, phrase: str) -> bool:
        return False

    def unavailable_reason(self) -> str:
        if not self._phrase:
            return "No wake phrase is configured."
        if self._display:
            return (f"Wake word model for {self._display} is not installed yet. "
                    f"No detector claims \"{self._phrase}\".")
        return f"No wake-word model is installed for \"{self._phrase}\"."


def _name_from_phrase(phrase: str) -> str:
    """"hey nova" -> "Nova".  Best-effort, for user-facing messages only."""
    words = normalise_wake_phrase(phrase).split()
    if len(words) >= 2 and words[0] == "hey":
        words = words[1:]
    return " ".join(w.capitalize() for w in words)


#: Builds a detector for one phrase.  Extra keyword arguments from the caller
#: (a microphone index, the "is the assistant speaking" guard) are forwarded.
DetectorFactory = Callable[..., WakeWordDetector]


@dataclass(frozen=True)
class WakeWordProvider:
    """One installed wake-word implementation and the phrase it detects.

    A provider is a *promise about one phrase*.  Wake-word models are trained
    per phrase, so "which detector do I use" is answered by the configured
    phrase and nothing else.
    """

    phrase: str
    factory: DetectorFactory
    label: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "phrase", normalise_wake_phrase(self.phrase))


class WakeWordRegistry:
    """Maps configured wake phrases to the detectors that can hear them.

    This is the seam the specification asked for.  Adding a validated "Hey
    Nova" model later is a :meth:`register` call in this module -- no change to
    the engine, the settings layer or the UI, none of which name a detector
    class.

    Two sources feed it.  Registered providers are the models that ship with
    the application.  A :class:`~speech.wakeword_store.CustomWakeWordStore` --
    the models the user enrolled -- is consulted when no shipped provider
    claims the phrase, and is read on every lookup rather than cached, so a
    detector enrolled a moment ago is found without anyone remembering to
    refresh anything.

    Shipped providers win a collision deliberately: the validated model is the
    one whose behaviour is known.
    """

    def __init__(self, custom_store=None) -> None:
        self._providers: Dict[str, WakeWordProvider] = {}
        self._custom_store = custom_store

    def register(self, provider: WakeWordProvider) -> None:
        if not provider.phrase:
            raise ValueError("a wake-word provider needs a phrase")
        self._providers[provider.phrase] = provider

    def phrases(self) -> Tuple[str, ...]:
        """Every phrase that can be detected right now, shipped or enrolled."""
        return tuple(sorted(set(self._providers) | set(self._custom_phrases())))

    def provider_for(self, phrase: str) -> Optional[WakeWordProvider]:
        normalised = normalise_wake_phrase(phrase)
        provider = self._providers.get(normalised)
        if provider is not None:
            return provider
        return self._custom_provider(normalised)

    def is_builtin(self, phrase: str) -> bool:
        """Whether a shipped model claims this phrase.

        The distinction matters to the settings screen: a shipped phrase is not
        something the user may re-enrol over, because the validated model is the
        one whose behaviour is known.
        """
        return normalise_wake_phrase(phrase) in self._providers

    def custom_model(self, phrase: str):
        """The enrolled model for ``phrase``, or ``None``.

        Exposed so the settings screen can show when a detector was enrolled
        and how it scored, without reaching into the store itself.
        """
        if self._custom_store is None:
            return None
        try:
            model = self._custom_store.get(phrase)
        except Exception:  # noqa: BLE001 - a damaged store is "no model"
            return None
        return model if (model is not None and model.is_intact()) else None

    def _custom_phrases(self) -> Tuple[str, ...]:
        if self._custom_store is None:
            return ()
        try:
            return tuple(m.phrase for m in self._custom_store.list())
        except Exception:  # noqa: BLE001
            logger.warning("Enrolled wake words could not be listed")
            return ()

    def _custom_provider(self, normalised: str) -> Optional[WakeWordProvider]:
        model = self.custom_model(normalised)
        if model is None:
            return None

        def factory(_model=model, **kwargs) -> WakeWordDetector:
            # Its own reference, its own weights and its own calibrated
            # threshold -- nothing is inherited from the shipped model but the
            # code that runs them.
            from speech.wakeword_cascade import CascadeWakeWordDetector

            return CascadeWakeWordDetector(
                phrase=_model.phrase,
                model_path=_model.verifier_path,
                reference_path=_model.reference_path,
                stage1_threshold=_model.stage1_threshold,
                stage2_threshold=_model.stage2_threshold,
                **kwargs)

        return WakeWordProvider(
            phrase=model.phrase, factory=factory,
            label=model.display_name or _name_from_phrase(model.phrase),
            description=(f"enrolled on {model.created_at or 'an unknown date'} "
                         f"from {model.enrollment_count} recordings "
                         f"(version {model.model_version})"))

    def create(self, phrase: str, *, display_name: str = "", **kwargs) -> WakeWordDetector:
        """A detector for ``phrase``.  Never returns ``None`` and never raises.

        An unknown phrase, or a provider whose construction fails, yields a
        :class:`MissingWakeWordDetector`: the caller gets a uniform object that
        reports why it cannot run, rather than an exception during start-up or
        a silent substitution.
        """
        provider = self.provider_for(phrase)
        if provider is None:
            return MissingWakeWordDetector(phrase, display_name=display_name)
        try:
            return provider.factory(**kwargs)
        except Exception as exc:  # noqa: BLE001 - a broken provider is unavailability
            logger.error("Wake-word provider failed to construct", extra={
                "context": {"phrase": provider.phrase, "error": str(exc)}})
            return MissingWakeWordDetector(phrase, display_name=display_name)


def _build_cascade(**kwargs) -> WakeWordDetector:
    # Imported here, not at module scope: the cascade imports this module for
    # the interface, and it pulls in torch, which a text-only session should
    # never load.
    from speech.wakeword_cascade import CascadeWakeWordDetector

    return CascadeWakeWordDetector(**kwargs)


#: The installed providers: one shipped model, plus whatever the user has
#: enrolled.
REGISTRY = WakeWordRegistry(custom_store=CUSTOM_STORE)
REGISTRY.register(WakeWordProvider(
    phrase="hey aariz",
    factory=_build_cascade,
    label="Aariz",
    description=("EfficientWord-Net stage 1 plus the Phase-15 learned far-field "
                 "verifier; the only phrase with a validated model in this build."),
))
