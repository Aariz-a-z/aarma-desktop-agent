"""The text-to-speech provider interface.

Everything above this line -- the engine, the GUI, the assistant -- depends only
on :class:`SpeechSpeaker`. Nothing knows which provider is speaking, which is
what allows an offline Windows voice and an online neural voice to be swapped
per request, and a better provider to be added later without touching the
application.

The interface is language-aware and voice-aware from the outset rather than
assuming English (specification section 7), and every provider must be able to
answer :meth:`SpeechSpeaker.describe_voice` *without* speaking -- the Settings
screen has to show whether a combination will work before the user commits to
it, and the engine needs it to choose a fallback.
"""

from __future__ import annotations

import html
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from speech.voices import (
    VOICE_PROFILES,
    ResolvedVoice,
    VoiceAvailability,
    VoiceProfile,
    VoiceQuality,
    VoiceStyle,
    get_profile,
)

#: Bounds on the user-facing delivery controls. Speech below or above these
#: stops being comfortable to listen to, which is the point of having a voice.
MIN_RATE, MAX_RATE = 0.5, 1.6
MIN_VOLUME, MAX_VOLUME = 0.0, 1.0
MIN_PITCH, MAX_PITCH = -10.0, 10.0


def clamp(value: float, lowest: float, highest: float) -> float:
    return max(lowest, min(highest, value))


@dataclass(frozen=True)
class VoiceDescriptor:
    """A concrete voice a provider can offer, for display in Settings."""

    voice_id: str
    display_name: str
    gender: str
    language: str = "en"
    quality: VoiceQuality = VoiceQuality.STANDARD
    provider: str = ""


@dataclass(frozen=True)
class SpeechRequest:
    """One thing to say, and how to say it."""

    text: str
    language: str = "en"
    voice_id: str = "female_warm"
    style: VoiceStyle = VoiceStyle.WARM
    rate: float = 1.0        # Multiplier: 1.0 is the voice's natural pace.
    volume: float = 0.9      # 0.0 - 1.0
    pitch: float = 0.0       # Semitone-ish offset; 0.0 is natural.

    @property
    def profile(self) -> VoiceProfile:
        return get_profile(self.voice_id)

    def normalised(self) -> "SpeechRequest":
        """A copy with every delivery control inside its supported range."""
        from dataclasses import replace

        return replace(
            self,
            rate=clamp(self.rate, MIN_RATE, MAX_RATE),
            volume=clamp(self.volume, MIN_VOLUME, MAX_VOLUME),
            pitch=clamp(self.pitch, MIN_PITCH, MAX_PITCH),
        )

    def escaped_text(self) -> str:
        """XML-safe text, for providers whose markup carries prosody."""
        return html.escape(self.text, quote=False)


@dataclass(frozen=True)
class SpeechResult:
    """The outcome of one speech attempt."""

    ok: bool
    voice: Optional[ResolvedVoice] = None
    provider: str = ""
    error: str = ""
    error_code: str = ""
    interrupted: bool = False

    @classmethod
    def success(cls, provider: str, voice: Optional[ResolvedVoice] = None) -> "SpeechResult":
        return cls(ok=True, provider=provider, voice=voice)

    @classmethod
    def failure(cls, provider: str, error: str, *,
                error_code: str = "speech_failed") -> "SpeechResult":
        return cls(ok=False, provider=provider, error=error, error_code=error_code)

    @classmethod
    def cancelled(cls, provider: str) -> "SpeechResult":
        return cls(ok=False, provider=provider, interrupted=True,
                   error_code="interrupted", error="Speech was stopped.")


class SpeechSpeaker(ABC):
    """A text-to-speech backend."""

    #: Stable identifier stored in configuration.
    name: str = "abstract"
    #: Shown in Settings. Never provider jargon (section 4).
    display_name: str = "Abstract"
    #: Whether this provider needs a network connection.
    requires_internet: bool = False
    #: The quality tier this provider delivers.
    quality: VoiceQuality = VoiceQuality.UNAVAILABLE

    @abstractmethod
    def is_available(self) -> bool:
        """True when this provider can synthesise right now on this machine."""

    @abstractmethod
    def describe_voice(self, language: str, voice_id: str,
                       style: Optional[VoiceStyle] = None) -> VoiceAvailability:
        """Report what this provider would use, without speaking."""

    @abstractmethod
    def speak(self, request: SpeechRequest) -> SpeechResult:
        """Synthesise and play ``request``. Blocks until finished or stopped.

        Called on the engine's worker thread, never the GUI thread. Must never
        raise: failures are reported as :class:`SpeechResult`.
        """

    @abstractmethod
    def stop(self) -> None:
        """Interrupt current speech. Safe from any thread, and when idle."""

    def voices(self, language: str = "en") -> List[VoiceDescriptor]:
        """Voices this provider offers for ``language``."""
        return []

    def unavailable_reason(self) -> str:
        return f"{self.display_name} is not available."

    def describe(self) -> str:
        state = "available" if self.is_available() else "unavailable"
        return f"{self.display_name} ({state})"


class NullSpeechSpeaker(SpeechSpeaker):
    """Used when no provider can run. Never fabricates success."""

    name = "none"
    display_name = "No voice"
    quality = VoiceQuality.UNAVAILABLE

    def __init__(self, reason: str = "No speech provider is available.") -> None:
        self._reason = reason

    def is_available(self) -> bool:
        return False

    def describe_voice(self, language: str, voice_id: str,
                       style: Optional[VoiceStyle] = None) -> VoiceAvailability:
        return VoiceAvailability(
            available=False, provider=self.name, quality=VoiceQuality.UNAVAILABLE,
            language=language, reason=self._reason,
            suggestion="Install a Windows voice, or connect to the internet for a natural voice.",
        )

    def speak(self, request: SpeechRequest) -> SpeechResult:
        return SpeechResult.failure(self.name, self._reason,
                                    error_code="no_provider")

    def stop(self) -> None:
        return None

    def unavailable_reason(self) -> str:
        return self._reason


#: Backwards-compatible view of the shipped profiles as descriptors, so callers
#: that only need a display list do not have to know about providers.
DECLARED_VOICES: tuple[VoiceDescriptor, ...] = tuple(
    VoiceDescriptor(
        voice_id=profile.voice_id,
        display_name=profile.display_name,
        gender=profile.gender,
    )
    for profile in VOICE_PROFILES.values()
)
