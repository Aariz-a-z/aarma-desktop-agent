"""Voice profiles, quality tiers, and the mapping onto concrete provider voices.

AARMA stores an **abstract** voice identifier (``female_warm``, ``male_calm``)
rather than a provider's voice name. That indirection is what lets the same
saved configuration keep working when the provider changes, when the user
switches language, or when a better voice becomes available -- the identifier
means "the warm female voice", and each provider answers that question in its
own vocabulary.

Two honesty rules are encoded here:

* :class:`VoiceQuality` distinguishes a **neural** voice from a **standard**
  one. The fallback is not presented as equivalent to the preferred voice; the
  UI shows which tier is actually in use.
* Where a language has only one voice per gender -- which is the case for most
  languages below -- the *style* affects delivery (rate and pitch) rather than
  selecting a different voice. :func:`resolve_online_voice` reports what it
  actually chose so nothing has to be guessed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple


class VoiceQuality(str, Enum):
    """How good the voice actually is, in the user's terms."""

    NEURAL = "neural"        # Modern synthesis. Natural prosody.
    STANDARD = "standard"    # Classic concatenative synthesis. Clearly synthetic.
    UNAVAILABLE = "unavailable"

    @property
    def label(self) -> str:
        return {
            VoiceQuality.NEURAL: "Natural",
            VoiceQuality.STANDARD: "Standard",
            VoiceQuality.UNAVAILABLE: "Unavailable",
        }[self]

    @property
    def description(self) -> str:
        return {
            VoiceQuality.NEURAL: "Natural-sounding neural voice.",
            VoiceQuality.STANDARD: (
                "Built-in Windows voice. Clear, but noticeably synthetic "
                "compared with a natural voice."
            ),
            VoiceQuality.UNAVAILABLE: "No voice available for this combination.",
        }[self]


class VoiceStyle(str, Enum):
    """User-facing delivery styles.

    Deliberately free of provider terminology (section 4) and free of persona
    logic (section 14) -- a style is a delivery preference, nothing more.
    """

    WARM = "warm"
    CLEAR = "clear"
    CALM = "calm"
    PROFESSIONAL = "professional"
    ENERGETIC = "energetic"

    @property
    def label(self) -> str:
        return self.value.capitalize()


@dataclass(frozen=True)
class StyleProsody:
    """How a style bends rate and pitch, as multipliers/offsets."""

    rate_scale: float = 1.0
    pitch_offset: float = 0.0
    description: str = ""


#: Modest adjustments on purpose. Large prosody shifts are what make synthetic
#: speech tiring to listen to, which is the opposite of the goal.
STYLE_PROSODY: Dict[VoiceStyle, StyleProsody] = {
    VoiceStyle.WARM: StyleProsody(0.96, -1.0, "Gentle and unhurried."),
    VoiceStyle.CLEAR: StyleProsody(1.00, 0.0, "Neutral and easy to follow."),
    VoiceStyle.CALM: StyleProsody(0.92, -2.0, "Slower, lower and steady."),
    VoiceStyle.PROFESSIONAL: StyleProsody(1.02, 0.0, "Crisp and businesslike."),
    VoiceStyle.ENERGETIC: StyleProsody(1.10, 2.0, "Brighter and quicker."),
}


@dataclass(frozen=True)
class VoiceProfile:
    """A selectable voice identity, independent of any provider."""

    voice_id: str
    display_name: str
    gender: str
    default_style: VoiceStyle
    description: str

    @property
    def is_female(self) -> bool:
        return self.gender == "female"


#: The voices offered in Settings and in first-run setup. ``voice_id`` values
#: are stable and are what appear in ``config.json``; the Phase 1 defaults
#: (``female_warm`` for Asma, ``male_calm`` for Aariz) are preserved.
VOICE_PROFILES: Dict[str, VoiceProfile] = {
    profile.voice_id: profile
    for profile in (
        VoiceProfile("female_warm", "Warm", "female", VoiceStyle.WARM,
                     "Gentle, friendly and unhurried. The default for Asma."),
        VoiceProfile("female_clear", "Clear", "female", VoiceStyle.CLEAR,
                     "Brighter and more precise. Good for noisy rooms."),
        VoiceProfile("male_calm", "Calm", "male", VoiceStyle.CALM,
                     "Steady, low and measured. The default for Aariz."),
        VoiceProfile("male_bright", "Bright", "male", VoiceStyle.PROFESSIONAL,
                     "Crisper and more energetic."),
    )
}

DEFAULT_FEMALE_VOICE = "female_warm"
DEFAULT_MALE_VOICE = "male_calm"


def get_profile(voice_id: str) -> VoiceProfile:
    """Look up a profile, falling back to the warm female voice."""
    return VOICE_PROFILES.get(voice_id, VOICE_PROFILES[DEFAULT_FEMALE_VOICE])


def profiles_for_gender(gender: str) -> Tuple[VoiceProfile, ...]:
    return tuple(p for p in VOICE_PROFILES.values() if p.gender == gender)


# ------------------------------------------------------- online voice mapping

#: Neural voices per language and gender, in preference order. The first entry
#: is the everyday choice; later entries are used by the brighter styles where a
#: language actually offers an alternative.
#:
#: English defaults to the Indian English voices: AARMA's assistants are named
#: Asma and Aariz and its Phase 2 language roadmap is Indian languages, so an
#: Indian English voice is the better default for its users than a US one.
ONLINE_VOICES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "en": {
        "female": ("en-IN-NeerjaNeural", "en-US-AvaNeural", "en-US-EmmaNeural"),
        "male": ("en-IN-PrabhatNeural", "en-US-AndrewNeural", "en-GB-RyanNeural"),
    },
    "hi": {"female": ("hi-IN-SwaraNeural",), "male": ("hi-IN-MadhurNeural",)},
    "kn": {"female": ("kn-IN-SapnaNeural",), "male": ("kn-IN-GaganNeural",)},
    "te": {"female": ("te-IN-ShrutiNeural",), "male": ("te-IN-MohanNeural",)},
    "ta": {"female": ("ta-IN-PallaviNeural",), "male": ("ta-IN-ValluvarNeural",)},
    "ml": {"female": ("ml-IN-SobhanaNeural",), "male": ("ml-IN-MidhunNeural",)},
    "ur": {"female": ("ur-IN-GulNeural",), "male": ("ur-IN-SalmanNeural",)},
    "ar": {"female": ("ar-EG-SalmaNeural",), "male": ("ar-EG-ShakirNeural",)},
    "es": {"female": ("es-ES-ElviraNeural",), "male": ("es-ES-AlvaroNeural",)},
    "fr": {"female": ("fr-FR-DeniseNeural",), "male": ("fr-FR-HenriNeural",)},
}

#: Which entry in the preference list a style reaches for. Clamped to whatever
#: the language actually offers, so a single-voice language simply uses its one
#: voice and expresses the style through prosody instead.
_STYLE_VOICE_INDEX: Dict[VoiceStyle, int] = {
    VoiceStyle.WARM: 0,
    VoiceStyle.CALM: 0,
    VoiceStyle.CLEAR: 2,
    VoiceStyle.PROFESSIONAL: 1,
    VoiceStyle.ENERGETIC: 1,
}


@dataclass(frozen=True)
class ResolvedVoice:
    """A concrete provider voice, plus how faithfully it matched the request."""

    name: str
    language: str
    gender: str
    quality: VoiceQuality
    #: True when the style chose this exact voice; False when the language
    #: offers only one voice for the gender and the style is prosody-only.
    style_matched_voice: bool = True


def resolve_online_voice(language: str, voice_id: str,
                         style: Optional[VoiceStyle] = None) -> Optional[ResolvedVoice]:
    """Pick the neural voice for a language and abstract voice id.

    Returns ``None`` when the language has no neural voice at all, so callers
    can report that honestly instead of silently substituting an unrelated one.
    """
    profile = get_profile(voice_id)
    table = ONLINE_VOICES.get(_base_language(language))
    if not table:
        return None
    candidates = table.get(profile.gender) or ()
    if not candidates:
        return None

    chosen_style = style or profile.default_style
    index = min(_STYLE_VOICE_INDEX.get(chosen_style, 0), len(candidates) - 1)
    return ResolvedVoice(
        name=candidates[index],
        language=_base_language(language),
        gender=profile.gender,
        quality=VoiceQuality.NEURAL,
        style_matched_voice=len(candidates) > 1,
    )


def online_languages() -> Tuple[str, ...]:
    """Language codes with a neural voice in this build."""
    return tuple(sorted(ONLINE_VOICES))


def _base_language(language: str) -> str:
    """``en-IN`` -> ``en``. AARMA's catalogue is keyed by base code."""
    return (language or "en").strip().lower().replace("_", "-").split("-", 1)[0]


def language_display_name(language: str) -> str:
    """"kn" -> "Kannada", for messages a person will read or hear.

    Error messages that name a language should name it the way the user does.
    "There's no offline kn voice" is a log line, not a sentence.
    """
    from language.manager import LANGUAGE_CATALOG

    info = LANGUAGE_CATALOG.get(_base_language(language))
    return info.english_name if info else (language or "that language")


# ------------------------------------------------------------- availability

@dataclass(frozen=True)
class VoiceAvailability:
    """What the user would actually hear for a given configuration.

    Carries the reason as well as the verdict, so the interface can say *why*
    something is unavailable rather than just greying a control out.
    """

    available: bool
    provider: str
    quality: VoiceQuality
    voice_name: str = ""
    language: str = "en"
    online: bool = False
    reason: str = ""
    #: A suggestion the user could act on, when one exists.
    suggestion: str = ""

    @property
    def status_label(self) -> str:
        if not self.available:
            return "Unavailable"
        return f"{self.quality.label} · {'Online' if self.online else 'Offline'}"

    def describe(self) -> str:
        if self.available:
            return f"{self.voice_name} ({self.status_label})"
        return self.reason or "No voice available."
