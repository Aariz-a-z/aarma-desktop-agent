"""Offline speech using the Windows built-in voices (SAPI 5).

This is the provider that works with no network, no account and no download, so
it is the floor AARMA can always stand on. Its quality is **standard**, not
natural, and nothing here pretends otherwise -- :attr:`SapiSpeaker.quality`
reports that honestly and the interface shows it.

Two details worth knowing:

**OneCore voices.** ``SAPI.SpVoice.GetVoices()`` returns only the legacy
"Desktop" voices -- typically just David and Zira. Windows also ships a OneCore
voice set (on this class of machine: Mark, and the Indian English pair Heera and
Ravi) which is newer and better. Those are reachable by enumerating a second
token category, with no registry modification and no extra dependency. If a
user installs Windows 11's "Natural" voices, they appear through the same route.

**Threading.** COM objects are apartment-bound, so the synthesiser is created
per-thread and speech is issued asynchronously and polled. That keeps
:meth:`stop` a simple thread-safe flag rather than a cross-thread COM call,
which would be neither simple nor safe.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

from speech.speaker import SpeechRequest, SpeechResult, SpeechSpeaker, VoiceDescriptor
from speech.voices import (
    ResolvedVoice,
    VoiceAvailability,
    VoiceQuality,
    VoiceStyle,
    get_profile,
    language_display_name,
)
from utils.logger import get_logger

logger = get_logger("speech.sapi")

#: The token category holding the newer OneCore voices.
ONECORE_CATEGORY = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices"

# SAPI constants.
_SVSFLAGS_ASYNC = 1
_SVSF_PURGE_BEFORE_SPEAK = 2
_SVSF_IS_XML = 8
_SPRS_IS_SPEAKING = 2

_POLL_INTERVAL = 0.05

#: Windows LCID prefixes to base language codes, for matching a voice to a
#: language. Only what the shipped Windows voice sets actually cover.
_LCID_LANGUAGES: Dict[str, str] = {
    "409": "en", "809": "en", "c09": "en", "4009": "en", "1009": "en",
    "439": "hi", "44b": "kn", "44a": "te", "449": "ta", "44c": "ml",
    "420": "ur", "401": "ar", "c0a": "es", "40a": "es", "40c": "fr",
}


def friendly_voice_name(description: str) -> str:
    """"Microsoft Zira Desktop - English (United States)" -> "Zira".

    Windows voice tokens carry vendor and locale in their description. The name
    is the part a user recognises; the rest is already shown elsewhere.
    """
    name = (description or "").replace("Microsoft ", "").replace(" Desktop", "")
    return name.split(" - ", 1)[0].strip() or "Windows voice"


class SapiSpeaker(SpeechSpeaker):
    """Windows SAPI 5, including the OneCore voice tokens."""

    name = "windows"
    display_name = "Windows built-in voice"
    requires_internet = False
    quality = VoiceQuality.STANDARD

    def __init__(self) -> None:
        self._local = threading.local()
        self._stop_requested = threading.Event()
        self._catalogue: Optional[List[VoiceDescriptor]] = None
        self._catalogue_lock = threading.Lock()

    # ----------------------------------------------------------- COM plumbing

    def _synthesiser(self):
        """This thread's SAPI object, created on first use."""
        voice = getattr(self._local, "voice", None)
        if voice is None:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            self._local.voice = voice
        return voice

    @staticmethod
    def _tokens() -> List[Tuple[object, str, str, str]]:
        """Every usable voice token: ``(token, description, gender, language)``.

        Reads the default category first, then OneCore, skipping duplicates by
        description so the same voice does not appear twice.
        """
        import win32com.client

        found: List[Tuple[object, str, str, str]] = []
        seen: set[str] = set()

        def collect(tokens) -> None:
            for index in range(tokens.Count):
                token = tokens.Item(index)
                try:
                    description = token.GetDescription()
                    if description in seen:
                        continue
                    seen.add(description)
                    gender = (token.GetAttribute("Gender") or "").lower()
                    lcid = (token.GetAttribute("Language") or "").split(";")[0].lower()
                    found.append((token, description, gender,
                                  _LCID_LANGUAGES.get(lcid, "en")))
                except Exception:  # noqa: BLE001 - a bad token must not hide good ones
                    continue

        try:
            collect(win32com.client.Dispatch("SAPI.SpVoice").GetVoices())
        except Exception:  # noqa: BLE001
            logger.exception("Could not enumerate default SAPI voices")

        try:
            category = win32com.client.Dispatch("SAPI.SpObjectTokenCategory")
            category.SetId(ONECORE_CATEGORY, False)
            collect(category.EnumerateTokens())
        except Exception:  # noqa: BLE001 - OneCore is absent on older Windows
            logger.debug("OneCore voice category unavailable")

        return found

    # ------------------------------------------------------------ availability

    def is_available(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            return bool(self._voice_catalogue())
        except Exception:  # noqa: BLE001
            return False

    def _voice_catalogue(self) -> List[VoiceDescriptor]:
        """Cached descriptors for every installed Windows voice."""
        with self._catalogue_lock:
            if self._catalogue is None:
                self._catalogue = [
                    VoiceDescriptor(
                        voice_id=description,
                        display_name=friendly_voice_name(description),
                        gender=gender or "female",
                        language=language,
                        quality=VoiceQuality.STANDARD,
                        provider=self.name,
                    )
                    for _token, description, gender, language in self._tokens()
                ]
            return list(self._catalogue)

    def voices(self, language: str = "en") -> List[VoiceDescriptor]:
        base = (language or "en").split("-")[0].lower()
        return [v for v in self._voice_catalogue() if v.language == base]

    def _select_token(self, language: str, gender: str):
        """Best token for a language and gender, or ``None``.

        Prefers a voice in the requested language *and* gender; falls back to
        the language with any gender. It never falls back to a different
        language -- an English voice reading Kannada is not a fallback, it is a
        different and worse failure.
        """
        base = (language or "en").split("-")[0].lower()
        candidates = [t for t in self._tokens() if t[3] == base]
        if not candidates:
            return None, None
        for token, description, token_gender, _lang in candidates:
            if token_gender == gender:
                return token, description
        token, description, _gender, _lang = candidates[0]
        return token, description

    def describe_voice(self, language: str, voice_id: str,
                       style: Optional[VoiceStyle] = None) -> VoiceAvailability:
        if not self.is_available():
            return VoiceAvailability(
                available=False, provider=self.name, quality=VoiceQuality.UNAVAILABLE,
                language=language,
                reason="No Windows speech voices are installed on this computer.",
                suggestion="Add a voice in Windows Settings, under Time & language → Speech.",
            )

        profile = get_profile(voice_id)
        _token, description = self._select_token(language, profile.gender)
        if description is None:
            spoken = language_display_name(language)
            return VoiceAvailability(
                available=False, provider=self.name, quality=VoiceQuality.UNAVAILABLE,
                language=language,
                reason=f"There's no offline {spoken} voice installed on this computer.",
                suggestion=(f"Connect to the internet to use a natural {spoken} voice, "
                            "or install the language's speech pack in Windows Settings."),
            )
        return VoiceAvailability(
            available=True, provider=self.name, quality=self.quality,
            voice_name=friendly_voice_name(description),
            language=language, online=False,
        )

    # ------------------------------------------------------------------ speech

    def speak(self, request: SpeechRequest) -> SpeechResult:
        options = request.normalised()
        profile = options.profile
        self._stop_requested.clear()

        try:
            synthesiser = self._synthesiser()
        except Exception as exc:  # noqa: BLE001 - COM/init failures vary
            logger.error("SAPI initialisation failed",
                         extra={"context": {"error": str(exc)}})
            return SpeechResult.failure(self.name,
                                        "The Windows speech engine could not start.",
                                        error_code="provider_unavailable")

        token, description = self._select_token(options.language, profile.gender)
        if token is None:
            spoken = language_display_name(options.language)
            return SpeechResult.failure(
                self.name,
                f"There's no offline {spoken} voice on this computer.",
                error_code="voice_unavailable",
            )

        try:
            synthesiser.Voice = token
            # SAPI rate is -10..10 around a natural 0.
            synthesiser.Rate = int(round((options.rate - 1.0) * 10))
            synthesiser.Volume = int(round(options.volume * 100))

            markup = options.escaped_text()
            flags = _SVSFLAGS_ASYNC
            if options.pitch:
                # Pitch is only reachable through SAPI's XML markup.
                markup = f'<pitch middle="{int(round(options.pitch))}"/>{markup}'
                flags |= _SVSF_IS_XML

            synthesiser.Speak(markup, flags)
            interrupted = self._wait_for_finish(synthesiser)
        except Exception as exc:  # noqa: BLE001 - COM raises a wide range
            logger.error("SAPI speech failed", extra={"context": {"error": str(exc)}})
            return SpeechResult.failure(self.name, "I couldn't play that out loud.",
                                        error_code="speech_failed")

        if interrupted:
            return SpeechResult.cancelled(self.name)

        return SpeechResult.success(self.name, ResolvedVoice(
            name=friendly_voice_name(description or ""),
            language=options.language,
            gender=profile.gender,
            quality=self.quality,
            style_matched_voice=False,
        ))

    def _wait_for_finish(self, synthesiser) -> bool:
        """Poll until speech ends. Returns True if it was interrupted."""
        while not self._stop_requested.is_set():
            try:
                if synthesiser.Status.RunningState != _SPRS_IS_SPEAKING:
                    return False
            except Exception:  # noqa: BLE001 - status can fail during teardown
                return False
            time.sleep(_POLL_INTERVAL)

        try:
            # An empty utterance with the purge flag clears the queue immediately.
            synthesiser.Speak("", _SVSF_PURGE_BEFORE_SPEAK)
        except Exception:  # noqa: BLE001
            logger.debug("Purge after stop failed")
        return True

    def stop(self) -> None:
        self._stop_requested.set()

    def unavailable_reason(self) -> str:
        if sys.platform != "win32":
            return "Windows voices are only available on Windows."
        return "No Windows speech voices are installed on this computer."
