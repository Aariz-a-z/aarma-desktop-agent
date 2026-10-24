"""Language catalogue, resolution and message lookup.

Phase 1 ships **English** as the only fully translated interface language.
Every other language in :data:`LANGUAGE_CATALOG` is *declared* -- it can be
selected, it persists across restarts, and the rest of the system routes
through it -- but its interface strings fall back to English until the
translation work in Phase 2 lands.  :meth:`LanguageManager.status_of` reports
that honestly rather than pretending a language is finished.

Splitting "which languages exist" from "which languages are translated" is what
lets Phases 2-4 add Hindi, Kannada, Telugu, Tamil and Malayalam without
touching any other subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional

from config.settings import ConfigManager
from utils.logger import get_logger

logger = get_logger("language")

DEFAULT_LANGUAGE = "en"


class LanguageStatus(str, Enum):
    """How complete a language is in the current build."""

    TRANSLATED = "translated"   # Interface strings exist in this language.
    DECLARED = "declared"       # Selectable and persisted; strings fall back to English.


@dataclass(frozen=True)
class LanguageInfo:
    code: str
    english_name: str
    native_name: str
    status: LanguageStatus
    phase: int  # Roadmap phase in which full support is planned.


def _info(code: str, english: str, native: str, status: LanguageStatus, phase: int) -> LanguageInfo:
    return LanguageInfo(code=code, english_name=english, native_name=native,
                        status=status, phase=phase)


#: Languages AARMA knows about.  Adding a language here makes it selectable in
#: the GUI, the CLI and via voice without any other code change.
LANGUAGE_CATALOG: Dict[str, LanguageInfo] = {
    "en": _info("en", "English", "English", LanguageStatus.TRANSLATED, 1),
    "hi": _info("hi", "Hindi", "हिन्दी", LanguageStatus.DECLARED, 2),
    "kn": _info("kn", "Kannada", "ಕನ್ನಡ", LanguageStatus.DECLARED, 2),
    "te": _info("te", "Telugu", "తెలుగు", LanguageStatus.DECLARED, 2),
    "ta": _info("ta", "Tamil", "தமிழ்", LanguageStatus.DECLARED, 2),
    "ml": _info("ml", "Malayalam", "മലയാളം", LanguageStatus.DECLARED, 2),
    "ur": _info("ur", "Urdu", "اردو", LanguageStatus.DECLARED, 3),
    "ar": _info("ar", "Arabic", "العربية", LanguageStatus.DECLARED, 3),
    "es": _info("es", "Spanish", "Español", LanguageStatus.DECLARED, 3),
    "fr": _info("fr", "French", "Français", LanguageStatus.DECLARED, 3),
}

#: Extra spellings users actually say, mapped onto catalogue codes.
_LANGUAGE_ALIASES: Dict[str, str] = {
    "english": "en", "eng": "en", "en-us": "en", "en-in": "en", "en_gb": "en",
    "hindi": "hi", "hindustani": "hi", "hi-in": "hi",
    "kannada": "kn", "kannad": "kn", "kn-in": "kn",
    "telugu": "te", "telegu": "te", "te-in": "te",
    "tamil": "ta", "tamizh": "ta", "ta-in": "ta",
    "malayalam": "ml", "malyalam": "ml", "ml-in": "ml",
    "urdu": "ur", "arabic": "ar", "spanish": "es", "espanol": "es",
    "french": "fr", "francais": "fr",
}


def resolve_language_code(value: str) -> Optional[str]:
    """Map a code or a spoken language name onto a catalogue code.

    Returns ``None`` when the language is not in the catalogue, which callers
    turn into a "language not supported" response rather than a crash.
    """
    if not value:
        return None
    needle = value.strip().lower().replace("_", "-")
    if needle in LANGUAGE_CATALOG:
        return needle
    if needle in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[needle]
    base = needle.split("-", 1)[0]
    if base in LANGUAGE_CATALOG:
        return base
    for info in LANGUAGE_CATALOG.values():
        if needle in (info.english_name.lower(), info.native_name.lower()):
            return info.code
    return None


#: Interface strings.  ``en`` is authoritative; any key missing from another
#: language falls back to the English entry.
MESSAGES: Dict[str, Dict[str, str]] = {
    "en": {
        "app.tagline": "Adaptive AI Voice-Controlled Desktop Agent",
        "response.greeting": "Hello, I'm {assistant}. How can I help?",
        "response.ready": "{assistant} is ready. Type a command, or 'help' to see what I can do.",
        "response.goodbye": "Goodbye.",
        "response.unknown_command": (
            "I didn't understand that yet. Type 'help' to see the commands I support in this build."
        ),
        "wake.renamed_reminder": "I'm {name} now, not {former}. Say \"{name}\" and I'll do it.",
        "wake.session_locked": "Say \"Hey Aariz\" first to start this session.",
        "wake.session_ended": "Voice session ended. Say \"Hey Aariz\" when you need me.",
        "wake.session_already_ended": "The voice session isn't active — say \"Hey Aariz\" to start it.",
        "wake.unlocked": "Listening for \"{name}\" now.",
        "wake.waiting": "Waiting for \"Hey Aariz\"",
        "wake.listening": "Listening for \"{phrase}\"",
        "wake.not_enrolled_reason": "No voice wake is enrolled for {name}. Record one from Settings → Wake word, or say \"Hey Aariz\" to wake me.",
        "wake.not_enrolled": "{name} — voice wake not enrolled",
        "response.confirm_superseded": "(I've cancelled the confirmation for '{action}'.)",
        "response.plan_unmet": "I didn't understand this part: {steps}.",
        "response.plan_stopped": "I stopped there, so I haven't done: {steps}.",
        "response.time": "The current time is {time}.",
        "response.date": "Today is {date}.",
        "response.name_changed": "Done. You can call me {assistant} now.",
        "response.language_changed": "Language set to {language}.",
        "response.language_declared": (
            "Language set to {language}. My interface text is still English in this build "
            "— full {language} support is planned for a later phase."
        ),
        "response.language_unknown": "I don't know the language '{value}' yet.",
        "response.permission_denied": (
            "I don't currently have permission to do that. Enable '{permission}' in Settings "
            "→ Permissions and ask me again."
        ),
        "response.offline": "You're currently offline, so I'll use local capabilities only.",
        "response.needs_internet": "That needs an internet connection and you're currently offline.",
        "response.action_failed": "I couldn't complete that action. {reason}",
        "response.confirm_required": (
            "'{action}' is a high-risk action. Reply 'yes' to confirm or 'no' to cancel."
        ),
        "response.confirm_cancelled": "Cancelled. I haven't changed anything.",
        "response.confirm_expired": "That confirmation expired, so I didn't run it. Ask me again if you still want it.",
        "response.invalid_parameters": "I need a bit more detail: {reason}",
        "response.security_locked": "I'm locked. Unlock me with your passphrase to continue.",
        "error.microphone_unavailable": "Microphone is unavailable. You can use text input instead.",
        "error.internal": "Something went wrong on my side. The details are in the log file.",
        "status.online": "Online",
        "status.offline": "Offline",
        "status.idle": "Idle",
        "status.listening": "Listening",
        "status.thinking": "Thinking",
        "status.working": "Working",
        "status.speaking": "Speaking",
        "status.listening_now": "Listening",
        "status.understanding": "Understanding",
    }
}


class LanguageManager:
    """Resolves the active language and renders every user-facing string.

    The manager reads from and writes to :class:`~config.settings.ConfigManager`
    so a language change made by voice, by the CLI or by the GUI is one and the
    same operation, and survives a restart.
    """

    def __init__(self, config: ConfigManager) -> None:
        self._config = config

    # ------------------------------------------------------------- accessors

    @property
    def config(self) -> ConfigManager:
        """The shared configuration this manager reads and writes."""
        return self._config

    @property
    def interface_language(self) -> str:
        return self._config.config.language.interface

    @property
    def input_language(self) -> str:
        return self._config.config.language.speech_input

    @property
    def output_language(self) -> str:
        return self._config.config.language.speech_output

    def info(self, code: Optional[str] = None) -> LanguageInfo:
        """Catalogue entry for ``code`` (defaults to the interface language)."""
        return LANGUAGE_CATALOG.get(code or self.interface_language, LANGUAGE_CATALOG[DEFAULT_LANGUAGE])

    def status_of(self, code: str) -> LanguageStatus:
        return self.info(code).status

    def display_name(self, code: Optional[str] = None) -> str:
        return self.info(code).english_name

    def available(self) -> Iterable[LanguageInfo]:
        return LANGUAGE_CATALOG.values()

    # --------------------------------------------------------------- mutation

    def set_language(self, value: str, *, scope: str = "all") -> Optional[LanguageInfo]:
        """Change the active language.

        ``scope`` selects which of the three independent language settings are
        affected: ``"all"``, ``"interface"``, ``"input"`` or ``"output"``.
        Returns the catalogue entry, or ``None`` if the language is unknown.
        """
        code = resolve_language_code(value)
        if code is None:
            logger.info("Rejected unknown language", extra={"context": {"value": value}})
            return None

        fields: Dict[str, str] = {}
        if scope in ("all", "interface"):
            fields["interface"] = code
        if scope in ("all", "input"):
            fields["speech_input"] = code
        if scope in ("all", "output"):
            fields["speech_output"] = code

        self._config.update_language(**fields)
        info = LANGUAGE_CATALOG[code]
        logger.info("Language changed",
                    extra={"context": {"code": code, "scope": scope, "status": info.status.value}})
        return info

    # ------------------------------------------------------------- rendering

    def translate(self, key: str, /, **params: object) -> str:
        """Render message ``key`` in the active interface language.

        Falls back to English, and then to the key itself, so a missing string
        degrades into something diagnosable instead of raising.
        """
        return self.translate_in(self.interface_language, key, **params)

    def translate_in(self, code: str, key: str, /, **params: object) -> str:
        table: Mapping[str, str] = MESSAGES.get(code, {})
        template = table.get(key) or MESSAGES[DEFAULT_LANGUAGE].get(key)
        if template is None:
            logger.warning("Missing message key", extra={"context": {"key": key, "language": code}})
            return key
        try:
            return template.format(**params)
        except (KeyError, IndexError) as exc:
            logger.warning("Message formatting failed",
                           extra={"context": {"key": key, "error": str(exc)}})
            return template

    #: Short alias -- ``lang.t("response.goodbye")`` reads well at call sites.
    t = translate
