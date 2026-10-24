"""The single source of truth for every user-adjustable AARMA setting.

Design rules enforced here:

* No module hardcodes an assistant name, language, wake word or permission.
  Everything is read from :class:`ConfigManager`.
* The on-disk representation is plain JSON so a user can inspect (and, if they
  really want to, repair) their own configuration.
* Writes are atomic -- a crash mid-save can never leave a truncated config.
* Permissions are stored as ``{permission_name: bool}`` strings rather than as
  the :class:`security.permissions.Permission` enum, which keeps ``config``
  free of any dependency on ``security`` and avoids an import cycle.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from utils.logger import get_logger
from utils.paths import get_paths

logger = get_logger("config")

#: Bumped whenever the persisted layout changes in a non-additive way.
CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AssistantProfile:
    """A shipped assistant persona offered during first-run setup."""

    key: str
    display_name: str
    gender: str
    voice_id: str
    wake_word: str
    description: str


#: The two default identities required by specification section 4.  These are
#: *starting points*: the user may rename the assistant or change its voice at
#: any time, and the chosen values are what the rest of the app reads.
ASSISTANT_PROFILES: Dict[str, AssistantProfile] = {
    "asma": AssistantProfile(
        key="asma",
        display_name="Asma",
        gender="female",
        voice_id="female_warm",
        wake_word="asma",
        description="Warm, concise and conversational.",
    ),
    "aariz": AssistantProfile(
        key="aariz",
        display_name="Aariz",
        gender="male",
        voice_id="male_calm",
        wake_word="hey aariz",
        description="Calm, direct and to the point.",
    ),
}

DEFAULT_PROFILE_KEY = "asma"


#: The wake phrase is the assistant's name, said on its own: "Friday" wakes
#: Friday.  Names are what people actually call each other, and requiring a
#: prefix would make the assistant answer to something nobody would say.
#:
#: One exception, and it is a fact about a model rather than a preference. The
#: shipped Phase-15 detector was trained, evaluated and frozen on the phrase
#: "hey aariz"; it hears that and nothing else. Giving the Aariz identity a
#: bare-name phrase would leave the only validated model in the project
#: unreachable, so that one identity keeps the phrase its model knows. Any
#: future shipped model trained on a bare name simply does not appear here.
LEGACY_WAKE_PHRASES: Dict[str, str] = {"aariz": "hey aariz"}


def normalise_wake_phrase(phrase: str) -> str:
    """Fold a wake phrase to the form used for storage and comparison.

    Case and stray whitespace are presentation, not identity: "Hey  Aariz",
    "hey aariz" and "HEY AARIZ" are the same phrase and must resolve to the
    same detector.  Returns "" for anything that folds away to nothing, which
    callers treat as "no wake phrase configured" rather than as a valid empty
    one.
    """
    return " ".join(str(phrase or "").split()).lower()


def wake_phrase_for_name(name: str) -> str:
    """The wake phrase implied by an assistant name: Friday -> "friday".

    Empty for a nameless assistant.  Deriving rather than storing an unrelated
    string is what keeps the spoken phrase and the displayed name in step after
    a rename -- including a rename to a name nothing has been trained on, which
    is a valid state and not an error.

    See :data:`LEGACY_WAKE_PHRASES` for the single identity that differs.
    """
    cleaned = " ".join(str(name or "").split())
    if not cleaned:
        return ""
    legacy = LEGACY_WAKE_PHRASES.get(cleaned.lower())
    if legacy:
        return normalise_wake_phrase(legacy)
    return normalise_wake_phrase(cleaned)


@dataclass
class AssistantIdentity:
    """Who the assistant currently is, from the user's point of view."""

    profile_key: str = DEFAULT_PROFILE_KEY
    name: str = ASSISTANT_PROFILES[DEFAULT_PROFILE_KEY].display_name
    gender: str = ASSISTANT_PROFILES[DEFAULT_PROFILE_KEY].gender
    voice_id: str = ASSISTANT_PROFILES[DEFAULT_PROFILE_KEY].voice_id
    wake_word: str = ASSISTANT_PROFILES[DEFAULT_PROFILE_KEY].wake_word
    personality: str = "balanced"

    @classmethod
    def from_profile(cls, profile: AssistantProfile) -> "AssistantIdentity":
        return cls(
            profile_key=profile.key,
            name=profile.display_name,
            gender=profile.gender,
            voice_id=profile.voice_id,
            wake_word=profile.wake_word,
        )


@dataclass
class LanguageSettings:
    """Interface, input and output languages are configured independently.

    Specification section 6 requires these to be separable: a user may want an
    English interface while speaking Hindi, or vice versa.
    """

    interface: str = "en"
    speech_input: str = "en"
    speech_output: str = "en"
    auto_detect: bool = False


@dataclass
class PermissionSettings:
    """Capability switches the user controls (specification section 12)."""

    granted: Dict[str, bool] = field(default_factory=dict)


@dataclass
class SecuritySettings:
    """Passphrase-protected mode.

    Only the *derivation parameters* and the resulting digest are stored -- the
    passphrase itself is never written to disk or to a log (section 13).
    """

    enabled: bool = False
    passphrase_hash: Optional[str] = None
    passphrase_salt: Optional[str] = None
    kdf_iterations: int = 0
    confirm_high_risk_actions: bool = True
    require_passphrase_for_high_risk: bool = False


@dataclass
class VoiceSettings:
    """How the assistant sounds (specification sections 3 and 4).

    Kept separate from :class:`AssistantIdentity`, which owns *which* voice the
    assistant uses. These are delivery controls: they apply to whatever voice is
    selected, and survive a change of voice, language or provider.
    """

    # -- shared: these apply to whichever voice is selected -----------------
    enabled: bool = True              # Speak replies out loud.
    provider_mode: str = "auto"       # auto | online | offline
    volume: float = 0.9               # 0.0 - 1.0
    speak_notifications: bool = True  # Reminders and alerts (Phase 3).

    # -- active: the selected voice's own delivery --------------------------
    style: str = "warm"               # warm | clear | calm | professional | energetic
    rate: float = 1.0                 # 1.0 is the voice's natural pace.
    pitch: float = 0.0                # Semitone-ish offset; 0.0 is natural.

    # -- remembered per presentation ----------------------------------------
    #
    # Flat fields rather than a nested dataclass, because `_build` constructs
    # settings from JSON one level deep and a nested one would come back as a
    # plain dict. Switching presentation copies the active values into the
    # outgoing set and the incoming set into the active ones, which is what
    # makes male -> female -> male restore what the user had rather than
    # leaving the other profile's rate and pitch behind.
    male_voice_id: str = "male_calm"
    male_style: str = "calm"
    male_rate: float = 1.0
    male_pitch: float = 0.0
    female_voice_id: str = "female_warm"
    female_style: str = "warm"
    female_rate: float = 1.0
    female_pitch: float = 0.0


@dataclass
class SpeechInputSettings:
    """Microphone and speech-recognition preferences (specification section 22).

    The input *language* is not duplicated here -- it lives in
    :class:`LanguageSettings.speech_input`, which is already the authority for
    what the user speaks.
    """

    enabled: bool = True
    #: Recogniser accuracy/size trade-off: tiny | base | small | medium.
    #: ``small`` is the benchmarked default -- see docs/SPEECH-INPUT.md.
    model_size: str = "small"
    #: Input device **name**; ``None`` means "use the Windows default".
    #: Stored by name because PortAudio indexes shift when hardware changes,
    #: so a saved index can quietly come to mean a different device.
    device_name: Optional[str] = None
    #: Last known index for the saved name. A hint only; never authoritative.
    device_index: Optional[int] = None
    #: Hard ceiling on one utterance.
    max_seconds: float = 15.0
    #: Stop this long after the speaker stops.
    silence_timeout: float = 1.2
    #: Give up this long after the button if nobody starts speaking, instead of
    #: making the user wait out ``max_seconds``.
    no_speech_timeout: float = 6.0
    #: Confidence required of a *long* utterance. Short commands are held to a
    #: proportionally lower bar -- see :func:`speech.listener.confidence_floor`.
    #: Lowered from 0.45 after measurement showed it rejecting correct readings
    #: of short commands like "Open Chrome."
    confidence_threshold: float = 0.35
    #: Write the captured audio to disk for debugging. Off, and stays off unless
    #: deliberately enabled; normal operation never touches the disk.
    save_debug_audio: bool = False
    #: 0 = only clear close speech, 1 = strains to hear quiet or distant speech.
    vad_sensitivity: float = 0.5
    #: Wait this long after the assistant stops talking before capturing, so
    #: its own voice cannot be transcribed as a command.
    post_speech_delay: float = 0.25


@dataclass
class RuntimeSettings:
    """Behavioural flags that are not tied to identity, language or security."""

    network_mode: str = "auto"  # auto | online | offline
    memory_enabled: bool = True
    notifications_enabled: bool = True
    llm_provider: str = "none"  # Phase 2 will add real providers.
    search_engine: str = "google"
    theme: str = "dark"
    log_level: str = "INFO"
    #: Where "what's the weather?" means, when the user does not name a place.
    #: Empty by default and never filled in automatically: guessing a location
    #: from the network address would answer about somewhere the user never
    #: mentioned, and silently.
    weather_location: str = ""


@dataclass
class WorkspaceSettings:
    """Directories the user has approved AARMA to work inside.

    Empty by default, and that is deliberate: the content folders Windows
    already reports (Documents, Desktop, Downloads...) are discovered at
    runtime and need no configuration, while anything beyond them is a
    decision only the user can make. A shipped default pointing at, say,
    ``C:\Projects`` would be a guess about someone else's machine.

    Stored as strings because a path is a string on disk; they are resolved and
    validated on every use rather than trusted as written.
    """

    roots: List[str] = field(default_factory=list)
    #: Whether discovery may also use the Windows content folders. On by
    #: default because the user already granted the filesystem permission for
    #: exactly those, and off is available for anyone who wants AARMA confined
    #: to explicitly named project directories.
    include_content_folders: bool = True


@dataclass
class AppConfig:
    """The complete persisted configuration document."""

    schema_version: int = CONFIG_SCHEMA_VERSION
    first_run_completed: bool = False
    assistant: AssistantIdentity = field(default_factory=AssistantIdentity)
    language: LanguageSettings = field(default_factory=LanguageSettings)
    permissions: PermissionSettings = field(default_factory=PermissionSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    voice: VoiceSettings = field(default_factory=VoiceSettings)
    speech_input: SpeechInputSettings = field(default_factory=SpeechInputSettings)
    workspace: WorkspaceSettings = field(default_factory=WorkspaceSettings)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AppConfig":
        """Rebuild a config from JSON, tolerating missing or unknown keys.

        Unknown keys are dropped and missing keys fall back to defaults, so a
        config written by an older (or slightly newer) build still loads.
        """
        return cls(
            schema_version=int(raw.get("schema_version", CONFIG_SCHEMA_VERSION)),
            first_run_completed=bool(raw.get("first_run_completed", False)),
            assistant=_build(AssistantIdentity, raw.get("assistant")),
            language=_build(LanguageSettings, raw.get("language")),
            permissions=_build(PermissionSettings, raw.get("permissions")),
            security=_build(SecuritySettings, raw.get("security")),
            runtime=_build(RuntimeSettings, raw.get("runtime")),
            voice=_build(VoiceSettings, raw.get("voice")),
            speech_input=_build(SpeechInputSettings, raw.get("speech_input")),
            workspace=_build(WorkspaceSettings, raw.get("workspace")),
        )


def _remember(voice: VoiceSettings, presentation: str) -> VoiceSettings:
    """Copy the active delivery settings into ``presentation``'s stored set."""
    if presentation not in ("male", "female"):
        return voice
    return replace(voice, **{f"{presentation}_style": voice.style,
                             f"{presentation}_rate": voice.rate,
                             f"{presentation}_pitch": voice.pitch})


def _load(voice: VoiceSettings, presentation: str) -> VoiceSettings:
    """Make ``presentation``'s stored delivery settings the active ones."""
    if presentation not in ("male", "female"):
        return voice
    return replace(voice,
                   style=getattr(voice, f"{presentation}_style"),
                   rate=getattr(voice, f"{presentation}_rate"),
                   pitch=getattr(voice, f"{presentation}_pitch"))


def _build(cls: type, raw: Any):
    """Instantiate a settings dataclass from ``raw``, ignoring unknown fields."""
    if not isinstance(raw, Mapping):
        return cls()
    allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return cls(**{key: value for key, value in raw.items() if key in allowed})


ConfigListener = Callable[[AppConfig], None]


class ConfigManager:
    """Loads, mutates and persists the :class:`AppConfig`.

    The manager is thread-safe because the GUI thread and the engine may both
    write settings.  Listeners are notified after every successful save, which
    is how the dashboard keeps its header in sync with voice/CLI commands such
    as "change your name to Nova".
    """

    def __init__(self, config_file: Optional[Path] = None, *, autosave: bool = True) -> None:
        self._path = config_file or get_paths().config_file
        self._autosave = autosave
        self._lock = threading.RLock()
        self._listeners: list[ConfigListener] = []
        self._config = AppConfig()

    # ------------------------------------------------------------------ load

    @property
    def path(self) -> Path:
        return self._path

    @property
    def config(self) -> AppConfig:
        """The current configuration.  Treat the returned object as read-only."""
        with self._lock:
            return self._config

    def load(self) -> AppConfig:
        """Read the config file, falling back to defaults if it is absent."""
        with self._lock:
            if not self._path.exists():
                logger.info("No config file found; starting from defaults",
                            extra={"context": {"path": str(self._path)}})
                self._config = AppConfig()
                return self._config
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._config = AppConfig.from_dict(raw)
                logger.info("Configuration loaded",
                            extra={"context": {"path": str(self._path),
                                               "schema_version": self._config.schema_version}})
            except (OSError, ValueError) as exc:
                # A corrupt config must never stop the app from starting.
                logger.error("Configuration unreadable; using defaults",
                             extra={"context": {"path": str(self._path), "error": str(exc)}})
                self._config = AppConfig()
            return self._config

    # ------------------------------------------------------------------ save

    def save(self) -> None:
        """Persist the configuration atomically."""
        with self._lock:
            payload = json.dumps(self._config.to_dict(), indent=2, sort_keys=True)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self._path)
            logger.debug("Configuration saved", extra={"context": {"path": str(self._path)}})

    # ---------------------------------------------------------------- mutate

    def update(self, **sections: Any) -> AppConfig:
        """Replace whole sections, e.g. ``update(assistant=new_identity)``."""
        with self._lock:
            self._config = replace(self._config, **sections)
            if self._autosave:
                self.save()
            snapshot = self._config
        self._notify(snapshot)
        return snapshot

    def update_assistant(self, **fields: Any) -> AppConfig:
        return self.update(assistant=replace(self.config.assistant, **fields))

    def update_language(self, **fields: Any) -> AppConfig:
        return self.update(language=replace(self.config.language, **fields))

    def update_runtime(self, **fields: Any) -> AppConfig:
        return self.update(runtime=replace(self.config.runtime, **fields))

    def update_security(self, **fields: Any) -> AppConfig:
        return self.update(security=replace(self.config.security, **fields))

    def update_voice(self, **fields: Any) -> AppConfig:
        return self.update(voice=replace(self.config.voice, **fields))

    # ------------------------------------------------------- voice profiles

    def select_voice(self, voice_id: str) -> AppConfig:
        """Choose a voice, and move the presentation with it.

        The presentation is a property of the voice, so it is written in the
        same update rather than left for something else to notice. Storing them
        separately is what produced a configuration claiming a female profile
        while a male voice spoke.
        """
        from speech.voices import VOICE_PROFILES

        profile = VOICE_PROFILES.get(voice_id)
        if profile is None:
            raise ValueError(f"unknown voice {voice_id!r}")
        voice = _remember(self.config.voice, self.config.assistant.gender)
        voice = replace(voice, style=profile.default_style.value)
        voice = _load(voice, profile.gender)
        return self.update(
            assistant=replace(self.config.assistant, voice_id=voice_id,
                              gender=profile.gender),
            voice=replace(voice, **{f"{profile.gender}_voice_id": voice_id}))

    def set_voice_presentation(self, presentation: str) -> Tuple[bool, str]:
        """Switch the whole profile. Returns ``(changed, explanation)``.

        Refuses rather than substituting when the requested presentation has no
        voice: answering a request for a male voice with a female one, while
        displaying "Male", is the failure mode this whole change exists to end.
        """
        from speech.voices import VOICE_PROFILES, profiles_for_gender

        wanted = str(presentation or "").strip().lower()
        if wanted not in ("male", "female"):
            return False, f"I don't have a {presentation!r} voice profile."

        current = self.config.assistant.gender
        remembered = getattr(self.config.voice, f"{wanted}_voice_id", "")
        # The remembered voice first, then any other voice of the same
        # presentation. Never one of the other presentation's.
        candidates = [remembered] if remembered else []
        candidates += [p.voice_id for p in profiles_for_gender(wanted)]
        chosen = next((v for v in candidates if v in VOICE_PROFILES
                       and VOICE_PROFILES[v].gender == wanted), "")
        if not chosen:
            return False, f"No available {wanted} voice is configured."

        if current == wanted and self.config.assistant.voice_id == chosen:
            return False, f"I'm already using a {wanted} voice."

        voice = _remember(self.config.voice, current)
        voice = _load(voice, wanted)
        self.update(
            assistant=replace(self.config.assistant, voice_id=chosen,
                              gender=wanted),
            voice=replace(voice, **{f"{wanted}_voice_id": chosen}))
        logger.info("Voice presentation changed", extra={"context": {
            "presentation": wanted, "voice_id": chosen}})
        return True, f"Switched to a {wanted} voice."

    def update_workspace(self, **fields: Any) -> AppConfig:
        return self.update(workspace=replace(self.config.workspace, **fields))

    def update_speech_input(self, **fields: Any) -> AppConfig:
        return self.update(speech_input=replace(self.config.speech_input, **fields))

    def reset_speech_input(self) -> AppConfig:
        """Restore speech input to defaults, leaving everything else alone.

        Scoped deliberately narrowly: identity, permissions, security, memories
        and every other setting are untouched. Someone resetting a microphone
        problem should not lose their assistant's name.
        """
        logger.info("Speech input settings reset to defaults")
        return self.update(speech_input=SpeechInputSettings())

    def reset_all_settings(self) -> AppConfig:
        """Restore every *setting* to first-run defaults.

        Stored data -- the conversation transcript, the audit trail, and any
        future memories -- lives in the database and is **not** touched here.
        Deleting a user's data is a separate, explicit action.
        """
        logger.info("All settings reset to defaults")
        return self.update(
            first_run_completed=False,
            assistant=AssistantIdentity(),
            language=LanguageSettings(),
            permissions=PermissionSettings(),
            security=SecuritySettings(),
            runtime=RuntimeSettings(),
            voice=VoiceSettings(),
            speech_input=SpeechInputSettings(),
        )

    def set_permission(self, permission_name: str, granted: bool) -> AppConfig:
        """Grant or revoke a single permission by name."""
        with self._lock:
            granted_map = dict(self.config.permissions.granted)
            granted_map[permission_name] = bool(granted)
            return self.update(permissions=PermissionSettings(granted=granted_map))

    def mark_first_run_complete(self) -> AppConfig:
        return self.update(first_run_completed=True)

    # ------------------------------------------------------------- listeners

    def add_listener(self, listener: ConfigListener) -> None:
        """Register a callback invoked after every successful mutation."""
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: ConfigListener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _notify(self, snapshot: AppConfig) -> None:
        for listener in list(self._listeners):
            try:
                listener(snapshot)
            except Exception:  # noqa: BLE001 - a bad listener must not break saving
                logger.exception("Configuration listener raised")


def load_config(config_file: Optional[Path] = None) -> ConfigManager:
    """Convenience helper: build a manager and immediately load from disk."""
    manager = ConfigManager(config_file)
    manager.load()
    return manager
