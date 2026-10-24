"""Voice input and output.

**Phase 2A ships voice output.** :class:`~speech.engine.SpeechEngine` is the
entry point: it chooses a provider, filters what may be spoken, and runs
synthesis on a worker thread so the interface never waits for audio.

Two providers ship. :class:`~speech.providers.edge.EdgeNeuralSpeaker` gives
natural neural voices in every language AARMA declares, and needs a connection.
:class:`~speech.providers.sapi.SapiSpeaker` uses Windows' own voices, works
offline, and is honestly labelled *standard* rather than natural. The engine
prefers the first and falls back to the second **explicitly**, telling the
caller when it does.

**Phase 2B ships voice input.** :class:`~speech.input_engine.ListeningEngine`
captures one utterance, transcribes it locally, and produces *text* -- it
executes nothing. The interface passes that text to the ordinary engine, so a
spoken command is audited exactly like a typed one. Recognition runs on-device
through :class:`~speech.providers.whisper_local.LocalWhisperRecognizer`; audio
is never written to disk and never uploaded.

Wake-word detection is provided by :class:`~speech.wakeword_cascade.CascadeWakeWordDetector`,
a two-stage engine behind the same :class:`~speech.wakeword.WakeWordDetector`
interface.  It reports itself unavailable -- with a reason -- when its optional
dependency or model assets are absent, so a machine without them behaves exactly
as it did before.
"""

from speech.audio import AudioPlayback, MciAudioPlayback, NullAudioPlayback, create_playback
from speech.capture import (
    MicrophoneCapture,
    MicrophoneDevice,
    default_device,
    has_microphone,
    list_devices,
)
from speech.engine import SpeechEngine, SpeechOutcome, SpeechPurpose, default_providers
from speech.input_engine import (
    ListeningEngine,
    ListeningReadiness,
    ListeningState,
    default_recognizer,
)
from speech.listener import (
    NullSpeechListener,
    NullSpeechRecognizer,
    RecognitionOutcome,
    RecognizerAvailability,
    SpeechListener,
    SpeechRecognizer,
    TranscriptionResult,
)
from speech.speaker import (
    DECLARED_VOICES,
    NullSpeechSpeaker,
    SpeechRequest,
    SpeechResult,
    SpeechSpeaker,
    VoiceDescriptor,
)
from speech.text import SpeechText, prepare_for_speech, preview_line
from speech.voices import (
    VOICE_PROFILES,
    VoiceAvailability,
    VoiceProfile,
    VoiceQuality,
    VoiceStyle,
    get_profile,
    resolve_online_voice,
)
from speech.wakeword_store import (
    STORE as CUSTOM_WAKE_WORD_STORE,
    CustomWakeWord,
    CustomWakeWordError,
    CustomWakeWordStore,
)
from speech.wakeword import (
    REGISTRY as WAKE_WORD_REGISTRY,
    MissingWakeWordDetector,
    NullWakeWordDetector,
    WakeWordDetector,
    WakeWordEvent,
    WakeWordProvider,
    WakeWordRegistry,
)
from speech.wakeword_cascade import CascadeWakeWordDetector

__all__ = [
    "AudioPlayback",
    "DECLARED_VOICES",
    "ListeningEngine",
    "ListeningReadiness",
    "ListeningState",
    "MciAudioPlayback",
    "MicrophoneCapture",
    "MicrophoneDevice",
    "NullAudioPlayback",
    "NullSpeechListener",
    "NullSpeechRecognizer",
    "NullSpeechSpeaker",
    "CascadeWakeWordDetector",
    "CUSTOM_WAKE_WORD_STORE",
    "CustomWakeWord",
    "CustomWakeWordError",
    "CustomWakeWordStore",
    "MissingWakeWordDetector",
    "NullWakeWordDetector",
    "WAKE_WORD_REGISTRY",
    "WakeWordProvider",
    "WakeWordRegistry",
    "RecognitionOutcome",
    "RecognizerAvailability",
    "SpeechEngine",
    "SpeechListener",
    "SpeechRecognizer",
    "SpeechOutcome",
    "SpeechPurpose",
    "SpeechRequest",
    "SpeechResult",
    "SpeechSpeaker",
    "SpeechText",
    "TranscriptionResult",
    "VOICE_PROFILES",
    "VoiceAvailability",
    "VoiceDescriptor",
    "VoiceProfile",
    "VoiceQuality",
    "VoiceStyle",
    "WakeWordDetector",
    "WakeWordEvent",
    "create_playback",
    "default_device",
    "default_providers",
    "default_recognizer",
    "get_profile",
    "has_microphone",
    "list_devices",
    "prepare_for_speech",
    "preview_line",
    "resolve_online_voice",
]
