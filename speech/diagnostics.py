"""Microphone diagnostics: is audio actually reaching AARMA, and if not, why?

Written after a user reported "No speech heard, 0.0000 RMS" while their
microphone worked in other applications. That message conflated at least six
different problems, and pointed at the wrong one — it blamed speech recognition
when the question was whether any signal existed at all.

Every distinct failure now has its own verdict:

============================  ====================================================
Verdict                       Meaning
============================  ====================================================
``NO_DEVICE``                 Nothing to record from
``PERMISSION_DENIED``         Windows is blocking desktop apps from the microphone
``OPEN_FAILED``               The device exists but the stream would not start
``FORMAT_UNSUPPORTED``        The device rejects every format AARMA can use
``SILENT_SAMPLES``            Stream runs, every sample is exactly zero
``NEAR_SILENT``               Stream runs, signal is at the noise floor: muted,
                              level at zero, or the wrong input
``SIGNAL_NO_SPEECH``          Real audio arriving, but no speech in it
``SPEECH_DETECTED``           Working
============================  ====================================================

Two rules this module exists to enforce:

* **Never report a recognition failure when there is no signal.** They have
  different causes and different fixes.
* **Never print a level as ``0.0000``.** Four decimal places renders a real
  measurement of 0.000017 as zero, which is exactly what made a working-but-quiet
  stream look like a dead one. :func:`format_level` keeps small values legible.

Nothing here changes any Windows setting. It measures and explains.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger("speech.diagnostics")

#: Below this peak, a stream is carrying noise floor rather than a usable signal.
NEAR_SILENT_PEAK = 0.002
#: Below this, samples are non-zero but meaninglessly small.
NOISE_FLOOR_PEAK = 0.0005


class CaptureStatus(str, Enum):
    """What a capture attempt actually found."""

    NO_DEVICE = "no_device"
    PERMISSION_DENIED = "permission_denied"
    OPEN_FAILED = "open_failed"
    FORMAT_UNSUPPORTED = "format_unsupported"
    SILENT_SAMPLES = "silent_samples"
    NEAR_SILENT = "near_silent"
    SIGNAL_NO_SPEECH = "signal_no_speech"
    SPEECH_DETECTED = "speech_detected"

    @property
    def ok(self) -> bool:
        return self in (CaptureStatus.SPEECH_DETECTED, CaptureStatus.SIGNAL_NO_SPEECH)

    @property
    def label(self) -> str:
        return {
            CaptureStatus.NO_DEVICE: "No microphone found",
            CaptureStatus.PERMISSION_DENIED: "Blocked by Windows",
            CaptureStatus.OPEN_FAILED: "Could not open",
            CaptureStatus.FORMAT_UNSUPPORTED: "Format not supported",
            CaptureStatus.SILENT_SAMPLES: "No audio signal",
            CaptureStatus.NEAR_SILENT: "Almost no signal",
            CaptureStatus.SIGNAL_NO_SPEECH: "Audio received",
            CaptureStatus.SPEECH_DETECTED: "Working",
        }[self]

    @property
    def symbol(self) -> str:
        if self.ok:
            return "✓"          # check mark
        if self in (CaptureStatus.NEAR_SILENT, CaptureStatus.SIGNAL_NO_SPEECH):
            return "⚠"          # warning sign
        return "✗"              # ballot X


def format_level(value: float) -> str:
    """Render an audio level without collapsing small values to "0.0000".

    A microphone reading 0.000017 RMS is *working and very quiet*, which is a
    completely different problem from one reading exactly zero. Four decimals
    hid that distinction and sent the diagnosis in the wrong direction.
    """
    if value <= 0.0:
        return "0 (exactly)"
    if value >= 0.001:
        return f"{value:.4f}"
    return f"{value:.2e}"


# ------------------------------------------------------------ windows consent

@dataclass(frozen=True)
class MicrophonePermission:
    """What Windows says about microphone access for desktop applications."""

    allowed: bool
    checked: bool = True
    global_value: str = ""
    desktop_value: str = ""
    detail: str = ""

    @property
    def label(self) -> str:
        if not self.checked:
            return "Not checked"
        return "Allowed" if self.allowed else "Blocked by Windows"

    def advice(self) -> str:
        if self.allowed or not self.checked:
            return ""
        return ("Open Windows Settings → Privacy & security → Microphone and turn on "
                "both 'Microphone access' and 'Let desktop apps access your microphone'.")


_CONSENT_PATH = (r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager"
                 r"\ConsentStore\microphone")


def check_windows_permission() -> MicrophonePermission:
    """Read the Windows consent store. Read-only; changes nothing.

    Windows does not fail a capture when desktop-app access is off — it hands
    the application a stream of silence. Without checking this, that is
    indistinguishable from a muted microphone, so it must be reported
    separately.
    """
    if sys.platform != "win32":
        return MicrophonePermission(allowed=True, checked=False,
                                    detail="Not applicable on this platform.")
    try:
        import winreg
    except ImportError:  # pragma: no cover - Windows only
        return MicrophonePermission(allowed=True, checked=False)

    def read(hive, path: str) -> str:
        try:
            with winreg.OpenKey(hive, path) as key:
                return str(winreg.QueryValueEx(key, "Value")[0])
        except OSError:
            return ""

    global_value = read(winreg.HKEY_CURRENT_USER, _CONSENT_PATH)
    desktop_value = read(winreg.HKEY_CURRENT_USER, _CONSENT_PATH + r"\NonPackaged")

    # An absent key means Windows has not been asked and the default applies,
    # which is permissive. Only an explicit Deny is treated as blocked.
    blocked = "Deny" in (global_value, desktop_value)
    return MicrophonePermission(
        allowed=not blocked,
        global_value=global_value or "(unset)",
        desktop_value=desktop_value or "(unset)",
        detail=("Windows is blocking desktop applications from the microphone."
                if blocked else ""))


# --------------------------------------------------------------- device probe

@dataclass
class DeviceProbe:
    """The result of opening one device and listening to it briefly."""

    index: int
    name: str
    host_api: str = ""
    channels: int = 0
    device_rate: float = 0.0
    capture_rate: int = 0
    is_default: bool = False
    is_selected: bool = False

    frames: int = 0
    rms: float = 0.0
    peak: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    nonzero: int = 0
    speech_detected: bool = False
    status: CaptureStatus = CaptureStatus.OPEN_FAILED
    error: str = ""

    @property
    def display_name(self) -> str:
        suffix = " (Windows default)" if self.is_default else ""
        return f"{self.name}{suffix}"

    @property
    def summary(self) -> str:
        if self.status is CaptureStatus.OPEN_FAILED:
            return self.error or "The device would not start."
        return (f"{self.frames} frames · RMS {format_level(self.rms)} · "
                f"peak {format_level(self.peak)}")

    def explain(self) -> str:
        """One sentence a person can act on."""
        return {
            CaptureStatus.NO_DEVICE: "No input device was found.",
            CaptureStatus.PERMISSION_DENIED:
                "Windows is blocking desktop apps from the microphone.",
            CaptureStatus.OPEN_FAILED:
                self.error or "This device could not be opened.",
            CaptureStatus.FORMAT_UNSUPPORTED:
                "This device does not accept a format AARMA can record.",
            CaptureStatus.SILENT_SAMPLES:
                "The device opened but returned nothing but zeros — it is muted, "
                "disabled, or held by another application.",
            CaptureStatus.NEAR_SILENT:
                "The device opened but the signal is at the noise floor. Its input "
                "level may be at zero, or this may be the wrong input.",
            CaptureStatus.SIGNAL_NO_SPEECH:
                "Audio is reaching AARMA, but no speech was detected in it.",
            CaptureStatus.SPEECH_DETECTED: "Audio and speech both detected.",
        }[self.status]


@dataclass
class MicrophoneDiagnosis:
    """The whole picture: permission, devices, and what AARMA is using."""

    permission: MicrophonePermission
    probes: List[DeviceProbe] = field(default_factory=list)
    selected: Optional[DeviceProbe] = None
    error: str = ""
    #: True when the user was asked to speak during the probe. A quiet reading
    #: is a fault only if we actually asked for sound; otherwise it just means
    #: the room was quiet, and calling that "broken" would be a false alarm.
    expected_speech: bool = False

    @property
    def usable(self) -> List[DeviceProbe]:
        """Devices that could be opened and returned non-zero samples."""
        return [p for p in self.probes
                if p.status not in (CaptureStatus.OPEN_FAILED,
                                    CaptureStatus.FORMAT_UNSUPPORTED,
                                    CaptureStatus.SILENT_SAMPLES)]

    @property
    def working(self) -> List[DeviceProbe]:
        """Devices that produced a real signal, loudest first."""
        return sorted((p for p in self.probes if p.status.ok),
                      key=lambda p: p.peak, reverse=True)

    @property
    def loudest(self) -> Optional[DeviceProbe]:
        candidates = sorted(self.usable, key=lambda p: p.peak, reverse=True)
        return candidates[0] if candidates else None

    @property
    def status(self) -> CaptureStatus:
        if self.selected is not None:
            return self.selected.status
        if not self.probes:
            return CaptureStatus.NO_DEVICE
        return CaptureStatus.SILENT_SAMPLES

    def recommendation(self) -> str:
        """What to do next, based only on what was actually measured."""
        if not self.permission.allowed:
            return self.permission.advice()
        if not self.probes:
            return "No input devices were found. Check that a microphone is connected."

        if self.status.ok:
            return "Microphone is working."

        if self.status is CaptureStatus.OPEN_FAILED:
            best = self.loudest
            if best is not None:
                return (f"The selected device could not be opened. "
                        f"'{best.name}' works — try selecting it above.")
            return "The selected device could not be opened."

        if self.status is CaptureStatus.SILENT_SAMPLES:
            best = self.loudest
            if best is not None:
                return (f"The selected device returns only silence. "
                        f"'{best.name}' is returning audio — try selecting it above.")
            return ("Every device returned silence. Check that the microphone is not "
                    "muted in Windows (Settings → System → Sound → Input).")

        # Quiet reading. Whether that is a fault depends on whether we asked
        # for sound.
        if not self.expected_speech:
            return ("Audio is arriving, but the room was quiet during the check. "
                    "Run the microphone check and speak to confirm the level.")

        working = self.working
        if working:
            best = working[0]
            return (f"AARMA heard almost nothing on the selected device, but "
                    f"'{best.name}' picked up sound. Try selecting it above.")
        return ("Audio is reaching AARMA, but at almost zero level. In Windows, open "
                "Settings → System → Sound → Input, choose this microphone, and check "
                "that its input volume is well above zero and not muted.")


@dataclass
class RecognitionDiagnostic:
    """Everything that happened during one listening attempt.

    Kept because "I couldn't understand that clearly" is the right thing to say
    to a person and the wrong thing to leave in a log. When recognition is
    refused, this records *which* of the eight possible reasons applied, what
    the recogniser actually returned before validation touched it, and the
    numbers behind the decision.
    """

    # Capture
    device: str = ""
    capture_seconds: float = 0.0        # Everything recorded.
    speech_seconds: float = 0.0         # What survived trimming.
    speech_start_sec: Optional[float] = None
    speech_end_sec: Optional[float] = None
    trailing_silence_sec: Optional[float] = None
    rms: float = 0.0
    peak: float = 0.0
    noise_rms: float = 0.0
    clipped: bool = False
    speech_detected: bool = False

    # Recognition
    model: str = ""
    language: str = ""
    raw_text: str = ""                  # Before validation. The key field.
    text: str = ""                      # What was accepted, if anything.
    confidence: Optional[float] = None
    average_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    threshold_used: Optional[float] = None
    detected_language: Optional[str] = None

    # Outcome
    accepted: bool = False
    reason: str = ""                    # Machine-readable rejection code.
    detail: str = ""                    # One line for a developer.
    latency: float = 0.0
    audio_path: str = ""                # Only when debug saving is enabled.

    @property
    def summary(self) -> str:
        if self.accepted:
            return f"accepted {self.text!r} (confidence {self.confidence_label})"
        return f"rejected: {self.reason or 'unknown'}"

    @property
    def confidence_label(self) -> str:
        return "n/a" if self.confidence is None else f"{self.confidence:.3f}"

    def report(self) -> str:
        """A block a developer or a user filing a bug can read."""
        lines = [
            f"Device        : {self.device or '-'}",
            f"Capture       : {self.capture_seconds:.2f}s recorded, "
            f"{self.speech_seconds:.2f}s speech",
            f"Levels        : RMS {format_level(self.rms)}, "
            f"peak {format_level(self.peak)}, noise {format_level(self.noise_rms)}"
            + (", CLIPPED" if self.clipped else ""),
            f"Voice detected: {'yes' if self.speech_detected else 'no'}",
            f"Model         : {self.model or '-'} ({self.language or '-'})",
            f"Raw STT       : {self.raw_text!r}",
            f"Confidence    : {self.confidence_label}"
            + (f"  (avg_logprob {self.average_logprob:.3f})"
               if self.average_logprob is not None else "")
            + (f"  no_speech {self.no_speech_prob:.3f}"
               if self.no_speech_prob is not None else "")
            + (f"  threshold {self.threshold_used:.3f}"
               if self.threshold_used is not None else ""),
            f"Decision      : {'ACCEPTED' if self.accepted else 'REJECTED'}"
            + (f" — {self.reason}" if self.reason else ""),
        ]
        if self.detail:
            lines.append(f"Detail        : {self.detail}")
        lines.append(f"Latency       : {self.latency:.2f}s")
        if self.audio_path:
            lines.append(f"Saved audio   : {self.audio_path}")
        return "\n".join(lines)


class DiagnosticLog:
    """A small ring buffer of recent recognition attempts."""

    def __init__(self, limit: int = 20) -> None:
        self._limit = limit
        self._entries: List[RecognitionDiagnostic] = []

    def add(self, entry: RecognitionDiagnostic) -> None:
        self._entries.append(entry)
        del self._entries[:-self._limit]

    @property
    def last(self) -> Optional[RecognitionDiagnostic]:
        return self._entries[-1] if self._entries else None

    def recent(self, count: int = 10) -> List[RecognitionDiagnostic]:
        return list(reversed(self._entries[-count:]))

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def classify(*, frames: int, peak: float, nonzero: int,
             speech: bool) -> CaptureStatus:
    """Turn raw measurements into a verdict."""
    if frames <= 0:
        return CaptureStatus.OPEN_FAILED
    if nonzero == 0:
        return CaptureStatus.SILENT_SAMPLES
    if peak < NOISE_FLOOR_PEAK:
        return CaptureStatus.NEAR_SILENT
    if speech:
        return CaptureStatus.SPEECH_DETECTED
    if peak < NEAR_SILENT_PEAK:
        return CaptureStatus.NEAR_SILENT
    return CaptureStatus.SIGNAL_NO_SPEECH
