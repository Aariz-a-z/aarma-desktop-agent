"""Microphone capture.

Rebuilt after the first version proved unreliable in real use. The failures and
their fixes:

======================================  ====================================
Failure                                 Fix
======================================  ====================================
Speaking immediately after pressing      No noise calibration in the capture
the button poisoned the noise            path at all. Speech detection is
measurement, so nothing was detected     :mod:`speech.vad`, which judges
again -- the "works once, then stops"    whether audio *sounds like* speech
report                                   rather than whether it is loud
Speech beyond ~1.5 m fell below a        No absolute level floor
fixed level floor
A room with any hum pushed the           Neural detection is unaffected by a
threshold above normal speech            constant noise floor
Every failure burned the full 15 s       Gives up early when no speech has
                                         started
First/last words could be clipped        The whole press-to-stop window is
                                         kept and trimmed by VAD with padding
======================================  ====================================

Audio is held in memory and **never written to disk** -- no temporary WAV, no
cache, no recordings folder.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from speech.vad import SAMPLE_RATE, VadTuning, VoiceActivityDetector, create_detector
from utils.logger import get_logger

logger = get_logger("speech.capture")

BLOCK_SECONDS = 0.032
#: How often the detector re-examines the buffer while recording.
VAD_INTERVAL_SECONDS = 0.25
#: Give up this long after the button if nobody has started speaking.
DEFAULT_NO_SPEECH_TIMEOUT = 6.0
#: Sample rates to try, best first. 16 kHz is what the recogniser wants.
PREFERRED_RATES = (16_000, 48_000, 44_100)


@dataclass(frozen=True)
class MicrophoneDevice:
    """An input device the user could choose.

    ``name`` is the stable identity used in configuration; ``index`` is only
    valid for the current enumeration.
    """

    index: int
    name: str
    channels: int
    sample_rate: float
    is_default: bool = False
    host_api: str = ""

    @property
    def display_name(self) -> str:
        return f"{self.name}{' (default)' if self.is_default else ''}"


@dataclass
class CalibrationReport:
    """What the microphone check found (specification section 6)."""

    ok: bool = False
    noise_rms: float = 0.0
    speech_rms: float = 0.0
    peak: float = 0.0
    clipped: bool = False
    speech_detected: bool = False
    device_name: str = ""
    error: str = ""

    @property
    def noise_label(self) -> str:
        if self.noise_rms < 0.004:
            return "Low"
        if self.noise_rms < 0.015:
            return "Moderate"
        return "High"

    @property
    def speech_label(self) -> str:
        if not self.speech_detected:
            return "Not detected"
        if self.speech_rms < 0.01:
            return "Quiet"
        if self.speech_rms < 0.25:
            return "Good"
        return "Very loud"

    @property
    def microphone_label(self) -> str:
        if self.error:
            return "Problem"
        if self.clipped:
            return "Too loud"
        if not self.speech_detected:
            return "No speech heard"
        return "Good"

    def advice(self) -> str:
        if self.error:
            return self.error
        if self.clipped:
            return "The signal is clipping. Lower the microphone level in Windows."
        if not self.speech_detected:
            return "I didn't hear speech. Check the right device is selected."
        if self.noise_rms >= 0.015:
            return "The room is noisy, which may reduce accuracy."
        if self.speech_rms < 0.01:
            return "Speech is quiet. Move closer or raise the microphone level."
        return "Microphone is working well."


@dataclass
class CaptureResult:
    """Captured speech plus what happened while collecting it."""

    audio: Optional["object"] = None      # numpy.ndarray float32 at 16 kHz
    duration: float = 0.0                 # Trimmed speech length.
    raw_duration: float = 0.0             # Everything recorded.
    peak: float = 0.0
    rms: float = 0.0                      # Whole recording, speech included.
    noise_rms: float = 0.0
    speech_seconds: float = 0.0
    #: Where the detector found speech within the recording. Reported so a
    #: clipped first word or a long trailing pause is visible rather than
    #: inferred. ``None`` when no speech was found at all.
    speech_start_sec: Optional[float] = None
    speech_end_sec: Optional[float] = None
    trailing_silence_sec: Optional[float] = None
    clipped: bool = False
    cancelled: bool = False
    error: str = ""
    error_code: str = ""

    @property
    def ok(self) -> bool:
        return self.audio is not None and not self.cancelled and not self.error

    @property
    def has_speech(self) -> bool:
        return self.speech_seconds > 0.0


def _sounddevice():
    import sounddevice  # noqa: PLC0415 - optional dependency

    return sounddevice


def is_supported() -> bool:
    import importlib.util

    return importlib.util.find_spec("sounddevice") is not None


def list_devices() -> List[MicrophoneDevice]:
    """Every input device, best-effort. Never raises."""
    if not is_supported():
        return []
    try:
        sd = _sounddevice()
        default_index = sd.default.device[0]
        devices: List[MicrophoneDevice] = []
        for index, info in enumerate(sd.query_devices()):
            if info.get("max_input_channels", 0) <= 0:
                continue
            try:
                host_api = str(sd.query_hostapis(info["hostapi"])["name"])
            except Exception:  # noqa: BLE001
                host_api = ""
            devices.append(MicrophoneDevice(
                index=index,
                name=str(info.get("name", f"Device {index}")).strip(),
                channels=int(info["max_input_channels"]),
                sample_rate=float(info.get("default_samplerate", SAMPLE_RATE)),
                is_default=(index == default_index),
                host_api=host_api,
            ))
        return devices
    except Exception:  # noqa: BLE001 - enumeration is inherently flaky
        logger.exception("Could not enumerate input devices")
        return []


def default_device() -> Optional[MicrophoneDevice]:
    devices = list_devices()
    for device in devices:
        if device.is_default:
            return device
    return devices[0] if devices else None


def has_microphone() -> bool:
    return bool(list_devices())


@dataclass(frozen=True)
class DeviceResolution:
    """Which device was chosen, and whether it was the one asked for."""

    device: Optional[MicrophoneDevice]
    #: True when a saved device could not be found and the default was used.
    fell_back: bool = False
    requested_name: str = ""

    @property
    def index(self) -> Optional[int]:
        return self.device.index if self.device else None

    def notice(self) -> str:
        if not self.fell_back:
            return ""
        return (f"The microphone '{self.requested_name}' is unavailable. "
                "I've switched to the Windows default microphone.")


def resolve_device(name: Optional[str]) -> DeviceResolution:
    """Find the saved microphone by name, falling back to the Windows default.

    Devices are stored by **name**, not index: PortAudio indexes are assigned in
    enumeration order and shift when hardware is plugged in, unplugged, or
    simply between reboots. A saved index can therefore silently come to mean a
    different device -- or a speaker -- which is precisely the class of bug that
    leaves an application attached to something that produces no audio.
    """
    devices = list_devices()
    if not devices:
        return DeviceResolution(device=None, requested_name=name or "")

    if not name:
        return DeviceResolution(device=default_device())

    for device in devices:
        if device.name == name:
            return DeviceResolution(device=device)
    # Same hardware can be reported with slight punctuation differences.
    simplified = name.strip().lower()
    for device in devices:
        if device.name.strip().lower() == simplified:
            return DeviceResolution(device=device)
    for device in devices:
        if simplified in device.name.strip().lower():
            return DeviceResolution(device=device)

    logger.info("Saved microphone not found; using the default",
                extra={"context": {"requested": name}})
    return DeviceResolution(device=default_device(), fell_back=True,
                            requested_name=name)


def negotiate_sample_rate(device_index: Optional[int]) -> int:
    """The best rate this device will actually accept.

    Asking for 16 kHz and hoping is how a driver ends up silently resampling
    badly, or refusing outright. Windows device defaults are not necessarily
    what the recogniser wants, so the rate is negotiated rather than assumed.
    """
    if not is_supported():
        return SAMPLE_RATE
    sd = _sounddevice()
    for rate in PREFERRED_RATES:
        try:
            sd.check_input_settings(device=device_index, channels=1,
                                    samplerate=rate, dtype="float32")
            return rate
        except Exception:  # noqa: BLE001 - unsupported combination
            continue
    try:
        info = sd.query_devices(device_index if device_index is not None
                                else sd.default.device[0])
        return int(info.get("default_samplerate", SAMPLE_RATE))
    except Exception:  # noqa: BLE001
        return SAMPLE_RATE


def resample_to_16k(audio, source_rate: int):
    """Downsample to the recogniser's rate, low-pass filtered.

    Plain decimation would alias voices above the new Nyquist limit into the
    speech band. A short windowed-sinc pass keeps the result clean.
    """
    import numpy as np

    if source_rate == SAMPLE_RATE or audio.size == 0:
        return audio.astype("float32")

    ratio = SAMPLE_RATE / float(source_rate)
    if ratio < 1.0:
        cutoff = 0.45 * ratio
        taps = 64
        n = np.arange(taps) - (taps - 1) / 2
        kernel = np.sinc(2 * cutoff * n) * np.hamming(taps)
        kernel /= kernel.sum()
        audio = np.convolve(audio, kernel, mode="same")

    target_length = max(1, int(round(audio.size * ratio)))
    source_positions = np.linspace(0, audio.size - 1, target_length)
    return np.interp(source_positions, np.arange(audio.size),
                     audio).astype("float32")


class MicrophoneStream:
    """An input stream with an explicit, repeatable lifecycle.

    ``open`` -> ``read`` … -> ``close``, reusable any number of times. The
    previous design created a stream inside the recording call and relied on a
    context manager; a cancelled session could leave a capture object that a
    later ``cancel()`` no longer referred to. Owning the lifecycle explicitly
    makes "start, stop, start, stop" twenty times over a testable property.
    """

    def __init__(self, device_index: Optional[int] = None) -> None:
        self._device_index = device_index
        self._stream = None
        self._rate = SAMPLE_RATE
        self._lock = threading.RLock()

    @property
    def sample_rate(self) -> int:
        return self._rate

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._stream is not None

    @property
    def device_index(self) -> Optional[int]:
        return self._device_index

    def open(self) -> None:
        """Open the device. Idempotent."""
        with self._lock:
            if self._stream is not None:
                return
            sd = _sounddevice()
            self._rate = negotiate_sample_rate(self._device_index)
            block = max(1, int(self._rate * BLOCK_SECONDS))
            self._stream = sd.InputStream(
                samplerate=self._rate, channels=1, dtype="float32",
                blocksize=block, device=self._device_index)
            self._stream.start()
            logger.info("Microphone opened", extra={"context": {
                "device": self._device_index, "rate": self._rate}})

    def read_block(self):
        """Read one block. Returns ``(samples, overflowed)``."""
        with self._lock:
            stream = self._stream
        if stream is None:
            raise RuntimeError("Microphone stream is not open")
        frames = max(1, int(self._rate * BLOCK_SECONDS))
        block, overflowed = stream.read(frames)
        return block.reshape(-1), overflowed

    def close(self) -> None:
        """Stop and release the device. Idempotent and safe to call twice."""
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:  # noqa: BLE001 - teardown must not raise
            logger.debug("Stream stop failed")
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            logger.debug("Stream close failed")
        logger.debug("Microphone closed")

    def __enter__(self) -> "MicrophoneStream":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MicrophoneCapture:
    """Records one utterance, with speech boundaries decided by the detector."""

    def __init__(self, *, device_index: Optional[int] = None,
                 detector: Optional[VoiceActivityDetector] = None) -> None:
        self._device_index = device_index
        self._detector = detector or create_detector()
        self._cancel = threading.Event()
        self._recording = threading.Event()
        self._stream: Optional[MicrophoneStream] = None
        self._lock = threading.RLock()

    @property
    def detector(self) -> VoiceActivityDetector:
        return self._detector

    @property
    def is_recording(self) -> bool:
        return self._recording.is_set()

    @property
    def device_index(self) -> Optional[int]:
        return self._device_index

    def cancel(self) -> None:
        """Stop an in-progress recording. Safe from any thread."""
        self._cancel.set()

    def close(self) -> None:
        """Release the device. Called on shutdown; safe when idle."""
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()

    def unavailable_reason(self) -> str:
        if not is_supported():
            return ("The microphone component isn't installed. "
                    "Run: pip install sounddevice")
        if not has_microphone():
            return "No microphone was detected."
        return ""

    def _acquire_stream(self) -> MicrophoneStream:
        """The stream for this device, opened. Reused across recordings."""
        with self._lock:
            if self._stream is None or self._stream.device_index != self._device_index:
                if self._stream is not None:
                    self._stream.close()
                self._stream = MicrophoneStream(self._device_index)
            self._stream.open()
            return self._stream

    # ------------------------------------------------------------- recording

    def record(
        self,
        *,
        max_seconds: float = 15.0,
        silence_timeout: float = 1.2,
        no_speech_timeout: float = DEFAULT_NO_SPEECH_TIMEOUT,
        min_speech_seconds: float = 0.25,
        tuning: Optional[VadTuning] = None,
        on_level: Optional[Callable[[float], None]] = None,
        on_speech_started: Optional[Callable[[], None]] = None,
    ) -> CaptureResult:
        """Capture one utterance. Blocks; call from a worker thread.

        Everything from the moment of pressing is kept, and the speech is cut
        out afterwards by the detector with padding. That is what guarantees the
        first word survives even when the user starts talking instantly -- the
        precise failure that made the previous build unusable.
        """
        reason = self.unavailable_reason()
        if reason:
            return CaptureResult(error=reason, error_code="no_microphone")

        try:
            import numpy as np
        except ImportError:
            return CaptureResult(error="The audio components aren't installed.",
                                 error_code="no_backend")

        self._cancel.clear()
        self._recording.set()
        options = tuning or VadTuning()

        chunks: List[object] = []
        started = time.monotonic()
        last_vad = 0.0
        speech_started = False
        last_speech_end_time = 0.0
        peak = 0.0
        clipped = False
        stream_rate = SAMPLE_RATE

        try:
            stream = self._acquire_stream()
            stream_rate = stream.sample_rate

            while True:
                if self._cancel.is_set():
                    return CaptureResult(cancelled=True,
                                         raw_duration=time.monotonic() - started)

                block, overflowed = stream.read_block()
                if overflowed:
                    logger.debug("Input overflow")
                chunks.append(block.copy())

                block_peak = float(np.max(np.abs(block))) if block.size else 0.0
                peak = max(peak, block_peak)
                if block_peak >= 0.99:
                    clipped = True
                if on_level is not None:
                    try:
                        on_level(float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0)
                    except Exception:  # noqa: BLE001 - a meter must not stop capture
                        pass

                elapsed = time.monotonic() - started

                # Ask the detector periodically rather than per block: it is
                # cheap but not free, and 250 ms is finer than human turn-taking.
                if elapsed - last_vad >= VAD_INTERVAL_SECONDS:
                    last_vad = elapsed
                    recent = np.concatenate(chunks)
                    if stream_rate != SAMPLE_RATE:
                        recent = resample_to_16k(recent, stream_rate)
                    end = self._detector.last_speech_end(recent, options)
                    if end is not None:
                        if not speech_started:
                            speech_started = True
                            logger.debug("Speech detected")
                            if on_speech_started is not None:
                                try:
                                    on_speech_started()
                                except Exception:  # noqa: BLE001
                                    pass
                        last_speech_end_time = end / SAMPLE_RATE

                    if speech_started and (elapsed - last_speech_end_time) >= silence_timeout:
                        break
                    # Nobody started talking: stop early instead of making the
                    # user wait out the full limit, which felt like a hang.
                    if not speech_started and elapsed >= no_speech_timeout:
                        break

                if elapsed >= max_seconds:
                    logger.info("Recording hit the time limit")
                    break
        except Exception as exc:  # noqa: BLE001 - audio stacks fail in many ways
            logger.error("Microphone capture failed",
                         extra={"context": {"error": type(exc).__name__}})
            self.close()          # A faulted device must not be reused as-is.
            return CaptureResult(error="I couldn't record from the microphone.",
                                 error_code="capture_failed")
        finally:
            self._recording.clear()

        import numpy as np

        if not chunks:
            return CaptureResult(error="No audio was captured.",
                                 error_code="empty_audio")

        raw = np.concatenate(chunks).astype("float32")
        if stream_rate != SAMPLE_RATE:
            raw = resample_to_16k(raw, stream_rate)
        raw_duration = raw.size / float(SAMPLE_RATE)
        noise_rms = _quiet_level(raw)
        overall_rms = float(np.sqrt(np.mean(raw ** 2))) if raw.size else 0.0

        # Ask the detector for the regions rather than the trimmed buffer, so
        # *where* it found speech can be reported and not only how much. The
        # slice below is byte-identical to what ``trim()`` returns -- this
        # records the boundaries the detector already decided, and changes no
        # threshold and no outcome.
        regions = self._detector.speech_regions(raw, options)
        speech = raw[regions[0].start:regions[-1].end] if regions else None
        speech_seconds = (speech.size / SAMPLE_RATE) if speech is not None else 0.0
        start_sec = (regions[0].start / SAMPLE_RATE) if regions else None
        end_sec = (regions[-1].end / SAMPLE_RATE) if regions else None
        trailing = (raw_duration - end_sec) if end_sec is not None else None

        if speech is None or speech_seconds < min_speech_seconds:
            return CaptureResult(
                audio=None, duration=0.0, raw_duration=raw_duration, peak=peak,
                rms=overall_rms, noise_rms=noise_rms,
                speech_seconds=speech_seconds, clipped=clipped,
                speech_start_sec=start_sec, speech_end_sec=end_sec,
                trailing_silence_sec=trailing,
                error="I didn't hear anything.", error_code="no_speech")

        return CaptureResult(
            audio=speech, duration=speech_seconds, raw_duration=raw_duration,
            peak=peak, rms=overall_rms, noise_rms=noise_rms,
            speech_seconds=speech_seconds, clipped=clipped,
            speech_start_sec=start_sec, speech_end_sec=end_sec,
            trailing_silence_sec=trailing)

    # ----------------------------------------------------------- calibration

    def calibrate(self, *, quiet_seconds: float = 2.0,
                  speech_seconds: float = 4.0,
                  on_level: Optional[Callable[[float], None]] = None,
                  on_phase: Optional[Callable[[str], None]] = None,
                  tuning: Optional[VadTuning] = None) -> CalibrationReport:
        """Measure the room and the speaker (specification section 6).

        Reports what it found and never adjusts anything on its own: blind
        amplification turns a quiet microphone into a loud noisy one.
        """
        reason = self.unavailable_reason()
        if reason:
            return CalibrationReport(error=reason)

        import numpy as np

        device = None
        for candidate in list_devices():
            if candidate.index == self._device_index or (
                    self._device_index is None and candidate.is_default):
                device = candidate
                break

        self._cancel.clear()
        try:
            stream = self._acquire_stream()
            rate = stream.sample_rate

            def gather(seconds: float) -> object:
                collected: List[object] = []
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    if self._cancel.is_set():
                        break
                    block, _ = stream.read_block()
                    collected.append(block.copy())
                    if on_level is not None:
                        try:
                            on_level(float(np.sqrt(np.mean(block ** 2))))
                        except Exception:  # noqa: BLE001
                            pass
                if not collected:
                    return np.zeros(0, dtype="float32")
                audio = np.concatenate(collected).astype("float32")
                return resample_to_16k(audio, rate) if rate != SAMPLE_RATE else audio

            if on_phase is not None:
                on_phase("quiet")
            quiet = gather(quiet_seconds)

            if on_phase is not None:
                on_phase("speak")
            spoken = gather(speech_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.error("Calibration failed",
                         extra={"context": {"error": type(exc).__name__}})
            self.close()
            return CalibrationReport(error="I couldn't read from the microphone.")

        noise_rms = float(np.sqrt(np.mean(quiet ** 2))) if quiet.size else 0.0
        peak = float(np.max(np.abs(spoken))) if spoken.size else 0.0
        regions = self._detector.speech_regions(spoken, tuning or VadTuning())
        speech_detected = bool(regions)
        if speech_detected:
            segment = spoken[regions[0].start:regions[-1].end]
            speech_rms = float(np.sqrt(np.mean(segment ** 2))) if segment.size else 0.0
        else:
            speech_rms = 0.0

        report = CalibrationReport(
            ok=speech_detected and peak < 0.99,
            noise_rms=noise_rms, speech_rms=speech_rms, peak=peak,
            clipped=peak >= 0.99, speech_detected=speech_detected,
            device_name=device.name if device else "")
        logger.info("Microphone calibrated", extra={"context": {
            "noise_rms": round(noise_rms, 5), "speech_rms": round(speech_rms, 5),
            "peak": round(peak, 4), "speech_detected": speech_detected}})
        return report


def probe_device(device: MicrophoneDevice, *, seconds: float = 1.2,
                 detector: Optional[VoiceActivityDetector] = None):
    """Open one device, listen briefly, and measure what actually arrived.

    This is the measurement that separates "no signal" from "no speech" -- the
    distinction the original single message could not make.
    """
    from speech.diagnostics import DeviceProbe, classify

    probe = DeviceProbe(
        index=device.index, name=device.name, host_api=device.host_api,
        channels=device.channels, device_rate=device.sample_rate,
        is_default=device.is_default)

    try:
        import numpy as np
    except ImportError:
        probe.error = "Audio components are not installed."
        return probe

    stream = MicrophoneStream(device.index)
    try:
        stream.open()
        probe.capture_rate = stream.sample_rate
        collected = []
        frames = 0
        # Some backends (DirectSound here) return immediately with silence
        # instead of blocking for the block duration. Without a cap the loop
        # spins and collects tens of millions of samples in under a second,
        # which wastes memory and reports a nonsense frame count.
        frame_limit = int(stream.sample_rate * (seconds + 0.5))
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and frames < frame_limit:
            block, _overflow = stream.read_block()
            collected.append(block.copy())
            frames += int(block.size)
    except Exception as exc:  # noqa: BLE001 - probing must never raise
        probe.error = f"{type(exc).__name__}: {str(exc)[:120]}"
        logger.info("Device probe failed", extra={"context": {
            "device": device.name, "error": type(exc).__name__}})
        return probe
    finally:
        stream.close()

    if not collected:
        probe.error = "No audio blocks were returned."
        return probe

    audio = np.concatenate(collected).astype("float32")
    if probe.capture_rate != SAMPLE_RATE:
        audio = resample_to_16k(audio, probe.capture_rate)

    probe.frames = int(audio.size)
    probe.rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    probe.peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    probe.minimum = float(np.min(audio)) if audio.size else 0.0
    probe.maximum = float(np.max(audio)) if audio.size else 0.0
    probe.nonzero = int(np.count_nonzero(audio))

    vad = detector or create_detector()
    probe.speech_detected = bool(vad.has_speech(audio))
    probe.status = classify(frames=probe.frames, peak=probe.peak,
                            nonzero=probe.nonzero, speech=probe.speech_detected)
    return probe


def diagnose(selected_name: Optional[str] = None, *, seconds: float = 1.2,
             expect_speech: bool = False,
             on_device=None):
    """Probe every input device and report the whole picture.

    ``expect_speech`` records whether the user was asked to speak, so a quiet
    reading is only treated as a fault when sound was actually expected.
    """
    from speech.diagnostics import MicrophoneDiagnosis, check_windows_permission

    permission = check_windows_permission()
    devices = list_devices()
    if not devices:
        return MicrophoneDiagnosis(permission=permission,
                                   error="No input devices were found.",
                                   expected_speech=expect_speech)

    resolution = resolve_device(selected_name)
    chosen_index = resolution.index
    detector = create_detector()

    probes = []
    for device in devices:
        if on_device is not None:
            try:
                on_device(device.name)
            except Exception:  # noqa: BLE001
                pass
        probe = probe_device(device, seconds=seconds, detector=detector)
        probe.is_selected = (device.index == chosen_index)
        probes.append(probe)

    selected = next((p for p in probes if p.is_selected), None)
    return MicrophoneDiagnosis(permission=permission, probes=probes,
                               selected=selected, expected_speech=expect_speech)


def _quiet_level(audio) -> float:
    """RMS of the quietest tenth of the buffer -- an estimate of the room."""
    import numpy as np

    if audio.size == 0:
        return 0.0
    window = max(1, int(SAMPLE_RATE * 0.03))
    usable = (audio.size // window) * window
    if usable == 0:
        return float(np.sqrt(np.mean(audio ** 2)))
    frames = audio[:usable].reshape(-1, window)
    levels = np.sqrt(np.mean(frames ** 2, axis=1))
    return float(np.percentile(levels, 10))
