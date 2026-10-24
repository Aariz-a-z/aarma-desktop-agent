"""The live Voice Command Benchmark.

Twenty known commands, spoken once each by a real person, driven through the
real pipeline as far as intent and action *resolution* — and stopped there.

This is a diagnostic screen alongside the microphone check and the speech test,
built on the same dialog pattern and launched from the same Settings section.
There is no second interface stack.

**Nothing runs.** The dispatch call carries ``dry_run=True``, the engine returns
at its resolution step, and the one place in AARMA that invokes a handler raises
:class:`~core.engine.BenchmarkExecutionError` if it is ever reached with that
flag set. The export refuses to be written unless every record carries
``executed: false``. Three locks, because one stray path would invalidate the
whole run.

**Capture is click-to-start with voice-activity endpointing**, not push-to-talk.
That is precisely what the microphone button does in ordinary use, and a
benchmark that measured a different capture mode would be measuring something
the user never does.

**Every attempt is kept.** A retry does not overwrite; it appends with the next
attempt number, and scoring uses the final one. Needing three tries is a
measurement, not a mess to tidy away.
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from benchmark.harness import META_INTENTS, BenchmarkSuite, ExpectedCommand
from benchmark.runner import STATUS_OK, STATUS_SKIPPED, BenchmarkRun, find_incomplete
from speech.diagnostics import format_level
from ui.responsive import Metrics, apply_initial_geometry, flow_container, scroll_wrap
from ui.theme import Palette, build_stylesheet, get_palette
from ui.widgets import divider, muted, section_title
from utils.logger import get_logger

logger = get_logger("ui.voice_benchmark")

IDEAL_SIZE = (860, 820)
MINIMUM_SIZE = (520, 440)

#: Hard ceiling on one recording. The production no-speech timeout is shorter,
#: so this only fires if capture genuinely hangs -- it is a watchdog, not a
#: setting, and it changes no configured value.
RECORDING_WATCHDOG_SECONDS = 10.0

#: The mic check must clear this before the run may start. Below it, twenty
#: commands would be wasted on a device that is not hearing anyone.
MIC_CHECK_SECONDS = 3.0
MIC_CHECK_MIN_PEAK = 0.01


class _CheckWorker(QObject):
    """Three-second "say anything" microphone check."""

    finished = Signal(object)

    def __init__(self, listening) -> None:
        super().__init__()
        self._listening = listening

    @Slot()
    def run(self) -> None:
        try:
            probe = self._listening.diagnose(seconds=MIC_CHECK_SECONDS)
        except Exception:  # noqa: BLE001 - a check must not crash the dialog
            logger.exception("Benchmark microphone check failed")
            probe = None
        self.finished.emit(probe)


class _CommandWorker(QObject):
    """Records one command and resolves it. Never executes."""

    completed = Signal(object)      # dict: the trace record.

    def __init__(self, engine, suite: BenchmarkSuite,
                 command: ExpectedCommand, attempt: int, run: BenchmarkRun) -> None:
        super().__init__()
        self._engine = engine
        self._suite = suite
        self._command = command
        self._attempt = attempt
        self._run = run

    @Slot()
    def run(self) -> None:
        record = build_record(self._engine, self._suite, self._command,
                              self._attempt, self._run)
        self.completed.emit(record)


def build_record(engine, suite: BenchmarkSuite, command: ExpectedCommand,
                 attempt: int, run: BenchmarkRun) -> Dict[str, Any]:
    """Capture, recognise, resolve, and describe — as one trace record.

    Every stage writes its field even when a later one fails, and a stage that
    never ran is ``None`` rather than absent. If anything throws, the exception
    is recorded and the record is still returned: one crash must not end a
    twenty-command run.
    """
    from utils.trace import ActionStage, AudioStage, PipelineTrace, SttStage, VadStage

    listening = engine.listening
    trace = PipelineTrace(
        mode="benchmark",
        expected_text=command.text,
        expected_intent=command.intent,
        expected_parameters=command.parameters,
    )
    audio_path = run.audio_path(command.id, attempt)

    try:
        result, diagnostic = listening.listen_once(audio_path=audio_path)
    except Exception as exc:  # noqa: BLE001 - keep the run alive
        logger.exception("Benchmark capture failed")
        trace.audio = AudioStage(sample_rate=16_000)
        trace.vad = VadStage(speech_detected=False)
        trace.finalise()
        trace.failure_layer = "VAD"
        trace.final_result = "FAILED"
        trace.failure_detail = f"{type(exc).__name__}: {exc}"
        return _as_record(trace, command, attempt, exception=traceback.format_exc())

    trace.audio = AudioStage(
        duration_sec=round(diagnostic.capture_seconds, 3),
        sample_rate=16_000,
        peak_amplitude=round(diagnostic.peak, 6),
        wav_path=diagnostic.audio_path or None,
    )
    trace.vad = VadStage(
        speech_detected=diagnostic.speech_detected,
        speech_start_sec=(round(diagnostic.speech_start_sec, 3)
                          if diagnostic.speech_start_sec is not None else None),
        speech_end_sec=(round(diagnostic.speech_end_sec, 3)
                        if diagnostic.speech_end_sec is not None else None),
        trailing_silence_sec=(round(diagnostic.trailing_silence_sec, 3)
                              if diagnostic.trailing_silence_sec is not None
                              else None),
    )
    trace.stt = SttStage(
        engine=listening.recognizer.name,
        model=getattr(listening.recognizer, "model_size", ""),
        raw_text=diagnostic.raw_text,          # Verbatim. Never the normalised form.
        confidence=diagnostic.confidence,
        alternatives=[],                       # beam_size is 1; recorded as fact.
        latency_ms=int(diagnostic.latency * 1000),
    )
    trace.validation = _validation_stage(engine, diagnostic, result)

    # Intent and action only run when a transcription survived. Otherwise these
    # stages genuinely did not happen and stay null.
    if result.ok and result.text.strip():
        trace.intent, trace.normalization, trace.action = _resolve(engine, result.text)
    else:
        trace.action = ActionStage(executed=False)

    trace.finalise()
    return _as_record(trace, command, attempt)


def _validation_stage(engine, diagnostic, result):
    """The gate as it actually resolved, not the configured constant."""
    from utils.trace import ValidationStage

    base = engine.config.config.speech_input.confidence_threshold
    words = len((diagnostic.raw_text or "").split())
    rejected = not result.ok and diagnostic.reason not in ("", "no_speech",
                                                           "cancelled")
    passed = result.ok
    return ValidationStage(
        result="PASS" if passed else ("REJECT" if rejected else None),
        raw_confidence=diagnostic.confidence,
        gate_base=base,
        gate_effective=diagnostic.threshold_used,
        length_scale_input={
            "duration_sec": round(diagnostic.speech_seconds, 3),
            "token_count": words,
        },
        reason=(diagnostic.detail or diagnostic.reason or
                (f"confidence {diagnostic.confidence:.3f} >= gate "
                 f"{diagnostic.threshold_used:.3f}"
                 if passed and diagnostic.confidence is not None
                 and diagnostic.threshold_used is not None else "accepted")),
    )


def _resolve(engine, text: str):
    """Parse and dispatch with ``dry_run=True``. Nothing runs."""
    from utils.trace import ActionStage, IntentStage, NormalizationStage

    intent = engine.intent_engine.parse(text)
    normalization = NormalizationStage(normalized_text=intent.normalised_text)
    intent_stage = IntentStage(
        name=intent.name,
        parameters=dict(intent.parameters),
        match_method=intent.match_method,
        match_score=intent.confidence,
    )

    if intent.name in META_INTENTS:
        # Answered by the planner; no registered action is involved, and that
        # is by design rather than a dispatch failure.
        return intent_stage, normalization, ActionStage(
            selected=intent.name, risk_level=None,
            would_require_confirmation=False, executed=False)

    response = engine.handle_text(text, source="benchmark", dry_run=True)
    resolution = (response.data or {}).get("resolution") or {}
    return intent_stage, normalization, ActionStage(
        selected=resolution.get("selected") or response.action,
        risk_level=resolution.get("risk_level"),
        would_require_confirmation=bool(
            resolution.get("would_require_confirmation", False)),
        executed=bool(resolution.get("executed", False)),
    )


def _as_record(trace, command: ExpectedCommand, attempt: int,
               exception: str = "") -> Dict[str, Any]:
    record = trace.to_dict()
    record.update({
        "command_id": command.id,
        "attempt": attempt,
        "status": STATUS_OK,
        "excluded": not command.resolvable,
        "mapping_note": command.mapping_note,
        "spec_intent": command.spec_intent,
        "stt_correct": trace.stt_correct,
        "stt_exact": trace.stt_exact,
        "exception": exception or None,
    })
    return record


def skipped_record(command: ExpectedCommand, attempt: int) -> Dict[str, Any]:
    from utils.trace import ActionStage, PipelineTrace

    trace = PipelineTrace(
        mode="benchmark", expected_text=command.text,
        expected_intent=command.intent,
        expected_parameters=command.parameters,
        action=ActionStage(executed=False),
    )
    trace.final_result = "FAILED"
    trace.failure_layer = "NONE"
    record = _as_record(trace, command, attempt)
    record["status"] = STATUS_SKIPPED
    return record


class VoiceBenchmarkDialog(QDialog):
    """Pre-flight, then twenty commands, then an export."""

    level_measured = Signal(float)

    def __init__(self, engine, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.engine = engine
        self.suite = BenchmarkSuite()
        self.commands: List[ExpectedCommand] = self.suite.commands()
        self.run = BenchmarkRun()
        self.metrics = Metrics.current()
        self.palette_tokens: Palette = get_palette(engine.config.config.runtime.theme)

        self._index = 0
        self._attempt = 1
        self._paused = False
        self._thread: Optional[QThread] = None
        self._worker: Optional[QObject] = None
        self._mic_ok = False
        self._watchdog = QTimer(self)
        self._watchdog.setSingleShot(True)
        self._watchdog.timeout.connect(self._on_watchdog)

        m = self.metrics
        self.setWindowTitle("Voice command benchmark")
        self.setObjectName("Root")
        if not self.styleSheet():
            self.setStyleSheet(build_stylesheet(self.palette_tokens, m))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(m.sp(24), m.sp(22), m.sp(24), m.sp(18))
        outer.setSpacing(m.sp(12))

        title = QLabel("Voice command benchmark")
        title.setObjectName("PageTitle")
        outer.addWidget(title)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_preflight())
        self.stack.addWidget(self._build_run_page())
        self.stack.addWidget(self._build_summary_page())
        outer.addWidget(self.stack, 1)

        self.level_measured.connect(self._on_level)
        self._level_emitter = self.level_measured.emit
        self.engine.listening.on_level(self._level_emitter)

        apply_initial_geometry(self, ideal=IDEAL_SIZE, minimum=MINIMUM_SIZE)
        self._offer_resume()

    # ------------------------------------------------------------- pre-flight

    def _build_preflight(self) -> QWidget:
        m = self.metrics
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(m.sp(10))

        layout.addWidget(section_title("Before you start"))
        layout.addWidget(muted(
            "You'll be asked to say 20 short commands, one at a time. Nothing "
            "you say will be carried out — no application opens, no screenshot "
            "is taken, nothing is searched. The run only works out which "
            "command you meant."))

        self.audio_notice = QLabel()
        self.audio_notice.setObjectName("ToggleTitle")
        self.audio_notice.setWordWrap(True)
        self.audio_notice.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.audio_notice)

        layout.addWidget(divider())
        layout.addWidget(section_title("Microphone"))
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        layout.addWidget(self.device_combo)
        self.device_detail = muted("")
        layout.addWidget(self.device_detail)

        self.meter = QProgressBar()
        self.meter.setObjectName("LevelMeter")
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(m.size(8))
        self.meter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.meter)

        self.check_result = QLabel("Run the microphone check to continue.")
        self.check_result.setObjectName("ToggleTitle")
        self.check_result.setWordWrap(True)
        layout.addWidget(self.check_result)

        layout.addWidget(divider())
        layout.addWidget(section_title("What this run will record"))
        self.environment = QLabel("")
        self.environment.setObjectName("Muted")
        self.environment.setWordWrap(True)
        self.environment.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.environment)
        layout.addStretch(1)

        buttons, row = flow_container(spacing=m.sp(8))
        self.check_button = QPushButton("Run 3-second microphone check")
        self.check_button.setObjectName("Primary")
        self.check_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.check_button.clicked.connect(self._run_mic_check)
        row.addWidget(self.check_button)

        self.start_button = QPushButton("Start benchmark")
        self.start_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self._start_run)
        row.addWidget(self.start_button)

        close_button = QPushButton("Close")
        close_button.setObjectName("Ghost")
        close_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        close_button.clicked.connect(self.reject)
        row.addWidget(close_button)

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(m.sp(10))
        wrapper_layout.addWidget(scroll_wrap(page), 1)
        wrapper_layout.addWidget(buttons)

        self._reload_devices()
        self._describe_environment()
        return wrapper

    def _reload_devices(self) -> None:
        from speech.capture import list_devices

        self.device_combo.blockSignals(True)
        try:
            self.device_combo.clear()
            self.device_combo.addItem("Windows default microphone", None)
            for device in list_devices():
                label = device.name
                if device.host_api:
                    label += f"  ·  {device.host_api}"
                self.device_combo.addItem(label, device.name)
            saved = self.engine.config.config.speech_input.device_name
            self.device_combo.setCurrentIndex(max(0, self.device_combo.findData(saved)))
        finally:
            self.device_combo.blockSignals(False)
        self._describe_device()

    def _describe_device(self) -> None:
        resolution = self.engine.listening.resolve_device()
        device = resolution.device
        if device is None:
            self.device_detail.setText("No input device is available.")
            return
        self.device_detail.setText(
            f"Using: {device.name} · {device.host_api or 'default API'} · "
            f"{device.channels} channel(s) · {device.sample_rate:.0f} Hz")

    def _on_device_changed(self) -> None:
        self.engine.config.update_speech_input(
            device_name=self.device_combo.currentData(), device_index=None)
        self._describe_device()
        self._mic_ok = False
        self.start_button.setEnabled(False)
        self.check_result.setText("Microphone changed. Run the check again.")

    def _describe_environment(self) -> None:
        """State the engine, model and gate, so the export is self-describing."""
        from ui.dashboard import APP_VERSION

        settings = self.engine.config.config.speech_input
        recognizer = self.engine.listening.recognizer
        resolution = self.engine.listening.resolve_device()
        device = resolution.device.name if resolution.device else "none"

        self.run.metadata = {
            "app_version": APP_VERSION,
            "started_at": None,
            "stt_engine": recognizer.name,
            "stt_model": getattr(recognizer, "model_size", ""),
            "gate_base": settings.confidence_threshold,
            "gate_note": ("length-scaled: x0.55 at 1-2 words, x0.80 at 3-4, "
                          "x1.00 at 5+"),
            "vad_sensitivity": settings.vad_sensitivity,
            "max_seconds": settings.max_seconds,
            "silence_timeout": settings.silence_timeout,
            "no_speech_timeout": settings.no_speech_timeout,
            "input_device": device,
            "language": self.engine.config.config.language.speech_input,
            "watchdog_seconds": RECORDING_WATCHDOG_SECONDS,
        }
        self.environment.setText(
            f"Speech engine: {recognizer.name} · model "
            f"{getattr(recognizer, 'model_size', '?')} · language "
            f"{self.engine.config.config.language.speech_input}\n"
            f"Confidence gate base: {settings.confidence_threshold} "
            f"({self.run.metadata['gate_note']})\n"
            f"Voice detection sensitivity: {settings.vad_sensitivity} · "
            f"recording watchdog {RECORDING_WATCHDOG_SECONDS:.0f}s\n"
            f"AARMA {APP_VERSION}")
        self.audio_notice.setText(
            "Your recordings will be saved on this computer, to:\n"
            f"{self.run.directory}\n"
            "They are never uploaded. You can delete them from the last screen.")

    def _run_mic_check(self) -> None:
        readiness = self.engine.listening.readiness()
        if not readiness.ready:
            self.check_result.setText(
                f"Can't use the microphone. {readiness.message} "
                f"{readiness.suggestion}")
            return
        self.check_button.setEnabled(False)
        self.check_result.setText("Say anything for three seconds…")
        self._start_thread(_CheckWorker(self.engine.listening), "run",
                           finished=self._on_mic_checked)

    @Slot(object)
    def _on_mic_checked(self, diagnosis) -> None:
        self.meter.setValue(0)
        self.check_button.setEnabled(True)
        self._stop_thread()

        if diagnosis is None:
            self._mic_check_failed("The check could not run at all.")
            return
        probe = next((p for p in diagnosis.probes if p.is_selected), None)
        if probe is None:
            self._mic_check_failed("The selected microphone was not found.")
            return
        if probe.peak < MIC_CHECK_MIN_PEAK:
            self._mic_check_failed(
                f"Almost no signal — peak {format_level(probe.peak)}. "
                f"{diagnosis.recommendation()}")
            return
        if not probe.speech_detected:
            self._mic_check_failed(
                f"Audio is arriving (peak {format_level(probe.peak)}) but voice "
                "detection did not trigger. Speak while the check runs, move "
                "closer, or raise the input level in Windows sound settings.")
            return

        self._mic_ok = True
        self.start_button.setEnabled(True)
        self.check_result.setText(
            f"✓ Microphone working — peak {format_level(probe.peak)}, "
            f"RMS {format_level(probe.rms)}, voice detected. "
            "You can start the benchmark.")

    def _mic_check_failed(self, message: str) -> None:
        self._mic_ok = False
        self.start_button.setEnabled(False)
        self.check_result.setText(
            f"✗ {message}\n\nThe benchmark can't start until this passes — "
            "otherwise all 20 commands would be wasted. Try a different device "
            "above, or use Settings → Speech input → Check microphone → "
            "Scan all devices to find one that hears you.")

    # ------------------------------------------------------------- run page

    def _build_run_page(self) -> QWidget:
        m = self.metrics
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(m.sp(10))

        self.progress_label = QLabel("Command 1 / 20")
        self.progress_label.setObjectName("FieldCaption")
        layout.addWidget(self.progress_label)

        self.expected_label = QLabel("")
        self.expected_label.setObjectName("PageTitle")
        self.expected_label.setWordWrap(True)
        layout.addWidget(self.expected_label)

        self.state_label = muted("")
        layout.addWidget(self.state_label)

        self.run_meter = QProgressBar()
        self.run_meter.setObjectName("LevelMeter")
        self.run_meter.setRange(0, 100)
        self.run_meter.setTextVisible(False)
        self.run_meter.setFixedHeight(m.size(8))
        self.run_meter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.run_meter)

        layout.addWidget(divider())

        grid = QGridLayout()
        grid.setHorizontalSpacing(m.sp(16))
        grid.setVerticalSpacing(m.sp(4))
        self._rows: Dict[str, QLabel] = {}
        for row, key in enumerate((
                "Audio", "VAD", "Heard", "Normalized", "Confidence",
                "Validation", "Intent", "Parameters", "Action",
                "Expected int", "Result", "Failure at")):
            caption = QLabel(key)
            caption.setObjectName("FieldCaption")
            value = QLabel("—")
            value.setObjectName("Muted")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(caption, row, 0, Qt.AlignLeft | Qt.AlignTop)
            grid.addWidget(value, row, 1, Qt.AlignLeft | Qt.AlignTop)
            self._rows[key] = value
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        layout.addStretch(1)

        buttons, row = flow_container(spacing=m.sp(8))
        self.record_button = QPushButton("Start recording")
        self.record_button.setObjectName("Primary")
        self.record_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.record_button.clicked.connect(self._record_current)
        row.addWidget(self.record_button)

        self.next_button = QPushButton("Next")
        self.next_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self._next_command)
        row.addWidget(self.next_button)

        self.retry_button = QPushButton("Retry")
        self.retry_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.retry_button.setEnabled(False)
        self.retry_button.clicked.connect(self._retry_current)
        row.addWidget(self.retry_button)

        self.skip_button = QPushButton("Skip")
        self.skip_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.skip_button.clicked.connect(self._skip_current)
        row.addWidget(self.skip_button)

        self.back_button = QPushButton("Back")
        self.back_button.setObjectName("Ghost")
        self.back_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.back_button.clicked.connect(self._previous_command)
        row.addWidget(self.back_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setObjectName("Ghost")
        self.pause_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.pause_button.clicked.connect(self._toggle_pause)
        row.addWidget(self.pause_button)

        self.abort_button = QPushButton("Abort")
        self.abort_button.setObjectName("Danger")
        self.abort_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.abort_button.clicked.connect(self._abort)
        row.addWidget(self.abort_button)

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(m.sp(10))
        wrapper_layout.addWidget(scroll_wrap(page), 1)
        wrapper_layout.addWidget(buttons)
        return wrapper

    # -------------------------------------------------------- summary page

    def _build_summary_page(self) -> QWidget:
        m = self.metrics
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(m.sp(10))

        layout.addWidget(section_title("Run complete"))
        self.summary_label = QLabel("")
        self.summary_label.setObjectName("ToggleTitle")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.summary_label)

        self.export_label = muted("")
        self.export_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.export_label)
        layout.addStretch(1)

        buttons, row = flow_container(spacing=m.sp(8))
        self.export_button = QPushButton("Export run")
        self.export_button.setObjectName("Primary")
        self.export_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.export_button.clicked.connect(self._export)
        row.addWidget(self.export_button)

        self.delete_audio_button = QPushButton("Delete this run's audio")
        self.delete_audio_button.setObjectName("Danger")
        self.delete_audio_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.delete_audio_button.clicked.connect(self._delete_audio)
        row.addWidget(self.delete_audio_button)

        done = QPushButton("Close")
        done.setObjectName("Ghost")
        done.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        done.clicked.connect(self.reject)
        row.addWidget(done)

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(m.sp(10))
        wrapper_layout.addWidget(scroll_wrap(page), 1)
        wrapper_layout.addWidget(buttons)
        return wrapper

    # --------------------------------------------------------------- resume

    def _offer_resume(self) -> None:
        partial = find_incomplete(len(self.commands))
        if partial is None:
            return
        done = len(partial.completed_ids())
        answer = QMessageBox.question(
            self, "Unfinished benchmark run",
            f"Run {partial.run_id} stopped after {done} of "
            f"{len(self.commands)} commands.\n\nResume it, or start a new run?\n\n"
            "Yes = resume    No = start fresh")
        if answer is QMessageBox.Yes:
            self.run = partial
            self._index = min(done, len(self.commands) - 1)
            self.check_result.setText(
                f"Resuming run {partial.run_id} at command {self._index + 1}.")
        self._describe_environment()

    # ------------------------------------------------------------ run control

    def _start_run(self) -> None:
        if not self._mic_ok:
            return
        from datetime import datetime, timezone

        self.run.metadata["started_at"] = datetime.now(timezone.utc).isoformat()
        self.run.save_metadata()
        self.stack.setCurrentIndex(1)
        self._show_command()

    @property
    def _command(self) -> ExpectedCommand:
        return self.commands[self._index]

    def _show_command(self) -> None:
        command = self._command
        self._attempt = self.run.next_attempt(command.id)
        self.progress_label.setText(
            f"Command {self._index + 1} / {len(self.commands)}"
            f"          [attempt {self._attempt}]")
        self.expected_label.setText(f"Say:  “{command.text}”")
        self.state_label.setText(
            "Press Start recording, say the phrase, then pause. "
            "Recording ends on its own when you stop speaking.")
        for label in self._rows.values():
            label.setText("—")
        self._rows["Expected int"].setText(
            f"{command.intent or 'NO_PRODUCTION_INTENT'}   "
            f"(mapped: {command.mapping_note})")
        self.next_button.setEnabled(False)
        self.retry_button.setEnabled(False)
        self.record_button.setEnabled(not self._paused)

    def _record_current(self) -> None:
        self._set_busy(True)
        self.state_label.setText("Listening… say the phrase now.")
        self._watchdog.start(int(RECORDING_WATCHDOG_SECONDS * 1000))
        worker = _CommandWorker(self.engine, self.suite, self._command,
                                self._attempt, self.run)
        self._start_thread(worker, "run", completed=self._on_command_done)

    @Slot()
    def _on_watchdog(self) -> None:
        """Never hang. Cancel capture; the cycle returns as a no-speech result."""
        logger.info("Benchmark recording watchdog fired")
        self.state_label.setText("Recording timed out. Moving on.")
        self.engine.listening.cancel()

    @Slot(object)
    def _on_command_done(self, record: Dict[str, Any]) -> None:
        self._watchdog.stop()
        self.run_meter.setValue(0)
        self._stop_thread()
        self._set_busy(False)

        self.run.append(record)          # On disk before anything else happens.
        self._display(record)
        self.next_button.setEnabled(True)
        self.retry_button.setEnabled(True)
        self.state_label.setText(
            "Recorded. Retry if you misspoke, or Next to continue.")

    def _display(self, record: Dict[str, Any]) -> None:
        audio = record.get("audio") or {}
        vad = record.get("vad") or {}
        stt = record.get("stt") or {}
        normalization = record.get("normalization") or {}
        validation = record.get("validation") or {}
        intent = record.get("intent") or {}
        action = record.get("action") or {}

        self._rows["Audio"].setText(
            f"{audio.get('duration_sec') or 0:.2f}s   "
            f"peak {format_level(audio.get('peak_amplitude') or 0.0)}   "
            f"→ {audio.get('wav_path') or '(not saved)'}")

        if vad.get("speech_detected"):
            self._rows["VAD"].setText(
                f"detected   speech {vad.get('speech_start_sec')}s–"
                f"{vad.get('speech_end_sec')}s   trailing silence "
                f"{vad.get('trailing_silence_sec')}s")
        else:
            self._rows["VAD"].setText("not detected")

        # Verbatim recogniser output. The normalised form has its own row.
        self._rows["Heard"].setText(
            f"\"{stt.get('raw_text')}\"" if stt.get("raw_text") else "(nothing)")
        self._rows["Normalized"].setText(
            normalization.get("normalized_text") or "—")
        confidence = stt.get("confidence")
        self._rows["Confidence"].setText(
            f"{confidence:.3f}" if confidence is not None else "n/a")

        gate_base = validation.get("gate_base")
        gate_effective = validation.get("gate_effective")
        gate_text = validation.get("result") or "—"
        if gate_base is not None and gate_effective is not None:
            gate_text += f"   (gate_base {gate_base} → gate_effective {gate_effective:.3f})"
        if validation.get("reason"):
            gate_text += f"\n{validation['reason']}"
        self._rows["Validation"].setText(gate_text)

        self._rows["Intent"].setText(
            f"{intent.get('name') or '—'}            "
            f"match: {intent.get('match_method') or '—'}   "
            f"score {intent.get('match_score')}")
        self._rows["Parameters"].setText(
            json.dumps(intent.get("parameters") or {}, ensure_ascii=False))
        self._rows["Action"].setText(
            f"{action.get('selected') or '—'}  |  risk: "
            f"{action.get('risk_level') or 'n/a'}  |  would confirm: "
            f"{'yes' if action.get('would_require_confirmation') else 'no'}  |  "
            f"EXECUTED: {'YES' if action.get('executed') else 'NO'}")
        self._rows["Result"].setText(record.get("final_result") or "—")
        self._rows["Failure at"].setText(
            record.get("failure_detail")
            or (record.get("failure_layer") if record.get("failure_layer") != "NONE"
                else "—"))

    def _next_command(self) -> None:
        if self._index + 1 >= len(self.commands):
            self._finish_run()
            return
        self._index += 1
        self._show_command()

    def _previous_command(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._show_command()

    def _retry_current(self) -> None:
        self._attempt = self.run.next_attempt(self._command.id)
        self.progress_label.setText(
            f"Command {self._index + 1} / {len(self.commands)}"
            f"          [attempt {self._attempt}]")
        self._record_current()

    def _skip_current(self) -> None:
        self.run.append(skipped_record(
            self._command, self.run.next_attempt(self._command.id)))
        self._next_command()

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.pause_button.setText("Resume" if self._paused else "Pause")
        self.record_button.setEnabled(not self._paused)
        self.state_label.setText(
            "Paused. Nothing is being recorded." if self._paused
            else "Ready. Press Start recording.")

    def _abort(self) -> None:
        answer = QMessageBox.question(
            self, "Abort benchmark",
            f"Stop the run at command {self._index + 1}?\n\n"
            "Everything recorded so far is already saved and can be exported "
            "or resumed later.")
        if answer is QMessageBox.Yes:
            self.engine.listening.cancel()
            self._finish_run()

    def _finish_run(self) -> None:
        self.stack.setCurrentIndex(2)
        summary = self.run.summary()
        stt = summary["stt_accuracy"]
        intent = summary["intent_accuracy_on_correct_transcripts"]
        end = summary["end_to_end_resolution"]
        self.summary_label.setText(
            f"Run {self.run.run_id} — {summary['total_attempts']} attempt(s) "
            f"across {summary['scored_commands']} scored command(s), "
            f"{summary['skipped']} skipped.\n\n"
            f"Speech recognition:      {stt['passed']} / {stt['total']}\n"
            f"Intent, on correct transcripts:  "
            f"{intent['passed']} / {intent['total']}\n"
            f"End-to-end resolution:   {end['passed']} / {end['total']}\n\n"
            "These are three separate measurements. They are not combined.")
        self.export_label.setText(
            f"Recordings and the incremental trace are in:\n{self.run.directory}")

    def _export(self) -> None:
        try:
            path = self.run.export(mapping=self.suite._mapping)
        except Exception as exc:  # noqa: BLE001 - includes ExportRefused
            QMessageBox.warning(self, "Export refused", str(exc))
            self.export_label.setText(f"Export refused: {exc}")
            return
        self.export_label.setText(
            f"Exported to:\n{path}\n\nSend this file back. It contains the run "
            "metadata, the mapping table, and every attempt including retries "
            "and skips.")

    def _delete_audio(self) -> None:
        answer = QMessageBox.question(
            self, "Delete recordings",
            f"Delete every .wav file from run {self.run.run_id}?\n\n"
            "The trace records and the export are not affected — only the "
            "audio.")
        if answer is not QMessageBox.Yes:
            return
        removed = self.run.delete_audio()
        self.export_label.setText(f"Deleted {removed} recording(s).")

    # ------------------------------------------------------------- plumbing

    def _set_busy(self, busy: bool) -> None:
        for button in (self.record_button, self.next_button, self.retry_button,
                       self.skip_button, self.back_button, self.pause_button):
            button.setEnabled(not busy)

    def _start_thread(self, worker: QObject, method: str, **connections) -> None:
        self._thread = QThread(self)
        self._worker = worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(getattr(worker, method))
        for signal_name, slot in connections.items():
            getattr(worker, signal_name).connect(slot)
        self._thread.start()

    def _stop_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(8000)
            self._thread = None
            self._worker = None

    @Slot(float)
    def _on_level(self, level: float) -> None:
        scaled = int(max(0.0, min(1.0, (level / 0.15) ** 0.5)) * 100)
        self.meter.setValue(scaled)
        self.run_meter.setValue(scaled)

    def _release(self) -> None:
        self._watchdog.stop()
        self.engine.listening.cancel()
        self._stop_thread()
        self.engine.listening.remove_level_listener(self._level_emitter)

    def reject(self) -> None:  # noqa: D102 - Qt naming
        self._release()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._release()
        super().closeEvent(event)
