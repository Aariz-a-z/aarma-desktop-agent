"""The microphone check.

Rewritten after a user reported "No speech heard — 0.0000 RMS" while their
microphone worked in other applications. That single message covered at least
six different problems and pointed at the wrong one: it implied speech
recognition had failed when the real question was whether any audio signal
existed at all.

This version answers, in order:

1. Does Windows allow desktop apps to use the microphone?
2. Does the stream open?
3. Are samples arriving, and are they non-zero?
4. What is the level — printed so a real 0.000017 is not shown as "0.0000"?
5. Is there speech in it?

and, when the selected device is silent, shows every other device's measured
level so the user can pick the one that is actually hearing them.

It changes nothing on the system. It measures, explains, and offers the two
actions that are actually useful: choose a different microphone, or reset the
speech settings.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from speech.capture import CalibrationReport
from speech.diagnostics import format_level
from ui.responsive import Metrics, apply_initial_geometry, flow_container, scroll_wrap
from ui.theme import Palette, build_stylesheet, get_palette
from ui.widgets import divider, muted, section_title
from utils.logger import get_logger

logger = get_logger("ui.microphone_test")

IDEAL_SIZE = (680, 720)
MINIMUM_SIZE = (460, 400)


class _Worker(QObject):
    """Runs measurements off the GUI thread."""

    phase = Signal(str)
    level = Signal(float)
    calibrated = Signal(object)     # CalibrationReport
    diagnosed = Signal(object)      # MicrophoneDiagnosis

    def __init__(self, listening) -> None:
        super().__init__()
        self._listening = listening

    @Slot()
    def run_check(self) -> None:
        try:
            report = self._listening.calibrate(
                on_phase=self.phase.emit, on_level=self.level.emit)
        except Exception:  # noqa: BLE001 - a check must never crash the app
            logger.exception("Microphone check failed")
            report = CalibrationReport(error="The microphone check couldn't run.")
        self.calibrated.emit(report)

    @Slot()
    def run_diagnosis(self) -> None:
        try:
            self.phase.emit("scanning")
            diagnosis = self._listening.diagnose(seconds=0.8)
        except Exception:  # noqa: BLE001
            logger.exception("Microphone diagnosis failed")
            diagnosis = None
        self.diagnosed.emit(diagnosis)


class MicrophoneTestDialog(QDialog):
    """Guided check, per-device scan, device selection and reset."""

    #: Emitted when the user changes or resets the microphone selection.
    settings_changed = Signal()

    def __init__(self, listening, config, assistant_name: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._listening = listening
        self._config = config
        self._assistant_name = assistant_name
        self.metrics = Metrics.current()
        self.palette_tokens: Palette = get_palette(config.config.runtime.theme)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None

        m = self.metrics
        self.setWindowTitle("Microphone check")
        self.setObjectName("Root")
        if not self.styleSheet():
            self.setStyleSheet(build_stylesheet(self.palette_tokens, m))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(m.sp(24), m.sp(22), m.sp(24), m.sp(18))
        outer.setSpacing(m.sp(12))

        title = QLabel("Microphone check")
        title.setObjectName("PageTitle")
        outer.addWidget(title)

        body = QWidget()
        body.setObjectName("ConversationCanvas")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, m.sp(10), 0)
        layout.setSpacing(m.sp(12))

        # --- device picker --------------------------------------------------
        layout.addWidget(section_title("Microphone"))
        self.device_combo = QComboBox()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        layout.addWidget(self.device_combo)
        self.device_detail = muted("")
        layout.addWidget(self.device_detail)

        layout.addWidget(divider())

        # --- live check -----------------------------------------------------
        self.instruction = QLabel("Press “Start check” and follow the prompts.")
        self.instruction.setObjectName("ToggleTitle")
        self.instruction.setWordWrap(True)
        layout.addWidget(self.instruction)

        self.meter = QProgressBar()
        self.meter.setObjectName("LevelMeter")
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(m.size(8))
        self.meter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.meter)

        grid = QGridLayout()
        grid.setHorizontalSpacing(m.sp(18))
        grid.setVerticalSpacing(m.sp(6))
        self._rows = {}
        for row, key in enumerate(("Windows permission", "Capture", "Samples",
                                   "Noise", "Speech", "Clipping", "Overall")):
            caption = QLabel(key)
            caption.setObjectName("FieldCaption")
            value = QLabel("—")
            value.setObjectName("ToggleTitle")
            value.setWordWrap(True)
            grid.addWidget(caption, row, 0, Qt.AlignLeft | Qt.AlignTop)
            grid.addWidget(value, row, 1, Qt.AlignLeft | Qt.AlignTop)
            self._rows[key] = value
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self.advice = muted("")
        layout.addWidget(self.advice)

        layout.addWidget(divider())

        # --- per-device scan ------------------------------------------------
        layout.addWidget(section_title("All input devices"))
        layout.addWidget(muted(
            "What each device is actually delivering. Use this to find the one "
            "that hears you."))
        self.scan_results = QLabel("Not scanned yet.")
        self.scan_results.setWordWrap(True)
        self.scan_results.setObjectName("Muted")
        self.scan_results.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.scan_results)
        layout.addStretch(1)

        outer.addWidget(scroll_wrap(body), 1)

        # --- actions --------------------------------------------------------
        actions, row = flow_container(spacing=m.sp(8))
        self.check_button = QPushButton("Start check")
        self.check_button.setObjectName("Primary")
        self.check_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.check_button.clicked.connect(self.start_check)
        row.addWidget(self.check_button)

        self.scan_button = QPushButton("Scan all devices")
        self.scan_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.scan_button.clicked.connect(self.start_scan)
        row.addWidget(self.scan_button)

        self.reset_button = QPushButton("Reset speech settings")
        self.reset_button.setObjectName("Danger")
        self.reset_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.reset_button.clicked.connect(self.reset_speech_settings)
        row.addWidget(self.reset_button)

        close_button = QPushButton("Close")
        close_button.setObjectName("Ghost")
        close_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        close_button.clicked.connect(self.reject)
        row.addWidget(close_button)
        outer.addWidget(actions)

        self._reload_devices()
        self._show_permission()
        apply_initial_geometry(self, ideal=IDEAL_SIZE, minimum=MINIMUM_SIZE)

    # ------------------------------------------------------------- devices

    def _reload_devices(self) -> None:
        """Rebuild the picker from the devices present right now."""
        from speech.capture import list_devices

        self.device_combo.blockSignals(True)
        try:
            self.device_combo.clear()
            self.device_combo.addItem("Windows default microphone", None)
            for device in list_devices():
                label = f"{device.name}"
                if device.host_api:
                    label += f"  ·  {device.host_api}"
                self.device_combo.addItem(label, device.name)
            saved = self._config.config.speech_input.device_name
            index = self.device_combo.findData(saved)
            self.device_combo.setCurrentIndex(max(0, index))
        finally:
            self.device_combo.blockSignals(False)
        self._describe_selection()

    def _describe_selection(self) -> None:
        resolution = self._listening.resolve_device()
        device = resolution.device
        if device is None:
            self.device_detail.setText("No input device is available.")
            return
        detail = (f"Using: {device.name} · {device.host_api or 'default API'} · "
                  f"{device.channels} channel(s) · device rate "
                  f"{device.sample_rate:.0f} Hz")
        if resolution.fell_back:
            detail = resolution.notice() + "  " + detail
        self.device_detail.setText(detail)

    def _on_device_changed(self) -> None:
        self._config.update_speech_input(
            device_name=self.device_combo.currentData(), device_index=None)
        self._describe_selection()
        self.settings_changed.emit()
        self.instruction.setText(
            "Microphone changed. Run the check to confirm it hears you.")

    def _show_permission(self) -> None:
        from speech.diagnostics import check_windows_permission

        permission = check_windows_permission()
        if not permission.checked:
            self._rows["Windows permission"].setText("Not applicable")
            return
        self._rows["Windows permission"].setText(
            "✓ Allowed" if permission.allowed
            else "✗ Blocked for desktop apps")
        if not permission.allowed:
            self.advice.setText(permission.advice())

    # --------------------------------------------------------------- check

    def start_check(self) -> None:
        readiness = self._listening.readiness()
        if not readiness.ready:
            self.instruction.setText(readiness.message)
            self.advice.setText(readiness.suggestion)
            self._rows["Overall"].setText(f"✗ {readiness.message}")
            return

        self._set_busy(True)
        self.instruction.setText("Please stay quiet for two seconds…")
        for key in ("Capture", "Samples", "Noise", "Speech", "Clipping", "Overall"):
            self._rows[key].setText("—")
        self.advice.setText("")
        self._start_worker("run_check")

    def start_scan(self) -> None:
        self._set_busy(True)
        self.instruction.setText("Testing every input device…")
        self.scan_results.setText("Scanning…")
        self._start_worker("run_diagnosis")

    def _start_worker(self, method: str) -> None:
        self._thread = QThread(self)
        self._worker = _Worker(self._listening)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(getattr(self._worker, method))
        self._worker.phase.connect(self._on_phase)
        self._worker.level.connect(self._on_level)
        self._worker.calibrated.connect(self._on_calibrated)
        self._worker.diagnosed.connect(self._on_diagnosed)
        self._thread.start()

    def _set_busy(self, busy: bool) -> None:
        for button in (self.check_button, self.scan_button, self.reset_button):
            button.setEnabled(not busy)
        self.device_combo.setEnabled(not busy)

    @Slot(str)
    def _on_phase(self, phase: str) -> None:
        if phase == "quiet":
            self.instruction.setText("Please stay quiet for two seconds…")
        elif phase == "scanning":
            self.instruction.setText("Testing every input device…")
        else:
            self.instruction.setText(f'Now say: "Hello {self._assistant_name}."')

    @Slot(float)
    def _on_level(self, level: float) -> None:
        self.meter.setValue(int(max(0.0, min(1.0, (level / 0.15) ** 0.5)) * 100))

    @Slot(object)
    def _on_calibrated(self, report: CalibrationReport) -> None:
        self.meter.setValue(0)
        self._set_busy(False)
        self.check_button.setText("Run again")
        self._stop_thread()

        if report.error:
            self.instruction.setText("The check couldn't finish.")
            self._rows["Capture"].setText("✗ Failed")
            self._rows["Overall"].setText("✗ Could not read the microphone")
            self.advice.setText(report.error)
            return

        self.instruction.setText("Check complete.")
        self._rows["Capture"].setText("✓ Stream opened")

        # The distinction the old message could not make.
        if report.peak <= 0.0:
            self._rows["Samples"].setText("✗ No audio signal — every sample was zero")
        elif report.peak < 0.0005:
            self._rows["Samples"].setText(
                f"⚠ Almost no signal (peak {format_level(report.peak)})")
        else:
            self._rows["Samples"].setText(
                f"✓ Receiving audio (peak {format_level(report.peak)})")

        self._rows["Noise"].setText(
            f"{report.noise_label} — {format_level(report.noise_rms)} RMS")
        if report.speech_detected:
            self._rows["Speech"].setText(
                f"✓ Detected — {format_level(report.speech_rms)} RMS")
        else:
            self._rows["Speech"].setText("✗ Not detected")
        self._rows["Clipping"].setText("Yes" if report.clipped else "None")

        if report.speech_detected and not report.clipped:
            self._rows["Overall"].setText("✓ Microphone working")
        elif report.peak < 0.0005:
            self._rows["Overall"].setText("⚠ AARMA is not receiving microphone audio")
        else:
            self._rows["Overall"].setText("⚠ Audio received, but no speech detected")
        self.advice.setText(report.advice())

        if report.peak < 0.0005:
            self.advice.setText(
                report.advice() + "  Use “Scan all devices” to find one that hears you.")

    @Slot(object)
    def _on_diagnosed(self, diagnosis) -> None:
        self.meter.setValue(0)
        self._set_busy(False)
        self._stop_thread()
        self.instruction.setText("Device scan complete.")

        if diagnosis is None:
            self.scan_results.setText("The scan could not run.")
            return

        lines = []
        for probe in diagnosis.probes:
            marker = "●" if probe.is_selected else "○"
            lines.append(
                f"{marker} {probe.status.symbol} {probe.name}\n"
                f"     {probe.host_api or 'default'} · {probe.channels} ch · "
                f"{probe.capture_rate or int(probe.device_rate)} Hz\n"
                f"     {probe.status.label} — frames {probe.frames}, "
                f"RMS {format_level(probe.rms)}, peak {format_level(probe.peak)}, "
                f"non-zero {probe.nonzero}")
        self.scan_results.setText("\n".join(lines) if lines else "No devices found.")
        self.advice.setText(diagnosis.recommendation())

    # --------------------------------------------------------------- reset

    def reset_speech_settings(self) -> None:
        confirmed = QMessageBox.question(
            self, "Reset speech input",
            "Reset speech input settings to recommended defaults?\n\n"
            "This clears the selected microphone, recognition quality, "
            "sensitivity and timeouts. Your assistant, language, permissions "
            "and data are not affected.")
        if confirmed is not QMessageBox.Yes:
            return
        self._config.reset_speech_input()
        self._reload_devices()
        self.instruction.setText("Speech input settings reset to defaults.")
        for key in ("Capture", "Samples", "Noise", "Speech", "Clipping", "Overall"):
            self._rows[key].setText("—")
        self.advice.setText("Run the check to confirm the default microphone works.")
        self.settings_changed.emit()

    # ------------------------------------------------------------ lifecycle

    def _stop_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(4000)
            self._thread = None
            self._worker = None

    def reject(self) -> None:  # noqa: D102 - Qt naming
        self._listening.cancel()
        self._stop_thread()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._listening.cancel()
        self._stop_thread()
        super().closeEvent(event)
