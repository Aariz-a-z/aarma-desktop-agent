"""Record a custom wake word from the settings screen.

The dialog prompts; :mod:`speech.wakeword_enrollment` decides.  That split is
deliberate -- the console front end and this one must never be able to disagree
about whether a recording is good enough, so neither owns the judgement.  Both
call :meth:`EnrollmentRecorder.capture`, which returns a
:class:`~speech.wakeword_enrollment.ClipQuality`; this file only renders it.

Recording and training both happen off the GUI thread.  The engine releases the
microphone before the dialog opens and takes it back when the dialog closes --
on success, on failure, on cancel and on the window being shut.

The layout is a single card with one job on screen at a time: which phrase, how
many are done, what is happening now, and what to do next.  An earlier version
used a large log pane, which spent most of its life empty and made a working
dialog look broken.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import (
    QMetaObject,
    QObject,
    Q_ARG,
    Qt,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.responsive import Metrics
from ui.theme import Palette, build_stylesheet, get_palette
from ui.widgets import card, divider, muted
from utils.logger import get_logger

logger = get_logger("ui.wakeword_enrol")

IDEAL_SIZE = (560, 600)
MINIMUM_SIZE = (420, 520)

#: Status colours are carried as text, not stylesheet ids, so the dialog needs
#: no additions to the theme.
_TONE = {"idle": "#9aa4b2", "busy": "#5aa9e6", "good": "#4cc38a", "warn": "#e6a23c",
         "bad": "#e05c5c"}


class _Worker(QObject):
    """Recording and training, off the GUI thread."""

    ready = Signal(bool, str)          # microphone usable, why
    captured = Signal(object)          # EnrollmentClip
    progress = Signal(str, str)
    finished = Signal(object)          # EnrollmentResult
    failed = Signal(str)

    def __init__(self, engine) -> None:
        super().__init__()
        self._engine = engine
        self._recorder = None

    @Slot()
    def open_microphone(self) -> None:
        from speech.wakeword_enrollment import EnrollmentRecorder

        try:
            self._recorder = EnrollmentRecorder()
            self._recorder.__enter__()
            self._recorder.measure_room()
            usable, why = self._recorder.readiness()
            logger.info("Enrollment microphone", extra={"context": {
                "device": self._recorder.device_name,
                "sample_rate": self._recorder.sample_rate,
                "floor_rms": round(self._recorder.floor_rms, 8),
                "usable": usable}})
            if usable:
                why = (f"{self._recorder.device_name} ready "
                       f"({self._recorder.sample_rate} Hz)")
            self.ready.emit(usable, why)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Enrollment microphone could not open")
            self.ready.emit(False, f"the microphone could not be opened: {exc}")

    @Slot(object)
    def capture(self, prompt) -> None:
        if self._recorder is None:
            self.failed.emit("The microphone is not open.")
            return
        try:
            self.captured.emit(self._recorder.capture(prompt))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Enrollment capture failed")
            self.failed.emit(f"That recording failed: {exc}")

    @Slot()
    def close_microphone(self) -> None:
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None

    @Slot(object, str)
    def train(self, clips, display_name: str) -> None:
        self.close_microphone()          # the trainer must not hold the device
        try:
            result = self._engine.enrol_wake_word(
                list(clips), display_name=display_name,
                progress=lambda stage, message: self.progress.emit(stage, message))
        except Exception as exc:  # noqa: BLE001 - a failure must not kill the app
            logger.exception("Enrollment failed")
            self.failed.emit(f"Enrollment could not finish: {exc}")
            return
        self.finished.emit(result)


class WakeWordEnrolDialog(QDialog):
    """Prompt for recordings, then train, validate and report."""

    #: Emitted once the store has changed, so the settings page can refresh.
    enrolled = Signal()

    def __init__(self, engine, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._engine = engine
        self._clips: List = []
        self._index = 0
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None
        self._training = False
        self._can_record = False

        state = engine.wake_identity()
        self._phrase = state["phrase"]
        self._name = state["name"]

        from speech.wakeword_enrollment import enrollment_plan

        self._prompts = enrollment_plan(self._phrase)

        m = self.metrics = Metrics.current()
        self.palette_tokens: Palette = get_palette(
            engine.config.config.runtime.theme)
        self.setWindowTitle("Enrol wake word")
        self.setObjectName("Root")
        if not self.styleSheet():
            self.setStyleSheet(build_stylesheet(self.palette_tokens, m))
        self.resize(*IDEAL_SIZE)
        self.setMinimumSize(*MINIMUM_SIZE)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(m.sp(18), m.sp(18), m.sp(18), m.sp(18))
        outer.setSpacing(m.sp(12))

        frame, body = card(m, spacing=12)
        outer.addWidget(frame)

        heading = QLabel("Teach AARMA a new wake word")
        heading.setObjectName("Heading")
        heading.setWordWrap(True)
        body.addWidget(heading)

        self.phrase_label = QLabel(f'"{self._phrase}"')
        self.phrase_label.setWordWrap(True)
        self.phrase_label.setStyleSheet(
            f"font-size: {m.sp(21)}px; font-weight: 600;")
        body.addWidget(self.phrase_label)

        body.addWidget(muted(
            f"Say it naturally {len(self._prompts)} times. Vary it on purpose — "
            "the same phrase said identically every time teaches me only that "
            "one way of saying it."))

        # --- progress -----------------------------------------------------
        self.counter = QLabel(f"0 / {len(self._prompts)}")
        self.counter.setAlignment(Qt.AlignCenter)
        self.counter.setStyleSheet(
            f"font-size: {m.sp(30)}px; font-weight: 700;")
        body.addWidget(self.counter)

        self.meter = QProgressBar()
        self.meter.setRange(0, len(self._prompts))
        self.meter.setValue(0)
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(m.sp(6))
        body.addWidget(self.meter)

        # --- live status --------------------------------------------------
        self.status = QLabel("Opening the microphone…")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        body.addWidget(self.status)

        self.detail = muted("")
        body.addWidget(self.detail)

        body.addWidget(divider())

        self.hint = muted("")
        body.addWidget(self.hint)

        # --- buttons ------------------------------------------------------
        row = QHBoxLayout()
        row.setSpacing(m.sp(8))
        self.skip_button = QPushButton("Skip")
        self.skip_button.setEnabled(False)
        self.skip_button.clicked.connect(self._skip)
        self.record_button = QPushButton("Record")
        self.record_button.setObjectName("Primary")
        self.record_button.setEnabled(False)
        self.record_button.setDefault(True)
        self.record_button.clicked.connect(self._record)
        row.addStretch(1)
        row.addWidget(self.skip_button)
        row.addWidget(self.record_button)
        outer.addLayout(row)

        close_row = QHBoxLayout()
        self.close_button = QPushButton("Cancel")
        self.close_button.setObjectName("Ghost")
        self.close_button.clicked.connect(self.reject)
        close_row.addStretch(1)
        close_row.addWidget(self.close_button)
        outer.addLayout(close_row)

        self._set_status("busy", "Opening the microphone…")
        self._start_thread()

    # ------------------------------------------------------------- machinery

    def _start_thread(self) -> None:
        self._thread = QThread(self)
        self._worker = _Worker(self._engine)
        self._worker.moveToThread(self._thread)
        self._worker.ready.connect(self._on_ready)
        self._worker.captured.connect(self._on_captured)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.started.connect(self._worker.open_microphone)
        self._thread.start()

    def _set_status(self, tone: str, text: str, detail: str = "") -> None:
        dot = {"busy": "●", "good": "✓", "warn": "⚠", "bad": "✕"}.get(tone, "●")
        self.status.setText(f"{dot}  {text}")
        self.status.setStyleSheet(
            f"color: {_TONE.get(tone, _TONE['idle'])}; font-weight: 600;")
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))

    def _set_buttons(self, *, record: bool, skip: bool) -> None:
        """The one place button state is decided.

        Centralised because the previous version could strand the Record button
        disabled when a state transition was missed, which looks identical to a
        broken application.
        """
        self.record_button.setEnabled(record and self._can_record)
        self.skip_button.setEnabled(skip and self._can_record)

    def _show_prompt(self) -> None:
        if self._index >= len(self._prompts):
            self._begin_training()
            return
        prompt = self._prompts[self._index]
        self.counter.setText(f"{len(self._clips)} / {len(self._prompts)}")
        self._set_status("idle", "Ready to record",
                         f'Press Record, then say "{self._phrase}".')
        self.hint.setText(f"Recording {self._index + 1}: {prompt.hint}")
        self._set_buttons(record=True, skip=True)
        self.record_button.setFocus()

    def _record(self) -> None:
        self._set_buttons(record=False, skip=False)
        self._set_status("busy", "Listening…", f'Say "{self._phrase}" now.')
        prompt = self._prompts[self._index]
        # Queued into the worker thread, so the window keeps painting while the
        # microphone is open.
        QMetaObject.invokeMethod(self._worker, "capture", Qt.QueuedConnection,
                                 Q_ARG(object, prompt))

    def _skip(self) -> None:
        self._index += 1
        self._show_prompt()

    def _begin_training(self) -> None:
        from speech.wakeword_enrollment import MIN_RECORDINGS

        if len(self._clips) < MIN_RECORDINGS:
            self._can_record = False
            self._set_buttons(record=False, skip=False)
            self._set_status(
                "warn", f"Only {len(self._clips)} usable recordings",
                f"I need at least {MIN_RECORDINGS} to train and still have some "
                "left to test with. Nothing has been changed.")
            self.hint.setText("")
            self.close_button.setText("Close")
            return

        self._training = True
        self._can_record = False
        self._set_buttons(record=False, skip=False)
        self.close_button.setEnabled(False)
        self.meter.setRange(0, 0)                 # indeterminate
        self._set_status("busy", "Training and validating",
                         "This takes a few minutes and uses no network. "
                         "You can leave this window open.")
        self.hint.setText("")
        QMetaObject.invokeMethod(self._worker, "train", Qt.QueuedConnection,
                                 Q_ARG(object, list(self._clips)),
                                 Q_ARG(str, self._name))

    # ---------------------------------------------------------------- slots

    @Slot(bool, str)
    def _on_ready(self, usable: bool, why: str) -> None:
        self._can_record = usable
        if not usable:
            self._set_status("bad", "Microphone not usable", why)
            self.hint.setText("")
            self._set_buttons(record=False, skip=False)
            self.close_button.setText("Close")
            return
        self.detail.setText(why)
        self._show_prompt()

    @Slot(object)
    def _on_captured(self, clip) -> None:
        quality = clip.quality
        logger.debug("Enrollment clip rendered", extra={
            "context": {"diagnostics": quality.diagnostics()}})
        if not quality.accepted:
            self._set_status("warn", quality.reason.capitalize(), quality.advice)
            self.hint.setText("Press Record to try that one again, or Skip it.")
            self._set_buttons(record=True, skip=True)
            return

        self._clips.append(clip)
        self._index += 1
        self.meter.setValue(len(self._clips))
        self.counter.setText(f"{len(self._clips)} / {len(self._prompts)}")
        if self._index >= len(self._prompts):
            self._begin_training()
            return
        self._set_status(
            "good", f"Recording {len(self._clips)} saved",
            f"{quality.speech_seconds:.1f} s of speech"
            + (f", {quality.snr_db:.0f} dB above the room"
               if quality.snr_db is not None else "")
            + ". Vary your voice a little for the next one.")
        self.hint.setText(f"Recording {self._index + 1}: "
                          f"{self._prompts[self._index].hint}")
        self._set_buttons(record=True, skip=True)
        self.record_button.setFocus()

    @Slot(str, str)
    def _on_progress(self, stage: str, message: str) -> None:
        if message:
            self.detail.setText(message)

    @Slot(object)
    def _on_finished(self, result) -> None:
        self._training = False
        self.meter.setRange(0, 1)
        self.meter.setValue(1 if result.installed else 0)
        self.close_button.setEnabled(True)
        self.close_button.setText("Close")

        failed = ", ".join(c.name.replace("_", " ") for c in result.failures())
        if result.installed:
            self._set_status("good", f'"{self._phrase}" is ready', result.message)
            self.hint.setText("Restart AARMA, or reopen Settings, to arm it.")
            self.enrolled.emit()
        elif result.passed:
            self._set_status("good", "Passed validation", result.message)
            self.hint.setText("")
        else:
            self._set_status("warn", "Did not pass validation",
                             f"Failed: {failed}. Nothing was installed, and any "
                             "wake word you already had is untouched.")
            self.hint.setText("More recordings, with more variety, usually help.")

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._training = False
        self._can_record = False
        self.meter.setRange(0, 1)
        self._set_status("bad", "Enrollment stopped", message)
        self.hint.setText("")
        self._set_buttons(record=False, skip=False)
        self.close_button.setEnabled(True)
        self.close_button.setText("Close")

    # --------------------------------------------------------------- closing

    def reject(self) -> None:
        if self._training:
            return
        super().reject()

    def closeEvent(self, event) -> None:      # noqa: N802 - Qt naming
        if self._training:
            # The worker is inside training and cannot service a blocking call,
            # so closing now would either deadlock or strand the thread.
            event.ignore()
            return
        self._shutdown()
        super().closeEvent(event)

    def done(self, code: int) -> None:
        if self._training:
            return
        self._shutdown()
        super().done(code)

    def _shutdown(self) -> None:
        # Only ever reached when the worker is idle -- reject(), done() and
        # closeEvent() all refuse while training -- so the blocking call is safe
        # and guarantees the device is released before the thread ends.
        if self._worker is not None:
            try:
                QMetaObject.invokeMethod(self._worker, "close_microphone",
                                         Qt.BlockingQueuedConnection)
            except Exception:  # noqa: BLE001
                logger.debug("Enrollment microphone close did not complete")
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(4000)
            self._thread = None
            self._worker = None
