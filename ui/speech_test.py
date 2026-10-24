"""The speech recognition test: a real person says known phrases, once each.

The microphone check (:mod:`ui.microphone_test`) answers "is audio arriving?".
This answers the next question — "is AARMA understanding it?" — and it is the
only way to answer it, because synthetic speech does not measure a microphone
in a room with a person in front of it. Nothing here simulates a voice.

For each phrase it shows what was expected, what the recogniser actually
returned **before validation**, the confidence, the threshold that confidence
had to clear, how much speech the detector found, and the verdict. Those columns
exist to separate three failures that all look identical from the outside:

* the recogniser misheard,
* the recogniser was right and AARMA's own checks discarded it,
* nothing was captured at all.

The test never runs the recognised text as a command. It calls
:meth:`~speech.input_engine.ListeningEngine.listen_once`, which does not notify
the result listeners — so saying "Open Chrome." during a test opens nothing.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from speech.accuracy import AccuracyReport, UtteranceScore, default_phrases
from ui.responsive import Metrics, apply_initial_geometry, flow_container, scroll_wrap
from ui.theme import Palette, build_stylesheet, get_palette
from ui.widgets import divider, muted, section_title
from utils.logger import get_logger

logger = get_logger("ui.speech_test")

IDEAL_SIZE = (760, 780)
MINIMUM_SIZE = (480, 400)

#: Rendered beside each row. Text, not colour alone.
_VERDICT_MARKS = {
    "PASS": "✓",
    "MISHEARD": "✗",
    "REJECTED (heard correctly)": "⚠",
    "REJECTED (misheard)": "✗",
    "NO SPEECH": "—",
    "SKIPPED": "·",
}


class _TestWorker(QObject):
    """Runs one listening cycle per phrase, off the GUI thread."""

    prompt = Signal(int)            # Index of the phrase to speak.
    scored = Signal(int, object)    # Index, UtteranceScore.
    finished = Signal()

    def __init__(self, listening, phrases: List[str]) -> None:
        super().__init__()
        self._listening = listening
        self._phrases = phrases
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    @Slot()
    def run(self) -> None:
        for index, phrase in enumerate(self._phrases):
            if self._stop:
                break
            self.prompt.emit(index)
            try:
                result, record = self._listening.listen_once()
                score = UtteranceScore(
                    expected=phrase,
                    detected=result.text,
                    raw_detected=record.raw_text,
                    confidence=record.confidence,
                    threshold=record.threshold_used,
                    speech_seconds=record.speech_seconds,
                    latency=record.latency,
                    accepted=result.ok,
                    reason=record.detail or record.reason,
                )
            except Exception as exc:  # noqa: BLE001 - a test must not crash the app
                logger.exception("Speech test cycle failed")
                score = UtteranceScore(expected=phrase,
                                       reason=f"{type(exc).__name__}")
            self.scored.emit(index, score)
        self.finished.emit()


class SpeechTestDialog(QDialog):
    """Guided real-voice accuracy test."""

    #: Carries the input level from the audio thread to the meter.
    level_measured = Signal(float)

    def __init__(self, listening, config, assistant_name: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._listening = listening
        self._config = config
        self._phrases = default_phrases(assistant_name)
        self._report = AccuracyReport()
        self.metrics = Metrics.current()
        self.palette_tokens: Palette = get_palette(config.config.runtime.theme)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_TestWorker] = None

        m = self.metrics
        self.setWindowTitle("Speech recognition test")
        self.setObjectName("Root")
        if not self.styleSheet():
            self.setStyleSheet(build_stylesheet(self.palette_tokens, m))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(m.sp(24), m.sp(22), m.sp(24), m.sp(18))
        outer.setSpacing(m.sp(12))

        title = QLabel("Speech recognition test")
        title.setObjectName("PageTitle")
        outer.addWidget(title)

        body = QWidget()
        body.setObjectName("ConversationCanvas")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, m.sp(10), 0)
        layout.setSpacing(m.sp(12))

        layout.addWidget(muted(
            "You'll be asked to say nine short phrases, one at a time. Say each "
            "one normally and then pause. Nothing you say here is treated as a "
            "command, and no audio is saved unless you switched that on."))

        self.prompt_label = QLabel("Press “Start test” to begin.")
        self.prompt_label.setObjectName("PageTitle")
        self.prompt_label.setWordWrap(True)
        layout.addWidget(self.prompt_label)

        self.state_label = muted("")
        layout.addWidget(self.state_label)

        self.meter = QProgressBar()
        self.meter.setObjectName("LevelMeter")
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(m.size(8))
        self.meter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.meter)

        layout.addWidget(divider())
        layout.addWidget(section_title("Results"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(m.sp(14))
        grid.setVerticalSpacing(m.sp(6))
        for column, heading in enumerate(
                ("Expected", "Detected", "Confidence", "Speech", "Verdict")):
            caption = QLabel(heading)
            caption.setObjectName("FieldCaption")
            grid.addWidget(caption, 0, column, Qt.AlignLeft | Qt.AlignBottom)

        self._cells: List[List[QLabel]] = []
        for row, phrase in enumerate(self._phrases, start=1):
            cells = []
            for column in range(5):
                cell = QLabel(phrase if column == 0 else "—")
                cell.setObjectName("ToggleTitle" if column == 0 else "Muted")
                cell.setWordWrap(column <= 1)
                cell.setTextInteractionFlags(Qt.TextSelectableByMouse)
                grid.addWidget(cell, row, column, Qt.AlignLeft | Qt.AlignTop)
                cells.append(cell)
            self._cells.append(cells)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 4)
        layout.addLayout(grid)

        layout.addWidget(divider())
        self.summary = QLabel("Not run yet.")
        self.summary.setObjectName("ToggleTitle")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.notes = muted("")
        layout.addWidget(self.notes)
        layout.addStretch(1)

        outer.addWidget(scroll_wrap(body), 1)

        actions, row = flow_container(spacing=m.sp(8))
        self.start_button = QPushButton("Start test")
        self.start_button.setObjectName("Primary")
        self.start_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.start_button.clicked.connect(self.start_test)
        row.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_test)
        row.addWidget(self.stop_button)

        self.copy_button = QPushButton("Copy report")
        self.copy_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self.copy_report)
        row.addWidget(self.copy_button)

        close_button = QPushButton("Close")
        close_button.setObjectName("Ghost")
        close_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        close_button.clicked.connect(self.reject)
        row.addWidget(close_button)
        outer.addWidget(actions)

        # Registered once and removed on close, so a dismissed dialog is not
        # still being called from the audio thread. The emitter is held rather
        # than re-derived, because removal compares by identity.
        self.level_measured.connect(self._on_level)
        self._level_emitter = self.level_measured.emit
        self._listening.on_level(self._level_emitter)

        apply_initial_geometry(self, ideal=IDEAL_SIZE, minimum=MINIMUM_SIZE)

    # ----------------------------------------------------------------- run

    def start_test(self) -> None:
        readiness = self._listening.readiness()
        if not readiness.ready:
            self.prompt_label.setText("The test can't run.")
            self.state_label.setText(f"{readiness.message} {readiness.suggestion}")
            return

        self._report = AccuracyReport()
        for cells in self._cells:
            for cell in cells[1:]:
                cell.setText("—")
        self.summary.setText("Running…")
        self.notes.setText("")
        self._set_busy(True)

        self._thread = QThread(self)
        self._worker = _TestWorker(self._listening, self._phrases)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.prompt.connect(self._on_prompt)
        self._worker.scored.connect(self._on_scored)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def stop_test(self) -> None:
        if self._worker is not None:
            self._worker.stop()
        self._listening.cancel()

    @Slot(int)
    def _on_prompt(self, index: int) -> None:
        self.prompt_label.setText(f'Say: “{self._phrases[index]}”')
        self.state_label.setText(f"Phrase {index + 1} of {len(self._phrases)} — "
                                 "listening now, then pause when you're done.")
        for column, cell in enumerate(self._cells[index]):
            if column:
                cell.setText("…")

    @Slot(float)
    def _on_level(self, level: float) -> None:
        self.meter.setValue(int(max(0.0, min(1.0, (level / 0.15) ** 0.5)) * 100))

    @Slot(int, object)
    def _on_scored(self, index: int, score: UtteranceScore) -> None:
        self._report.scores.append(score)
        cells = self._cells[index]

        heard = score.detected or score.raw_detected
        cells[1].setText(heard if heard else "(nothing)")
        cells[2].setText(
            "n/a" if score.confidence is None
            else (f"{score.confidence:.3f}"
                  + ("" if score.threshold is None
                     else f" / {score.threshold:.3f}")))
        cells[3].setText(f"{score.speech_seconds:.2f}s")
        mark = _VERDICT_MARKS.get(score.verdict, "")
        cells[4].setText(f"{mark} {score.verdict}".strip())
        self.summary.setText(self._report.headline())

    @Slot()
    def _on_finished(self) -> None:
        self.meter.setValue(0)
        self._set_busy(False)
        self._stop_thread()
        self.prompt_label.setText("Test complete.")
        self.state_label.setText("")
        self.copy_button.setEnabled(bool(self._report.total))
        self.summary.setText(self._report.headline())
        self.notes.setText(self._advice())
        logger.info("Speech test finished", extra={"context": {
            "attempted": self._report.total, "passed": self._report.passes,
            "recognised": self._report.recognised,
            "rejected_correct": self._report.rejected_correct,
            "mean_latency": round(self._report.mean_latency, 2),
        }})

    def _advice(self) -> str:
        """What the numbers actually mean. No congratulation, no blame."""
        report = self._report
        if not report.total:
            return "Nothing was measured."
        parts = [f"Mean word error rate {report.mean_error_rate * 100:.1f}%, "
                 f"mean response {report.mean_latency:.1f}s."]
        if report.rejected_correct:
            parts.append(
                f"{report.rejected_correct} phrase(s) were transcribed correctly "
                "and then refused — that is AARMA's confidence check being too "
                "strict, not a microphone problem.")
        elif report.recognised < report.total:
            parts.append(
                "The remaining phrases were misheard rather than refused. "
                "A more accurate recognition quality in Settings usually helps, "
                "at the cost of speed.")
        if report.meets_target:
            parts.append(f"This run met the {report.target * 100:.0f}% target. "
                         "One run of nine phrases is a small sample.")
        elif report.total:
            parts.append(f"This run did not reach the {report.target * 100:.0f}% "
                         "target.")
        return "  ".join(parts)

    # ------------------------------------------------------------ plumbing

    def copy_report(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self._report.report())
        self.notes.setText("Report copied to the clipboard.")

    def _set_busy(self, busy: bool) -> None:
        self.start_button.setEnabled(not busy)
        self.start_button.setText("Run again" if self._report.total else "Start test")
        self.stop_button.setEnabled(busy)

    def _stop_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(6000)
            self._thread = None
            self._worker = None

    def _release(self) -> None:
        self.stop_test()
        self._stop_thread()
        self._listening.remove_level_listener(self._level_emitter)

    def reject(self) -> None:  # noqa: D102 - Qt naming
        self._release()
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._release()
        super().closeEvent(event)
