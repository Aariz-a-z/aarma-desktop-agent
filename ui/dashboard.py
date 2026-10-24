"""The AARMA desktop dashboard.

The window is a thin shell over :class:`~core.engine.AarmaEngine`. It renders
five pages -- Conversation, Activity, Permissions, Settings and About -- and
every one of them reads its data from the engine or the configuration rather
than keeping its own copy.

Structural decisions worth calling out:

* **The engine runs on a worker thread.** Launching an application, probing
  connectivity or writing an audit row can all block for a moment; doing any of
  that on the GUI thread produces the frozen-window look that makes desktop
  software feel amateur.
* **Confirmations are inline, not modal.** A high-risk action appears as a card
  in the conversation with its own Yes/Cancel buttons, so the user can read the
  surrounding context before deciding, and can simply ignore it.
* **Nothing carries a hardcoded size.** The window opens at a size derived from
  the screen's work area; the sidebar is a share of the window; chat bubbles are
  a share of the viewport; settings grids reflow to one column when narrow; the
  starter prompts wrap. Every page scrolls, so content is never clipped —
  including when Windows text scaling makes everything larger.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config.settings import (
    ASSISTANT_PROFILES,
    AppConfig,
    normalise_wake_phrase,
    wake_phrase_for_name,
)
from core.engine import AarmaEngine, EngineResponse, ResponseKind
from language.manager import LANGUAGE_CATALOG, LanguageStatus
from security.passphrase import PassphraseError, PassphraseManager
from security.permissions import Permission
from speech.providers.whisper_local import MODEL_SIZES
from speech.voices import (VOICE_PROFILES, VoiceStyle, get_profile,
                           profiles_for_gender)
from ui.responsive import (
    BREAKPOINT_COMPACT,
    BREAKPOINT_NARROW,
    Metrics,
    apply_initial_geometry,
    breakpoint_threshold,
    bubble_width_for,
    flow_container,
    grid_columns_for,
    scroll_wrap,
    sidebar_width_for,
)
from ui.theme import Palette, app_icon, build_stylesheet, get_palette
from ui.widgets import (
    Avatar,
    ConfirmationCard,
    ElidingLabel,
    ListeningIndicator,
    MessageBubble,
    Pill,
    ToggleRow,
    card,
    divider,
    labelled_field,
    muted,
    section_title,
)
from utils.logger import get_logger
from utils.paths import get_paths

logger = get_logger("ui.dashboard")

APP_VERSION = "0.3.1"
PHASE_LABEL = "Phase 2B · voice in and out"

#: Comfortable size and usable floor. The window actually opens at whatever
#: `apply_initial_geometry` derives from the screen; these are only bounds.
IDEAL_SIZE = (1240, 820)
MINIMUM_SIZE = (720, 460)

#: Starter prompts shown under the composer so a first-time user has somewhere
#: to click instead of guessing what the assistant understands.
SUGGESTIONS = (
    "What time is it?",
    "Open Chrome",
    "List files in Documents",
    "Help",
)


# --------------------------------------------------------------------- worker

class EngineWorker(QObject):
    """Runs engine calls off the GUI thread."""

    finished = Signal(object)  # EngineResponse

    def __init__(self, engine: AarmaEngine) -> None:
        super().__init__()
        self._engine = engine

    @Slot(str)
    def handle_text(self, text: str) -> None:
        self.finished.emit(self._engine.handle_text(text, source="gui"))

    @Slot(str)
    def handle_voice(self, text: str) -> None:
        """Same pipeline as typing; only the audit source differs.

        Recording the source matters for the audit trail: reviewing what an
        assistant did should show whether a command was typed or spoken.
        """
        self.finished.emit(self._engine.handle_text(text, source="voice"))

    @Slot(str, bool)
    def resolve_confirmation(self, token: str, approved: bool) -> None:
        self.finished.emit(self._engine.resolve_confirmation(token, approved))


class ConversationBridge(QObject):
    """Marshals conversation events onto the GUI thread.

    A wake-word turn publishes from its own worker thread, and Qt widgets may
    not be touched from there. The sink does nothing but emit; everything
    visible happens in the slot.
    """

    event = Signal(object)  # ConversationEvent


class ConfigBridge(QObject):
    """Marshals configuration changes from any thread onto the GUI thread."""

    changed = Signal(object)  # AppConfig


class ReminderBridge(QObject):
    """Marshals due reminders onto the GUI thread.

    The scheduler polls on its own daemon thread, and Qt widgets may not be
    touched from it.  The sink therefore does nothing but emit; everything
    visible happens in the slot, on the GUI thread.
    """

    fired = Signal(object)  # ReminderEvent


class ListeningBridge(QObject):
    """Marshals microphone events onto the GUI thread.

    Capture callbacks arrive on the audio thread and transcription callbacks on
    the listening worker; Qt widgets may be touched from neither.
    """

    state_changed = Signal(object, str)   # ListeningState, message
    level = Signal(float)
    result = Signal(object)               # TranscriptionResult


class SpeechBridge(QObject):
    """Marshals speech events from the speech worker onto the GUI thread.

    The speech engine is deliberately Qt-free and calls its listeners on its own
    worker; a Qt signal is the safe way to bring that back to the interface.
    """

    started = Signal()
    finished = Signal(object)  # SpeechOutcome


# ------------------------------------------------------------ responsive grid

class ReflowGrid(QWidget):
    """A grid of equal-width cells that drops to fewer columns as it narrows.

    Settings pairs read well side by side on a wide window and badly when each
    is squeezed to 140 pixels. Rather than picking a column count for one screen
    size, the count is recomputed from the actual width on every resize.
    """

    def __init__(self, metrics: Metrics, *, max_columns: int = 2,
                 min_column_width: int = 240) -> None:
        super().__init__()
        self._metrics = metrics
        self._max_columns = max_columns
        self._min_column_width = metrics.size(min_column_width)
        self._cells: List[QWidget] = []
        self._columns = 0

        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(metrics.sp(16))
        self._grid.setVerticalSpacing(metrics.sp(12))
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

    def add(self, widget: QWidget) -> None:
        self._cells.append(widget)
        self._relayout(force=True)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self, *, force: bool = False) -> None:
        columns = grid_columns_for(max(1, self.width()),
                                   min_column_width=self._min_column_width,
                                   max_columns=self._max_columns)
        if columns == self._columns and not force:
            return
        self._columns = columns
        for cell in self._cells:
            self._grid.removeWidget(cell)
        for index, cell in enumerate(self._cells):
            self._grid.addWidget(cell, index // columns, index % columns)
        for column in range(self._max_columns):
            self._grid.setColumnStretch(column, 1 if column < columns else 0)


# ----------------------------------------------------------------------- pages

class ChatPage(QWidget):
    """The conversation: transcript, composer and starter prompts."""

    submitted = Signal(str)
    confirmation_decided = Signal(str, bool)
    microphone_clicked = Signal()

    def __init__(self, assistant_name: str, metrics: Metrics) -> None:
        super().__init__()
        self.metrics = metrics
        self._bubbles: List[QWidget] = []
        m = metrics

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(m.sp(14))

        self.canvas = QWidget()
        self.canvas.setObjectName("ConversationCanvas")
        self.canvas_layout = QVBoxLayout(self.canvas)
        self.canvas_layout.setContentsMargins(m.sp(2), m.sp(2), m.sp(12), m.sp(2))
        self.canvas_layout.setSpacing(m.sp(12))
        self.canvas_layout.addStretch(1)

        self.scroll = scroll_wrap(self.canvas, object_name="Conversation")
        layout.addWidget(self.scroll, 1)

        # Scrolling on a timer races the layout: a tall bubble changes the
        # scroll range *after* the timer fires, leaving the newest message
        # partly below the fold. rangeChanged fires once the range is final.
        # Start at the bottom: an empty transcript has nothing to read back.
        self._follow_tail = True
        self.scroll.verticalScrollBar().rangeChanged.connect(self._on_range_changed)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_scrolled)

        # --- composer -------------------------------------------------------
        composer = QFrame()
        composer.setObjectName("Composer")
        composer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        composer_layout = QHBoxLayout(composer)
        composer_layout.setContentsMargins(m.sp(14), m.sp(10), m.sp(12), m.sp(10))
        composer_layout.setSpacing(m.sp(10))

        self.mic_button = QPushButton("\U0001F3A4")
        self.mic_button.setObjectName("IconButton")
        mic_size = m.size(36)
        self.mic_button.setFixedSize(mic_size, mic_size)  # Round button: square by definition.
        self.mic_button.setCursor(Qt.PointingHandCursor)
        self.mic_button.setToolTip("Click to speak")
        self.mic_button.clicked.connect(self.microphone_clicked)
        composer_layout.addWidget(self.mic_button)

        self.input = QLineEdit()
        self.input.setObjectName("CommandInput")
        self.input.setPlaceholderText(f"Ask {assistant_name} something…")
        self.input.setClearButtonEnabled(True)
        self.input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.input.returnPressed.connect(self._emit_submit)
        composer_layout.addWidget(self.input, 1)

        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("Primary")
        self.send_button.clicked.connect(self._emit_submit)
        composer_layout.addWidget(self.send_button)

        layout.addWidget(composer)

        # Visible only while the microphone is genuinely capturing.
        self.listening_indicator = ListeningIndicator(m)
        layout.addWidget(self.listening_indicator)

        # --- starter prompts ------------------------------------------------
        # A flow layout, so the chips wrap onto a second row on a narrow window
        # rather than running off the edge.
        self.suggestion_bar, chips = flow_container(spacing=m.sp(8))
        hint = muted("Try:", wrap=False)
        hint.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        chips.addWidget(hint)
        for text in SUGGESTIONS:
            chip = QPushButton(text)
            chip.setObjectName("Ghost")
            chip.setCursor(Qt.PointingHandCursor)
            chip.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            chip.clicked.connect(lambda _=False, value=text: self._fill(value))
            chips.addWidget(chip)
        layout.addWidget(self.suggestion_bar)

    # ---------------------------------------------------------------- helpers

    def _fill(self, text: str) -> None:
        self.input.setText(text)
        self.input.setFocus()

    def _emit_submit(self) -> None:
        text = self.input.text().strip()
        if text:
            self.input.clear()
            self.submitted.emit(text)

    def set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.send_button.setText("…" if busy else "Send")
        self.input.setEnabled(not busy)
        if not busy:
            self.input.setFocus()

    def set_placeholder(self, assistant_name: str) -> None:
        self.input.setPlaceholderText(f"Ask {assistant_name} something…")

    def set_microphone_state(self, state, message: str = "", *,
                             ready: bool = True, reason: str = "") -> None:
        """Reflect the true microphone state in the composer.

        Never shows a listening appearance unless capture is actually running,
        and the tooltip always explains a disabled button rather than leaving
        the user to guess.
        """
        from speech.input_engine import ListeningState

        listening = state is ListeningState.LISTENING
        processing = state is ListeningState.PROCESSING

        name = "IconButtonActive" if listening else "IconButton"
        if self.mic_button.objectName() != name:
            self.mic_button.setObjectName(name)
            self.mic_button.style().unpolish(self.mic_button)
            self.mic_button.style().polish(self.mic_button)

        self.mic_button.setEnabled(ready and not processing)
        if listening:
            self.mic_button.setToolTip("Listening — click to stop")
            self.listening_indicator.show_state(
                "Listening…", hint="Click the microphone to stop")
        elif processing:
            self.mic_button.setToolTip("Working out what you said")
            self.listening_indicator.show_state("Understanding…", metering=False)
        else:
            self.listening_indicator.stop()
            self.mic_button.setToolTip(
                "Click to speak" if ready else (reason or message or "Microphone unavailable"))

    def set_microphone_level(self, level: float) -> None:
        if self.listening_indicator.isVisible():
            self.listening_indicator.set_level(level)

    # ------------------------------------------------------------- responsive

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._apply_bubble_widths()

    def _bubble_max_width(self) -> int:
        return bubble_width_for(self.scroll.viewport().width(), self.metrics)

    def _apply_bubble_widths(self) -> None:
        """Reflow every bubble to the current viewport.

        Applied to existing bubbles too, not just new ones, so resizing the
        window re-wraps the whole transcript instead of leaving old messages at
        a width chosen for the previous size.
        """
        maximum = self._bubble_max_width()
        for bubble in self._bubbles:
            bubble.setMaximumWidth(maximum)

    # --------------------------------------------------------------- messages

    def _append(self, widget: QWidget, *, align_right: bool) -> None:
        widget.setMaximumWidth(self._bubble_max_width())
        self._bubbles.append(widget)

        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        if align_right:
            row.addStretch(1)
            row.addWidget(widget)
        else:
            row.addWidget(widget)
            row.addStretch(1)
        self.canvas_layout.insertWidget(self.canvas_layout.count() - 1, container)
        # Deliberately does not re-enable following. A new message arriving is
        # not a reason to yank the view away from what the user is reading.
        if self._follow_tail:
            QTimer.singleShot(0, self._scroll_to_bottom)

    #: How close to the bottom still counts as following the conversation.
    #: Roughly one line: far enough that a small overshoot or a bubble
    #: re-wrapping does not detach the view, close enough that deliberately
    #: scrolling up does.
    NEAR_BOTTOM_PX = 48

    def _on_scrolled(self, value: int) -> None:
        """Follow the tail only while the user is sitting at it."""
        bar = self.scroll.verticalScrollBar()
        self._follow_tail = (bar.maximum() - value) <= self.NEAR_BOTTOM_PX

    def follow_tail(self) -> None:
        """Jump to the newest message and keep following it.

        Called when the user sends something: having just spoken, they are
        looking at the end of the conversation whatever they were reading a
        moment ago.
        """
        self._follow_tail = True
        QTimer.singleShot(0, self._scroll_to_bottom)

    def _on_range_changed(self, _minimum: int, maximum: int) -> None:
        if self._follow_tail:
            self.scroll.verticalScrollBar().setValue(maximum)

    def _scroll_to_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def add_user_message(self, text: str, *, spoken: bool = False,
                         confidence: Optional[float] = None) -> None:
        """Add what the user said or typed.

        Spoken turns are labelled, so the transcript makes clear which words
        AARMA *heard* rather than read — and a marginal recognition is flagged
        instead of being presented as certain.
        """
        meta = "you"
        if spoken:
            meta = "you · spoken"
            if confidence is not None and confidence < 0.6:
                meta = "you · spoken, unclear"
        self._append(MessageBubble(text, role="user", meta=meta, metrics=self.metrics),
                     align_right=True)

    def add_assistant_message(self, text: str, assistant_name: str,
                              *, tone: str = "normal") -> None:
        self._append(MessageBubble(text, role="assistant", meta=assistant_name,
                                   tone=tone, metrics=self.metrics),
                     align_right=False)

    def add_confirmation(self, token: str, text: str) -> None:
        widget = ConfirmationCard(token, text, self.metrics)
        widget.decided.connect(self.confirmation_decided)
        self._append(widget, align_right=False)


class ActivityPage(QWidget):
    """The audit trail: what ran, at what risk, and whether it was allowed."""

    COLUMNS = ("Time", "Action", "Risk", "Source", "Allowed", "Outcome")

    def __init__(self, engine: AarmaEngine, metrics: Metrics) -> None:
        super().__init__()
        self._engine = engine
        m = metrics

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(m.sp(14))

        header = QHBoxLayout()
        header.setSpacing(m.sp(12))
        titles = QVBoxLayout()
        titles.setSpacing(m.sp(2))
        title = QLabel("Activity")
        title.setObjectName("PageTitle")
        titles.addWidget(title)
        titles.addWidget(muted(
            "Every action AARMA attempted this session and earlier, with the "
            "permission decision behind it."
        ))
        header.addLayout(titles, 1)
        refresh = QPushButton("Refresh")
        refresh.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        refresh.clicked.connect(self.refresh)
        header.addWidget(refresh, 0, Qt.AlignTop)
        layout.addLayout(header)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setMinimumSize(1, m.size(120))
        # Per-pixel scrolling, so a table narrower than its content stays fully
        # reachable instead of jumping a whole column at a time.
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(len(self.COLUMNS) - 1, QHeaderView.Stretch)
        header_view.setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        self.empty_note = muted("Nothing has run yet. Ask AARMA to do something.")
        layout.addWidget(self.empty_note)

    def refresh(self) -> None:
        entries = self._engine.database.recent_actions(limit=100)
        self.empty_note.setVisible(not entries)
        self.table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            values = (
                entry.created_at.replace("T", " ").replace("+00:00", " UTC"),
                entry.action,
                entry.risk,
                entry.source,
                "yes" if entry.allowed else "no",
                (entry.message or ("ok" if entry.succeeded else "failed")).split("\n")[0][:90],
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))


class PermissionsPage(QWidget):
    """Live capability switches, grouped the way the catalogue orders them."""

    def __init__(self, engine: AarmaEngine, metrics: Metrics) -> None:
        super().__init__()
        self._engine = engine
        self._rows: Dict[str, ToggleRow] = {}
        m = metrics

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(m.sp(14))

        title = QLabel("Permissions")
        title.setObjectName("PageTitle")
        outer.addWidget(title)
        outer.addWidget(muted(
            "AARMA can only use a capability you have switched on. Anything that "
            "watches you — microphone, camera, screen capture, keyboard and mouse — "
            "stays off until you turn it on. Changes take effect on the very next command."
        ))

        content = QWidget()
        content.setObjectName("ConversationCanvas")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, m.sp(12), 0)
        content_layout.setSpacing(m.sp(14))

        unlocked_by = engine.registry.required_permissions()
        for group, infos in engine.permissions.grouped().items():
            frame, frame_layout = card(m)
            frame_layout.addWidget(section_title(group))
            for index, info in enumerate(infos):
                if index:
                    frame_layout.addWidget(divider())
                actions = unlocked_by.get(info.permission, [])
                description = info.description
                if actions:
                    description += f"  ·  unlocks: {', '.join(sorted(actions))}"
                row = ToggleRow(info.permission.value, info.display_name, description,
                                engine.permissions.is_granted(info.permission), m)
                row.toggled.connect(self._on_toggled)
                self._rows[info.permission.value] = row
                frame_layout.addWidget(row)
            content_layout.addWidget(frame)

        content_layout.addStretch(1)
        outer.addWidget(scroll_wrap(content), 1)

    permission_changed = Signal()

    def _on_toggled(self, key: str, granted: bool) -> None:
        self._engine.permissions.set(Permission(key), granted)
        # Microphone and speaker availability follow their permissions, so the
        # composer must not keep offering a control that is now denied.
        self.permission_changed.emit()

    def refresh(self) -> None:
        for key, row in self._rows.items():
            row.set_checked(self._engine.permissions.is_granted(Permission(key)))


class SettingsPage(QWidget):
    """Assistant identity, language, appearance, connectivity and security."""

    identity_changed = Signal()
    theme_changed = Signal(str)

    def __init__(self, engine: AarmaEngine, metrics: Metrics) -> None:
        super().__init__()
        self._engine = engine
        self.metrics = metrics
        m = metrics

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(m.sp(14))

        title = QLabel("Settings")
        title.setObjectName("PageTitle")
        outer.addWidget(title)
        outer.addWidget(muted("Everything here is written to your local configuration file "
                              "and restored the next time AARMA starts."))

        content = QWidget()
        content.setObjectName("ConversationCanvas")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, m.sp(12), 0)
        layout.setSpacing(m.sp(14))

        layout.addWidget(self._assistant_card())
        layout.addWidget(self._voice_card())
        layout.addWidget(self._speech_input_card())
        layout.addWidget(self._language_card())
        layout.addWidget(self._experience_card())
        layout.addWidget(self._security_card())
        layout.addStretch(1)

        outer.addWidget(scroll_wrap(content), 1)
        self.refresh()

    def _action_row(self, button: QPushButton) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button)
        return row

    # ------------------------------------------------------------------ cards

    def _assistant_card(self) -> QFrame:
        m = self.metrics
        frame, layout = card(m)
        layout.addWidget(section_title("Assistant"))

        grid = ReflowGrid(m)

        self.name_input = QLineEdit()
        self.name_input.setMaxLength(32)
        grid.add(labelled_field("Name", self.name_input,
                                "What you call the assistant.", m))

        self.profile_combo = QComboBox()
        for key, profile in ASSISTANT_PROFILES.items():
            self.profile_combo.addItem(f"{profile.display_name} · {profile.gender}", key)
        self.profile_combo.addItem("Custom", "custom")
        grid.add(labelled_field("Profile", self.profile_combo,
                                "Switching resets name, voice and wake word.", m))

        self.voice_combo = QComboBox()
        for profile in VOICE_PROFILES.values():
            self.voice_combo.addItem(profile.display_name, profile.voice_id)
        grid.add(labelled_field("Voice", self.voice_combo,
                                "Full voice controls are in the Voice section below.", m))

        self.wake_input = QLineEdit()
        self.wake_input.setMaxLength(40)
        grid.add(labelled_field("Wake word", self.wake_input,
                                "Derived from the name unless you set it yourself.", m))

        layout.addWidget(grid)

        # Whether the phrase can actually be heard depends on which
        # models are installed, so it is reported rather than assumed.
        self.wake_status = muted("")
        layout.addWidget(self.wake_status)

        self.enrol_button = QPushButton("Enrol wake word")
        self.enrol_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.enrol_button.setEnabled(False)
        self.enrol_button.clicked.connect(self._enrol_wake_word)
        layout.addLayout(self._action_row(self.enrol_button))

        save = QPushButton("Save assistant settings")
        save.setObjectName("Primary")
        save.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        save.clicked.connect(self._save_assistant)
        layout.addLayout(self._action_row(save))
        return frame

    def _slider(self, minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return slider

    def _voice_card(self) -> QFrame:
        """Voice selection, delivery controls and a live availability readout."""
        m = self.metrics
        frame, layout = card(m)
        layout.addWidget(section_title("Voice"))

        self.voice_status = QLabel()
        self.voice_status.setWordWrap(True)
        layout.addWidget(self.voice_status)
        self.voice_detail = muted("")
        layout.addWidget(self.voice_detail)

        grid = ReflowGrid(m)

        # Presentation first: it selects a whole profile, and the voice list
        # below it is that profile's own voices. Choosing from either keeps the
        # two in step, because both go through ConfigManager rather than
        # writing one field each.
        self.voice_presentation_combo = QComboBox()
        self.voice_presentation_combo.addItem("Female", "female")
        self.voice_presentation_combo.addItem("Male", "male")
        grid.add(labelled_field("Voice profile", self.voice_presentation_combo,
                                "Switches the voice, delivery and pitch together.", m))

        self.voice_profile_combo = QComboBox()
        grid.add(labelled_field("Voice", self.voice_profile_combo,
                                "Which voice the assistant speaks with.", m))

        self.voice_style_combo = QComboBox()
        for style in VoiceStyle:
            self.voice_style_combo.addItem(
                style.label, style.value)
        grid.add(labelled_field("Delivery", self.voice_style_combo,
                                "How it is spoken, not what is said.", m))

        self.voice_mode_combo = QComboBox()
        self.voice_mode_combo.addItem("Automatic (best available)", "auto")
        self.voice_mode_combo.addItem("Natural voice (needs internet)", "online")
        self.voice_mode_combo.addItem("Offline voice only", "offline")
        grid.add(labelled_field("Voice source", self.voice_mode_combo,
                                "Automatic uses the natural voice when you're online.", m))

        speed_row = QWidget()
        speed_layout = QHBoxLayout(speed_row)
        speed_layout.setContentsMargins(0, 0, 0, 0)
        speed_layout.setSpacing(m.sp(10))
        self.voice_rate_slider = self._slider(50, 160, 100)
        self.voice_rate_label = muted("", wrap=False)
        speed_layout.addWidget(self.voice_rate_slider, 1)
        speed_layout.addWidget(self.voice_rate_label)
        grid.add(labelled_field("Speaking speed", speed_row, metrics=m))

        volume_row = QWidget()
        volume_layout = QHBoxLayout(volume_row)
        volume_layout.setContentsMargins(0, 0, 0, 0)
        volume_layout.setSpacing(m.sp(10))
        self.voice_volume_slider = self._slider(0, 100, 90)
        self.voice_volume_label = muted("", wrap=False)
        volume_layout.addWidget(self.voice_volume_slider, 1)
        volume_layout.addWidget(self.voice_volume_label)
        grid.add(labelled_field("Volume", volume_row, metrics=m))

        pitch_row = QWidget()
        pitch_layout = QHBoxLayout(pitch_row)
        pitch_layout.setContentsMargins(0, 0, 0, 0)
        pitch_layout.setSpacing(m.sp(10))
        self.voice_pitch_slider = self._slider(-10, 10, 0)
        self.voice_pitch_label = muted("", wrap=False)
        pitch_layout.addWidget(self.voice_pitch_slider, 1)
        pitch_layout.addWidget(self.voice_pitch_label)
        grid.add(labelled_field("Pitch", pitch_row,
                                "Only some voices support this.", m))

        layout.addWidget(grid)

        self.speak_replies_toggle = ToggleRow(
            "voice_enabled", "Speak replies out loud",
            "The assistant reads its answers aloud as well as showing them.",
            True, m)
        self.speak_replies_toggle.toggled.connect(self._on_voice_toggle)
        layout.addWidget(self.speak_replies_toggle)

        self.speak_notifications_toggle = ToggleRow(
            "voice_notifications", "Speak reminders and alerts",
            "Applies to reminders and notifications, which arrive in a later phase.",
            True, m)
        self.speak_notifications_toggle.toggled.connect(self._on_voice_toggle)
        layout.addWidget(self.speak_notifications_toggle)

        buttons, row = flow_container(spacing=m.sp(8))
        self.test_voice_button = QPushButton("Test voice")
        self.test_voice_button.setObjectName("Primary")
        self.test_voice_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.test_voice_button.clicked.connect(self._test_voice)
        row.addWidget(self.test_voice_button)

        self.stop_voice_button = QPushButton("Stop")
        self.stop_voice_button.setObjectName("Ghost")
        self.stop_voice_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.stop_voice_button.clicked.connect(self._stop_voice)
        row.addWidget(self.stop_voice_button)
        layout.addWidget(buttons)

        # Saved as soon as they change: a voice setting is only meaningful when
        # you can hear the result, so Test voice must use the current values.
        self.voice_presentation_combo.currentIndexChanged.connect(
            self._switch_voice_presentation)
        self.voice_profile_combo.currentIndexChanged.connect(self._save_voice)
        self.voice_style_combo.currentIndexChanged.connect(self._save_voice)
        self.voice_mode_combo.currentIndexChanged.connect(self._save_voice)
        for slider in (self.voice_rate_slider, self.voice_volume_slider,
                       self.voice_pitch_slider):
            slider.valueChanged.connect(self._save_voice)
        return frame

    def _speech_input_card(self) -> QFrame:
        """Microphone and recognition settings (specification section 22)."""
        m = self.metrics
        frame, layout = card(m)
        layout.addWidget(section_title("Speech input"))

        self.mic_status = QLabel()
        self.mic_status.setWordWrap(True)
        layout.addWidget(self.mic_status)
        self.mic_detail = muted("")
        layout.addWidget(self.mic_detail)
        # Where the audio goes is stated on the screen, not only in the docs.
        self.mic_privacy = muted("")
        layout.addWidget(self.mic_privacy)

        grid = ReflowGrid(m)

        self.mic_device_combo = QComboBox()
        grid.add(labelled_field(
            "Microphone", self.mic_device_combo,
            "Windows default follows whatever you choose in Windows.", m))

        self.mic_model_combo = QComboBox()
        for size in MODEL_SIZES.values():
            self.mic_model_combo.addItem(
                f"{size.display_name} · ~{size.latency_seconds:g}s · "
                f"{size.approximate_mb} MB", size.key)
        grid.add(labelled_field(
            "Recognition quality", self.mic_model_combo,
            "Larger is more accurate but slower to answer.", m))

        timeout_row = QWidget()
        timeout_layout = QHBoxLayout(timeout_row)
        timeout_layout.setContentsMargins(0, 0, 0, 0)
        timeout_layout.setSpacing(m.sp(10))
        self.mic_timeout_slider = self._slider(5, 30, 15)
        self.mic_timeout_label = muted("", wrap=False)
        timeout_layout.addWidget(self.mic_timeout_slider, 1)
        timeout_layout.addWidget(self.mic_timeout_label)
        grid.add(labelled_field("Maximum listening time", timeout_row, metrics=m))

        sensitivity_row = QWidget()
        sensitivity_layout = QHBoxLayout(sensitivity_row)
        sensitivity_layout.setContentsMargins(0, 0, 0, 0)
        sensitivity_layout.setSpacing(m.sp(10))
        self.mic_sensitivity_slider = self._slider(0, 100, 50)
        self.mic_sensitivity_label = muted("", wrap=False)
        sensitivity_layout.addWidget(self.mic_sensitivity_slider, 1)
        sensitivity_layout.addWidget(self.mic_sensitivity_label)
        grid.add(labelled_field(
            "Hearing sensitivity", sensitivity_row,
            "Raise it for quiet or distant speech; lower it in a noisy room.", m))

        confidence_row = QWidget()
        confidence_layout = QHBoxLayout(confidence_row)
        confidence_layout.setContentsMargins(0, 0, 0, 0)
        confidence_layout.setSpacing(m.sp(10))
        self.mic_confidence_slider = self._slider(10, 90, 45)
        self.mic_confidence_label = muted("", wrap=False)
        confidence_layout.addWidget(self.mic_confidence_slider, 1)
        confidence_layout.addWidget(self.mic_confidence_label)
        grid.add(labelled_field(
            "Required certainty", confidence_row,
            "Below this, I'll ask you to repeat rather than guess.", m))

        layout.addWidget(grid)

        check_buttons, check_row = flow_container(spacing=m.sp(8))
        self.mic_test_button = QPushButton("Check microphone")
        self.mic_test_button.setObjectName("Primary")
        self.mic_test_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.mic_test_button.clicked.connect(self._open_microphone_test)
        check_row.addWidget(self.mic_test_button)

        self.mic_reset_button = QPushButton("Reset speech input settings")
        self.mic_reset_button.setObjectName("Ghost")
        self.mic_reset_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.mic_reset_button.clicked.connect(self._reset_speech_input)
        check_row.addWidget(self.mic_reset_button)
        layout.addWidget(check_buttons)

        self.listen_enabled_toggle = ToggleRow(
            "speech_input_enabled", "Listen for voice commands",
            "The microphone is used only while you hold a listening session open.",
            True, m)
        self.listen_enabled_toggle.toggled.connect(self._on_listen_toggle)
        layout.addWidget(self.listen_enabled_toggle)

        layout.addWidget(muted(
            "The language you speak is set in the Language section, under "
            "'Speech input'."))

        layout.addWidget(divider())
        layout.addWidget(section_title("Advanced diagnostics"))
        layout.addWidget(muted(
            "What actually happened on the last listening attempt. Useful when "
            "AARMA says it couldn't understand you and you want to know why."))

        diagnostics_grid = QGridLayout()
        diagnostics_grid.setHorizontalSpacing(m.sp(16))
        diagnostics_grid.setVerticalSpacing(m.sp(4))
        self._diagnostic_rows = {}
        for row, key in enumerate((
                "Microphone", "Recording", "Levels", "Voice detected",
                "Model", "Heard", "Certainty", "Decision", "Response time")):
            caption = QLabel(key)
            caption.setObjectName("FieldCaption")
            value = QLabel("—")
            value.setObjectName("Muted")
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            diagnostics_grid.addWidget(caption, row, 0, Qt.AlignLeft | Qt.AlignTop)
            diagnostics_grid.addWidget(value, row, 1, Qt.AlignLeft | Qt.AlignTop)
            self._diagnostic_rows[key] = value
        diagnostics_grid.setColumnStretch(1, 1)
        layout.addLayout(diagnostics_grid)

        self.save_audio_toggle = ToggleRow(
            "save_debug_audio", "Save diagnostic audio",
            "Writes each recording to a folder on this computer so you can hear "
            "what AARMA heard. Off by default. Nothing is ever uploaded, and "
            "only the last few recordings are kept.",
            False, m)
        self.save_audio_toggle.toggled.connect(self._on_save_audio_toggle)
        layout.addWidget(self.save_audio_toggle)

        diagnostic_buttons, diagnostic_row = flow_container(spacing=m.sp(8))
        self.speech_test_button = QPushButton("Run speech test")
        self.speech_test_button.setObjectName("Primary")
        self.speech_test_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.speech_test_button.clicked.connect(self._open_speech_test)
        diagnostic_row.addWidget(self.speech_test_button)

        self.diagnostic_refresh_button = QPushButton("Refresh")
        self.diagnostic_refresh_button.setObjectName("Ghost")
        self.diagnostic_refresh_button.setSizePolicy(QSizePolicy.Maximum,
                                                     QSizePolicy.Fixed)
        self.diagnostic_refresh_button.clicked.connect(self._refresh_diagnostics)
        diagnostic_row.addWidget(self.diagnostic_refresh_button)

        self.voice_benchmark_button = QPushButton("Voice command benchmark")
        self.voice_benchmark_button.setSizePolicy(QSizePolicy.Maximum,
                                                  QSizePolicy.Fixed)
        self.voice_benchmark_button.clicked.connect(self._open_voice_benchmark)
        diagnostic_row.addWidget(self.voice_benchmark_button)

        self.diagnostic_copy_button = QPushButton("Copy details")
        self.diagnostic_copy_button.setObjectName("Ghost")
        self.diagnostic_copy_button.setSizePolicy(QSizePolicy.Maximum,
                                                  QSizePolicy.Fixed)
        self.diagnostic_copy_button.clicked.connect(self._copy_diagnostics)
        diagnostic_row.addWidget(self.diagnostic_copy_button)
        layout.addWidget(diagnostic_buttons)

        self.mic_device_combo.currentIndexChanged.connect(self._save_speech_input)
        self.mic_model_combo.currentIndexChanged.connect(self._save_speech_input)
        self.mic_timeout_slider.valueChanged.connect(self._save_speech_input)
        self.mic_confidence_slider.valueChanged.connect(self._save_speech_input)
        self.mic_sensitivity_slider.valueChanged.connect(self._save_speech_input)
        return frame

    def _open_microphone_test(self) -> None:
        from ui.microphone_test import MicrophoneTestDialog

        dialog = MicrophoneTestDialog(self._engine.listening, self._engine.config,
                                      self._engine.assistant_name, self)
        dialog.settings_changed.connect(self._refresh_speech_input)
        dialog.exec()
        self._refresh_speech_input()

    def _open_speech_test(self) -> None:
        from ui.speech_test import SpeechTestDialog

        dialog = SpeechTestDialog(self._engine.listening, self._engine.config,
                                  self._engine.assistant_name, self)
        dialog.exec()
        self._refresh_diagnostics()

    def _open_voice_benchmark(self) -> None:
        from ui.voice_benchmark import VoiceBenchmarkDialog

        dialog = VoiceBenchmarkDialog(self._engine, self)
        dialog.exec()
        self._refresh_diagnostics()

    def _on_save_audio_toggle(self, _key: str, enabled: bool) -> None:
        self._engine.config.update_speech_input(save_debug_audio=bool(enabled))
        if enabled:
            from utils.paths import get_paths

            self._diagnostic_rows["Decision"].setText(
                f"Recordings will be saved to {get_paths().diagnostic_dir}")

    def _refresh_diagnostics(self) -> None:
        """Show the last listening attempt in full.

        The user gets one friendly sentence when recognition fails; this is
        where the real reason lives, so "I couldn't understand that clearly"
        is never the end of the story.
        """
        from speech.diagnostics import format_level

        rows = getattr(self, "_diagnostic_rows", None)
        if not rows:
            return

        record = self._engine.listening.diagnostics.last
        if record is None:
            for value in rows.values():
                value.setText("—")
            rows["Decision"].setText("No listening attempt yet.")
            return

        rows["Microphone"].setText(record.device or "Windows default")
        rows["Recording"].setText(
            f"{record.capture_seconds:.2f}s recorded, "
            f"{record.speech_seconds:.2f}s of speech")
        rows["Levels"].setText(
            f"RMS {format_level(record.rms)}, peak {format_level(record.peak)}, "
            f"background {format_level(record.noise_rms)}"
            + ("  ·  CLIPPED" if record.clipped else ""))
        rows["Voice detected"].setText("Yes" if record.speech_detected else "No")
        rows["Model"].setText(
            f"{record.model or '—'} ({record.language or '—'})"
            + (f", detected {record.detected_language}"
               if record.detected_language
               and record.detected_language != record.language else ""))
        # The raw recogniser output, before validation touched it.
        rows["Heard"].setText(record.raw_text or "(nothing returned)")
        certainty = record.confidence_label
        if record.threshold_used is not None:
            certainty += f"  (needed {record.threshold_used:.3f})"
        if record.average_logprob is not None:
            certainty += f"  ·  avg logprob {record.average_logprob:.3f}"
        rows["Certainty"].setText(certainty)
        decision = "ACCEPTED" if record.accepted else "REJECTED"
        if record.reason:
            decision += f" — {record.reason}"
        if record.detail:
            decision += f" ({record.detail})"
        if record.audio_path:
            decision += f"\nAudio saved: {record.audio_path}"
        rows["Decision"].setText(decision)
        rows["Response time"].setText(f"{record.latency:.2f}s")

    def _copy_diagnostics(self) -> None:
        record = self._engine.listening.diagnostics.last
        if record is None:
            return
        QApplication.clipboard().setText(record.report())
        self._diagnostic_rows["Decision"].setText(
            self._diagnostic_rows["Decision"].text() + "\nCopied to the clipboard.")

    def _reset_speech_input(self) -> None:
        confirmed = QMessageBox.question(
            self, "Reset speech input",
            "Reset speech input settings to recommended defaults?\n\n"
            "This clears the selected microphone, recognition quality, "
            "sensitivity and timeouts. Your assistant, language, permissions "
            "and data are not affected.")
        if confirmed is not QMessageBox.Yes:
            return
        self._engine.config.reset_speech_input()
        self._refresh_speech_input()

    def _reset_all_settings(self) -> None:
        confirmed = QMessageBox.question(
            self, "Reset all settings",
            "This will reset AARMA configuration to first-run defaults.\n\n"
            "Your conversation history and activity log will remain unless you "
            "separately choose to delete them.\n\nReset all settings?")
        if confirmed is not QMessageBox.Yes:
            return
        self._engine.config.reset_all_settings()
        self.refresh()
        self.identity_changed.emit()
        QMessageBox.information(
            self, "Reset all settings",
            "Settings have been reset. Your data was not deleted.")

    def _language_card(self) -> QFrame:
        m = self.metrics
        frame, layout = card(m)
        layout.addWidget(section_title("Language"))
        layout.addWidget(muted(
            "Interface, speech input and speech output are configured separately so "
            "you can read one language and speak another."
        ))

        grid = ReflowGrid(m, max_columns=3, min_column_width=200)
        self.language_combos: Dict[str, QComboBox] = {}
        for key, caption in (("interface", "Interface"), ("input", "Speech input"),
                             ("output", "Speech output")):
            combo = QComboBox()
            for info in LANGUAGE_CATALOG.values():
                suffix = "" if info.status is LanguageStatus.TRANSLATED else f"  (Phase {info.phase})"
                combo.addItem(f"{info.english_name}{suffix}", info.code)
            self.language_combos[key] = combo
            grid.add(labelled_field(caption, combo, metrics=m))
        layout.addWidget(grid)

        save = QPushButton("Save language settings")
        save.setObjectName("Primary")
        save.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        save.clicked.connect(self._save_languages)
        layout.addLayout(self._action_row(save))
        return frame

    def _experience_card(self) -> QFrame:
        m = self.metrics
        frame, layout = card(m)
        layout.addWidget(section_title("Appearance and connectivity"))

        grid = ReflowGrid(m)

        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.addItem("Light", "light")
        self.theme_combo.currentIndexChanged.connect(self._apply_theme)
        grid.add(labelled_field("Theme", self.theme_combo, metrics=m))

        self.network_combo = QComboBox()
        self.network_combo.addItem("Automatic", "auto")
        self.network_combo.addItem("Always online", "online")
        self.network_combo.addItem("Offline only", "offline")
        grid.add(labelled_field("Network mode", self.network_combo,
                                "'Offline only' blocks every online action.", m))

        self.search_combo = QComboBox()
        for engine_key in ("google", "duckduckgo", "bing"):
            self.search_combo.addItem(engine_key.capitalize(), engine_key)
        grid.add(labelled_field("Search engine", self.search_combo, metrics=m))

        self.provider_label = QLabel()
        self.provider_label.setWordWrap(True)
        self.provider_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        grid.add(labelled_field("AI provider", self.provider_label, metrics=m))

        layout.addWidget(grid)

        save = QPushButton("Save preferences")
        save.setObjectName("Primary")
        save.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        save.clicked.connect(self._save_experience)
        layout.addLayout(self._action_row(save))
        return frame

    def _security_card(self) -> QFrame:
        m = self.metrics
        frame, layout = card(m)
        layout.addWidget(section_title("Security"))
        layout.addWidget(muted(
            "A passphrase is stored only as a salted PBKDF2 hash — never in plain text, "
            "and never written to the log. High-risk actions always ask for confirmation "
            "whether or not a passphrase is set."
        ))

        self.security_status = QLabel()
        self.security_status.setWordWrap(True)
        layout.addWidget(self.security_status)

        # A wrapping row, so the two buttons stack rather than overflow when narrow.
        buttons, row = flow_container(spacing=m.sp(8))
        self.security_button = QPushButton()
        self.security_button.setObjectName("Primary")
        self.security_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.security_button.clicked.connect(self._toggle_security)
        row.addWidget(self.security_button)

        clear_button = QPushButton("Clear conversation history")
        clear_button.setObjectName("Danger")
        clear_button.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        clear_button.clicked.connect(self._clear_history)
        row.addWidget(clear_button)

        reset_all = QPushButton("Reset all settings")
        reset_all.setObjectName("Danger")
        reset_all.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        reset_all.clicked.connect(self._reset_all_settings)
        row.addWidget(reset_all)
        layout.addWidget(buttons)
        layout.addWidget(muted(
            "Resetting settings does not delete your conversation history or "
            "activity log."))
        return frame

    # -------------------------------------------------------------- behaviour

    def refresh(self) -> None:
        config = self._engine.config.config
        self.name_input.setText(config.assistant.name)
        self.wake_input.setText(config.assistant.wake_word)
        self._refresh_wake_status()
        self._select(self.profile_combo, config.assistant.profile_key)
        self._select(self.voice_combo, config.assistant.voice_id)
        self._select(self.language_combos["interface"], config.language.interface)
        self._select(self.language_combos["input"], config.language.speech_input)
        self._select(self.language_combos["output"], config.language.speech_output)
        self._select(self.theme_combo, config.runtime.theme)
        self._select(self.network_combo, config.runtime.network_mode)
        self._select(self.search_combo, config.runtime.search_engine)
        self.provider_label.setText(
            "None configured — AARMA uses its built-in rule engine. "
            "Model providers arrive in Phase 2."
        )
        self._refresh_voice()
        self._refresh_speech_input()
        self._refresh_security()

    # ----------------------------------------------------------- speech input

    def _refresh_speech_input(self) -> None:
        settings = self._engine.config.config.speech_input
        blocked = (self.mic_device_combo, self.mic_model_combo,
                   self.mic_timeout_slider, self.mic_confidence_slider,
                   self.mic_sensitivity_slider)
        for widget in blocked:
            widget.blockSignals(True)
        try:
            # Devices are keyed by name, not index: indexes shift between runs.
            self.mic_device_combo.clear()
            self.mic_device_combo.addItem("Windows default microphone", None)
            for device in self._engine.listening.devices():
                label = device.name
                if device.host_api:
                    label += f"  ·  {device.host_api}"
                self.mic_device_combo.addItem(label, device.name)
            index = self.mic_device_combo.findData(settings.device_name)
            self.mic_device_combo.setCurrentIndex(max(0, index))

            self._select(self.mic_model_combo, settings.model_size)
            self.mic_timeout_slider.setValue(int(round(settings.max_seconds)))
            self.mic_confidence_slider.setValue(
                int(round(settings.confidence_threshold * 100)))
            self.mic_sensitivity_slider.setValue(
                int(round(settings.vad_sensitivity * 100)))
        finally:
            for widget in blocked:
                widget.blockSignals(False)

        self.listen_enabled_toggle.set_checked(settings.enabled)
        self.save_audio_toggle.set_checked(settings.save_debug_audio)
        self._refresh_speech_slider_labels()
        self._refresh_diagnostics()
        self.refresh_microphone_status()

    def _refresh_speech_slider_labels(self) -> None:
        self.mic_timeout_label.setText(f"{self.mic_timeout_slider.value()}s")
        self.mic_confidence_label.setText(f"{self.mic_confidence_slider.value()}%")
        sensitivity = self.mic_sensitivity_slider.value()
        self.mic_sensitivity_label.setText(
            "low" if sensitivity < 34 else "normal" if sensitivity < 67 else "high")

    def refresh_microphone_status(self) -> None:
        """Show whether the microphone is usable, and where audio is processed."""
        readiness = self._engine.listening.readiness()
        language = self._engine.language.info(
            self._engine.config.config.language.speech_input).english_name

        if readiness.ready:
            device = readiness.device_name or "System default"
            state = "Ready" if not readiness.needs_setup else "Ready · one-time setup"
            self.mic_status.setText(f"{device} · {language} · {state}")
            self.mic_detail.setText(readiness.message or
                                    "Press the microphone in the conversation to speak.")
        else:
            self.mic_status.setText(f"{language} · Unavailable")
            detail = readiness.message
            if readiness.suggestion:
                detail += f"  {readiness.suggestion}"
            self.mic_detail.setText(detail)

        self.mic_privacy.setText(readiness.privacy_label)

    def _save_speech_input(self) -> None:
        self._refresh_speech_slider_labels()
        self._engine.config.update_speech_input(
            device_name=self.mic_device_combo.currentData(),
            device_index=None,
            model_size=self.mic_model_combo.currentData(),
            max_seconds=float(self.mic_timeout_slider.value()),
            confidence_threshold=self.mic_confidence_slider.value() / 100,
            vad_sensitivity=self.mic_sensitivity_slider.value() / 100,
        )
        recognizer = self._engine.listening.recognizer
        if hasattr(recognizer, "set_model_size"):
            recognizer.set_model_size(self.mic_model_combo.currentData())
        self.refresh_microphone_status()

    def _on_listen_toggle(self, _key: str, enabled: bool) -> None:
        self._engine.config.update_speech_input(enabled=enabled)
        if not enabled:
            self._engine.listening.cancel()
        self.refresh_microphone_status()

    # ------------------------------------------------------------------ voice

    def _refresh_voice(self) -> None:
        """Reload the voice controls and re-read what is actually available."""
        voice = self._engine.config.config.voice
        blocked = (self.voice_presentation_combo, self.voice_profile_combo,
                   self.voice_style_combo, self.voice_mode_combo,
                   self.voice_rate_slider, self.voice_volume_slider,
                   self.voice_pitch_slider)
        for widget in blocked:
            widget.blockSignals(True)
        try:
            # Read the presentation off the selected voice rather than off the
            # stored gender. They are kept in step now, but the voice is the
            # one the synthesiser actually uses, so it is the honest source for
            # a label that claims to describe what the user will hear.
            selected_voice = self._engine.config.config.assistant.voice_id
            presentation = get_profile(selected_voice).gender
            self._select(self.voice_presentation_combo, presentation)
            self._fill_voice_list(presentation)
            self._select(self.voice_profile_combo, selected_voice)
            self._select(self.voice_style_combo, voice.style)
            self._select(self.voice_mode_combo, voice.provider_mode)
            self.voice_rate_slider.setValue(int(round(voice.rate * 100)))
            self.voice_volume_slider.setValue(int(round(voice.volume * 100)))
            self.voice_pitch_slider.setValue(int(round(voice.pitch)))
        finally:
            for widget in blocked:
                widget.blockSignals(False)

        self.speak_replies_toggle.set_checked(voice.enabled)
        self.speak_notifications_toggle.set_checked(voice.speak_notifications)
        self._refresh_slider_labels()
        self.refresh_voice_status()

    def _refresh_slider_labels(self) -> None:
        self.voice_rate_label.setText(f"{self.voice_rate_slider.value() / 100:.2f}×")
        self.voice_volume_label.setText(f"{self.voice_volume_slider.value()}%")
        pitch = self.voice_pitch_slider.value()
        self.voice_pitch_label.setText("natural" if pitch == 0 else f"{pitch:+d}")

    def refresh_voice_status(self) -> None:
        """Show whether the selected voice is usable, and which one it is."""
        availability = self._engine.speech.availability()
        assistant = self._engine.config.config.assistant.name
        language = self._engine.language.info(
            self._engine.config.config.language.speech_output).english_name

        if availability.available:
            self.voice_status.setText(
                f"{assistant} · {language} · {availability.status_label}")
            detail = f"Using {availability.voice_name}. {availability.quality.description}"
            if not availability.online:
                detail += " Works without an internet connection."
            self.voice_detail.setText(detail)
        else:
            self.voice_status.setText(f"{assistant} · {language} · Unavailable")
            detail = availability.reason
            if availability.suggestion:
                detail += f"  {availability.suggestion}"
            self.voice_detail.setText(detail)

        self.test_voice_button.setEnabled(availability.available)

    def _fill_voice_list(self, presentation: str) -> None:
        """Offer only this profile's voices.

        A list mixing both is how a "Male" profile ends up holding a female
        voice; if the user wants the other one, the profile selector is the
        control that says so.
        """
        self.voice_profile_combo.clear()
        for profile in profiles_for_gender(presentation):
            self.voice_profile_combo.addItem(profile.display_name,
                                             profile.voice_id)

    def _switch_voice_presentation(self) -> None:
        """Load the whole other profile, or say why it cannot be loaded."""
        wanted = self.voice_presentation_combo.currentData()
        changed, explanation = self._engine.config.set_voice_presentation(wanted)
        if not changed and "already" not in explanation:
            # Never leave the control claiming a profile that is not in use.
            self._engine.conversation.say_assistant(explanation, source="gui",
                                                    status="error")
        self.refresh_voice_status()
        self.refresh()

    def _save_voice(self) -> None:
        self._refresh_slider_labels()
        selected = self.voice_profile_combo.currentData()
        if selected:
            # `select_voice` moves the presentation with the voice; writing
            # `voice_id` alone is what left the two disagreeing.
            self._engine.config.select_voice(selected)
        self._engine.config.update_voice(
            style=self.voice_style_combo.currentData(),
            provider_mode=self.voice_mode_combo.currentData(),
            rate=self.voice_rate_slider.value() / 100,
            volume=self.voice_volume_slider.value() / 100,
            pitch=float(self.voice_pitch_slider.value()),
        )
        self.refresh_voice_status()

    def _on_voice_toggle(self, key: str, enabled: bool) -> None:
        if key == "voice_enabled":
            self._engine.config.update_voice(enabled=enabled)
        else:
            self._engine.config.update_voice(speak_notifications=enabled)
        self.refresh_voice_status()

    def _test_voice(self) -> None:
        """Play the short preview (section 10), reporting failure clearly."""
        if not self._engine.speech.speak_preview():
            availability = self._engine.speech.availability()
            QMessageBox.information(
                self, "Test voice",
                availability.reason or "The voice isn't available right now.")

    def _stop_voice(self) -> None:
        self._engine.speech.stop()

    @staticmethod
    def _select(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _refresh_wake_status(self) -> None:
        """Say which of the wake-word states this identity is actually in.

        Ready, enrolled and ready, not yet taught, or a model whose files have
        gone missing -- they need different things from the user, so they are
        not collapsed into one "unavailable". The wording never claims voice
        wake-up works when no detector exists.
        """
        try:
            state = self._engine.wake_identity()
        except Exception as exc:  # noqa: BLE001 - diagnostics must not break settings
            self.wake_status.setText(f"Wake-word status unavailable: {exc}")
            self.enrol_button.setEnabled(False)
            return

        can_enrol = bool(state.get("can_enrol"))
        phrase, name = state["phrase"], state["name"]
        self.enrol_button.setEnabled(can_enrol)
        self.enrol_button.setText(f"Re-teach {name}" if state.get("enrolled")
                                  else f"Teach {name}")

        # Before the session is unlocked, the identity's detector is not the
        # one listening -- so its readiness is not what to report.
        session = self._engine.wake_state
        if session.get("bootstrap_required"):
            self.wake_status.setText(
                '○ Waiting for "Hey Aariz" — say it once to start this '
                "session. Typing works without it.")
            return

        if state["available"]:
            self.wake_status.setText(
                f'● Active — say "{name}" followed by your command. '
                f'Say "go to sleep" to stop listening.')
            return

        # Active, but this identity has nothing that can hear it. Saying
        # "active" alone here would be the claim the user cannot check: they
        # would say the name and nothing would happen.
        self.wake_status.setText(
            f"● Active for typing. {name} has no voice wake word yet, so "
            f'saying "{name}" will not reach me — teach me one below, or '
            "run: python main.py --enrol-wake-word")
        return

        if state["available"] and state.get("enrolled"):
            self.wake_status.setText(
                f'● {name} — say "{phrase}" to wake AARMA. Taught by you '
                f'on {(state.get("enrolled_at") or "")[:10]} from '
                f'{state.get("enrollment_count", 0)} recordings '
                f'(version {state.get("enrolled_version", 1)}).')
        elif state["available"]:
            self.wake_status.setText(
                f'● {name} — say "{phrase}" to wake AARMA.')
        elif can_enrol:
            self.wake_status.setText(
                f"⚠ {name} isn't enrolled yet, so voice wake-up is off. "
                f'Teach me "{phrase}" here, or run '
                "python main.py --enrol-wake-word")
        else:
            self.wake_status.setText(f"⚠ {state['reason']}")

    def _enrol_wake_word(self) -> None:
        from ui.wakeword_enrol import WakeWordEnrolDialog

        dialog = WakeWordEnrolDialog(self._engine, self)
        dialog.enrolled.connect(self.identity_changed.emit)
        dialog.exec()
        self._refresh_wake_status()


    def _enrol_wake_word(self) -> None:
        from ui.wakeword_enrol import WakeWordEnrolDialog

        dialog = WakeWordEnrolDialog(self._engine, self)
        dialog.enrolled.connect(self.identity_changed.emit)
        dialog.exec()
        self._refresh_wake_status()


    def _save_assistant(self) -> None:
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Name required", "The assistant needs a name.")
            return
        profile_key = self.profile_combo.currentData()
        config = self._engine.config
        if profile_key in ASSISTANT_PROFILES and profile_key != config.config.assistant.profile_key:
            from config.settings import AssistantIdentity

            config.update(assistant=AssistantIdentity.from_profile(ASSISTANT_PROFILES[profile_key]))
        config.update_assistant(
            name=name,
            voice_id=self.voice_combo.currentData(),
            wake_word=(normalise_wake_phrase(self.wake_input.text())
                       or wake_phrase_for_name(name)),
            profile_key=self.profile_combo.currentData() or "custom",
        )
        self.refresh()
        self.identity_changed.emit()

    def _save_languages(self) -> None:
        self._engine.config.update_language(
            interface=self.language_combos["interface"].currentData(),
            speech_input=self.language_combos["input"].currentData(),
            speech_output=self.language_combos["output"].currentData(),
        )
        self.identity_changed.emit()

    def _save_experience(self) -> None:
        self._engine.config.update_runtime(
            network_mode=self.network_combo.currentData(),
            search_engine=self.search_combo.currentData(),
            theme=self.theme_combo.currentData(),
        )
        self._engine.connectivity.invalidate()
        self.identity_changed.emit()

    def _apply_theme(self) -> None:
        self.theme_changed.emit(self.theme_combo.currentData())

    def _refresh_security(self) -> None:
        manager = PassphraseManager(self._engine.config)
        if manager.is_enabled:
            self.security_status.setText("Security mode is ON. A passphrase is set.")
            self.security_button.setText("Disable security mode")
        else:
            self.security_status.setText("Security mode is OFF. No passphrase is set.")
            self.security_button.setText("Set a passphrase")

    def _toggle_security(self) -> None:
        manager = self._engine.passphrase
        if manager.is_enabled:
            current, ok = self._prompt_passphrase("Disable security", "Current passphrase:")
            if not ok:
                return
            try:
                manager.disable(current)
            except PassphraseError as exc:
                QMessageBox.warning(self, "Security", str(exc))
        else:
            new, ok = self._prompt_passphrase("Set a passphrase",
                                              "New passphrase (8 characters or more):")
            if not ok:
                return
            try:
                manager.enable(new)
            except PassphraseError as exc:
                QMessageBox.warning(self, "Security", str(exc))
        self._refresh_security()

    def _prompt_passphrase(self, title: str, label: str) -> tuple[str, bool]:
        from PySide6.QtWidgets import QInputDialog

        value, ok = QInputDialog.getText(self, title, label, QLineEdit.Password)
        return value, bool(ok and value)

    def _clear_history(self) -> None:
        confirm = QMessageBox.question(
            self, "Clear history",
            "Delete the stored conversation transcript? The action audit trail is kept.",
        )
        if confirm is QMessageBox.Yes:
            removed = self._engine.database.clear_conversation()
            self._engine.context.clear()
            QMessageBox.information(self, "Clear history", f"Removed {removed} message(s).")


class AboutPage(QWidget):
    """What this build actually contains -- capabilities, paths, phase status."""

    def __init__(self, engine: AarmaEngine, metrics: Metrics) -> None:
        super().__init__()
        self._engine = engine
        m = metrics

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(m.sp(14))

        title = QLabel("About AARMA")
        title.setObjectName("PageTitle")
        outer.addWidget(title)
        outer.addWidget(muted(
            f"Version {APP_VERSION} · {PHASE_LABEL}. "
            "The assistant speaks and listens, with recognition running on this "
            "computer. Wake word, language models and reminders are not "
            "implemented in this build."
        ))

        content = QWidget()
        content.setObjectName("ConversationCanvas")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, m.sp(12), 0)
        layout.setSpacing(m.sp(14))

        # Registered capabilities.
        frame, frame_layout = card(m)
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        frame_layout.addWidget(section_title(f"Registered actions ({len(engine.registry)})"))
        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(("Action", "Category", "Risk", "Requires"))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setShowGrid(False)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        specs = engine.registry.all()
        table.setRowCount(len(specs))
        for row, spec in enumerate(specs):
            requires = ", ".join(p.value for p in spec.permissions) or "—"
            if spec.requires_internet:
                requires += " · internet"
            for column, value in enumerate((spec.name, spec.category, spec.risk_label, requires)):
                table.setItem(row, column, QTableWidgetItem(value))
        header_view = table.horizontalHeader()
        header_view.setSectionResizeMode(QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(3, QHeaderView.Stretch)
        # Tall enough to be worth reading, small enough not to force the window
        # taller than a small screen; the page scrolls either way.
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        table.setMinimumSize(1, m.size(160))
        frame_layout.addWidget(table)
        layout.addWidget(frame, 1)

        # Where things live on disk.
        paths = get_paths()
        frame2, frame2_layout = card(m)
        frame2_layout.addWidget(section_title("Local data"))
        grid = ReflowGrid(m, max_columns=2, min_column_width=280)
        for caption, value in (
            ("Data folder", paths.root),
            ("Configuration", paths.config_file),
            ("Database", paths.database_file),
            ("Log file", paths.log_file),
            ("Screenshots", paths.screenshot_dir),
        ):
            value_label = QLabel(str(value))
            value_label.setObjectName("Muted")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value_label.setWordWrap(True)
            value_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
            grid.add(labelled_field(caption, value_label, metrics=m))
        frame2_layout.addWidget(grid)
        layout.addWidget(frame2)

        layout.addStretch(1)
        outer.addWidget(scroll_wrap(content), 1)


# ------------------------------------------------------------------- main window

class Dashboard(QMainWindow):
    """The application window."""

    submit_text = Signal(str)
    submit_voice = Signal(str)
    submit_confirmation = Signal(str, bool)

    NAV_ITEMS = (
        ("Conversation", "chat"),
        ("Activity", "activity"),
        ("Permissions", "permissions"),
        ("Settings", "settings"),
        ("About", "about"),
    )

    def __init__(self, engine: AarmaEngine) -> None:
        super().__init__()
        self.engine = engine
        self.metrics = Metrics.current()
        self.palette_tokens: Palette = get_palette(engine.config.config.runtime.theme)
        m = self.metrics

        self.setWindowTitle("AARMA")
        self.setWindowIcon(app_icon(self.palette_tokens))

        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        # Stop Qt writing the layout's own minimum onto the window; the minimum
        # is chosen from the screen, and the content scrolls to meet it.
        root_layout.setSizeConstraint(QLayout.SetNoConstraint)

        self.sidebar = self._build_sidebar()
        root_layout.addWidget(self.sidebar)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(m.sp(22), m.sp(22), m.sp(22), m.sp(22))
        content_layout.setSpacing(m.sp(16))
        content_layout.addWidget(self._build_header())

        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.setMinimumSize(1, 1)
        self.chat_page = ChatPage(engine.assistant_name, m)
        self.activity_page = ActivityPage(engine, m)
        self.permissions_page = PermissionsPage(engine, m)
        self.settings_page = SettingsPage(engine, m)
        self.about_page = AboutPage(engine, m)
        for page in (self.chat_page, self.activity_page, self.permissions_page,
                     self.settings_page, self.about_page):
            self.stack.addWidget(page)
        content_layout.addWidget(self.stack, 1)
        root_layout.addWidget(content, 1)

        # --- wiring ---------------------------------------------------------
        self.chat_page.submitted.connect(self._on_submitted)
        self.chat_page.confirmation_decided.connect(self._on_confirmation)
        self.settings_page.identity_changed.connect(self._refresh_identity)
        self.settings_page.theme_changed.connect(self.apply_theme)
        self.permissions_page.permission_changed.connect(self.refresh_microphone_state)
        self.stack.currentChanged.connect(self._on_page_changed)

        self._bridge = ConfigBridge()
        self._bridge.changed.connect(self._on_config_changed)
        engine.config.add_listener(self._bridge.changed.emit)

        self._speech_bridge = SpeechBridge()
        self._speech_bridge.started.connect(self._on_speech_started)
        self._speech_bridge.finished.connect(self._on_speech_finished)
        engine.speech.on_started(self._speech_bridge.started.emit)
        engine.speech.on_finished(self._speech_bridge.finished.emit)

        # The bridge is built here; the scheduler is *not* started here.
        # Constructing a window must not start background work -- a half-built
        # QMainWindow with a live thread pointed at it is how a headless test
        # run turns into an access violation, which is exactly what happened.
        # `main.run_gui` calls `start_reminders()` after `show()`, the same
        # point at which the wake loop is armed.
        # Subscribed here rather than in `start_reminders`, because a turn can
        # arrive before the window is shown and a dropped first message is
        # exactly the class of bug this bridge exists to end.
        self._conversation_bridge = ConversationBridge()
        self._conversation_bridge.event.connect(self._on_conversation_event)
        self._unsubscribe_conversation = engine.conversation.subscribe(
            self._conversation_bridge.event.emit)

        self._reminder_bridge = ReminderBridge()
        self._reminder_bridge.fired.connect(self._on_reminder)

        self._listen_bridge = ListeningBridge()
        self._listen_bridge.state_changed.connect(self._on_listen_state)
        self._listen_bridge.level.connect(self.chat_page.set_microphone_level)
        self._listen_bridge.result.connect(self._on_transcription)
        engine.listening.on_state_changed(self._listen_bridge.state_changed.emit)
        engine.listening.on_level(self._listen_bridge.level.emit)
        engine.listening.on_result(self._listen_bridge.result.emit)
        self.chat_page.microphone_clicked.connect(self._toggle_listening)

        self._thread = QThread(self)
        self._worker = EngineWorker(engine)
        self._worker.moveToThread(self._thread)
        self.submit_text.connect(self._worker.handle_text)
        self.submit_voice.connect(self._worker.handle_voice)
        self.submit_confirmation.connect(self._worker.resolve_confirmation)
        self._worker.finished.connect(self._on_response)
        self._thread.start()

        QShortcut(QKeySequence("Ctrl+L"), self, activated=self._focus_input)
        # Escape silences the assistant. Being able to stop a voice instantly,
        # without hunting for a control, is the difference between a voice
        # people leave on and one they switch off.
        QShortcut(QKeySequence("Esc"), self, activated=self._stop_speaking)

        apply_initial_geometry(self, ideal=IDEAL_SIZE, minimum=MINIMUM_SIZE)
        self.apply_theme(self.palette_tokens.name)
        self._refresh_identity()
        self._apply_breakpoints(self.width())
        self.refresh_microphone_state()
        engine.conversation.say_assistant(engine.start().text, source="gui")
        self.chat_page.input.setFocus()

    # ----------------------------------------------------------------- layout

    def _build_sidebar(self) -> QFrame:
        m = self.metrics
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        # Bounded, then sized from the window in `_apply_breakpoints`. The lower
        # bound is the width of the longest nav label, so no label is ever cut.
        sidebar.setMinimumWidth(m.size(150))
        sidebar.setMaximumWidth(m.size(264))
        sidebar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(m.sp(18), m.sp(22), m.sp(18), m.sp(18))
        layout.setSpacing(m.sp(16))

        brand = QHBoxLayout()
        brand.setSpacing(m.sp(12))
        self.brand_mark = Avatar("A", 38, self.palette_tokens, m)
        brand.addWidget(self.brand_mark)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        self.brand_name = ElidingLabel("AARMA", object_name="BrandName",
                                       letter_spacing=m.sp(2))
        self.brand_tag = ElidingLabel("Desktop AI Agent", object_name="BrandTag")
        brand_text.addWidget(self.brand_name)
        brand_text.addWidget(self.brand_tag)
        brand.addLayout(brand_text, 1)
        layout.addLayout(brand)
        layout.addWidget(divider())

        # The nav list scrolls, so a short window (or a large system font) can
        # never make the sidebar dictate the window's minimum height.
        nav_holder = QWidget()
        nav_holder.setObjectName("ConversationCanvas")
        nav_layout = QVBoxLayout(nav_holder)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(m.sp(6))

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        for index, (label, _key) in enumerate(self.NAV_ITEMS):
            button = QPushButton(label)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setChecked(index == 0)
            button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.nav_group.addButton(button, index)
            nav_layout.addWidget(button)
        self.nav_group.idClicked.connect(self._on_nav)

        nav_layout.addStretch(1)
        layout.addWidget(scroll_wrap(nav_holder), 1)

        layout.addWidget(divider())
        self.sidebar_footer = QLabel()
        self.sidebar_footer.setObjectName("SidebarFooter")
        self.sidebar_footer.setWordWrap(True)
        layout.addWidget(self.sidebar_footer)
        return sidebar

    def _build_header(self) -> QFrame:
        m = self.metrics
        header = QFrame()
        header.setObjectName("HeaderBar")
        header.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(m.sp(20), m.sp(16), m.sp(20), m.sp(16))
        layout.setSpacing(m.sp(16))

        self.avatar = Avatar(self.engine.assistant_name, 52, self.palette_tokens, m)
        layout.addWidget(self.avatar)

        identity = QVBoxLayout()
        identity.setSpacing(m.sp(2))
        self.assistant_label = QLabel(self.engine.assistant_name)
        self.assistant_label.setObjectName("AssistantName")
        self.assistant_tag = QLabel(self.engine.language.t("app.tagline"))
        self.assistant_tag.setObjectName("AssistantTag")
        self.assistant_tag.setWordWrap(True)
        identity.addWidget(self.assistant_label)
        identity.addWidget(self.assistant_tag)
        layout.addLayout(identity, 1)

        # Pills wrap rather than overflow, and the lower-priority ones drop out
        # entirely on a narrow window -- see `_apply_breakpoints`.
        pill_bar, pills = flow_container(spacing=m.sp(8))
        self.pill_state = Pill("Idle")
        self.pill_language = Pill("English")
        self.pill_voice = Pill("Voice: Phase 2")
        self.pill_network = Pill("Offline")
        for pill in (self.pill_state, self.pill_language, self.pill_voice, self.pill_network):
            pills.addWidget(pill)
        pill_bar.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
        layout.addWidget(pill_bar, 0, Qt.AlignVCenter)
        return header

    # ------------------------------------------------------------- responsive

    def compact_breakpoint(self) -> int:
        """Width below which decorative header chrome is dropped."""
        return breakpoint_threshold(BREAKPOINT_COMPACT, MINIMUM_SIZE[0], self.metrics,
                                    headroom=60)

    def narrow_breakpoint(self) -> int:
        """Width below which only essential status remains."""
        return breakpoint_threshold(BREAKPOINT_NARROW, MINIMUM_SIZE[0], self.metrics,
                                    headroom=20)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._apply_breakpoints(self.width())

    def _apply_breakpoints(self, width: int) -> None:
        """Re-shape the interface for the current window width.

        Two adaptations, both about protecting the content column: the sidebar
        takes a share of the window rather than a constant, and the decorative
        header chrome steps aside before it can crowd the assistant's name.
        """
        m = self.metrics
        # Pin both bounds to the computed value. A range alone is not enough:
        # the sidebar's own size hint collapsed once its nav list became
        # scrollable, so the layout parked it at the minimum and it stopped
        # responding to the window at all. The value is still derived from the
        # window on every resize -- pinned, not hardcoded.
        sidebar_width = sidebar_width_for(width, m)
        self.sidebar.setMinimumWidth(sidebar_width)
        self.sidebar.setMaximumWidth(sidebar_width)

        compact = width < self.compact_breakpoint()
        narrow = width < self.narrow_breakpoint()

        self.pill_voice.setVisible(not compact)
        self.pill_language.setVisible(not narrow)
        self.assistant_tag.setVisible(not compact)
        self.brand_tag.setVisible(not narrow)
        self.brand_mark.setVisible(
            self._wordmark_fits_beside_the_mark(sidebar_width) and not narrow
        )

    def _wordmark_fits_beside_the_mark(self, sidebar_width: int) -> bool:
        """Whether "AARMA" still fits once the gradient mark takes its share.

        Measured against the real font rather than assumed from a pixel
        threshold, so the mark yields at exactly the point the wordmark would
        otherwise be truncated -- at any text size.
        """
        m = self.metrics
        chrome = m.sp(18) * 2 + self.brand_mark.sizeHint().width() + m.sp(12)
        return (sidebar_width - chrome) >= self.brand_name.natural_width() + m.sp(4)

    # ---------------------------------------------------------------- actions

    def _focus_input(self) -> None:
        self.nav_group.button(0).setChecked(True)
        self.stack.setCurrentIndex(0)
        self.chat_page.input.setFocus()

    def _on_nav(self, index: int) -> None:
        self.stack.setCurrentIndex(index)

    def _on_page_changed(self, index: int) -> None:
        if self.stack.widget(index) is self.activity_page:
            self.activity_page.refresh()
        elif self.stack.widget(index) is self.permissions_page:
            self.permissions_page.refresh()
        elif self.stack.widget(index) is self.settings_page:
            self.settings_page.refresh()

    @Slot(object)
    def _on_conversation_event(self, event) -> None:
        """Draw one turn. The only place messages enter the transcript.

        Every route publishes to the bus -- typing, the microphone button, a
        wake-word turn, a confirmation answered by button -- so adding another
        route cannot skip the transcript by forgetting to call a renderer.
        """
        if not (event.text or "").strip():
            # Never an empty bubble. A turn with nothing to show is a turn with
            # nothing to say, and a blank message reads as a bug.
            return
        if event.role == "user":
            self.chat_page.add_user_message(
                event.text, spoken=event.spoken, confidence=event.confidence)
            return
        if event.status == "confirmation" and event.confirmation_token:
            self.chat_page.add_confirmation(event.confirmation_token, event.text)
            return
        self.chat_page.add_assistant_message(
            event.text, self.engine.assistant_name,
            tone="error" if event.status == "error" else "normal")

    def _on_submitted(self, text: str, *, spoken: bool = False,
                      confidence: Optional[float] = None) -> None:
        # The transcript is drawn by `_on_conversation_event`; this only puts
        # the window into its working state -- and returns the view to the end
        # of the conversation, since the user has just added to it.
        self.chat_page.follow_tail()
        self.chat_page.set_busy(True)
        self.pill_state.set_state(self.engine.language.t("status.thinking"), highlighted=True)
        if spoken:
            self.submit_voice.emit(text)
        else:
            self.submit_text.emit(text)

    def _on_confirmation(self, token: str, approved: bool) -> None:
        self.chat_page.set_busy(True)
        self.pill_state.set_state(self.engine.language.t("status.working"), highlighted=True)
        self.submit_confirmation.emit(token, approved)

    @Slot(object)
    def _on_response(self, response: EngineResponse) -> None:
        self.chat_page.set_busy(False)
        self.pill_state.set_state(self.engine.language.t("status.idle"))

        # The reply is already on screen: the engine published it the moment it
        # existed, which is what keeps the text from waiting on synthesis.
        # Speaking is all that is left. The engine filters and shortens the
        # text, and returns immediately -- synthesis happens on its own worker.
        self.engine.speech.speak(response.text)

        self._refresh_status_pills()
        if response.should_exit:
            QTimer.singleShot(600, self.close)

    # --------------------------------------------------------------- listening

    def _toggle_listening(self) -> None:
        """Microphone button: start listening, or stop if already listening."""
        listening = self.engine.listening
        if listening.is_listening:
            listening.cancel()
            return

        # Stop talking before listening, or the assistant hears itself.
        self.engine.speech.stop()

        readiness = listening.readiness()
        if not readiness.ready:
            self.engine.conversation.say_assistant(
                f"{readiness.message} {readiness.suggestion}".strip(),
                source="voice", status="error")
            self.refresh_microphone_state()
            return

        if readiness.needs_setup:
            # The one-time model download blocks the first recognition; say so
            # rather than leaving the user watching an idle "Understanding…".
            self.engine.conversation.say_assistant(readiness.message,
                                                   source="voice")

        listening.start_listening()

    def refresh_microphone_state(self) -> None:
        """Re-read readiness and update the composer accordingly."""
        readiness = self.engine.listening.readiness()
        self.chat_page.set_microphone_state(
            self.engine.listening.state, ready=readiness.ready,
            reason=f"{readiness.message} {readiness.suggestion}".strip())

    @Slot(object, str)
    def _on_listen_state(self, state, message: str) -> None:
        from speech.input_engine import ListeningState

        readiness_ready = state not in (ListeningState.DISABLED,
                                        ListeningState.UNAVAILABLE)
        self.chat_page.set_microphone_state(state, message, ready=readiness_ready,
                                            reason=message)
        if state is ListeningState.LISTENING:
            self.pill_state.set_state(self.engine.language.t("status.listening_now"),
                                      highlighted=True)
        elif state is ListeningState.PROCESSING:
            self.pill_state.set_state(self.engine.language.t("status.understanding"),
                                      highlighted=True)
        elif not self.engine.speech.is_speaking:
            self.pill_state.set_state(self.engine.language.t("status.idle"))

    @Slot(object)
    def _on_transcription(self, result) -> None:
        """Route recognised speech through the ordinary command path.

        A successful transcription is handed to `_on_submitted`, which is the
        same method the text box uses — so voice never gets a second execution
        route, and every permission, risk and confirmation check applies
        unchanged.
        """
        from speech.listener import RecognitionOutcome

        self.refresh_microphone_state()

        if result.outcome is RecognitionOutcome.CANCELLED:
            return
        if not result.ok:
            # Nothing recognised: say why, and never invent a command or leave
            # an empty user bubble behind. Published rather than drawn, so this
            # message reaches the transcript the same way every other one does.
            if result.error:
                self.engine.conversation.say_assistant(
                    result.error, source="voice", status="error")
            return

        self._on_submitted(result.text, spoken=True, confidence=result.confidence)

    # ------------------------------------------------------------------ speech

    def _stop_speaking(self) -> None:
        self.engine.speech.stop()
        self.engine.listening.cancel()

    @Slot()
    def _on_speech_started(self) -> None:
        self.pill_state.set_state(self.engine.language.t("status.speaking"),
                                  highlighted=True)

    @Slot(object)
    def _on_speech_finished(self, outcome) -> None:
        self.pill_state.set_state(self.engine.language.t("status.idle"))
        if outcome.interrupted:
            return
        # A failure to speak is worth one quiet line, not a dialog: the reply is
        # already on screen and the user has lost nothing but the audio.
        if not outcome.ok and outcome.error:
            self.engine.conversation.say_assistant(outcome.error, source="voice",
                                                   status="error")
        elif outcome.notice:
            self.settings_page.refresh_voice_status()

    @Slot(object)
    def _on_config_changed(self, _config: AppConfig) -> None:
        """Follow an identity change from anywhere, not just this window.

        A rename can arrive from the settings form, from the setup wizard, or
        from the user simply saying "from now on I'll call you Friday" in the
        chat.  All three land here, so the header, the placeholder and the
        wake-word status all update live rather than waiting for a restart.
        """
        self._refresh_identity()
        try:
            self.settings_page.refresh()
        except Exception:  # noqa: BLE001 - a refresh must not break the window
            logger.debug("Settings page could not refresh after a config change")

    # --------------------------------------------------------------- refresh

    def _refresh_identity(self) -> None:
        name = self.engine.assistant_name
        self.assistant_label.setText(name)
        self.assistant_tag.setText(self.engine.language.t("app.tagline"))
        self.avatar.set_letter(name)
        self.chat_page.set_placeholder(name)
        self.setWindowTitle(f"AARMA · {name}")
        # The full data path is long enough to swamp the sidebar; it lives on
        # the About page, and here it is only a tooltip.
        self.sidebar_footer.setText(f"v{APP_VERSION} · {PHASE_LABEL}")
        self.sidebar_footer.setToolTip(f"Local data: {get_paths().root}")
        self._refresh_status_pills()

    def _refresh_status_pills(self) -> None:
        config = self.engine.config.config
        info = self.engine.language.info()
        self.pill_language.set_state(info.english_name)
        self.pill_voice.set_state(f"Voice: {config.assistant.voice_id}")
        online = self.engine.online
        self.pill_network.set_state(
            self.engine.language.t("status.online" if online else "status.offline"),
            highlighted=online,
        )

    def apply_theme(self, name: str) -> None:
        self.palette_tokens = get_palette(name)
        self.metrics = Metrics.current()
        application = QApplication.instance()
        if application is not None:
            application.setStyleSheet(build_stylesheet(self.palette_tokens, self.metrics))
        self.setWindowIcon(app_icon(self.palette_tokens))
        self.avatar.set_palette_tokens(self.palette_tokens, self.engine.assistant_name)
        self.brand_mark.set_palette_tokens(self.palette_tokens, "A")
        self.engine.config.update_runtime(theme=self.palette_tokens.name)

    # ----------------------------------------------------------- reminders

    def start_reminders(self) -> bool:
        """Begin ringing reminders into this window.

        Called after :meth:`show`, never from ``__init__``: the sink hands
        events to a window, and a window that is still being built is not
        somewhere to put them.
        """
        return self.engine.start_reminders(self._deliver_reminder)

    def _deliver_reminder(self, event) -> bool:
        """The scheduler's sink.  Called on the scheduler thread.

        Only emits.  Returning True says the reminder reached the user, and it
        is honest here because the signal is queued to the GUI thread and will
        be shown -- the window exists for as long as this sink is registered,
        since :meth:`closeEvent` stops the scheduler before tearing it down.
        """
        self._reminder_bridge.fired.emit(event)
        return True

    def _on_reminder(self, event) -> None:
        """Show a due reminder.  Runs on the GUI thread."""
        self.engine.conversation.say_assistant(event.spoken_text(),
                                               source="reminder")
        try:
            if (self.engine.config.config.voice.speak_notifications
                    and not event.missed):
                # Missed reminders are reported, never sounded: an alarm for
                # something hours past reads as an alarm for something now.
                self.engine.speech.speak(event.spoken_text())
        except Exception:  # noqa: BLE001 - a mute speaker is not a lost reminder
            pass

    # -------------------------------------------------------------- shutdown

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Unsubscribe first: a turn published during teardown would otherwise
        # be marshalled to widgets that are on their way out.
        try:
            self._unsubscribe_conversation()
        except Exception:  # noqa: BLE001 - teardown must not raise
            logger.debug("Conversation bridge would not unsubscribe")
        # Stopped before the window goes, so a reminder cannot arrive at a
        # widget that is being torn down.
        self.engine.stop_reminders()
        self._thread.quit()
        self._thread.wait(3000)
        self.engine.shutdown()
        super().closeEvent(event)


def build_pages_index() -> List[str]:
    """Page keys, used by tests to assert the navigation stays in sync."""
    return [key for _label, key in Dashboard.NAV_ITEMS]
