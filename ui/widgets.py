"""Reusable presentation widgets.

Kept separate from :mod:`ui.dashboard` so the window file stays about *layout*
and the pieces here stay about *appearance*. None of these widgets know
anything about the engine; they take plain strings and emit plain signals.

Sizing rule for everything in this module: a dimension is either derived from
:class:`~ui.responsive.Metrics` (so it follows the user's text size) or it is a
*bound* rather than a fixed value. The only genuinely fixed dimensions left are
the ones where a fixed value is the point — a circular avatar has to be square,
and a hairline divider has to be one pixel.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.responsive import Metrics
from ui.theme import Palette, avatar_pixmap_for


def divider() -> QFrame:
    """A one-pixel rule. Genuinely fixed: a hairline is a hairline at any DPI."""
    line = QFrame()
    line.setObjectName("Divider")
    line.setFrameShape(QFrame.HLine)
    line.setFixedHeight(1)
    line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return line


def section_title(text: str) -> QLabel:
    label = QLabel(text.upper())
    label.setObjectName("SectionTitle")
    label.setWordWrap(True)
    return label


def muted(text: str, *, wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Muted")
    label.setWordWrap(wrap)
    if wrap:
        # Wrapped text must be allowed to trade width for height, otherwise the
        # label demands its full single-line width and forces the window wider.
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    return label


def card(metrics: Optional[Metrics] = None, *, spacing: int = 14,
         margins: Tuple[int, int, int, int] = (20, 18, 20, 18)) -> Tuple[QFrame, QVBoxLayout]:
    """A rounded surface panel plus its vertical layout, spaced by the user's text size."""
    m = metrics or Metrics.current()
    frame = QFrame()
    frame.setObjectName("Card")
    frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(*(m.sp(value) for value in margins))
    layout.setSpacing(m.sp(spacing))
    return frame, layout


class ElidingLabel(QLabel):
    """A single-line label that shortens with an ellipsis rather than clipping.

    A plain ``QLabel`` given less width than its text simply cuts mid-glyph —
    which is how a sidebar brand ends up reading "AARM". Eliding degrades
    legibly instead, and the full text stays available as a tooltip. The
    ``Ignored`` horizontal policy is what lets the label shrink below its own
    text width in the first place.
    """

    def __init__(self, text: str = "", *, object_name: str = "",
                 letter_spacing: float = 0.0) -> None:
        super().__init__()
        if object_name:
            self.setObjectName(object_name)
        self._full_text = text
        # Letter spacing is applied to the *font* rather than through the
        # stylesheet, because QSS letter-spacing is invisible to QFontMetrics --
        # measuring one thing while rendering another is how a label ends up
        # elided even though the caller calculated that it would fit.
        self._letter_spacing = letter_spacing
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._apply_letter_spacing()
        self._refresh()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        from PySide6.QtCore import QEvent

        super().changeEvent(event)
        if event.type() == QEvent.FontChange:
            # The stylesheet sets the font size after construction; re-apply the
            # spacing and re-measure whenever that happens.
            self._apply_letter_spacing()
            self._refresh()

    def _apply_letter_spacing(self) -> None:
        from PySide6.QtGui import QFont

        if not self._letter_spacing:
            return
        font = self.font()
        if (font.letterSpacingType() != QFont.AbsoluteSpacing
                or font.letterSpacing() != self._letter_spacing):
            font.setLetterSpacing(QFont.AbsoluteSpacing, self._letter_spacing)
            self.setFont(font)   # Re-enters changeEvent once, then the guard holds.

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self._full_text = text
        self._refresh()

    def full_text(self) -> str:
        return self._full_text

    def natural_width(self) -> int:
        """Width the untruncated text needs, for callers deciding what to drop.

        Measured rather than estimated, so the answer stays correct at any font
        size or text-scaling setting.
        """
        from PySide6.QtGui import QFontMetrics

        return QFontMetrics(self.font()).horizontalAdvance(self._full_text)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        from PySide6.QtGui import QFontMetrics

        metrics = QFontMetrics(self.font())
        available = max(0, self.width())
        elided = metrics.elidedText(self._full_text, Qt.ElideRight, available)
        QLabel.setText(self, elided if available else self._full_text)
        self.setToolTip("" if elided == self._full_text else self._full_text)


class Pill(QLabel):
    """A small status chip, e.g. "Online" or "English"."""

    def __init__(self, text: str = "", *, highlighted: bool = False) -> None:
        super().__init__(text)
        self.setObjectName("PillOnline" if highlighted else "Pill")
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

    def set_state(self, text: str, *, highlighted: bool = False) -> None:
        self.setText(text)
        name = "PillOnline" if highlighted else "Pill"
        if self.objectName() != name:
            self.setObjectName(name)
            # Object-name changes need a style refresh to repaint.
            self.style().unpolish(self)
            self.style().polish(self)


class Avatar(QLabel):
    """The circular gradient mark that stands in for the assistant.

    Square by definition, so its size *is* fixed — but the value is derived from
    the user's text size rather than hardcoded, and the pixmap is repainted at
    the device pixel ratio of whichever screen the widget is on. Dragging the
    window between a 100% and a 200% monitor re-renders it sharp.
    """

    def __init__(self, letter: str, design_size: int, palette: Palette,
                 metrics: Optional[Metrics] = None) -> None:
        super().__init__()
        self._metrics = metrics or Metrics.current()
        self._design_size = design_size
        self._palette = palette
        self._letter = letter or "A"
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._rescale()

    @property
    def _pixel_size(self) -> int:
        return self._metrics.size(self._design_size)

    def _rescale(self) -> None:
        size = self._pixel_size
        self.setFixedSize(size, size)
        self._repaint()

    def _repaint(self) -> None:
        self.setPixmap(avatar_pixmap_for(self, self._letter, self._pixel_size, self._palette))

    def set_letter(self, letter: str) -> None:
        self._letter = letter or "A"
        self._repaint()

    def set_palette_tokens(self, palette: Palette, letter: str) -> None:
        self._palette = palette
        self._letter = letter or "A"
        self._repaint()

    def set_metrics(self, metrics: Metrics) -> None:
        self._metrics = metrics
        self._rescale()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # The real screen (and so the real pixel ratio) is only known once the
        # widget has been shown; repaint then so the first frame is not blurry.
        super().showEvent(event)
        self._repaint()


class MessageBubble(QFrame):
    """One conversation turn.

    Carries no width of its own: :class:`~ui.dashboard.ChatPage` sets a maximum
    derived from the current viewport, so bubbles reflow when the window is
    resized rather than staying at a width chosen for someone else's monitor.
    """

    def __init__(self, text: str, *, role: str = "assistant", meta: str = "",
                 tone: str = "normal", metrics: Optional[Metrics] = None) -> None:
        super().__init__()
        m = metrics or Metrics.current()
        object_name = {
            ("assistant", "error"): "BubbleError",
            ("assistant", "confirm"): "BubbleConfirm",
        }.get((role, tone), "BubbleUser" if role == "user" else "BubbleAssistant")
        self.setObjectName(object_name)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(m.sp(16), m.sp(12), m.sp(16), m.sp(12))
        layout.setSpacing(m.sp(5))

        if meta:
            header = QLabel(meta.upper())
            header.setObjectName("BubbleMeta")
            layout.addWidget(header)

        body = QLabel(text)
        body.setObjectName("BubbleText")
        body.setWordWrap(True)
        body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(body)
        self._body = body


class ConfirmationCard(QFrame):
    """A high-risk action waiting for approval, with the buttons attached.

    The buttons sit in a layout that wraps to a column when the card is too
    narrow for a row, so Yes and Cancel stay reachable however small the window
    gets. A confirmation control that scrolls out of reach would be the worst
    possible thing to clip.
    """

    decided = Signal(str, bool)  # token, approved

    def __init__(self, token: str, text: str, metrics: Optional[Metrics] = None) -> None:
        super().__init__()
        m = metrics or Metrics.current()
        self.setObjectName("BubbleConfirm")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self._token = token
        self._metrics = m

        layout = QVBoxLayout(self)
        layout.setContentsMargins(m.sp(16), m.sp(12), m.sp(16), m.sp(12))
        layout.setSpacing(m.sp(10))

        header = QLabel("CONFIRMATION REQUIRED")
        header.setObjectName("BubbleMeta")
        layout.addWidget(header)

        body = QLabel(text)
        body.setObjectName("BubbleText")
        body.setWordWrap(True)
        body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout.addWidget(body)

        self._button_row = QHBoxLayout()
        self._button_row.setSpacing(m.sp(8))
        approve = QPushButton("Yes, do it")
        approve.setObjectName("Primary")
        decline = QPushButton("Cancel")
        decline.setObjectName("Ghost")
        self._button_row.addWidget(approve)
        self._button_row.addWidget(decline)
        self._button_row.addStretch(1)
        layout.addLayout(self._button_row)

        self._buttons = (approve, decline)
        approve.clicked.connect(lambda: self._decide(True))
        decline.clicked.connect(lambda: self._decide(False))

    def _decide(self, approved: bool) -> None:
        for button in self._buttons:
            button.setEnabled(False)
        self._buttons[0].setText("Approved" if approved else "Yes, do it")
        self._buttons[1].setText("Cancelled" if not approved else "Cancel")
        self.decided.emit(self._token, approved)

    def disable(self) -> None:
        for button in self._buttons:
            button.setEnabled(False)


class ListeningIndicator(QWidget):
    """Shows that the microphone is actually capturing, and how loudly.

    Hidden whenever capture is inactive. That is the point: an indicator that
    lingers implies the microphone is still open, which is exactly the thing a
    voice assistant must never suggest falsely (specification sections 2 and 20).

    The level is a single smoothed bar rather than a spectrum -- enough to show
    the microphone is hearing something, not a distraction.
    """

    def __init__(self, metrics: Optional[Metrics] = None) -> None:
        super().__init__()
        m = metrics or Metrics.current()
        self._metrics = m
        self._level = 0.0
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(m.sp(4), m.sp(2), m.sp(4), m.sp(2))
        layout.setSpacing(m.sp(10))

        self.caption = QLabel("Listening…")
        self.caption.setObjectName("ListeningCaption")
        layout.addWidget(self.caption)

        self.meter = QProgressBar()
        self.meter.setObjectName("LevelMeter")
        self.meter.setRange(0, 100)
        self.meter.setValue(0)
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(m.size(6))
        self.meter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.meter, 1)

        self.hint = QLabel("")
        self.hint.setObjectName("Muted")
        layout.addWidget(self.hint)

        self.setVisible(False)

    def show_state(self, caption: str, *, hint: str = "", metering: bool = True) -> None:
        self.caption.setText(caption)
        self.hint.setText(hint)
        self.meter.setVisible(metering)
        if not metering:
            self.set_level(0.0)
        self.setVisible(True)

    def set_level(self, level: float) -> None:
        """Update the meter from an RMS value, smoothed and compressed.

        Speech RMS spans a wide, quiet range; a linear mapping barely moves.
        The square root spreads the useful part across the bar.
        """
        normalised = max(0.0, min(1.0, (level / 0.15) ** 0.5))
        # Ease towards the new value so the bar does not flicker per block.
        self._level = self._level * 0.5 + normalised * 0.5
        self.meter.setValue(int(self._level * 100))

    def stop(self) -> None:
        self._level = 0.0
        self.meter.setValue(0)
        self.setVisible(False)


class ToggleRow(QWidget):
    """A labelled on/off switch with an explanatory line underneath."""

    toggled = Signal(str, bool)

    def __init__(self, key: str, title: str, description: str, checked: bool,
                 metrics: Optional[Metrics] = None) -> None:
        super().__init__()
        m = metrics or Metrics.current()
        self._key = key
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, m.sp(6), 0, m.sp(6))
        layout.setSpacing(m.sp(14))

        text_column = QVBoxLayout()
        text_column.setSpacing(m.sp(2))
        heading = QLabel(title)
        heading.setObjectName("ToggleTitle")
        heading.setWordWrap(True)
        text_column.addWidget(heading)
        text_column.addWidget(muted(description))
        layout.addLayout(text_column, 1)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(checked)
        self.checkbox.setCursor(Qt.PointingHandCursor)
        self.checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.checkbox.toggled.connect(lambda state: self.toggled.emit(self._key, state))
        layout.addWidget(self.checkbox, 0, Qt.AlignVCenter)

    def set_checked(self, checked: bool) -> None:
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(checked)
        self.checkbox.blockSignals(False)


def labelled_field(label_text: str, widget: QWidget, hint: Optional[str] = None,
                   metrics: Optional[Metrics] = None) -> QWidget:
    """A settings row: caption above the control, optional hint below.

    Stacked rather than side-by-side so the row narrows gracefully instead of
    forcing a minimum width that a small window cannot satisfy.
    """
    m = metrics or Metrics.current()
    container = QWidget()
    container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(m.sp(6))

    caption = QLabel(label_text)
    caption.setObjectName("FieldCaption")
    caption.setWordWrap(True)
    layout.addWidget(caption)
    layout.addWidget(widget)
    if hint:
        layout.addWidget(muted(hint))
    return container
