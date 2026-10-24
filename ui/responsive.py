"""The sizing foundation every AARMA screen is built on.

AARMA is meant to be downloaded and installed by people whose machines look
nothing like a developer's. Three variables move independently and this module
is where each one is handled once, rather than being guessed at per screen:

**Screen size.** Window sizes are *derived from* the work area, never chosen
and then clamped against it. :func:`compute_window_placement` is a pure
function over a rectangle, so the full matrix of resolutions and scale factors
can be tested without owning the monitors.

**Display scaling (DPI).** Qt 6 already maps logical pixels to physical pixels,
so layout code stays in logical units. What Qt does *not* do automatically is
render manually-painted pixmaps at the device pixel ratio —
:func:`device_pixel_ratio_for` and the painters in :mod:`ui.theme` handle that,
which is the difference between a crisp avatar and a blurry one at 150%.

**Font scaling.** Windows' "make text bigger" changes the system font size but
not the DPI. A layout measured in fixed pixels ignores it and starts clipping.
:class:`Metrics` derives every font size and every spacing value from the
*user's own font*, so the whole interface grows with their preference.

The design was drawn against a 9pt Segoe UI baseline; :meth:`Metrics.font_pt`
and :meth:`Metrics.sp` translate those original design pixels into values that
track whatever the user actually has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

#: The baseline the visual design was drawn against: Segoe UI 9pt at 96 DPI,
#: whose QFontMetrics height is 16 logical pixels and whose "1.0" body size is
#: 12 pixels. These are reference constants for a *ratio*, not target sizes.
REFERENCE_POINT_SIZE = 9.0
REFERENCE_FONT_HEIGHT = 16.0
REFERENCE_DESIGN_PX = 12.0

#: Bounds on how far font scaling is allowed to stretch the layout. Below the
#: floor the interface would stop being readable; above the ceiling a single
#: screen could no longer hold a usable amount of content.
MIN_SCALE = 0.80
MAX_SCALE = 2.00


def _clamp(value: float, lowest: float, highest: float) -> float:
    return max(lowest, min(highest, value))


@dataclass(frozen=True)
class Rect:
    """A plain rectangle, so placement logic is testable without a Qt screen."""

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_qrect(cls, qrect) -> "Rect":
        return cls(qrect.x(), qrect.y(), qrect.width(), qrect.height())

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def contains(self, other: "Rect") -> bool:
        return (other.x >= self.x and other.y >= self.y
                and other.right <= self.right and other.bottom <= self.bottom)


# ------------------------------------------------------------------- metrics

@dataclass(frozen=True)
class Metrics:
    """Font-derived sizing. One instance drives the whole stylesheet."""

    point_size: float = REFERENCE_POINT_SIZE
    scale: float = 1.0

    @classmethod
    def from_font(cls, font, font_metrics=None) -> "Metrics":
        """Build metrics from a font and, if available, its real metrics.

        ``QFontMetrics`` height is preferred over the nominal point size
        because it reflects the font that was actually resolved, not the one
        that was asked for.
        """
        point_size = float(font.pointSizeF()) if font.pointSizeF() > 0 else REFERENCE_POINT_SIZE
        if font_metrics is not None and font_metrics.height() > 0:
            scale = font_metrics.height() / REFERENCE_FONT_HEIGHT
        else:
            scale = point_size / REFERENCE_POINT_SIZE
        return cls(point_size=point_size, scale=_clamp(scale, MIN_SCALE, MAX_SCALE))

    @classmethod
    def current(cls) -> "Metrics":
        """Metrics for the running application, or the reference if there is none."""
        try:
            from PySide6.QtGui import QFontMetrics
            from PySide6.QtWidgets import QApplication
        except ImportError:  # pragma: no cover - PySide6 is optional
            return cls()
        application = QApplication.instance()
        if application is None:
            return cls()
        font = application.font()
        return cls.from_font(font, QFontMetrics(font))

    # -------------------------------------------------------------- scaling

    def font_pt(self, design_px: float) -> float:
        """Convert a design pixel size into points that follow the user's font.

        Points rather than pixels: a QSS ``px`` value is fixed regardless of the
        user's text-size preference, whereas ``pt`` tracks it. At the 9pt / 96
        DPI baseline the two are identical, so the design is unchanged for a
        default configuration and grows for anyone who needs it to.
        """
        return round(self.point_size * design_px / REFERENCE_DESIGN_PX, 2)

    def sp(self, design_px: float) -> int:
        """Scale a spacing, padding or radius value by the font scale."""
        return max(1, round(design_px * self.scale)) if design_px else 0

    def size(self, design_px: float) -> int:
        """Scale a widget dimension (avatar diameter, icon button, and so on)."""
        return max(1, round(design_px * self.scale))


# --------------------------------------------------------------- window size

@dataclass(frozen=True)
class WindowPlacement:
    """Where and how large a top-level window should open."""

    x: int
    y: int
    width: int
    height: int

    def as_rect(self) -> Rect:
        return Rect(self.x, self.y, self.width, self.height)


#: Space reserved for the native window frame. ``resize()`` sets the *client*
#: area, and Windows draws a title bar and borders outside it; without this
#: reservation a window sized to the full work area has its title bar pushed
#: off-screen.
DEFAULT_FRAME_ALLOWANCE = (16, 56)


def compute_window_placement(
    work_area: Rect,
    *,
    ideal: Tuple[int, int],
    minimum: Tuple[int, int],
    coverage: Tuple[float, float] = (0.82, 0.86),
    frame_allowance: Tuple[int, int] = DEFAULT_FRAME_ALLOWANCE,
) -> WindowPlacement:
    """Choose an initial window rectangle for a given work area.

    ``work_area`` is in logical pixels and already excludes the taskbar, so the
    same call behaves correctly whether the taskbar is at the bottom, the side,
    hidden, or on a secondary monitor.

    The size is the smaller of *what the layout wants* (``ideal``) and *a
    comfortable share of what exists* (``coverage``), never allowed below
    ``minimum`` unless the screen itself is smaller — in which case the window
    fills the work area rather than hanging off it. The result is then centred.

    Pure and Qt-free, so every resolution and scaling combination AARMA claims
    to support can be asserted in the test suite.
    """
    frame_width, frame_height = frame_allowance
    usable_width = max(1, work_area.width - frame_width)
    usable_height = max(1, work_area.height - frame_height)

    target_width = min(ideal[0], round(work_area.width * coverage[0]))
    target_height = min(ideal[1], round(work_area.height * coverage[1]))

    # Grow to the minimum where there is room, but never past the work area.
    width = min(max(target_width, minimum[0]), usable_width)
    height = min(max(target_height, minimum[1]), usable_height)

    x = work_area.x + max(0, (work_area.width - width) // 2)
    y = work_area.y + max(0, (work_area.height - height) // 2)
    return WindowPlacement(x=x, y=y, width=width, height=height)


def work_area_for(widget=None) -> Rect:
    """The available work area for ``widget``'s screen, taskbar already excluded."""
    try:
        from PySide6.QtGui import QGuiApplication
    except ImportError:  # pragma: no cover - PySide6 is optional
        return Rect(0, 0, 1024, 768)

    screen = None
    if widget is not None:
        screen = widget.screen()
    if screen is None:
        screen = QGuiApplication.primaryScreen()
    if screen is None:  # pragma: no cover - headless
        return Rect(0, 0, 1024, 768)
    return Rect.from_qrect(screen.availableGeometry())


def apply_initial_geometry(
    widget,
    *,
    ideal: Tuple[int, int],
    minimum: Tuple[int, int],
    coverage: Tuple[float, float] = (0.82, 0.86),
) -> WindowPlacement:
    """Size and centre a top-level window against its own screen's work area.

    ``minimum`` is applied as the widget's real minimum size only when the
    screen can accommodate it. On a work area smaller than the minimum, forcing
    it would make the window larger than the screen and push its controls out
    of reach — so the minimum is relaxed and the content scrolls instead.
    """
    area = work_area_for(widget)
    placement = compute_window_placement(
        area, ideal=ideal, minimum=minimum, coverage=coverage
    )
    widget.setMinimumSize(min(minimum[0], placement.width),
                          min(minimum[1], placement.height))
    widget.resize(placement.width, placement.height)
    widget.move(placement.x, placement.y)
    return placement


def device_pixel_ratio_for(widget=None) -> float:
    """Device pixel ratio for ``widget``'s screen, for painting crisp pixmaps."""
    try:
        from PySide6.QtGui import QGuiApplication
    except ImportError:  # pragma: no cover
        return 1.0
    if widget is not None:
        try:
            ratio = float(widget.devicePixelRatioF())
            if ratio > 0:
                return ratio
        except (AttributeError, RuntimeError):
            pass
    screen = QGuiApplication.primaryScreen()
    return float(screen.devicePixelRatio()) if screen is not None else 1.0


# ------------------------------------------------------------------ scrolling
#
# Qt's default: a QComboBox, QSlider or QSpinBox under the pointer *accepts*
# the wheel event and changes its own value.  Inside a scroll area that is two
# faults at once -- the setting moves, and because the event was accepted it
# never reaches the viewport, so the page does not scroll either.  Measured on
# this build: one notch over a combo moved it from index 0 to 1 and scrolled
# the page by zero pixels.
#
# That behaviour makes sense for a spin box sitting alone on a form.  It is
# wrong for a long settings page, where the wheel means "move the page" every
# single time and a value change is never what was intended.

#: Control types that eat wheel events by default and must not, in a page that
#: scrolls.  Checkboxes and buttons are deliberately absent: they already
#: ignore the wheel, so there is nothing to fix and a filter would be noise.
_WHEEL_HUNGRY = ("QComboBox", "QSlider", "QAbstractSpinBox", "QDial")


class _WheelToScrollArea:
    """Event filter: give a hovered control's wheel event to the page instead.

    Unconditional rather than "only when the control lacks focus", and that is
    deliberate.  Focus follows tabbing and the first focusable widget on a
    page, so a focus test still lets an arbitrary control eat the scroll -- it
    was exactly that half-measure which left a focused combo box changing
    language while the page stayed put.  Scrolling is never a value-change
    gesture here, whatever has focus.

    Keyboard and mouse behaviour are untouched: arrow keys, clicking, dragging
    a slider and opening a dropdown all still work, because none of them is a
    wheel event.  An open dropdown's popup is a separate window and is not
    filtered, so scrolling a long language list still works once it is open.
    """

    def __init__(self) -> None:
        # Imported lazily: ui.responsive is imported by headless code paths
        # that must not require Qt.
        from PySide6.QtCore import QObject

        class _Filter(QObject):
            def eventFilter(self, watched, event):  # noqa: N802 - Qt naming
                from PySide6.QtCore import QEvent
                from PySide6.QtWidgets import QApplication, QScrollArea

                if event.type() != QEvent.Type.Wheel:
                    return False
                parent = watched.parentWidget()
                while parent is not None:
                    if isinstance(parent, QScrollArea):
                        QApplication.sendEvent(parent.viewport(), event)
                        break
                    parent = parent.parentWidget()
                # Consumed either way: a control outside a scroll area has no
                # page to scroll, and silently changing it would still be wrong.
                return True

        self._filter = _Filter()

    @property
    def filter(self):
        return self._filter


#: One filter for the whole application.  Event filters are stateless here, and
#: a per-control instance would be a needless object per widget.
_WHEEL_FILTER = None


def _wheel_filter():
    global _WHEEL_FILTER
    if _WHEEL_FILTER is None:
        _WHEEL_FILTER = _WheelToScrollArea()
    return _WHEEL_FILTER.filter


def guard_wheel(root) -> int:
    """Stop every wheel-hungry control under ``root`` from eating the scroll.

    Applied to the content of a scroll area rather than to each control at its
    construction site, so a control added to a page later is covered without
    anyone having to remember.  Returns how many were guarded, which is what
    the regression test asserts on.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget

    guarded = 0
    for child in root.findChildren(QWidget):
        if not any(child.inherits(name) for name in _WHEEL_HUNGRY):
            continue
        # WheelFocus would additionally let a passing wheel *take* focus, so a
        # scroll could leave a control focused and armed for the next one.
        if child.focusPolicy() == Qt.WheelFocus:
            child.setFocusPolicy(Qt.StrongFocus)
        child.installEventFilter(_wheel_filter())
        guarded += 1
    return guarded


def scroll_wrap(content, *, object_name: str = "ResponsiveScroll"):
    """Put ``content`` in a vertical scroll area that can shrink to nothing.

    This is the mechanism behind the rule that controls are never clipped. A
    page keeps its natural size when there is room and scrolls when there is
    not, instead of forcing the window taller than the screen.

    The explicit tiny minimum matters: a ``QScrollArea`` otherwise inherits its
    content's minimum size hint, which would defeat the entire purpose by
    propagating the content's height back up to the window.

    Every scrollable page in AARMA is built through here, which is why the
    wheel guard is applied here too: one place, and no page can forget it.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QScrollArea, QSizePolicy

    area = QScrollArea()
    area.setObjectName(object_name)
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    area.setMinimumSize(1, 1)
    area.setWidget(content)
    guard_wheel(content)
    return area


def make_scrollable_page(build_content, *, object_name: str = "ResponsiveScroll"):
    """Build a page widget and return it already wrapped in a scroll area."""
    content = build_content()
    content.setObjectName(content.objectName() or "ConversationCanvas")
    return scroll_wrap(content, object_name=object_name)


# ----------------------------------------------------------- responsive rules

#: Window widths at which the interface changes shape. Expressed in logical
#: pixels because that is the unit layouts are written in; DPI is Qt's problem,
#: not the layout's.
BREAKPOINT_COMPACT = 900     # Below this, non-essential header chrome is dropped.
BREAKPOINT_NARROW = 780      # Below this, the sidebar shrinks to its floor.

#: Sidebar width is a share of the window, bounded so it can neither crowd out
#: the content nor become too narrow for its longest label ("Conversation").
SIDEBAR_WIDTH_RATIO = 0.20
SIDEBAR_MIN_WIDTH = 150
SIDEBAR_MAX_WIDTH = 264


def breakpoint_threshold(design_width: int, minimum_window_width: int,
                         metrics: Optional[Metrics] = None, *, headroom: int = 40) -> int:
    """A breakpoint that scales with the user's font but stays reachable.

    Breakpoints have to grow with the text size — a larger font needs more width
    before the full layout still fits. But scaling alone is not enough: a user
    whose font is *smaller* than the reference would push the threshold below
    the smallest window the app supports, and the adaptation would never fire at
    all. Anchoring the floor above the minimum window width keeps every
    breakpoint live at every text size.
    """
    m = metrics or Metrics.current()
    return max(m.size(design_width), minimum_window_width + m.sp(headroom))


def sidebar_width_for(window_width: int, metrics: Optional[Metrics] = None) -> int:
    """Sidebar width for a given window width, scaled by the user's font."""
    scale = metrics.scale if metrics else 1.0
    lowest = round(SIDEBAR_MIN_WIDTH * scale)
    highest = round(SIDEBAR_MAX_WIDTH * scale)
    return int(_clamp(round(window_width * SIDEBAR_WIDTH_RATIO), lowest, highest))


def grid_columns_for(available_width: int, *, min_column_width: int,
                     max_columns: int) -> int:
    """How many columns fit in ``available_width`` without crushing any of them."""
    if min_column_width <= 0:
        return max_columns
    fits = int(available_width // min_column_width)
    return max(1, min(max_columns, fits))


def bubble_width_for(viewport_width: int, metrics: Optional[Metrics] = None) -> int:
    """Maximum chat-bubble width: a share of the viewport, with a floor.

    Long messages stay readable on a wide window instead of stretching into
    unreadable full-width lines, and stay usable on a narrow one instead of
    being squeezed into a column two words wide.
    """
    scale = metrics.scale if metrics else 1.0
    floor = round(240 * scale)
    ceiling = round(760 * scale)
    return int(_clamp(round(viewport_width * 0.82), floor, max(floor, ceiling)))


# ---------------------------------------------------------------- flow layout

def _flow_layout_class():
    """Build the FlowLayout class lazily so this module imports without Qt."""
    from PySide6.QtCore import QPoint, QRect, QSize, Qt
    from PySide6.QtWidgets import QLayout

    class FlowLayout(QLayout):
        """A layout that wraps its children onto new rows as width shrinks.

        Qt ships no flow layout, and a ``QHBoxLayout`` of chips simply overflows
        its container when the window narrows — which is clipping by another
        name. This wraps instead, so the starter prompts and any future chip row
        stay fully visible at every window size.
        """

        def __init__(self, parent=None, spacing: int = 8) -> None:
            super().__init__(parent)
            self._items = []
            self._spacing = spacing
            self.setContentsMargins(0, 0, 0, 0)

        # -- QLayout plumbing ------------------------------------------------
        def addItem(self, item) -> None:  # noqa: N802 - Qt naming
            self._items.append(item)

        def count(self) -> int:
            return len(self._items)

        def itemAt(self, index):  # noqa: N802 - Qt naming
            return self._items[index] if 0 <= index < len(self._items) else None

        def takeAt(self, index):  # noqa: N802 - Qt naming
            return self._items.pop(index) if 0 <= index < len(self._items) else None

        def expandingDirections(self):  # noqa: N802 - Qt naming
            return Qt.Orientations(0)

        def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt naming
            return True

        def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt naming
            return self._arrange(QRect(0, 0, width, 0), apply=False)

        def setGeometry(self, rect) -> None:  # noqa: N802 - Qt naming
            super().setGeometry(rect)
            self._arrange(rect, apply=True)

        def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
            # *Preferred* is everything on one row. Returning the minimum here
            # instead tells the parent layout one child's width is enough, and
            # the row wraps even when there is ample space for it.
            width = 0
            height = 0
            for index, item in enumerate(self._items):
                hint = item.sizeHint()
                width += hint.width() + (self._spacing if index else 0)
                height = max(height, hint.height())
            margins = self.contentsMargins()
            return QSize(width + margins.left() + margins.right(),
                         height + margins.top() + margins.bottom())

        def minimumSize(self) -> QSize:  # noqa: N802 - Qt naming
            # The widest single child, not the sum: the row is allowed to wrap,
            # so it can shrink to whatever its largest element needs.
            size = QSize(0, 0)
            for item in self._items:
                size = size.expandedTo(item.minimumSize())
            margins = self.contentsMargins()
            return size + QSize(margins.left() + margins.right(),
                                margins.top() + margins.bottom())

        # -- arrangement -----------------------------------------------------
        def _arrange(self, rect, *, apply: bool) -> int:
            margins = self.contentsMargins()
            effective = rect.adjusted(margins.left(), margins.top(),
                                      -margins.right(), -margins.bottom())
            x, y, row_height = effective.x(), effective.y(), 0
            # `QRect.right()` is x + width - 1, so comparing against it wraps a
            # row that fits exactly. Use the exclusive edge instead.
            right_edge = effective.x() + effective.width()

            for item in self._items:
                hint = item.sizeHint()
                next_x = x + hint.width() + self._spacing
                if next_x - self._spacing > right_edge and row_height > 0:
                    x = effective.x()
                    y += row_height + self._spacing
                    next_x = x + hint.width() + self._spacing
                    row_height = 0
                if apply:
                    item.setGeometry(QRect(QPoint(x, y), hint))
                x = next_x
                row_height = max(row_height, hint.height())

            return y + row_height - rect.y() + margins.bottom()

    return FlowLayout


def flow_layout(parent=None, spacing: int = 8):
    """Create a wrapping row layout."""
    return _flow_layout_class()(parent, spacing)


def flow_container(spacing: int = 8):
    """A widget hosting a :func:`flow_layout`, wired for height-for-width.

    Qt only consults a layout's ``heightForWidth`` if the *widget's* size policy
    says it should. Without that flag the parent layout negotiates using a
    single-row height while allocating a narrower width, and the row wraps while
    there is still space for it. Returns ``(widget, layout)``.
    """
    from PySide6.QtWidgets import QSizePolicy, QWidget

    host = QWidget()
    policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    policy.setHeightForWidth(True)
    host.setSizePolicy(policy)
    layout = flow_layout(host, spacing)
    return host, layout
