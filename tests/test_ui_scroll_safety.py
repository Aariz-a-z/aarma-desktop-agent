"""Scrolling a settings page must scroll it, and change nothing.

The bug: on a long settings page, scrolling with the pointer over a dropdown or
a slider changed that control instead of moving the page.  Qt's default is that
``QComboBox``, ``QSlider`` and ``QAbstractSpinBox`` *accept* wheel events, which
fails twice over -- the setting moves, and because the event was accepted it
never reaches the scroll area's viewport, so the page stays where it is.

Measured on this build before the fix: one wheel notch over a combo box moved
it from index 0 to index 1 and scrolled the page by zero pixels.

Every test here sends real ``QWheelEvent`` objects at real widgets on Qt's
offscreen platform.  None of them inspects source: the bug was in behaviour
Qt supplies by default, so only behaviour can prove it is gone.

The rule being defended, in the user's words:

    SCROLL = SCROLL, HOVER = NO CHANGE, CLICK = ACTIVATE,
    KEYBOARD = CHANGE ONLY WHEN FOCUSED/INTENDED
"""

from __future__ import annotations

import os
from typing import Iterator, List

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ui.responsive import guard_wheel, scroll_wrap  # noqa: E402

#: One wheel notch, downward, as Qt reports it.
NOTCH = -120


@pytest.fixture(scope="session")
def qapp() -> Iterator[QApplication]:
    application = QApplication.instance() or QApplication([])
    yield application


def wheel_at(widget, delta: int = NOTCH) -> QWheelEvent:
    """A wheel event aimed at the centre of ``widget``."""
    centre = QPointF(widget.rect().center())
    return QWheelEvent(centre,
                       QPointF(widget.mapToGlobal(widget.rect().center())),
                       QPoint(0, 0), QPoint(0, delta),
                       Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)


def scroll_over(widget, delta: int = NOTCH) -> None:
    QApplication.sendEvent(widget, wheel_at(widget, delta))
    QApplication.processEvents()


class SettingsLike:
    """A page shaped like AARMA's settings: tall, scrollable, mixed controls."""

    def __init__(self, *, guarded: bool = True) -> None:
        self.content = QWidget()
        layout = QVBoxLayout(self.content)

        for index in range(25):
            layout.addWidget(QLabel(f"filler {index}"))

        self.combo = QComboBox()
        self.combo.addItems(["English", "Hindi", "Urdu", "Arabic"])
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.spin = QSpinBox()
        self.spin.setRange(0, 100)
        self.spin.setValue(50)
        self.checkbox = QCheckBox("Speak replies")
        self.button = QPushButton("Save")

        self.toggles: List[bool] = []
        self.clicks: List[int] = []
        self.checkbox.toggled.connect(self.toggles.append)
        self.button.clicked.connect(lambda: self.clicks.append(1))

        for control in (self.combo, self.slider, self.spin, self.checkbox,
                        self.button):
            layout.addWidget(control)
        for index in range(25):
            layout.addWidget(QLabel(f"tail {index}"))

        if guarded:
            self.area = scroll_wrap(self.content)
        else:
            # The unguarded arrangement, to prove the tests detect the bug.
            from PySide6.QtWidgets import QScrollArea

            self.area = QScrollArea()
            self.area.setWidgetResizable(True)
            self.area.setWidget(self.content)

        self.area.resize(420, 300)
        self.area.show()
        QApplication.processEvents()
        self.bar = self.area.verticalScrollBar()
        self.bar.setValue(0)
        QApplication.processEvents()

    def close(self) -> None:
        self.area.hide()
        self.area.deleteLater()
        QApplication.processEvents()


@pytest.fixture
def page(qapp) -> Iterator[SettingsLike]:
    built = SettingsLike(guarded=True)
    yield built
    built.close()


# ------------------------------------------------- the bug is really a bug

class TestTheBugExists:
    """Without the guard, the tests below would pass vacuously.

    A regression test that cannot fail on the broken code is not a regression
    test, so the broken arrangement is exercised here directly.
    """

    def test_unguarded_combo_changes_and_page_does_not_move(self, qapp) -> None:
        broken = SettingsLike(guarded=False)
        try:
            before_index, before_scroll = broken.combo.currentIndex(), broken.bar.value()
            scroll_over(broken.combo)
            assert broken.combo.currentIndex() != before_index, (
                "Qt no longer eats the wheel here; this test needs revisiting")
            assert broken.bar.value() == before_scroll
        finally:
            broken.close()

    def test_unguarded_slider_moves(self, qapp) -> None:
        broken = SettingsLike(guarded=False)
        try:
            before = broken.slider.value()
            scroll_over(broken.slider)
            assert broken.slider.value() != before
        finally:
            broken.close()


# ------------------------------------------------------- SCROLL = SCROLL

class TestScrollingScrollsAndChangesNothing:

    def test_scroll_over_empty_space_scrolls(self, page) -> None:
        before = page.bar.value()
        scroll_over(page.area.viewport())
        assert page.bar.value() > before

    def test_scroll_over_a_dropdown_does_not_change_it(self, page) -> None:
        before_index = page.combo.currentIndex()
        before_scroll = page.bar.value()
        scroll_over(page.combo)
        assert page.combo.currentIndex() == before_index
        assert page.bar.value() > before_scroll, "the page should have scrolled"

    def test_scroll_over_a_slider_does_not_change_it(self, page) -> None:
        before_value = page.slider.value()
        before_scroll = page.bar.value()
        scroll_over(page.slider)
        assert page.slider.value() == before_value
        assert page.bar.value() > before_scroll

    def test_scroll_over_a_spin_box_does_not_change_it(self, page) -> None:
        before_value = page.spin.value()
        before_scroll = page.bar.value()
        scroll_over(page.spin)
        assert page.spin.value() == before_value
        assert page.bar.value() > before_scroll

    def test_scroll_over_a_toggle_does_not_toggle_it(self, page) -> None:
        before = page.checkbox.isChecked()
        scroll_over(page.checkbox)
        assert page.checkbox.isChecked() == before
        assert page.toggles == [], "toggled was emitted by a scroll"

    def test_scroll_over_a_button_does_not_click_it(self, page) -> None:
        scroll_over(page.button)
        assert page.clicks == [], "clicked was emitted by a scroll"

    def test_a_long_scroll_changes_nothing_anywhere(self, page) -> None:
        """The real gesture: several notches straight down the page."""
        snapshot = (page.combo.currentIndex(), page.slider.value(),
                    page.spin.value(), page.checkbox.isChecked())
        for control in (page.combo, page.slider, page.spin, page.checkbox,
                        page.button, page.area.viewport()):
            for _ in range(3):
                scroll_over(control)
        assert (page.combo.currentIndex(), page.slider.value(),
                page.spin.value(), page.checkbox.isChecked()) == snapshot
        assert page.toggles == [] and page.clicks == []
        assert page.bar.value() > 0

    def test_scrolling_up_also_changes_nothing(self, page) -> None:
        page.bar.setValue(page.bar.maximum())
        QApplication.processEvents()
        snapshot = (page.combo.currentIndex(), page.slider.value())
        scroll_over(page.combo, delta=+120)
        scroll_over(page.slider, delta=+120)
        assert (page.combo.currentIndex(), page.slider.value()) == snapshot


# ----------------------------------------- focus must not re-open the hole

class TestFocusDoesNotReopenTheBug:
    """A focus-conditional guard was tried first and was not enough.

    Focus follows tabbing, and the first focusable widget on a page has it from
    the moment the page appears -- so "block the wheel unless focused" left a
    focused combo box still eating the scroll and still changing language.
    """

    def test_a_focused_dropdown_still_does_not_change_on_scroll(self, page) -> None:
        page.combo.setFocus()
        QApplication.processEvents()
        assert page.combo.hasFocus()

        before = page.combo.currentIndex()
        scroll_over(page.combo)
        assert page.combo.currentIndex() == before

    def test_a_focused_slider_still_does_not_change_on_scroll(self, page) -> None:
        page.slider.setFocus()
        QApplication.processEvents()
        before = page.slider.value()
        scroll_over(page.slider)
        assert page.slider.value() == before

    def test_scrolling_does_not_steal_focus(self, page) -> None:
        """WheelFocus would let a passing scroll arm the next one."""
        assert page.combo.focusPolicy() != Qt.WheelFocus
        assert page.spin.focusPolicy() != Qt.WheelFocus


# -------------------------------- CLICK = ACTIVATE, KEYBOARD still works

class TestNormalInteractionStillWorks:
    """The guard must not have made the controls read-only."""

    def test_a_toggle_still_toggles_when_clicked(self, page) -> None:
        page.checkbox.click()
        QApplication.processEvents()
        assert page.checkbox.isChecked() is True
        assert page.toggles == [True]

    def test_a_button_still_fires_when_clicked(self, page) -> None:
        page.button.click()
        QApplication.processEvents()
        assert page.clicks == [1]

    def test_a_dropdown_can_still_be_set(self, page) -> None:
        page.combo.setCurrentIndex(2)
        QApplication.processEvents()
        assert page.combo.currentIndex() == 2

    def test_a_slider_can_still_be_set(self, page) -> None:
        page.slider.setValue(80)
        QApplication.processEvents()
        assert page.slider.value() == 80

    def test_keyboard_still_changes_a_focused_slider(self, page) -> None:
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        page.slider.setFocus()
        QApplication.processEvents()
        before = page.slider.value()
        press = QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.NoModifier)
        QApplication.sendEvent(page.slider, press)
        QApplication.processEvents()
        assert page.slider.value() > before, (
            "the guard blocked the keyboard, which it must not")

    def test_keyboard_still_changes_a_focused_spin_box(self, page) -> None:
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        page.spin.setFocus()
        QApplication.processEvents()
        before = page.spin.value()
        QApplication.sendEvent(page.spin,
                               QKeyEvent(QEvent.KeyPress, Qt.Key_Up, Qt.NoModifier))
        QApplication.processEvents()
        assert page.spin.value() > before


# ------------------------------------------------- applied where it matters

class TestTheGuardIsActuallyInstalled:

    def test_scroll_wrap_guards_what_it_wraps(self, qapp) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        combo = QComboBox()
        combo.addItems(["a", "b", "c"])
        layout.addWidget(combo)
        area = scroll_wrap(content)
        try:
            assert combo.focusPolicy() == Qt.StrongFocus
        finally:
            area.deleteLater()
            QApplication.processEvents()

    def test_guard_wheel_reports_what_it_guarded(self, qapp) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        for widget in (QComboBox(), QSlider(Qt.Horizontal), QSpinBox(),
                       QCheckBox(), QPushButton("x"), QLabel("y")):
            layout.addWidget(widget)
        # Three wheel-hungry controls; the checkbox, button and label are not.
        assert guard_wheel(content) == 3

    def test_the_real_settings_page_is_covered(self, qapp, engine) -> None:
        """Built for real, then every dropdown and slider on it checked."""
        from ui.dashboard import SettingsPage
        from ui.responsive import Metrics

        page = SettingsPage(engine, Metrics.current())
        try:
            controls = [child for child in page.findChildren(QWidget)
                        if child.inherits("QComboBox") or child.inherits("QSlider")
                        or child.inherits("QAbstractSpinBox")]
            assert controls, "no controls found; the page did not build"
            for control in controls:
                assert control.focusPolicy() != Qt.WheelFocus, (
                    f"{control.objectName() or type(control).__name__} can still "
                    "take focus from a passing wheel")
            # And the behaviour, on the real page's own widgets.
            for control in controls:
                if control.inherits("QComboBox"):
                    before = control.currentIndex()
                    scroll_over(control)
                    assert control.currentIndex() == before
                elif control.inherits("QSlider"):
                    before = control.value()
                    scroll_over(control)
                    assert control.value() == before
        finally:
            page.deleteLater()
            QApplication.processEvents()
