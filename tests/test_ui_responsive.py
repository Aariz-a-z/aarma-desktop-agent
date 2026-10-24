"""Responsive layout, window management and high-DPI behaviour.

Two layers of test, because they answer different questions.

**Pure geometry.** :func:`~ui.responsive.compute_window_placement` is a function
over a rectangle, so every resolution and scaling factor AARMA claims to
support is asserted here directly. No monitor is required, and none is claimed
to have been used -- these simulate the *work areas* those configurations
produce.

**Live widgets.** The wizard and the dashboard are built on Qt's offscreen
platform and driven through real resizes, which is the only way to catch a
minimum size hint that quietly forces a window taller than a small screen. The
central assertion is that the navigation controls stay inside the window at the
smallest supported size, on every page.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Tuple

import pytest

from ui.responsive import (
    MAX_SCALE,
    MIN_SCALE,
    Metrics,
    Rect,
    bubble_width_for,
    compute_window_placement,
    grid_columns_for,
    sidebar_width_for,
)

#: Physical resolutions AARMA is expected to run on.
RESOLUTIONS: Tuple[Tuple[int, int], ...] = (
    (1280, 720),
    (1366, 768),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
)

#: Windows display-scaling factors.
SCALE_FACTORS: Tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0)

#: The Windows taskbar, in logical pixels. It scales with DPI, so its logical
#: height is roughly constant -- which is why it is subtracted after scaling.
TASKBAR_LOGICAL_HEIGHT = 48


def work_area(resolution: Tuple[int, int], scale: float) -> Rect:
    """The logical work area a resolution and scale factor produce."""
    width = int(resolution[0] / scale)
    height = int(resolution[1] / scale) - TASKBAR_LOGICAL_HEIGHT
    return Rect(0, 0, width, max(1, height))


def all_environments() -> List[Tuple[Tuple[int, int], float]]:
    return [(resolution, scale) for resolution in RESOLUTIONS for scale in SCALE_FACTORS]


ENVIRONMENT_IDS = [f"{w}x{h}@{int(s * 100)}%" for (w, h) in RESOLUTIONS for s in SCALE_FACTORS]


# ------------------------------------------------------------ pure geometry

class TestWindowPlacement:
    """The window must fit, be centred, and never depend on one screen size."""

    @pytest.mark.parametrize(("resolution", "scale"), all_environments(), ids=ENVIRONMENT_IDS)
    def test_window_always_fits_inside_the_work_area(self, resolution, scale) -> None:
        area = work_area(resolution, scale)
        placement = compute_window_placement(
            area, ideal=(1240, 820), minimum=(720, 460)
        )
        assert area.contains(placement.as_rect()), (
            f"{placement} escapes the work area {area}"
        )

    @pytest.mark.parametrize(("resolution", "scale"), all_environments(), ids=ENVIRONMENT_IDS)
    def test_window_leaves_room_for_the_frame(self, resolution, scale) -> None:
        """The title bar must not be pushed off the top of the screen."""
        area = work_area(resolution, scale)
        placement = compute_window_placement(area, ideal=(1240, 820), minimum=(720, 460))
        assert placement.height <= area.height - 56 or placement.height == area.height - 56
        assert placement.width <= area.width - 16

    @pytest.mark.parametrize(("resolution", "scale"), all_environments(), ids=ENVIRONMENT_IDS)
    def test_window_is_centred(self, resolution, scale) -> None:
        area = work_area(resolution, scale)
        placement = compute_window_placement(area, ideal=(1240, 820), minimum=(720, 460))
        left_gap = placement.x - area.x
        right_gap = area.right - (placement.x + placement.width)
        assert abs(left_gap - right_gap) <= 1

    @pytest.mark.parametrize(("resolution", "scale"), all_environments(), ids=ENVIRONMENT_IDS)
    def test_window_is_a_reasonable_share_of_the_screen(self, resolution, scale) -> None:
        """Not shrunk to a postage stamp just to be safe."""
        area = work_area(resolution, scale)
        placement = compute_window_placement(area, ideal=(1240, 820), minimum=(720, 460))
        assert placement.width >= min(720, area.width - 16)
        assert placement.height >= min(460, area.height - 56)

    def test_large_screens_do_not_produce_an_absurd_window(self) -> None:
        """A 4K monitor gets a comfortable window, not a full-screen one."""
        area = work_area((3840, 2160), 1.0)
        placement = compute_window_placement(area, ideal=(1240, 820), minimum=(720, 460))
        assert placement.width == 1240
        assert placement.height == 820

    def test_small_screens_use_the_space_available(self) -> None:
        area = work_area((1366, 768), 1.5)   # 910 x 464 logical
        placement = compute_window_placement(area, ideal=(1240, 820), minimum=(720, 460))
        assert placement.width < 1240
        assert area.contains(placement.as_rect())

    def test_work_area_smaller_than_the_minimum_still_fits(self) -> None:
        """The minimum is a preference; the screen is a fact."""
        tiny = Rect(0, 0, 500, 300)
        placement = compute_window_placement(tiny, ideal=(1240, 820), minimum=(720, 460))
        assert tiny.contains(placement.as_rect())
        assert placement.width < 720
        assert placement.height < 460

    def test_placement_respects_a_non_zero_work_area_origin(self) -> None:
        """A taskbar on the left, or a secondary monitor, offsets the origin."""
        area = Rect(1920, 40, 1600, 900)
        placement = compute_window_placement(area, ideal=(1240, 820), minimum=(720, 460))
        assert area.contains(placement.as_rect())
        assert placement.x >= 1920
        assert placement.y >= 40

    @pytest.mark.parametrize(("resolution", "scale"), all_environments(), ids=ENVIRONMENT_IDS)
    def test_wizard_placement_fits_everywhere_too(self, resolution, scale) -> None:
        from ui.setup_wizard import IDEAL_SIZE, MINIMUM_SIZE

        area = work_area(resolution, scale)
        placement = compute_window_placement(area, ideal=IDEAL_SIZE, minimum=MINIMUM_SIZE)
        assert area.contains(placement.as_rect())

    def test_no_two_environments_share_a_hardcoded_size(self) -> None:
        """Proof the size is derived, not constant: small screens differ from large."""
        small = compute_window_placement(work_area((1280, 720), 1.5),
                                         ideal=(1240, 820), minimum=(720, 460))
        large = compute_window_placement(work_area((2560, 1440), 1.0),
                                         ideal=(1240, 820), minimum=(720, 460))
        assert (small.width, small.height) != (large.width, large.height)


# ------------------------------------------------------------------- metrics

class TestMetrics:
    """Sizing follows the user's font, which is how Windows text scaling works."""

    def test_reference_metrics_reproduce_the_original_design(self) -> None:
        m = Metrics()
        assert m.font_pt(12) == pytest.approx(9.0)   # Design body size == 9pt baseline.
        assert m.sp(16) == 16
        assert m.size(40) == 40

    def test_larger_system_font_scales_the_whole_interface(self) -> None:
        m = Metrics(point_size=12.0, scale=1.5)
        assert m.font_pt(12) > Metrics().font_pt(12)
        assert m.sp(16) > Metrics().sp(16)

    def test_scale_is_clamped_to_a_usable_range(self) -> None:
        class _Font:
            def pointSizeF(self):  # noqa: N802 - mimics the Qt API
                return 40.0

        class _FontMetrics:
            def height(self):
                return 400

        assert Metrics.from_font(_Font(), _FontMetrics()).scale == MAX_SCALE

        class _TinyMetrics:
            def height(self):
                return 1

        assert Metrics.from_font(_Font(), _TinyMetrics()).scale == MIN_SCALE

    def test_metrics_fall_back_without_a_running_application(self) -> None:
        assert Metrics.current().scale > 0

    def test_spacing_never_collapses_to_zero(self) -> None:
        assert Metrics(scale=MIN_SCALE).sp(1) >= 1


class TestResponsiveRules:
    def test_sidebar_shrinks_with_the_window_but_keeps_a_floor(self) -> None:
        assert sidebar_width_for(400) == 150          # Floor: fits "Conversation".
        assert sidebar_width_for(760) == 152          # Proportional between the bounds.
        assert sidebar_width_for(1240) > sidebar_width_for(800)
        assert sidebar_width_for(3000) == 264         # Ceiling: never crowds content.

    def test_breakpoints_stay_above_the_smallest_supported_window(self) -> None:
        """A breakpoint below the minimum window width would never fire."""
        from ui.responsive import breakpoint_threshold

        for scale in (MIN_SCALE, 1.0, 1.25, MAX_SCALE):
            metrics = Metrics(point_size=9.0 * scale, scale=scale)
            assert breakpoint_threshold(760, 620, metrics) > 620

    def test_breakpoints_grow_with_the_user_font(self) -> None:
        from ui.responsive import breakpoint_threshold

        small = breakpoint_threshold(900, 720, Metrics())
        large = breakpoint_threshold(900, 720, Metrics(point_size=13.5, scale=1.5))
        assert large > small

    def test_sidebar_floor_grows_with_the_user_font(self) -> None:
        assert sidebar_width_for(760, Metrics(scale=1.5)) > sidebar_width_for(760)

    def test_bubbles_track_the_viewport(self) -> None:
        assert bubble_width_for(400) < bubble_width_for(1000)
        assert bubble_width_for(120) >= 240           # Readable floor.
        assert bubble_width_for(4000) <= 760          # Readable ceiling.

    @pytest.mark.parametrize(("width", "expected"), [
        (200, 1), (300, 1), (500, 2), (900, 2),
    ])
    def test_grids_reflow_to_fewer_columns_when_narrow(self, width, expected) -> None:
        assert grid_columns_for(width, min_column_width=240, max_columns=2) == expected

    def test_grid_never_reports_zero_columns(self) -> None:
        assert grid_columns_for(10, min_column_width=240, max_columns=3) == 1


# ------------------------------------------------------------- live widgets

pytest.importorskip("PySide6", reason="Desktop interface tests need PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialog,
    QPushButton,
    QScrollArea,
    QWidget,
)

#: The smallest window either surface is expected to remain usable at.
SMALLEST_TEST_SIZE = (620, 420)

#: Widths swept during the resize tests, spanning the supported range.
SWEEP_WIDTHS = (620, 760, 900, 1100, 1400, 1920)


@pytest.fixture(scope="session")
def qapp() -> Iterator[QApplication]:
    application = QApplication.instance() or QApplication([])
    yield application


def _is_inside(child: QWidget, window: QWidget) -> bool:
    """True when ``child`` is fully within ``window``'s visible bounds."""
    top_left = child.mapTo(window, QPoint(0, 0))
    return (top_left.x() >= 0 and top_left.y() >= 0
            and top_left.x() + child.width() <= window.width()
            and top_left.y() + child.height() <= window.height())


class TestSetupWizardWindow:
    """The wizard must behave like an ordinary application window."""

    @pytest.fixture
    def wizard(self, qapp, config):
        from ui.setup_wizard import SetupWizard

        widget = SetupWizard(config)
        widget.show()
        qapp.processEvents()
        yield widget
        widget.close()
        widget.deleteLater()
        qapp.processEvents()

    def test_is_not_a_dialog(self, wizard) -> None:
        """QDialog is what withheld the minimize and maximize buttons."""
        assert not isinstance(wizard, QDialog)

    def test_offers_minimize_maximize_and_close(self, wizard) -> None:
        flags = wizard.windowFlags()
        assert flags & Qt.WindowMinimizeButtonHint
        assert flags & Qt.WindowMaximizeButtonHint
        assert flags & Qt.WindowCloseButtonHint
        assert flags & Qt.Window

    def test_can_be_minimized_and_restored(self, wizard, qapp) -> None:
        wizard.showMinimized()
        qapp.processEvents()
        assert wizard.isMinimized()
        wizard.showNormal()
        qapp.processEvents()
        assert not wizard.isMinimized()

    def test_can_be_maximized_and_restored(self, wizard, qapp) -> None:
        wizard.showMaximized()
        qapp.processEvents()
        assert wizard.isMaximized()
        wizard.showNormal()
        qapp.processEvents()
        assert not wizard.isMaximized()

    def test_is_freely_resizable(self, wizard) -> None:
        """No setFixedSize anywhere in the chain."""
        assert wizard.maximumWidth() >= 16_777_215
        assert wizard.maximumHeight() >= 16_777_215

    def test_layout_minimum_fits_a_small_screen(self, wizard) -> None:
        hint = wizard.minimumSizeHint()
        assert hint.width() <= SMALLEST_TEST_SIZE[0]
        assert hint.height() <= SMALLEST_TEST_SIZE[1]

    def test_resizing_to_the_minimum_actually_applies(self, wizard, qapp) -> None:
        """A resize that is silently clamped is a window that cannot fit a screen.

        The step rail used to force its full height onto the window, so asking
        for 430px of height produced 452 and the wizard overflowed short work
        areas. Scrolling the rail removed the floor.
        """
        wizard.resize(*SMALLEST_TEST_SIZE)
        qapp.processEvents()
        assert wizard.width() <= SMALLEST_TEST_SIZE[0]
        assert wizard.height() <= SMALLEST_TEST_SIZE[1]

    def test_fits_the_shortest_supported_work_area(self, wizard, qapp, monkeypatch) -> None:
        """1366x768 at 150% scaling is the tightest realistic Windows laptop.

        Its 464px-tall work area is shorter than the wizard's preferred 420px
        minimum plus a window frame, so the minimum has to yield — a window that
        insists on being taller than the screen is the original bug.
        """
        from ui import responsive
        from ui.setup_wizard import IDEAL_SIZE, MINIMUM_SIZE

        area = work_area((1366, 768), 1.5)
        monkeypatch.setattr(responsive, "work_area_for", lambda widget=None: area)
        placement = responsive.apply_initial_geometry(
            wizard, ideal=IDEAL_SIZE, minimum=MINIMUM_SIZE)
        qapp.processEvents()

        assert area.contains(placement.as_rect())
        assert wizard.minimumHeight() <= area.height - 56
        assert wizard.height() <= area.height - 56
        assert _is_inside(wizard.next_button, wizard)

    def test_every_page_is_scrollable(self, wizard) -> None:
        from ui.setup_wizard import STEPS

        assert wizard.stack.count() == len(STEPS)
        for index in range(wizard.stack.count()):
            assert isinstance(wizard.stack.widget(index), QScrollArea)

    def test_all_eight_steps_are_preserved(self, wizard) -> None:
        from ui.setup_wizard import STEPS

        assert STEPS == ("Welcome", "Assistant", "Language", "Voice",
                         "Name", "Permissions", "Security", "Finish")

    @pytest.mark.parametrize("page", range(8))
    def test_navigation_stays_on_screen_at_the_smallest_size(
        self, wizard, qapp, page: int
    ) -> None:
        """The failure this whole change exists to prevent."""
        wizard.resize(*SMALLEST_TEST_SIZE)
        wizard.stack.setCurrentIndex(page)
        wizard._update_navigation()
        qapp.processEvents()

        for button in (wizard.back_button, wizard.next_button):
            assert button.isVisible()
            assert _is_inside(button, wizard), (
                f"{button.text()} is clipped on step {page + 1}"
            )

    @pytest.mark.parametrize("width", SWEEP_WIDTHS)
    def test_navigation_survives_a_width_sweep(self, wizard, qapp, width: int) -> None:
        wizard.resize(width, 460)
        qapp.processEvents()
        assert _is_inside(wizard.next_button, wizard)

    @pytest.mark.parametrize("width", SWEEP_WIDTHS)
    def test_brand_wordmark_is_never_truncated(self, wizard, qapp, width: int) -> None:
        wizard.resize(width, 600)
        qapp.processEvents()
        if wizard.rail.isVisible():
            assert wizard.brand_name.text() == "AARMA", f"truncated at {width}px"

    @pytest.mark.parametrize("width", (800, 1000, 1400, 1920))
    def test_rail_width_tracks_the_window(self, wizard, qapp, width: int) -> None:
        wizard.resize(width, 600)
        qapp.processEvents()
        if wizard.rail.isVisible():
            # Against the achieved width, not the requested one: a resize can be
            # clamped by the screen, and that is correct behaviour.
            assert wizard.rail.width() == sidebar_width_for(wizard.width(), wizard.metrics)

    def test_content_scrolls_rather_than_forcing_the_window_taller(
        self, wizard, qapp
    ) -> None:
        wizard.stack.setCurrentIndex(5)          # Permissions: the tallest page.
        wizard.resize(*SMALLEST_TEST_SIZE)
        qapp.processEvents()
        area = wizard.stack.currentWidget()
        assert isinstance(area, QScrollArea)
        assert area.widget().sizeHint().height() > area.viewport().height()
        assert area.verticalScrollBar().maximum() > 0

    def test_rail_hides_on_a_narrow_window(self, wizard, qapp) -> None:
        threshold = wizard.rail_breakpoint()
        wizard.resize(threshold + 120, 600)
        qapp.processEvents()
        assert wizard.rail.isVisible()

        wizard.resize(max(SMALLEST_TEST_SIZE[0], threshold - 60), 600)
        qapp.processEvents()
        assert not wizard.rail.isVisible()
        # Progress is still shown, just in the content column.
        assert "Step" in wizard.step_caption.text()

    def test_closing_reports_completion_once(self, qapp, config) -> None:
        from ui.setup_wizard import SetupWizard

        widget = SetupWizard(config)
        seen: List[bool] = []
        widget.finished.connect(seen.append)
        widget.show()
        qapp.processEvents()
        widget.close()
        qapp.processEvents()
        assert seen == [False]        # Closed without finishing; emitted exactly once.
        assert not config.config.first_run_completed


class TestDashboardWindow:
    @pytest.fixture
    def dashboard(self, qapp, engine):
        from ui.dashboard import Dashboard

        window = Dashboard(engine)
        window.show()
        qapp.processEvents()
        yield window
        window.close()
        qapp.processEvents()

    def test_is_freely_resizable(self, dashboard) -> None:
        assert dashboard.maximumWidth() >= 16_777_215
        assert dashboard.maximumHeight() >= 16_777_215

    def test_can_be_maximized_and_restored(self, dashboard, qapp) -> None:
        dashboard.showMaximized()
        qapp.processEvents()
        assert dashboard.isMaximized()
        dashboard.showNormal()
        qapp.processEvents()
        assert not dashboard.isMaximized()

    def test_layout_minimum_fits_a_small_screen(self, dashboard) -> None:
        hint = dashboard.minimumSizeHint()
        assert hint.width() <= 720
        assert hint.height() <= 460

    def test_resizing_to_the_minimum_actually_applies(self, dashboard, qapp) -> None:
        dashboard.resize(720, 460)
        qapp.processEvents()
        assert dashboard.width() <= 720
        assert dashboard.height() <= 460

    def test_fits_the_shortest_supported_work_area(self, dashboard, qapp, monkeypatch) -> None:
        from ui import responsive
        from ui.dashboard import IDEAL_SIZE, MINIMUM_SIZE

        area = work_area((1366, 768), 1.5)
        monkeypatch.setattr(responsive, "work_area_for", lambda widget=None: area)
        placement = responsive.apply_initial_geometry(
            dashboard, ideal=IDEAL_SIZE, minimum=MINIMUM_SIZE)
        qapp.processEvents()

        assert area.contains(placement.as_rect())
        assert dashboard.minimumHeight() <= area.height - 56
        assert dashboard.height() <= area.height - 56
        assert _is_inside(dashboard.chat_page.send_button, dashboard)

    @pytest.mark.parametrize("width", SWEEP_WIDTHS)
    def test_composer_stays_reachable_across_widths(self, dashboard, qapp, width) -> None:
        dashboard.resize(width, 460)
        qapp.processEvents()
        assert _is_inside(dashboard.chat_page.send_button, dashboard)
        assert _is_inside(dashboard.chat_page.input, dashboard)

    @pytest.mark.parametrize("page", range(5))
    def test_every_page_survives_the_smallest_size(self, dashboard, qapp, page) -> None:
        dashboard.stack.setCurrentIndex(page)
        dashboard.resize(720, 460)
        qapp.processEvents()
        widget = dashboard.stack.widget(page)
        assert widget.width() > 0 and widget.height() > 0

    def test_sidebar_adapts_to_the_window(self, dashboard, qapp) -> None:
        """Asserts the *rendered* width: bounds alone left it parked at its floor."""
        dashboard.resize(1400, 800)
        qapp.processEvents()
        wide = dashboard.sidebar.width()

        dashboard.resize(760, 500)
        qapp.processEvents()
        narrow = dashboard.sidebar.width()

        assert narrow < wide
        assert narrow == sidebar_width_for(dashboard.width(), dashboard.metrics)

    def test_header_sheds_chrome_before_it_crowds(self, dashboard, qapp) -> None:
        threshold = dashboard.compact_breakpoint()
        dashboard.resize(threshold + 200, 800)
        qapp.processEvents()
        assert dashboard.pill_voice.isVisible()

        dashboard.resize(max(720, threshold - 60), 500)
        qapp.processEvents()
        assert not dashboard.pill_voice.isVisible()
        assert dashboard.pill_network.isVisible()   # Connectivity is never hidden.

    @pytest.mark.parametrize("width", SWEEP_WIDTHS)
    def test_brand_wordmark_is_never_truncated(self, dashboard, qapp, width) -> None:
        """"AARMA" rendered as "AARM" was the visible symptom of a rigid sidebar.

        The gradient mark yields whenever it would cost the wordmark its last
        letters, so the brand reads in full at every supported window width.
        """
        dashboard.resize(width, 600)
        qapp.processEvents()
        label = dashboard.brand_name
        assert label.full_text() == "AARMA"
        assert label.text() == "AARMA", f"wordmark truncated at {width}px"

    def test_chat_bubbles_reflow_when_the_window_resizes(self, dashboard, qapp) -> None:
        dashboard.resize(1400, 800)
        qapp.processEvents()
        dashboard.chat_page.add_assistant_message("A reasonably long message.", "Asma")
        qapp.processEvents()
        wide = dashboard.chat_page._bubbles[-1].maximumWidth()

        dashboard.resize(760, 500)
        qapp.processEvents()
        narrow = dashboard.chat_page._bubbles[-1].maximumWidth()

        assert narrow < wide, "existing bubbles must reflow, not just new ones"

    def test_starter_prompts_wrap_instead_of_overflowing(self, dashboard, qapp) -> None:
        dashboard.resize(1400, 800)
        qapp.processEvents()
        wide_height = dashboard.chat_page.suggestion_bar.height()

        dashboard.resize(720, 500)
        qapp.processEvents()
        narrow_height = dashboard.chat_page.suggestion_bar.height()

        assert narrow_height >= wide_height


class TestGeneratedArtIsDpiAware:
    def test_avatar_is_rendered_at_the_device_pixel_ratio(self, qapp) -> None:
        """A logical-size pixmap upscaled by Qt is what makes a mark look soft."""
        from ui.theme import DARK, gradient_avatar

        standard = gradient_avatar("A", 64, DARK, 1.0)
        retina = gradient_avatar("A", 64, DARK, 2.0)

        assert standard.width() == 64
        assert retina.width() == 128           # Twice the pixels...
        assert retina.devicePixelRatio() == 2.0
        assert retina.deviceIndependentSize().width() == 64   # ...same logical size.

    def test_icon_covers_the_sizes_windows_asks_for(self, qapp) -> None:
        from ui.theme import DARK, app_icon

        sizes = {size.width() for size in app_icon(DARK).availableSizes()}
        assert {16, 32, 48, 256}.issubset(sizes)


class TestStylesheetScalesWithTheUserFont:
    def test_font_sizes_are_points_not_pixels(self, qapp) -> None:
        """px ignores the Windows text-size setting; pt follows it."""
        from ui.theme import DARK, build_stylesheet

        sheet = build_stylesheet(DARK, Metrics())
        assert "pt;" in sheet
        assert "font-size: 14px" not in sheet

    def test_larger_user_font_produces_a_larger_stylesheet_scale(self, qapp) -> None:
        from ui.theme import DARK, build_stylesheet

        normal = build_stylesheet(DARK, Metrics(point_size=9.0, scale=1.0))
        large = build_stylesheet(DARK, Metrics(point_size=13.5, scale=1.5))
        assert normal != large

    def test_stylesheet_has_no_leftover_hardcoded_font_pixels(self, qapp) -> None:
        import re

        from ui.theme import DARK, build_stylesheet

        sheet = build_stylesheet(DARK, Metrics())
        assert not re.search(r"font-size:\s*\d+px", sheet)


class TestFlowLayout:
    def test_wraps_children_onto_further_rows_as_width_shrinks(self, qapp) -> None:
        from ui.responsive import flow_layout

        host = QWidget()
        layout = flow_layout(host, spacing=8)
        for index in range(6):
            button = QPushButton(f"Chip number {index}")
            button.setFixedSize(140, 30)
            layout.addWidget(button)

        assert layout.heightForWidth(1200) < layout.heightForWidth(300)

    def test_minimum_width_is_one_child_not_all_of_them(self, qapp) -> None:
        from ui.responsive import flow_layout

        host = QWidget()
        layout = flow_layout(host, spacing=8)
        for _ in range(6):
            button = QPushButton("Chip")
            button.setFixedSize(140, 30)
            layout.addWidget(button)

        assert layout.minimumSize().width() < 6 * 140

    def test_a_row_that_fits_exactly_does_not_wrap(self, qapp) -> None:
        """Off-by-one at the right edge cost the header its last status pill."""
        from ui.responsive import flow_layout

        host = QWidget()
        layout = flow_layout(host, spacing=8)
        for _ in range(3):
            button = QPushButton("Chip")
            button.setFixedSize(100, 30)
            layout.addWidget(button)

        exact = layout.sizeHint().width()          # 3 * 100 + 2 * 8
        assert layout.heightForWidth(exact) == layout.heightForWidth(exact + 200)

    def test_preferred_width_is_a_single_row(self, qapp) -> None:
        """Reporting the minimum as the hint makes rows wrap with space to spare."""
        from ui.responsive import flow_layout

        host = QWidget()
        layout = flow_layout(host, spacing=8)
        for _ in range(4):
            button = QPushButton("Chip")
            button.setFixedSize(140, 30)
            layout.addWidget(button)

        assert layout.sizeHint().width() >= 4 * 140
        assert layout.sizeHint().width() > layout.minimumSize().width()
