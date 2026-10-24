"""Design tokens, the generated Qt stylesheet, and DPI-aware generated art.

Two rules hold the visual system together:

* **No widget names a colour.** Every colour comes from a :class:`Palette`, so
  the dark/light switch in Settings is a palette swap rather than a sweep
  through the UI code.
* **No widget names a size.** Every font size and every spacing value is
  derived from :class:`~ui.responsive.Metrics`, so the interface grows with the
  user's Windows text-size setting instead of clipping against it.

Font sizes are emitted in **points**, not pixels. A QSS ``px`` is a fixed
logical pixel and ignores the user's text-size preference entirely; ``pt``
follows it. At the 9pt / 96 DPI baseline the design was drawn against, the two
are identical — so a default configuration looks exactly as before, and a
configuration with larger text actually gets larger text.

Generated art (the window icon and the assistant avatar) is painted at the
screen's device pixel ratio. Painting at logical size and letting Qt upscale is
what makes an icon look soft on a high-DPI display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QLinearGradient,
    QPainter,
    QPixmap,
)

from ui.responsive import Metrics, device_pixel_ratio_for


@dataclass(frozen=True)
class Palette:
    """One complete colour scheme."""

    name: str
    base: str            # Window background, furthest back.
    surface: str         # Cards and panels.
    surface_alt: str     # Raised elements inside a card.
    sidebar: str
    border: str
    border_strong: str
    text: str
    text_muted: str
    text_faint: str
    accent: str
    accent_alt: str      # Second gradient stop.
    accent_soft: str     # Tinted background for accented chips.
    success: str
    warning: str
    danger: str
    bubble_user: str
    bubble_assistant: str


DARK = Palette(
    name="dark",
    base="#0B0F14",
    surface="#121822",
    surface_alt="#18202C",
    sidebar="#0E141C",
    border="#212C3B",
    border_strong="#2E3C50",
    text="#E7EEF7",
    text_muted="#93A4B8",
    text_faint="#63748A",
    accent="#4CC2FF",
    accent_alt="#7C6BFF",
    accent_soft="#16324A",
    success="#3ED598",
    warning="#FFB454",
    danger="#FF6B6B",
    bubble_user="#1D3A52",
    bubble_assistant="#18202C",
)

LIGHT = Palette(
    name="light",
    base="#F4F7FB",
    surface="#FFFFFF",
    surface_alt="#F0F4F9",
    sidebar="#FFFFFF",
    border="#DFE7F0",
    border_strong="#C6D3E2",
    text="#101A26",
    text_muted="#556579",
    text_faint="#8494A8",
    accent="#0B7FC7",
    accent_alt="#6154E0",
    accent_soft="#E2F1FC",
    success="#127A52",
    warning="#A65B00",
    danger="#C33A3A",
    bubble_user="#DEEEFB",
    bubble_assistant="#F0F4F9",
)

PALETTES: Dict[str, Palette] = {DARK.name: DARK, LIGHT.name: LIGHT}


def get_palette(name: str) -> Palette:
    return PALETTES.get((name or "dark").strip().lower(), DARK)


# --------------------------------------------------------------------- fonts

def ui_font_family() -> str:
    """Pick the best available modern UI face without shipping a font file."""
    try:
        available = set(QFontDatabase.families())
    except RuntimeError:  # pragma: no cover - no QApplication yet
        return "Segoe UI"
    for candidate in ("Segoe UI Variable Text", "Segoe UI", "Inter", "SF Pro Text",
                      "Helvetica Neue", "Noto Sans"):
        if candidate in available:
            return candidate
    return QFont().defaultFamily()


# ---------------------------------------------------------------- stylesheet

def build_stylesheet(palette: Palette, metrics: Optional[Metrics] = None) -> str:
    """Render the application stylesheet for ``palette`` at the user's text size.

    ``m.font_pt(n)`` converts a design pixel into points; ``m.sp(n)`` scales a
    spacing value. Nothing below is a bare constant.
    """
    m = metrics or Metrics.current()
    font = ui_font_family()

    return f"""
    * {{
        font-family: "{font}";
        outline: none;
    }}

    QWidget#Root {{
        background-color: {palette.base};
    }}

    QWidget {{
        color: {palette.text};
        font-size: {m.font_pt(14)}pt;
    }}

    /* ---------------------------------------------------------- sidebar */

    QFrame#Sidebar {{
        background-color: {palette.sidebar};
        border-right: 1px solid {palette.border};
    }}

    /* Letter spacing for the wordmark is applied on the font in
       ui.widgets.ElidingLabel so QFontMetrics can measure it. */
    QLabel#BrandName {{
        font-size: {m.font_pt(19)}pt;
        font-weight: 700;
        color: {palette.text};
    }}

    QLabel#BrandTag {{
        font-size: {m.font_pt(11)}pt;
        color: {palette.text_faint};
        letter-spacing: 0.4px;
    }}

    QPushButton#NavButton {{
        background: transparent;
        border: none;
        border-radius: {m.sp(10)}px;
        padding: {m.sp(11)}px {m.sp(14)}px;
        text-align: left;
        font-size: {m.font_pt(14)}pt;
        font-weight: 500;
        color: {palette.text_muted};
    }}

    QPushButton#NavButton:hover {{
        background-color: {palette.surface_alt};
        color: {palette.text};
    }}

    QPushButton#NavButton:checked {{
        background-color: {palette.accent_soft};
        color: {palette.accent};
        font-weight: 600;
    }}

    QLabel#SidebarFooter {{
        color: {palette.text_faint};
        font-size: {m.font_pt(11)}pt;
    }}

    QLabel#WizardStep {{
        font-size: {m.font_pt(13)}pt;
        color: {palette.text_faint};
        padding: {m.sp(3)}px 0px;
    }}

    QLabel#WizardStepActive {{
        font-size: {m.font_pt(13)}pt;
        font-weight: 700;
        color: {palette.accent};
        padding: {m.sp(3)}px 0px;
    }}

    QLabel#WizardStepDone {{
        font-size: {m.font_pt(13)}pt;
        color: {palette.text_muted};
        padding: {m.sp(3)}px 0px;
    }}

    /* ------------------------------------------------------------ cards */

    QFrame#Card, QFrame#HeaderBar {{
        background-color: {palette.surface};
        border: 1px solid {palette.border};
        border-radius: {m.sp(16)}px;
    }}

    QLabel#AssistantName {{
        font-size: {m.font_pt(20)}pt;
        font-weight: 700;
    }}

    QLabel#AssistantTag {{
        font-size: {m.font_pt(12)}pt;
        color: {palette.text_muted};
    }}

    QLabel#PageTitle {{
        font-size: {m.font_pt(22)}pt;
        font-weight: 700;
    }}

    QLabel#SectionTitle {{
        font-size: {m.font_pt(13)}pt;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: {palette.text_faint};
    }}

    QLabel#FieldCaption {{
        font-size: {m.font_pt(13)}pt;
        font-weight: 600;
    }}

    QLabel#ToggleTitle {{
        font-size: {m.font_pt(14)}pt;
        font-weight: 600;
    }}

    QLabel#Muted {{
        color: {palette.text_muted};
        font-size: {m.font_pt(12)}pt;
    }}

    /* ------------------------------------------------------------ pills */

    QLabel#Pill {{
        background-color: {palette.surface_alt};
        border: 1px solid {palette.border};
        border-radius: {m.sp(11)}px;
        padding: {m.sp(3)}px {m.sp(11)}px;
        font-size: {m.font_pt(11)}pt;
        font-weight: 600;
        color: {palette.text_muted};
    }}

    QLabel#PillOnline {{
        background-color: {palette.accent_soft};
        border: 1px solid {palette.accent};
        border-radius: {m.sp(11)}px;
        padding: {m.sp(3)}px {m.sp(11)}px;
        font-size: {m.font_pt(11)}pt;
        font-weight: 600;
        color: {palette.accent};
    }}

    /* ------------------------------------------------------ conversation */

    QScrollArea#Conversation, QScrollArea#ResponsiveScroll {{
        background: transparent;
        border: none;
    }}

    QWidget#ConversationCanvas {{
        background: transparent;
    }}

    QFrame#BubbleUser {{
        background-color: {palette.bubble_user};
        border: 1px solid {palette.border};
        border-radius: {m.sp(14)}px;
    }}

    QFrame#BubbleAssistant {{
        background-color: {palette.bubble_assistant};
        border: 1px solid {palette.border};
        border-radius: {m.sp(14)}px;
    }}

    QFrame#BubbleError {{
        background-color: {palette.bubble_assistant};
        border: 1px solid {palette.danger};
        border-radius: {m.sp(14)}px;
    }}

    QFrame#BubbleConfirm {{
        background-color: {palette.bubble_assistant};
        border: 1px solid {palette.warning};
        border-radius: {m.sp(14)}px;
    }}

    QLabel#BubbleText {{
        font-size: {m.font_pt(14)}pt;
    }}

    QLabel#BubbleMeta {{
        font-size: {m.font_pt(10)}pt;
        font-weight: 700;
        letter-spacing: 1px;
        color: {palette.text_faint};
    }}

    /* ------------------------------------------------------------ input */

    QFrame#Composer {{
        background-color: {palette.surface};
        border: 1px solid {palette.border};
        border-radius: {m.sp(18)}px;
    }}

    QLineEdit#CommandInput {{
        background: transparent;
        border: none;
        font-size: {m.font_pt(15)}pt;
        padding: {m.sp(6)}px {m.sp(4)}px;
        color: {palette.text};
    }}

    QLineEdit, QComboBox, QSpinBox {{
        background-color: {palette.surface_alt};
        border: 1px solid {palette.border};
        border-radius: {m.sp(10)}px;
        padding: {m.sp(8)}px {m.sp(12)}px;
        selection-background-color: {palette.accent};
        selection-color: {palette.base};
    }}

    QLineEdit:focus, QComboBox:focus {{
        border: 1px solid {palette.accent};
    }}

    QComboBox::drop-down {{
        border: none;
        width: {m.sp(22)}px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {palette.surface};
        border: 1px solid {palette.border};
        selection-background-color: {palette.accent_soft};
        selection-color: {palette.accent};
        padding: {m.sp(4)}px;
    }}

    QListWidget {{
        background-color: {palette.surface_alt};
        border: 1px solid {palette.border};
        border-radius: {m.sp(10)}px;
        padding: {m.sp(4)}px;
        font-size: {m.font_pt(13)}pt;
    }}

    QListWidget::item {{
        padding: {m.sp(8)}px {m.sp(10)}px;
        border-radius: {m.sp(7)}px;
    }}

    QListWidget::item:selected {{
        background-color: {palette.accent_soft};
        color: {palette.accent};
    }}

    QRadioButton {{
        font-size: {m.font_pt(14)}pt;
        spacing: {m.sp(9)}px;
        padding: {m.sp(3)}px 0px;
    }}

    /* ---------------------------------------------------------- buttons */

    QPushButton {{
        background-color: {palette.surface_alt};
        border: 1px solid {palette.border};
        border-radius: {m.sp(10)}px;
        padding: {m.sp(9)}px {m.sp(16)}px;
        font-size: {m.font_pt(13)}pt;
        font-weight: 600;
        color: {palette.text};
    }}

    QPushButton:hover {{
        border: 1px solid {palette.border_strong};
    }}

    QPushButton:disabled {{
        color: {palette.text_faint};
        border: 1px solid {palette.border};
    }}

    QPushButton#Primary {{
        background-color: {palette.accent};
        border: 1px solid {palette.accent};
        color: #06121C;
    }}

    QPushButton#Primary:hover {{
        background-color: {palette.accent_alt};
        border: 1px solid {palette.accent_alt};
        color: #FFFFFF;
    }}

    QPushButton#Primary:disabled {{
        background-color: {palette.surface_alt};
        border: 1px solid {palette.border};
        color: {palette.text_faint};
    }}

    QPushButton#Danger {{
        background-color: transparent;
        border: 1px solid {palette.danger};
        color: {palette.danger};
    }}

    QPushButton#Ghost {{
        background: transparent;
        border: 1px solid {palette.border};
        color: {palette.text_muted};
    }}

    QPushButton#IconButton {{
        background-color: {palette.surface_alt};
        border: 1px solid {palette.border};
        border-radius: {m.sp(18)}px;
        padding: 0px;
        font-size: {m.font_pt(15)}pt;
    }}

    /* The microphone while it is actually capturing. */
    QPushButton#IconButtonActive {{
        background-color: {palette.accent};
        border: 2px solid {palette.accent_alt};
        border-radius: {m.sp(18)}px;
        padding: 0px;
        font-size: {m.font_pt(15)}pt;
        color: #06121C;
    }}

    /* ------------------------------------------------------- listening */

    QLabel#ListeningCaption {{
        font-size: {m.font_pt(12)}pt;
        font-weight: 700;
        color: {palette.accent};
    }}

    QProgressBar#LevelMeter {{
        background-color: {palette.surface_alt};
        border: 1px solid {palette.border};
        border-radius: {m.size(3)}px;
    }}

    QProgressBar#LevelMeter::chunk {{
        background-color: {palette.accent};
        border-radius: {m.size(3)}px;
    }}

    /* ---------------------------------------------------------- toggles */

    QCheckBox {{
        spacing: {m.sp(10)}px;
        font-size: {m.font_pt(13)}pt;
    }}

    QCheckBox::indicator {{
        width: {m.size(40)}px;
        height: {m.size(22)}px;
        border-radius: {m.size(11)}px;
        border: 1px solid {palette.border_strong};
        background-color: {palette.surface_alt};
    }}

    QCheckBox::indicator:checked {{
        background-color: {palette.accent};
        border: 1px solid {palette.accent};
    }}

    /* ----------------------------------------------------------- tables */

    QTableWidget {{
        background-color: {palette.surface};
        border: 1px solid {palette.border};
        border-radius: {m.sp(12)}px;
        gridline-color: {palette.border};
        font-size: {m.font_pt(13)}pt;
    }}

    QTableWidget::item {{
        padding: {m.sp(7)}px {m.sp(10)}px;
        border: none;
    }}

    QHeaderView::section {{
        background-color: {palette.surface_alt};
        color: {palette.text_faint};
        border: none;
        border-bottom: 1px solid {palette.border};
        padding: {m.sp(9)}px {m.sp(10)}px;
        font-size: {m.font_pt(11)}pt;
        font-weight: 700;
        letter-spacing: 0.8px;
    }}

    /* --------------------------------------------------------- scrollbar */

    QScrollBar:vertical {{
        background: transparent;
        width: {m.size(10)}px;
        margin: {m.sp(4)}px {m.sp(2)}px;
    }}

    QScrollBar::handle:vertical {{
        background: {palette.border_strong};
        border-radius: {m.size(5)}px;
        min-height: {m.size(32)}px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {palette.text_faint};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        height: 0px;
        background: none;
    }}

    QScrollBar:horizontal {{
        background: transparent;
        height: {m.size(10)}px;
        margin: {m.sp(2)}px {m.sp(4)}px;
    }}

    QScrollBar::handle:horizontal {{
        background: {palette.border_strong};
        border-radius: {m.size(5)}px;
        min-width: {m.size(32)}px;
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        width: 0px;
        background: none;
    }}

    /* ------------------------------------------------------------- misc */

    QFrame#Divider {{
        background-color: {palette.border};
        border: none;
    }}

    QToolTip {{
        background-color: {palette.surface_alt};
        color: {palette.text};
        border: 1px solid {palette.border_strong};
        border-radius: {m.sp(6)}px;
        padding: {m.sp(6)}px {m.sp(9)}px;
        font-size: {m.font_pt(12)}pt;
    }}
    """


# --------------------------------------------------------------- generated art

def gradient_avatar(letter: str, size: int, palette: Palette,
                    device_pixel_ratio: float = 1.0) -> QPixmap:
    """Draw the circular assistant avatar at the screen's pixel density.

    The backing pixmap is allocated in *physical* pixels and tagged with the
    device pixel ratio, so painting still happens in logical coordinates while
    the result stays sharp at 125%, 150%, 200% and beyond. Allocating at
    logical size and letting Qt upscale is what produces a soft, blurry mark on
    a high-DPI display.

    Generated rather than shipped as an image so it always matches the current
    palette and the current assistant name, including after a rename.
    """
    ratio = max(1.0, float(device_pixel_ratio))
    pixmap = QPixmap(int(round(size * ratio)), int(round(size * ratio)))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.TextAntialiasing)

    gradient = QLinearGradient(0, 0, size, size)
    gradient.setColorAt(0.0, QColor(palette.accent))
    gradient.setColorAt(1.0, QColor(palette.accent_alt))
    painter.setBrush(QBrush(gradient))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(QRectF(0, 0, size, size))

    font = QFont(ui_font_family())
    font.setPixelSize(max(1, int(size * 0.46)))
    font.setWeight(QFont.Bold)
    painter.setFont(font)
    painter.setPen(QColor("#08121B"))
    painter.drawText(QRectF(0, 0, size, size), Qt.AlignCenter,
                     (letter or "A")[:1].upper())
    painter.end()
    return pixmap


def app_icon(palette: Palette = DARK) -> QIcon:
    """The window and taskbar icon, drawn at every size Windows asks for.

    Supplying the full ladder up to 256px means Windows never has to scale a
    small bitmap up for a high-DPI taskbar or the Alt-Tab switcher.
    """
    icon = QIcon()
    for size in (16, 20, 24, 32, 40, 48, 64, 96, 128, 256):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        gradient = QLinearGradient(0, 0, size, size)
        gradient.setColorAt(0.0, QColor(palette.accent))
        gradient.setColorAt(1.0, QColor(palette.accent_alt))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        radius = size * 0.26
        painter.drawRoundedRect(QRectF(0, 0, size, size), radius, radius)

        font = QFont(ui_font_family())
        font.setPixelSize(max(1, int(size * 0.58)))
        font.setWeight(QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor("#06121C"))
        painter.drawText(QRectF(0, 0, size, size), Qt.AlignCenter, "A")
        painter.end()
        icon.addPixmap(pixmap)
    return icon


def avatar_pixmap_for(widget, letter: str, size: int, palette: Palette) -> QPixmap:
    """Avatar rendered for the screen ``widget`` currently lives on."""
    return gradient_avatar(letter, size, palette, device_pixel_ratio_for(widget))
