"""First-run setup (specification section 21).

Eight steps: welcome, assistant, language, voice, name, permissions, optional
security, finish. Nothing is written to disk until the user reaches the final
step and presses Finish, so backing out at any point leaves the configuration
untouched.

**Window architecture.** This is a top-level :class:`QWidget`, not a
``QDialog``. A dialog on Windows is created without minimize and maximize
buttons and resists being treated as an ordinary window — which is exactly the
behaviour that made the wizard feel unlike real desktop software. A plain
parentless widget with explicit window hints gets the full native frame:
minimize, maximize, restore, resize and close all behave as the user expects.
Completion is reported through the :attr:`SetupWizard.finished` signal instead
of ``exec()``'s return value.

**Nothing is ever clipped.** Every one of the eight pages lives inside its own
scroll area, while the step rail and the Back/Next footer sit outside it. When
vertical space runs short the *content* scrolls; the navigation the user needs
in order to continue stays on screen. That is the difference between a window
that adapts and one that hides its own Finish button.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from config.settings import (
    ASSISTANT_PROFILES,
    AssistantIdentity,
    ConfigManager,
    wake_phrase_for_name,
)
from language.manager import LANGUAGE_CATALOG, LanguageManager, LanguageStatus
from security.passphrase import PassphraseError, PassphraseManager
from security.permissions import PERMISSION_CATALOG, Permission, PermissionManager
from speech.voices import VOICE_PROFILES
from ui.responsive import (
    Metrics,
    apply_initial_geometry,
    breakpoint_threshold,
    scroll_wrap,
    sidebar_width_for,
)
from ui.theme import Palette, app_icon, build_stylesheet, get_palette
from ui.widgets import Avatar, ElidingLabel, ToggleRow, divider, muted, section_title
from utils.logger import get_logger

logger = get_logger("ui.setup")

STEPS = (
    "Welcome",
    "Assistant",
    "Language",
    "Voice",
    "Name",
    "Permissions",
    "Security",
    "Finish",
)

#: The size the layout is comfortable at, and the smallest size it stays usable
#: at. Neither is a target: :func:`~ui.responsive.apply_initial_geometry` picks
#: the actual size from the screen's work area and only uses these as bounds.
IDEAL_SIZE = (980, 680)
MINIMUM_SIZE = (620, 420)

#: Below this window width the step rail is hidden and the compact "Step N of 8"
#: caption in the content column becomes the sole progress indicator.
RAIL_BREAKPOINT = 760


class SetupWizard(QWidget):
    """Collects the settings a new user must choose before AARMA starts."""

    #: ``True`` when the user pressed Finish, ``False`` when they closed the
    #: window. Emitted exactly once.
    finished = Signal(bool)

    def __init__(self, config: ConfigManager, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._completed = False
        self._finished_emitted = False
        self.metrics = Metrics.current()
        self.palette_tokens: Palette = get_palette(config.config.runtime.theme)

        # A normal application window: minimize, maximize, restore, resize, close.
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setWindowTitle("Welcome to AARMA")
        self.setWindowIcon(app_icon(self.palette_tokens))
        self.setObjectName("Root")

        if not (QApplication.instance() and QApplication.instance().styleSheet()):
            self.setStyleSheet(build_stylesheet(self.palette_tokens, self.metrics))

        # Working copy of the answers; committed only on Finish.
        self._profile_key = "asma"
        self._language_code = "en"
        self._voice_id = ASSISTANT_PROFILES["asma"].voice_id
        self._permission_state: Dict[str, bool] = {
            info.permission.value: info.default_granted
            for info in PERMISSION_CATALOG.values()
        }

        m = self.metrics
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Without this, Qt writes the layout's own minimum onto the window,
        # overriding the deliberately small minimum chosen from the screen --
        # and the window then refuses to shrink no matter how much of its
        # content can scroll.
        layout.setSizeConstraint(QLayout.SetNoConstraint)

        self.rail = self._build_rail()
        layout.addWidget(self.rail)

        right = QWidget()
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(m.sp(30), m.sp(26), m.sp(30), m.sp(20))
        right_layout.setSpacing(m.sp(14))

        self.step_caption = section_title("Step 1 of 8 · Welcome")
        right_layout.addWidget(self.step_caption)

        # Each page is wrapped in its own scroll area, so the stack's minimum
        # height is a scrollbar's worth rather than the tallest page's.
        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.setMinimumSize(1, 1)
        builders: tuple[Callable[[], QWidget], ...] = (
            self._page_welcome, self._page_assistant, self._page_language,
            self._page_voice, self._page_name, self._page_permissions,
            self._page_security, self._page_finish,
        )
        for build in builders:
            content = build()
            content.setObjectName("ConversationCanvas")
            self.stack.addWidget(scroll_wrap(content))
        right_layout.addWidget(self.stack, 1)

        right_layout.addWidget(divider())
        right_layout.addLayout(self._build_footer())
        layout.addWidget(right, 1)

        apply_initial_geometry(self, ideal=IDEAL_SIZE, minimum=MINIMUM_SIZE)
        self._update_navigation()
        self._apply_breakpoints(self.width())

    # ---------------------------------------------------------------- chrome

    def _build_rail(self) -> QFrame:
        m = self.metrics
        rail = QFrame()
        rail.setObjectName("Sidebar")
        # Bounded, not fixed: the actual width is recomputed from the window in
        # `_apply_breakpoints`, so the rail shrinks with a narrow window instead
        # of squeezing the content it is meant to accompany.
        rail.setMinimumWidth(m.size(150))
        rail.setMaximumWidth(m.size(264))
        rail.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        layout = QVBoxLayout(rail)
        layout.setContentsMargins(m.sp(20), m.sp(26), m.sp(20), m.sp(20))
        layout.setSpacing(m.sp(14))

        brand = QHBoxLayout()
        brand.setSpacing(m.sp(12))
        self.brand_mark = Avatar("A", 40, self.palette_tokens, m)
        brand.addWidget(self.brand_mark)
        text = QVBoxLayout()
        text.setSpacing(0)
        self.brand_name = ElidingLabel("AARMA", object_name="BrandName",
                                       letter_spacing=m.sp(2))
        self.brand_tag = ElidingLabel("Setup", object_name="BrandTag")
        text.addWidget(self.brand_name)
        text.addWidget(self.brand_tag)
        brand.addLayout(text, 1)
        layout.addLayout(brand)
        layout.addWidget(divider())

        # Eight step labels plus a footnote are taller than a short window, and
        # a non-scrolling rail would force its full height onto the window --
        # capping how small the whole wizard can go. Scrolling the list keeps
        # the rail's contribution to the minimum height near zero.
        steps_holder = QWidget()
        steps_holder.setObjectName("ConversationCanvas")
        steps_layout = QVBoxLayout(steps_holder)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(m.sp(12))

        self.step_labels: List[QLabel] = []
        for index, title in enumerate(STEPS, start=1):
            label = QLabel(f"{index}.  {title}")
            label.setObjectName("WizardStep")
            label.setWordWrap(True)
            self.step_labels.append(label)
            steps_layout.addWidget(label)

        steps_layout.addStretch(1)
        steps_layout.addWidget(muted("You can change all of this later in Settings."))
        layout.addWidget(scroll_wrap(steps_holder), 1)
        return rail

    def _build_footer(self) -> QHBoxLayout:
        """Back / Next. Deliberately outside every scroll area — always reachable."""
        m = self.metrics
        row = QHBoxLayout()
        row.setSpacing(m.sp(10))
        self.back_button = QPushButton("Back")
        self.back_button.setObjectName("Ghost")
        self.back_button.clicked.connect(self._go_back)
        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("Primary")
        self.next_button.setDefault(True)
        self.next_button.clicked.connect(self._go_next)
        row.addStretch(1)
        row.addWidget(self.back_button)
        row.addWidget(self.next_button)
        return row

    # ----------------------------------------------------------------- pages

    def _page_welcome(self) -> QWidget:
        m = self.metrics
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(m.sp(16))
        layout.addStretch(1)

        centre = QHBoxLayout()
        centre.addStretch(1)
        centre.addWidget(Avatar("A", 88, self.palette_tokens, m))
        centre.addStretch(1)
        layout.addLayout(centre)

        title = QLabel("Welcome to AARMA")
        title.setObjectName("PageTitle")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        layout.addWidget(title)

        subtitle = muted(
            "An adaptive desktop assistant that understands what you ask, checks what "
            "it is allowed to do, and then does only that.\n\n"
            "This short setup chooses who your assistant is and what it may access. "
            "Everything is stored locally on this machine."
        )
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        note = muted(
            "This is the Phase 1 foundation build: text commands work today; "
            "voice, wake word and AI reasoning arrive in later phases."
        )
        note.setAlignment(Qt.AlignCenter)
        layout.addWidget(note)
        layout.addStretch(2)
        return page

    def _page_assistant(self) -> QWidget:
        m = self.metrics
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(m.sp(16))
        layout.addWidget(self._heading(
            "Choose your assistant",
            "Two identities ship with AARMA. Pick a starting point."))

        self.profile_group = QButtonGroup(self)
        row = QHBoxLayout()
        row.setSpacing(m.sp(14))
        for index, profile in enumerate(ASSISTANT_PROFILES.values()):
            frame = QFrame()
            frame.setObjectName("Card")
            frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(m.sp(20), m.sp(20), m.sp(20), m.sp(20))
            frame_layout.setSpacing(m.sp(10))

            avatar_row = QHBoxLayout()
            avatar_row.addStretch(1)
            avatar_row.addWidget(Avatar(profile.display_name, 64, self.palette_tokens, m))
            avatar_row.addStretch(1)
            frame_layout.addLayout(avatar_row)

            radio = QRadioButton(f"{profile.display_name}  ·  {profile.gender}")
            radio.setChecked(index == 0)
            self.profile_group.addButton(radio, index)
            frame_layout.addWidget(radio, 0, Qt.AlignCenter)
            frame_layout.addWidget(muted(profile.description))
            frame_layout.addWidget(muted(f"Wake word: \"{profile.wake_word}\""))
            row.addWidget(frame)
        layout.addLayout(row)
        layout.addStretch(1)
        self.profile_group.idClicked.connect(self._on_profile_selected)
        return page

    def _page_language(self) -> QWidget:
        m = self.metrics
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(m.sp(16))
        layout.addWidget(self._heading(
            "Choose a language",
            "English is fully translated in this build. The others can be selected now "
            "and are wired through the whole system; their interface text arrives in a "
            "later phase."))

        self.language_list = QListWidget()
        self.language_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.language_list.setMinimumHeight(m.size(140))
        for info in LANGUAGE_CATALOG.values():
            suffix = ("fully translated" if info.status is LanguageStatus.TRANSLATED
                      else f"selectable now · translation in Phase {info.phase}")
            item = QListWidgetItem(f"{info.english_name}   ({info.native_name})   —   {suffix}")
            item.setData(Qt.UserRole, info.code)
            self.language_list.addItem(item)
        self.language_list.setCurrentRow(0)
        self.language_list.currentItemChanged.connect(self._on_language_selected)
        layout.addWidget(self.language_list, 1)
        return page

    def _page_voice(self) -> QWidget:
        m = self.metrics
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(m.sp(16))
        layout.addWidget(self._heading(
            "Choose a voice",
            "AARMA uses a natural voice when you're online and a Windows voice when "
            "you're not. You can change this, and test it, in Settings at any time."))

        self.voice_group = QButtonGroup(self)
        container = QFrame()
        container.setObjectName("Card")
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(m.sp(20), m.sp(18), m.sp(20), m.sp(18))
        container_layout.setSpacing(m.sp(10))
        container_layout.addWidget(section_title("Available voice profiles"))
        for index, profile in enumerate(VOICE_PROFILES.values()):
            radio = QRadioButton(f"{profile.display_name}   ·   {profile.gender}")
            radio.setProperty("voice_id", profile.voice_id)
            radio.setChecked(profile.voice_id == self._voice_id)
            self.voice_group.addButton(radio, index)
            container_layout.addWidget(radio)
            container_layout.addWidget(muted(profile.description))
        layout.addWidget(container)
        layout.addStretch(1)
        self.voice_group.idClicked.connect(self._on_voice_selected)
        return page

    def _page_name(self) -> QWidget:
        m = self.metrics
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(m.sp(16))
        layout.addWidget(self._heading(
            "Name your assistant",
            "You can rename it any time, by voice or in Settings."))

        self.name_input = QLineEdit()
        self.name_input.setMaxLength(32)
        self.name_input.setPlaceholderText("Asma")
        self.name_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.name_input)
        self.name_preview = muted("")
        layout.addWidget(self.name_preview)
        self.name_input.textChanged.connect(self._update_name_preview)
        layout.addStretch(1)
        return page

    def _page_permissions(self) -> QWidget:
        m = self.metrics
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(m.sp(14))
        layout.addWidget(self._heading(
            "Choose what AARMA may access",
            "Least privilege by default: everyday capabilities are on, and anything that "
            "can watch you or control the machine is off until you say otherwise."))

        grouped: Dict[str, list] = {}
        for info in PERMISSION_CATALOG.values():
            grouped.setdefault(info.group, []).append(info)

        for group, infos in grouped.items():
            frame = QFrame()
            frame.setObjectName("Card")
            frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(m.sp(20), m.sp(16), m.sp(20), m.sp(16))
            frame_layout.setSpacing(m.sp(8))
            frame_layout.addWidget(section_title(group))
            for index, info in enumerate(infos):
                if index:
                    frame_layout.addWidget(divider())
                row = ToggleRow(info.permission.value, info.display_name,
                                info.description, info.default_granted, m)
                row.toggled.connect(self._on_permission_toggled)
                frame_layout.addWidget(row)
            layout.addWidget(frame)

        layout.addStretch(1)
        return page

    def _page_security(self) -> QWidget:
        m = self.metrics
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(m.sp(16))
        layout.addWidget(self._heading(
            "Set a passphrase (optional)",
            "Leave both boxes empty to skip. A passphrase is stored only as a salted "
            "PBKDF2 hash — never as plain text, and never in the log file."))

        self.passphrase_input = QLineEdit()
        self.passphrase_input.setEchoMode(QLineEdit.Password)
        self.passphrase_input.setPlaceholderText("Passphrase (8 characters or more)")
        layout.addWidget(self.passphrase_input)

        self.passphrase_confirm = QLineEdit()
        self.passphrase_confirm.setEchoMode(QLineEdit.Password)
        self.passphrase_confirm.setPlaceholderText("Repeat passphrase")
        layout.addWidget(self.passphrase_confirm)

        self.passphrase_note = muted(
            "High-risk actions always ask for a yes or no, with or without a passphrase. "
            "Setting one additionally locks AARMA when it starts."
        )
        layout.addWidget(self.passphrase_note)
        layout.addStretch(1)
        return page

    def _page_finish(self) -> QWidget:
        m = self.metrics
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(m.sp(16))
        layout.addStretch(1)

        centre = QHBoxLayout()
        centre.addStretch(1)
        self.finish_avatar = Avatar("A", 80, self.palette_tokens, m)
        centre.addWidget(self.finish_avatar)
        centre.addStretch(1)
        layout.addLayout(centre)

        self.finish_title = QLabel("All set")
        self.finish_title.setObjectName("PageTitle")
        self.finish_title.setAlignment(Qt.AlignCenter)
        self.finish_title.setWordWrap(True)
        layout.addWidget(self.finish_title)

        self.finish_summary = muted("")
        self.finish_summary.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.finish_summary)
        layout.addStretch(2)
        return page

    def _heading(self, title: str, subtitle: str) -> QWidget:
        m = self.metrics
        container = QWidget()
        container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(m.sp(6))
        label = QLabel(title)
        label.setObjectName("PageTitle")
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(muted(subtitle))
        return container

    # ------------------------------------------------------------ responsive

    def rail_breakpoint(self) -> int:
        """Window width below which the step rail steps aside for the content."""
        return breakpoint_threshold(RAIL_BREAKPOINT, MINIMUM_SIZE[0], self.metrics)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._apply_breakpoints(self.width())

    def _apply_breakpoints(self, width: int) -> None:
        """Re-shape the layout for the current window width.

        The rail is a share of the window rather than a constant, and drops out
        entirely on a narrow window so the content keeps a usable column. The
        step caption in the content header means progress is still visible when
        it does.
        """
        show_rail = width >= self.rail_breakpoint()
        self.rail.setVisible(show_rail)
        if show_rail:
            # Both bounds pinned to the computed width -- a range alone leaves
            # the rail parked at its (now collapsed) size hint. Still derived
            # from the window on every resize.
            rail_width = sidebar_width_for(width, self.metrics)
            self.rail.setMinimumWidth(rail_width)
            self.rail.setMaximumWidth(rail_width)
            # Drop the mark at exactly the point the wordmark would otherwise be
            # truncated, measured against the real font rather than guessed.
            m = self.metrics
            chrome = m.sp(20) * 2 + self.brand_mark.sizeHint().width() + m.sp(12)
            self.brand_mark.setVisible(
                (rail_width - chrome) >= self.brand_name.natural_width() + m.sp(4)
            )

    # ------------------------------------------------------------ selections

    def _on_profile_selected(self, index: int) -> None:
        profile = list(ASSISTANT_PROFILES.values())[index]
        self._profile_key = profile.key
        self._voice_id = profile.voice_id
        self.name_input.setPlaceholderText(profile.display_name)
        self._update_name_preview()

    def _on_language_selected(self, item: Optional[QListWidgetItem]) -> None:
        if item is not None:
            self._language_code = item.data(Qt.UserRole)

    def _on_voice_selected(self, index: int) -> None:
        button = self.voice_group.button(index)
        if button is not None:
            self._voice_id = button.property("voice_id")

    def _on_permission_toggled(self, key: str, granted: bool) -> None:
        self._permission_state[key] = granted

    def _update_name_preview(self) -> None:
        name = self._chosen_name()
        self.name_preview.setText(
            f"You'll be able to say \"hey {name.lower()}\" once voice lands in Phase 2. "
            f"Until then, just type."
        )
        self.finish_avatar.set_letter(name)

    def _chosen_name(self) -> str:
        typed = self.name_input.text().strip()
        return typed or ASSISTANT_PROFILES[self._profile_key].display_name

    # ------------------------------------------------------------ navigation

    def _update_navigation(self) -> None:
        index = self.stack.currentIndex()
        self.back_button.setEnabled(index > 0)
        last = index == self.stack.count() - 1
        self.next_button.setText("Finish" if last else "Next")
        self.step_caption.setText(f"Step {index + 1} of {len(STEPS)} · {STEPS[index]}")

        # Object names rather than inline stylesheets, so the step rail follows
        # the palette and the user's text size like everything else.
        for step, label in enumerate(self.step_labels):
            name = ("WizardStepActive" if step == index
                    else "WizardStepDone" if step < index else "WizardStep")
            if label.objectName() != name:
                label.setObjectName(name)
                label.style().unpolish(label)
                label.style().polish(label)

        if last:
            name = self._chosen_name()
            language = LANGUAGE_CATALOG[self._language_code].english_name
            granted = sum(1 for value in self._permission_state.values() if value)
            self.finish_title.setText(f"{name} is ready")
            self.finish_summary.setText(
                f"Language: {language}   ·   Voice: {self._voice_id}   ·   "
                f"{granted} of {len(self._permission_state)} permissions granted\n\n"
                "Press Finish to save these choices and open the dashboard."
            )

    def _go_back(self) -> None:
        self.stack.setCurrentIndex(max(0, self.stack.currentIndex() - 1))
        self._update_navigation()

    def _go_next(self) -> None:
        if self.stack.currentIndex() == self.stack.count() - 1:
            self._commit()
            return
        self.stack.setCurrentIndex(self.stack.currentIndex() + 1)
        self._update_navigation()

    # ---------------------------------------------------------------- commit

    def _commit(self) -> None:
        """Write every choice, then close. This is the first disk write."""
        profile = ASSISTANT_PROFILES[self._profile_key]
        name = self._chosen_name()

        passphrase = self.passphrase_input.text()
        if passphrase and passphrase != self.passphrase_confirm.text():
            self._fail_on_security_step("Those two passphrases don't match. Try again.")
            return

        identity = AssistantIdentity(
            profile_key=self._profile_key if name == profile.display_name else "custom",
            name=name,
            gender=profile.gender,
            voice_id=self._voice_id,
            wake_word=wake_phrase_for_name(name),
        )
        self._config.update(assistant=identity)
        LanguageManager(self._config).set_language(self._language_code)

        PermissionManager(self._config).apply_selection(
            {Permission(key): value for key, value in self._permission_state.items()}
        )

        if passphrase:
            try:
                PassphraseManager(self._config).enable(passphrase)
            except PassphraseError as exc:
                self._fail_on_security_step(str(exc))
                return

        self._config.mark_first_run_complete()
        logger.info("First-run setup completed",
                    extra={"context": {"language": self._language_code,
                                       "profile": self._profile_key}})
        self._completed = True
        self._emit_finished()
        self.close()

    def _fail_on_security_step(self, message: str) -> None:
        self.stack.setCurrentIndex(STEPS.index("Security"))
        self.passphrase_note.setText(message)
        self._update_navigation()

    # -------------------------------------------------------------- lifecycle

    def _emit_finished(self) -> None:
        if not self._finished_emitted:
            self._finished_emitted = True
            self.finished.emit(self._completed)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # Covers both Finish and the window's close button, so whoever is
        # waiting on `finished` is never left waiting forever.
        self._emit_finished()
        super().closeEvent(event)

    @property
    def completed(self) -> bool:
        return self._completed


def run_setup_wizard(config: ConfigManager) -> bool:
    """Show the wizard and block until it closes. Returns True if completed.

    A local event loop rather than ``QDialog.exec()``: the wizard is an ordinary
    window, so the caller supplies the modality it needs (none, here) instead of
    inheriting a dialog's.
    """
    from PySide6.QtCore import QEventLoop

    wizard = SetupWizard(config)
    loop = QEventLoop()
    wizard.finished.connect(lambda _completed: loop.quit())
    wizard.show()
    wizard.raise_()
    wizard.activateWindow()
    loop.exec()
    return wizard.completed
