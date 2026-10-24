"""The text interface.

Specification section 35: while voice is not implemented, there must be a
reliable text mode that exercises the whole architecture.  This is that mode --
the same :class:`~core.engine.AarmaEngine` the dashboard drives, rendered as a
prompt.

It is also the fastest way to demonstrate the security model in a viva: type a
high-risk command and the confirmation prompt appears in the terminal exactly
as it would in the GUI, because both go through the same engine.
"""

from __future__ import annotations

import sys

from config.settings import (
    ASSISTANT_PROFILES,
    AssistantIdentity,
    ConfigManager,
    wake_phrase_for_name,
)
from core.engine import AarmaEngine, EngineResponse, ResponseKind
from language.manager import LANGUAGE_CATALOG, LanguageStatus
from security.permissions import PERMISSION_CATALOG, PermissionManager
from utils.logger import get_logger

logger = get_logger("ui.console")

BANNER = r"""
    _        _     ____   __  __     _
   / \      / \   |  _ \ |  \/  |   / \      Adaptive AI
  / _ \    / _ \  | |_) || |\/| |  / _ \     Voice-Controlled
 / ___ \  / ___ \ |  _ < | |  | | / ___ \    Desktop Agent
/_/   \_\/_/   \_\|_| \_\|_|  |_|/_/   \_\   Phase 2B · text mode with voice
"""


class Palette:
    """ANSI colours, disabled automatically when the terminal can't show them."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def accent(self, text: str) -> str:
        return self._wrap("38;5;81", text)

    def muted(self, text: str) -> str:
        return self._wrap("38;5;245", text)

    def success(self, text: str) -> str:
        return self._wrap("38;5;114", text)

    def warn(self, text: str) -> str:
        return self._wrap("38;5;215", text)

    def error(self, text: str) -> str:
        return self._wrap("38;5;203", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)


def _enable_ansi() -> bool:
    """Turn on virtual-terminal processing on Windows; report whether colour works."""
    if not sys.stdout.isatty():
        return False
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        # 7 == STD_OUTPUT_HANDLE (-11 as an unsigned handle), 0x0007 enables
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING alongside the existing flags.
        return bool(kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7))
    except Exception:  # noqa: BLE001 - colour is cosmetic; never fail the app for it
        return False


class ConsoleInterface:
    """A read-eval-print loop over the engine."""

    def __init__(self, engine: AarmaEngine) -> None:
        self.engine = engine
        self.palette = Palette(_enable_ansi())

    # ------------------------------------------------------------- rendering

    def _render(self, response: EngineResponse) -> str:
        if response.kind is ResponseKind.CONFIRMATION_REQUIRED:
            return self.palette.warn(response.text)
        if response.kind is ResponseKind.ERROR or not response.ok:
            return self.palette.error(response.text)
        if response.kind is ResponseKind.RESULT:
            return self.palette.success(response.text)
        return response.text

    def _print_response(self, response: EngineResponse) -> None:
        prompt = self.palette.accent(f"{self.engine.assistant_name} >")
        body = self._render(response).replace("\n", "\n  ")
        print(f"{prompt} {body}\n")
        # Speaking is the interface's decision, not the pipeline's, so the text
        # mode opts in here rather than the core engine speaking for everyone.
        if self.engine.speech.enabled:
            self.engine.speech.speak(response.text)

    # -------------------------------------------------------- notifications

    def show_reminder(self, event) -> bool:
        """A reminder sink for the text interface.

        Runs on the scheduler's thread while the main thread is parked in
        ``input()``.  Printing from here interleaves with whatever the user is
        halfway through typing, which is untidy -- and still far better than
        the alternative, because a reminder that waits for the user to press
        Return is not a reminder.  The leading newline and the re-drawn prompt
        keep the damage to one line.

        Returns True because the text genuinely reached the terminal.  The
        scheduler treats that as delivery and closes the reminder out; had this
        returned False it would keep retrying, which is the right behaviour for
        a sink that could not present it and the wrong one here.
        """
        text = event.spoken_text()
        print()
        print("  " + self.palette.warn(text))
        try:
            if self.engine.config.config.voice.speak_notifications and \
                    self.engine.speech.enabled and not event.missed:
                # Missed reminders are history: an alarm for something hours
                # past reads as an alarm for something now.
                self.engine.speech.speak(text)
        except Exception:  # noqa: BLE001 - a mute speaker is not a lost reminder
            pass
        print(self.palette.bold("you > "), end="", flush=True)
        return True

    def print_banner(self) -> None:
        print(self.palette.accent(BANNER))
        print("  " + self.palette.muted(self.engine.status_line()))
        if not self.engine.online:
            print("  " + self.palette.muted(
                self.engine.language.t("response.offline")))
        print()

    # ------------------------------------------------------------------ loop

    def run(self) -> int:
        """Run until the user exits.  Returns a process exit code."""
        self.print_banner()
        # Start ringing reminders now that there is somewhere to show them.
        # The engine deliberately does not do this for itself: "written to the
        # log" is not "the user was told", and a scheduler that believed
        # otherwise would mark reminders delivered on a headless machine.
        self.engine.start_reminders(self.show_reminder)
        self._print_response(self.engine.start())

        while True:
            try:
                raw = input(self.palette.bold("you > "))
            except (EOFError, KeyboardInterrupt):
                print()
                print(self.palette.muted("  Session ended."))
                return 0

            if not raw.strip():
                continue

            response = self.engine.handle_text(raw, source="text")
            self._print_response(response)
            if response.should_exit:
                return 0


# --------------------------------------------------------------- first run

def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return answer or default


def run_console_setup(config: ConfigManager) -> None:
    """A compact first-run setup for people who never open the GUI.

    Mirrors the wizard in :mod:`ui.setup_wizard`: same settings, same defaults,
    written to the same configuration file.
    """
    print("\n  AARMA first-run setup\n  " + "-" * 40)

    print("\n  1. Choose an assistant")
    for index, profile in enumerate(ASSISTANT_PROFILES.values(), start=1):
        print(f"     {index}. {profile.display_name} ({profile.gender}) — {profile.description}")
    choice = _ask("     Enter 1 or 2", "1")
    profiles = list(ASSISTANT_PROFILES.values())
    profile = profiles[1] if choice.strip() == "2" else profiles[0]
    identity = AssistantIdentity.from_profile(profile)

    print("\n  2. Name your assistant")
    name = _ask("     Assistant name", profile.display_name)
    if name != profile.display_name:
        identity = AssistantIdentity(
            profile_key="custom", name=name, gender=profile.gender,
            voice_id=profile.voice_id, wake_word=wake_phrase_for_name(name),
        )
    config.update(assistant=identity)

    print("\n  3. Choose a language")
    for info in LANGUAGE_CATALOG.values():
        note = ("" if info.status is LanguageStatus.TRANSLATED
                else f"  (selectable now, translation in Phase {info.phase})")
        print(f"     {info.code}  {info.english_name}{note}")
    language = _ask("     Language code", "en")
    from language.manager import LanguageManager

    LanguageManager(config).set_language(language)

    print("\n  4. Permissions")
    print("     Defaults grant browsing, files and reminders; microphone, camera,")
    print("     screen capture, keyboard, mouse and system control stay OFF.")
    PermissionManager(config).apply_defaults()
    if _ask("     Review each permission now? (y/N)", "n").lower().startswith("y"):
        manager = PermissionManager(config)
        for info in PERMISSION_CATALOG.values():
            current = "granted" if manager.is_granted(info.permission) else "denied"
            answer = _ask(f"     {info.display_name} [{current}] — grant? (y/n)",
                          "y" if current == "granted" else "n")
            manager.set(info.permission, answer.lower().startswith("y"))

    print("\n  5. Security passphrase (optional)")
    print("     Leave blank to skip. It is stored only as a salted hash.")
    passphrase = _ask("     Passphrase")
    if passphrase:
        from security.passphrase import PassphraseError, PassphraseManager

        try:
            PassphraseManager(config).enable(passphrase)
            print("     Security mode enabled.")
        except PassphraseError as exc:
            print(f"     Skipped: {exc}")

    config.mark_first_run_complete()
    print(f"\n  Setup complete. Say hello to {config.config.assistant.name}.\n")
