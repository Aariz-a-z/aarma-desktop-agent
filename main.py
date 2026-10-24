"""AARMA entry point.

Usage
-----
    python main.py               Launch the desktop dashboard.
    python main.py --cli         Launch the text interface.
    python main.py --setup       Re-run first-run setup, then continue.
    python main.py --doctor      Print an environment report and exit.

The two interfaces are peers: both construct the same
:class:`~core.engine.AarmaEngine` and neither contains any command handling of
its own.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

APP_VERSION = "0.3.1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aarma",
        description="AARMA — Adaptive AI Voice-Controlled Desktop Agent (Phase 1 foundation)",
    )
    parser.add_argument("--cli", "--text", dest="cli", action="store_true",
                        help="run the text interface instead of the dashboard")
    parser.add_argument("--setup", action="store_true",
                        help="run first-run setup even if it has already been completed")
    parser.add_argument("--doctor", action="store_true",
                        help="print an environment report and exit")
    parser.add_argument("--benchmark", action="store_true",
                        help="open the voice command benchmark and exit")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="override the configured log level")
    parser.add_argument("--enrol-wake-word", dest="enrol_wake_word",
                        action="store_true",
                        help="record and validate a wake word for the current "
                             "assistant name")
    parser.add_argument("--enrol-name", default="",
                        help="enrol for this name instead of the current one")
    parser.add_argument("--enrol-count", type=int, default=0,
                        help="how many recordings enrollment should ask for")
    parser.add_argument("--enrol-replay", dest="enrol_replay", default="",
                        help="re-run enrollment training on a saved recording "
                             "session instead of recording again")
    parser.add_argument("--enrol-diagnose", dest="enrol_diagnose",
                        action="store_true",
                        help="print the measurements behind every accepted or "
                             "rejected enrollment recording")
    parser.add_argument("--version", action="version", version=f"AARMA {APP_VERSION}")
    return parser


def _configure(args: argparse.Namespace):
    """Set up paths, logging and configuration.  Returns the config manager."""
    from config.settings import load_config
    from utils.logger import configure_logging
    from utils.paths import get_paths

    get_paths()  # Creates the data, log and screenshot directories.
    config = load_config()
    level_name = args.log_level or config.config.runtime.log_level
    configure_logging(level=getattr(logging, level_name.upper(), logging.INFO))
    return config


def run_doctor() -> int:
    """Report what is installed and what this build can therefore do."""
    import platform

    from actions.registry import build_default_registry
    from config.settings import load_config
    from utils.paths import get_paths

    paths = get_paths()
    config = load_config()
    registry = build_default_registry()

    def probe(module: str) -> str:
        try:
            __import__(module)
            return "installed"
        except ImportError:
            return "missing"

    print(f"\nAARMA {APP_VERSION} — environment report\n" + "=" * 46)
    print(f"  Python            {sys.version.split()[0]} ({sys.platform})")
    print(f"  OS                {platform.system()} {platform.release()}")
    print(f"  Data folder       {paths.root}")
    print(f"  Configuration     {paths.config_file}")
    print(f"  Database          {paths.database_file}")
    print(f"  Log file          {paths.log_file}")
    print(f"\n  Assistant         {config.config.assistant.name}")
    print(f"  Language          {config.config.language.interface}")
    print(f"  First-run setup   {'done' if config.config.first_run_completed else 'not completed'}")
    print(f"  Registered actions {len(registry)}")

    print("\n  Optional components")
    for label, module in (("Desktop GUI (PySide6)", "PySide6"),
                          ("Screen capture (Pillow)", "PIL"),
                          ("Recycle-bin delete (Send2Trash)", "send2trash"),
                          ("Natural voice (edge-tts)", "edge_tts"),
                          ("Windows voice (pywin32)", "win32com"),
                          ("Microphone (sounddevice)", "sounddevice"),
                          ("Speech recognition (faster-whisper)", "faster_whisper")):
        print(f"    {label:<38}{probe(module)}")

    print("\n  Voice")
    try:
        from core.connectivity import ConnectivityMonitor
        from speech.engine import SpeechEngine

        speech = SpeechEngine(config, connectivity=ConnectivityMonitor(config),
                              autostart=False)
        for provider in speech.providers:
            state = "available" if provider.is_available() else "unavailable"
            tier = provider.quality.label
            print(f"    {provider.display_name:<34}{state} ({tier})")
        availability = speech.availability()
        print(f"    {'Selected voice':<34}{availability.describe()}")
        if not availability.available and availability.suggestion:
            print(f"    {'':<34}{availability.suggestion}")
    except Exception as exc:  # noqa: BLE001 - diagnostics must never fail hard
        print(f"    voice check failed: {exc}")

    print("\n  Microphone")
    try:
        from security.permissions import Permission, PermissionManager
        from speech.capture import diagnose
        from speech.diagnostics import format_level
        from speech.input_engine import ListeningEngine

        permissions = PermissionManager(config)
        listening = ListeningEngine(config, permissions=permissions)

        # Every device is actually opened and measured. Reporting "ready" from
        # configuration alone is what let a microphone returning silence look
        # healthy, and sent the last diagnosis in the wrong direction.
        diagnosis = diagnose(config.config.speech_input.device_name, seconds=0.6)

        print(f"    {'Windows permission':<38}{diagnosis.permission.label}")
        print(f"    {'AARMA permission':<38}"
              f"{'granted' if permissions.is_granted(Permission.MICROPHONE) else 'denied'}")

        resolution = listening.resolve_device()
        chosen = resolution.device
        print(f"    {'Selected device':<38}{chosen.name if chosen else 'none'}"
              f"{'  (fell back to default)' if resolution.fell_back else ''}")
        if chosen:
            print(f"    {'Host API / channels / rate':<38}"
                  f"{chosen.host_api or '-'} · {chosen.channels} ch · "
                  f"{chosen.sample_rate:.0f} Hz")

        selected = diagnosis.selected
        if selected is not None:
            print(f"    {'Stream':<38}"
                  f"{'opened' if selected.frames else 'failed to open'}")
            print(f"    {'Input signal':<38}"
                  f"frames {selected.frames}, non-zero {selected.nonzero}")
            print(f"    {'RMS / peak':<38}"
                  f"{format_level(selected.rms)} / {format_level(selected.peak)}")
            print(f"    {'Status':<38}{selected.status.symbol} {selected.status.label}")

        print(f"    {'All input devices':<38}{len(diagnosis.probes)} found")
        for entry in diagnosis.probes:
            mark = "*" if entry.is_selected else " "
            print(f"      {mark} {entry.status.symbol} {entry.name[:36]:<38}"
                  f"peak {format_level(entry.peak):>11}  {entry.status.label}")

        readiness = listening.readiness()
        print(f"    {'Speech input':<38}"
              f"{'ready' if readiness.ready else 'unavailable'}")
        print(f"    {'Privacy':<38}{readiness.privacy_label}")
        print(f"    {'Recommendation':<38}{diagnosis.recommendation()}")
    except Exception as exc:  # noqa: BLE001
        print(f"    microphone check failed: {type(exc).__name__}: {exc}")

    print("\n  Not implemented: language models, reminders,")
    print("  keyboard/mouse automation.\n")
    return 0


def _start_wake_listening(engine, *, background: bool = False) -> None:
    """Arm the wake word for the active identity, and say what happened.

    Silence would be the wrong default here: an assistant that is not listening
    looks exactly like one that is, until you talk to it.  So both the ready
    case and every not-ready case print one line.

    Gated on the existing ``speech_input.enabled`` setting rather than a new
    one -- someone who has turned speech input off has already said they do not
    want the microphone opened.
    """
    if background:
        # Loading the two models takes a few seconds; on the desktop interface
        # that must not happen on the thread painting the window.
        import threading

        threading.Thread(target=_start_wake_listening, args=(engine,),
                         name="wake-start", daemon=True).start()
        return
    if not engine.config.config.speech_input.enabled:
        print("  Wake word: off (speech input is disabled in settings).")
        return
    try:
        state = engine.wake_identity()
    except Exception as exc:  # noqa: BLE001 - never block start-up on this
        print(f"  Wake word: unavailable ({type(exc).__name__}: {exc})")
        return
    if not state["available"]:
        # Name what *would* work. The reason on its own ("no detector claims
        # 'hey boss'") tells the user their wake word is broken without telling
        # them how to get one back, and a configuration left pointing at a
        # phrase that was never installed is exactly how this happens.
        print(f"  Wake word: off -- {state['reason']}")
        installed = [p for p in state.get("installed_phrases", ()) if p]
        if installed:
            spoken = ", ".join(f'"{phrase}"' for phrase in installed)
            print(f"             I can currently hear {spoken}. "
                  f"Say 'set the wake word to {installed[0]}' to use one,")
            print(f"             or enrol \"{state.get('name', '')}\" from "
                  "Settings to record your own.")
        return
    if engine.start_wake_loop():
        print(f"  Wake word: listening for \"{state['phrase']}\".")
    else:
        print("  Wake word: could not start; see the log for details.")


def run_cli(config, *, force_setup: bool) -> int:
    from core.engine import AarmaEngine
    from ui.console import ConsoleInterface, run_console_setup

    if force_setup or not config.config.first_run_completed:
        run_console_setup(config)

    engine = AarmaEngine(config)
    _start_wake_listening(engine)
    try:
        return ConsoleInterface(engine).run()
    finally:
        engine.shutdown()


def _configure_high_dpi() -> None:
    """Set Qt's DPI policy before any QApplication exists.

    Qt 6 already converts logical pixels to physical ones, which is why layout
    code never mentions DPI. What it does not do by default on every platform is
    honour *fractional* factors exactly -- rounding 150% to 100% or 200% is what
    makes an interface look either cramped or oversized on a scaled display.
    PassThrough keeps the user's real scale factor.

    Must run before the QApplication is constructed; afterwards it has no effect.
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication
    except ImportError:  # pragma: no cover - PySide6 is optional
        return
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )


def run_gui(config, *, force_setup: bool) -> int:
    try:
        _configure_high_dpi()
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("The desktop interface needs PySide6.\n"
              "  Install it with:  pip install -r requirements.txt\n"
              "  Or use the text interface:  python main.py --cli")
        return 2

    from core.engine import AarmaEngine
    from ui.dashboard import Dashboard
    from ui.responsive import Metrics
    from ui.setup_wizard import run_setup_wizard
    from ui.theme import app_icon, build_stylesheet, get_palette

    application = QApplication(sys.argv[:1])
    application.setApplicationName("AARMA")
    application.setOrganizationName("AARMA")
    application.setApplicationVersion(APP_VERSION)

    palette = get_palette(config.config.runtime.theme)
    # Metrics are read after the QApplication exists, so they reflect the user's
    # actual system font -- including a raised Windows text-size setting.
    application.setStyleSheet(build_stylesheet(palette, Metrics.current()))
    application.setWindowIcon(app_icon(palette))

    if force_setup or not config.config.first_run_completed:
        # A cancelled wizard still opens the dashboard with defaults; setup is
        # simply offered again next launch rather than blocking the user out.
        run_setup_wizard(config)

    engine = AarmaEngine(config)
    window = Dashboard(engine)
    window.show()
    # After show(), and off the GUI thread, so the window is painted and
    # usable while the detector loads its models.
    _start_wake_listening(engine, background=True)
    # Likewise after show(): the reminder sink delivers into this window, so
    # there has to be a window first.
    window.start_reminders()
    return application.exec()


def run_benchmark(config) -> int:
    """Open the voice command benchmark on its own, without the dashboard.

    A measurement run should not require navigating a settings screen first,
    and it should not leave the rest of the application open to be clicked
    during it. The same dialog is also reachable from Settings → Speech input →
    Advanced diagnostics; this is only a shortcut to it.
    """
    try:
        _configure_high_dpi()
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("The benchmark needs PySide6.\n"
              "  Install it with:  pip install -r requirements.txt")
        return 2

    from core.engine import AarmaEngine
    from ui.responsive import Metrics
    from ui.theme import app_icon, build_stylesheet, get_palette
    from ui.voice_benchmark import VoiceBenchmarkDialog

    application = QApplication(sys.argv[:1])
    application.setApplicationName("AARMA")
    application.setApplicationVersion(APP_VERSION)
    palette = get_palette(config.config.runtime.theme)
    application.setStyleSheet(build_stylesheet(palette, Metrics.current()))
    application.setWindowIcon(app_icon(palette))

    engine = AarmaEngine(config)
    try:
        dialog = VoiceBenchmarkDialog(engine)
        dialog.show()
        return application.exec()
    finally:
        engine.shutdown()


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.doctor:
        return run_doctor()

    config = _configure(args)
    from utils.logger import get_logger

    logger = get_logger("main")
    logger.info("AARMA starting",
                extra={"context": {"version": APP_VERSION,
                                   "interface": "cli" if args.cli else "gui"}})
    try:
        if args.enrol_wake_word:
            from tools.enrol_wakeword import run as run_enrolment

            return run_enrolment(config, name=args.enrol_name,
                                 count=args.enrol_count,
                                 diagnose=args.enrol_diagnose,
                                 replay=args.enrol_replay)
        if args.benchmark:
            return run_benchmark(config)
        if args.cli:
            return run_cli(config, force_setup=args.setup)
        return run_gui(config, force_setup=args.setup)
    except KeyboardInterrupt:
        return 0
    except Exception:  # noqa: BLE001 - top-level guard; details go to the log
        logger.exception("AARMA terminated with an unhandled error")
        print("AARMA hit an unexpected problem and had to stop. "
              "The details are in the log file.")
        return 1
    finally:
        logger.info("AARMA stopped")


if __name__ == "__main__":
    sys.exit(main())
