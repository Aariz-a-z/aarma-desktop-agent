"""Shared fixtures.

Three rules the whole suite follows.  The third was bought the hard way.

* **Nothing touches the real user profile.**  Every fixture redirects
  ``AARMA_HOME`` into pytest's ``tmp_path`` and roots the filesystem sandbox in
  a scratch directory, so a test can exercise create/rename/move/delete without
  any chance of reaching real documents.
* **Nothing launches a real application or opens a browser.**  Tests that cover
  those paths patch the launcher and assert on what it was asked to do.
* **Nothing can reach a real operating-system side effect, whether or not the
  test remembered to prevent it.**  :func:`block_real_os_side_effects` is
  autouse and severs every such boundary for the whole suite.

That third rule exists because the second one was a convention, and a
convention is exactly what the next test breaks.  A mutation test disabled the
confirmation gate on purpose -- which is a reasonable thing for a mutation test
to do -- and then sent "shut down the computer" through a real engine.  The
engine builds its own action context, which carries no injected process runner,
so ``actions.power`` fell through to the real :func:`subprocess.run` and issued
``shutdown /s /t 20``.  The suite reported a pass; the machine powered off
twenty seconds later.

So the safety is no longer per-test discipline.  It is structural, it is on by
default, and a test that wants a real boundary has to reach past a guard that
fails loudly rather than silently.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterator

import pytest

# Allow `python -m pytest` from the repository root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from actions.registry import ActionContext, build_default_registry  # noqa: E402
from config.settings import ConfigManager  # noqa: E402
from core.engine import AarmaEngine  # noqa: E402
from database.db import Database  # noqa: E402
from language.manager import LanguageManager  # noqa: E402
from security.permissions import PermissionManager  # noqa: E402
from security.validator import SafePathResolver  # noqa: E402
from utils.paths import reset_paths_cache  # noqa: E402


class RealOSCallBlocked(BaseException):
    """A test reached a boundary that would have changed the real machine.

    Deliberately a ``BaseException``.  Production code catches broad
    ``Exception`` in two places on this path -- ``actions.power._execute`` and
    ``AarmaEngine._invoke`` -- so that one failing action cannot crash the
    assistant.  That is right in production and wrong here: it would swallow
    this guard and turn "a test tried to shut the machine down" into a quiet
    failed-action result, reported as a pass.  ``BaseException`` travels
    through both and fails the run.
    """


class _UnusableBackend:
    """Stands in for a real OS backend and reports itself unusable.

    Providers ask ``usable()`` before doing anything, so returning ``False``
    makes them report themselves unavailable and every handler above them
    refuse cleanly -- which is the behaviour a test should see.  Any call that
    somehow gets past that check raises instead of doing it.
    """

    def usable(self) -> bool:
        return False

    def __getattr__(self, name: str):
        def _blocked(*_args, **_kwargs):
            raise RealOSCallBlocked(
                f"A test called {name}() on a real OS backend. Inject a fake "
                "provider through the ActionContext instead."
            )

        return _blocked


class _BlockedSubprocess:
    """A drop-in for the :mod:`subprocess` module that spawns nothing.

    The constants and exception types are re-exported from the real module
    because the code under test catches and references them; only the four
    functions that actually start a process are replaced.
    """

    DEVNULL = subprocess.DEVNULL
    PIPE = subprocess.PIPE
    STDOUT = subprocess.STDOUT
    SubprocessError = subprocess.SubprocessError
    TimeoutExpired = subprocess.TimeoutExpired
    CalledProcessError = subprocess.CalledProcessError

    @staticmethod
    def _blocked(*args, **_kwargs):
        raise RealOSCallBlocked(
            f"A test tried to spawn a real process: {args[0] if args else args!r}"
        )

    run = _blocked
    Popen = _blocked
    call = _blocked
    check_output = _blocked


@pytest.fixture(autouse=True)
def block_real_os_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sever every route from the test suite to a real machine change.

    Applied to every test.  A test that legitimately exercises one of these
    paths supplies its own fake -- an injected ``process_runner``, a patched
    ``_launch``, an injected provider -- and that fake wins, because the test's
    own ``monkeypatch`` is applied after this fixture.

    What is blocked, and why each one is here:

    * **Power commands.**  ``shutdown``/``restart``/``sleep``/``sign out``/
      ``lock``. The incident that prompted all of this.
    * **Process spawning** in :mod:`actions.applications` and
      :mod:`actions.filesystem` -- launching a program, ``taskkill``, and
      opening a file with its default handler.
    * **The browser**, which would otherwise open real tabs on the desktop of
      whoever is running the suite.
    * **Window management**, which can minimise and close real windows --
      ``close_window`` posts ``WM_CLOSE``, and someone's unsaved document is
      not a test fixture.
    * **Keyboard and mouse**, which type and click into whatever has focus.
    * **Media keys**, which change the machine's volume.

    Screen capture is deliberately *not* blocked: it writes a PNG into the
    per-test temporary home and changes nothing, and the screenshot tests need
    a real grab to prove a real file appears.
    """
    import actions.applications
    import actions.browser
    import actions.filesystem
    import actions.power
    import providers.desktop
    import providers.input
    import providers.media

    def _blocked(what: str):
        def _refuse(*args, **_kwargs):
            raise RealOSCallBlocked(f"A test tried to {what}: {args!r}")

        return _refuse

    # 1. Power. The fallback is never the real subprocess.run inside a test.
    #    `actions.power` guards this itself as well; this is the second lock on
    #    the same door, because this is the door that came off its hinges.
    def _guarded_power_runner(ctx):
        injected = getattr(ctx, "process_runner", None)
        if injected is not None:
            return injected
        return _blocked("run a real power command")

    monkeypatch.setattr(actions.power, "_runner", _guarded_power_runner)

    # 2. Launching, killing and opening things.
    monkeypatch.setattr(actions.applications, "_launch",
                        _blocked("launch a real application"))
    monkeypatch.setattr(actions.applications, "subprocess", _BlockedSubprocess)
    monkeypatch.setattr(actions.filesystem, "_open_with_default_app",
                        _blocked("open a file with its real handler"))
    monkeypatch.setattr(actions.filesystem, "subprocess", _BlockedSubprocess)
    monkeypatch.setattr(actions.browser, "_open_in_browser",
                        _blocked("open a real browser"))

    # 3. The desktop providers. Patching the backend inside each process-wide
    #    singleton reaches every importer, however the name was imported.
    monkeypatch.setattr(providers.desktop.WINDOWS, "_backend", _UnusableBackend())
    monkeypatch.setattr(providers.input.INPUT, "_backend", _UnusableBackend())
    monkeypatch.setattr(providers.media.MEDIA, "_user32", None)
    monkeypatch.setattr(providers.media.MEDIA, "_resolved", True)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point every path helper at a throwaway directory for the duration of a test."""
    home = tmp_path / "aarma-home"
    home.mkdir()
    monkeypatch.setenv("AARMA_HOME", str(home))
    reset_paths_cache()
    yield home
    reset_paths_cache()


@pytest.fixture
def config(tmp_path: Path) -> ConfigManager:
    manager = ConfigManager(tmp_path / "config.json")
    manager.load()
    return manager


@pytest.fixture
def language(config: ConfigManager) -> LanguageManager:
    return LanguageManager(config)


@pytest.fixture
def permissions(config: ConfigManager) -> PermissionManager:
    return PermissionManager(config)


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A scratch directory that stands in for the user's content folders."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("hello", encoding="utf-8")
    (workspace / "Reports").mkdir()
    return workspace


@pytest.fixture
def resolver(sandbox: Path) -> SafePathResolver:
    return SafePathResolver([sandbox])


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    db = Database(tmp_path / "test.db")
    db.initialise()
    yield db
    db.close()


@pytest.fixture
def registry():
    return build_default_registry()


@pytest.fixture
def action_context(config: ConfigManager, language: LanguageManager,
                   database: Database, permissions: PermissionManager,
                   resolver: SafePathResolver) -> ActionContext:
    return ActionContext(
        config=config, language=language, database=database,
        permissions=permissions, paths=resolver, session_id="test", online=False,
    )


@pytest.fixture
def engine(config: ConfigManager, database: Database,
           resolver: SafePathResolver) -> Iterator[AarmaEngine]:
    """A fully wired engine confined to the scratch sandbox and offline by default."""
    config.update_runtime(network_mode="offline")
    instance = AarmaEngine(config, database=database, path_resolver=resolver,
                           session_id="test-session")
    yield instance
    instance.shutdown()
