"""Lock, sleep, restart and shut down.

Risk grading here is the whole design, so it is worth stating plainly:

* **Lock** is LOW. It interrupts nothing, loses nothing, and is undone by
  typing a password. Asking "are you sure?" before locking would train the user
  to dismiss confirmations without reading them, which is how a confirmation
  prompt stops working.
* **Sleep** is MEDIUM. Nothing is lost, but it ends the session's
  responsiveness and a misheard command is disruptive.
* **Restart and shutdown** are HIGH. They close every open application, and any
  unsaved work goes with them. These are exactly the commands a speech
  recogniser must never be trusted to have heard correctly on its own.

The confirmation itself is not implemented here. The registry marks the risk,
and :class:`~security.confirmation.ConfirmationBroker` in the engine decides
what needs confirming -- one policy in one place, rather than each handler
inventing its own prompt.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional

from actions.registry import ActionContext, ActionResult, ActionSpec, ParameterSpec
from security.permissions import Permission
from security.validator import RiskLevel
from utils.logger import get_logger

logger = get_logger("actions.power")

#: Seconds Windows waits before a scheduled shutdown or restart.  Non-zero on
#: purpose: it leaves a window in which `shutdown /a` can abort, which is the
#: last line of defence if a confirmation was given by mistake.
GRACE_SECONDS = 20


class UnsafePowerExecution(BaseException):
    """A real power command was about to run under a test runner.

    A ``BaseException`` rather than an ``Exception``, and that is the whole
    point of it.  :func:`_execute` below and
    :meth:`core.engine.AarmaEngine._invoke` both catch broad ``Exception`` so
    that one failing action cannot crash the assistant -- correct in
    production, and exactly wrong here, because it would turn "a test tried to
    shut this machine down" into a quiet failed-action result that the suite
    reports as a pass.  Deriving from ``BaseException`` makes it travel through
    every one of those handlers and fail the run out loud.
    """


def _under_test() -> bool:
    """Whether this process is a test run rather than a real session.

    Two independent signals, OR-ed, because this decides whether to fail
    closed: ``PYTEST_CURRENT_TEST`` is set by pytest around every test call,
    and the module check also covers collection, fixtures and conftest import,
    where the environment variable is not yet set.

    Production code does not normally know it is being tested, and this is a
    deliberate exception to that rule.  The alternative -- relying on every
    test to remember to inject a fake runner -- is a convention, and a
    convention is precisely what the next test breaks.
    """
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) or "pytest" in sys.modules


def _refuse_real_execution(command: Any = None, **_kwargs: Any):
    """Fail closed.  Never returns; never runs anything."""
    raise UnsafePowerExecution(
        "A power command was about to run for real inside a test: "
        f"{list(command) if command else command!r}. Tests must inject a fake "
        "through ActionContext.process_runner. Refusing to touch this machine."
    )


def _runner(ctx: ActionContext) -> Callable[..., Any]:
    """The subprocess runner for this call.

    Resolution order, and the order matters:

    1. A runner injected into the context -- what every test that legitimately
       exercises these actions supplies, and what lets them assert on the exact
       argv without a machine changing state.
    2. Under a test runner with nothing injected: :func:`_refuse_real_execution`.
       This is the guard that was missing.  A mutation test that disables the
       confirmation gate now reaches a refusal instead of ``shutdown /s``.
    3. Otherwise: the real :func:`subprocess.run`.

    Step 2 exists because step 1 failed once in practice. A mutation test set
    ``RiskPolicy.requires_confirmation`` to return ``False`` and then sent
    "shut down the computer" through a real engine; the engine builds its own
    context, which carries no injected runner, so this function handed back the
    real ``subprocess.run`` and the machine shut down twenty seconds later --
    while the suite reported a pass.
    """
    injected = getattr(ctx, "process_runner", None)
    if injected is not None:
        return injected
    if _under_test():
        logger.error("Blocked a real power command inside a test run")
        return _refuse_real_execution
    return subprocess.run


def _windows(ctx: ActionContext) -> bool:
    return (getattr(ctx, "platform_system", None) or platform.system()) == "Windows"


def _unsupported(ctx: ActionContext, what: str) -> ActionResult:
    system = getattr(ctx, "platform_system", None) or platform.system()
    return ActionResult.failure(f"I can only {what} on Windows, and this is {system}.",
                                error_code="unsupported_platform")


def _execute(ctx: ActionContext, command: List[str], *, success: str,
             failure: str, **data) -> ActionResult:
    run = _runner(ctx)
    try:
        completed = run(command, capture_output=True, text=True, timeout=15,
                        check=False)
    except FileNotFoundError:
        logger.error("Power command missing", extra={"context": {"command": command[0]}})
        return ActionResult.failure(f"{failure} -- {command[0]} isn't available.",
                                    error_code="command_missing")
    except subprocess.TimeoutExpired:
        return ActionResult.failure(f"{failure} -- the command timed out.",
                                    error_code="timeout")
    except Exception as exc:  # noqa: BLE001 - one failed action, not a crash
        logger.error("Power command failed", extra={
            "context": {"command": command, "error": str(exc)}})
        return ActionResult.failure(failure, error_code="command_failed")

    code = getattr(completed, "returncode", 0)
    if code not in (0, None):
        detail = (getattr(completed, "stderr", "") or "").strip().splitlines()
        # The OS message can name paths and accounts; keep the first line only.
        hint = f" ({detail[0][:120]})" if detail else ""
        logger.warning("Power command refused", extra={
            "context": {"command": command, "returncode": code}})
        return ActionResult.failure(f"{failure}{hint}", error_code="command_refused")
    return ActionResult.success(success, **data)


def _lock(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    if not _windows(ctx):
        return _unsupported(ctx, "lock the screen")
    return _execute(ctx, ["rundll32.exe", "user32.dll,LockWorkStation"],
                    success="Locked.", failure="I couldn't lock the screen.")


def _sleep(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    if not _windows(ctx):
        return _unsupported(ctx, "put the machine to sleep")
    # The documented invocation. The trailing 0,1,0 asks for sleep rather than
    # hibernate and declines to force applications closed.
    return _execute(ctx, ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
                    success="Going to sleep.",
                    failure="I couldn't put the machine to sleep.")


def _restart(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    if not _windows(ctx):
        return _unsupported(ctx, "restart the machine")
    return _execute(
        ctx, ["shutdown", "/r", "/t", str(GRACE_SECONDS)],
        success=f"Restarting in {GRACE_SECONDS} seconds. "
                "Say \"cancel shutdown\" if that was a mistake.",
        failure="I couldn't restart the machine.",
        grace_seconds=GRACE_SECONDS)


def _shutdown(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    if not _windows(ctx):
        return _unsupported(ctx, "shut the machine down")
    return _execute(
        ctx, ["shutdown", "/s", "/t", str(GRACE_SECONDS)],
        success=f"Shutting down in {GRACE_SECONDS} seconds. "
                "Say \"cancel shutdown\" if that was a mistake.",
        failure="I couldn't shut the machine down.",
        grace_seconds=GRACE_SECONDS)


def _cancel_shutdown(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Abort a pending shutdown or restart.  LOW risk: it only ever prevents."""
    if not _windows(ctx):
        return _unsupported(ctx, "cancel a shutdown")
    return _execute(ctx, ["shutdown", "/a"],
                    success="Cancelled the pending shutdown.",
                    failure="There wasn't a shutdown to cancel.")


def _sign_out(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    if not _windows(ctx):
        return _unsupported(ctx, "sign out")
    return _execute(ctx, ["shutdown", "/l"], success="Signing out.",
                    failure="I couldn't sign out.")


def get_actions() -> List[ActionSpec]:
    """Every power action, each behind the system-control permission.

    Requiring :data:`~security.permissions.Permission.SYSTEM_CONTROL` here is a
    second, independent barrier.  Until now confirmation was the *only* thing
    between a misheard sentence and a real shutdown, and a single barrier is one
    bug away from none -- which is how a mutation test that disabled exactly
    that barrier managed to power the machine off.  The permission is off by
    default, so on a fresh install these actions cannot run at all, whatever
    the confirmation policy says.

    Two actions are deliberately left ungated, for the same underlying reason:
    a barrier is only worth having where the thing behind it can cost the user
    something.

    ``cancel_shutdown`` only ever *prevents* a shutdown.  Needing a permission
    to abort one -- possibly started by something else entirely -- would make
    the escape hatch harder to reach than the hazard.

    ``lock_screen`` interrupts nothing, loses nothing, and is undone by typing
    a password.  Gating it would mean "lock my PC" fails on a fresh install
    with a permissions error, which teaches the user that AARMA refuses simple
    things -- and a user who has learned to expect refusals is a user who stops
    reading them.  The permission exists to protect against *loss*, and there
    is none here.
    """
    system = (Permission.SYSTEM_CONTROL,)
    return [
        ActionSpec(
            name="lock_screen",
            description="Lock the screen.",
            category="Power", risk=RiskLevel.LOW, handler=_lock,
            examples=("lock my PC", "lock the computer", "lock my screen"),
        ),
        ActionSpec(
            name="cancel_shutdown",
            description="Cancel a shutdown or restart that hasn't happened yet.",
            category="Power", risk=RiskLevel.LOW, handler=_cancel_shutdown,
            examples=("cancel shutdown", "stop the restart", "never mind"),
        ),
        ActionSpec(
            name="sleep_computer",
            description="Put the computer to sleep.",
            category="Power", risk=RiskLevel.MEDIUM, handler=_sleep,
            permissions=system,
            examples=("put the computer to sleep", "go to sleep"),
        ),
        ActionSpec(
            name="sign_out",
            description="Sign out of Windows.",
            category="Power", risk=RiskLevel.HIGH, handler=_sign_out,
            permissions=system,
            examples=("sign me out", "log out"),
        ),
        ActionSpec(
            name="restart_computer",
            description="Restart the computer, closing everything that's open.",
            category="Power", risk=RiskLevel.HIGH, handler=_restart,
            permissions=system,
            examples=("restart the computer", "reboot my PC"),
        ),
        ActionSpec(
            name="shutdown_computer",
            description="Shut the computer down, closing everything that's open.",
            category="Power", risk=RiskLevel.HIGH, handler=_shutdown,
            permissions=system,
            examples=("shut down the computer", "turn off my PC"),
        ),
    ]
