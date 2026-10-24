"""How long a parked confirmation stays answerable.

A confirmation token is a promise that "yes" means one specific thing. The
failure this file exists for was found live: "close all applications" parked a
confirmation, the user said "no, take a screenshot instead", the screenshot ran
and the park survived -- so a bare "yes" several turns later still approved
closing every application. Nothing in the safety system caught it; what stopped
it was an unrelated validation bug, which is not a safety mechanism.

The lifetime tests drive a shutdown rather than a bulk close, because a
shutdown reaches the confirmation gate without needing a desktop: the autouse
conftest guard replaces the power runner, so the action can be approved in a
test without any possibility of a real one running.
"""

from __future__ import annotations

import pytest

from core.engine import AarmaEngine, ResponseKind
from providers import Availability, ProviderResult
from providers.desktop import WindowInfo
from security.permissions import Permission


@pytest.fixture
def armed(engine: AarmaEngine) -> AarmaEngine:
    """An engine allowed to reach the confirmation gate."""
    engine.permissions.grant(Permission.SYSTEM_CONTROL)
    return engine


def _park(engine: AarmaEngine):
    response = engine.handle_text("shut down the computer")
    assert response.kind is ResponseKind.CONFIRMATION_REQUIRED, response.text
    assert engine.confirmations.latest() is not None
    return response


class TestAParkDoesNotOutliveItsRequest:
    def test_running_a_different_action_cancels_the_park(
            self, armed: AarmaEngine) -> None:
        _park(armed)
        armed.handle_text("what time is it")
        assert armed.confirmations.latest() is None, \
            "a stale park lets a later bare 'yes' approve a forgotten action"

    def test_the_cancellation_is_said_out_loud(self, armed: AarmaEngine) -> None:
        """The user has no other way to learn "yes" no longer does that."""
        _park(armed)
        response = armed.handle_text("what time is it")
        assert "cancelled the confirmation" in response.text.lower()
        assert "shutdown_computer" in response.text

    def test_yes_afterwards_approves_nothing(self, armed: AarmaEngine) -> None:
        _park(armed)
        armed.handle_text("what time is it")
        assert not armed.handle_text("yes").ok

    def test_answering_no_immediately_still_works(self, armed: AarmaEngine) -> None:
        """The guard must not make a confirmation impossible to answer."""
        _park(armed)
        response = armed.handle_text("no")
        assert response.ok
        assert armed.confirmations.latest() is None

    # Note: there is deliberately no test that approves the *shutdown*. The
    # autouse conftest guard raises on a real power call, and it caught an
    # earlier draft of this file doing exactly that. Approval is proved below
    # against a bulk close driven by a fake desktop, where saying yes cannot
    # reach anything real.


# --------------------------------------------------------------------------
# What the prompt says
# --------------------------------------------------------------------------

class _Desktop:
    """A desktop with a known window list, so the summary has something to say.

    Built on the real ``WindowInfo`` and ``ProviderResult`` rather than
    look-alikes: a fake that answers a different shape than the thing it stands
    in for tests the fake, not the code.
    """

    def __init__(self, *windows: WindowInfo) -> None:
        self._windows = list(windows)
        self.closed: list = []

    def is_available(self) -> bool:
        return True

    def availability(self):
        return Availability(available=True)

    def list_windows(self):
        return list(self._windows)

    def close(self, window: WindowInfo) -> ProviderResult:
        """Records instead of closing, so approval can be tested end to end."""
        self.closed.append(window.process_name)
        return ProviderResult.success(f"Closed {window.label()}.")


@pytest.fixture
def desktop(armed: AarmaEngine, monkeypatch: pytest.MonkeyPatch) -> AarmaEngine:
    import actions.desktop_control as desktop_control

    fake = _Desktop(
        WindowInfo(handle=1, title="GitHub - Chrome", process_name="chrome.exe"),
        WindowInfo(handle=2, title="notes.txt - Notepad", process_name="notepad.exe"))
    monkeypatch.setattr(desktop_control, "_windows", lambda ctx: fake)
    armed.fake_desktop = fake       # type: ignore[attr-defined]
    return armed


class TestTheConfirmationSaysWhatItWouldDo:
    def test_a_bulk_close_names_the_applications(self, desktop: AarmaEngine) -> None:
        """The list is the decision; a count alone cannot be judged."""
        response = desktop.handle_text("close all applications")
        assert response.kind is ResponseKind.CONFIRMATION_REQUIRED, response.text
        assert "Chrome" in response.text
        assert "Notepad" in response.text
        assert "unsaved work" in response.text.lower()

    def test_an_exception_is_named_as_spared(self, desktop: AarmaEngine) -> None:
        response = desktop.handle_text("close everything except Chrome")
        assert response.kind is ResponseKind.CONFIRMATION_REQUIRED, response.text
        assert "Leaving open: chrome" in response.text
        assert "Chrome," not in response.text, "a spared app must not be listed to close"

    def test_no_blank_parameters_are_rendered(self, desktop: AarmaEngine) -> None:
        """The prompt used to read "(except: )" -- a default, not a choice."""
        response = desktop.handle_text("close all applications")
        assert "except: )" not in response.text

    def test_approval_still_reaches_the_action(self, desktop: AarmaEngine) -> None:
        """The staleness guard must not make a confirmation unanswerable.

        Driven against the fake desktop, so "yes" closes recorded windows
        rather than anything real.
        """
        parked = desktop.handle_text("close all applications")
        assert parked.kind is ResponseKind.CONFIRMATION_REQUIRED, parked.text
        response = desktop.handle_text("yes")
        assert response.ok, response.text
        assert desktop.fake_desktop.closed, "approval did not reach the handler"
        assert desktop.confirmations.latest() is None
