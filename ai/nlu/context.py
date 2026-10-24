"""What is true right now, so "close this" can mean something.

A third of natural instructions point at the situation rather than naming a
target: *close this*, *go back*, *refresh*, *open another one*.  None of them is
resolvable from the sentence alone, and the difference between a useful
assistant and a dangerous one is whether it admits that.

So context here is deliberately thin.  It holds the front window and the last
thing that was resolved, and nothing else.  A richer memory would let an
instruction from four turns ago supply a target for this one, which is exactly
the failure the pending-action store was built narrow to avoid.

The important method is :meth:`ResolutionContext.referent_kind`.  When it
returns ``None`` the resolver must ask rather than assume -- "close this" with
nothing in front is a question, not a command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from utils.logger import get_logger

logger = get_logger("ai.nlu.context")

#: Process names that mean "a browser is in front".  Read from the desktop
#: provider rather than duplicated, so the two can never disagree.
def _browser_names() -> Dict[str, str]:
    from providers.desktop import BROWSER_PROCESSES

    return BROWSER_PROCESSES


@dataclass
class ResolutionContext:
    """The situation a sentence is being interpreted against.

    Every field is optional and every consumer must cope with it being absent:
    a text-only session has no window manager, and a headless test has no
    desktop at all.  Absent context narrows what can be resolved; it never
    causes a guess.
    """

    #: Title of the window in front, when known.
    active_window_title: str = ""
    #: Process that owns it, e.g. ``chrome.exe``.
    active_process: str = ""
    #: The action resolved on the previous turn, for "another one" / "again".
    last_action: str = ""
    #: Entities from the previous turn, for "search machine learning instead".
    last_entities: Dict[str, Any] = field(default_factory=dict)

    # -- what is in front --------------------------------------------------

    @property
    def browser(self) -> Optional[str]:
        """The browser in front, or ``None`` when it is not a browser."""
        if not self.active_process:
            return None
        return _browser_names().get(self.active_process.lower())

    @property
    def has_window_context(self) -> bool:
        return bool(self.active_process or self.active_window_title)

    def referent_kind(self) -> Optional[str]:
        """What "this" refers to: ``"tab"``, ``"window"``, or ``None``.

        ``None`` is the honest answer whenever the front window is unknown, and
        the resolver turns it into a question.  Guessing "tab" would mean
        "close this" closes a browser tab while the user is looking at an
        unsaved document.
        """
        if not self.has_window_context:
            return None
        return "tab" if self.browser else "window"

    def as_dict(self) -> Dict[str, Any]:
        return {"active_window_title": self.active_window_title,
                "active_process": self.active_process,
                "browser": self.browser,
                "last_action": self.last_action}


#: A callable returning the live context.  Injected so the resolver never
#: reaches for a window manager itself -- which keeps it testable and keeps a
#: text-only session from constructing a desktop provider it will not use.
ContextProvider = Callable[[], ResolutionContext]


def empty_context() -> ResolutionContext:
    """The context of a session that cannot see a desktop."""
    return ResolutionContext()


def from_window_provider(provider: Any = None) -> ResolutionContext:
    """Read the live desktop, degrading to empty context on any failure.

    Wrapped broadly on purpose: this runs on the path of every unmatched
    sentence, and a window manager that is unavailable, slow or throwing must
    cost the user a less specific answer rather than an error.
    """
    try:
        if provider is None:
            from providers.desktop import WINDOWS

            provider = WINDOWS
        if not provider.is_available():
            return ResolutionContext()
        window = provider.active_window()
        if window is None:
            return ResolutionContext()
        return ResolutionContext(active_window_title=window.title,
                                 active_process=window.process_name)
    except Exception as exc:  # noqa: BLE001 - context is a convenience
        logger.debug("Could not read window context",
                     extra={"context": {"error": str(exc)}})
        return ResolutionContext()


__all__ = ["ContextProvider", "ResolutionContext", "empty_context",
           "from_window_provider"]
