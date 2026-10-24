"""User interfaces.

Two front ends ship in Phase 1 -- the styled desktop dashboard
(:mod:`ui.dashboard`) and the text interface (:mod:`ui.console`) -- and both
are thin.  Neither one parses commands, checks permissions or touches the
operating system; they render what :class:`core.engine.AarmaEngine` returns.

Keeping the interfaces this thin is what specification section 36 means by
"the GUI should be separated from the core AI engine": swapping the toolkit, or
adding the voice loop as a third front end, does not put any security logic at
risk.
"""

__all__ = ["console", "dashboard", "responsive", "setup_wizard", "theme", "widgets"]
