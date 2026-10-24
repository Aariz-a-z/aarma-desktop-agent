"""Opening web pages and running web searches in the user's default browser.

Both actions funnel through :func:`security.validator.normalize_url`, so only
``http`` and ``https`` addresses ever reach the browser.  That check is what
stops "open this link" from becoming "run this local file": ``file:``,
``javascript:`` and custom application schemes are rejected before
:mod:`webbrowser` is involved.

AARMA does not embed a browser or scrape pages in this phase -- it hands the
address to whatever browser the user has already chosen.
"""

from __future__ import annotations

import webbrowser
from typing import Dict, List
from urllib.parse import quote_plus, urlparse

from actions.registry import (
    ActionContext,
    ActionResult,
    ActionSpec,
    ParameterSpec,
    ParameterType,
)
from providers.desktop import WINDOWS, active_browser
from providers.input import INPUT, UnknownKeyError, parse_chord
from security.permissions import Permission
from security.validator import RiskLevel, UnsafeInputError, normalize_url
from utils.logger import get_logger

logger = get_logger("actions.browser")

#: Search engines the user may choose in Settings.  ``{query}`` is substituted
#: with a URL-encoded query, so a search string can never alter the URL shape.
SEARCH_ENGINES: Dict[str, str] = {
    "google": "https://www.google.com/search?q={query}",
    "duckduckgo": "https://duckduckgo.com/?q={query}",
    "bing": "https://www.bing.com/search?q={query}",
}

DEFAULT_SEARCH_ENGINE = "google"

#: Sites users refer to by bare name, so "open youtube" does something sensible.
SITE_SHORTCUTS: Dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "github": "https://github.com",
    "wikipedia": "https://www.wikipedia.org",
    "stack overflow": "https://stackoverflow.com",
    "stackoverflow": "https://stackoverflow.com",
    "linkedin": "https://www.linkedin.com",
    "chatgpt": "https://chat.openai.com",
    "claude": "https://claude.ai",
}


def _open_in_browser(url: str) -> bool:
    """Hand ``url`` to the default browser.  Returns False if none is available."""
    try:
        return bool(webbrowser.open(url, new=2))
    except webbrowser.Error as exc:  # pragma: no cover - depends on the host
        logger.error("Browser launch failed", extra={"context": {"error": str(exc)}})
        return False


def _open_url(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    raw = str(params["url"]).strip()

    shortcut = SITE_SHORTCUTS.get(raw.lower())
    try:
        url = shortcut or normalize_url(raw)
    except UnsafeInputError as exc:
        return ActionResult.failure(str(exc), error_code="unsafe_url")

    if not _open_in_browser(url):
        return ActionResult.failure(
            "I couldn't find a browser to open that with.", error_code="no_browser"
        )

    host = urlparse(url).netloc or url
    logger.info("Opened URL", extra={"context": {"host": host}})
    return ActionResult.success(f"Opening {host}.", url=url, host=host)


def _search_site(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Search one site directly, using that site's own search URL.

    "Search YouTube for python tutorials" has an obvious shortest form -- open
    YouTube's results page -- and the alternative of opening the home page and
    then synthesising keystrokes into a search box is slower, fragile, and
    needs keyboard permission for something a URL already does.

    Only sites in :data:`ai.nlu.lexicon.SITE_SEARCH` are offered, because those
    are the ones whose search endpoint is actually known. Asking for a site
    that is not on that list is refused and redirected to a web search rather
    than guessed at, which would mean inventing a URL pattern.
    """
    from ai.nlu.lexicon import SITE_SEARCH, canonical_site

    raw_site = str(params.get("site") or "").strip()
    query = str(params.get("query") or "").strip()
    site = canonical_site(raw_site) or raw_site.lower()

    template = SITE_SEARCH.get(site)
    if template is None:
        known = ", ".join(sorted(SITE_SEARCH))
        return ActionResult.failure(
            f"I don't know how to search {raw_site} directly. I can search "
            f"{known}, or search the web for it instead.",
            error_code="site_search_unknown", site=site)
    if not query:
        return ActionResult.failure(f"What should I search {site} for?",
                                    error_code="missing_query", needs="query")

    url = template.format(query=quote_plus(query))
    if not _open_in_browser(url):
        return ActionResult.failure(
            "I couldn't find a browser to search with.", error_code="no_browser")

    logger.info("Site search opened", extra={"context": {"site": site}})
    return ActionResult.success(
        f"Searching {site} for \"{query}\".", url=url, site=site, query=query)


def _search_web(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    query = str(params["query"]).strip()
    engine_key = (ctx.config.config.runtime.search_engine or DEFAULT_SEARCH_ENGINE).lower()
    template = SEARCH_ENGINES.get(engine_key, SEARCH_ENGINES[DEFAULT_SEARCH_ENGINE])
    url = template.format(query=quote_plus(query))

    if not _open_in_browser(url):
        return ActionResult.failure(
            "I couldn't find a browser to search with.", error_code="no_browser"
        )

    logger.info("Web search opened", extra={"context": {"engine": engine_key}})
    return ActionResult.success(
        f"Searching the web for \"{query}\".", url=url, engine=engine_key, query=query
    )




# ------------------------------------------------------------- tab control
#
# AARMA does not drive the browser through an automation protocol.  Doing that
# means either a debugging port (which anything on the machine could then use
# to read the user's logged-in sessions) or an extension the user has to
# install, and neither is something to switch on unasked.  So tab control uses
# the keyboard shortcuts every browser already has.
#
# That choice has one dangerous consequence, and the whole of the code below
# exists to contain it: a keystroke goes to whatever has focus.  Ctrl+W closes
# a tab in Chrome and discards an unsaved buffer in some editors.  So every
# shortcut here is gated on :func:`providers.desktop.active_browser`, which
# resolves the foreground window and its owning process *first* and refuses
# when it is not a browser.  The shortcut is never sent optimistically and the
# result never claims a tab was closed unless a browser was in front.


def _window_provider(ctx: ActionContext):
    """The window provider, overridable by a test through the context."""
    return getattr(ctx, "window_provider", None) or WINDOWS


def _input_provider(ctx: ActionContext):
    return getattr(ctx, "input_provider", None) or INPUT


def _browser_shortcut(ctx: ActionContext, keys: str, *, success: str,
                      **data: object) -> ActionResult:
    """Send one shortcut, but only into a window known to be a browser."""
    window, reason = active_browser(_window_provider(ctx))
    if window is None:
        return ActionResult.failure(reason or "I can't tell what's in front.",
                                    error_code="not_a_browser")
    provider = _input_provider(ctx)
    if not provider.is_available():
        return ActionResult.failure(provider.availability().detail,
                                    error_code="input_unavailable")
    try:
        chord = parse_chord(keys)
    except UnknownKeyError as exc:  # pragma: no cover - the keys are literals
        return ActionResult.failure(str(exc), error_code="bad_shortcut")
    result = provider.press(chord)
    if not result.ok:
        return ActionResult.failure(result.message,
                                    error_code=result.error_code or "press_failed")
    logger.info("Browser shortcut sent", extra={"context": {
        "browser": window.browser, "keys": chord.spelling()}})
    return ActionResult.success(success.format(browser=window.browser),
                                browser=window.browser, keys=chord.spelling(),
                                window=window.title, **data)


def _new_tab(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    return _browser_shortcut(ctx, "ctrl+t", success="New tab open in {browser}.")


def _close_tab(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    return _browser_shortcut(ctx, "ctrl+w", success="Closed the tab in {browser}.")


def _reopen_tab(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    return _browser_shortcut(ctx, "ctrl+shift+t",
                             success="Reopened the last closed tab in {browser}.")


def _refresh_page(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    return _browser_shortcut(ctx, "f5", success="Refreshed the page in {browser}.")


def _go_back(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    return _browser_shortcut(ctx, "alt+left", success="Went back in {browser}.")


def _go_forward(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    return _browser_shortcut(ctx, "alt+right", success="Went forward in {browser}.")


def _next_tab(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    return _browser_shortcut(ctx, "ctrl+tab", success="Switched tab in {browser}.")


def _browser_status(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Say whether a browser is in front, so the user can tell before asking.

    Exists because "close this tab" failing with "that isn't a browser" is much
    easier to act on if you can also just ask what AARMA thinks is in front.
    """
    provider = _window_provider(ctx)
    if not provider.is_available():
        return ActionResult.failure(provider.availability().detail,
                                    error_code="windows_unavailable")
    window = provider.active_window()
    if window is None:
        return ActionResult.failure("I can't tell which window is in front.",
                                    error_code="no_active_window")
    if window.is_browser:
        return ActionResult.success(
            f"{window.browser} is in front — tab commands will go there.",
            browser=window.browser, is_browser=True, window=window.title)
    return ActionResult.success(
        f"{window.label()} is in front, and it isn't a browser, so I won't "
        "send browser shortcuts to it.",
        browser=None, is_browser=False, window=window.title)


#: Tab actions: (name, description, handler, examples).  Declared as data so
#: the specs below stay readable -- they differ only in wording.
_TAB_ACTIONS = (
    ("new_tab", "Open a new browser tab.", _new_tab,
     ("open a new tab", "new tab")),
    ("close_tab", "Close the browser tab you're looking at.", _close_tab,
     ("close this tab", "close the current tab")),
    ("reopen_tab", "Reopen the tab you just closed.", _reopen_tab,
     ("reopen the last tab", "undo close tab")),
    ("refresh_page", "Reload the page you're looking at.", _refresh_page,
     ("refresh this page", "reload the page")),
    ("go_back", "Go back to the previous page.", _go_back,
     ("go back", "back")),
    ("go_forward", "Go forward again.", _go_forward,
     ("go forward", "forward")),
    ("next_tab", "Switch to the next browser tab.", _next_tab,
     ("next tab", "switch tabs")),
)


def get_actions() -> List[ActionSpec]:
    specs: List[ActionSpec] = [
        ActionSpec(
            name="open_url",
            description="Open a website in your default browser.",
            category="Web",
            risk=RiskLevel.LOW,
            handler=_open_url,
            permissions=(Permission.BROWSER,),
            parameters=(
                ParameterSpec(
                    name="url",
                    type=ParameterType.STRING,
                    description="the website address to open",
                    max_length=2000,
                ),
            ),
            requires_internet=True,
            examples=("open youtube.com", "go to github", "open https://example.com"),
        ),
        ActionSpec(
            name="search_site",
            description="Search a specific website, such as YouTube or GitHub.",
            category="Web",
            risk=RiskLevel.LOW,
            handler=_search_site,
            permissions=(Permission.BROWSER, Permission.INTERNET),
            parameters=(
                ParameterSpec(name="site", type=ParameterType.STRING,
                              description="which site to search",
                              max_length=40),
                ParameterSpec(name="query", type=ParameterType.TEXT,
                              description="what to search for",
                              required=False, default="", max_length=400),
            ),
            requires_internet=True,
            examples=("search youtube for python tutorials",
                      "open python tutorials on youtube",
                      "find me python tutorials on youtube"),
        ),
        ActionSpec(
            name="search_web",
            description="Search the web and open the results in your browser.",
            category="Web",
            risk=RiskLevel.LOW,
            handler=_search_web,
            permissions=(Permission.BROWSER, Permission.INTERNET),
            parameters=(
                ParameterSpec(
                    name="query",
                    type=ParameterType.TEXT,
                    description="what to search for",
                    max_length=400,
                ),
            ),
            requires_internet=True,
            examples=("search for python tutorials",
                      "search the internet for the latest news about NVIDIA"),
        ),
    ]

    specs.extend(
        ActionSpec(
            name=name,
            description=description,
            category="Web",
            # LOW: every one of these is undone by another keystroke, and
            # closing a tab has Ctrl+Shift+T behind it. Asking "are you sure?"
            # before each would train the user to dismiss the prompt, which is
            # how a confirmation stops protecting anything that matters.
            risk=RiskLevel.LOW,
            handler=handler,
            permissions=(Permission.BROWSER, Permission.KEYBOARD),
            platforms=("win32",),
            examples=examples,
        )
        for name, description, handler, examples in _TAB_ACTIONS
    )
    specs.append(
        ActionSpec(
            name="browser_status",
            description="Say whether a browser window is in front.",
            category="Web",
            risk=RiskLevel.LOW,
            handler=_browser_status,
            platforms=("win32",),
            examples=("is a browser open", "what window is in front"),
        )
    )
    return specs
