"""Screen capture.

Gated behind the ``screenshot`` permission, which is **off by default** -- a
desktop agent that can silently photograph the screen is exactly the kind of
capability that should require a deliberate opt-in.

Two things this file is careful about:

**Nothing is ever overwritten.**  Timestamps to the second are not unique:
"take a screenshot" twice in the same second used to produce one file, and the
second capture silently replaced the first.  :func:`unique_path` walks a
counter suffix until it finds a name that does not exist, so a capture either
creates a new file or fails -- it never destroys an earlier one.

**The reported path is the file that exists.**  The save is what is reported,
not the intention to save, and the size is read back off disk.  A "screenshot
saved" that names a path with nothing at it would be undetectable until the
user went looking.

Pillow is an optional dependency.  If it is missing the actions stay registered
-- so the user gets a clear explanation rather than "unknown command" -- but
report honestly that the component is unavailable.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from actions.registry import ActionContext, ActionResult, ActionSpec
from providers.desktop import WINDOWS
from security.permissions import Permission
from security.validator import RiskLevel
from utils.logger import get_logger
from utils.paths import get_paths

logger = get_logger("actions.screenshot")

#: Upper bound on the collision suffix.  Reaching it means something is wrong
#: with the directory, and failing is better than looping.
MAX_NAME_ATTEMPTS = 1000


def is_available() -> bool:
    """True when a screen-capture backend is installed on this machine.

    Checked without importing the module, so a probe from the settings UI does
    not pull Pillow into memory for a capability the user may never use.
    """
    try:
        return importlib.util.find_spec("PIL.ImageGrab") is not None
    except (ImportError, ValueError):
        return False


def unique_path(directory: Path, stem: str, suffix: str = ".png") -> Path:
    """A path in ``directory`` that does not exist yet.

    Deliberately a loop over candidate names rather than an ``exists`` check
    followed by a write: between the two, a second capture could claim the
    name.  The final ``open(..., "xb")`` in :func:`_save` is what actually
    makes the claim atomic; this just finds a sensible candidate first.
    """
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    for index in range(2, MAX_NAME_ATTEMPTS):
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise OSError(f"No free filename for {stem} in {directory}.")


def _save(image: Any, destination: Path) -> int:
    """Write ``image`` to a path that must not already exist.  Returns bytes.

    ``"xb"`` fails if the file appeared in the meantime, which turns a silent
    overwrite into an error the caller can report.
    """
    with open(destination, "xb") as handle:
        image.save(handle, "PNG")
    return destination.stat().st_size


def _capture(bbox: Optional[Tuple[int, int, int, int]], stem: str
             ) -> Tuple[Optional[Path], Optional[int], Optional[str]]:
    """Grab the screen (or a region) and save it.  Returns (path, size, error)."""
    try:
        from PIL import ImageGrab
    except ImportError:
        return None, None, (
            "Screen capture needs the Pillow component, which isn't installed. "
            "Run: pip install Pillow")

    target_dir = get_paths().screenshot_dir
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        destination = unique_path(target_dir, stem)
        image = ImageGrab.grab(bbox=bbox) if bbox else ImageGrab.grab()
        size = _save(image, destination)
    except FileExistsError:
        logger.error("Screenshot name was taken between check and write")
        return None, None, "Something else wrote that file first. Try again."
    except Exception as exc:  # noqa: BLE001 - backend errors vary by platform
        logger.error("Screenshot failed", extra={"context": {"error": str(exc)}})
        return None, None, "I couldn't capture the screen."
    return destination, size, None


def _stem(prefix: str = "aarma") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def _take_screenshot(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    destination, size, error = _capture(None, _stem())
    if error is not None or destination is None:
        return ActionResult.failure(error or "I couldn't capture the screen.",
                                    error_code="capture_failed")
    logger.info("Screenshot saved", extra={"context": {
        "file": destination.name, "bytes": size}})
    return ActionResult.success(f"Screenshot saved to {destination}.",
                                path=str(destination), name=destination.name,
                                bytes=size, region="screen")


def _capture_window(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    """Capture just the window in front.

    Needs the window manager as well as Pillow, because the crop box has to be
    the window's real rectangle.  If either is missing the action says which --
    it does not fall back to a full-screen grab and call it a window capture,
    which would hand the user a file that is not what they asked for.
    """
    provider = getattr(ctx, "window_provider", None) or WINDOWS
    if not provider.is_available():
        return ActionResult.failure(
            provider.availability().detail
            + " I can still take a full-screen shot if you'd like.",
            error_code="windows_unavailable")
    window = provider.active_window()
    if window is None:
        return ActionResult.failure("I can't tell which window is in front.",
                                    error_code="no_active_window")
    box = provider.rect(window)
    if box is None:
        return ActionResult.failure(
            f"Windows wouldn't tell me where {window.label()} is on screen, "
            "so I can't crop to it.", error_code="no_window_rect")

    destination, size, error = _capture(box, _stem("aarma-window"))
    if error is not None or destination is None:
        return ActionResult.failure(error or "I couldn't capture that window.",
                                    error_code="capture_failed")
    logger.info("Window screenshot saved", extra={"context": {
        "file": destination.name, "bytes": size}})
    return ActionResult.success(
        f"Captured {window.label()} to {destination}.",
        path=str(destination), name=destination.name, bytes=size,
        region="window", window=window.title)


def get_actions() -> List[ActionSpec]:
    return [
        ActionSpec(
            name="take_screenshot",
            description="Capture the screen and save it as a PNG.",
            category="Desktop",
            risk=RiskLevel.LOW,
            handler=_take_screenshot,
            permissions=(Permission.SCREENSHOT,),
            examples=("take a screenshot", "capture my screen",
                      "save a screenshot"),
        ),
        ActionSpec(
            name="capture_window",
            description="Capture just the window in front.",
            category="Desktop",
            risk=RiskLevel.LOW,
            handler=_capture_window,
            permissions=(Permission.SCREENSHOT,),
            platforms=("win32",),
            examples=("capture this window", "screenshot this window"),
        ),
    ]
