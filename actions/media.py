"""Volume and playback actions.

All LOW risk: changing the volume is instantly obvious and instantly
reversible, which is the test for whether something needs confirming.

Volume is stepped, not set. :meth:`providers.media.MediaProvider.set_volume`
explains why it refuses an absolute level rather than approximating one.
"""

from __future__ import annotations

from typing import Dict, List

from actions.registry import ActionContext, ActionResult, ActionSpec, ParameterSpec
from providers.media import DEFAULT_STEPS, MEDIA
from security.validator import RiskLevel
from utils.logger import get_logger

logger = get_logger("actions.media")


def _provider(ctx: ActionContext):
    return getattr(ctx, "media_provider", None) or MEDIA


def _report(result) -> ActionResult:
    """Turn a provider result into an action result without inventing success."""
    if result.ok:
        return ActionResult.success(result.message, **result.data)
    return ActionResult.failure(result.message,
                                error_code=result.error_code or "media_failed",
                                **result.data)


def _steps(params: Dict[str, object]) -> int:
    raw = params.get("amount")
    if raw in (None, ""):
        return DEFAULT_STEPS
    try:
        return max(1, min(50, int(raw)))
    except (TypeError, ValueError):
        return DEFAULT_STEPS


def _volume_up(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    result = _provider(ctx).volume_up(_steps(params))
    return _report(result) if not result.ok else ActionResult.success(
        "Turned the volume up.", **result.data)


def _volume_down(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    result = _provider(ctx).volume_down(_steps(params))
    return _report(result) if not result.ok else ActionResult.success(
        "Turned the volume down.", **result.data)


def _set_volume(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    try:
        percent = int(params["percent"])
    except (KeyError, TypeError, ValueError):
        return ActionResult.failure("I need a number between 0 and 100.",
                                    error_code="invalid_volume")
    if not 0 <= percent <= 100:
        return ActionResult.failure("Volume has to be between 0 and 100.",
                                    error_code="invalid_volume")
    return _report(_provider(ctx).set_volume(percent))


def _mute(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    result = _provider(ctx).mute()
    if not result.ok:
        return _report(result)
    # The key is a toggle and Windows will not say which way it went, so the
    # reply must not claim a direction it cannot know.
    return ActionResult.success("Toggled mute.", **result.data)


def _play_pause(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    result = _provider(ctx).play_pause()
    return _report(result) if not result.ok else ActionResult.success(
        "Sent play/pause.", **result.data)


def _next_track(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    result = _provider(ctx).next_track()
    return _report(result) if not result.ok else ActionResult.success(
        "Skipped to the next track.", **result.data)


def _previous_track(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    result = _provider(ctx).previous_track()
    return _report(result) if not result.ok else ActionResult.success(
        "Went back a track.", **result.data)


def _stop_playback(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    result = _provider(ctx).stop()
    return _report(result) if not result.ok else ActionResult.success(
        "Stopped playback.", **result.data)


def get_actions() -> List[ActionSpec]:
    amount = ParameterSpec(name="amount", description="how many steps",
                           required=False, default=None)
    return [
        ActionSpec(
            name="volume_up", description="Turn the volume up.",
            category="Media", risk=RiskLevel.LOW, handler=_volume_up,
            parameters=(amount,),
            examples=("turn the volume up", "increase the volume", "louder"),
        ),
        ActionSpec(
            name="volume_down", description="Turn the volume down.",
            category="Media", risk=RiskLevel.LOW, handler=_volume_down,
            parameters=(amount,),
            examples=("turn the volume down", "turn the sound down a little"),
        ),
        ActionSpec(
            name="set_volume", description="Set the volume to a percentage.",
            category="Media", risk=RiskLevel.LOW, handler=_set_volume,
            parameters=(ParameterSpec(name="percent",
                                      description="0 to 100"),),
            examples=("set volume to 30 percent", "set the volume to 40"),
        ),
        ActionSpec(
            name="mute", description="Toggle mute.",
            category="Media", risk=RiskLevel.LOW, handler=_mute,
            examples=("mute", "unmute", "mute the sound"),
        ),
        ActionSpec(
            name="play_pause", description="Play or pause whatever is playing.",
            category="Media", risk=RiskLevel.LOW, handler=_play_pause,
            examples=("pause the music", "play", "pause"),
        ),
        ActionSpec(
            name="next_track", description="Skip to the next track.",
            category="Media", risk=RiskLevel.LOW, handler=_next_track,
            examples=("next song", "play the next song", "skip"),
        ),
        ActionSpec(
            name="previous_track", description="Go back to the previous track.",
            category="Media", risk=RiskLevel.LOW, handler=_previous_track,
            examples=("previous song", "go back a track"),
        ),
        ActionSpec(
            name="stop_playback", description="Stop playback.",
            category="Media", risk=RiskLevel.LOW, handler=_stop_playback,
            examples=("stop the music",),
        ),
    ]
