"""Actions that reconfigure the assistant itself.

"Change your name to Nova" and "speak Hindi" are, architecturally, ordinary
tool calls: they go through the same registry, permission check and risk
grading as opening a browser.  Routing them this way -- instead of
special-casing them in the parser -- means the GUI, the text interface and the
future voice pipeline all reconfigure the assistant through one audited path.

Changing settings is graded MEDIUM (specification section 11: "modify
non-sensitive settings"), so it is logged and reversible but not confirmed.
"""

from __future__ import annotations

from typing import Dict, List

from actions.registry import ActionContext, ActionResult, ActionSpec, ParameterSpec
from config.settings import (
    ASSISTANT_PROFILES,
    normalise_wake_phrase,
    wake_phrase_for_name,
)
from language.manager import LANGUAGE_CATALOG, LanguageStatus
from security.validator import RiskLevel, UnsafeInputError, ensure_text
from utils.logger import get_logger

logger = get_logger("actions.assistant")

MAX_NAME_LENGTH = 32


def _wake_note(phrase: str, name: str) -> str:
    """One honest sentence about whether that phrase can actually wake me.

    Wake-word models are trained per phrase, so renaming the assistant does not
    conjure a detector for the new name.  Saying so here is the difference
    between a setting that works and a setting that only looks like it does.
    """
    phrase = normalise_wake_phrase(phrase)
    if not phrase:
        return ""
    try:
        from speech.wakeword import REGISTRY
    except Exception:  # noqa: BLE001 - speech is optional at this layer
        return ""
    if REGISTRY.provider_for(phrase) is not None:
        return f' You can wake me by saying "{phrase}".'
    # Say what is missing and what to do about it.  "It didn't work" without a
    # next step is the least useful true thing an assistant can say -- and
    # claiming voice wake-up works when no model exists would be worse still.
    return (f" I haven't learned the {name} wake word yet, so voice wake-up is "
            "off until you teach me. Typing still works. You can teach me in "
            "Settings, or run: python main.py --enrol-wake-word")


def _set_assistant_name(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    raw = str(params["name"])
    try:
        name = ensure_text(raw, field="name", max_length=MAX_NAME_LENGTH)
    except UnsafeInputError as exc:
        return ActionResult.failure(str(exc), error_code="invalid_name")

    if not any(ch.isalpha() for ch in name):
        return ActionResult.failure("A name needs at least one letter.", error_code="invalid_name")

    previous = ctx.config.config.assistant.name
    # The wake word tracks the name unless the user has customised it away from
    # the previous default, so "hey nova" starts working after a rename.
    assistant = ctx.config.config.assistant
    updates: Dict[str, object] = {"name": name, "profile_key": "custom"}
    phrase = normalise_wake_phrase(assistant.wake_word)
    if phrase == wake_phrase_for_name(previous):
        updates["wake_word"] = wake_phrase_for_name(name)
    ctx.config.update_assistant(**updates)
    # Re-read rather than trusting the update: the reply may only claim the
    # rename happened if the settings layer actually stored it.
    stored = ctx.config.config.assistant
    if stored.name != name:
        return ActionResult.failure("The new name could not be saved.",
                                    error_code="save_failed")

    logger.info("Assistant renamed", extra={"context": {"from": previous, "to": name}})
    return ActionResult.success(
        ctx.language.t("response.name_changed", assistant=name)
        + _wake_note(stored.wake_word, name),
        name=name, previous=previous, wake_word=stored.wake_word,
    )


def _set_assistant_profile(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    key = str(params["profile"]).strip().lower()
    profile = ASSISTANT_PROFILES.get(key)
    if profile is None:
        options = ", ".join(f"{p.display_name} ({p.key})" for p in ASSISTANT_PROFILES.values())
        return ActionResult.failure(f"I don't have a '{key}' profile. Available: {options}.",
                                    error_code="unknown_profile")

    from config.settings import AssistantIdentity

    ctx.config.update(assistant=AssistantIdentity.from_profile(profile))
    logger.info("Assistant profile switched", extra={"context": {"profile": profile.key}})
    return ActionResult.success(
        ctx.language.t("response.name_changed", assistant=profile.display_name)
        + _wake_note(profile.wake_word, profile.display_name),
        profile=profile.key, name=profile.display_name,
    )


def _set_language(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    requested = str(params["language"])
    scope = str(params.get("scope") or "all")

    info = ctx.language.set_language(requested, scope=scope)
    if info is None:
        supported = ", ".join(entry.english_name for entry in LANGUAGE_CATALOG.values())
        return ActionResult.failure(
            ctx.language.t("response.language_unknown", value=requested)
            + f" I currently know: {supported}.",
            error_code="unknown_language",
        )

    key = ("response.language_changed" if info.status is LanguageStatus.TRANSLATED
           else "response.language_declared")
    return ActionResult.success(
        ctx.language.translate(key, language=info.english_name),
        language=info.code, status=info.status.value,
    )


def _set_wake_word(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    try:
        phrase = ensure_text(str(params["phrase"]), field="phrase", max_length=40).lower()
    except UnsafeInputError as exc:
        return ActionResult.failure(str(exc), error_code="invalid_wake_word")

    if len(phrase.split()) > 4:
        return ActionResult.failure("A wake word works best at four words or fewer.",
                                    error_code="invalid_wake_word")

    phrase = normalise_wake_phrase(phrase)
    ctx.config.update_assistant(wake_word=phrase)
    logger.info("Wake word changed")
    return ActionResult.success(
        f"Wake word set to \"{phrase}\"." + _wake_note(phrase, ctx.config.config.assistant.name),
        wake_word=phrase,
    )


def _list_languages(ctx: ActionContext, params: Dict[str, object]) -> ActionResult:
    current = ctx.language.interface_language
    lines = []
    for info in ctx.language.available():
        marker = "*" if info.code == current else " "
        note = "fully translated" if info.status is LanguageStatus.TRANSLATED else \
               f"selectable now, translation planned for Phase {info.phase}"
        lines.append(f" {marker} {info.english_name} ({info.code}) — {note}")
    return ActionResult.success("Languages I know about:\n" + "\n".join(lines), current=current)


def _set_voice_presentation(ctx: ActionContext,
                            params: Dict[str, object]) -> ActionResult:
    """Switch the whole voice profile, the same way the settings screen does.

    Deliberately delegates to :meth:`ConfigManager.set_voice_presentation`
    rather than writing the fields here. A command-only path would be a second
    place that decides what "male" means, and the two would drift -- which is
    how the configuration ended up claiming a female profile while a male voice
    spoke.
    """
    wanted = str(params.get("presentation") or "").strip().lower()
    changed, explanation = ctx.config.set_voice_presentation(wanted)
    if not changed:
        # "Already using that" is not a failure, and neither reading is a
        # reason to pretend the voice changed when it did not.
        already = explanation.startswith("I'm already")
        return (ActionResult.success(explanation) if already
                else ActionResult.failure(explanation,
                                          error_code="voice_unavailable"))

    from speech.voices import get_profile

    profile = get_profile(ctx.config.config.assistant.voice_id)
    logger.info("Voice presentation set", extra={"context": {
        "presentation": wanted, "voice": profile.voice_id}})
    return ActionResult.success(
        f"{explanation} I'll sound like this from now on.",
        presentation=wanted, voice_id=profile.voice_id,
        voice_name=profile.display_name)


def get_actions() -> List[ActionSpec]:
    return [
        ActionSpec(
            name="set_assistant_name",
            description="Change what you call me.",
            category="Assistant",
            risk=RiskLevel.MEDIUM,
            handler=_set_assistant_name,
            parameters=(
                ParameterSpec(name="name", description="my new name", max_length=MAX_NAME_LENGTH),
            ),
            examples=("change your name to Nova", "call yourself Iris"),
        ),
        ActionSpec(
            name="set_assistant_profile",
            description="Switch between the built-in assistant profiles.",
            category="Assistant",
            risk=RiskLevel.MEDIUM,
            handler=_set_assistant_profile,
            parameters=(
                ParameterSpec(
                    name="profile",
                    description="which profile to use",
                    choices=tuple(ASSISTANT_PROFILES),
                ),
            ),
            examples=("switch to the Aariz profile",),
        ),
        ActionSpec(
            name="set_voice_presentation",
            description="Switch between the male and female voice profiles.",
            category="Assistant",
            risk=RiskLevel.LOW,
            handler=_set_voice_presentation,
            parameters=(
                ParameterSpec(name="presentation",
                              description="male or female",
                              choices=("male", "female")),
            ),
            examples=("use a male voice", "switch to a female voice",
                      "change your voice to male"),
        ),
        ActionSpec(
            name="set_language",
            description="Change the language I use.",
            category="Assistant",
            risk=RiskLevel.LOW,
            handler=_set_language,
            parameters=(
                ParameterSpec(name="language", description="the language to switch to",
                              max_length=40),
                ParameterSpec(
                    name="scope",
                    description="which language setting to change",
                    required=False,
                    default="all",
                    choices=("all", "interface", "input", "output"),
                ),
            ),
            examples=("change language to Hindi", "speak Kannada", "use English"),
        ),
        ActionSpec(
            name="list_languages",
            description="Show which languages I support.",
            category="Assistant",
            risk=RiskLevel.LOW,
            handler=_list_languages,
            examples=("what languages do you speak",),
        ),
        ActionSpec(
            name="set_wake_word",
            description="Change the phrase that will wake me once voice is enabled.",
            category="Assistant",
            risk=RiskLevel.MEDIUM,
            handler=_set_wake_word,
            parameters=(
                ParameterSpec(name="phrase", description="the new wake phrase", max_length=40),
            ),
            examples=("set the wake word to hey nova",),
        ),
    ]
