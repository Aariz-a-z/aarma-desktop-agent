"""The AARMA engine: one command in, one response out.

Every path through this class follows the same order, and there is no way to
skip a step:

    text -> intent -> plan -> registry lookup -> parameter validation
         -> permission check -> connectivity check -> risk / confirmation
         -> passphrase gate -> execute -> audit -> response

The engine is the *only* component that calls an action handler.  The text
interface, the dashboard and the future voice loop all call
:meth:`AarmaEngine.handle_text`, which means none of them can accidentally
execute something without the checks above -- and neither can a language model
in Phase 2, because a model's proposal enters at exactly the same point as a
typed sentence.

Handlers may raise.  The engine catches everything, writes the detail to the
log and returns a plain sentence to the user (specification section 30): a
capstone demo should never show a stack trace to an evaluator.
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance for type hints only
    from speech.engine import SpeechEngine
    from speech.input_engine import ListeningEngine

from actions.registry import (
    ActionContext,
    ActionRegistry,
    ActionRequest,
    ActionResult,
    ActionSpec,
    UnknownActionError,
    ValidationFailed,
    build_default_registry,
)
from ai.intent import IntentEngine
from ai.memory import ConversationContext
from ai.planner import PlanKind, TaskPlanner
from config.settings import ConfigManager, load_config, normalise_wake_phrase
from core.connectivity import ConnectivityMonitor
from core.conversation import ConversationBus, ConversationEvent
from core.pending import (
    PendingActionStore,
    interpret_answer as interpret_slot_answer,
    is_cancellation,
)
from core.scheduler import ReminderEvent, ReminderScheduler
from core.wake_session import (BOOTSTRAP_PHRASE, SessionState,
                               WakeSession)
from database.db import Database
from language.manager import LanguageManager
from security.confirmation import ConfirmationBroker, ConfirmationOutcome, interpret_answer
from security.passphrase import PassphraseManager
from security.permissions import PermissionManager, describe as describe_permissions
from security.validator import RiskLevel, RiskPolicy, SafePathResolver
from utils.logger import get_logger

logger = get_logger("core.engine")


class BenchmarkExecutionError(RuntimeError):
    """Raised if an action handler is entered during a dry run.

    The benchmark must resolve intents without launching anything. "Return
    before calling the handler" is a rule that holds only as long as every
    future code path remembers it, so this makes the rule structural: the one
    place that invokes a handler refuses to do so when the dry-run flag is set,
    loudly.

    A silent no-op would be worse than useless here -- it would let a bug that
    routes around the resolution path go unnoticed, and the whole value of the
    benchmark rests on nothing having run.
    """


@dataclass(frozen=True)
class ActionResolution:
    """What *would* have happened, worked out without doing it.

    Every gate an ordinary command passes -- registry, platform, parameters,
    permissions, connectivity, precheck, risk -- is evaluated and recorded
    here. Only the final call is withheld.
    """

    action: Optional[str] = None
    risk_level: Optional[str] = None
    would_require_confirmation: bool = False
    permitted: bool = True
    refusal: str = ""
    #: Always False, and asserted before any export is written.
    executed: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected": self.action,
            "risk_level": self.risk_level,
            "would_require_confirmation": self.would_require_confirmation,
            "permitted": self.permitted,
            "refusal": self.refusal,
            "executed": self.executed,
        }


class ResponseKind(str, Enum):
    """What kind of thing the engine is saying back."""

    REPLY = "reply"                             # Conversational, no action ran.
    RESULT = "result"                           # An action ran and succeeded.
    ERROR = "error"                             # Something was refused or failed.
    CONFIRMATION_REQUIRED = "confirmation"      # Waiting for the user to approve.
    EXIT = "exit"                               # Session should end.


@dataclass(frozen=True)
class EngineResponse:
    """A complete answer, ready to be rendered by any interface."""

    text: str
    kind: ResponseKind = ResponseKind.REPLY
    ok: bool = True
    action: Optional[str] = None
    risk: Optional[RiskLevel] = None
    data: Dict[str, Any] = field(default_factory=dict)
    confirmation_token: Optional[str] = None

    @property
    def should_exit(self) -> bool:
        return self.kind is ResponseKind.EXIT


def _status_of(response: "EngineResponse") -> str:
    """How a display should treat this reply.

    Derived here rather than in each interface, so the console and the desktop
    window cannot end up disagreeing about what counts as an error.
    """
    if response.kind is ResponseKind.CONFIRMATION_REQUIRED:
        return "confirmation"
    return "ok" if response.ok else "error"


def _wake_registry():
    """The installed wake-word providers, imported on demand.

    A module-level import would drag the whole speech package -- and its audio
    back ends -- into a text-only session, which the lazy :attr:`speech` and
    :attr:`listening` properties exist precisely to avoid.
    """
    from speech.wakeword import REGISTRY

    return REGISTRY


class AarmaEngine:
    """Owns the pipeline and every long-lived subsystem."""

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        *,
        database: Optional[Database] = None,
        registry: Optional[ActionRegistry] = None,
        path_resolver: Optional[SafePathResolver] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.config = config or load_config()
        self.language = LanguageManager(self.config)
        self.permissions = PermissionManager(self.config)
        self.passphrase = PassphraseManager(self.config)
        self.connectivity = ConnectivityMonitor(self.config)
        self.registry = registry or build_default_registry()
        self.database = database or Database()
        self.database.initialise()
        self.context = ConversationContext()
        self.confirmations = ConfirmationBroker()
        #: One half-finished action, waiting on one answer. See
        #: :mod:`core.pending` for why it is deliberately no larger than that.
        self.pending = PendingActionStore()
        #: Announces every visible turn. Interfaces subscribe rather than
        #: relying on being the caller -- which is how a wake-word turn used to
        #: run, speak and never appear in the conversation view.
        self.conversation = ConversationBus()
        #: Rings reminders. Not started here -- an interface calls
        #: :meth:`start_reminders` once it has somewhere to show them, so a
        #: headless test never spawns a thread it did not ask for.
        self.reminders = ReminderScheduler(self.database)
        # Given the registry so the nearest-intent resolver can only ever
        # propose capabilities that exist, and a context provider so "close
        # this" can mean the tab in front rather than a guess.
        self.intent_engine = IntentEngine(
            self.config, registry=self.registry,
            context_provider=self._resolution_context)
        self.planner = TaskPlanner(self.registry, self.language)
        self.paths = path_resolver or SafePathResolver()
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._speech: Optional["SpeechEngine"] = None
        self._listening: Optional["ListeningEngine"] = None
        #: The last action that succeeded and the target it acted on, so the
        #: next sentence can say "it" or "there". See :meth:`_remember_target`.
        self._last_action: str = ""
        self._last_target: Dict[str, str] = {}
        #: What the assistant is called, and what it was called before that.
        #: The former name exists so a spoken command addressed to it can be
        #: corrected instead of obeyed. See :meth:`_addressed_to_former_name`.
        self._known_name: str = self.config.config.assistant.name
        self._former_name: str = ""

        #: Locked until "hey aariz" is heard. Constructed per engine, so a new
        #: process starts locked again -- the bootstrap is a session-level
        #: activation step, not a setting.
        self.wake_session = WakeSession()

        #: Wake-word runtime, created on first use like :attr:`speech` and
        #: :attr:`listening`, and rebuilt whenever the identity changes.
        self._wake = None
        #: The bootstrap detector -- a different object from the identity's,
        #: retired the moment the session unlocks.
        self._bootstrap_detector = None
        self._wake_phrase = ""
        self._wake_session = None
        self._wake_callbacks = (None, None)
        self._wake_running = False
        self._wake_lock = threading.RLock()
        # The assistant can be renamed from the GUI, the console or a spoken
        # command; all three land here, so the detector follows the identity
        # without any of them knowing that a detector exists.
        self.config.add_listener(self._on_identity_changed)

        logger.info("Engine initialised", extra={"context": {
            "session": self.session_id,
            "assistant": self.config.config.assistant.name,
            "language": self.language.interface_language,
            "actions": len(self.registry),
        }})

    # ------------------------------------------------------------- accessors

    @property
    def assistant_name(self) -> str:
        return self.config.config.assistant.name

    @property
    def online(self) -> bool:
        return self.connectivity.is_online()

    @property
    def is_locked(self) -> bool:
        return not self.passphrase.is_unlocked

    @property
    def speech(self) -> "SpeechEngine":
        """The voice, created on first use.

        Lazy because constructing it starts a worker thread and probes the
        machine's voices: a text-only session -- the CLI, or the test suite --
        should pay neither cost. Interfaces decide *when* to speak; the core
        pipeline never speaks on its own.
        """
        if self._speech is None:
            from speech.engine import SpeechEngine

            self._speech = SpeechEngine(
                self.config,
                connectivity=self.connectivity,
                permissions=self.permissions,
            )
        return self._speech

    @property
    def speech_started(self) -> bool:
        """Whether the voice has been constructed, without constructing it."""
        return self._speech is not None

    @property
    def listening(self) -> "ListeningEngine":
        """The microphone, created on first use.

        Lazy for the same reason as :attr:`speech`: a text-only session should
        not load a speech model. The listening engine produces *text* only --
        the interface hands that text to :meth:`handle_text`, so voice commands
        travel the same audited path as typed ones.
        """
        if self._listening is None:
            from speech.input_engine import ListeningEngine

            self._listening = ListeningEngine(
                self.config, permissions=self.permissions,
                # Given the voice so it can stop the assistant talking before
                # opening the microphone, rather than transcribing its own reply.
                speech_engine=self.speech)
            self._listening.preload()
        return self._listening

    @property
    def listening_started(self) -> bool:
        return self._listening is not None

    def status_line(self) -> str:
        """One-line summary used by the CLI banner and the dashboard header."""
        connectivity = self.language.t("status.online" if self.online else "status.offline")
        return (f"{self.assistant_name} · {self.language.info().english_name} · "
                f"{connectivity} · {len(self.registry)} actions")

    def help_text(self) -> str:
        return self.planner.help_text()

    # ------------------------------------------------------------- lifecycle

    def start(self) -> EngineResponse:
        """Greeting shown when a session opens."""
        if self.is_locked:
            return EngineResponse(self.language.t("response.security_locked"),
                                  kind=ResponseKind.ERROR, ok=False)
        text = self.language.t("response.ready", assistant=self.assistant_name)
        self.context.add_assistant(text, self.language.interface_language)
        return EngineResponse(text, kind=ResponseKind.REPLY)

    # ------------------------------------------------------------- wake word

    @property
    def wake_phrase(self) -> str:
        """This identity's own phrase, normalised.

        Not necessarily the one armed at this moment -- see :attr:`armed_phrase`
        for that. This is what the assistant answers to once the session is
        unlocked, and it is maintained by the rename system: a custom name gets
        the bare name, and Aariz keeps "hey aariz" because that is the phrase
        its shipped model was trained on.

        Read from configuration on every access rather than cached: settings,
        the setup wizard and the rename action all write it, and the wake layer
        must never drift from what they stored.
        """
        return normalise_wake_phrase(self.config.config.assistant.wake_word)

    @property
    def armed_phrase(self) -> str:
        """What the detector is listening for right now.

        The bootstrap phrase until the session is unlocked, then the identity's
        own. These are genuinely different things and collapsing them is the
        mistake the wake session exists to prevent: the shipped model hears
        "hey aariz", and a session that has not heard it yet cannot be listening
        for a name that nobody has enrolled a detector for.
        """
        if self.wake_session.bootstrap_required:
            return BOOTSTRAP_PHRASE
        return self.wake_phrase

    @property
    def wake_state(self) -> Dict[str, Any]:
        """Bootstrap state plus what it means, for the interface to show."""
        state = dict(self.wake_session.as_dict())
        identity_phrase = self.wake_phrase
        state.update({
            "name": self.config.config.assistant.name,
            "phrase": self.armed_phrase,
            "identity_phrase": identity_phrase,
            # After unlocking, whether the identity can actually be heard is a
            # question about enrolment, not about configuration.
            "identity_detectable": self._has_detector(identity_phrase),
            "waiting_for_bootstrap": self.wake_session.bootstrap_required,
        })
        return state

    @staticmethod
    def _has_detector(phrase: str) -> bool:
        """Whether a phrase really has a model, shipped or enrolled.

        Asked of the registry rather than assumed, because "the user renamed
        the assistant" and "the user enrolled a detector" are different events
        and only the second one makes a name audible.
        """
        if not phrase:
            return False
        try:
            return _wake_registry().provider_for(phrase) is not None
        except Exception:  # noqa: BLE001 - an unreadable registry hears nothing
            logger.debug("Wake registry could not be consulted")
            return False

    def _addressed_to_former_name(self, utterance: str) -> bool:
        """Does this sentence open by addressing the name we used to have?

        Only at the start, and only followed by a word boundary, so "tell me
        about Aariz" and "open the Aariz folder" are untouched -- the name is
        being *used* there, not being spoken to.
        """
        former = (self._former_name or "").strip().lower()
        if not former or former == self.assistant_name.strip().lower():
            return False
        spoken = " ".join(str(utterance or "").strip().lower().split())
        for opener in (former, f"hey {former}", f"okay {former}", f"ok {former}"):
            if spoken == opener:
                return True
            if spoken.startswith(opener) and not spoken[len(opener)].isalnum():
                return True
        return False

    def end_voice_session(self) -> bool:
        """Stop responding to ambient voice until the bootstrap is said again.

        Nothing to do with the computer sleeping -- see
        :mod:`actions.session` for why those must not share a phrase.
        """
        if self.wake_session.bootstrap_required:
            return False
        self.wake_session.reset()
        # The identity's detector is no longer the armed one, so retire it and
        # let the bootstrap detector take the microphone back.
        self._invalidate_wake_detector()
        if self._wake_running:
            on_event, on_command = self._wake_callbacks
            self.start_wake_loop(on_event=on_event, on_command=on_command)
        return True

    def complete_bootstrap(self, phrase: str = BOOTSTRAP_PHRASE) -> bool:
        """Unlock the session and re-arm for the current identity.

        Returns False when the phrase was not the bootstrap phrase, leaving the
        session locked -- "something else was heard" has to stay distinguishable
        from "the bootstrap was completed".
        """
        if not self.wake_session.complete(phrase):
            return False
        # The armed phrase has just changed, so retire the bootstrap detector
        # rather than leaving it holding the microphone.
        with self._wake_lock:
            retiring, self._bootstrap_detector = self._bootstrap_detector, None
        if retiring is not None:
            try:
                retiring.stop()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.debug("Bootstrap detector would not stop")
        return True

    @property
    def wake_detector(self):
        """The detector for the current *identity*, built on first use.

        Not necessarily the one that is running: until the session is unlocked
        that is :meth:`armed_detector`. The distinction matters because "which
        detector belongs to this assistant" is a question about the identity,
        and the settings screen, the rename flow and the diagnostics all want
        that answer whether or not the bootstrap has happened yet.

        Rebuilt whenever the configured phrase changes.  Construction is
        deferred because it reaches into the speech package, and a text-only
        session should never pay for a microphone it will not open.  The models
        themselves load later still, on the detector's own thread.
        """
        phrase = self.wake_phrase
        with self._wake_lock:
            if self._wake is not None and self._wake_phrase == phrase:
                return self._wake
            previous = self._wake
            self._wake = None
        if previous is not None:
            # Never leave a retired identity holding the microphone.
            try:
                previous.stop()
            except Exception:  # noqa: BLE001
                logger.debug("Previous wake detector would not stop")
        detector = _wake_registry().create(
            phrase,
            display_name=self.config.config.assistant.name,
            # The assistant must not hear itself.  This is the same guard the
            # listening engine already applies, reused rather than reinvented.
            # Wrapped in a lambda so that merely asking which detector is
            # active -- as the settings screen does -- does not build the
            # speech-input engine and load its models.
            speaking=lambda: self.listening._assistant_is_speaking(),
        )
        with self._wake_lock:
            self._wake, self._wake_phrase = detector, phrase
        return detector

    @property
    def armed_detector(self):
        """The detector actually listening, which is the bootstrap one at first.

        Only this one is ever started, so there is never a second reader on the
        microphone -- an arrangement Windows does not support, and which the
        voice turn already takes care to avoid.
        """
        if not self.wake_session.bootstrap_required:
            return self.wake_detector
        with self._wake_lock:
            if self._bootstrap_detector is not None:
                return self._bootstrap_detector
        detector = _wake_registry().create(
            BOOTSTRAP_PHRASE,
            display_name=self.config.config.assistant.name,
            speaking=lambda: self.listening._assistant_is_speaking(),
        )
        with self._wake_lock:
            self._bootstrap_detector = detector
        return detector

    def wake_identity(self) -> Dict[str, Any]:
        """What the wake layer will actually do, for settings and diagnostics.

        Answerable without starting anything, so the UI can show "model not
        installed" before the user discovers it by shouting at the laptop.
        """
        phrase = self.wake_phrase
        detector = _wake_registry().create(
            phrase, display_name=self.config.config.assistant.name)
        available = bool(phrase) and detector.is_available() \
            and detector.supports_phrase(phrase)
        enrolled = _wake_registry().custom_model(phrase)
        return {
            "name": self.config.config.assistant.name,
            "phrase": phrase,
            "detector": detector.name,
            "available": available,
            "reason": "" if available else detector.unavailable_reason(),
            "listening": self._wake_running,
            "installed_phrases": _wake_registry().phrases(),
            # Enrollment is a different question from availability: a phrase can
            # be unavailable because nothing is enrolled, or because something is
            # enrolled and its files have gone missing.
            "enrolled": enrolled is not None,
            "enrolled_version": enrolled.model_version if enrolled else 0,
            "enrolled_at": enrolled.created_at if enrolled else "",
            "enrollment_count": enrolled.enrollment_count if enrolled else 0,
            # Re-enrollment is allowed and replaces version N with N+1;
            # a shipped model is not something to record over.
            "can_enrol": bool(phrase) and not _wake_registry().is_builtin(phrase),
        }

    def wake_availability(self):
        """``(available, reason)`` without starting anything."""
        state = self.wake_identity()
        return state["available"], state["reason"]

    def start_wake_loop(self, *, on_event=None, on_command=None) -> bool:
        """Listen for the active wake phrase and handle complete voice turns.

        Returns ``False`` when the current identity has no usable detector -- a
        model that is not installed, a missing optional dependency.  That must
        leave the rest of the assistant working, so this reports rather than
        raises, and never substitutes a different identity's model.

        ``on_event`` fires when the phrase is heard and ``on_command`` with the
        :class:`EngineResponse` once the turn finishes; both are optional and
        exist so an interface can show progress.
        """
        phrase = self.armed_phrase
        if not phrase:
            logger.warning("Nothing to arm for the current identity")
            return False
        detector = self.armed_detector
        if not detector.is_available():
            logger.warning("Wake loop unavailable", extra={
                "context": {"phrase": phrase, "reason": detector.unavailable_reason()}})
            return False
        if not detector.supports_phrase(phrase):
            logger.warning("Wake phrase unsupported", extra={"context": {
                "phrase": phrase, "reason": detector.unavailable_reason()}})
            return False

        self._wake_callbacks = (on_event, on_command)

        def _woken(event) -> None:
            # This runs on the detector's own thread, so the turn is handed to
            # another one: stopping the detector from inside its own callback
            # would deadlock on its join.
            with self._wake_lock:
                busy = (self._wake_session is not None
                        and self._wake_session.is_alive())
            if busy:
                logger.debug("Wake ignored; a turn is already in progress")
                return
            worker = threading.Thread(
                target=self._run_voice_turn, args=(event,),
                name="wake-turn", daemon=True)
            with self._wake_lock:
                self._wake_session = worker
            worker.start()

        started = detector.start(phrase, _woken)
        self._wake_running = started
        return started

    def stop_wake_loop(self) -> None:
        """Stop listening for the wake phrase.  Safe when nothing is running.

        Both detectors are stopped. Only one of them is ever started, but which
        one depends on whether the session was unlocked, and shutdown is not
        the place to be clever about that.
        """
        self._wake_running = False
        for detector in (self._wake, self._bootstrap_detector):
            if detector is None:
                continue
            try:
                detector.stop()
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.debug("Wake detector stop failed")

    def _on_identity_changed(self, snapshot) -> None:
        """Follow a rename: retire the old detector, arm the new identity.

        Called by :class:`ConfigManager` after every settings change, from
        whichever thread made it.  A change arriving mid-turn only invalidates
        the detector: restarting here would open a second reader on a
        microphone the command capture is already using, so the turn's own
        resume step picks up the new identity instead.
        """
        phrase = normalise_wake_phrase(snapshot.assistant.wake_word)
        # Kept so a command addressed to the old name can be answered rather
        # than silently obeyed. One name deep: the name before this one is the
        # one someone might still be using out of habit.
        previous = self._known_name
        if previous and previous.lower() != snapshot.assistant.name.lower():
            self._former_name = previous
        self._known_name = snapshot.assistant.name
        with self._wake_lock:
            if phrase == self._wake_phrase:
                return
            detector, was_running = self._wake, self._wake_running
            self._wake, self._wake_phrase = None, ""
            mid_turn = (self._wake_session is not None
                        and self._wake_session.is_alive())
        if detector is not None:
            try:
                detector.stop()
            except Exception:  # noqa: BLE001
                logger.debug("Retired wake detector would not stop")
        logger.info("Wake identity changed", extra={"context": {
            "phrase": phrase, "name": snapshot.assistant.name}})
        if was_running and not mid_turn:
            on_event, on_command = self._wake_callbacks
            if not self.start_wake_loop(on_event=on_event, on_command=on_command):
                logger.warning("New identity has no usable wake detector",
                               extra={"context": {"phrase": phrase}})

    def _invalidate_wake_detector(self) -> None:
        """Force the next lookup to rebuild from configuration and the store.

        Called after a detector is enrolled or removed: the phrase has not
        changed, so the ordinary "did the phrase change?" check would keep
        handing back the old answer -- which is the one that said no model
        existed.
        """
        with self._wake_lock:
            detector, self._wake, self._wake_phrase = self._wake, None, ""
        if detector is not None:
            try:
                detector.stop()
            except Exception:  # noqa: BLE001
                logger.debug("Replaced wake detector would not stop")

    @contextmanager
    def wake_paused(self):
        """Release the microphone for something else, then restore the loop.

        Enrollment records through the same capture stack the detector uses, so
        the detector has to let go first: one owner of the microphone, always.
        The loop is restarted afterwards only if it was running, and it is
        restarted whether the caller succeeded, failed or was cancelled.
        """
        was_running = self._wake_running
        self.stop_wake_loop()
        try:
            yield
        finally:
            if was_running:
                on_event, on_command = self._wake_callbacks
                try:
                    self.start_wake_loop(on_event=on_event, on_command=on_command)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Wake listening did not resume after enrollment",
                                 extra={{"context": {{"error": str(exc)}}}})

    def enrol_wake_word(self, clips, *, display_name: str = "",
                        progress=None, install: bool = True, **options):
        """Train and validate a detector for the active identity's phrase.

        ``clips`` are :class:`~speech.wakeword_enrollment.EnrollmentClip`
        recordings; capturing them is the caller's job, because a console and a
        settings dialog prompt for them very differently.  Everything after
        that -- training, threshold calibration, the quality gate and the
        decision to install -- happens here so that all three front ends get
        the same answer.

        The wake loop is stopped for the duration and restored afterwards, and
        a detector that fails validation is not installed.
        """
        from speech.wakeword_enrollment import enrol

        phrase = self.wake_phrase
        name = display_name or self.config.config.assistant.name
        with self.wake_paused():
            result = enrol(display_name=name, clips=clips, phrase=phrase,
                           progress=progress, install=install, **options)
            if result.installed:
                # The store changed under the registry; drop the cached
                # "no model" detector so the new one is picked up.
                self._invalidate_wake_detector()
        return result

    def remove_wake_word(self, phrase: str = "") -> bool:
        """Delete an enrolled detector.  Shipped models are never touched."""
        from speech.wakeword_store import STORE

        target = normalise_wake_phrase(phrase) or self.wake_phrase
        with self.wake_paused():
            removed = STORE.remove(target)
            if removed:
                self._invalidate_wake_detector()
        return removed

    def _run_voice_turn(self, event) -> None:
        """One turn: wake -> command -> action -> spoken reply -> listen again.

        Wake detection is stopped for the duration.  Two readers on one
        microphone is not a supported arrangement on Windows, and the command
        capture needs the device the detector is holding.
        """
        on_event, on_command = self._wake_callbacks
        detector = (self._bootstrap_detector
                    if self.wake_session.bootstrap_required else self._wake)
        # Hearing the bootstrap phrase is what unlocks the session. Done before
        # anything else in the turn so that the re-arm at the end picks up the
        # identity's own phrase rather than the bootstrap one again.
        #
        # The turn then continues normally: "Hey Aariz, shut down the computer"
        # carries a command, and unlocking grants no authority over it -- the
        # permission and confirmation gates are the same ones typing goes
        # through, because the command still travels through `handle_text`.
        was_bootstrap = self.wake_session.bootstrap_required
        if was_bootstrap:
            self.complete_bootstrap()
            self.conversation.say_system(
                self.language.t("wake.unlocked", name=self.assistant_name),
                source="wake")
        try:
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:  # noqa: BLE001
                    logger.debug("Wake event callback raised")

            if detector is not None:
                detector.stop()                 # release the microphone

            try:
                result, _record = self.listening.listen_once()
            except Exception as exc:  # noqa: BLE001
                logger.error("Command capture failed",
                             extra={"context": {"error": str(exc)}})
                result = None

            text = (getattr(result, "text", "") or "").strip() if result else ""
            if not text:
                # Nothing said, or nothing recognised. Not an error: the phrase
                # may have been triggered in passing, and the bootstrap wake is
                # expected to carry no command at all. Deliberately silent in
                # the transcript -- an empty user bubble for every passing
                # trigger is noise, not a record.
                logger.info("Wake produced no command")
            else:
                response = self.handle_text(text, source="wake")
                if on_command is not None:
                    try:
                        on_command(response)
                    except Exception:  # noqa: BLE001
                        logger.debug("Command callback raised")
                if response.text:
                    try:
                        self.speech.speak(response.text)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Reply could not be spoken",
                                     extra={"context": {"error": str(exc)}})
        except Exception as exc:  # noqa: BLE001 - a bad turn must not end the loop
            logger.error("Voice turn failed", extra={"context": {"error": str(exc)}})
        finally:
            # Wait until the assistant has stopped talking before listening
            # again, or the tail of its own reply is the first thing it hears.
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                try:
                    if not self.listening._assistant_is_speaking():
                        break
                except Exception:  # noqa: BLE001
                    break
                time.sleep(0.1)
            with self._wake_lock:
                self._wake_session = None
            if not self._wake_running:
                return                          # shut down while we were busy
            try:
                # Re-read the identity: this turn may itself have been a rename.
                if not self.start_wake_loop(on_event=on_event, on_command=on_command):
                    logger.warning("Wake listening did not resume",
                                   extra={"context": {"phrase": self.wake_phrase}})
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not resume wake listening",
                             extra={"context": {"error": str(exc)}})

    # --------------------------------------------------------- reminders

    def start_reminders(self, sink=None) -> bool:
        """Begin ringing reminders, delivering them through ``sink``.

        The sink returns True when it actually reached the user.  That return
        value is not decoration: the scheduler leaves a reminder pending and
        retries when nothing accepted it, so a reminder that arrives while the
        assistant is mid-sentence is delayed rather than lost.

        Called by whichever interface is running.  The engine does not add a
        sink of its own, because "written to the log" is not "the user was
        told" and a scheduler that believed otherwise would drop reminders on a
        headless machine.
        """
        if sink is not None:
            self.reminders.notifier.add_sink(sink)
        return self.reminders.start()

    def stop_reminders(self) -> None:
        self.reminders.stop()

    def speak_reminder(self, event: "ReminderEvent") -> bool:
        """A sink that speaks a reminder, if the voice is free to speak it.

        Returns False while the assistant is already talking, which is what
        makes the scheduler hold the reminder for the next poll rather than
        cutting across a sentence.  Missed reminders are never spoken: they are
        history, and an alarm for something hours past reads as an alarm for
        something now.
        """
        if event.missed or not self.config.config.voice.speak_notifications:
            return False
        if not self.speech_started:
            # Constructing the voice here would start a worker thread inside a
            # scheduler tick, on a machine that may have no speaker permission.
            return False
        try:
            if self.speech.is_speaking:
                return False
            from speech.engine import SpeechPurpose  # noqa: PLC0415

            # ``speak`` returns False when the voice is disabled or the
            # speaker permission is off. Passing that straight through is what
            # makes an unspoken reminder count as undelivered rather than done.
            return bool(self.speech.speak(event.spoken_text(),
                                          purpose=SpeechPurpose.NOTIFICATION,
                                          interrupt=False))
        except Exception:  # noqa: BLE001 - a mute speaker is not a lost reminder
            logger.exception("Could not speak a reminder")
            return False

    def shutdown(self) -> None:
        self.confirmations.cancel_all()
        self.pending.clear("shutdown")
        self.reminders.stop()
        self.stop_wake_loop()
        self.config.remove_listener(self._on_identity_changed)
        if self._listening is not None:
            self._listening.shutdown()
            self._listening = None
        if self._speech is not None:
            self._speech.shutdown()
            self._speech = None
        self.database.close()
        logger.info("Engine shut down", extra={"context": {"session": self.session_id}})

    # ---------------------------------------------------------------- intake

    def handle_text(self, text: str, *, source: str = "text",
                    dry_run: bool = False) -> EngineResponse:
        """Process one user utterance.  Never raises.

        ``dry_run`` additionally suppresses the two side effects that are not
        actions but are still side effects: the conversation transcript and the
        audit log. Twenty benchmark utterances should not appear in the user's
        history as though they had been said to the assistant.
        """
        utterance = (text or "").strip()
        if not utterance:
            return EngineResponse(self.language.t("response.unknown_command"),
                                  kind=ResponseKind.REPLY)

        # Spoken to a name that is no longer this assistant's: answer, do not
        # act. The action rules are unanchored, so "Aariz, open Chrome" would
        # otherwise open Chrome with the address simply ignored.
        if source in ("voice", "wake") and self._addressed_to_former_name(utterance):
            reply = EngineResponse(
                self.language.t("wake.renamed_reminder",
                                former=self._former_name,
                                name=self.assistant_name),
                kind=ResponseKind.REPLY, ok=True)
            if not dry_run:
                self.conversation.say_user(utterance, source=source)
                self.conversation.say_assistant(reply.text, source=source,
                                                status="ok")
            return reply

        # Ambient voice is gated on the session; typing and the microphone
        # button are not. See `core.wake_session.GATED_SOURCES` for why.
        if not self.wake_session.allows(source):
            refusal = EngineResponse(
                self.language.t("wake.session_locked"),
                kind=ResponseKind.ERROR, ok=False)
            if not dry_run:
                self.conversation.say_assistant(refusal.text, source=source,
                                                status="error")
            logger.info("Refused a voice command before activation",
                        extra={"context": {"source": source}})
            return refusal

        language_code = self.language.interface_language
        if not dry_run:
            self.context.add_user(utterance, language_code)
            self._persist("user", utterance)
            # Published before the action runs, so the transcript shows what
            # was asked while it is being worked on rather than afterwards.
            self.conversation.say_user(utterance, source=source)

        try:
            response = self._route(utterance, source=source, dry_run=dry_run)
        except BenchmarkExecutionError:
            raise            # A benchmark that executed something must not be swallowed.
        except Exception:  # noqa: BLE001 - the engine is the last line of defence
            logger.exception("Unhandled error while processing input")
            response = EngineResponse(self.language.t("error.internal"),
                                      kind=ResponseKind.ERROR, ok=False)

        if not dry_run:
            self.context.add_assistant(response.text, language_code)
            self._persist("assistant", response.text)
            # Published as soon as the reply exists. Speaking it is a separate
            # concern and happens afterwards: text that waits for synthesis is
            # text the user watches a silent window for.
            self.conversation.say_assistant(
                response.text, source=source, action=response.action or "",
                status=_status_of(response),
                confirmation_token=response.confirmation_token or "",
                data=dict(response.data))
        return response

    def _route(self, utterance: str, *, source: str,
               dry_run: bool = False) -> EngineResponse:
        # A locked assistant answers nothing except an unlock attempt.
        if self.is_locked:
            return self._handle_locked(utterance)

        # A bare "yes"/"no" belongs to the outstanding confirmation, if any.
        # Skipped in a dry run: resolving a real confirmation would execute the
        # parked action, which is precisely what must not happen.
        pending = self.confirmations.latest()
        if pending is not None and not dry_run:
            answer = interpret_answer(utterance)
            if answer is not None:
                # handle_text announces this turn on the way out.
                return self.resolve_confirmation(pending.token, answer,
                                                 announce=False)

        intent = self.intent_engine.parse(utterance)

        # A half-finished action gets first refusal on this utterance -- but
        # only if the utterance is not itself a command. See
        # :meth:`_continue_pending` for the ordering and why it matters.
        if not dry_run and self.pending.is_open:
            continued = self._continue_pending(utterance, intent, source=source)
            if continued is not None:
                return continued

        # One sentence can carry more than one instruction. Tried before the
        # single-intent planner because a rule that matched only part of the
        # sentence would otherwise run that part and silently discard the rest.
        multi = self.intent_engine.parse_plan(utterance,
                                              self._resolution_context())
        if multi is not None and (multi.is_multi or multi.unmet):
            return self._run_plan(multi, source=source, dry_run=dry_run)

        plan = self.planner.plan(intent, source=source)

        if plan.kind is PlanKind.EXIT:
            return EngineResponse(plan.reply or "Goodbye.", kind=ResponseKind.EXIT)
        if plan.kind in (PlanKind.REPLY, PlanKind.UNKNOWN):
            return EngineResponse(plan.reply or "",
                                  kind=ResponseKind.REPLY,
                                  ok=plan.kind is PlanKind.REPLY)

        # Phase 1 plans hold a single step; the loop is what Phase 2 grows into.
        response: Optional[EngineResponse] = None
        for request in plan.steps:
            response = self.execute(request, dry_run=dry_run)
            if not response.ok or response.kind is ResponseKind.CONFIRMATION_REQUIRED:
                break
        return response or EngineResponse(self.language.t("response.unknown_command"),
                                          kind=ResponseKind.REPLY, ok=False)

    # ------------------------------------------------------ multi-step actions

    def _run_plan(self, plan, *, source: str, dry_run: bool) -> EngineResponse:
        """Run an instruction that asked for more than one thing.

        Three rules, all of them about not overstating what happened:

        A later step never runs on a failed earlier one. "Open my repo in VS
        Code and open a terminal there" has no *there* once the repository was
        not found, and running the second step against whatever the previous
        context held would be worse than stopping.

        Partial success is reported as partial, naming what did not run. "Done"
        after one step of two is the summary that leaves someone believing
        their terminal is open when it is not.

        A step that needs confirmation ends the plan. The broker parks one
        action, not a queue, so the steps after it are abandoned -- and the
        reply says which, rather than letting the user assume that answering
        "yes" will still run them.
        """
        done: List[str] = []
        for index, step in enumerate(plan.steps):
            remaining = [s.text for s in plan.steps[index + 1:]]
            request = ActionRequest(action=step.action,
                                    parameters=dict(step.parameters),
                                    source=source, raw_utterance=step.text)
            response = self.execute(request, dry_run=dry_run)

            if response.kind is ResponseKind.CONFIRMATION_REQUIRED:
                return self._partial(done, response, remaining, plan.unmet,
                                     kind=ResponseKind.CONFIRMATION_REQUIRED,
                                     ok=response.ok,
                                     token=response.confirmation_token,
                                     action=response.action, risk=response.risk)
            if not response.ok:
                return self._partial(done, response, remaining, plan.unmet,
                                     kind=response.kind, ok=False,
                                     action=response.action)
            done.append(response.text.strip())

        parts = [t for t in done if t]
        if plan.unmet:
            parts.append(self.language.t("response.plan_unmet",
                                         steps="; ".join(plan.unmet)))
        return EngineResponse(" ".join(parts), kind=ResponseKind.RESULT,
                              ok=not plan.unmet,
                              data={"steps": [s.as_dict() for s in plan.steps],
                                    "unmet": list(plan.unmet)})

    def _partial(self, done: "List[str]", response: EngineResponse,
                 remaining: "List[str]", unmet: "List[str]", *,
                 kind: ResponseKind, ok: bool,
                 token: Optional[str] = None, action: Optional[str] = None,
                 risk: Optional[RiskLevel] = None) -> EngineResponse:
        """Say what ran, what is happening now, and what was dropped."""
        parts = [t for t in done if t]
        parts.append(response.text.strip())
        if remaining:
            parts.append(self.language.t("response.plan_stopped",
                                         steps="; ".join(remaining)))
        if unmet:
            parts.append(self.language.t("response.plan_unmet",
                                         steps="; ".join(unmet)))
        return EngineResponse(" ".join(p for p in parts if p), kind=kind, ok=ok,
                              action=action, risk=risk,
                              confirmation_token=token,
                              data=dict(response.data))

    # ----------------------------------------------------- multi-turn actions

    def _continue_pending(self, utterance: str, intent, *, source: str
                          ) -> Optional[EngineResponse]:
        """Try to read ``utterance`` as the answer to the open question.

        Returns ``None`` to mean "this was not an answer, carry on with normal
        routing".  The order of the checks below is the whole safety property:

        1. **Cancellation wins.**  "Never mind" clears the pending action and
           nothing else happens.
        2. **A recognised command wins next.**  If the intent engine matched a
           real action, the user has moved on; the pending action is abandoned
           rather than fed a sentence that was meant as an instruction.
        3. **Only then is it treated as an answer**, and only if it fits the
           slot -- a time slot takes something parseable as a time or nothing
           at all.

        Without rule 2, "schedule a meeting tomorrow" / "what time?" / "check
        my email" would book a meeting called "check my email".
        """
        pending = self.pending.current()
        if pending is None:
            return None

        if is_cancellation(utterance):
            self.pending.clear("user cancelled")
            return EngineResponse("Cancelled.", kind=ResponseKind.REPLY, ok=True)

        if intent is not None and intent.is_action:
            self.pending.clear("superseded by a new command")
            return None

        value = interpret_slot_answer(pending, utterance)
        if value is None:
            # Not an answer and not a command. Drop the question rather than
            # asking it again -- repeating it is how a dialogue gets stuck.
            dropped = self.pending.clear("answer did not fit the slot")
            logger.info("Pending answer rejected", extra={"context": {
                "action": dropped.action if dropped else "", "slot": pending.needs}})
            return None

        filled = self.pending.fill(value)
        if filled is None:  # pragma: no cover - current() already proved it open
            return None
        logger.info("Pending slot filled", extra={"context": {
            "action": filled.action, "slot": filled.needs}})
        return self.execute(ActionRequest(action=filled.action,
                                          parameters=dict(filled.parameters),
                                          source=source,
                                          raw_utterance=utterance))

    def _open_pending(self, spec: ActionSpec, parameters: Dict[str, Any],
                      response: EngineResponse, source: str) -> EngineResponse:
        """Turn a handler's "I still need X" into a question worth answering.

        Handlers already report the missing field -- ``needs="title"`` and the
        like -- so the engine does not have to guess which of several optional
        parameters was the problem.  When the action is not one that may be
        completed conversationally, this changes nothing and the handler's own
        message is returned as before.
        """
        slot = str(response.data.get("needs") or "")
        if not slot:
            return response
        # Carry what the handler received, minus the blanks.
        #
        # ``parameters`` here is post-validation, so every optional field the
        # spec declares is present -- filled in with its default, usually "".
        # Handing those back to ``validate`` on the retry fails, because an
        # empty string is not a valid value for a parameter that was *supplied*.
        # Dropping them restores the distinction between "not given" and
        # "given as nothing", which is exactly what the next turn depends on.
        carried = {key: value for key, value in parameters.items()
                   if value not in ("", None)}
        opened = self.pending.open(spec.name, carried, needs=slot,
                                   question=response.text, source=source)
        if opened is None:
            return response
        return EngineResponse(response.text, kind=ResponseKind.REPLY, ok=True,
                              action=spec.name, risk=spec.risk,
                              data={"pending": opened.as_dict()})

    def _handle_locked(self, utterance: str) -> EngineResponse:
        """While locked, the only thing that works is the passphrase."""
        if self.passphrase.unlock(utterance):
            return EngineResponse(f"Unlocked. Hello again — I'm {self.assistant_name}.",
                                  kind=ResponseKind.REPLY)
        return EngineResponse(self.language.t("response.security_locked"),
                              kind=ResponseKind.ERROR, ok=False)

    # ------------------------------------------------------------- execution

    def execute(self, request: ActionRequest, *, pre_approved: bool = False,
                dry_run: bool = False) -> EngineResponse:
        """Run one action request through every gate, in order.

        ``dry_run`` is the benchmark flag. It changes nothing about how the
        gates are evaluated -- the action is still looked up, its parameters
        still validated, its permissions still checked, its risk still
        graded -- and then the method returns the resolution instead of
        invoking the handler. It is threaded in from the caller rather than
        held as engine state, so two callers cannot disagree about whether a
        run is live.
        """
        # 0. A confirmation parked for something else is stale the moment a
        #    different action runs. Leaving it standing means a later bare
        #    "yes" -- about anything at all -- approves an action the user
        #    stopped talking about several turns ago.
        dropped = "" if pre_approved else self._drop_stale_confirmation(request.action)

        # 1. The action must exist in the registry.
        try:
            spec = self.registry.require(request.action)
        except UnknownActionError:
            self._audit(request.action, {}, "unknown", request.source,
                        allowed=False, succeeded=False, message="unregistered action")
            return EngineResponse(self.language.t("response.unknown_command"),
                                  kind=ResponseKind.ERROR, ok=False, action=request.action)

        # 2. The action must be supported on this platform.
        if spec.platforms and sys.platform not in spec.platforms:
            message = f"'{spec.name}' isn't supported on this operating system yet."
            self._audit(spec.name, request.parameters, spec.risk_label, request.source,
                        allowed=False, succeeded=False, message=message)
            return EngineResponse(message, kind=ResponseKind.ERROR, ok=False,
                                  action=spec.name, risk=spec.risk)

        # 3. Parameters must satisfy the declared schema.
        try:
            parameters = spec.validate(request.parameters)
        except ValidationFailed as exc:
            return EngineResponse(
                self.language.t("response.invalid_parameters", reason=str(exc)),
                kind=ResponseKind.ERROR, ok=False, action=spec.name, risk=spec.risk,
            )

        # 4. The user must have granted every permission the action declares.
        decision = self.permissions.check(spec.permissions)
        if not decision.allowed:
            for permission in decision.missing:
                self.database.record_permission_decision(
                    permission=permission.value, action=spec.name,
                    decision="denied", reason="not granted by user",
                )
            self._audit(spec.name, parameters, spec.risk_label, request.source,
                        allowed=False, succeeded=False,
                        message=f"missing permissions: {decision.describe()}")
            return EngineResponse(
                self.language.t("response.permission_denied",
                                permission=describe_permissions(decision.missing)),
                kind=ResponseKind.ERROR, ok=False, action=spec.name, risk=spec.risk,
                data={"missing_permissions": [p.value for p in decision.missing]},
            )

        # 5. Online-only actions must not pretend to work offline.
        if spec.requires_internet and not self.online:
            self._audit(spec.name, parameters, spec.risk_label, request.source,
                        allowed=False, succeeded=False, message="offline")
            return EngineResponse(self.language.t("response.needs_internet"),
                                  kind=ResponseKind.ERROR, ok=False,
                                  action=spec.name, risk=spec.risk)

        # 6. Cheap pre-check, so we never ask someone to approve an action that
        #    was always going to be refused (a path outside the sandbox, say).
        if spec.precheck is not None:
            refusal = spec.precheck(self._build_context(), parameters)
            if refusal:
                self._audit(spec.name, parameters, spec.risk_label, request.source,
                            allowed=False, succeeded=False, message=refusal)
                return EngineResponse(refusal, kind=ResponseKind.ERROR, ok=False,
                                      action=spec.name, risk=spec.risk)

        policy = RiskPolicy.from_settings(self.config.config.security)
        would_confirm = (not pre_approved
                         and policy.requires_confirmation(spec.risk))

        # 6b. BENCHMARK STOP. Everything above has been evaluated; nothing
        #     below is reached. No confirmation token is created (it would sit
        #     in the broker and swallow the next yes/no), no audit row is
        #     written, and above all no handler is called.
        if dry_run:
            resolution = ActionResolution(
                action=spec.name, risk_level=spec.risk_label,
                would_require_confirmation=would_confirm,
                permitted=True, executed=False)
            logger.info("Resolved action without executing", extra={"context": {
                "action": spec.name, "risk": spec.risk_label,
                "would_confirm": would_confirm}})
            return EngineResponse(
                f"[dry run] {spec.name}", kind=ResponseKind.RESULT, ok=True,
                action=spec.name, risk=spec.risk,
                data={"resolution": resolution.as_dict(), "dry_run": True})

        # 7. High-risk actions need an explicit yes from the human.
        if would_confirm:
            # Blanks are dropped for the same reason as in `_open_pending`:
            # `parameters` is post-validation, so every optional field the spec
            # declares is present as "". Re-validating those on approval fails,
            # because an empty string is not a valid value for a parameter that
            # was *supplied*. Without this, approving "close all applications"
            # died on "'except' can't be empty".
            supplied = {key: value for key, value in parameters.items()
                        if value not in ("", None)}
            confirmation = self.confirmations.create(
                spec.name, self._describe_action(spec, parameters), supplied
            )
            return EngineResponse(
                self.language.t("response.confirm_required", action=spec.name)
                + f"\n  {confirmation.summary}",
                kind=ResponseKind.CONFIRMATION_REQUIRED, ok=True,
                action=spec.name, risk=spec.risk,
                confirmation_token=confirmation.token,
            )

        # 8. If the user tightened security, high-risk actions also need the passphrase.
        if policy.requires_passphrase(spec.risk) and self.is_locked:
            return EngineResponse(self.language.t("response.security_locked"),
                                  kind=ResponseKind.ERROR, ok=False, action=spec.name)

        response = self._invoke(spec, parameters, request.source, dry_run=dry_run)
        if dropped:
            # Said out loud rather than logged. The user has no other way to
            # learn that "yes" will no longer do what they last asked about.
            response = replace(response, text=f"{response.text} {dropped}".strip())
        return response

    def _drop_stale_confirmation(self, action: str) -> str:
        """Cancel a park for a different action, and say what was cancelled."""
        parked = self.confirmations.latest()
        if parked is None or parked.action_name == action:
            return ""
        self.confirmations.cancel_all()
        logger.info("Dropped a stale confirmation", extra={"context": {
            "parked": parked.action_name, "superseded_by": action}})
        return self.language.t("response.confirm_superseded",
                               action=parked.action_name)

    def _build_context(self) -> ActionContext:
        """Fresh context per call, so handlers always see current settings."""
        return ActionContext(
            config=self.config,
            language=self.language,
            database=self.database,
            permissions=self.permissions,
            paths=self.paths,
            registry=self.registry,
            session_id=self.session_id,
            online=self.online,
            session_control=self.end_voice_session,
        )

    def _invoke(self, spec: ActionSpec, parameters: Dict[str, Any],
                source: str, *, dry_run: bool = False) -> EngineResponse:
        """Call the handler, time it, audit it and translate the outcome.

        Line 387 of this file is the only place in AARMA that calls
        ``spec.handler``. The guard below is therefore the last gate before any
        side effect in the entire application, which is exactly where a
        benchmark's "nothing runs" promise has to be enforced.
        """
        if dry_run:
            # Unreachable while `execute` returns at step 6b. It exists so that
            # a future path which forgets to return trips an error instead of
            # quietly launching Chrome during a measurement run.
            raise BenchmarkExecutionError(
                f"Dry run reached the executor for {spec.name!r}. No action "
                "may run during a benchmark; this is a bug in the dispatch path."
            )

        context = self._build_context()

        started = time.perf_counter()
        try:
            result = spec.handler(context, parameters)
        except Exception as exc:  # noqa: BLE001 - handler faults must not escape
            logger.exception("Action handler raised",
                             extra={"context": {"action": spec.name}})
            result = ActionResult.failure(str(exc), error_code="handler_exception")
        duration_ms = int((time.perf_counter() - started) * 1000)

        self._audit(spec.name, parameters, spec.risk_label, source,
                    allowed=True, succeeded=result.ok, message=result.message,
                    duration_ms=duration_ms)

        if result.ok:
            # Done: whatever question was open is answered or irrelevant.
            self.pending.clear("action completed")
            self._remember_target(spec.name, parameters)
            return EngineResponse(result.message, kind=ResponseKind.RESULT, ok=True,
                                  action=spec.name, risk=spec.risk, data=result.data)

        # Handler messages are already user-facing; only wrap the generic ones.
        text = result.message or self.language.t("response.action_failed", reason="")
        failure = EngineResponse(text.strip(), kind=ResponseKind.ERROR, ok=False,
                                 action=spec.name, risk=spec.risk, data=result.data)
        if result.data.get("needs"):
            return self._open_pending(spec, parameters, failure, source)
        self.pending.clear("action failed for another reason")
        return failure

    # ------------------------------------------------------- nlu context

    #: Parameters worth remembering across a turn. A target is a thing the
    #: next sentence might point at; a flag or a body of text is not.
    _TARGET_KEYS = ("name", "site", "location", "editor", "url")

    def _remember_target(self, action: str, parameters: Dict[str, Any]) -> None:
        """Keep the target of the last successful action, for "it" and "there".

        One turn deep and targets only. The temptation is to remember more, but
        a longer memory is what lets "open it in VS Code" act on something the
        user stopped thinking about several sentences ago -- and an action on a
        misremembered target is worse than a question.
        """
        target = {key: str(value) for key, value in parameters.items()
                  if key in self._TARGET_KEYS and value not in ("", None)}
        self._last_action = action
        self._last_target = target

    def _resolution_context(self):
        """What is in front, and what the last turn was about.

        Read lazily and defensively: a text-only session has no window
        manager, and failing to read one must cost a less specific
        interpretation rather than an error.
        """
        from ai.nlu.context import from_window_provider

        context = from_window_provider()
        context.last_action = self._last_action
        context.last_entities = dict(self._last_target)
        return context

    # ---------------------------------------------------------- confirmations

    def resolve_confirmation(self, token: str, approved: bool, *,
                             announce: bool = True) -> EngineResponse:
        """Approve or decline a parked high-risk action.

        ``announce`` is False when the answer arrived as an utterance:
        ``handle_text`` publishes that turn itself, and announcing here as well
        put the same assistant message in the transcript twice. An interface
        resolving from a button leaves it True, because nothing else will.
        """
        outcome, request = self.confirmations.resolve(token, approved)

        if outcome is ConfirmationOutcome.UNKNOWN:
            return EngineResponse(self.language.t("response.confirm_expired"),
                                  kind=ResponseKind.ERROR, ok=False)
        if outcome is ConfirmationOutcome.EXPIRED:
            return EngineResponse(self.language.t("response.confirm_expired"),
                                  kind=ResponseKind.ERROR, ok=False)
        if outcome is ConfirmationOutcome.DECLINED:
            assert request is not None
            self._audit(request.action_name, request.parameters, "High", "confirmation",
                        allowed=False, succeeded=False, message="declined by user")
            declined = EngineResponse(self.language.t("response.confirm_cancelled"),
                                      kind=ResponseKind.REPLY, ok=True,
                                      action=request.action_name)
            if announce:
                self.conversation.say_assistant(
                    declined.text, source="confirmation",
                    action=request.action_name, status="ok")
            return declined

        assert request is not None
        approved_request = ActionRequest(
            action=request.action_name,
            parameters=dict(request.parameters),
            source="confirmation",
        )
        response = self.execute(approved_request, pre_approved=True)
        if announce:
            self.conversation.say_assistant(
                response.text, source="confirmation",
                action=response.action or "", status=_status_of(response),
                data=dict(response.data))
        return response

    @property
    def pending_confirmation_token(self) -> Optional[str]:
        pending = self.confirmations.latest()
        return pending.token if pending else None

    # ------------------------------------------------------------- internals

    def _describe_action(self, spec: ActionSpec, parameters: Dict[str, Any]) -> str:
        """Plain-language summary shown in the confirmation prompt.

        A spec may supply its own, which is how an action whose scope is the
        decision gets to state that scope. Wrapped broadly: a summary that
        throws must cost the user detail, never the confirmation itself, since
        failing closed here would block the action rather than protect it.
        """
        if spec.summarise is not None:
            try:
                summary = spec.summarise(self._build_context(), parameters)
                if summary:
                    return summary
            except Exception:  # noqa: BLE001 - detail is not worth a failure
                logger.exception("Action summary failed",
                                 extra={"context": {"action": spec.name}})

        # Blanks are the schema's defaults, not anything the user said, and
        # rendering them produced prompts like "(except: )".
        stated = {key: value for key, value in parameters.items()
                  if value not in ("", None)}
        if not stated:
            return spec.description
        rendered = ", ".join(f"{key}: {value}" for key, value in sorted(stated.items()))
        return f"{spec.description} ({rendered})"

    def _persist(self, role: str, text: str) -> None:
        try:
            self.database.add_message(self.session_id, role, text,
                                      self.language.interface_language)
        except Exception:  # noqa: BLE001 - a transcript failure must not break the reply
            logger.exception("Failed to persist conversation message")

    def _audit(self, action: str, parameters: Dict[str, Any], risk: str, source: str,
               *, allowed: bool, succeeded: bool, message: str = "",
               duration_ms: int = 0) -> None:
        try:
            self.database.record_action(
                session_id=self.session_id, action=action, parameters=parameters,
                risk=risk, source=source, allowed=allowed, succeeded=succeeded,
                message=message, duration_ms=duration_ms,
            )
        except Exception:  # noqa: BLE001 - auditing must not break execution
            logger.exception("Failed to write action audit record")
