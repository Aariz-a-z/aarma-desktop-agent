"""The voice turn: wake, capture, act, speak, listen again.

The real loop needs a microphone, a recogniser and a voice, so the pieces
around it are replaced with stubs and what remains under test is the engine's
own sequencing -- which is where the mistakes live.  Four things are pinned
down here, each of which was a real bug or a real risk during integration:

* the whole command survives, so "Hey Aariz, open Chrome" does not arrive as
  "Chrome";
* the assistant does not hear itself, and does not resume while it is talking;
* one utterance produces one turn;
* a failure anywhere in a turn does not end the loop or strand the microphone.

Nothing here opens an audio device.
"""

from __future__ import annotations

import threading
from typing import Iterator, List, Optional, Tuple

import pytest

from core.engine import AarmaEngine, EngineResponse, ResponseKind
from speech.wakeword import WakeWordDetector, WakeWordProvider, WakeWordRegistry


class _Recognised:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeListening:
    """Stands in for :class:`ListeningEngine`, without a microphone."""

    def __init__(self, heard: Optional[List[str]] = None) -> None:
        self.heard = list(heard or [])
        self.calls = 0
        self.speaking = False
        self.raise_on_listen = False

    def _assistant_is_speaking(self) -> bool:
        return self.speaking

    def listen_once(self, *, audio_path=None) -> Tuple[_Recognised, object]:
        self.calls += 1
        if self.raise_on_listen:
            raise RuntimeError("microphone went away")
        text = self.heard.pop(0) if self.heard else ""
        return _Recognised(text), object()

    def preload(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class _FakeSpeech:
    def __init__(self) -> None:
        self.said: List[str] = []

    def speak(self, text: str, **kwargs) -> None:
        self.said.append(text)

    def shutdown(self) -> None:
        return None


class _ScriptedDetector(WakeWordDetector):
    """A detector whose "microphone" is a method call."""

    name = "scripted"

    def __init__(self, phrase: str = "hey aariz", **kwargs) -> None:
        self.phrase = phrase
        self.speaking = kwargs.get("speaking")
        self.callback = None
        self.starts = 0
        self.stops = 0
        self.running = False

    def is_available(self) -> bool:
        return True

    def supports_phrase(self, phrase: str) -> bool:
        return phrase.strip().lower() == self.phrase

    def start(self, phrase: str, callback) -> bool:
        self.callback = callback
        self.starts += 1
        self.running = True
        return True

    def stop(self) -> None:
        self.stops += 1
        self.running = False

    def fire(self) -> None:
        assert self.callback is not None, "detector was never started"
        self.callback("wake-event")


@pytest.fixture
def voice(engine: AarmaEngine, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict]:
    """An engine wired to fake audio, with one scripted detector installed."""
    listening, speech = _FakeListening(), _FakeSpeech()
    detectors: List[_ScriptedDetector] = []

    registry = WakeWordRegistry()

    def factory(**kwargs):
        detectors.append(_ScriptedDetector(**kwargs))
        return detectors[-1]

    registry.register(WakeWordProvider(phrase="hey aariz", factory=factory))
    monkeypatch.setattr("core.engine._wake_registry", lambda: registry)

    engine._listening = listening
    engine._speech = speech
    engine.config.update_assistant(name="Aariz", wake_word="hey aariz")

    yield {"engine": engine, "listening": listening, "speech": speech,
           "detectors": detectors}


def _settle(engine: AarmaEngine, timeout: float = 5.0) -> None:
    """Wait for the turn thread to finish, so assertions are not racing it."""
    deadline = threading.Event()
    for _ in range(int(timeout / 0.02)):
        session = engine._wake_session
        if session is None or not session.is_alive():
            return
        deadline.wait(0.02)
    raise AssertionError("the voice turn did not finish")


class TestOneTurn:
    def test_the_whole_command_reaches_the_pipeline(self, voice: dict) -> None:
        engine, listening = voice["engine"], voice["listening"]
        listening.heard = ["open chrome"]
        seen: List[str] = []

        def fake_handle(text, *, source="text", dry_run=False):
            seen.append(text)
            return EngineResponse("Opening Chrome.", kind=ResponseKind.REPLY)

        engine.handle_text = fake_handle          # type: ignore[assignment]
        assert engine.start_wake_loop() is True
        voice["detectors"][0].fire()
        _settle(engine)

        # First-word protection: not "chrome", not "" -- the command arrives whole.
        assert seen == ["open chrome"]

    def test_the_command_is_marked_as_ambient_voice(self, voice: dict) -> None:
        """"wake", not "voice": the room triggered this, nobody pressed anything.

        The session gate depends on telling those apart -- ambient voice waits
        for the bootstrap phrase, a deliberate microphone press does not -- and
        so does the audit trail.
        """
        engine, listening = voice["engine"], voice["listening"]
        listening.heard = ["what time is it"]
        sources: List[str] = []

        def fake_handle(text, *, source="text", dry_run=False):
            sources.append(source)
            return EngineResponse("It is late.", kind=ResponseKind.REPLY)

        engine.handle_text = fake_handle          # type: ignore[assignment]
        engine.start_wake_loop()
        voice["detectors"][0].fire()
        _settle(engine)
        assert sources == ["wake"]

    def test_the_reply_is_spoken(self, voice: dict) -> None:
        engine, listening, speech = voice["engine"], voice["listening"], voice["speech"]
        listening.heard = ["say hello"]
        engine.handle_text = lambda text, *, source="text", dry_run=False: EngineResponse(
            "Hello.", kind=ResponseKind.REPLY)                    # type: ignore[assignment]
        engine.start_wake_loop()
        voice["detectors"][0].fire()
        _settle(engine)
        assert speech.said == ["Hello."]

    def test_the_microphone_is_released_before_the_command_is_captured(
            self, voice: dict) -> None:
        engine, listening = voice["engine"], voice["listening"]
        order: List[str] = []
        detector = None

        def watching_listen(*, audio_path=None):
            order.append("listen")
            assert detector is not None and detector.running is False
            return _Recognised(""), object()

        listening.listen_once = watching_listen   # type: ignore[assignment]
        engine.start_wake_loop()
        detector = voice["detectors"][0]
        detector.fire()
        _settle(engine)
        assert order == ["listen"]

    def test_wake_listening_resumes_afterwards(self, voice: dict) -> None:
        engine = voice["engine"]
        voice["listening"].heard = [""]
        engine.start_wake_loop()
        voice["detectors"][0].fire()
        _settle(engine)
        assert engine._wake_running is True
        assert voice["detectors"][-1].running is True

    def test_silence_after_the_wake_word_is_not_an_error(self, voice: dict) -> None:
        engine = voice["engine"]
        voice["listening"].heard = [""]
        called: List[str] = []
        engine.handle_text = lambda *a, **k: called.append("x")  # type: ignore[assignment]
        engine.start_wake_loop()
        voice["detectors"][0].fire()
        _settle(engine)
        assert called == []
        assert engine._wake_running is True       # still listening


class TestSelfTriggerGuard:
    def test_the_detector_is_given_the_existing_speaking_guard(self, voice: dict) -> None:
        engine, listening = voice["engine"], voice["listening"]
        engine.start_wake_loop()
        guard = voice["detectors"][0].speaking
        assert callable(guard)

        listening.speaking = True
        assert guard() is True                    # delegates, not a second guess
        listening.speaking = False
        assert guard() is False

    def test_the_loop_does_not_resume_while_the_assistant_is_talking(
            self, voice: dict) -> None:
        engine, listening = voice["engine"], voice["listening"]
        listening.heard = ["say hello"]
        engine.handle_text = lambda text, *, source="text", dry_run=False: EngineResponse(
            "Hello.", kind=ResponseKind.REPLY)                    # type: ignore[assignment]

        # Speech is still playing when the reply returns.
        def speak_and_hold(text, **kwargs):
            listening.speaking = True

        voice["speech"].speak = speak_and_hold    # type: ignore[assignment]
        engine.start_wake_loop()
        first = voice["detectors"][0]
        first.fire()

        # The turn must sit in its wait loop rather than re-arm the detector.
        threading.Event().wait(0.4)
        assert len(voice["detectors"]) == 1
        assert first.running is False

        listening.speaking = False
        _settle(engine)
        assert voice["detectors"][-1].running is True


class TestRobustness:
    def test_one_wake_produces_one_turn(self, voice: dict) -> None:
        engine = voice["engine"]
        voice["listening"].heard = ["", "", ""]
        engine.start_wake_loop()
        detector = voice["detectors"][0]
        detector.fire()
        _settle(engine)
        # Three overlapping windows above threshold, one after the other.
        assert voice["listening"].calls == 1

    def test_a_capture_failure_does_not_end_the_loop(self, voice: dict) -> None:
        engine, listening = voice["engine"], voice["listening"]
        listening.raise_on_listen = True
        engine.start_wake_loop()
        voice["detectors"][0].fire()
        _settle(engine)
        assert engine._wake_running is True
        assert voice["detectors"][-1].running is True

    def test_a_handler_failure_does_not_end_the_loop(self, voice: dict) -> None:
        engine = voice["engine"]
        voice["listening"].heard = ["explode"]

        def boom(text, *, source="text", dry_run=False):
            raise RuntimeError("intent engine fell over")

        engine.handle_text = boom                 # type: ignore[assignment]
        engine.start_wake_loop()
        voice["detectors"][0].fire()
        _settle(engine)
        assert engine._wake_running is True

    def test_shutdown_during_a_turn_does_not_re_arm(self, voice: dict) -> None:
        engine, listening = voice["engine"], voice["listening"]
        started = threading.Event()
        release = threading.Event()

        def slow_listen(*, audio_path=None):
            started.set()
            release.wait(timeout=5.0)
            return _Recognised(""), object()

        listening.listen_once = slow_listen       # type: ignore[assignment]
        engine.start_wake_loop()
        voice["detectors"][0].fire()
        assert started.wait(timeout=5.0)

        engine.stop_wake_loop()
        release.set()
        _settle(engine)
        assert engine._wake_running is False
        assert all(not d.running for d in voice["detectors"])

    def test_a_rename_mid_turn_is_picked_up_on_resume(self, voice: dict) -> None:
        engine, listening = voice["engine"], voice["listening"]
        started, release = threading.Event(), threading.Event()

        def slow_listen(*, audio_path=None):
            started.set()
            release.wait(timeout=5.0)
            return _Recognised(""), object()

        listening.listen_once = slow_listen       # type: ignore[assignment]
        engine.start_wake_loop()
        voice["detectors"][0].fire()
        assert started.wait(timeout=5.0)

        # The rename must not open a second reader while capture is in flight.
        engine.config.update_assistant(name="Nova", wake_word="hey nova")
        assert len(voice["detectors"]) == 1

        release.set()
        _settle(engine)
        # No Nova model is registered, so the loop honestly stops rather than
        # re-arming the Aariz detector under a new name.
        assert engine.wake_identity()["available"] is False
        assert all(not d.running for d in voice["detectors"])
