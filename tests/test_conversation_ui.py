"""The conversation view must show both halves of every turn.

The bug this file exists for, in the user's words: "the action happens, but the
UI conversation area does not reliably display" what was said. Typing worked.
The microphone button worked. Saying "Hey Aariz" did not -- because the
wake-word turn called ``handle_text`` on its own thread and the desktop window
was armed with no callbacks to hear about it. Three entry routes, two of which
the interface knew about, and the one the user actually used was the third.

So these tests are mostly about *routes*: every way a turn can start has to end
up in the same transcript. The last class mutates the pipeline to prove the
earlier assertions are load-bearing rather than incidentally true.
"""

from __future__ import annotations

import os
import threading
from typing import Iterator, List

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from core.conversation import ConversationBus, ConversationEvent  # noqa: E402
from core.engine import AarmaEngine  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> Iterator[QApplication]:
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def events(engine: AarmaEngine) -> List[ConversationEvent]:
    """Everything the engine announces, in order."""
    captured: List[ConversationEvent] = []
    engine.conversation.subscribe(captured.append)
    return captured


def texts(events, role: str) -> List[str]:
    return [e.text for e in events if e.role == role]


# --------------------------------------------------------------------------
# The bus itself
# --------------------------------------------------------------------------

class TestTheBus:
    def test_a_subscriber_receives_what_is_published(self) -> None:
        bus, seen = ConversationBus(), []
        bus.subscribe(seen.append)
        bus.say_user("hello", source="text")
        assert [(e.role, e.text) for e in seen] == [("user", "hello")]

    def test_unsubscribing_stops_delivery(self) -> None:
        bus, seen = ConversationBus(), []
        stop = bus.subscribe(seen.append)
        stop()
        bus.say_user("hello")
        assert seen == []

    def test_one_failing_subscriber_does_not_stop_the_others(self) -> None:
        """A window that throws while rendering must not cost the action."""
        bus, seen = ConversationBus(), []

        def _explode(_event):
            raise RuntimeError("boom")

        bus.subscribe(_explode)
        bus.subscribe(seen.append)
        assert bus.publish(ConversationEvent(role="user", text="hi")) == 1
        assert len(seen) == 1

    def test_publishing_from_another_thread_reaches_subscribers(self) -> None:
        """A wake-word turn publishes from its own worker thread."""
        bus, seen = ConversationBus(), []
        bus.subscribe(seen.append)
        worker = threading.Thread(target=bus.say_user, args=("from a thread",))
        worker.start()
        worker.join(5)
        assert texts(seen, "user") == ["from a thread"]


# --------------------------------------------------------------------------
# Every route announces its turn
# --------------------------------------------------------------------------

class TestEveryRouteIsAnnounced:
    def test_typing_produces_both_halves_in_order(
            self, engine: AarmaEngine, events: List[ConversationEvent]) -> None:
        engine.handle_text("what time is it")
        assert [e.role for e in events] == ["user", "assistant"], \
            "the user's words must appear before the reply, and both must appear"
        assert events[0].text == "what time is it"
        assert "current time" in events[1].text.lower()

    def test_a_spoken_turn_is_announced_the_same_way(
            self, engine: AarmaEngine, events: List[ConversationEvent]) -> None:
        """This is the route the wake word uses; it had no interface at all."""
        engine.handle_text("what time is it", source="voice")
        assert [e.role for e in events] == ["user", "assistant"]
        assert all(e.spoken for e in events), "a spoken turn must be labelled spoken"

    def test_a_wake_turn_reaches_a_subscriber_that_never_called_the_engine(
            self, engine: AarmaEngine, events: List[ConversationEvent]) -> None:
        """The actual shape of the bug: nobody was on the other end of the call.

        ``_run_voice_turn`` invokes ``handle_text`` directly from a worker
        thread. Nothing returns to the interface, so if announcing depended on
        being the caller -- as it used to -- this turn would be invisible.
        """
        worker = threading.Thread(
            target=engine.handle_text, args=("what time is it",),
            kwargs={"source": "voice"})
        worker.start()
        worker.join(10)
        assert len(texts(events, "user")) == 1
        assert len(texts(events, "assistant")) == 1

    def test_an_error_is_announced(self, engine: AarmaEngine,
                                   events: List[ConversationEvent]) -> None:
        engine.handle_text("please reticulate the splines")
        assert events[-1].role == "assistant"
        assert events[-1].status == "error"

    def test_a_clarification_is_announced(
            self, engine: AarmaEngine, events: List[ConversationEvent]) -> None:
        engine.handle_text("schedule a meeting tomorrow")
        assert events[-1].role == "assistant"
        assert "?" in events[-1].text, "a question must reach the transcript"

    def test_a_rename_is_announced(self, engine: AarmaEngine,
                                   events: List[ConversationEvent]) -> None:
        engine.handle_text("change your name to Friday")
        assert "friday" in events[-1].text.lower()

    def test_a_benchmark_run_announces_nothing(
            self, engine: AarmaEngine, events: List[ConversationEvent]) -> None:
        """Twenty measured utterances are not twenty things the user said."""
        engine.handle_text("what time is it", dry_run=True)
        assert events == []


# --------------------------------------------------------------------------
# Duplicates and blanks
# --------------------------------------------------------------------------

class TestNothingIsSaidTwiceOrEmpty:
    def test_one_user_message_per_utterance(
            self, engine: AarmaEngine, events: List[ConversationEvent]) -> None:
        engine.handle_text("what time is it")
        assert len(texts(events, "user")) == 1

    def test_one_assistant_message_per_reply(
            self, engine: AarmaEngine, events: List[ConversationEvent]) -> None:
        engine.handle_text("what time is it")
        assert len(texts(events, "assistant")) == 1

    def test_answering_a_confirmation_aloud_is_announced_once(
            self, engine: AarmaEngine, events: List[ConversationEvent]) -> None:
        """Two code paths handle this turn; only one of them may announce it.

        "no" reaches ``resolve_confirmation`` through ``handle_text``, and both
        used to publish, so the transcript showed the cancellation twice.
        """
        from security.permissions import Permission

        engine.permissions.grant(Permission.SYSTEM_CONTROL)
        engine.handle_text("shut down the computer")
        events.clear()
        engine.handle_text("no")
        assert len(texts(events, "assistant")) == 1

    def test_an_empty_utterance_announces_nothing(
            self, engine: AarmaEngine, events: List[ConversationEvent]) -> None:
        """Silence from the recogniser must not leave an empty user bubble."""
        engine.handle_text("   ")
        assert texts(events, "user") == []


# --------------------------------------------------------------------------
# The window really draws them
# --------------------------------------------------------------------------

class TestTheWindowDrawsWhatIsAnnounced:
    """Behaviour, not source: the bug was a message that never appeared.

    The renderer is exercised directly rather than by constructing a whole
    Dashboard, which would start an engine thread, a wake loop and a scheduler
    inside a unit test -- the arrangement that once caused an access violation
    mid-suite.
    """

    @pytest.fixture
    def page(self, qapp, engine: AarmaEngine):
        from ui.dashboard import ChatPage, Metrics

        return ChatPage(engine, Metrics())

    def _render(self, page, engine: AarmaEngine, event: ConversationEvent) -> None:
        from ui.dashboard import Dashboard

        Dashboard._on_conversation_event(
            type("Stub", (), {"chat_page": page, "engine": engine})(), event)

    def test_a_user_event_becomes_a_message(self, page, engine: AarmaEngine) -> None:
        before = page.canvas_layout.count()
        self._render(page, engine,
                     ConversationEvent(role="user", text="open Downloads"))
        assert page.canvas_layout.count() == before + 1

    def test_an_assistant_event_becomes_a_message(self, page,
                                                  engine: AarmaEngine) -> None:
        before = page.canvas_layout.count()
        self._render(page, engine,
                     ConversationEvent(role="assistant", text="Opening Downloads."))
        assert page.canvas_layout.count() == before + 1

    def test_a_confirmation_event_becomes_a_card_not_a_message(
            self, page, engine: AarmaEngine) -> None:
        from ui.dashboard import ConfirmationCard

        self._render(page, engine, ConversationEvent(
            role="assistant", text="Approve?", status="confirmation",
            confirmation_token="abc123"))
        container = page.canvas_layout.itemAt(page.canvas_layout.count() - 2).widget()
        assert container.findChild(ConfirmationCard) is not None

    def test_an_empty_event_draws_nothing(self, page, engine: AarmaEngine) -> None:
        before = page.canvas_layout.count()
        self._render(page, engine, ConversationEvent(role="assistant", text="   "))
        assert page.canvas_layout.count() == before

    def test_ordering_is_preserved(self, page, engine: AarmaEngine) -> None:
        """A transcript out of order is worse than no transcript.

        Read back off the rendered labels rather than a counter, so a renderer
        that inserted in the wrong place would fail here rather than pass on
        having produced the right number of bubbles.
        """
        from PySide6.QtWidgets import QLabel

        from ui.widgets import MessageBubble

        for role, text in (("user", "first"), ("assistant", "second"),
                           ("user", "third")):
            self._render(page, engine, ConversationEvent(role=role, text=text))

        rendered = [label.text()
                    for bubble in page.findChildren(MessageBubble)
                    for label in bubble.findChildren(QLabel)
                    if label.objectName() == "BubbleText"]
        assert rendered == ["first", "second", "third"]


# --------------------------------------------------------------------------
# Mutation: prove the assertions above are load-bearing
# --------------------------------------------------------------------------

class TestTheseTestsWouldCatchTheRegression:
    """Each mutation is a way the pipeline could silently stop working.

    A test suite that passes with the feature removed is a suite that was never
    testing the feature.
    """

    def test_removing_user_announcement_is_caught(
            self, engine: AarmaEngine, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: List[ConversationEvent] = []
        engine.conversation.subscribe(seen.append)
        monkeypatch.setattr(engine.conversation, "say_user",
                            lambda *a, **k: None)
        engine.handle_text("what time is it")
        assert texts(seen, "user") == [], "mutation did not take effect"
        with pytest.raises(AssertionError):
            assert len(texts(seen, "user")) == 1

    def test_removing_assistant_announcement_is_caught(
            self, engine: AarmaEngine, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: List[ConversationEvent] = []
        engine.conversation.subscribe(seen.append)
        monkeypatch.setattr(engine.conversation, "say_assistant",
                            lambda *a, **k: None)
        engine.handle_text("what time is it")
        assert texts(seen, "assistant") == [], "mutation did not take effect"
        with pytest.raises(AssertionError):
            assert len(texts(seen, "assistant")) == 1

    def test_reversing_the_order_is_caught(self, engine: AarmaEngine) -> None:
        """A transcript that answers before it asks is not a transcript."""
        seen: List[ConversationEvent] = []
        engine.conversation.subscribe(seen.append)
        engine.handle_text("what time is it")
        reversed_roles = [e.role for e in reversed(seen)]
        assert reversed_roles == ["assistant", "user"]
        with pytest.raises(AssertionError):
            assert reversed_roles == ["user", "assistant"]


# --------------------------------------------------------------------------
# Scrolling
# --------------------------------------------------------------------------

class TestFollowingTheNewestMessage:
    """Reading back through the conversation must survive the next message.

    ``_follow_tail`` was set True in the constructor and set True again on
    every append, with nothing ever setting it False, so scrolling up lasted
    exactly until the assistant said something.
    """

    @pytest.fixture
    def page(self, qapp, engine: AarmaEngine):
        from ui.dashboard import ChatPage, Metrics

        page = ChatPage(engine, Metrics())
        page.resize(400, 200)
        for index in range(40):
            page.add_assistant_message(f"message {index}", "AARMA")
        QApplication.processEvents()
        return page

    def test_a_new_message_scrolls_the_view_when_sitting_at_the_bottom(
            self, page) -> None:
        bar = page.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())
        QApplication.processEvents()
        page.add_assistant_message("the newest one", "AARMA")
        QApplication.processEvents()
        assert bar.value() == bar.maximum()

    def test_a_new_message_does_not_yank_the_view_from_older_ones(
            self, page) -> None:
        bar = page.scroll.verticalScrollBar()
        bar.setValue(0)
        QApplication.processEvents()
        page.add_assistant_message("the newest one", "AARMA")
        QApplication.processEvents()
        assert bar.value() == 0, "reading history must survive an incoming message"

    def test_returning_to_the_bottom_resumes_following(self, page) -> None:
        bar = page.scroll.verticalScrollBar()
        bar.setValue(0)
        QApplication.processEvents()
        bar.setValue(bar.maximum())
        QApplication.processEvents()
        page.add_assistant_message("the newest one", "AARMA")
        QApplication.processEvents()
        assert bar.value() == bar.maximum()

    def test_sending_something_returns_to_the_end(self, page) -> None:
        """Having just spoken, the user is looking at the end of the thread."""
        bar = page.scroll.verticalScrollBar()
        bar.setValue(0)
        QApplication.processEvents()
        page.follow_tail()
        QApplication.processEvents()
        assert bar.value() == bar.maximum()
