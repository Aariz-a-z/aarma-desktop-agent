"""The neural voice must not leave a live socket pointing at a closed loop.

After a run of the wake-word corpus tool, Python printed::

    Exception ignored in: <function _ProactorBasePipeTransport.__del__ ...>
    RuntimeError: Event loop is closed

The traceback itself is cosmetic -- it is raised inside ``__del__``, so the
interpreter swallows it and the process still exits 0. What is *not* cosmetic is
what it reveals: a network transport outliving the loop that owns it, on a code
path that also writes audio files. A teardown that is wrong in that order is a
teardown that could truncate a write.

**These tests run the interpreter as a subprocess on purpose.** The fault only
appears when the garbage collector reaches the transport, which for a short
script is at interpreter shutdown -- after pytest would have finished asserting.
In-process testing cannot observe it.

They need a network connection (edge-tts is an online service) and are marked
``network`` so the default suite stays offline and fast::

    python -m pytest tests/test_speech_async_teardown.py -m network
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Five syntheses per process. One is not enough to expose the fault reliably
#: -- measured at 1 in 5 runs -- while five reproduced it in 5 of 5 before the
#: fix. The corpus tool performs fifteen, which is why the user saw it.
SYNTHESES = 5

_DRIVER = textwrap.dedent(
    """
    import sys, tempfile
    from pathlib import Path
    sys.path.insert(0, {root!r})

    from speech.providers.edge import EdgeNeuralSpeaker

    speaker = EdgeNeuralSpeaker()
    if not speaker.is_installed():
        print("SKIP: edge-tts not installed")
        raise SystemExit(0)

    out = Path(tempfile.mkdtemp(prefix="aarma-teardown-"))
    phrases = ["Hey Asma is ready.", "I am Asma.",
               "Hey Aariz, that is your name.",
               "Nova, I've opened Chrome for you.",
               "I couldn't find Zara in your contacts."]
    for index, phrase in enumerate(phrases[:{count}]):
        target = out / f"clip-{{index}}.mp3"
        speaker._synthesise(text=phrase, voice="en-US-AriaNeural",
                            rate="+0%", volume="+0%", pitch="+0Hz",
                            destination=target)
        assert target.stat().st_size > 0, "synthesis produced an empty file"
        print(f"OK {{target.stat().st_size}}")
    """
)


def _run_driver(count: int = SYNTHESES) -> subprocess.CompletedProcess:
    script = _DRIVER.format(root=str(REPO_ROOT), count=count)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300,
    )


def _assert_clean(result: subprocess.CompletedProcess, label: str) -> None:
    if "SKIP:" in result.stdout:
        pytest.skip("edge-tts is not installed")
    assert result.returncode == 0, (
        f"{label}: exit {result.returncode}\nstderr:\n{result.stderr}")
    # Exit code alone is not the criterion: an exception inside __del__ never
    # sets one. stderr has to be clean too.
    assert "Exception ignored" not in result.stderr, (
        f"{label}: transport outlived the loop\n{result.stderr}")
    assert "Traceback" not in result.stderr, (
        f"{label}: unexpected traceback\n{result.stderr}")
    assert "Event loop is closed" not in result.stderr, (
        f"{label}: closed-loop access\n{result.stderr}")


@pytest.mark.network
class TestNeuralVoiceTeardown:
    def test_a_single_run_leaves_no_traceback(self) -> None:
        _assert_clean(_run_driver(), "single run")

    @pytest.mark.parametrize("attempt", range(1, 6))
    def test_five_consecutive_runs_are_clean(self, attempt: int) -> None:
        """Consecutively, because this was a race before the fix.

        One clean run proved nothing: the unfixed code passed 4 of 5 times with
        a single synthesis. Five consecutive runs of five syntheses is the bar.
        """
        _assert_clean(_run_driver(), f"run {attempt}")

    def test_synthesis_still_produces_audio(self, tmp_path) -> None:
        """The teardown fix must not cost us the thing being torn down."""
        from speech.providers.edge import EdgeNeuralSpeaker

        speaker = EdgeNeuralSpeaker()
        if not speaker.is_installed():
            pytest.skip("edge-tts is not installed")
        target = tmp_path / "clip.mp3"
        speaker._synthesise(text="Hey Asma is ready.", voice="en-US-AriaNeural",
                            rate="+0%", volume="+0%", pitch="+0Hz",
                            destination=target)
        assert target.is_file()
        assert target.stat().st_size > 1000, "audio looks truncated"


class TestTeardownContract:
    """Offline checks on the shape of the fix. No network required."""

    def test_the_loop_is_closed_exactly_once_and_last(self) -> None:
        """Order is the whole fix: drain, collect, then close."""
        import asyncio

        from speech.providers.edge import _shutdown_loop

        loop = asyncio.new_event_loop()
        _shutdown_loop(loop)
        assert loop.is_closed()

    def test_it_closes_the_loop_even_if_draining_fails(self) -> None:
        """A device left holding an open loop is worse than a noisy log."""
        import asyncio

        from speech.providers.edge import _shutdown_loop

        loop = asyncio.new_event_loop()

        def explode() -> None:
            raise RuntimeError("drain failed")

        loop.shutdown_asyncgens = explode  # type: ignore[assignment]
        _shutdown_loop(loop)
        assert loop.is_closed()

    def test_no_suppression_technique_is_used(self) -> None:
        """The forbidden list, asserted rather than promised.

        Every one of these would silence the symptom while leaving a socket
        pointing at a dead loop, which is the actual defect.
        """
        source = (REPO_ROOT / "speech" / "providers" / "edge.py").read_text(
            encoding="utf-8")
        for forbidden in ("filterwarnings", "unraisablehook",
                          "_ProactorBasePipeTransport", "simplefilter",
                          "WindowsSelectorEventLoopPolicy"):
            assert forbidden not in source, f"suppression found: {forbidden}"

    def test_the_drain_does_not_rely_on_a_settle_sleep(self) -> None:
        """``asyncio.sleep`` here would be waiting and hoping, not closing."""
        import inspect

        from speech.providers.edge import _shutdown_loop

        source = inspect.getsource(_shutdown_loop)
        assert "sleep" not in source
        assert "gc.collect()" in source
        assert "shutdown_asyncgens" in source
        assert "shutdown_default_executor" in source
