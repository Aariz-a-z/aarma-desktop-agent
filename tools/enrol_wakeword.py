"""Record a custom wake word and, if it validates, install it.

    python main.py --enrol-wake-word
    python main.py --enrol-wake-word --name Friday --count 18
    python tools/enrol_wakeword.py --dry-run

The phrase comes from the assistant's current name unless ``--name`` overrides
it, so the ordinary route is: rename the assistant, then enrol.  Nothing here
decides whether the detector is good enough -- that is
:func:`speech.wakeword_enrollment.enrol`, and this module only prompts, plays
back what was measured, and refuses to pretend a failure was a success.

The microphone is opened once, for the whole session, through the same
:class:`~speech.capture.MicrophoneStream` the detector uses.  If the assistant's
wake loop is running it is stopped first and restored afterwards, including on
Ctrl-C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BAR = "-" * 68


def _ask(prompt: str, options: str) -> str:
    try:
        return input(f"{prompt}\n   {options} > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def _report(result) -> None:
    """Print the gate honestly: every check, with its denominator."""
    print()
    print(BAR)
    verdict = "PASSED" if result.passed else "FAILED"
    print(f"  Quality gate: {verdict}")
    print(BAR)
    for check in result.checks:
        mark = "ok  " if check.passed else "FAIL"
        count = f"  [{check.numerator}/{check.denominator}]" if check.denominator else ""
        print(f"  {mark}  {check.name:<18}{check.detail}{count}")
    summary = result.summary or {}
    if summary:
        print()
        print(f"  recordings          {summary.get('recordings_used', 0)} used, "
              f"{summary.get('recordings_held_back', 0)} held back")
        print(f"  distances recorded  "
              f"{', '.join(summary.get('distances_recorded') or ['unknown'])}")
        print(f"  training examples   {summary.get('training_examples', 0):,} "
              f"({summary.get('training_positives', 0):,} positive, "
              f"{summary.get('training_negatives', 0):,} negative)")
        print(f"  model               {summary.get('parameters', 0):,} parameters, "
              f"{summary.get('checkpoint_bytes', 0) / 1024:.0f} KB")
        print(f"  trained in          {summary.get('train_seconds', 0):.0f} s")
    print(f"  stage-2 threshold   {result.stage2_threshold:.2f} "
          "(calibrated for this model, not inherited)")
    print()
    print(f"  {result.message}")
    if not result.passed:
        print()
        print("  Nothing was installed. What usually helps:")
        print("    * record more examples, and vary them more")
        print("    * say the phrase the same way you actually would")
        print("    * a quieter room, or a microphone closer than 1 m")
    print(BAR)


def run(config, *, name: str = "", count: int = 0, dry_run: bool = False,
        seed: Optional[int] = None, diagnose: bool = False,
        replay: str = "") -> int:
    """Prompt, record, train, validate.  Returns a process exit code."""
    from config.settings import wake_phrase_for_name
    from core.engine import AarmaEngine
    from speech.wakeword_enrollment import (
        MIN_RECORDINGS,
        TARGET_RECORDINGS,
        EnrollmentRecorder,
        EnrollmentSession,
        enrollment_plan,
        load_session,
    )

    engine = AarmaEngine(config)
    try:
        display_name = (name or engine.config.config.assistant.name).strip()
        phrase = wake_phrase_for_name(display_name) if name else engine.wake_phrase
        if not phrase:
            print("  There is no wake phrase to enrol. Set a name first.")
            return 2

        from speech.wakeword import REGISTRY

        if REGISTRY.is_builtin(phrase):
            print(f'  "{phrase}" already has a validated model that ships with '
                  "AARMA.\n  Re-recording over it would replace a measured "
                  "detector with an unmeasured one, so this refuses.")
            return 2

        prompts = enrollment_plan(phrase, count or TARGET_RECORDINGS)
        existing = REGISTRY.custom_model(phrase)

        print()
        print(BAR)
        print(f'  Enrolling the wake word "{phrase}" for {display_name}')
        print(BAR)
        if existing is not None:
            print(f"  A version {existing.model_version} detector already exists. "
                  "It stays\n  active unless the new one passes validation.")
        print(f"  I will ask for {len(prompts)} recordings. Vary them on purpose:")
        print("  the same phrase said fifteen identical ways teaches it one way.")
        print("  Each prompt records for about 3 seconds after you press Enter.")
        print()

        if replay:
            # Re-run training on a saved session, with no microphone at all.
            # A failed enrollment is worth understanding, and understanding it
            # should not cost the user fifteen more recordings.
            saved_phrase, clips = load_session(Path(replay))
            print(f"  Replaying {len(clips)} saved recordings of "
                  f"\"{saved_phrase}\" from {replay}")
            usable = [c for c in clips if c.usable]
            for clip in clips:
                mark = "ok " if clip.usable else "x  "
                print(f"   {mark} {clip.quality.diagnostics()}")
            if len(usable) < MIN_RECORDINGS:
                print(f"\nOnly {len(usable)} usable; at least {MIN_RECORDINGS} "
                      "are needed. Nothing was changed.")
                return 1
            print()

            def replay_progress(stage: str, message: str) -> None:
                print(f"   [{stage}] {message}", flush=True)

            options = {} if seed is None else {"seed": seed}
            result = engine.enrol_wake_word(
                usable, display_name=display_name, progress=replay_progress,
                install=not dry_run, **options)
            _report(result)
            return 0 if result.passed else 1

        clips: List = []
        session = EnrollmentSession(phrase)
        with engine.wake_paused():
            with EnrollmentRecorder() as recorder:
                print(f"  Microphone: {recorder.device_name} "
                      f"@ {recorder.sample_rate} Hz")
                print("  Measuring the room; stay quiet for a moment...")
                floor = recorder.measure_room()
                ready, why = recorder.readiness()
                print(f"  Room floor RMS {floor:.5f} -- {why}")
                if not ready:
                    print()
                    print("  Enrollment needs a working microphone. "
                          "Nothing was changed.")
                    return 1
                print()

                index = 0
                while index < len(prompts):
                    prompt = prompts[index]
                    answer = _ask(
                        f'  [{index + 1}/{len(prompts)}] Say "{phrase}" -- {prompt.hint}',
                        "[Enter]=record  s=skip  q=quit")
                    if answer == "q":
                        print("\n  Cancelled. Nothing was changed.")
                        return 1
                    if answer == "s":
                        index += 1
                        continue

                    print("   listening...", flush=True)
                    clip = recorder.capture(prompt)
                    quality = clip.quality
                    if diagnose:
                        print(f"   {quality.diagnostics()}")
                    if not quality.accepted:
                        print(f"   x  {quality.reason}.")
                        print(f"      {quality.advice}")
                        continue
                    print(f"   ok  {quality.speech_seconds:.2f} s of speech, "
                          f"peak {quality.peak:.3f}"
                          + (f", SNR {quality.snr_db:.0f} dB" if quality.snr_db
                             is not None else ""))
                    clips.append(clip)
                    session.save_clip(len(clips), clip)
                    index += 1

            if len(clips) < MIN_RECORDINGS:
                print(f"\n  Only {len(clips)} usable recordings; at least "
                      f"{MIN_RECORDINGS} are needed to train and still have "
                      "something left to test with.\n  Nothing was changed.")
                return 1

            print()
            print(f"  Training on {len(clips)} recordings. This takes a few minutes"
                  " and uses no network.")
            print()

            def progress(stage: str, message: str) -> None:
                print(f"   [{stage}] {message}", flush=True)

            options = {} if seed is None else {"seed": seed}
            result = engine.enrol_wake_word(
                clips, display_name=display_name, progress=progress,
                install=not dry_run, **options)

        session.save_result(result)
        _report(result)
        print(f"  Recordings kept in: {session.directory}")
        print("  Re-run training on them without recording again:")
        print(f"    python main.py --enrol-wake-word --enrol-replay "
              f"\"{session.directory}\"")
        if result.installed:
            print()
            print(f'  Say "{phrase}" to wake me. If the assistant is running, '
                  "restart it\n  or reopen Settings so the new detector is armed.")
        return 0 if result.passed else 1
    except KeyboardInterrupt:
        print("\n  Cancelled. Nothing was changed.")
        return 1
    finally:
        engine.shutdown()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="",
                        help="enrol for this name instead of the current one")
    parser.add_argument("--count", type=int, default=0,
                        help="how many recordings to ask for")
    parser.add_argument("--dry-run", action="store_true",
                        help="train and validate, but install nothing")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--diagnose", action="store_true",
                        help="print the measurements behind every accept or reject")
    parser.add_argument("--replay", default="",
                        help="re-run training on a saved enrollment session "
                             "instead of recording")
    args = parser.parse_args(argv)

    from config.settings import load_config

    return run(load_config(), name=args.name, count=args.count,
               dry_run=args.dry_run, seed=args.seed,
               diagnose=args.diagnose, replay=args.replay)


if __name__ == "__main__":
    raise SystemExit(main())
