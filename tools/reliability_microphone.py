"""Repeated-use reliability check for the real microphone.

The original speech input worked once and then stopped. This exercises the
lifecycle the way a person does -- press, speak, stop, press again -- twenty
times over, plus the awkward orderings (cancel mid-capture, double start,
stop when idle), and reports whether anything degrades.

    python tools/reliability_microphone.py            # 20 cycles
    python tools/reliability_microphone.py --cycles 40

Uses the real device. Speak or stay silent; either is fine, because what is
being measured is the *lifecycle*, not the transcription: no stuck device, no
growing latency, no second-attempt failure, no leaked threads or streams.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def process_rss_mb() -> float:
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize / 1e6
    except Exception:  # noqa: BLE001
        return 0.0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Microphone reliability check.")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=1.5,
                        help="how long to hold each capture open")
    args = parser.parse_args(argv)

    from speech.capture import MicrophoneCapture, has_microphone, list_devices

    if not has_microphone():
        print("No microphone detected; nothing to test.")
        return 1

    device = next((d for d in list_devices() if d.is_default), list_devices()[0])
    print(f"device      : {device.display_name}")
    print(f"cycles      : {args.cycles}")

    capture = MicrophoneCapture()
    baseline_threads = threading.active_count()
    baseline_rss = process_rss_mb()

    durations: List[float] = []
    outcomes: List[str] = []
    failures = 0

    print(f"\n{'#':>3} {'wall':>7} {'raw':>7} {'speech':>7}  outcome")
    for cycle in range(1, args.cycles + 1):
        started = time.monotonic()

        # Hold the capture open briefly, then stop it, like a user would.
        stopper = threading.Timer(args.seconds, capture.cancel)
        stopper.daemon = True
        stopper.start()
        try:
            result = capture.record(max_seconds=8.0, silence_timeout=0.8,
                                    no_speech_timeout=4.0)
        except Exception as exc:  # noqa: BLE001 - a raise is itself a failure
            print(f"{cycle:>3}  EXCEPTION {type(exc).__name__}: {exc}")
            failures += 1
            continue
        finally:
            stopper.cancel()

        elapsed = time.monotonic() - started
        durations.append(elapsed)
        outcome = ("cancelled" if result.cancelled
                   else "speech" if result.ok else (result.error_code or "error"))
        outcomes.append(outcome)
        if outcome not in ("cancelled", "speech", "no_speech"):
            failures += 1
        print(f"{cycle:>3} {elapsed:>6.2f}s {result.raw_duration:>6.2f}s "
              f"{result.speech_seconds:>6.2f}s  {outcome}")

    print("\n--- awkward orderings ---")
    checks = []

    capture.cancel()                       # Cancel while idle.
    checks.append(("cancel while idle", True))

    t = threading.Timer(0.4, capture.cancel)
    t.daemon = True
    t.start()
    r = capture.record(max_seconds=5.0, no_speech_timeout=3.0)
    checks.append(("cancel mid-capture reports cancelled", r.cancelled))
    t.cancel()

    r = capture.record(max_seconds=2.0, no_speech_timeout=1.0)
    checks.append(("capture works right after a cancel", not r.cancelled))

    capture.close()
    checks.append(("close while idle", not capture.is_recording))
    r = capture.record(max_seconds=2.0, no_speech_timeout=1.0)
    checks.append(("capture works after close (reopens)", r.error_code != "capture_failed"))

    for label, ok in checks:
        print(f"   {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            failures += 1

    capture.close()
    time.sleep(0.4)

    leaked_threads = threading.active_count() - baseline_threads
    grew_mb = process_rss_mb() - baseline_rss

    print("\n--- summary ---")
    if durations:
        first_half = durations[:max(1, len(durations) // 2)]
        second_half = durations[max(1, len(durations) // 2):]
        print(f"   median cycle   : {statistics.median(durations):.2f}s")
        print(f"   first half avg : {statistics.mean(first_half):.2f}s")
        print(f"   second half avg: {statistics.mean(second_half):.2f}s")
        drift = statistics.mean(second_half) - statistics.mean(first_half)
        print(f"   latency drift  : {drift:+.2f}s "
              f"({'OK' if abs(drift) < 0.75 else 'DEGRADING'})")
    tally = ", ".join(f"{o}={outcomes.count(o)}" for o in sorted(set(outcomes)))
    print(f"   outcomes       : {tally}")
    print(f"   leaked threads : {leaked_threads}")
    print(f"   memory growth  : {grew_mb:+.1f} MB")
    print(f"   hard failures  : {failures}")
    verdict = "PASS" if failures == 0 and leaked_threads <= 0 else "CHECK"
    print(f"\n   RESULT: {verdict}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
