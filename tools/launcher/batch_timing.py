"""Timing helpers shared by the SITL batch launchers."""

import argparse
import math
import time


def nonnegative_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("must be a finite, non-negative number of seconds")
    return seconds


def wait_between_runs(delay_s: float, stop: dict) -> bool:
    """Wait before the next attempt, returning False if cancellation is requested."""
    if delay_s <= 0:
        return not stop["flag"]
    print(f"[WAIT] Waiting {delay_s:g}s before the next SITL attempt...")
    deadline = time.monotonic() + delay_s
    while not stop["flag"]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(remaining, 0.1))
    return False
