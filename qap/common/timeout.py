"""Wall-clock limit for one planner trial.

An IG motion-planning solver has no time limit of its own. In some scenes it can spend a
very long time on a single plan, e.g. G4 (the L3-only object is parked out of reach at
x=2.5 m, so the L3 solver plans towards an unreachable pose). One stuck trial blocks a pool
worker forever, so every solver / planner call in qap runs inside ``time_limit``.

    with time_limit(trial_timeout()):
        left, right = solve(env, seed=seed)

A timeout raises TrialTimeout (an Exception), which callers treat as a planning failure.
Default limit: QAP_TRIAL_TIMEOUT seconds (env var), 300 s if unset; 0 disables it.
Uses SIGALRM, so it is active only on POSIX in the main thread of a process (pool workers
qualify); elsewhere it is a no-op.
"""

from __future__ import annotations

import contextlib
import os
import signal
import threading
from typing import Iterator, Optional

DEFAULT_TRIAL_TIMEOUT = 300.0


class TrialTimeout(Exception):
    """A planner trial exceeded its wall-clock limit."""


def trial_timeout(default: float = DEFAULT_TRIAL_TIMEOUT) -> float:
    try:
        return float(os.environ.get("QAP_TRIAL_TIMEOUT", default))
    except ValueError:
        return default


@contextlib.contextmanager
def time_limit(seconds: Optional[float]) -> Iterator[None]:
    usable = (seconds is not None and seconds > 0 and hasattr(signal, "SIGALRM")
              and threading.current_thread() is threading.main_thread())
    if not usable:
        yield
        return

    active = [True]

    def _raise(signum, frame):
        if active[0]:
            raise TrialTimeout(f"trial exceeded {seconds:.0f}s")

    previous = signal.signal(signal.SIGALRM, _raise)
    # After the first expiry the alarm repeats every second: code inside the trial that does
    # `except Exception: pass` (e.g. the recorder's get_state guard) can swallow one
    # TrialTimeout, and a one-shot alarm would then never fire again and the trial would hang.
    signal.setitimer(signal.ITIMER_REAL, float(seconds), 1.0)
    try:
        yield
    finally:
        active[0] = False
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
