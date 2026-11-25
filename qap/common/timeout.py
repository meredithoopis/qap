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

    def _raise(signum, frame):
        raise TrialTimeout(f"trial exceeded {seconds:.0f}s")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
