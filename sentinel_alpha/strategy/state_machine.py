"""Two-state hysteresis machine: risk_on <-> risk_off with dwell threshold."""
from __future__ import annotations
import numpy as np


def hysteresis(signal: np.ndarray, enter: float, exit_: float, dwell: int,
               start_state: int = 0) -> np.ndarray:
    """Convert a signal series to a 0/1 state series.

    State 0 = risk_on, state 1 = risk_off.
    Transition 0 -> 1 if `signal > enter` for `dwell` consecutive ticks.
    Transition 1 -> 0 if `signal < exit_` for `dwell` consecutive ticks.

    Returns the state in effect AT each tick (same length as `signal`).
    """
    s = np.asarray(signal, dtype=float).ravel()
    n = s.size
    state = int(start_state)
    out = np.empty(n, dtype=np.int8)
    consec_up = 0
    consec_down = 0
    for t in range(n):
        if state == 0:
            if s[t] > enter:
                consec_up += 1; consec_down = 0
            else:
                consec_up = 0
            if consec_up >= dwell:
                state = 1; consec_up = 0
        else:
            if s[t] < exit_:
                consec_down += 1; consec_up = 0
            else:
                consec_down = 0
            if consec_down >= dwell:
                state = 0; consec_down = 0
        out[t] = state
    return out
