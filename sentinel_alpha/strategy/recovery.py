"""V2: leading-recovery composite + hysteresis-with-override.

Three economically-motivated leading indicators (credit-spread tightening rate,
VIX-descent velocity, equity momentum), each causally smoothed and z-scored,
averaged into a single recovery composite. Used as a conjunctive exit override
on top of the existing dwell-based exit -- the override never blocks the
existing path, only opens an additional one.

Validation: cross-regime synthetic-stress simulation in `extras.ipynb` §12.
Pre-committed decision rules pass with `override_kind="strong", thr=1.5`:

  Sharpe deltas vs V1 (sym dwell=3) across 20 bootstrap instances per scenario:
    V_shape         +0.36   (87% of the gap to bench closed)
    Slow_grind      +0.08
    Rolling_crisis  +0.04   (home turf preserved)
    Stagflation     +0.06
    Calm_bull        0.00   (override never triggers, as designed)
    Black_swan      -0.02   (within noise)
  Max-DD deltas: within +/- 0.5pp in every scenario.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def recovery_composite(
    F_panel: pd.DataFrame,
    window: int = 13,
    z_win: int = 52,
) -> pd.Series:
    """Return the weekly leading-recovery composite z-score.

    Three components, each a causal z-score of a smoothed signal:
      - credit-tightening rate   : z_{z_win}[ rolling_mean_{window} of ENG_credit_excess ]
      - VIX-descent velocity     : z_{z_win}[ rolling_mean_{window} of -ENG_vix_d ]
      - equity momentum          : z_{z_win}[ rolling_mean_{window} of ENG_eq_dm ]

    Output is the simple average of the three (no learned weights). Higher
    means a broader, stronger recovery signal -- positive values reliably
    appear in the 13-week window after every named historical crisis;
    negative values reliably appear inside the named crises.
    """
    required = ("ENG_credit_excess", "ENG_vix_d", "ENG_eq_dm")
    missing = [c for c in required if c not in F_panel.columns]
    if missing:
        raise KeyError(
            f"recovery_composite requires engineered features {required}; "
            f"missing: {missing}"
        )

    def _z(series: pd.Series) -> pd.Series:
        smooth = series.rolling(window, min_periods=4).mean()
        mu = smooth.rolling(z_win, min_periods=8).mean()
        sd = smooth.rolling(z_win, min_periods=8).std(ddof=0).replace(0.0, np.nan)
        return (smooth - mu) / sd

    cz = _z(F_panel["ENG_credit_excess"])
    vz = _z(-F_panel["ENG_vix_d"])
    ez = _z(F_panel["ENG_eq_dm"])
    return ((cz + vz + ez) / 3.0).fillna(0.0)


def hysteresis_with_override(
    signal: np.ndarray,
    p_raw: np.ndarray,
    rec: np.ndarray,
    enter: float,
    exit_: float,
    dwell: int = 3,
    override_kind: str = "strong",
    thr: float = 1.5,
    start_state: int = 0,
) -> np.ndarray:
    """Hysteresis state machine + an additional recovery-override exit path.

    Standard transitions (unchanged):
      0 -> 1  iff signal > enter for `dwell` consecutive weeks
      1 -> 0  iff signal < exit_ for `dwell` consecutive weeks

    Additional EXIT path while in state 1, parameterised by `override_kind`:
      "none"       : disabled (returns the original hysteresis)
      "strong"     : exit if rec[t] > thr             (one-week event)
      "persistent" : exit if rec[t] > thr for 2 consecutive weeks
      "joint"      : exit if rec[t] > thr AND p_raw[t] < 0.40

    The override only ADDS an exit path, never blocks the dwell one.

    Shipped default (validated in extras.ipynb §12): override_kind="strong",
    thr=1.5. The 1.3-1.7 region is a stable plateau on the synthetic grid.
    """
    s = np.asarray(signal, dtype=float).ravel()
    p = np.asarray(p_raw,  dtype=float).ravel()
    r = np.asarray(rec,    dtype=float).ravel()
    assert s.shape == p.shape == r.shape, "signal / p_raw / rec must align"

    n = s.size
    state = int(start_state)
    out = np.empty(n, dtype=np.int8)
    cu = cd = cr_persist = 0

    for t in range(n):
        # Override path: only active while in state 1.
        force_exit = False
        if state == 1:
            if override_kind == "strong":
                force_exit = r[t] > thr
            elif override_kind == "persistent":
                cr_persist = cr_persist + 1 if r[t] > thr else 0
                force_exit = cr_persist >= 2
            elif override_kind == "joint":
                force_exit = (r[t] > thr) and (p[t] < 0.40)
            elif override_kind != "none":
                raise ValueError(f"unknown override_kind {override_kind!r}")

        if state == 0:
            cu = cu + 1 if s[t] > enter else 0
            if cu >= dwell:
                state = 1
                cu = 0; cd = 0; cr_persist = 0
        else:
            cd = cd + 1 if s[t] < exit_ else 0
            if cd >= dwell or force_exit:
                state = 0
                cu = 0; cd = 0; cr_persist = 0
        out[t] = state

    return out
