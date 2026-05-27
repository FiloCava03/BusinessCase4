"""Continuous-allocation strategy.

Instead of a binary risk-on / risk-off state, the defensive weight is a
*continuous* function of the calibrated probability `p` and the risk-appetite
composite `RA`. This lets the strategy participate in V-shaped recoveries
linearly as `p` drops, rather than waiting for two consecutive sub-threshold
weeks like the hysteresis state machine.

Mathematically:

    w_off[t] = clip(g * (p_smooth[t] - center), 0, 1)   if RA[t] < tau
             = 0                                        otherwise

where `p_smooth` is a causal exponential moving average of `p` (alpha controls
the half-life). With `g = 1`, `center = 0`, `alpha = 1.0`, this collapses to
`w_off = p` -- the simplest sensible linear policy. With `alpha < 1` the
weight is smoothed; with `g > 1` the policy is more aggressive around the
decision boundary; with `center > 0` we require a stronger signal before
allocating any defensive weight.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd

from sentinel_alpha.strategy.backtest import run_backtest


def continuous_weights(
    p: np.ndarray | pd.Series,
    risk_appetite: np.ndarray | pd.Series,
    tau: float,
    gain: float = 1.0,
    center: float = 0.0,
    alpha: float = 1.0,
    lo: float = 0.0,
    hi: float = 1.0,
) -> np.ndarray:
    """Compute the defensive weight w_off[t] in [0, 1].

    Parameters
    ----------
    p : array-like (T,)
        Calibrated risk-off probability per week.
    risk_appetite : array-like (T,)
        Risk-appetite composite per week. Same length as p.
    tau : float
        Gate threshold. When RA[t] >= tau, force w_off[t] = 0.
        Use a very large value (e.g. 1e9) to disable the gate.
    gain : float >= 0
        Multiplier applied to `(p_smooth - center)`. Higher = steeper allocation
        as p crosses center. With gain=1 and center=0 the policy is `w = p`.
    center : float in [0, 1]
        Probability below which the strategy stays fully risk-on.
    alpha : float in (0, 1]
        EWM smoothing factor on p. alpha=1.0 disables smoothing.
    lo, hi : float in [0, 1]
        Clip limits on the resulting weight.
    """
    p_arr = np.asarray(p, dtype=float).ravel()
    ra_arr = np.asarray(risk_appetite, dtype=float).ravel()
    assert p_arr.shape == ra_arr.shape

    if 0.0 < alpha < 1.0:
        # Causal EMA: p_smooth[t] = alpha * p[t] + (1 - alpha) * p_smooth[t-1]
        p_smooth = pd.Series(p_arr).ewm(alpha=alpha, adjust=False).mean().values
    else:
        p_smooth = p_arr

    raw = gain * (p_smooth - center)
    w = np.clip(raw, lo, hi)
    w = np.where(ra_arr < tau, w, 0.0)
    return w


@dataclass(frozen=True)
class ContinuousGrid:
    """Search grid for continuous-allocation hyperparameters."""
    gains:   tuple[float, ...] = (0.8, 1.0, 1.25, 1.5, 2.0)
    centers: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
    alphas:  tuple[float, ...] = (0.4, 0.6, 0.8, 1.0)
    taus:    tuple[float, ...] = (-1.0, -0.25, 0.25, 1.0)


def tune_continuous(
    p: pd.Series,
    risk_appetite: pd.Series,
    risk_on_simple: pd.Series,
    defensive_simple: pd.Series,
    grid: ContinuousGrid = ContinuousGrid(),
    tc_bps_per_leg: float = 10.0,
    objective: str = "sharpe",
    min_avg_w_off: float = 0.05,
) -> dict:
    """Grid search over (gain, center, alpha, tau) by maximizing `objective`.

    `objective` may be "sharpe", "calmar", or "sortino". A configuration is
    rejected if it averages less than `min_avg_w_off` defensive weight over
    the sample (degenerate "never go defensive" optimum).
    """
    assert objective in {"sharpe", "calmar", "sortino"}, objective
    best: dict = {objective: -np.inf}
    p_v = p.values; ra_v = risk_appetite.values
    idx = risk_on_simple.index

    for gain, center, alpha, tau in product(grid.gains, grid.centers, grid.alphas, grid.taus):
        w = continuous_weights(p_v, ra_v, tau=tau, gain=gain, center=center, alpha=alpha)
        if w.mean() < min_avg_w_off:
            continue
        ws = pd.Series(w, index=idx)
        res = run_backtest(ws, risk_on_simple, defensive_simple, tc_bps_per_leg=tc_bps_per_leg)
        val = float(res.metrics.get(objective, -np.inf))
        if not np.isfinite(val):
            continue
        if val > best[objective]:
            best = {
                objective: val,
                "gain": float(gain), "center": float(center),
                "alpha": float(alpha), "tau": float(tau),
                "mean_w_off": float(w.mean()),
                "ann_return": float(res.metrics.get("ann_return", 0.0)),
                "max_drawdown": float(res.metrics.get("max_drawdown", 0.0)),
                "calmar": float(res.metrics.get("calmar", 0.0)),
            }
    if best[objective] == -np.inf:
        # Fallback: simple linear policy with no gate.
        return {objective: 0.0, "gain": 1.0, "center": 0.0, "alpha": 1.0,
                "tau": 1e9, "mean_w_off": 0.0, "ann_return": 0.0,
                "max_drawdown": 0.0, "calmar": 0.0}
    return best
