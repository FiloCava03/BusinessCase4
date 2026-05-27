"""In-fold tuner for the state-machine thresholds (enter/exit/dwell/tau)."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd

from sentinel_alpha.strategy.gate import apply_gate
from sentinel_alpha.strategy.state_machine import hysteresis
from sentinel_alpha.strategy.backtest import run_backtest


@dataclass(frozen=True)
class ThresholdGrid:
    enters: tuple[float, ...] = (0.40, 0.50, 0.60, 0.70)
    exits:  tuple[float, ...] = (0.20, 0.30, 0.40)
    dwells: tuple[int, ...]   = (1, 2, 3)
    taus:   tuple[float, ...] = (-1.0, -0.25, 0.25, 1.0)  # 1.0 ~ effectively off


def tune_thresholds(
    p: pd.Series,
    risk_appetite: pd.Series,
    risk_on_simple: pd.Series,
    defensive_simple: pd.Series,
    grid: ThresholdGrid = ThresholdGrid(),
    tc_bps_per_leg: float = 10.0,
    minimum_off_weeks: int = 4,
) -> dict:
    """Grid search over (enter, exit, dwell, tau) by maximizing net Sharpe.

    A configuration is rejected if it triggers fewer than `minimum_off_weeks`
    risk-off weeks in total (avoids picking the always-risk-on degenerate
    optimum simply because the bench Sharpe is positive).
    """
    best: dict = {"sharpe": -np.inf}
    p = p.values; ra = risk_appetite.values
    idx = risk_on_simple.index
    for enter, exit_, dwell, tau in product(grid.enters, grid.exits, grid.dwells, grid.taus):
        if exit_ >= enter:
            continue
        signal = apply_gate(p, ra, tau=tau)
        states = hysteresis(signal, enter=enter, exit_=exit_, dwell=dwell)
        n_off = int(np.sum(states == 1))
        if n_off < minimum_off_weeks:
            continue
        states_s = pd.Series(states, index=idx)
        res = run_backtest(states_s, risk_on_simple, defensive_simple,
                           tc_bps_per_leg=tc_bps_per_leg)
        sh = float(res.metrics.get("sharpe", -np.inf))
        if sh > best["sharpe"]:
            best = {
                "sharpe": sh,
                "enter": float(enter), "exit": float(exit_),
                "dwell": int(dwell), "tau": float(tau),
                "n_off_weeks": n_off,
                "ann_return": float(res.metrics.get("ann_return", 0.0)),
                "max_drawdown": float(res.metrics.get("max_drawdown", 0.0)),
            }
    if best["sharpe"] == -np.inf:
        # fall back to defaults if no configuration triggers off
        return {"sharpe": 0.0, "enter": 0.5, "exit": 0.3, "dwell": 2, "tau": 1.0,
                "n_off_weeks": 0, "ann_return": 0.0, "max_drawdown": 0.0}
    return best
