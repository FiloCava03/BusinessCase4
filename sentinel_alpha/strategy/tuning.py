"""In-fold tuner for the state-machine thresholds (enter/exit/dwell/tau)."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd

from sentinel_alpha.config import (
    DEFAULT_TUNING_OBJECTIVE,
    DWELL_WEEKS,
    ENTER_RISK_OFF,
    EXIT_RISK_OFF,
    TC_BPS_PER_LEG,
)
from sentinel_alpha.strategy.gate import apply_gate
from sentinel_alpha.strategy.state_machine import hysteresis
from sentinel_alpha.strategy.backtest import run_backtest

_log = logging.getLogger(__name__)


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
    tc_bps_per_leg: float = TC_BPS_PER_LEG,
    minimum_off_weeks: int = 4,
    objective: str = DEFAULT_TUNING_OBJECTIVE,
) -> dict:
    """Grid search over (enter, exit, dwell, tau) by maximizing ``objective``.

    Parameters
    ----------
    objective : str, default = ``DEFAULT_TUNING_OBJECTIVE`` (= "calmar")
        Metric to maximize. Must be a key produced by ``run_backtest`` --
        typically one of "calmar" (Risk Management default), "sharpe", "sortino".
    minimum_off_weeks : int
        A configuration is rejected if it triggers fewer than this many
        risk-off weeks (avoids the always-risk-on degenerate optimum simply
        because the bench return is positive).

    Returns
    -------
    dict
        Best configuration, with the objective value under key ``objective``.
        If the grid yields no valid configuration, returns a fallback dict and
        emits a warning -- callers should treat ``n_off_weeks == 0`` as a
        signal that no flip ever fires under the tested grid.
    """
    best: dict = {objective: -np.inf}
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
        val = float(res.metrics.get(objective, -np.inf))
        if not np.isfinite(val):
            continue
        if val > best[objective]:
            best = {
                objective: val,
                "enter": float(enter), "exit": float(exit_),
                "dwell": int(dwell), "tau": float(tau),
                "n_off_weeks": n_off,
                "ann_return": float(res.metrics.get("ann_return", 0.0)),
                "max_drawdown": float(res.metrics.get("max_drawdown", 0.0)),
                "sharpe": float(res.metrics.get("sharpe", 0.0)),
                "calmar": float(res.metrics.get("calmar", 0.0)),
            }
    if best[objective] == -np.inf:
        # No configuration on the grid produced any risk-off triggers --
        # return config defaults and warn loudly so the caller can react.
        _log.warning(
            "tune_thresholds: no configuration produced >= %d risk-off weeks "
            "(objective=%s). Falling back to config defaults "
            "(enter=%.2f, exit=%.2f, dwell=%d, tau=0.0). "
            "Inspect p_t distribution and grid coverage.",
            minimum_off_weeks, objective,
            ENTER_RISK_OFF, EXIT_RISK_OFF, DWELL_WEEKS,
        )
        return {objective: 0.0,
                "enter": ENTER_RISK_OFF, "exit": EXIT_RISK_OFF,
                "dwell": DWELL_WEEKS, "tau": 0.0,
                "n_off_weeks": 0, "ann_return": 0.0, "max_drawdown": 0.0,
                "sharpe": 0.0, "calmar": 0.0,
                "_fallback": True}
    return best
