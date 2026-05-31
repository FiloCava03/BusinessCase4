"""Backtest KPI invariants: crisis-table schema, dd_reduction sign convention."""
from __future__ import annotations
import numpy as np
import pandas as pd

from sentinel_alpha.strategy.backtest import run_backtest


def _toy_inputs(n: int = 60):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-05", periods=n, freq="W")
    risk_on = pd.Series(rng.normal(0.001, 0.02, size=n), index=idx)
    defensive = pd.Series(rng.normal(0.0005, 0.004, size=n), index=idx)
    return idx, risk_on, defensive


def test_crisis_metrics_schema_includes_new_kpis():
    idx, risk_on, defensive = _toy_inputs(80)
    states = pd.Series(np.zeros(len(idx)), index=idx)
    crises = {"toy_crisis": (str(idx[10].date()), str(idx[40].date()))}
    bt = run_backtest(states, risk_on, defensive, crises=crises)
    assert not bt.crisis_metrics.empty
    expected_cols = {
        "crisis", "weeks", "off_weeks", "off_rate",
        "strategy_ret", "bench_ret", "excess_ret",
        "strategy_max_dd", "bench_max_dd", "dd_reduction",
    }
    missing = expected_cols - set(bt.crisis_metrics.columns)
    assert not missing, f"crisis_metrics missing columns: {missing}"


def test_dd_reduction_sign_when_strategy_defends():
    """If the strategy stays fully defensive while bench drops, dd_reduction
    must be positive (strategy max-DD > bench max-DD, since both are negative)."""
    idx = pd.date_range("2008-01-04", periods=30, freq="W")
    # bench drops 30% then recovers; defensive stays flat.
    bench = pd.Series(
        [-0.05] * 10 + [0.0] * 10 + [0.01] * 10, index=idx
    )
    defensive = pd.Series([0.0] * 30, index=idx)
    states = pd.Series(np.ones(30), index=idx)  # always defensive
    crises = {"toy": (str(idx[0].date()), str(idx[-1].date()))}
    bt = run_backtest(states, bench, defensive, crises=crises)
    row = bt.crisis_metrics.iloc[0]
    assert row["strategy_max_dd"] >= row["bench_max_dd"], \
        "defensive strategy should have a shallower (less negative) max-DD"
    assert row["dd_reduction"] >= 0.0, \
        f"dd_reduction should be >= 0 when strategy defends; got {row['dd_reduction']}"


def test_excess_return_positive_when_strategy_avoids_bench_loss():
    idx = pd.date_range("2008-01-04", periods=30, freq="W")
    bench = pd.Series([-0.02] * 30, index=idx)  # bench steady loss
    defensive = pd.Series([0.0] * 30, index=idx)
    states = pd.Series(np.ones(30), index=idx)
    crises = {"toy": (str(idx[0].date()), str(idx[-1].date()))}
    bt = run_backtest(states, bench, defensive, crises=crises)
    row = bt.crisis_metrics.iloc[0]
    assert row["excess_ret"] > 0.0


def test_run_backtest_headline_metrics_present():
    idx, risk_on, defensive = _toy_inputs(60)
    states = pd.Series(np.zeros(len(idx)), index=idx)
    bt = run_backtest(states, risk_on, defensive)
    for k in ("sharpe", "calmar", "max_drawdown", "ann_return", "ann_vol", "sortino"):
        assert k in bt.metrics, f"missing headline metric {k!r}"
