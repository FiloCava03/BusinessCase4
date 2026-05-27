"""Strategy invariants: hysteresis dwell, gate, backtest identity case."""
from __future__ import annotations
import numpy as np
import pandas as pd

from sentinel_alpha.strategy import apply_gate, hysteresis, run_backtest


def test_gate_blocks_when_ra_above_tau():
    p = np.array([0.9, 0.9, 0.9])
    ra = np.array([1.0, -1.0, 0.5])
    out = apply_gate(p, ra, tau=0.0)
    # ra < 0 only at index 1 -> only that one passes
    np.testing.assert_allclose(out, np.array([0.0, 0.9, 0.0]))


def test_hysteresis_dwell_prevents_whipsaw():
    # Signal spikes above enter for 1 week, then drops back -> should NOT transition
    sig = np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.0])
    states = hysteresis(sig, enter=0.55, exit_=0.35, dwell=2)
    assert (states == 0).all(), f"unexpected transition: {states}"


def test_hysteresis_enters_after_two_above():
    sig = np.array([0.0, 0.7, 0.7, 0.7, 0.0, 0.0, 0.0])
    states = hysteresis(sig, enter=0.55, exit_=0.35, dwell=2)
    # transitions on the 2nd consecutive sample > enter -> index 2
    assert states[0] == 0 and states[1] == 0 and states[2] == 1 and states[3] == 1
    # exit needs 2 consecutive below -> not yet at index 4, but yes at 5
    assert states[4] == 1 and states[5] == 0


def test_backtest_identity_when_always_risk_on():
    idx = pd.date_range("2020-01-01", periods=10, freq="W")
    states = pd.Series(0, index=idx)  # never risk-off
    rng = np.random.default_rng(0)
    risk_on = pd.Series(rng.normal(0.001, 0.02, size=10), index=idx)
    defensive = pd.Series(rng.normal(0.0001, 0.005, size=10), index=idx)
    r = run_backtest(states, risk_on, defensive, tc_bps_per_leg=10.0)
    # No flips -> strategy returns == risk_on returns
    np.testing.assert_allclose(r.weekly_strategy.values, risk_on.values, atol=1e-12)
    assert r.metrics["n_flips"] == 0


def test_backtest_charges_tc_on_flip():
    idx = pd.date_range("2020-01-01", periods=6, freq="W")
    # toggle each week starting after one warmup -> flips at weeks 1,2,3,4,5
    states = pd.Series([0, 0, 1, 1, 0, 0], index=idx)
    risk_on = pd.Series(0.0, index=idx)   # flat returns
    defensive = pd.Series(0.0, index=idx)
    r = run_backtest(states, risk_on, defensive, tc_bps_per_leg=10.0)
    # 2 flips after the 1-week lag: state goes 0,0,0,1,1,0 (lagged) -> flips at idx 3 and idx 5
    assert r.metrics["n_flips"] == 2
    # Each flip costs 2 * 10 bps = 20 bps = 0.002 in fraction
    expected = np.array([0.0, 0.0, 0.0, -0.002, 0.0, -0.002])
    np.testing.assert_allclose(r.weekly_strategy.values, expected, atol=1e-12)
