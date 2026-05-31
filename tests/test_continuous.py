"""Continuous-allocation strategy invariants."""
from __future__ import annotations
import numpy as np
import pandas as pd

from sentinel_alpha.strategy.continuous import continuous_weights, tune_continuous, ContinuousGrid


def test_continuous_weights_in_unit_interval():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, size=200)
    ra = rng.normal(0.0, 1.0, size=200)
    w = continuous_weights(p, ra, tau=1e9)  # gate disabled
    assert w.min() >= 0.0 and w.max() <= 1.0


def test_continuous_weights_gate_zeroes_when_ra_high():
    p = np.array([0.9, 0.9, 0.9, 0.9])
    ra = np.array([2.0, 2.0, 2.0, 2.0])  # all above tau=0
    w = continuous_weights(p, ra, tau=0.0)
    assert (w == 0.0).all()


def test_continuous_weights_identity_when_alpha_one_gain_one():
    # alpha=1 disables EMA, gain=1, center=0 -> w = p (clipped).
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    ra = np.array([-1.0, -1.0, -1.0, -1.0, -1.0])  # always below tau=0 -> gate open
    w = continuous_weights(p, ra, tau=0.0, gain=1.0, center=0.0, alpha=1.0)
    np.testing.assert_allclose(w, p)


def test_continuous_weights_smoothing_dampens_spikes():
    p = np.array([0.0] * 50 + [1.0] + [0.0] * 50)
    ra = np.full_like(p, -1.0)
    w_raw = continuous_weights(p, ra, tau=0.0, alpha=1.0)
    w_smooth = continuous_weights(p, ra, tau=0.0, alpha=0.3)
    # Smoothed peak should be strictly lower than the raw spike.
    assert w_smooth.max() < w_raw.max()


def test_tune_continuous_returns_well_formed_dict():
    """Smoke test: tuner must return a dict with the objective metric set."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2010-01-01", periods=200, freq="W")
    p = pd.Series(rng.uniform(0.0, 1.0, size=200), index=idx)
    ra = pd.Series(rng.normal(0.0, 1.0, size=200), index=idx)
    risk_on = pd.Series(rng.normal(0.001, 0.02, size=200), index=idx)
    defensive = pd.Series(rng.normal(0.0005, 0.005, size=200), index=idx)
    # Tiny grid for speed.
    grid = ContinuousGrid(gains=(1.0,), centers=(0.0,), alphas=(1.0,), taus=(1e9,))
    out = tune_continuous(p, ra, risk_on, defensive, grid=grid, objective="calmar")
    assert "calmar" in out
    assert "gain" in out and "center" in out and "alpha" in out and "tau" in out
