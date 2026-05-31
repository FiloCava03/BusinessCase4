"""Calibration invariants: isotonic monotonicity, output bounds, ECE/Brier ranges."""
from __future__ import annotations
import numpy as np

from sentinel_alpha.stack.calibrate import (
    IsotonicCalibrator,
    brier_score,
    expected_calibration_error,
)


def test_isotonic_output_in_unit_interval():
    rng = np.random.default_rng(0)
    p_raw = rng.uniform(0.0, 1.0, size=500)
    y = (rng.uniform(0.0, 1.0, size=500) < p_raw).astype(int)
    cal = IsotonicCalibrator().fit(p_raw, y)
    p_cal = cal.transform(p_raw)
    assert p_cal.min() >= 0.0 and p_cal.max() <= 1.0


def test_isotonic_is_monotone_non_decreasing():
    rng = np.random.default_rng(1)
    p_raw = rng.uniform(0.0, 1.0, size=400)
    y = (p_raw + 0.1 * rng.normal(size=400) > 0.5).astype(int)
    cal = IsotonicCalibrator().fit(p_raw, y)
    grid = np.linspace(0.0, 1.0, 101)
    p_cal_grid = cal.transform(grid)
    # Monotone non-decreasing in p_raw.
    diffs = np.diff(p_cal_grid)
    assert (diffs >= -1e-12).all(), "isotonic calibrator must be non-decreasing"


def test_brier_score_perfect_predictions_is_zero():
    y = np.array([0, 0, 1, 1, 0, 1])
    assert brier_score(y, y.astype(float)) == 0.0


def test_brier_score_in_unit_interval():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=200)
    p = rng.uniform(0.0, 1.0, size=200)
    bs = brier_score(y, p)
    assert 0.0 <= bs <= 1.0


def test_ece_in_unit_interval():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=200)
    p = rng.uniform(0.0, 1.0, size=200)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_ece_perfectly_calibrated_low():
    # Synthetic perfectly-calibrated data: p ~ U(0,1), y ~ Bernoulli(p).
    rng = np.random.default_rng(4)
    p = rng.uniform(0.0, 1.0, size=5000)
    y = (rng.uniform(0.0, 1.0, size=5000) < p).astype(int)
    ece = expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.05, f"perfectly calibrated synth data ECE {ece:.3f} too high"
