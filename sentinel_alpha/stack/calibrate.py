"""Isotonic calibration on top of a probability source."""
from __future__ import annotations
import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso_.fit(np.asarray(p_raw, dtype=float).ravel(),
                      np.asarray(y, dtype=float).ravel())
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        return self.iso_.transform(np.asarray(p_raw, dtype=float).ravel())


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = y.size
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)
