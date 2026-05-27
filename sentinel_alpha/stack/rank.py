"""Convert raw anomaly scores to training-fold empirical quantiles."""
from __future__ import annotations
import numpy as np


class EmpiricalQuantile:
    """Fit the empirical CDF on a 1-D training score vector and apply it to new data.

    Right-continuous: each query value x is mapped to mean( train_scores <= x ).
    Output is in [0, 1].
    """

    def fit(self, scores: np.ndarray) -> "EmpiricalQuantile":
        s = np.asarray(scores, dtype=float).ravel()
        self.sorted_ = np.sort(s)
        self.n_ = s.size
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float).ravel()
        ranks = np.searchsorted(self.sorted_, s, side="right")
        return ranks / float(self.n_)

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        return self.fit(scores).transform(scores)
