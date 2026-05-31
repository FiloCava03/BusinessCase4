"""Convert raw anomaly scores to training-fold empirical quantiles."""
from __future__ import annotations
import numpy as np


class EmpiricalQuantile:
    """Fit the empirical CDF on a 1-D training score vector and apply it to new data.

    The transform maps each query value ``x`` to ``mean(train_scores <= x)``,
    i.e. its right-continuous empirical-CDF rank, and returns a number in
    ``[0, 1]``.

    Out-of-range behaviour (important in an EWS)
    --------------------------------------------
    * A query value **strictly above** ``max(train_scores)`` maps to ``1.0``
      (saturation). Any test sample more anomalous than every training row
      is treated as "maximally anomalous" -- a deliberately conservative
      choice for risk-management use cases: extreme tail events get the
      largest possible rank-feature signal rather than an extrapolated one.
    * A query value **strictly below** ``min(train_scores)`` maps to ``0.0``.
    * Ties are handled with ``side="right"`` so a query equal to ``s_k`` in
      the training set receives rank ``k / n``.

    Use ``EmpiricalQuantile().fit(...).transform(...)`` or ``fit_transform``.
    """

    def fit(self, scores: np.ndarray) -> "EmpiricalQuantile":
        s = np.asarray(scores, dtype=float).ravel()
        if s.size == 0:
            raise ValueError("EmpiricalQuantile.fit requires non-empty scores")
        self.sorted_ = np.sort(s)
        self.n_ = s.size
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float).ravel()
        ranks = np.searchsorted(self.sorted_, s, side="right")
        return ranks / float(self.n_)

    def fit_transform(self, scores: np.ndarray) -> np.ndarray:
        return self.fit(scores).transform(scores)
