"""Local Outlier Factor anomaly detector (novelty mode)."""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from sentinel_alpha.detectors.base import AnomalyDetector


class LOFDetector(AnomalyDetector):
    """LOF in novelty mode: fit on rows where y == 0, score new points.

    The LOF anomaly score is the negative of `score_samples` from sklearn
    (which is defined so that *higher* means more normal); we flip the sign
    so that higher score == more anomalous, matching the package convention.
    """

    def __init__(self, n_neighbors: int = 20, metric: str = "minkowski",
                 contamination: float | str = "auto") -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.contamination = contamination

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LOFDetector":
        Xf = np.asarray(X, dtype=float)
        if y is not None:
            mask = (np.asarray(y) == 0)
            if mask.sum() >= max(self.n_neighbors + 2, 30):
                Xf_fit = Xf[mask]
            else:
                Xf_fit = Xf
        else:
            Xf_fit = Xf
        n_neighbors = int(min(self.n_neighbors, max(2, Xf_fit.shape[0] - 1)))
        self.model_ = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            metric=self.metric,
            contamination=self.contamination,
            novelty=True,
            n_jobs=1,
        ).fit(Xf_fit)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # sklearn: higher score_samples = more normal. Negate so higher = more anomalous.
        return -self.model_.score_samples(np.asarray(X, dtype=float))
