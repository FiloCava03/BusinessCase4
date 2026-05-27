"""Isolation Forest wrapper."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest

from sentinel_alpha.detectors.base import AnomalyDetector
from sentinel_alpha.config import SEED


class IForestDetector(AnomalyDetector):
    def __init__(self, n_estimators: int = 400, contamination: float | str = "auto",
                 max_samples: int | str = "auto", random_state: int = SEED) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IForestDetector":
        cont: float | str
        if isinstance(self.contamination, str):
            cont = self.contamination
        else:
            cont = float(np.clip(self.contamination, 1e-4, 0.5))
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=cont,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=1,
        ).fit(np.asarray(X, dtype=float))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # decision_function: higher = more normal. Negate so higher = more anomalous.
        return -self.model_.decision_function(np.asarray(X, dtype=float))
