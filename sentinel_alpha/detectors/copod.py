"""Copula-Based Outlier Detection (PyOD wrapper)."""
from __future__ import annotations
import numpy as np

from sentinel_alpha.detectors.base import AnomalyDetector


class COPODDetector(AnomalyDetector):
    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "COPODDetector":
        from pyod.models.copod import COPOD
        self.model_ = COPOD()
        self.model_.fit(np.asarray(X, dtype=float))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        # PyOD: higher decision score -> more anomalous (matches our convention).
        return self.model_.decision_function(np.asarray(X, dtype=float))
