"""Anomaly-detector ABC. Higher score == more anomalous."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator


class AnomalyDetector(ABC, BaseEstimator):
    """All detectors expose `fit(X, y=None)` and `score_samples(X) -> ndarray[N]`.

    The y parameter is ignored by unsupervised detectors but accepted to keep a
    uniform sklearn-style signature. Detectors that consume labels (e.g. the
    novelty AE trained on Y=0 rows only) read them inside `fit`.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AnomalyDetector":
        ...

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_score(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).score_samples(X)
