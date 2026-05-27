"""Supervised detectors: Random Forest and Logistic Regression.

These are wrapped as `AnomalyDetector` so the stacker sees them as one more
score function. The score is the predicted probability of class 1 -- already
in [0, 1] and monotone in "anomalousness".
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sentinel_alpha.detectors.base import AnomalyDetector
from sentinel_alpha.config import SEED


class RFDetector(AnomalyDetector):
    def __init__(self, n_estimators: int = 300, max_depth: int | None = None,
                 class_weight: str | dict = "balanced",
                 random_state: int = SEED) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "RFDetector":
        if y is None:
            raise ValueError("RFDetector requires labels y at fit time.")
        y_arr = np.asarray(y).ravel()
        if len(np.unique(y_arr)) < 2:
            # Degenerate fold: fall back to a constant predictor.
            self.model_ = None
            self.const_ = float(y_arr.mean())
            return self
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=1,
        ).fit(np.asarray(X, dtype=float), y_arr)
        self.const_ = None
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.model_ is None:
            return np.full(X.shape[0], self.const_, dtype=float)
        return self.model_.predict_proba(X)[:, 1]


class LogRegDetector(AnomalyDetector):
    def __init__(self, C: float = 1.0, max_iter: int = 2000,
                 class_weight: str | dict = "balanced",
                 random_state: int = SEED) -> None:
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LogRegDetector":
        if y is None:
            raise ValueError("LogRegDetector requires labels y at fit time.")
        y_arr = np.asarray(y).ravel()
        if len(np.unique(y_arr)) < 2:
            self.model_ = None
            self.const_ = float(y_arr.mean())
            return self
        self.model_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state, solver="lbfgs",
        ).fit(np.asarray(X, dtype=float), y_arr)
        self.const_ = None
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.model_ is None:
            return np.full(X.shape[0], self.const_, dtype=float)
        return self.model_.predict_proba(X)[:, 1]
