"""L2 logistic-regression stacker on rank-quantile detector features."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression

from sentinel_alpha.config import SEED


class LogRegStacker:
    """Tiny wrapper around `LogisticRegression` that returns probability of Y=1."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000,
                 class_weight: str | dict = "balanced", random_state: int = SEED) -> None:
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, Q: np.ndarray, y: np.ndarray) -> "LogRegStacker":
        # sklearn 1.8 deprecated penalty='l2'; the default is L2 with C= the
        # inverse regularization strength, so we just pass C.
        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver="lbfgs",
        ).fit(np.asarray(Q, dtype=float), np.asarray(y).ravel())
        return self

    def predict_proba(self, Q: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(np.asarray(Q, dtype=float))[:, 1]

    @property
    def coef_(self) -> np.ndarray:
        return self.model_.coef_[0]

    @property
    def intercept_(self) -> float:
        return float(self.model_.intercept_[0])
