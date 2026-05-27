"""Gaussian Mixture Model (k=3 by default) trained unsupervised on the full batch."""
from __future__ import annotations
import numpy as np
from sklearn.mixture import GaussianMixture

from sentinel_alpha.detectors.base import AnomalyDetector
from sentinel_alpha.config import SEED


class GMMDetector(AnomalyDetector):
    def __init__(self, n_components: int = 3, covariance_type: str = "full",
                 reg_covar: float = 1e-4, random_state: int = SEED) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "GMMDetector":
        Xf = np.asarray(X, dtype=float)
        if y is not None:
            mask = (np.asarray(y) == 0)
            if mask.sum() >= max(3 * self.n_components, 30):
                Xf = Xf[mask]
        self.model_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=self.random_state,
            max_iter=300,
            n_init=3,
        ).fit(Xf)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self.model_.score_samples(np.asarray(X, dtype=float))
