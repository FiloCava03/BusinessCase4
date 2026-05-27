"""Kernel PCA reconstruction-error anomaly detector."""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import KernelPCA

from sentinel_alpha.detectors.base import AnomalyDetector
from sentinel_alpha.config import SEED


class KPCADetector(AnomalyDetector):
    def __init__(self, n_components: int = 8, kernel: str = "rbf",
                 gamma: float | None = None, random_state: int = SEED) -> None:
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "KPCADetector":
        Xf = np.asarray(X, dtype=float)
        if y is not None:
            mask = (np.asarray(y) == 0)
            if mask.sum() >= max(self.n_components + 2, 30):
                Xf = Xf[mask]
        self.model_ = KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma,
            fit_inverse_transform=True,
            random_state=self.random_state,
            n_jobs=1,
        ).fit(Xf)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        Xf = np.asarray(X, dtype=float)
        Xrec = self.model_.inverse_transform(self.model_.transform(Xf))
        return np.mean((Xf - Xrec) ** 2, axis=1)
