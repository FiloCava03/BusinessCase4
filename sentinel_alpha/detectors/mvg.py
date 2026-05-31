"""Multivariate Gaussian with Ledoit-Wolf shrinkage. Trained on normal data only."""
from __future__ import annotations
import logging
import numpy as np
from sklearn.covariance import LedoitWolf
from scipy.stats import multivariate_normal

from sentinel_alpha.detectors.base import AnomalyDetector

_log = logging.getLogger(__name__)


class LedoitWolfMVG(AnomalyDetector):
    """Score = negative log-pdf under N(mu, Sigma_shrunk).

    If `y` is provided, the model is fit on rows where y == 0 (novelty setup);
    otherwise on the whole batch (outlier setup).

    If `y` is provided but contains fewer than ``n_features + 2`` normal rows,
    the detector falls back to fitting on the whole batch (warning logged).
    """

    def __init__(self, allow_singular: bool = True) -> None:
        self.allow_singular = allow_singular

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "LedoitWolfMVG":
        Xf = np.asarray(X, dtype=float)
        if y is not None:
            mask = (np.asarray(y) == 0)
            min_required = Xf.shape[1] + 2
            if mask.sum() < min_required:
                # not enough normal rows to fit; fall back to full batch
                _log.warning(
                    "LedoitWolfMVG: only %d normal rows (need %d). "
                    "Falling back to full-batch fit (outlier setup).",
                    int(mask.sum()), int(min_required),
                )
                Xf_fit = Xf
            else:
                Xf_fit = Xf[mask]
        else:
            Xf_fit = Xf
        lw = LedoitWolf().fit(Xf_fit)
        self.mu_ = lw.location_
        self.sigma_ = lw.covariance_
        self.dist_ = multivariate_normal(
            mean=self.mu_, cov=self.sigma_, allow_singular=self.allow_singular
        )
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return -self.dist_.logpdf(np.asarray(X, dtype=float))
