"""Detector contracts: shape, monotonicity on injected outlier, determinism."""
from __future__ import annotations
import numpy as np
import pytest

from sentinel_alpha.config import SEED
from sentinel_alpha.detectors import DETECTOR_REGISTRY


def _make_data(n_normal: int = 400, n_anom: int = 5, d: int = 12, seed: int = SEED):
    rng = np.random.default_rng(seed)
    cov = np.eye(d)
    Xn = rng.multivariate_normal(mean=np.zeros(d), cov=cov, size=n_normal)
    # Anomalies are placed far from the normal blob along all axes.
    Xa = rng.multivariate_normal(mean=12 * np.ones(d), cov=cov, size=n_anom)
    X = np.vstack([Xn, Xa])
    y = np.array([0] * n_normal + [1] * n_anom)
    return X.astype(float), y


@pytest.mark.parametrize("name", list(DETECTOR_REGISTRY.keys()))
def test_detector_shape_and_top_recovery(name: str):
    """Detector must place anomalies disproportionately in the top of the score
    distribution. We check that at least 60% of the injected outliers fall in
    the top-(2 * n_anom) scores -- a mild rank-based contract that all six
    detectors should be able to satisfy on this trivially-separated synthetic
    set without overfitting to the GMM unsupervised-collapse failure mode.
    """
    X, y = _make_data()
    det = DETECTOR_REGISTRY[name]()
    det.fit(X, y)
    s = det.score_samples(X)
    assert s.shape == (X.shape[0],)
    assert np.all(np.isfinite(s))
    n_anom = int(y.sum())
    top_k = 2 * n_anom
    top_idx = np.argsort(-s)[:top_k]
    hits = (y[top_idx] == 1).sum()
    assert hits >= int(0.6 * n_anom), (
        f"{name}: only {hits}/{n_anom} outliers in top-{top_k} (rank check)"
    )


@pytest.mark.parametrize("name", list(DETECTOR_REGISTRY.keys()))
def test_detector_deterministic(name: str):
    X, y = _make_data()
    s1 = DETECTOR_REGISTRY[name]().fit(X, y).score_samples(X)
    s2 = DETECTOR_REGISTRY[name]().fit(X, y).score_samples(X)
    np.testing.assert_allclose(s1, s2, rtol=1e-5, atol=1e-6,
                               err_msg=f"{name} not deterministic")
