"""Smoke test for the stack pipeline on synthetic data."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score

from sentinel_alpha.stack import StackPipeline


def test_stack_pipeline_synthetic():
    rng = np.random.default_rng(42)
    d = 10
    n_normal_train, n_anom_train = 600, 60
    n_normal_test, n_anom_test = 200, 20
    Xtr = np.vstack([
        rng.standard_normal((n_normal_train, d)),
        rng.standard_normal((n_anom_train, d)) + 4.5,
    ])
    ytr = np.array([0]*n_normal_train + [1]*n_anom_train)
    Xte = np.vstack([
        rng.standard_normal((n_normal_test, d)),
        rng.standard_normal((n_anom_test, d)) + 4.5,
    ])
    yte = np.array([0]*n_normal_test + [1]*n_anom_test)
    # shuffle train so the cal sub-fold has some anomalies
    perm = rng.permutation(len(ytr))
    Xtr, ytr = Xtr[perm], ytr[perm]

    pipe = StackPipeline().fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)
    assert p.shape == (Xte.shape[0],)
    assert ((p >= 0.0) & (p <= 1.0)).all()
    auc = roc_auc_score(yte, p)
    assert auc > 0.9, f"stack AUC {auc:.3f} too low on easy synthetic"
