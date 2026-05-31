"""Smoke tests for the explain/projection.py UMAP wrappers."""
from __future__ import annotations
import importlib
import numpy as np
import pandas as pd
import pytest

from sentinel_alpha.explain.projection import confusion_buckets


def test_confusion_buckets_assigns_every_row():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0])
    buckets = confusion_buckets(y_true, y_pred)
    expected = np.array(["TN", "FP", "TP", "FN", "TP", "TN"])
    np.testing.assert_array_equal(buckets, expected)


def test_confusion_buckets_handles_all_negatives():
    y = np.zeros(10, dtype=int)
    buckets = confusion_buckets(y, y)
    assert (buckets == "TN").all()


@pytest.mark.skipif(
    importlib.util.find_spec("umap") is None,
    reason="umap-learn not installed",
)
def test_umap_embed_smoke():
    """Run UMAP on a tiny synthetic matrix. Just checks shape and finiteness."""
    from sentinel_alpha.explain.projection import umap_embed
    rng = np.random.default_rng(0)
    F = pd.DataFrame(rng.normal(size=(60, 8)))
    emb = umap_embed(F, n_neighbors=5)
    assert emb.shape == (60, 2)
    assert np.isfinite(emb).all()
