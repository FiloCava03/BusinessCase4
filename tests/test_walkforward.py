"""Walk-forward CV invariants."""
from __future__ import annotations
import numpy as np
import pandas as pd

from sentinel_alpha.data.loader import load_dataset
from sentinel_alpha.data.transforms import stationarize
from sentinel_alpha.features.engineer import add_engineered
from sentinel_alpha.cv.walkforward import PurgedExpandingSplit
from sentinel_alpha.config import PURGE_WEEKS, HOLDOUT_START, HOLDOUT_END


def _index() -> pd.DatetimeIndex:
    d = load_dataset()
    Z = stationarize(d.X, d.type_map)
    F, _ = add_engineered(Z)
    return F.index


def test_split_yields_folds():
    splitter = PurgedExpandingSplit()
    folds = splitter.folds(_index())
    assert len(folds) >= 10, f"expected >=10 folds, got {len(folds)}"


def test_no_train_val_overlap():
    splitter = PurgedExpandingSplit()
    for f in splitter.folds(_index()):
        assert np.intersect1d(f.train_idx, f.val_idx).size == 0


def test_purge_respected():
    splitter = PurgedExpandingSplit()
    for f in splitter.folds(_index()):
        gap = int(f.val_idx.min()) - int(f.train_idx.max())
        assert gap >= 1 + PURGE_WEEKS, f"gap {gap} < required {1 + PURGE_WEEKS}"


def test_holdout_never_appears():
    idx = _index()
    splitter = PurgedExpandingSplit()
    ho = splitter.holdout_idx(idx)
    assert ho.size > 0
    ho_set = set(ho.tolist())
    for f in splitter.folds(idx):
        assert ho_set.isdisjoint(f.train_idx.tolist())
        assert ho_set.isdisjoint(f.val_idx.tolist())


def test_train_window_expands_monotonically():
    splitter = PurgedExpandingSplit()
    folds = splitter.folds(_index())
    prev_max = -1
    for f in folds:
        cur_max = int(f.train_idx.max())
        assert cur_max > prev_max
        prev_max = cur_max
