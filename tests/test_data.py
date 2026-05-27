"""Invariants on the data layer."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from sentinel_alpha.data.loader import load_dataset
from sentinel_alpha.data.transforms import stationarize, NonPositiveValuesError


def test_load_dataset_invariants():
    d = load_dataset()
    assert d.X.shape[0] == 1111
    assert d.X.shape[1] == 42
    assert d.y.shape[0] == 1111
    assert d.y.isin([0, 1]).all()
    # weekly cadence
    gaps = d.X.index.to_series().diff().dt.days.dropna().unique()
    assert len(gaps) == 1 and gaps[0] == 7
    # every X column has a metadata type
    assert set(d.X.columns) <= set(d.type_map)


def test_stationarize_no_nan_and_drops_first_row():
    d = load_dataset()
    Z = stationarize(d.X, d.type_map)
    assert not Z.isna().any().any()
    assert len(Z) == len(d.X) - 1
    # VIX produces both level and difference columns
    assert "VIX_lvl" in Z.columns and "VIX_d" in Z.columns
    # at least one bps_diff column for rates (Germany 2Y can go negative -> must not be logret)
    assert "GTDEM2Y_dbps" in Z.columns


def test_logret_raises_on_negatives():
    s = pd.Series([1.0, 2.0, -0.5, 1.0], name="bad")
    X = s.to_frame()
    type_map = {"bad": "Equity Index"}
    with pytest.raises(NonPositiveValuesError):
        stationarize(X, type_map)


def test_bps_diff_scale():
    # First difference of yields is in percent points * 100 = bps
    s = pd.Series([2.50, 2.55, 2.40], name="r")  # in percent
    X = s.to_frame()
    type_map = {"r": "Bond Yield"}
    Z = stationarize(X, type_map)
    np.testing.assert_allclose(Z["r_dbps"].values, np.array([5.0, -15.0]))
