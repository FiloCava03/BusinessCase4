"""No look-ahead invariants on engineered features."""
from __future__ import annotations
import numpy as np
import pandas as pd

from sentinel_alpha.data.loader import load_dataset
from sentinel_alpha.data.transforms import stationarize
from sentinel_alpha.features.engineer import add_engineered


def test_features_no_lookahead():
    """Mutating a row at time t must not change any feature value at time < t."""
    d = load_dataset()
    Z = stationarize(d.X, d.type_map)
    F1, _ = add_engineered(Z)

    perturb_loc = 600
    perturb_time = Z.index[perturb_loc]
    Z2 = Z.copy()
    cols_to_perturb = ["MXUS_logret", "VIX_lvl", "LUACTRUU_logret"]
    for c in cols_to_perturb:
        Z2.iloc[perturb_loc, Z2.columns.get_loc(c)] += 100.0
    F2, _ = add_engineered(Z2)

    # Strict causality: any feature at time strictly < perturb_time must match.
    before = F1.index[F1.index < perturb_time]
    diff = (F1.loc[before] - F2.loc[before]).abs().max().max()
    assert diff < 1e-10, f"look-ahead detected: max diff before perturbation = {diff}"


def test_features_shape_and_no_nan():
    d = load_dataset()
    Z = stationarize(d.X, d.type_map)
    F, ra = add_engineered(Z)
    assert not F.isna().any().any()
    assert not ra.isna().any()
    # Should have engineered columns
    expected_eng = {"ENG_eq_vol4w", "ENG_eq_credit_corr4w", "ENG_term_spread_d",
                    "ENG_credit_excess", "ENG_vix_z52", "ENG_vix_d", "ENG_eq_dm"}
    assert expected_eng <= set(F.columns)
