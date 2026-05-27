"""Stationarity transforms by asset type. No look-ahead."""
from __future__ import annotations
import numpy as np
import pandas as pd

from sentinel_alpha.config import TRANSFORM_BY_TYPE, TRANSFORM_OVERRIDES


class NonPositiveValuesError(ValueError):
    """Raised when log-returns are requested on a series with non-positive values."""


def _logret(s: pd.Series) -> pd.Series:
    if (s <= 0).any():
        raise NonPositiveValuesError(
            f"Cannot compute log-returns: '{s.name}' has non-positive values "
            f"(min={s.min():.4f}). Use first-differences instead."
        )
    return np.log(s).diff()


def _bps_diff(s: pd.Series) -> pd.Series:
    # First difference, expressed in basis points (×100 since input is in %).
    return s.diff() * 100.0


def _level(s: pd.Series) -> pd.Series:
    return s.astype(float)


def stationarize(
    X: pd.DataFrame,
    type_map: dict[str, str],
) -> pd.DataFrame:
    """Apply per-type transforms using the rule in `config.TRANSFORM_BY_TYPE`.

    Returns a DataFrame indexed identically to `X`, with the first row dropped
    (because of the differencing).
    """
    cols: list[pd.Series] = []
    for c in X.columns:
        rule = TRANSFORM_OVERRIDES.get(c)
        if rule is None:
            typ = type_map[c]
            rule = TRANSFORM_BY_TYPE.get(typ)
            if rule is None:
                raise KeyError(f"No transform rule for type '{typ}' (column '{c}')")
        s = X[c]
        if rule == "logret":
            out = _logret(s).rename(f"{c}_logret")
            cols.append(out)
        elif rule == "bps_diff":
            out = _bps_diff(s).rename(f"{c}_dbps")
            cols.append(out)
        elif rule == "level":
            cols.append(_level(s).rename(f"{c}_lvl"))
        elif rule == "level_and_diff":
            cols.append(_level(s).rename(f"{c}_lvl"))
            cols.append(s.diff().rename(f"{c}_d"))
        else:
            raise ValueError(f"Unknown rule '{rule}' for column '{c}'")
    out = pd.concat(cols, axis=1).iloc[1:]  # drop first row (NaN from diff/logret)
    assert not out.isna().any().any(), "NaNs leaked through stationarize()"
    return out
