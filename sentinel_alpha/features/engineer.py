"""Engineered features on top of the stationarized matrix. No look-ahead."""
from __future__ import annotations
import numpy as np
import pandas as pd

from sentinel_alpha.config import VOL_WIN, CORR_WIN, Z_WIN

# Pruning policy (see notebook Step 4 diagnostic, all computed pre-hold-out).
# ENG candidates with univariate AUC < 0.55 -- dropped.
WEAK_ENG: tuple[str, ...] = ("ENG_term_spread_d", "ENG_eq_credit_corr4w")
# Lag-1 allow-list: of the 43 lag-1 candidates, only VIX_lvl_lag1 survives the
# ablation: it is the single most predictive feature in the panel (uAUC ~ 0.85)
# and adds CV signal on top of the contemporaneous block. The other 42 lag-1
# columns add net noise to a linear stacker, so we drop them.
KEEP_LAG1: tuple[str, ...] = ("VIX_lvl_lag1",)


def _safe_get(Z: pd.DataFrame, name: str) -> pd.Series:
    if name not in Z.columns:
        raise KeyError(f"required column '{name}' missing from stationarized matrix")
    return Z[name]


def _rolling_zscore(s: pd.Series, win: int) -> pd.Series:
    """Causal rolling z-score with expanding warm-up. Never bfill (would leak)."""
    mu = s.rolling(window=win, min_periods=8).mean()
    sd = s.rolling(window=win, min_periods=8).std(ddof=0)
    z = (s - mu) / sd.replace(0.0, np.nan)
    # warm-up region (first <8 obs) is filled with 0.0 -- causal, no leakage
    return z.fillna(0.0)


def add_engineered(Z: pd.DataFrame, prune_weak: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Add engineered columns to `Z`. Returns (augmented_Z, risk_appetite).

    The risk_appetite series is exported separately because the strategy gate
    consumes it directly, distinct from the detector feature matrix.

    When ``prune_weak=True`` (default), applies the post-benchmark pruning
    policy described in the Step 4 diagnostic (notebook): drop the two ENG
    features below uAUC 0.55, and keep only ``VIX_lvl_lag1`` from the lag-1
    block (the rest are net noise to the linear stacker).
    """
    out = Z.copy()

    mxwo_proxy = _safe_get(Z, "MXUS_logret")           # broad equity proxy
    mxeu = _safe_get(Z, "MXEU_logret")
    mxjp = _safe_get(Z, "MXJP_logret")
    vix_lvl = _safe_get(Z, "VIX_lvl")
    vix_d = _safe_get(Z, "VIX_d")
    luac = _safe_get(Z, "LUACTRUU_logret")             # US Corporate IG
    lf98 = _safe_get(Z, "LF98TRUU_logret")             # US Corporate HY
    us30 = _safe_get(Z, "USGG30YR_dbps")
    us3m = _safe_get(Z, "USGG3M_dbps")

    # Equity composite (DM): average of US/EU/JP weekly log-returns.
    eq_dm = ((mxwo_proxy + mxeu + mxjp) / 3.0).rename("ENG_eq_dm")
    out[eq_dm.name] = eq_dm

    # 4-week realized vol of equity composite.
    out["ENG_eq_vol4w"] = eq_dm.rolling(VOL_WIN, min_periods=2).std(ddof=0).fillna(0.0)

    # 4-week rolling correlation of equity and IG credit (equity-credit decoupling).
    out["ENG_eq_credit_corr4w"] = (
        mxwo_proxy.rolling(CORR_WIN, min_periods=3).corr(luac).fillna(0.0)
    )

    # Term spread change (curve steepening/flattening).
    out["ENG_term_spread_d"] = (us30 - us3m).rename("ENG_term_spread_d")

    # Credit spread proxy (HY - IG weekly return spread).
    out["ENG_credit_excess"] = (lf98 - luac).rename("ENG_credit_excess")

    # VIX regime: rolling z of level.
    out["ENG_vix_z52"] = _rolling_zscore(vix_lvl, Z_WIN)
    out["ENG_vix_d"] = vix_d

    # Risk-appetite composite: positive means risk-on.
    # eq_dm rolling 4w mean minus 0.5 * VIX rolling-z (causal).
    eq_dm_4w = eq_dm.rolling(VOL_WIN, min_periods=2).mean().fillna(0.0)
    risk_appetite = (eq_dm_4w - 0.5 * out["ENG_vix_z52"]).rename("risk_appetite")

    # Lag-1 of every transformed feature (no look-ahead).
    base_cols = [c for c in Z.columns]
    lagged = Z[base_cols].shift(1).add_suffix("_lag1")
    out = pd.concat([out, lagged], axis=1)

    # Drop the first row introduced by lag-1.
    out = out.iloc[1:]
    risk_appetite = risk_appetite.loc[out.index]

    if prune_weak:
        weak_eng = [c for c in WEAK_ENG if c in out.columns]
        all_lag1 = [c for c in out.columns if c.endswith("_lag1")]
        drop_lag1 = [c for c in all_lag1 if c not in KEEP_LAG1]
        out = out.drop(columns=weak_eng + drop_lag1)

    # Hard guard: same rationale as stationarize() -- use `raise` so the check
    # is not silently stripped by `python -O`.
    if out.isna().any().any():
        bad = out.columns[out.isna().any()].tolist()
        raise ValueError(
            f"NaNs leaked through add_engineered() in columns: {bad[:10]}"
            + (f" ... (+{len(bad) - 10} more)" if len(bad) > 10 else "")
        )
    return out, risk_appetite
