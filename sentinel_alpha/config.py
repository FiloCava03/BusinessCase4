"""Global configuration: seed, paths, hyperparameters, defensive-sleeve weights.

This module is the **single source of truth** for every constant used across
the package. Magic numbers in submodules must import from here.

Business angle
--------------
Sentinel-alpha is framed as a **Risk Management** (defensive overlay) system,
not a Quant Strategy. The Early Warning System raises calibrated risk-off
probabilities that drive a long-only switch between an equity book (MXUS) and
a defensive sleeve. All hyperparameter choices below reflect this framing:
- objective metric: ``calmar`` (return per unit of drawdown), not ``sharpe``;
- thresholds tuned to favour false positives over false negatives in crisis;
- hysteresis dwell to suppress costly flip-flopping in calm regimes.
"""
from __future__ import annotations
from pathlib import Path

SEED: int = 42

PACKAGE_ROOT: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_ROOT.parent
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

DATA_FILE: Path = PROJECT_ROOT / "Dataset4_EWS.xlsx"

# ---------------------------------------------------------------------------
# Walk-forward CV (Lopez de Prado, ch. 7)
# ---------------------------------------------------------------------------
INITIAL_TRAIN_END: str = "2004-12-31"
HOLDOUT_START: str = "2019-01-01"
HOLDOUT_END: str = "2021-04-20"
PURGE_WEEKS: int = 5      # gap between train_end and val_start (no leakage of train into val)
EMBARGO_WEEKS: int = 2    # gap after val_end before the next val (no leakage of val into next train)
VAL_LEN_WEEKS: int = 52   # one calendar year per validation window

# ---------------------------------------------------------------------------
# Feature engineering windows (data/features layer)
# ---------------------------------------------------------------------------
VOL_WIN: int = 4          # 4-week realized vol of equity composite
CORR_WIN: int = 4         # 4-week rolling correlation (eq vs IG credit)
Z_WIN: int = 52           # 1-year z-score window (VIX regime, recovery composite)

# ---------------------------------------------------------------------------
# Stacker / calibration
# ---------------------------------------------------------------------------
# Fraction of the training fold reserved for isotonic calibration. 0.20 is the
# smallest value that consistently yields at least 2 positives in the calibration
# sub-fold across early walk-forward iterations.
CAL_FRAC: float = 0.20

# Validation fraction inside the AE / AE-ensemble training loop (for early
# stopping). Kept small so the AE sees as much of the training fold as possible.
AE_VAL_FRAC: float = 0.15

# ---------------------------------------------------------------------------
# Strategy thresholds (defaults; tuned in-fold via tune_thresholds / Optuna)
# ---------------------------------------------------------------------------
# Tuned on CV with objective = calmar; values reflect an asymmetric overlay
# (enter at a higher probability than we exit) to avoid premature re-risking.
ENTER_RISK_OFF: float = 0.55
EXIT_RISK_OFF: float = 0.35
DWELL_WEEKS: int = 2          # consecutive sub-/super-threshold weeks before flipping state
GATE_TAU: float = 0.0         # risk-appetite gate: signal blocked when RA >= tau

# Default optimization objective for in-fold and Optuna tuners. Coherent with
# the Risk Management angle: maximize return per unit of max drawdown.
DEFAULT_TUNING_OBJECTIVE: str = "calmar"

# ---------------------------------------------------------------------------
# Defensive sleeve composition (used by backtest)
# ---------------------------------------------------------------------------
DEFENSIVE_WEIGHTS: dict[str, float] = {
    "LF94TRUU": 0.50,  # Global Inflation-Linked
    "XAUBGNL":  0.25,  # Gold spot
    "CASH":     0.25,  # USGG3M / 52 weekly
}

# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
TC_BPS_PER_LEG: float = 10.0  # 10 bps each side per leg flip

# ---------------------------------------------------------------------------
# Named crises (start, end) inclusive -- used for diagnostic reporting and for
# the per-crisis stress table at the heart of the Risk Management story.
# ---------------------------------------------------------------------------
CRISES: dict[str, tuple[str, str]] = {
    "Dotcom":             ("2000-03-01", "2002-12-31"),
    "GFC":                ("2007-10-01", "2009-06-30"),
    "EU sovereign":       ("2011-07-01", "2012-12-31"),
    "2015-16 China/oil":  ("2015-08-01", "2016-02-29"),
    "2018 Q4":            ("2018-10-01", "2018-12-31"),
    "COVID":              ("2020-02-01", "2020-12-31"),
}

# ---------------------------------------------------------------------------
# Per-ticker overrides where the Metadata "Type" is misleading.
# BDIY is a positive freight-cost index, logret-able (Zenti groups it with commodities).
# ---------------------------------------------------------------------------
TRANSFORM_OVERRIDES: dict[str, str] = {
    "BDIY": "logret",
}

# Stationarity transform rule by metadata "Type"
TRANSFORM_BY_TYPE: dict[str, str] = {
    "Bond Yield":       "bps_diff",
    "Interest rate":    "bps_diff",
    "Bond Index":       "logret",
    "Equity Index":     "logret",
    "Currency":         "logret",
    "Commodity":        "logret",
    "Commodity Index":  "logret",
    "Hedge Fund Index": "logret",
    "Futures Contract": "logret",
    "Volatility Index": "level_and_diff",
    "Economic Index":   "level",
}
