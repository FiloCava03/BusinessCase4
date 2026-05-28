"""Global configuration: seed, paths, hyper-priors, defensive-sleeve weights."""
from __future__ import annotations
from pathlib import Path

SEED: int = 42

PACKAGE_ROOT: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_ROOT.parent
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

DATA_FILE: Path = PROJECT_ROOT / "Dataset4_EWS.xlsx"

# Walk-forward CV
INITIAL_TRAIN_END: str = "2004-12-31"
HOLDOUT_START: str = "2019-01-01"
HOLDOUT_END: str = "2021-04-20"
PURGE_WEEKS: int = 5
EMBARGO_WEEKS: int = 2
VAL_LEN_WEEKS: int = 52

# Strategy thresholds (defaults; tuned in-fold)
ENTER_RISK_OFF: float = 0.55
EXIT_RISK_OFF: float = 0.35
DWELL_WEEKS: int = 2
GATE_TAU: float = 0.0

# Defensive sleeve composition (used by backtest)
DEFENSIVE_WEIGHTS: dict[str, float] = {
    "LF94TRUU": 0.50,  # Global Inflation-Linked
    "XAUBGNL":  0.25,  # Gold spot
    "CASH":     0.25,  # USGG3M / 52 weekly
}

# Costs
TC_BPS_PER_LEG: float = 10.0  # 10 bps each side per leg flip

# Named crises (start, end) inclusive
CRISES: dict[str, tuple[str, str]] = {
    "Dotcom":             ("2000-03-01", "2002-12-31"),
    "GFC":                ("2007-10-01", "2009-06-30"),
    "EU sovereign":       ("2011-07-01", "2012-12-31"),
    "2015-16 China/oil":  ("2015-08-01", "2016-02-29"),
    "2018 Q4":            ("2018-10-01", "2018-12-31"),
    "COVID":              ("2020-02-01", "2020-12-31"),
}

# Per-ticker overrides where the Metadata "Type" is misleading.
# BDIY is a positive freight-cost index, logret-able (Zenti groups it with commodities).
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
