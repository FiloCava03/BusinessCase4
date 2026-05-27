"""Load Dataset4_EWS.xlsx into a tidy DataFrame plus an asset-class meta map."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

from sentinel_alpha.config import DATA_FILE


@dataclass(frozen=True)
class MarketData:
    """Container with strict invariants enforced at load time."""
    X: pd.DataFrame                  # DatetimeIndex weekly, no NaN
    y: pd.Series                     # binary, aligned with X
    type_map: dict[str, str]         # column -> asset-class string from Metadata


# Tickers present in metadata but absent from the Markets sheet (no leakage risk).
_OPTIONAL_TICKERS: set[str] = {
    "MXWO", "MXWD", "LEGATRUU", "HFRXGL",
    "RX1", "TY1", "GC1", "CO1", "ES1", "VG1", "NQ1", "LLL1", "TP1", "DU1", "TU2",
}


def load_dataset(path: Path | str = DATA_FILE) -> MarketData:
    """Load the Bloomberg EWS dataset with strict invariants:

    - weekly cadence (every gap exactly 7 days);
    - no NaNs in the Markets sheet;
    - all numeric columns aligned with their Metadata `Type`.

    Raises AssertionError on violation rather than silently coercing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    markets = pd.read_excel(path, sheet_name="Markets")
    meta = pd.read_excel(path, sheet_name="Metadata")

    if "Data" not in markets.columns:
        raise ValueError("Markets sheet missing 'Data' column")
    markets = markets.sort_values("Data").reset_index(drop=True)
    dt = pd.to_datetime(markets["Data"])
    gaps = dt.diff().dt.days.dropna().unique()
    if not (len(gaps) == 1 and gaps[0] == 7):
        raise AssertionError(f"Non-weekly cadence detected: gap days = {gaps}")
    markets = markets.set_index(pd.DatetimeIndex(dt, name="date")).drop(columns=["Data"])

    if markets.isna().any().any():
        bad = markets.columns[markets.isna().any()].tolist()
        raise AssertionError(f"NaNs found in columns: {bad}")

    if "Y" not in markets.columns:
        raise ValueError("Markets sheet missing 'Y' label column")
    y = markets["Y"].astype(int).rename("Y")
    X = markets.drop(columns=["Y"]).astype(float)

    # Build the type map for present tickers only.
    meta = meta.rename(columns={"Variable name": "name", "Type": "type"})
    type_map: dict[str, str] = {}
    for _, row in meta.iterrows():
        # The Metadata sheet uses inconsistent whitespace (e.g. "XAU BGNL"
        # vs the column header "XAUBGNL"). Collapse spaces for matching.
        name = "".join(str(row["name"]).split())
        typ = str(row["type"]).strip()
        if name in X.columns:
            type_map[name] = typ
    # Sanity: every column in X must have a type.
    missing_type = sorted(set(X.columns) - set(type_map))
    if missing_type:
        raise AssertionError(f"Columns without metadata type: {missing_type}")

    return MarketData(X=X, y=y, type_map=type_map)
