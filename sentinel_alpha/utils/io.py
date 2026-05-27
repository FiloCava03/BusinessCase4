"""Lightweight artifact persistence helpers."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

from sentinel_alpha.config import ARTIFACTS_DIR


def save_parquet(df: pd.DataFrame, name: str) -> Path:
    path = ARTIFACTS_DIR / f"{name}.parquet"
    df.to_parquet(path)
    return path


def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(ARTIFACTS_DIR / f"{name}.parquet")


def save_json(obj: dict, name: str) -> Path:
    path = ARTIFACTS_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path
