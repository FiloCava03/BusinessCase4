"""Walk-forward expanding CV with purge and embargo (Lopez de Prado, ch. 7)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Generator
import numpy as np
import pandas as pd

from sentinel_alpha.config import (
    INITIAL_TRAIN_END,
    HOLDOUT_START,
    HOLDOUT_END,
    PURGE_WEEKS,
    EMBARGO_WEEKS,
    VAL_LEN_WEEKS,
)


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex


class PurgedExpandingSplit:
    """Expanding-window CV with `purge` weeks between train and val, and `embargo`
    weeks after each val block before the next can start.

    The hold-out window [HOLDOUT_START, HOLDOUT_END] is never returned.
    """

    def __init__(
        self,
        initial_train_end: str = INITIAL_TRAIN_END,
        holdout_start: str = HOLDOUT_START,
        holdout_end: str = HOLDOUT_END,
        purge_weeks: int = PURGE_WEEKS,
        embargo_weeks: int = EMBARGO_WEEKS,
        val_len_weeks: int = VAL_LEN_WEEKS,
    ) -> None:
        self.initial_train_end = pd.Timestamp(initial_train_end)
        self.holdout_start = pd.Timestamp(holdout_start)
        self.holdout_end = pd.Timestamp(holdout_end)
        self.purge = purge_weeks
        self.embargo = embargo_weeks
        self.val_len = val_len_weeks

    def split(self, index: pd.DatetimeIndex) -> Generator[Fold, None, None]:
        """Yield Fold objects walking from `initial_train_end` to `holdout_start`.

        Leakage controls
        ----------------
        * **Purge** (``self.purge`` weeks): gap *before* every validation block,
          preventing train labels from leaking into val.
        * **Embargo** (``self.embargo`` weeks): gap *after* every validation
          block. These weeks are explicitly **excluded from the next fold's
          training set** -- they form a post-val "trust gap" that prevents
          val_k labels from leaking into train_{k+1}.
        """
        idx = pd.DatetimeIndex(index)
        n = len(idx)
        pos = lambda ts: int(np.searchsorted(idx.values, np.datetime64(ts), side="left"))

        train_end_pos = pos(self.initial_train_end) - 1
        holdout_start_pos = pos(self.holdout_start)

        fold_id = 0
        # First fold: pre-val gap = purge only (no prior val exists).
        # Subsequent folds: pre-val gap = purge + embargo, so the embargo
        # weeks immediately after val_{k-1} never enter train_k either.
        gap_before_val = self.purge
        while True:
            val_start = train_end_pos + 1 + gap_before_val
            val_end = val_start + self.val_len  # exclusive
            if val_end > holdout_start_pos or val_end > n:
                break
            train_idx = np.arange(0, train_end_pos + 1, dtype=int)
            val_idx = np.arange(val_start, val_end, dtype=int)
            yield Fold(
                fold_id=fold_id,
                train_idx=train_idx,
                val_idx=val_idx,
                train_dates=idx[train_idx],
                val_dates=idx[val_idx],
            )
            fold_id += 1
            # Expanding training: include the val block we just produced, but
            # NOT the embargo window after it (that gap stays excluded forever).
            train_end_pos = val_end - 1
            gap_before_val = self.purge + self.embargo

    def holdout_idx(self, index: pd.DatetimeIndex) -> np.ndarray:
        idx = pd.DatetimeIndex(index)
        mask = (idx >= self.holdout_start) & (idx <= self.holdout_end)
        return np.where(mask)[0]

    def folds(self, index: pd.DatetimeIndex) -> list[Fold]:
        return list(self.split(index))
