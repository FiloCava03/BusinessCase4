"""Aggregate evaluation metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, fbeta_score,
    precision_score, recall_score, brier_score_loss,
)

from sentinel_alpha.stack.calibrate import expected_calibration_error


def classification_report_dict(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    p = np.asarray(p).ravel()
    y_pred = (p >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) == 2 else float("nan"),
        "pr_auc":  float(average_precision_score(y_true, p)) if len(np.unique(y_true)) == 2 else float("nan"),
        "f1":      float(f1_score(y_true, y_pred, zero_division=0)),
        "f0_5":    float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":  float(recall_score(y_true, y_pred, zero_division=0)),
        "brier":   float(brier_score_loss(y_true, p)) if len(np.unique(y_true)) == 2 else float("nan"),
        "ece":     float(expected_calibration_error(y_true, p)),
        "n":       int(y_true.size),
        "n_pos":   int(y_true.sum()),
    }


def regime_wise_report(
    y_true: pd.Series, p: pd.Series, crises: dict[str, tuple[str, str]],
    threshold: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for name, (s, e) in crises.items():
        m = (y_true.index >= pd.Timestamp(s)) & (y_true.index <= pd.Timestamp(e))
        if not m.any():
            continue
        y = y_true[m].values; pp = p[m].values
        if y.sum() == 0 or y.sum() == len(y):
            row = {"crisis": name, "n": int(m.sum()), "n_pos": int(y.sum())}
        else:
            r = classification_report_dict(y, pp, threshold=threshold)
            row = {"crisis": name, **r}
        rows.append(row)
    return pd.DataFrame(rows)
