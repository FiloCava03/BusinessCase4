"""SHAP attribution on the logistic stacker."""
from __future__ import annotations
import numpy as np
import pandas as pd


def stacker_shap_values(
    stacker, Q_train: np.ndarray, Q_val: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """SHAP values for a linear stacker on the val set.

    Uses `shap.LinearExplainer` with an `independent` masker fit on Q_train,
    which is appropriate for a sklearn LogisticRegression in [0,1]-bounded
    rank-quantile space. Returns a DataFrame [n_val x n_features].
    """
    import shap
    explainer = shap.LinearExplainer(stacker.model_, masker=shap.maskers.Independent(Q_train))
    sv = explainer.shap_values(np.asarray(Q_val, dtype=float))
    if isinstance(sv, list):  # multi-output fallback
        sv = sv[1] if len(sv) > 1 else sv[0]
    return pd.DataFrame(sv, columns=feature_names)


def aggregate_shap_by_crisis(
    sv: pd.DataFrame, index: pd.DatetimeIndex,
    crises: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Mean SHAP per detector aggregated within each named crisis window."""
    sv = sv.copy(); sv.index = index
    rows = []
    for name, (s, e) in crises.items():
        m = (sv.index >= pd.Timestamp(s)) & (sv.index <= pd.Timestamp(e))
        if m.any():
            rows.append({"crisis": name, **sv.loc[m].mean().to_dict()})
    return pd.DataFrame(rows).set_index("crisis") if rows else pd.DataFrame()
