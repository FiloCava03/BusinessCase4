"""Head-to-head comparison: prof baselines vs Sentinel-alpha, same hold-out.

Run:  python -m sentinel_alpha.compare
Outputs:
    artifacts/comparison_table.parquet  (long-format pandas)
    artifacts/comparison_table.md       (Markdown table for the deck/README)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, brier_score_loss,
)

from sentinel_alpha.config import ARTIFACTS_DIR, SEED, CRISES
from sentinel_alpha.utils.seeding import set_global_seed
from sentinel_alpha.data.loader import load_dataset
from sentinel_alpha.data.transforms import stationarize
from sentinel_alpha.features.engineer import add_engineered
from sentinel_alpha.cv.walkforward import PurgedExpandingSplit
from sentinel_alpha.stack import StackPipeline
from sentinel_alpha.strategy import (
    apply_gate, hysteresis, run_backtest, build_strategy_returns,
)
from sentinel_alpha.stack.calibrate import expected_calibration_error
from sentinel_alpha.baseline import PROF_BASELINES


def _classifier_row(name: str, y_true: np.ndarray, p: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": name,
        "AUC":       float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) == 2 else np.nan,
        "PR-AUC":    float(average_precision_score(y_true, p)) if len(np.unique(y_true)) == 2 else np.nan,
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "F1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "Brier":     float(brier_score_loss(y_true, _to01(p))) if len(np.unique(y_true)) == 2 else np.nan,
        "ECE":       float(expected_calibration_error(y_true, _to01(p))),
    }


def _to01(x: np.ndarray) -> np.ndarray:
    """Min-max compress to [0,1] for Brier/ECE when the score is not a probability."""
    x = np.asarray(x, dtype=float)
    if x.min() >= 0.0 and x.max() <= 1.0:
        return x
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _backtest_row(states_eff: pd.Series, ron: pd.Series, dfd: pd.Series,
                  ho_index: pd.DatetimeIndex) -> dict:
    states_eff = states_eff.reindex(ho_index).fillna(0).astype(int)
    bt = run_backtest(states_eff, ron, dfd, crises=CRISES)
    covid_off = 0.0
    if not bt.crisis_metrics.empty:
        mask = bt.crisis_metrics["crisis"] == "COVID"
        if mask.any():
            covid_off = float(bt.crisis_metrics.loc[mask, "off_rate"].iloc[0])
    return {
        "AnnRet":      bt.metrics["ann_return"],
        "AnnVol":      bt.metrics["ann_vol"],
        "Sharpe":      bt.metrics["sharpe"],
        "Sortino":     bt.metrics["sortino"],
        "MaxDD":       bt.metrics["max_drawdown"],
        "Calmar":      bt.metrics["calmar"],
        "Turnover/y":  bt.metrics["turnover_per_year"],
        "Flips":       int(bt.metrics["n_flips"]),
        "OffRate_COVID": covid_off,
    }


def _naive_strategy(preds: np.ndarray, ho_index: pd.DatetimeIndex) -> pd.Series:
    """Convert binary predictions to a state series, no hysteresis / gate."""
    return pd.Series(np.asarray(preds, dtype=int), index=ho_index, name="state")


def run_comparison() -> pd.DataFrame:
    set_global_seed(SEED)

    # 1. Data ----------------------------------------------------------------
    d = load_dataset()
    Z = stationarize(d.X, d.type_map)
    F, ra = add_engineered(Z)
    y = d.y.reindex(F.index).astype(int)

    splitter = PurgedExpandingSplit()
    ho_idx = splitter.holdout_idx(F.index)
    train_mask = np.ones(len(F), dtype=bool); train_mask[ho_idx] = False
    Xtr_arr, ytr_arr = F.values[train_mask], y.values[train_mask]
    Xho_arr, yho_arr = F.values[ho_idx],     y.values[ho_idx]
    ho_index = F.index[ho_idx]

    # 2. Returns (raw prices) ------------------------------------------------
    prices_ho = d.X.loc[ho_index]
    ron, dfd = build_strategy_returns(prices_ho)

    rows: list[dict] = []

    # 3. Buy-and-hold (no signal) -------------------------------------------
    bh_states = _naive_strategy(np.zeros(ho_idx.size, dtype=int), ho_index)
    bh_class = {
        "model": "Buy_and_hold", "AUC": np.nan, "PR-AUC": np.nan,
        "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
        "Brier": float(np.mean((yho_arr - yho_arr.mean()) ** 2)),
        "ECE": float("nan"),
    }
    rows.append({**bh_class, **_backtest_row(bh_states, ron, dfd, ho_index)})

    # 4. Prof baselines ------------------------------------------------------
    for name, fn in PROF_BASELINES.items():
        try:
            res = fn(Xtr_arr, ytr_arr, Xho_arr)
        except Exception as e:
            print(f"[compare] {name}: skipped ({type(e).__name__}: {e})")
            continue
        cls = _classifier_row(name, yho_arr, res.scores_holdout, res.preds_holdout)
        states = _naive_strategy(res.preds_holdout, ho_index)
        bt = _backtest_row(states, ron, dfd, ho_index)
        rows.append({**cls, **bt})

    # 5. Sentinel-alpha ------------------------------------------------------
    pipe = StackPipeline().fit(Xtr_arr, ytr_arr)
    p_raw_ho = pipe.predict_proba_raw(Xho_arr)
    p_cal_ho = pipe.predict_proba(Xho_arr)

    # Use the same threshold-tuning recipe as run.py (CV-only) ---------------
    try:
        cv_proba = pd.read_parquet(ARTIFACTS_DIR / "cv_probabilities.parquet")
        ra_aligned = ra.reindex(cv_proba.index).dropna()
        cv_proba = cv_proba.loc[ra_aligned.index]
        prices_cv = d.X.loc[cv_proba.index]
        from sentinel_alpha.strategy import tune_thresholds, ThresholdGrid
        risk_on_cv, defensive_cv = build_strategy_returns(prices_cv)
        tuned = tune_thresholds(
            p=cv_proba["p_raw"], risk_appetite=ra_aligned,
            risk_on_simple=risk_on_cv, defensive_simple=defensive_cv,
            grid=ThresholdGrid(),
        )
        enter, exit_, dwell, tau = tuned["enter"], tuned["exit"], tuned["dwell"], tuned["tau"]
    except FileNotFoundError:
        enter, exit_, dwell, tau = 0.55, 0.35, 2, 1.0

    signal = apply_gate(p_raw_ho, ra.iloc[ho_idx].values, tau=tau)
    sentinel_states = pd.Series(
        hysteresis(signal, enter=enter, exit_=exit_, dwell=dwell),
        index=ho_index,
    )
    # F1-based threshold on p_raw at the same recall regime as baselines.
    # We pick the threshold maximising F1 on the CV out-of-fold predictions,
    # so it's *not* tuned on the hold-out.
    from sklearn.metrics import f1_score
    cv_y = y.reindex(cv_proba.index).values
    grid = np.quantile(cv_proba["p_raw"].values, np.linspace(0.5, 0.99, 50))
    f1s = [f1_score(cv_y, (cv_proba["p_raw"].values >= g).astype(int), zero_division=0) for g in grid]
    f1_thr = float(grid[int(np.argmax(f1s))])
    sentinel_preds_for_class = (p_raw_ho >= f1_thr).astype(int)

    cls_row = _classifier_row("Sentinel_alpha_stack", yho_arr, p_raw_ho, sentinel_preds_for_class)
    cls_row["Brier"] = float(brier_score_loss(yho_arr, p_cal_ho))
    cls_row["ECE"] = float(expected_calibration_error(yho_arr, p_cal_ho))
    bt_row = _backtest_row(sentinel_states, ron, dfd, ho_index)
    rows.append({**cls_row, **bt_row})

    df = pd.DataFrame(rows)
    return df.set_index("model")


def to_markdown(df: pd.DataFrame) -> str:
    metric_order = [
        "AUC", "PR-AUC", "Precision", "Recall", "F1", "Brier", "ECE",
        "AnnRet", "AnnVol", "Sharpe", "Sortino", "MaxDD", "Calmar",
        "Turnover/y", "Flips", "OffRate_COVID",
    ]
    df = df[metric_order].copy()
    fmt = {
        "AUC": "{:.3f}", "PR-AUC": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
        "F1": "{:.3f}", "Brier": "{:.3f}", "ECE": "{:.3f}",
        "AnnRet": "{:.1%}", "AnnVol": "{:.1%}", "Sharpe": "{:.2f}", "Sortino": "{:.2f}",
        "MaxDD": "{:.1%}", "Calmar": "{:.2f}", "Turnover/y": "{:.2f}", "Flips": "{:.0f}",
        "OffRate_COVID": "{:.0%}",
    }
    out_lines = []
    out_lines.append("| Model | " + " | ".join(metric_order) + " |")
    out_lines.append("|---|" + "|".join(["---"] * len(metric_order)) + "|")
    for model, row in df.iterrows():
        cells = []
        for col in metric_order:
            v = row[col]
            if pd.isna(v):
                cells.append("—")
            else:
                cells.append(fmt[col].format(v))
        out_lines.append(f"| **{model}** | " + " | ".join(cells) + " |")
    return "\n".join(out_lines)


def main() -> None:
    df = run_comparison()
    df.to_parquet(ARTIFACTS_DIR / "comparison_table.parquet")
    md = to_markdown(df)
    (ARTIFACTS_DIR / "comparison_table.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
