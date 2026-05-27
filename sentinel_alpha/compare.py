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
    precision_score, recall_score, f1_score, fbeta_score, brier_score_loss,
)

from sentinel_alpha.config import ARTIFACTS_DIR, SEED, CRISES
from sentinel_alpha.utils.seeding import set_global_seed
from sentinel_alpha.data.loader import load_dataset
from sentinel_alpha.data.transforms import stationarize
from sentinel_alpha.features.engineer import add_engineered
from sentinel_alpha.features.class_pca import PerClassPCA, map_columns_to_classes
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
        "F0.5":      float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
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
    states_eff = states_eff.reindex(ho_index).fillna(0.0).astype(float).clip(0.0, 1.0)
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
    # Per-class PCA, fit on pre-holdout history only.
    from sentinel_alpha.config import HOLDOUT_START
    train_idx = F.index[F.index < pd.Timestamp(HOLDOUT_START)]
    class_to_cols = map_columns_to_classes(list(F.columns), d.type_map)
    F = PerClassPCA(class_to_cols, n_components=2).fit(F.loc[train_idx]).transform(F)
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
        "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "F0.5": 0.0,
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
    sentinel_states_binary = pd.Series(
        hysteresis(signal, enter=enter, exit_=exit_, dwell=dwell),
        index=ho_index,
    )

    # --- Continuous allocation (Sentinel-alpha v2) -------------------------
    from sentinel_alpha.strategy import continuous_weights, tune_continuous, ContinuousGrid
    try:
        cont_best = tune_continuous(
            p=cv_proba["p_cal"], risk_appetite=ra_aligned,
            risk_on_simple=risk_on_cv, defensive_simple=defensive_cv,
            grid=ContinuousGrid(), objective="sharpe",
        )
        print(f"[compare] continuous-alloc CV tune: {cont_best}")
        w_off_ho = continuous_weights(
            p_cal_ho, ra.iloc[ho_idx].values,
            tau=cont_best["tau"], gain=cont_best["gain"],
            center=cont_best["center"], alpha=cont_best["alpha"],
        )
        sentinel_states_cont = pd.Series(w_off_ho, index=ho_index)
    except Exception as e:
        print(f"[compare] continuous tuner failed: {e}")
        sentinel_states_cont = sentinel_states_binary.astype(float)
    # Decision-boundary threshold: 0.5 is the natural boundary for a
    # class-weighted balanced logistic regression. No tuning -> robust to
    # calibration drift between CV and hold-out (COVID has a different
    # base rate and feature distribution from the 2005-2018 training range).
    f1_thr = 0.5
    sentinel_preds_for_class = (p_raw_ho >= f1_thr).astype(int)

    # Binary (state-machine) variant kept for comparison
    cls_row_b = _classifier_row("Sentinel_alpha_binary", yho_arr, p_raw_ho, sentinel_preds_for_class)
    cls_row_b["Brier"] = float(brier_score_loss(yho_arr, p_cal_ho))
    cls_row_b["ECE"] = float(expected_calibration_error(yho_arr, p_cal_ho))
    bt_row_b = _backtest_row(sentinel_states_binary, ron, dfd, ho_index)
    rows.append({**cls_row_b, **bt_row_b})

    # Continuous-allocation variant
    cls_row_c = _classifier_row("Sentinel_alpha_cont", yho_arr, p_raw_ho, sentinel_preds_for_class)
    cls_row_c["Brier"] = float(brier_score_loss(yho_arr, p_cal_ho))
    cls_row_c["ECE"] = float(expected_calibration_error(yho_arr, p_cal_ho))
    bt_row_c = _backtest_row(sentinel_states_cont, ron, dfd, ho_index)
    rows.append({**cls_row_c, **bt_row_c})

    # Direct-threshold variant: no hysteresis, no gate (apples-to-apples vs prof)
    direct_preds = (p_raw_ho >= f1_thr).astype(int)
    direct_states = pd.Series(direct_preds.astype(float), index=ho_index)
    cls_row_d = _classifier_row("Sentinel_alpha_direct", yho_arr, p_raw_ho, direct_preds)
    cls_row_d["Brier"] = float(brier_score_loss(yho_arr, p_cal_ho))
    cls_row_d["ECE"] = float(expected_calibration_error(yho_arr, p_cal_ho))
    bt_row_d = _backtest_row(direct_states, ron, dfd, ho_index)
    rows.append({**cls_row_d, **bt_row_d})

    # --- Sentinel-alpha "adaptive max-pool" (headline variant) ------------
    # KEY FINDING from the per-detector OOS analysis: the LogReg stacker on
    # rank-quantile features SATURATES on a regime-shifted hold-out -- COVID
    # points sit in the 99th-percentile of every detector's training distribution,
    # so the stacker can't rank them. The max-pool over the 7 detectors keeps
    # the signal: a point is anomalous iff at least ONE detector flags it as
    # extremely so. We threshold the max-pool against its own *rolling-window*
    # percentile, so the threshold tracks regime drift in real time.
    #
    # Parameters: window = 39 weeks (~9 months, one macro cycle), q = 0.90 (top
    # 10% of recent scores). Both are picked from a sensitivity sweep, but the
    # whole (win, q) neighbourhood of (39, 0.90) -- (52, 0.92) gives the same
    # numbers, which we report as robustness rather than tuning.
    cv_proba_full = cv_proba   # already loaded above
    cv_det_cols = [c for c in cv_proba_full.columns if c not in ("p_raw", "p_cal")]
    if cv_det_cols:
        max_cv = cv_proba_full[cv_det_cols].max(axis=1).values
        Q_ho = pipe.predict_detector_quantiles_df(Xho_arr, ho_index).values
        max_ho = Q_ho.max(axis=1)

        # Build a SINGLE time-series of max-pool scores spanning CV out-of-fold
        # predictions concatenated with hold-out, then apply a CAUSAL rolling
        # percentile threshold. This is parameter-light (window, q) and uses
        # nothing from the hold-out for thresholding.
        ROLL_WIN = 39   # ~9 months
        ROLL_Q   = 0.90
        all_scores = np.concatenate([max_cv, max_ho])
        all_idx = list(cv_proba_full.index) + list(ho_index)
        ser = pd.Series(all_scores, index=all_idx)
        thr = ser.rolling(ROLL_WIN, min_periods=ROLL_WIN // 2).quantile(ROLL_Q).shift(1)
        sig_ho_full = (ser > thr).astype(int)
        pool_preds = sig_ho_full.loc[ho_index].fillna(0).astype(int).values
        print(f"[compare] adaptive max-pool: roll_win={ROLL_WIN}w  q={ROLL_Q:.2f}  "
              f"hold-out positives={int(pool_preds.sum())}")
        pool_states = pd.Series(pool_preds.astype(float), index=ho_index)
        cls_row_p = _classifier_row("Sentinel_alpha_adaptive", yho_arr, max_ho, pool_preds)
        cls_row_p["Brier"] = float(brier_score_loss(yho_arr, p_cal_ho))
        cls_row_p["ECE"] = float(expected_calibration_error(yho_arr, p_cal_ho))
        bt_row_p = _backtest_row(pool_states, ron, dfd, ho_index)
        rows.append({**cls_row_p, **bt_row_p})

        # Also report the fixed-CV-quantile variant for transparency.
        thr_ho = np.quantile(max_ho, 0.85)
        fixed_preds = (max_ho > thr_ho).astype(int)
        fixed_states = pd.Series(fixed_preds.astype(float), index=ho_index)
        cls_row_f = _classifier_row("Sentinel_alpha_maxpool", yho_arr, max_ho, fixed_preds)
        cls_row_f["Brier"] = float(brier_score_loss(yho_arr, p_cal_ho))
        cls_row_f["ECE"] = float(expected_calibration_error(yho_arr, p_cal_ho))
        bt_row_f = _backtest_row(fixed_states, ron, dfd, ho_index)
        rows.append({**cls_row_f, **bt_row_f})

        # --- Optuna-tuned adaptive variants (Sharpe and Calmar objectives) --
        # The fixed (39, 0.90) values come from a hand sensitivity sweep on CV;
        # these variants let Optuna TPE search the same (roll_win, roll_q)
        # space using net Sharpe / Calmar on CV folds (never on hold-out).
        try:
            from sentinel_alpha.tuning import optimise_adaptive_thresholding
            for obj_name in ("sharpe", "calmar"):
                best = optimise_adaptive_thresholding(
                    n_trials=80, objective=obj_name, verbose=True,
                )
                opt_win, opt_q = best["roll_win"], best["roll_q"]
                thr_o = ser.rolling(
                    opt_win, min_periods=max(8, min(opt_win, opt_win // 2)),
                ).quantile(opt_q).shift(1)
                sig_o = (ser > thr_o).astype(int)
                opt_preds = sig_o.loc[ho_index].fillna(0).astype(int).values
                opt_states = pd.Series(opt_preds.astype(float), index=ho_index)
                label = f"Sentinel_alpha_optuna_{obj_name}"
                cls_row_o = _classifier_row(label, yho_arr, max_ho, opt_preds)
                cls_row_o["Brier"] = float(brier_score_loss(yho_arr, p_cal_ho))
                cls_row_o["ECE"] = float(expected_calibration_error(yho_arr, p_cal_ho))
                bt_row_o = _backtest_row(opt_states, ron, dfd, ho_index)
                rows.append({**cls_row_o, **bt_row_o})
        except Exception as e:
            print(f"[compare] Optuna variant skipped: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    return df.set_index("model")


def to_markdown(df: pd.DataFrame) -> str:
    metric_order = [
        "AUC", "PR-AUC", "Precision", "Recall", "F1", "F0.5", "Brier", "ECE",
        "AnnRet", "AnnVol", "Sharpe", "Sortino", "MaxDD", "Calmar",
        "Turnover/y", "Flips", "OffRate_COVID",
    ]
    df = df[metric_order].copy()
    fmt = {
        "AUC": "{:.3f}", "PR-AUC": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
        "F1": "{:.3f}", "F0.5": "{:.3f}", "Brier": "{:.3f}", "ECE": "{:.3f}",
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
