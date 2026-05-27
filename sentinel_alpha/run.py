"""End-to-end runner: data -> features -> CV -> stack -> backtest -> artifacts.

Usage
-----
    python -m sentinel_alpha.run --stage all
    python -m sentinel_alpha.run --stage data
    python -m sentinel_alpha.run --stage cv
    python -m sentinel_alpha.run --stage backtest
"""
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from sentinel_alpha.config import (
    SEED, ARTIFACTS_DIR, CRISES,
    ENTER_RISK_OFF, EXIT_RISK_OFF, DWELL_WEEKS, GATE_TAU,
)
from sentinel_alpha.utils.seeding import set_global_seed
from sentinel_alpha.utils.io import save_parquet, save_json
from sentinel_alpha.data.loader import load_dataset
from sentinel_alpha.data.transforms import stationarize
from sentinel_alpha.features.engineer import add_engineered
from sentinel_alpha.cv.walkforward import PurgedExpandingSplit
from sentinel_alpha.stack import StackPipeline
from sentinel_alpha.strategy import (
    apply_gate, hysteresis, run_backtest, build_strategy_returns,
    tune_thresholds, ThresholdGrid,
)


def _log(msg: str) -> None:
    print(f"[sentinel_alpha] {msg}", flush=True)


def stage_data() -> None:
    set_global_seed(SEED)
    d = load_dataset()
    Z = stationarize(d.X, d.type_map)
    F, ra = add_engineered(Z)
    save_parquet(F, "features")
    save_parquet(d.y.to_frame().rename(columns={0: "Y"}), "labels")  # binary
    ra.to_frame().to_parquet(ARTIFACTS_DIR / "risk_appetite.parquet")
    d.X.to_parquet(ARTIFACTS_DIR / "prices_raw.parquet")
    _log(f"saved features {F.shape}, labels {d.y.shape}, risk_appetite, prices_raw")


def stage_cv() -> None:
    set_global_seed(SEED)
    d = load_dataset()
    Z = stationarize(d.X, d.type_map)
    F, _ = add_engineered(Z)
    y = d.y.reindex(F.index).astype(int)
    splitter = PurgedExpandingSplit()
    folds = splitter.folds(F.index)

    rows = []
    proba_all: list[pd.Series] = []
    proba_raw_all: list[pd.Series] = []
    detector_q_all: list[pd.DataFrame] = []
    for fold in folds:
        Xtr = F.values[fold.train_idx]; ytr = y.values[fold.train_idx]
        Xva = F.values[fold.val_idx];   yva = y.values[fold.val_idx]
        if ytr.sum() < 5:
            rows.append({"fold": fold.fold_id, "skipped": "train has <5 positives"})
            continue
        pipe = StackPipeline().fit(Xtr, ytr)
        p_raw = pipe.predict_proba_raw(Xva)
        p_cal = pipe.predict_proba(Xva)
        proba_raw_all.append(pd.Series(p_raw, index=fold.val_dates, name="p_raw"))
        proba_all.append(pd.Series(p_cal, index=fold.val_dates, name="p_cal"))
        detector_q_all.append(
            pipe.predict_detector_quantiles_df(Xva, fold.val_dates)
        )
        auc = roc_auc_score(yva, p_raw) if len(np.unique(yva)) == 2 else np.nan
        ap = average_precision_score(yva, p_raw) if len(np.unique(yva)) == 2 else np.nan
        rows.append({
            "fold": fold.fold_id,
            "train_start": fold.train_dates[0], "train_end": fold.train_dates[-1],
            "val_start": fold.val_dates[0], "val_end": fold.val_dates[-1],
            "n_train": int(len(ytr)), "n_train_pos": int(ytr.sum()),
            "n_val": int(len(yva)), "n_val_pos": int(yva.sum()),
            "auc_raw": float(auc), "pr_auc_raw": float(ap),
        })
    folds_df = pd.DataFrame(rows)
    save_parquet(folds_df.assign(
        train_start=folds_df.get("train_start", pd.NaT),
        train_end=folds_df.get("train_end", pd.NaT),
        val_start=folds_df.get("val_start", pd.NaT),
        val_end=folds_df.get("val_end", pd.NaT),
    ), "cv_folds")
    if proba_all:
        proba_cal_concat = pd.concat(proba_all)
        proba_raw_concat = pd.concat(proba_raw_all)
        detector_concat = pd.concat(detector_q_all)
        out_cv = pd.concat(
            [proba_raw_concat.rename("p_raw"), proba_cal_concat.rename("p_cal"),
             detector_concat], axis=1)
        save_parquet(out_cv, "cv_probabilities")
    _log(f"CV done: {len(folds_df)} fold rows, AUC mean = "
         f"{folds_df['auc_raw'].mean(skipna=True):.3f}")


def stage_backtest() -> None:
    set_global_seed(SEED)
    d = load_dataset()
    Z = stationarize(d.X, d.type_map)
    F, ra = add_engineered(Z)
    y = d.y.reindex(F.index).astype(int)

    splitter = PurgedExpandingSplit()
    ho_idx = splitter.holdout_idx(F.index)
    if ho_idx.size == 0:
        raise RuntimeError("Empty hold-out window")

    # Use ALL non-holdout data as the training set for the final pipeline.
    train_mask = np.ones(len(F), dtype=bool); train_mask[ho_idx] = False
    Xtr = F.values[train_mask]; ytr = y.values[train_mask]
    Xho = F.values[ho_idx];     yho = y.values[ho_idx]
    ra_ho = ra.iloc[ho_idx]
    holdout_index = F.index[ho_idx]

    pipe = StackPipeline().fit(Xtr, ytr)
    p_raw_ho = pipe.predict_proba_raw(Xho)
    p_cal_ho = pipe.predict_proba(Xho)

    # ---- Tune thresholds on CV (out-of-fold) predictions, never on hold-out ---
    try:
        cv_proba = pd.read_parquet(ARTIFACTS_DIR / "cv_probabilities.parquet")
        ra_aligned = ra.reindex(cv_proba.index).dropna()
        cv_proba = cv_proba.loc[ra_aligned.index]
        # Build cv-period returns from raw prices for the tuner.
        prices_cv = d.X.loc[cv_proba.index]
        risk_on_cv, defensive_cv = build_strategy_returns(prices_cv)
        # Tune on p_raw (less compressed than p_cal, more sensitive to thresholds).
        tuned = tune_thresholds(
            p=cv_proba["p_raw"],
            risk_appetite=ra_aligned,
            risk_on_simple=risk_on_cv,
            defensive_simple=defensive_cv,
            grid=ThresholdGrid(),
        )
        _log(f"tuned thresholds on CV: {tuned}")
        enter, exit_, dwell, tau = tuned["enter"], tuned["exit"], tuned["dwell"], tuned["tau"]
        # Use p_raw on the hold-out too (consistent with tuner).
        p_for_signal = p_raw_ho
    except FileNotFoundError:
        _log("cv_probabilities not found; falling back to config defaults")
        enter, exit_, dwell, tau = ENTER_RISK_OFF, EXIT_RISK_OFF, DWELL_WEEKS, GATE_TAU
        p_for_signal = p_cal_ho

    signal = apply_gate(p_for_signal, ra_ho.values, tau=tau)
    states = hysteresis(signal, enter=enter, exit_=exit_, dwell=dwell)
    states_s = pd.Series(states, index=holdout_index, name="state")

    # Build returns from raw prices on the hold-out window.
    prices_ho = d.X.loc[holdout_index]
    risk_on, defensive = build_strategy_returns(prices_ho)
    bt = run_backtest(states_s, risk_on, defensive, crises=CRISES)

    save_parquet(pd.DataFrame({
        "p_raw": p_raw_ho, "p_cal": p_cal_ho,
        "risk_appetite": ra_ho.values, "signal": signal,
        "state": states, "y": yho,
    }, index=holdout_index), "holdout_probabilities")
    save_parquet(pd.concat(
        [bt.equity_strategy.rename("equity_strategy"),
         bt.equity_bench.rename("equity_bench"),
         bt.weekly_strategy.rename("weekly_strategy"),
         bt.weekly_bench.rename("weekly_bench"),
         bt.states.rename("state_eff")], axis=1
    ), "holdout_equity")
    if not bt.crisis_metrics.empty:
        save_parquet(bt.crisis_metrics, "holdout_crisis")

    summary = {
        "seed": SEED,
        "n_train": int(train_mask.sum()),
        "n_holdout": int(ho_idx.size),
        "n_train_pos": int(ytr.sum()),
        "n_holdout_pos": int(yho.sum()),
        "holdout_start": str(holdout_index[0].date()),
        "holdout_end": str(holdout_index[-1].date()),
        "auc_raw_holdout": float(roc_auc_score(yho, p_raw_ho)) if len(np.unique(yho)) == 2 else None,
        "pr_auc_raw_holdout": float(average_precision_score(yho, p_raw_ho)) if len(np.unique(yho)) == 2 else None,
        "backtest_metrics": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in bt.metrics.items()},
        "thresholds": {
            "enter": float(enter), "exit": float(exit_),
            "dwell": int(dwell), "gate_tau": float(tau),
        },
    }
    save_json(summary, "summary")
    _log(f"holdout AUC = {summary['auc_raw_holdout']}  PR-AUC = {summary['pr_auc_raw_holdout']}")
    _log(f"strategy Sharpe = {bt.metrics.get('sharpe'):.3f}  "
         f"vs bench Sharpe = {bt.metrics.get('bench_sharpe'):.3f}")
    _log(f"strategy max DD = {bt.metrics.get('max_drawdown'):.3f}  "
         f"vs bench max DD = {bt.metrics.get('bench_max_drawdown'):.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["data", "cv", "backtest", "all"], default="all",
    )
    args = parser.parse_args(argv)

    if args.stage in ("data", "all"):
        _log("=== stage: data ===")
        stage_data()
    if args.stage in ("cv", "all"):
        _log("=== stage: cv ===")
        stage_cv()
    if args.stage in ("backtest", "all"):
        _log("=== stage: backtest ===")
        stage_backtest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
