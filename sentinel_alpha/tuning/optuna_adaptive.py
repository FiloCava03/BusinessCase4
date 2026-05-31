"""Optuna TPE search over the adaptive-thresholding hyperparameters.

Search space (small on purpose -- we're tuning two real-valued knobs on the
*CV out-of-fold* predictions only):

    roll_win in [13, 104]    # 3 months to 2 years
    roll_q   in [0.80, 0.97] # top-3% to top-20%

Objective: by default the **net Calmar ratio** (return per unit of max
drawdown) of a TC-aware backtest run on the CV-period max-pool score, with
the rolling-window causal quantile threshold. This default reflects the
Risk Management framing of the system; pass ``objective="sharpe"`` or
``"sortino"`` to compare. The hold-out is never touched during the search.

The search prints best parameters and persists them to
`artifacts/optuna_adaptive.json`.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import optuna

from sentinel_alpha.config import ARTIFACTS_DIR, SEED, DEFAULT_TUNING_OBJECTIVE
from sentinel_alpha.strategy import run_backtest, build_strategy_returns


def _backtest_for_params(
    max_cv: np.ndarray, cv_index: pd.DatetimeIndex,
    risk_on_cv: pd.Series, defensive_cv: pd.Series,
    roll_win: int, roll_q: float,
    objective: str = DEFAULT_TUNING_OBJECTIVE,
) -> float:
    ser = pd.Series(max_cv, index=cv_index)
    thr = ser.rolling(roll_win, min_periods=max(8, min(roll_win, roll_win // 2))).quantile(roll_q).shift(1)
    sig = (ser > thr).astype(int).fillna(0).astype(float)
    states = pd.Series(sig.values, index=cv_index)
    res = run_backtest(states, risk_on_cv, defensive_cv)
    val = float(res.metrics.get(objective, -np.inf))
    n_off = int((states.values >= 0.5).sum())
    if n_off < 4:
        return -np.inf
    return val if np.isfinite(val) else -np.inf


def optimise_adaptive_thresholding(
    cv_proba_path: Path | None = None,
    prices_raw_path: Path | None = None,
    n_trials: int = 80,
    seed: int = SEED,
    objective: str = DEFAULT_TUNING_OBJECTIVE,
    verbose: bool = True,
) -> dict:
    """Run the Optuna study; return the best params + diagnostics."""
    cv_proba_path = cv_proba_path or (ARTIFACTS_DIR / "cv_probabilities.parquet")
    prices_raw_path = prices_raw_path or (ARTIFACTS_DIR / "prices_raw.parquet")

    cv_proba = pd.read_parquet(cv_proba_path)
    det_cols = [c for c in cv_proba.columns if c not in ("p_raw", "p_cal")]
    max_cv = cv_proba[det_cols].max(axis=1).values
    cv_index = cv_proba.index

    prices = pd.read_parquet(prices_raw_path)
    prices_cv = prices.loc[cv_index]
    risk_on_cv, defensive_cv = build_strategy_returns(prices_cv)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective_fn(trial: optuna.Trial) -> float:
        roll_win = trial.suggest_int("roll_win", 13, 104)
        roll_q = trial.suggest_float("roll_q", 0.80, 0.97)
        return _backtest_for_params(
            max_cv, cv_index, risk_on_cv, defensive_cv, roll_win, roll_q,
            objective=objective,
        )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=False)

    best = {
        "roll_win": int(study.best_params["roll_win"]),
        "roll_q":   float(study.best_params["roll_q"]),
        f"cv_{objective}": float(study.best_value),
        "objective": objective,
        "n_trials": n_trials,
    }
    out_path = ARTIFACTS_DIR / f"optuna_adaptive_{objective}.json"
    out_path.write_text(json.dumps(best, indent=2))
    if verbose:
        print(f"[optuna/{objective}] best: {best}")
        print(f"[optuna/{objective}] saved to {out_path}")
    return best
