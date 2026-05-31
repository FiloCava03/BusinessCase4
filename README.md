# Sentinel-Alpha

A calibrated, multi-view Early Warning System (EWS) for systematic
**risk-on / risk-off equity allocation**, framed as a **Risk Management**
defensive overlay against the MSCI World book.

> Business case 4 — PoliMI Fintech Lab 2024/25 — Prof. Zenti.

## Business angle

We chose the **Risk Management** angle (defensive overlay), not Quant
Strategy. The goal is to **reduce the max drawdown** and **improve the
Calmar ratio** of an equity exposure during systemic-stress windows by
re-allocating into a defensive sleeve (50% inflation-linked, 25% gold,
25% cash) when the calibrated risk-off probability `p_t` exceeds in-fold
tuned thresholds.

This choice drives every downstream design decision:

- the default tuning objective is **Calmar**, not Sharpe;
- the in-fold threshold tuner asymmetrically penalises premature re-risking;
- the per-crisis stress table (Dotcom → COVID) is the **headline KPI**;
- the hysteresis state-machine uses `dwell ≥ 2` to suppress costly flips
  in calm regimes.

## Pipeline

```
raw weekly Bloomberg
  -> stationarity transforms (logret / bps_diff / level)
  -> engineered features (vol, corr, term spread, credit excess, VIX z, ...)
  -> per-class PCA on macro buckets
  -> 7-detector heterogeneous stack
       MVG | GMM | iForest | KPCA | COPOD | AE-ensemble | LOF
  -> empirical-quantile rank features (per detector)
  -> L2 logistic stacker
  -> isotonic calibration  ->  p_t in [0, 1]
  -> asymmetry gate (risk-appetite < tau)
  -> hysteresis state machine (default dwell = 2)
  -> TC-aware backtest vs MSCI World
```

## Methodology highlights

- **Walk-forward CV** with both **purge** (5 weeks before val) and
  **embargo** (2 weeks after val, excluded from the next training set).
  See `sentinel_alpha/cv/walkforward.py`.
- **Stratified calibration sub-fold** (20% of training, seeded) guarantees
  positives in the isotonic-fit set even in early walk-forward iterations.
- **AE ensemble** with disagreement (std across members) available as an
  uncertainty proxy (`AEEnsembleDetector.score_with_uncertainty`).
- **Adaptive thresholding** via Optuna TPE on the CV-period out-of-fold
  predictions; the hold-out is never touched during the search.
- **V2 recovery override** (`strategy/recovery.py`): a leading-recovery
  composite of credit-spread, VIX, and equity momentum can force an exit
  from the risk-off state on a strong recovery signal. Validated on a
  synthetic stress simulation (see `extras.ipynb` §12).

## Layout

| Path                          | Role                                       |
|-------------------------------|--------------------------------------------|
| `sentinel_alpha/`             | Python package, sklearn-compatible API     |
| `tests/`                      | Pytest suite (run-of-the-mill invariants)  |
| `main.ipynb`                  | Final pipeline & report                    |
| `extras.ipynb`                | Stress simulation, ablations, "what didn't work" |
| `artifacts/`                  | Produced outputs (parquet, json, png)      |
| `Dataset4_EWS.xlsx`           | Weekly Bloomberg input data                |

## Reproducibility

All randomness is seeded through `sentinel_alpha.config.SEED` (default 42).
Two consecutive runs must produce identical numbers.

```bash
pip install -e .[dev]
pytest tests/
python -m sentinel_alpha.run --stage all
jupyter nbconvert --to notebook --execute main.ipynb extras.ipynb
```
