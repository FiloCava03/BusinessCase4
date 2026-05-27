# Sentinel-alpha

A calibrated multi-view nowcaster for systematic risk-on / risk-off allocation, built on the
PoliMI Business Case #4 Bloomberg weekly dataset (2000-2021).

## Pipeline

```
raw weekly Bloomberg -> stationarity transforms -> engineered features
                     -> 6-detector heterogeneous stack
                     -> rank-quantile features
                     -> L2 logistic stacker
                     -> isotonic calibration  ->  p_t in [0,1]
                     -> asymmetry gate (risk-appetite < tau)
                     -> hysteresis state machine (dwell=2)
                     -> TC-aware backtest vs MSCI World
```

## Layout

- `sentinel_alpha/` package (sklearn-compatible interfaces).
- `tests/` unit tests (pytest).
- `notebooks/` 5 thin orchestrators that import the package.
- `artifacts/` produced outputs (parquet, png, json).

## Run

```bash
pip install -e .[dev]
pytest tests/
python -m sentinel_alpha.run --stage all
jupyter nbconvert --to notebook --execute notebooks/*.ipynb
```

## Seeds

All randomness is seeded through `sentinel_alpha.config.SEED` (default 42).
Two consecutive runs must produce identical numbers.
