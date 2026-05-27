"""Strategy layer: gate, state machine, backtest, in-fold tuner, continuous allocation."""
from sentinel_alpha.strategy.gate import apply_gate
from sentinel_alpha.strategy.state_machine import hysteresis
from sentinel_alpha.strategy.backtest import (
    run_backtest, build_strategy_returns, BacktestResult,
)
from sentinel_alpha.strategy.tuning import tune_thresholds, ThresholdGrid
from sentinel_alpha.strategy.continuous import (
    continuous_weights, tune_continuous, ContinuousGrid,
)

__all__ = ["apply_gate", "hysteresis", "run_backtest",
           "build_strategy_returns", "BacktestResult",
           "tune_thresholds", "ThresholdGrid",
           "continuous_weights", "tune_continuous", "ContinuousGrid"]
