"""Strategy layer: gate, state machine, backtest, in-fold tuner, continuous allocation, V2 recovery override."""
from sentinel_alpha.strategy.gate import apply_gate
from sentinel_alpha.strategy.state_machine import hysteresis
from sentinel_alpha.strategy.backtest import (
    run_backtest, build_strategy_returns, BacktestResult,
)
from sentinel_alpha.strategy.tuning import tune_thresholds, ThresholdGrid
from sentinel_alpha.strategy.continuous import (
    continuous_weights, tune_continuous, ContinuousGrid,
)
from sentinel_alpha.strategy.recovery import (
    recovery_composite, hysteresis_with_override,
)

__all__ = ["apply_gate", "hysteresis", "run_backtest",
           "build_strategy_returns", "BacktestResult",
           "tune_thresholds", "ThresholdGrid",
           "continuous_weights", "tune_continuous", "ContinuousGrid",
           "recovery_composite", "hysteresis_with_override"]
