"""Weekly P&L engine for the risk-on/risk-off switch with transaction costs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from sentinel_alpha.config import DEFENSIVE_WEIGHTS, TC_BPS_PER_LEG


WEEKS_PER_YEAR = 52


@dataclass
class BacktestResult:
    equity_strategy: pd.Series       # cumulative net return curve (gross of fees, net of TC)
    equity_bench: pd.Series          # MXWO buy-and-hold
    weekly_strategy: pd.Series       # weekly net returns
    weekly_bench: pd.Series          # weekly bench returns
    states: pd.Series                # 0/1 state in effect during the week
    metrics: dict[str, float]
    crisis_metrics: pd.DataFrame     # rows = named crises, columns = per-crisis stats


def _ann_metrics(weekly: pd.Series) -> dict[str, float]:
    if weekly.empty:
        return {}
    mu = float(weekly.mean()); sd = float(weekly.std(ddof=0))
    ann_ret = (1.0 + mu) ** WEEKS_PER_YEAR - 1.0
    ann_vol = sd * np.sqrt(WEEKS_PER_YEAR)
    sharpe = (mu / sd) * np.sqrt(WEEKS_PER_YEAR) if sd > 0 else 0.0
    downside = weekly[weekly < 0].std(ddof=0)
    sortino = (mu / downside) * np.sqrt(WEEKS_PER_YEAR) if downside > 0 else 0.0
    equity = (1.0 + weekly).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1.0)
    max_dd = float(dd.min())
    calmar = (ann_ret / abs(max_dd)) if max_dd < 0 else np.inf
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def _safe_loc(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(default, index=df.index)


def build_strategy_returns(
    prices: pd.DataFrame,
    risk_on_ticker: str = "MXUS",
    defensive_weights: dict[str, float] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """From the *raw* (untransformed) DataFrame, build weekly simple returns
    for (risk-on book, defensive book).

    The defensive book is rebalanced weekly to the target weights in
    `defensive_weights` (default: from config). Cash uses USGG3M / 52.
    """
    weights = defensive_weights or DEFENSIVE_WEIGHTS
    if risk_on_ticker not in prices.columns:
        raise KeyError(f"risk_on_ticker '{risk_on_ticker}' not in prices")
    risk_on_simple = prices[risk_on_ticker].pct_change().fillna(0.0)

    parts: list[pd.Series] = []
    for ticker, w in weights.items():
        if ticker == "CASH":
            if "USGG3M" not in prices.columns:
                raise KeyError("CASH leg needs USGG3M (3-month US T-bill yield)")
            # USGG3M is in percent; weekly return = yield/52 in fraction
            cash_w = prices["USGG3M"] / 100.0 / WEEKS_PER_YEAR
            parts.append(w * cash_w)
        else:
            if ticker not in prices.columns:
                raise KeyError(f"Defensive leg needs '{ticker}' in prices")
            parts.append(w * prices[ticker].pct_change().fillna(0.0))
    defensive_simple = sum(parts)
    return risk_on_simple, defensive_simple


def run_backtest(
    states: pd.Series,
    risk_on_simple: pd.Series,
    defensive_simple: pd.Series,
    tc_bps_per_leg: float = TC_BPS_PER_LEG,
    crises: dict[str, tuple[str, str]] | None = None,
) -> BacktestResult:
    """Execute the switch strategy and compute headline + per-crisis metrics.

    Convention: state[t] is the regime decided using p[t]. With a 1-week
    execution lag the strategy holds state[t-1]'s allocation during week t.
    """
    idx = states.index
    states_eff = states.shift(1).fillna(0).astype(int)  # 1-week execution lag

    risk_on_simple = risk_on_simple.reindex(idx).fillna(0.0)
    defensive_simple = defensive_simple.reindex(idx).fillna(0.0)

    # state 0 = 100% risk-on; state 1 = 100% defensive
    w_on = (1 - states_eff).astype(float)
    w_off = states_eff.astype(float)
    gross = w_on * risk_on_simple + w_off * defensive_simple

    mxwo_simple = risk_on_simple  # alias kept for clarity below

    # Transaction costs: when state flips, pay tc_bps_per_leg twice (sell + buy)
    # on the entire portfolio weight that moves.
    flips = (states_eff != states_eff.shift(1).fillna(0)).astype(int)
    # Each flip moves 100% of NAV; cost = 2 * tc_bps in bps -> /10000 in fraction.
    tc = flips * (2.0 * tc_bps_per_leg / 10000.0)
    net = gross - tc

    equity = (1.0 + net).cumprod()
    equity_bench = (1.0 + mxwo_simple).cumprod()

    metrics = _ann_metrics(net)
    metrics_bench = _ann_metrics(mxwo_simple)
    metrics.update({f"bench_{k}": v for k, v in metrics_bench.items()})
    metrics["turnover_per_year"] = float(flips.sum() * WEEKS_PER_YEAR / max(1, len(net)))
    metrics["n_flips"] = int(flips.sum())

    # Hit rate: fraction of crisis weeks during which the strategy was in risk-off.
    crisis_rows = []
    if crises is None:
        crises = {}
    for name, (start, end) in crises.items():
        mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
        if not mask.any():
            continue
        n_weeks = int(mask.sum())
        n_off = int((states_eff[mask] == 1).sum())
        cum_strat = float((1.0 + net[mask]).prod() - 1.0)
        cum_bench = float((1.0 + mxwo_simple[mask]).prod() - 1.0)
        max_dd_strat = float(((1.0 + net[mask]).cumprod() / (1.0 + net[mask]).cumprod().cummax() - 1.0).min())
        max_dd_bench = float(((1.0 + mxwo_simple[mask]).cumprod() / (1.0 + mxwo_simple[mask]).cumprod().cummax() - 1.0).min())
        crisis_rows.append({
            "crisis": name,
            "weeks": n_weeks,
            "off_weeks": n_off,
            "off_rate": n_off / n_weeks,
            "strategy_ret": cum_strat,
            "bench_ret": cum_bench,
            "strategy_max_dd": max_dd_strat,
            "bench_max_dd": max_dd_bench,
        })
    crisis_df = pd.DataFrame(crisis_rows)

    return BacktestResult(
        equity_strategy=equity,
        equity_bench=equity_bench,
        weekly_strategy=net,
        weekly_bench=mxwo_simple,
        states=states_eff,
        metrics=metrics,
        crisis_metrics=crisis_df,
    )
