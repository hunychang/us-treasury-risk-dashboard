"""Risk and performance metrics.

Every function operates on the standard :class:`BacktestResult` produced
by the backtesting engine, so that all models and benchmarks are
evaluated with identical bookkeeping.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from backtester.engine import BacktestResult


# ------------------------------------------------------------------
# Individual metrics
# ------------------------------------------------------------------

def annualized_return(returns: pd.Series, factor: int = 252) -> float:
    """Annualized mean return."""
    return float(returns.mean() * factor)


def annualized_volatility(returns: pd.Series, factor: int = 252) -> float:
    """Annualized standard deviation of returns."""
    return float(returns.std() * np.sqrt(factor))


def sharpe_ratio(
    returns: pd.Series, rf: float = 0.0, factor: int = 252
) -> float:
    """Annualized Sharpe ratio.

    Parameters
    ----------
    rf : Daily risk-free rate (default 0).
    """
    excess = returns - rf
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((excess.mean() * factor) / (std * np.sqrt(factor)))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (a negative number)."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def average_turnover(turnover_series: pd.Series) -> float:
    """Mean turnover per rebalance."""
    return float(turnover_series.mean())


def var_95(returns: pd.Series, method: str = "historical") -> float:
    """Value-at-Risk at the 95 % confidence level (daily).

    Returns a negative number — the 5th-percentile return.
    """
    clean = returns.dropna()
    if len(clean) == 0:
        return 0.0
    if method == "historical":
        return float(np.percentile(clean, 5))
    elif method == "parametric":
        from scipy.stats import norm
        return float(clean.mean() + norm.ppf(0.05) * clean.std())
    raise ValueError(f"Unknown VaR method: {method!r}")


# ------------------------------------------------------------------
# Time-series helpers (for charts)
# ------------------------------------------------------------------

def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Cumulative wealth path starting at 1."""
    return (1 + returns).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Continuous drawdown time series."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak


def rolling_volatility(
    returns: pd.Series, window: int = 756
) -> pd.Series:
    """Rolling annualized volatility (default window = 756 days ~ 3 years)."""
    return returns.rolling(window).std() * np.sqrt(252)


# ------------------------------------------------------------------
# Aggregation helpers
# ------------------------------------------------------------------

def compute_all_metrics(result: BacktestResult) -> Dict[str, float]:
    """Compute the full metric set for a single backtest result."""
    rets = result.portfolio_returns
    return {
        "Annualized Return": annualized_return(rets),
        "Annualized Volatility": annualized_volatility(rets),
        "Sharpe Ratio": sharpe_ratio(rets),
        "Max Drawdown": max_drawdown(rets),
        "Average Turnover": average_turnover(result.turnover),
        "VaR 95%": var_95(rets),
    }


def metrics_comparison_table(
    results: Dict[str, BacktestResult],
) -> pd.DataFrame:
    """Build a comparison DataFrame with one row per model / benchmark."""
    rows = {}
    for name, result in results.items():
        rows[name] = compute_all_metrics(result)
    return pd.DataFrame(rows).T
