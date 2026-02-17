"""Benchmark portfolio constructors.

Each function returns a :class:`BacktestResult` so that benchmarks can
be compared with optimized portfolios using the same metrics and charts.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from backtester.engine import BacktestResult


def equal_weight_benchmark(
    returns: pd.DataFrame,
    oos_start: pd.Timestamp,
) -> BacktestResult:
    """1/N equal-weight portfolio, rebalanced daily (constant weights).

    Parameters
    ----------
    returns : Full return history.
    oos_start : Start of the out-of-sample window.

    Returns
    -------
    BacktestResult for the equal-weight strategy.
    """
    oos = returns.loc[returns.index >= oos_start]
    n = len(returns.columns)
    w = np.ones(n) / n

    port_rets = oos.dot(w)
    port_rets.name = "equal_weight"

    weights_df = pd.DataFrame(
        np.tile(w, (len(oos), 1)),
        index=oos.index,
        columns=returns.columns,
    )
    turnover = pd.Series(0.0, index=oos.index, name="turnover")

    return BacktestResult(
        model_name="equal_weight",
        portfolio_returns=port_rets,
        weights_history=weights_df,
        turnover=turnover,
        rebalance_dates=list(oos.index),
    )


def sixty_forty_benchmark(
    returns: pd.DataFrame,
    oos_start: pd.Timestamp,
    long_asset: str = "DGS10",
    short_asset: str = "DGS2",
) -> BacktestResult:
    """60/40 duration proxy: 60 % long-tenor, 40 % short-tenor.

    Parameters
    ----------
    returns : Full return history.
    oos_start : Start of the out-of-sample window.
    long_asset : Column name for the long-duration instrument (default DGS10).
    short_asset : Column name for the short-duration instrument (default DGS2).

    Returns
    -------
    BacktestResult for the 60/40 strategy.
    """
    oos = returns.loc[returns.index >= oos_start]
    n = len(returns.columns)
    cols = list(returns.columns)

    w = np.zeros(n)
    if long_asset in cols:
        w[cols.index(long_asset)] = 0.60
    if short_asset in cols:
        w[cols.index(short_asset)] = 0.40

    # If either asset is missing, fall back to equal weight for safety
    if w.sum() == 0:
        w = np.ones(n) / n

    port_rets = oos.dot(w)
    port_rets.name = "60_40_proxy"

    weights_df = pd.DataFrame(
        np.tile(w, (len(oos), 1)),
        index=oos.index,
        columns=returns.columns,
    )
    turnover = pd.Series(0.0, index=oos.index, name="turnover")

    return BacktestResult(
        model_name="60_40_proxy",
        portfolio_returns=port_rets,
        weights_history=weights_df,
        turnover=turnover,
        rebalance_dates=list(oos.index),
    )
