"""Benchmark portfolio constructors.

Each function returns a :class:`BacktestResult` so that benchmarks can
be compared with optimized portfolios using the same metrics and charts.
"""

from __future__ import annotations

from typing import Dict, List, Optional

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


# Default approximate modified durations for DV01-parity calculation
_TREASURY_DURATIONS = {
    "DGS1": 1.0,
    "DGS2": 1.9,
    "DGS5": 4.5,
    "DGS10": 8.5,
}


def duration_weighted_benchmark(
    returns: pd.DataFrame,
    oos_start: pd.Timestamp,
    treasury_durations: Optional[Dict[str, float]] = None,
) -> BacktestResult:
    """DV01-parity benchmark: weights inversely proportional to duration.

    Each Treasury instrument contributes equal interest-rate risk (DV01).
    Weight formula: ``w_i = (1/D_i) / sum(1/D_j)`` for Treasury instruments.
    Non-Treasury instruments receive zero weight.

    Parameters
    ----------
    returns : Full return history.
    oos_start : Start of the out-of-sample window.
    treasury_durations : Mapping of column name -> modified duration.
        Defaults to approximate durations for DGS1/2/5/10.

    Returns
    -------
    BacktestResult for the DV01-parity strategy.
    """
    if treasury_durations is None:
        treasury_durations = _TREASURY_DURATIONS

    oos = returns.loc[returns.index >= oos_start]
    n = len(returns.columns)
    cols = list(returns.columns)

    # Identify which Treasury instruments are present
    present = {c: d for c, d in treasury_durations.items() if c in cols}

    if not present:
        # Fallback: equal weight everything
        w = np.ones(n) / n
    else:
        # Inverse-duration weights (DV01 parity)
        inv_durations = {c: 1.0 / d for c, d in present.items()}
        total_inv_dur = sum(inv_durations.values())
        w = np.zeros(n)
        for c, inv_d in inv_durations.items():
            w[cols.index(c)] = inv_d / total_inv_dur

    port_rets = oos.dot(w)
    port_rets.name = "duration_weighted"

    weights_df = pd.DataFrame(
        np.tile(w, (len(oos), 1)),
        index=oos.index,
        columns=returns.columns,
    )
    turnover = pd.Series(0.0, index=oos.index, name="turnover")

    return BacktestResult(
        model_name="duration_weighted",
        portfolio_returns=port_rets,
        weights_history=weights_df,
        turnover=turnover,
        rebalance_dates=list(oos.index),
    )


_TREASURY_COLS = ["DGS1", "DGS2", "DGS5", "DGS10"]


def treasuries_only_benchmark(
    returns: pd.DataFrame,
    oos_start: pd.Timestamp,
    treasury_assets: List[str] | None = None,
) -> BacktestResult:
    """Equal-weight benchmark using only the 4 Treasury yield instruments.

    Useful for answering *"does adding spreads to the portfolio help?"*

    Parameters
    ----------
    returns : Full return history (must contain the Treasury columns).
    oos_start : Start of the out-of-sample window.
    treasury_assets : Column names of Treasury instruments.  Defaults to
        ``["DGS1", "DGS2", "DGS5", "DGS10"]``.

    Returns
    -------
    BacktestResult for the Treasuries-only strategy.
    """
    if treasury_assets is None:
        treasury_assets = _TREASURY_COLS

    oos = returns.loc[returns.index >= oos_start]
    n = len(returns.columns)
    cols = list(returns.columns)

    # Build weight vector: equal weight across Treasury cols, zero elsewhere
    present = [c for c in treasury_assets if c in cols]
    if not present:
        # Fallback: equal weight everything
        w = np.ones(n) / n
    else:
        w = np.zeros(n)
        for c in present:
            w[cols.index(c)] = 1.0 / len(present)

    port_rets = oos.dot(w)
    port_rets.name = "treasuries_only"

    weights_df = pd.DataFrame(
        np.tile(w, (len(oos), 1)),
        index=oos.index,
        columns=returns.columns,
    )
    turnover = pd.Series(0.0, index=oos.index, name="turnover")

    return BacktestResult(
        model_name="treasuries_only",
        portfolio_returns=port_rets,
        weights_history=weights_df,
        turnover=turnover,
        rebalance_dates=list(oos.index),
    )
