"""Tests for performance and risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics.performance import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    var_95,
    cumulative_returns,
    drawdown_series,
)


def _make_returns(n: int = 500, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    dates = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(np.random.randn(n) * 0.01, index=dates)


def test_volatility_positive():
    rets = _make_returns()
    vol = annualized_volatility(rets)
    assert vol > 0


def test_max_drawdown_negative():
    rets = _make_returns()
    mdd = max_drawdown(rets)
    assert mdd <= 0


def test_var_95_negative():
    rets = _make_returns(1000)
    v = var_95(rets)
    assert v < 0


def test_cumulative_returns_starts_near_one():
    rets = _make_returns()
    cum = cumulative_returns(rets)
    # First value should be close to 1 (= 1 + first tiny return)
    assert abs(cum.iloc[0] - 1.0) < 0.1


def test_drawdown_series_non_positive():
    rets = _make_returns()
    dd = drawdown_series(rets)
    assert dd.max() <= 1e-10  # should be <= 0 everywhere


def test_sharpe_ratio_sign():
    # With positive drift, Sharpe should be positive
    np.random.seed(0)
    dates = pd.bdate_range("2010-01-01", periods=500)
    rets = pd.Series(np.random.randn(500) * 0.01 + 0.001, index=dates)
    sr = sharpe_ratio(rets)
    assert sr > 0


def test_annualized_return():
    np.random.seed(0)
    dates = pd.bdate_range("2010-01-01", periods=252)
    rets = pd.Series(np.ones(252) * 0.001, index=dates)
    ar = annualized_return(rets)
    np.testing.assert_almost_equal(ar, 0.252, decimal=3)
