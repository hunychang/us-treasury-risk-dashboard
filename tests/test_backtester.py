"""Tests for the backtesting engine and benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.engine import BacktestEngine
from backtester.benchmarks import equal_weight_benchmark, sixty_forty_benchmark
from models.rolling_cov import RollingCovarianceModel
from optimizer.min_variance import MinVarianceOptimizer


def test_backtest_runs(sample_returns):
    model = RollingCovarianceModel(window=252)
    optimizer = MinVarianceOptimizer(long_only=True)
    engine = BacktestEngine(
        sample_returns,
        [model],
        optimizer,
        "monthly",
        sample_returns.index[300],
    )
    results = engine.run()
    assert "rolling_cov" in results
    res = results["rolling_cov"]
    assert len(res.portfolio_returns) > 0
    assert len(res.weights_history) > 0


def test_weights_sum_to_one(sample_returns):
    model = RollingCovarianceModel(window=252)
    optimizer = MinVarianceOptimizer(long_only=True)
    engine = BacktestEngine(
        sample_returns,
        [model],
        optimizer,
        "monthly",
        sample_returns.index[300],
    )
    results = engine.run()
    weights = results["rolling_cov"].weights_history
    sums = weights.sum(axis=1)
    np.testing.assert_array_almost_equal(sums, 1.0, decimal=4)


def test_equal_weight_benchmark(sample_returns):
    res = equal_weight_benchmark(sample_returns, sample_returns.index[300])
    assert res.model_name == "equal_weight"
    assert len(res.portfolio_returns) > 0
    # Weights should all be 0.25
    np.testing.assert_array_almost_equal(
        res.weights_history.iloc[0].values, [0.25] * 4
    )


def test_sixty_forty_benchmark(sample_returns):
    res = sixty_forty_benchmark(sample_returns, sample_returns.index[300])
    assert res.model_name == "60_40_proxy"
    assert len(res.portfolio_returns) > 0
    w = res.weights_history.iloc[0].values
    # DGS10 = 60%, DGS2 = 40%
    cols = list(sample_returns.columns)
    assert w[cols.index("DGS10")] == 0.60
    assert w[cols.index("DGS2")] == 0.40
