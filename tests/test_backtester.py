"""Tests for the backtesting engine and benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.engine import BacktestEngine
from backtester.benchmarks import (
    equal_weight_benchmark,
    sixty_forty_benchmark,
    treasuries_only_benchmark,
)
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


def test_treasuries_only_benchmark_4_assets(sample_returns):
    """When all 4 columns are Treasury instruments, all get equal weight."""
    res = treasuries_only_benchmark(sample_returns, sample_returns.index[300])
    assert res.model_name == "treasuries_only"
    assert len(res.portfolio_returns) > 0
    # All 4 columns are DGS1/2/5/10 — so each should get 0.25
    np.testing.assert_array_almost_equal(
        res.weights_history.iloc[0].values, [0.25] * 4
    )


def test_treasuries_only_benchmark_10_assets():
    """When there are 10 columns (4 Treasury + 6 spreads), only Treasuries get weight."""
    np.random.seed(99)
    dates = pd.bdate_range("2000-01-03", periods=1000)
    cols = [
        "DGS1", "DGS2", "DGS5", "DGS10",
        "T10Y2Y", "BAMLC0A0CM", "BAMLH0A0HYM2", "BAA10Y", "T5YIE", "VIXCLS",
    ]
    data = np.random.randn(1000, 10) * 0.01
    returns = pd.DataFrame(data, index=dates, columns=cols)

    res = treasuries_only_benchmark(returns, returns.index[300])
    assert res.model_name == "treasuries_only"
    w = res.weights_history.iloc[0]
    # Treasury columns each get 0.25
    for tc in ["DGS1", "DGS2", "DGS5", "DGS10"]:
        assert abs(w[tc] - 0.25) < 1e-10
    # Spread columns get 0
    for sc in ["T10Y2Y", "BAMLC0A0CM", "BAMLH0A0HYM2", "BAA10Y", "T5YIE", "VIXCLS"]:
        assert w[sc] == 0.0
