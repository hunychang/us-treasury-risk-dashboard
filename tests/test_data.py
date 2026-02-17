"""Tests for return computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.returns import compute_returns


@pytest.fixture
def sample_yields() -> pd.DataFrame:
    """Synthetic yield levels."""
    np.random.seed(123)
    dates = pd.bdate_range("2020-01-01", periods=100)
    base = np.array([1.5, 2.0, 3.0, 4.0])
    # Random walk around base levels
    changes = np.random.randn(100, 4) * 0.05
    levels = base + np.cumsum(changes, axis=0)
    # Ensure positive (yields must be > 0 for log returns)
    levels = np.abs(levels) + 0.1
    return pd.DataFrame(levels, index=dates, columns=["DGS1", "DGS2", "DGS5", "DGS10"])


def test_log_returns_shape(sample_yields):
    rets = compute_returns(sample_yields, method="log")
    # One fewer row (first is NaN, then dropped)
    assert rets.shape[0] == sample_yields.shape[0] - 1
    assert rets.shape[1] == 4


def test_diff_returns_shape(sample_yields):
    rets = compute_returns(sample_yields, method="diff")
    assert rets.shape[0] == sample_yields.shape[0] - 1


def test_simple_returns_shape(sample_yields):
    rets = compute_returns(sample_yields, method="simple")
    assert rets.shape[0] == sample_yields.shape[0] - 1


def test_no_nans_after_drop(sample_yields):
    rets = compute_returns(sample_yields, method="log", missing_handling="drop")
    assert not rets.isna().any().any()


def test_invalid_method_raises(sample_yields):
    with pytest.raises(ValueError, match="Unknown return method"):
        compute_returns(sample_yields, method="foobar")
