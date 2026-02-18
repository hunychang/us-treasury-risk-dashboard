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


@pytest.fixture
def yields_with_negatives() -> pd.DataFrame:
    """Synthetic levels that include negative values (like inverted yield curve spreads)."""
    np.random.seed(456)
    dates = pd.bdate_range("2020-01-01", periods=100)
    # Simulate a spread that oscillates around zero (e.g., T10Y2Y during inversion)
    spread = np.cumsum(np.random.randn(100) * 0.05) - 0.5
    # Simulate a credit spread that stays positive
    credit = np.abs(np.cumsum(np.random.randn(100) * 0.02)) + 0.5
    return pd.DataFrame(
        {"T10Y2Y": spread, "BAMLC0A0CM": credit},
        index=dates,
    )


def test_diff_returns_with_negatives(yields_with_negatives):
    """diff returns should work fine even when levels go negative."""
    rets = compute_returns(yields_with_negatives, method="diff")
    assert rets.shape[0] == yields_with_negatives.shape[0] - 1
    assert not rets.isna().any().any()
    # First differences should match manual computation
    expected = yields_with_negatives.diff().dropna()
    pd.testing.assert_frame_equal(rets, expected)


def test_log_returns_fail_on_negatives(yields_with_negatives):
    """log returns produce NaN/inf when levels go negative — this is why we use diff."""
    rets = compute_returns(yields_with_negatives, method="log", missing_handling="ffill")
    # With negative values, log produces NaN — after dropping, we lose rows
    rets_dropped = compute_returns(yields_with_negatives, method="log", missing_handling="drop")
    # Should have fewer rows than diff because NaN rows from log(negative) are dropped
    assert rets_dropped.shape[0] < yields_with_negatives.shape[0] - 1
