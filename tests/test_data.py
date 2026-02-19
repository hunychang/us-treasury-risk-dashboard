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


# --- Duration-adjusted return tests ---


def test_duration_adj_returns_shape(sample_yields):
    """duration_adj should produce the same number of columns and rows-1."""
    rets = compute_returns(sample_yields, method="duration_adj")
    assert rets.shape[0] == sample_yields.shape[0] - 1
    assert rets.shape[1] == sample_yields.shape[1]


def test_duration_adj_treasury_sign():
    """When yield increases, bond price should decrease (negative return)."""
    dates = pd.bdate_range("2020-01-01", periods=3)
    yields = pd.DataFrame(
        {"DGS10": [4.00, 4.10, 4.05]},  # +10bp then -5bp
        index=dates,
    )
    rets = compute_returns(yields, method="duration_adj")
    # DGS10 duration ~8.5; +10bp = +0.10 pp; return = -8.5 * 0.10/100 = -0.0085
    assert rets["DGS10"].iloc[0] < 0  # yield went up -> price down
    assert rets["DGS10"].iloc[1] > 0  # yield went down -> price up
    np.testing.assert_almost_equal(rets["DGS10"].iloc[0], -8.5 * 0.001, decimal=6)


def test_duration_adj_spread():
    """Spread returns should use spread-duration scaling: -SD * diff * scale."""
    dates = pd.bdate_range("2020-01-01", periods=3)
    yields = pd.DataFrame(
        {"BAMLC0A0CM": [1.50, 1.55, 1.45]},
        index=dates,
    )
    rets = compute_returns(yields, method="duration_adj")
    # diff = +0.05 (spread widened), SD=7.0, scale=0.01
    # return = -7.0 * 0.05 * 0.01 = -0.0035 (widening = loss)
    np.testing.assert_almost_equal(rets["BAMLC0A0CM"].iloc[0], -0.0035, decimal=6)
    # diff = -0.10 (spread tightened) -> return = -7.0 * (-0.10) * 0.01 = +0.007
    np.testing.assert_almost_equal(rets["BAMLC0A0CM"].iloc[1], 0.007, decimal=6)


def test_duration_adj_vix():
    """VIX should use simple percent change."""
    dates = pd.bdate_range("2020-01-01", periods=3)
    yields = pd.DataFrame(
        {"VIXCLS": [20.0, 22.0, 21.0]},
        index=dates,
    )
    rets = compute_returns(yields, method="duration_adj")
    # simple pct_change: (22-20)/20 = 0.10
    np.testing.assert_almost_equal(rets["VIXCLS"].iloc[0], 0.10, decimal=6)


def test_duration_adj_mixed_instruments():
    """Full 10-instrument test with mixed types."""
    np.random.seed(789)
    dates = pd.bdate_range("2020-01-01", periods=100)
    cols = ["DGS1", "DGS2", "DGS5", "DGS10",
            "T10Y2Y", "BAMLC0A0CM", "BAMLH0A0HYM2", "BAA10Y", "T5YIE", "VIXCLS"]
    # Realistic yield levels
    base = np.array([1.5, 2.0, 3.0, 4.0, 0.5, 1.2, 3.5, 2.0, 2.5, 20.0])
    changes = np.random.randn(100, 10) * np.array([0.02] * 4 + [0.05] * 5 + [1.0])
    levels = base + np.cumsum(changes, axis=0)
    levels[:, 9] = np.abs(levels[:, 9]) + 10  # VIX must be positive
    levels[:, 0:4] = np.abs(levels[:, 0:4]) + 0.1  # yields must be positive
    df = pd.DataFrame(levels, index=dates, columns=cols)

    rets = compute_returns(df, method="duration_adj")
    assert rets.shape == (99, 10)
    assert not rets.isna().any().any()
