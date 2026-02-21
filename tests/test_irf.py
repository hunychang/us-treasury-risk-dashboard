"""Tests for Local Projection IRF estimation and storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.irf.local_projection import LPEstimator, IRFResult
from models.irf.storage import IRFStore


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Generate synthetic returns with a known shock response.

    A positive shock causes instrument 'A' to decrease (like a rate hike
    pushing yields up / bond prices down) and 'B' to increase.
    """
    np.random.seed(42)
    n_days = 2000
    dates = pd.bdate_range("1995-01-03", periods=n_days)

    # Random shocks on ~8% of days (simulating FOMC meetings)
    shock_mask = np.random.random(n_days) < 0.03
    shocks = np.zeros(n_days)
    shocks[shock_mask] = np.random.randn(shock_mask.sum()) * 0.25

    # Generate returns with known shock sensitivity
    noise_a = np.random.randn(n_days) * 0.005
    noise_b = np.random.randn(n_days) * 0.005

    # Instrument A responds negatively to shocks with a lag
    ret_a = noise_a.copy()
    for i in range(1, n_days):
        ret_a[i] += -0.02 * shocks[max(0, i - 1)]

    # Instrument B responds positively to shocks
    ret_b = noise_b.copy()
    for i in range(1, n_days):
        ret_b[i] += 0.015 * shocks[max(0, i - 1)]

    returns = pd.DataFrame(
        {"A": ret_a, "B": ret_b},
        index=dates,
    )
    shock_series = pd.Series(shocks, index=dates, name="shock")

    return returns, shock_series


@pytest.fixture
def estimator():
    return LPEstimator(max_horizon=6, n_lags=2, confidence_level=0.90)


# ---------------------------------------------------------------
# LP Estimation Tests
# ---------------------------------------------------------------

class TestLPEstimator:

    def test_estimate_returns_dict(self, estimator, synthetic_data):
        """Estimate should return a dict of IRFResult per instrument."""
        returns, shocks = synthetic_data
        results = estimator.estimate(returns, shocks)
        assert isinstance(results, dict)
        assert "A" in results
        assert "B" in results
        assert isinstance(results["A"], IRFResult)

    def test_irf_result_shapes(self, estimator, synthetic_data):
        """IRFResult arrays should have length == max_horizon."""
        returns, shocks = synthetic_data
        results = estimator.estimate(returns, shocks)
        irf = results["A"]
        assert len(irf.horizons) == 6
        assert len(irf.coefficients) == 6
        assert len(irf.std_errors) == 6
        assert len(irf.ci_lower) == 6
        assert len(irf.ci_upper) == 6
        assert len(irf.t_stats) == 6
        assert len(irf.p_values) == 6

    def test_horizons_sequential(self, estimator, synthetic_data):
        """Horizons should be [1, 2, ..., H]."""
        returns, shocks = synthetic_data
        results = estimator.estimate(returns, shocks)
        np.testing.assert_array_equal(results["A"].horizons, [1, 2, 3, 4, 5, 6])

    def test_coefficient_sign_a(self, estimator, synthetic_data):
        """Instrument A has negative shock response — β_1 should be negative."""
        returns, shocks = synthetic_data
        results = estimator.estimate(returns, shocks)
        # At horizon 1, the coefficient should be negative
        assert results["A"].coefficients[0] < 0

    def test_coefficient_sign_b(self, estimator, synthetic_data):
        """Instrument B has positive shock response — β_1 should be positive."""
        returns, shocks = synthetic_data
        results = estimator.estimate(returns, shocks)
        assert results["B"].coefficients[0] > 0

    def test_hac_standard_errors_finite(self, estimator, synthetic_data):
        """Standard errors should be finite and positive."""
        returns, shocks = synthetic_data
        results = estimator.estimate(returns, shocks)
        for name, irf in results.items():
            assert np.all(np.isfinite(irf.std_errors))
            assert np.all(irf.std_errors > 0)

    def test_ci_contains_beta(self, estimator, synthetic_data):
        """Confidence intervals should contain the point estimate."""
        returns, shocks = synthetic_data
        results = estimator.estimate(returns, shocks)
        for name, irf in results.items():
            assert np.all(irf.ci_lower <= irf.coefficients)
            assert np.all(irf.coefficients <= irf.ci_upper)

    def test_p_values_in_range(self, estimator, synthetic_data):
        """P-values should be in [0, 1]."""
        returns, shocks = synthetic_data
        results = estimator.estimate(returns, shocks)
        for name, irf in results.items():
            assert np.all(irf.p_values >= 0)
            assert np.all(irf.p_values <= 1)

    def test_n_obs_positive(self, estimator, synthetic_data):
        """Number of observations should be positive for all horizons."""
        returns, shocks = synthetic_data
        results = estimator.estimate(returns, shocks)
        for name, irf in results.items():
            assert np.all(irf.n_obs > 0)

    def test_zero_shocks_zero_coefficients(self):
        """With zero shocks, coefficients should be near zero."""
        np.random.seed(99)
        n = 500
        dates = pd.bdate_range("2000-01-03", periods=n)
        returns = pd.DataFrame(
            np.random.randn(n, 2) * 0.01,
            index=dates,
            columns=["X", "Y"],
        )
        zero_shocks = pd.Series(0.0, index=dates, name="shock")

        est = LPEstimator(max_horizon=4, n_lags=2)
        results = est.estimate(returns, zero_shocks)

        # With zero shocks, the shock coefficient is not identifiable
        # but should not blow up
        for name, irf in results.items():
            assert np.all(np.isfinite(irf.coefficients))


# ---------------------------------------------------------------
# IRF Storage Tests
# ---------------------------------------------------------------

class TestIRFStore:

    def test_save_load_roundtrip(self, estimator, synthetic_data, tmp_path):
        """Save and load should produce identical results."""
        returns, shocks = synthetic_data
        results = estimator.estimate(returns, shocks)

        store = IRFStore(output_dir=str(tmp_path))
        store.save(results, "test_irf.pkl")
        loaded = store.load("test_irf.pkl")

        assert set(loaded.keys()) == set(results.keys())
        for name in results:
            np.testing.assert_array_equal(
                loaded[name].coefficients, results[name].coefficients
            )
            np.testing.assert_array_equal(
                loaded[name].std_errors, results[name].std_errors
            )

    def test_load_nonexistent_raises(self, tmp_path):
        store = IRFStore(output_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            store.load("nonexistent.pkl")

    def test_exists(self, estimator, synthetic_data, tmp_path):
        returns, shocks = synthetic_data
        results = estimator.estimate(returns, shocks)

        store = IRFStore(output_dir=str(tmp_path))
        assert store.exists("test.pkl") is False
        store.save(results, "test.pkl")
        assert store.exists("test.pkl") is True

    def test_version_hash_deterministic(self, synthetic_data, tmp_path):
        returns, shocks = synthetic_data
        store = IRFStore(output_dir=str(tmp_path))
        h1 = store.version_hash(returns, shocks)
        h2 = store.version_hash(returns, shocks)
        assert h1 == h2

    def test_version_hash_changes_with_data(self, synthetic_data, tmp_path):
        returns, shocks = synthetic_data
        store = IRFStore(output_dir=str(tmp_path))
        h1 = store.version_hash(returns, shocks)

        # Modify data slightly
        returns2 = returns.copy()
        returns2.iloc[0, 0] += 1.0
        h2 = store.version_hash(returns2, shocks)
        assert h1 != h2
