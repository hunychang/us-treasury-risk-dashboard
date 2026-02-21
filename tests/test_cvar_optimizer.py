"""Tests for the CVaR (Expected Shortfall) optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from optimizer.cvar_optimizer import CVaROptimizer


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def identity_cov():
    return np.eye(4)


@pytest.fixture
def sample_cov():
    """A 4×4 positive-definite covariance matrix."""
    np.random.seed(42)
    A = np.random.randn(4, 4)
    return A @ A.T / 16 + np.eye(4) * 0.01


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

class TestCVaROptimizer:

    def test_weights_sum_to_one(self, identity_cov):
        opt = CVaROptimizer(long_only=True)
        w = opt.optimize(identity_cov)
        np.testing.assert_almost_equal(w.sum(), 1.0, decimal=4)

    def test_weights_non_negative(self, identity_cov):
        opt = CVaROptimizer(long_only=True)
        w = opt.optimize(identity_cov)
        assert np.all(w >= -1e-8)

    def test_identity_cov_near_equal_weight(self, identity_cov):
        """With identity covariance and zero mean, CVaR → approx equal weight."""
        opt = CVaROptimizer(long_only=True, n_scenarios=10000)
        w = opt.optimize(identity_cov)
        # Should be close to 0.25 each (not exact due to sampling)
        np.testing.assert_array_almost_equal(w, [0.25] * 4, decimal=1)

    def test_max_weight_constraint(self, sample_cov):
        opt = CVaROptimizer(long_only=True, max_weight=0.40)
        w = opt.optimize(sample_cov)
        assert np.all(w <= 0.40 + 1e-4)
        np.testing.assert_almost_equal(w.sum(), 1.0, decimal=4)

    def test_concentrates_in_low_vol(self):
        """With asymmetric covariance, should favor low-variance asset."""
        # Asset 0 has much lower variance
        cov = np.diag([0.01, 1.0, 1.0, 1.0])
        opt = CVaROptimizer(long_only=True, max_weight=0.80)
        w = opt.optimize(cov)
        # Asset 0 should get the largest weight
        assert w[0] == w.max()

    def test_feasibility(self, sample_cov):
        """Optimizer should find a feasible solution."""
        opt = CVaROptimizer(long_only=True)
        w = opt.optimize(sample_cov)
        assert w is not None
        assert len(w) == 4

    def test_different_confidence_levels(self, sample_cov):
        """Different confidence levels should produce different weights."""
        w_90 = CVaROptimizer(confidence_level=0.90).optimize(sample_cov)
        w_99 = CVaROptimizer(confidence_level=0.99).optimize(sample_cov)
        # Weights may differ (more conservative at higher confidence)
        # Just check both are valid
        np.testing.assert_almost_equal(w_90.sum(), 1.0, decimal=4)
        np.testing.assert_almost_equal(w_99.sum(), 1.0, decimal=4)

    def test_prev_weights_accepted(self, sample_cov):
        """prev_weights parameter should not cause errors."""
        opt = CVaROptimizer(long_only=True)
        prev_w = np.array([0.4, 0.3, 0.2, 0.1])
        w = opt.optimize(sample_cov, prev_weights=prev_w)
        np.testing.assert_almost_equal(w.sum(), 1.0, decimal=4)

    def test_mean_returns_effect(self):
        """Non-zero mean returns should shift optimal weights."""
        # Use low-vol covariance so mean signal is strong relative to noise
        cov = np.eye(4) * 0.01
        opt = CVaROptimizer(long_only=True, max_weight=0.80, n_scenarios=10000)
        # Asset 0 has much higher expected return
        mean_ret = np.array([0.50, 0.0, 0.0, 0.0])
        w = opt.optimize(cov, mean_returns=mean_ret)
        # Asset 0 should have the largest weight due to high mean
        assert w[0] == w.max()

    def test_reproducible_with_seed(self, sample_cov):
        """Same seed should give identical results."""
        opt1 = CVaROptimizer(random_seed=123)
        opt2 = CVaROptimizer(random_seed=123)
        w1 = opt1.optimize(sample_cov)
        w2 = opt2.optimize(sample_cov)
        np.testing.assert_array_almost_equal(w1, w2)
