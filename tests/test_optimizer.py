"""Tests for the minimum-variance optimizer."""

from __future__ import annotations

import numpy as np

from optimizer.min_variance import MinVarianceOptimizer


def test_weights_sum_to_one(sample_cov):
    opt = MinVarianceOptimizer(long_only=True)
    w = opt.optimize(sample_cov)
    np.testing.assert_almost_equal(w.sum(), 1.0, decimal=6)


def test_weights_non_negative(sample_cov):
    opt = MinVarianceOptimizer(long_only=True)
    w = opt.optimize(sample_cov)
    assert np.all(w >= -1e-10)


def test_identity_cov_gives_equal_weight():
    """With identity covariance, min-var = equal weight."""
    cov = np.eye(4)
    opt = MinVarianceOptimizer(long_only=True)
    w = opt.optimize(cov)
    np.testing.assert_array_almost_equal(w, [0.25] * 4, decimal=4)


def test_turnover_penalty():
    """Turnover penalty should bias towards previous weights."""
    cov = np.eye(4)
    prev_w = np.array([0.4, 0.3, 0.2, 0.1])
    opt_no_tc = MinVarianceOptimizer(long_only=True, transaction_cost_bps=0)
    opt_tc = MinVarianceOptimizer(long_only=True, transaction_cost_bps=100)

    w_no_tc = opt_no_tc.optimize(cov, prev_w)
    w_tc = opt_tc.optimize(cov, prev_w)

    # With turnover penalty, weights should be closer to prev_w
    dist_no_tc = np.sum(np.abs(w_no_tc - prev_w))
    dist_tc = np.sum(np.abs(w_tc - prev_w))
    assert dist_tc <= dist_no_tc + 1e-6
