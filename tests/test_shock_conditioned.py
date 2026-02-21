"""Tests for the shock-conditioned covariance model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.ewma import EWMAModel
from models.shock_conditioned import ShockConditionedModel
from models.irf.local_projection import IRFResult


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def sample_returns():
    """4-asset returns over 500 days."""
    np.random.seed(42)
    dates = pd.bdate_range("2000-01-03", periods=500)
    data = np.random.randn(500, 4) * 0.01
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C", "D"])


@pytest.fixture
def baseline_model():
    return EWMAModel(lambda_=0.94)


@pytest.fixture
def irf_results():
    """Fake IRF results for 4 instruments."""
    results = {}
    for name, coefs in [
        ("A", np.array([0.03, 0.02, 0.01, 0.005])),
        ("B", np.array([-0.02, -0.01, -0.005, -0.002])),
        ("C", np.array([0.01, 0.005, 0.002, 0.001])),
        ("D", np.array([0.0, 0.0, 0.0, 0.0])),  # no response
    ]:
        results[name] = IRFResult(
            instrument=name,
            horizons=np.arange(1, 5),
            coefficients=coefs,
            std_errors=np.ones(4) * 0.01,
            ci_lower=coefs - 0.02,
            ci_upper=coefs + 0.02,
            t_stats=coefs / 0.01,
            p_values=np.array([0.01, 0.05, 0.10, 0.50]),
            n_obs=np.full(4, 400),
        )
    return results


@pytest.fixture
def shock_series(sample_returns):
    """Shock series with a few non-zero events."""
    shocks = pd.Series(0.0, index=sample_returns.index, name="shock")
    # Place a positive shock near the middle
    shocks.iloc[250] = 0.50
    # Place a negative shock later
    shocks.iloc[400] = -0.30
    return shocks


@pytest.fixture
def conditioned_model(baseline_model, irf_results, shock_series):
    return ShockConditionedModel(
        baseline_model=baseline_model,
        irf_results=irf_results,
        shock_series=shock_series,
        scale_factor=1.0,
        response_horizon=4,
    )


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------

class TestShockConditionedModel:

    def test_name(self, conditioned_model):
        assert conditioned_model.name() == "ShockCond(ewma)"

    def test_min_history(self, conditioned_model):
        """min_history should delegate to baseline."""
        assert conditioned_model.min_history() == 60

    def test_output_shape(self, conditioned_model, sample_returns):
        as_of = sample_returns.index[300]
        cov = conditioned_model.estimate(sample_returns, as_of)
        assert cov.shape == (4, 4)

    def test_output_symmetric(self, conditioned_model, sample_returns):
        as_of = sample_returns.index[300]
        cov = conditioned_model.estimate(sample_returns, as_of)
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_output_psd(self, conditioned_model, sample_returns):
        """Output must be positive semi-definite."""
        as_of = sample_returns.index[300]
        cov = conditioned_model.estimate(sample_returns, as_of)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)

    def test_zero_shock_equals_baseline(
        self, baseline_model, irf_results, sample_returns
    ):
        """With no shocks, conditioned cov should equal baseline + ridge."""
        zero_shocks = pd.Series(0.0, index=sample_returns.index, name="shock")
        model = ShockConditionedModel(
            baseline_model=baseline_model,
            irf_results=irf_results,
            shock_series=zero_shocks,
            scale_factor=1.0,
            response_horizon=4,
        )
        as_of = sample_returns.index[300]
        cov_cond = model.estimate(sample_returns, as_of)
        cov_base = baseline_model.estimate(sample_returns, as_of)

        # The conditioned model adds 1e-6*I ridge on top of baseline
        # The baseline also has 1e-6*I ridge
        # So conditioned should be baseline + 1e-6*I (extra ridge)
        # But with zero shock, D^cond = D_baseline, so Σ^cond = Σ_baseline + 1e-6*I
        # vs Σ_baseline which already includes 1e-6*I
        # The difference should be exactly 1e-6*I
        diff = cov_cond - cov_base
        np.testing.assert_array_almost_equal(
            diff, np.eye(4) * 1e-6, decimal=10
        )

    def test_positive_shock_inflates_vol(
        self, baseline_model, irf_results, sample_returns
    ):
        """A positive shock should inflate diagonal entries for responding instruments."""
        # Create a shock just before the as_of_date
        shocks = pd.Series(0.0, index=sample_returns.index, name="shock")
        shocks.iloc[298] = 1.0  # Large shock

        model = ShockConditionedModel(
            baseline_model=baseline_model,
            irf_results=irf_results,
            shock_series=shocks,
            scale_factor=1.0,
            response_horizon=4,
        )
        as_of = sample_returns.index[300]
        cov_cond = model.estimate(sample_returns, as_of)
        cov_base = baseline_model.estimate(sample_returns, as_of)

        # Instrument A has large positive IRF → vol should increase
        assert cov_cond[0, 0] > cov_base[0, 0]
        # Instrument B has negative IRF (but squared, so still inflates)
        assert cov_cond[1, 1] > cov_base[1, 1]
        # Instrument D has zero IRF → vol should be unchanged (within ridge)
        np.testing.assert_almost_equal(
            cov_cond[3, 3], cov_base[3, 3] + 1e-6, decimal=8
        )

    def test_correlation_preservation(
        self, baseline_model, irf_results, sample_returns
    ):
        """Off-diagonal correlation structure should be preserved."""
        shocks = pd.Series(0.0, index=sample_returns.index, name="shock")
        shocks.iloc[298] = 0.5

        model = ShockConditionedModel(
            baseline_model=baseline_model,
            irf_results=irf_results,
            shock_series=shocks,
            scale_factor=1.0,
            response_horizon=4,
        )
        as_of = sample_returns.index[300]
        cov_cond = model.estimate(sample_returns, as_of)
        cov_base = baseline_model.estimate(sample_returns, as_of)

        # Extract correlation matrices
        def to_corr(c):
            d = np.sqrt(np.diag(c))
            d = np.maximum(d, 1e-10)
            d_inv = np.diag(1.0 / d)
            return d_inv @ c @ d_inv

        corr_cond = to_corr(cov_cond)
        corr_base = to_corr(cov_base)

        # Correlations should be very close (the small ridge adds a tiny diff)
        np.testing.assert_array_almost_equal(corr_cond, corr_base, decimal=2)

    def test_scale_factor_effect(
        self, baseline_model, irf_results, sample_returns
    ):
        """Higher scale factor should produce larger vol inflation."""
        shocks = pd.Series(0.0, index=sample_returns.index, name="shock")
        shocks.iloc[298] = 0.5

        as_of = sample_returns.index[300]

        model_low = ShockConditionedModel(
            baseline_model=baseline_model,
            irf_results=irf_results,
            shock_series=shocks,
            scale_factor=0.5,
            response_horizon=4,
        )
        model_high = ShockConditionedModel(
            baseline_model=baseline_model,
            irf_results=irf_results,
            shock_series=shocks,
            scale_factor=2.0,
            response_horizon=4,
        )

        cov_low = model_low.estimate(sample_returns, as_of)
        cov_high = model_high.estimate(sample_returns, as_of)

        # Higher scale → bigger diagonal
        assert cov_high[0, 0] > cov_low[0, 0]
