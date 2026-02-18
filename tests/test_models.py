"""Tests for the three risk models."""

from __future__ import annotations

import numpy as np
import pytest

from models.rolling_cov import RollingCovarianceModel
from models.ewma import EWMAModel
from models.var_model import VARModel


class TestRollingCovariance:

    def test_shape(self, sample_returns):
        model = RollingCovarianceModel(window=252)
        cov = model.estimate(sample_returns, sample_returns.index[500])
        assert cov.shape == (4, 4)

    def test_psd(self, sample_returns):
        model = RollingCovarianceModel(window=252)
        cov = model.estimate(sample_returns, sample_returns.index[500])
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)

    def test_symmetric(self, sample_returns):
        model = RollingCovarianceModel(window=252)
        cov = model.estimate(sample_returns, sample_returns.index[500])
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_insufficient_data_raises(self, sample_returns):
        model = RollingCovarianceModel(window=252)
        with pytest.raises(ValueError, match="need 252"):
            model.estimate(sample_returns, sample_returns.index[100])

    def test_min_history(self):
        model = RollingCovarianceModel(window=252)
        assert model.min_history() == 252

    def test_name(self):
        model = RollingCovarianceModel()
        assert model.name() == "rolling_cov"


class TestEWMA:

    def test_shape(self, sample_returns):
        model = EWMAModel(lambda_=0.94)
        cov = model.estimate(sample_returns, sample_returns.index[500])
        assert cov.shape == (4, 4)

    def test_psd(self, sample_returns):
        model = EWMAModel(lambda_=0.94)
        cov = model.estimate(sample_returns, sample_returns.index[500])
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)

    def test_symmetric(self, sample_returns):
        model = EWMAModel(lambda_=0.94)
        cov = model.estimate(sample_returns, sample_returns.index[500])
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_name(self):
        model = EWMAModel()
        assert model.name() == "ewma"


class TestVAR:

    def test_shape(self, sample_returns):
        model = VARModel(lags=1, window=504)
        cov = model.estimate(sample_returns, sample_returns.index[700])
        assert cov.shape == (4, 4)

    def test_psd(self, sample_returns):
        model = VARModel(lags=1, window=504)
        cov = model.estimate(sample_returns, sample_returns.index[700])
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)

    def test_name(self):
        model = VARModel()
        assert model.name() == "var1_cov"
