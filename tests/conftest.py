"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Synthetic daily returns for 4 correlated assets over 1 000 days."""
    np.random.seed(42)
    dates = pd.bdate_range("2000-01-03", periods=1000)
    raw = np.random.randn(1000, 4) * 0.01  # ~1 % daily vol

    # Introduce correlation via a lower-triangular matrix
    L = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.866, 0.0, 0.0],
            [0.3, 0.2, 0.933, 0.0],
            [0.4, 0.3, 0.2, 0.837],
        ]
    )
    data = raw @ L.T
    return pd.DataFrame(
        data, index=dates, columns=["DGS1", "DGS2", "DGS5", "DGS10"]
    )


@pytest.fixture
def sample_cov() -> np.ndarray:
    """A 4x4 positive-definite covariance matrix."""
    np.random.seed(42)
    A = np.random.randn(4, 4)
    return A @ A.T / 16 + np.eye(4) * 0.01
