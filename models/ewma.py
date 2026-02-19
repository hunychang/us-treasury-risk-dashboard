"""Exponentially Weighted Moving Average (EWMA) covariance model.

Implements the RiskMetrics-style recursive covariance estimator:

    Sigma_t = lambda * Sigma_{t-1} + (1 - lambda) * r_{t-1} r_{t-1}^T
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.base_model import RiskModel


class EWMAModel(RiskModel):
    """EWMA covariance estimator (RiskMetrics style).

    Parameters
    ----------
    lambda_ : Decay factor (default 0.94).
    annualization_factor : Multiply the final covariance by this factor
        to annualize (default 252).
    """

    _INIT_WINDOW = 60  # observations used to seed the recursion

    def __init__(
        self,
        lambda_: float = 0.94,
        annualization_factor: int = 252,
    ) -> None:
        self._lambda = lambda_
        self._ann = annualization_factor

    # ------------------------------------------------------------------
    # RiskModel interface
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "ewma"

    def min_history(self) -> int:
        return self._INIT_WINDOW

    def estimate(
        self, returns: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> np.ndarray:
        mask = returns.index <= as_of_date
        data = returns.loc[mask].values  # shape (T, n)
        T, n = data.shape

        if T < self._INIT_WINDOW:
            raise ValueError(
                f"EWMA: need at least {self._INIT_WINDOW} observations, "
                f"got {T}"
            )

        # Seed with sample covariance of the first INIT_WINDOW rows
        cov = np.cov(data[: self._INIT_WINDOW], rowvar=False)

        # Recursive update
        lam = self._lambda
        for t in range(self._INIT_WINDOW, T):
            r = data[t].reshape(-1, 1)  # (n, 1)
            cov = lam * cov + (1 - lam) * (r @ r.T)

        # Ridge regularization for numerical stability
        cov = cov + np.eye(n) * 1e-6

        return cov * self._ann
