"""Exponentially Weighted Moving Average (EWMA) covariance model.

Implements the RiskMetrics-style recursive covariance estimator:

    Sigma_t = lambda * Sigma_{t-1} + (1 - lambda) * r_{t-1} r_{t-1}^T

Supports incremental caching: when called repeatedly with the same
``returns`` DataFrame (same object) and advancing ``as_of_date``, only
processes new observations since the last call.
"""

from __future__ import annotations

from typing import Optional

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
        # Incremental cache state
        self._cached_cov: Optional[np.ndarray] = None
        self._cached_t: int = 0
        self._cached_data_id: Optional[int] = None

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
        # Use searchsorted for fast slicing
        end_pos = returns.index.searchsorted(as_of_date, side="right")
        data = returns.values[:end_pos]  # shape (T, n)
        T, n = data.shape

        if T < self._INIT_WINDOW:
            raise ValueError(
                f"EWMA: need at least {self._INIT_WINDOW} observations, "
                f"got {T}"
            )

        data_id = id(returns)
        lam = self._lambda

        # Resume from cached state if same DataFrame and valid position
        if (
            self._cached_cov is not None
            and self._cached_data_id == data_id
            and self._cached_t >= self._INIT_WINDOW
            and self._cached_t <= T
        ):
            cov = self._cached_cov.copy()
            start_t = self._cached_t
        else:
            # Cold start: seed with sample covariance of first INIT_WINDOW rows
            cov = np.cov(data[: self._INIT_WINDOW], rowvar=False)
            start_t = self._INIT_WINDOW

        # Process only NEW observations since last call
        for t in range(start_t, T):
            r = data[t].reshape(-1, 1)  # (n, 1)
            cov = lam * cov + (1 - lam) * (r @ r.T)

        # Save state for next call
        self._cached_cov = cov.copy()
        self._cached_t = T
        self._cached_data_id = data_id

        # Ridge regularization for numerical stability
        cov = cov + np.eye(n) * 1e-6

        return cov * self._ann

    def reset_cache(self) -> None:
        """Clear the incremental cache (e.g., between independent backtests)."""
        self._cached_cov = None
        self._cached_t = 0
        self._cached_data_id = None
