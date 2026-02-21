"""Rolling-window sample covariance with optional Ledoit-Wolf shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.base_model import RiskModel


class RollingCovarianceModel(RiskModel):
    """Estimate the covariance matrix using a fixed-length trailing window.

    Parameters
    ----------
    window : Number of trading days in the rolling window (default 252).
    shrinkage : ``"none"`` for the raw sample covariance, or
        ``"ledoit_wolf"`` for the Ledoit-Wolf shrinkage estimator.
    annualization_factor : Multiply the daily covariance by this factor
        to annualize (default 252).
    """

    def __init__(
        self,
        window: int = 252,
        shrinkage: str = "none",
        annualization_factor: int = 252,
    ) -> None:
        self._window = window
        self._shrinkage = shrinkage
        self._ann = annualization_factor

    # ------------------------------------------------------------------
    # RiskModel interface
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "rolling_cov"

    def min_history(self) -> int:
        return self._window

    def estimate(
        self, returns: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> np.ndarray:
        # Select trailing window up to (and including) as_of_date
        end_pos = returns.index.searchsorted(as_of_date, side="right")
        start_pos = max(0, end_pos - self._window)
        trailing = returns.iloc[start_pos:end_pos]

        if len(trailing) < self._window:
            raise ValueError(
                f"RollingCov: need {self._window} observations, "
                f"got {len(trailing)}"
            )

        if self._shrinkage == "ledoit_wolf":
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf().fit(trailing.values)
            cov = lw.covariance_ * self._ann
        else:
            cov = trailing.cov().values * self._ann

        return cov
