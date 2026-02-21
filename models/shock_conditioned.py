"""Shock-conditioned covariance model.

Wraps any baseline ``RiskModel`` and adjusts the covariance matrix
using IRF-implied volatility scaling:

    Σ^cond = D^cond × Corr_baseline × D^cond

where D^cond inflates the volatility of each instrument proportional
to the magnitude of the recent monetary policy shock and the LP-estimated
impulse response coefficients.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from models.base_model import RiskModel
from models.irf.local_projection import IRFResult


class ShockConditionedModel(RiskModel):
    """IRF-adjusted covariance model.

    Parameters
    ----------
    baseline_model : Any RiskModel whose covariance to adjust.
    irf_results : IRF estimates from ``LPEstimator.estimate()``.
    shock_series : Series of shock magnitudes indexed by date.  Non-event
        dates should be 0 or NaN.
    scale_factor : Multiplier for the IRF vol adjustment (default 1.0).
    response_horizon : Number of LP horizons to aggregate (default 12).
    """

    def __init__(
        self,
        baseline_model: RiskModel,
        irf_results: Dict[str, IRFResult],
        shock_series: pd.Series,
        scale_factor: float = 1.0,
        response_horizon: int = 12,
    ) -> None:
        self._baseline = baseline_model
        self._irf = irf_results
        self._shocks = shock_series
        self._scale = scale_factor
        self._horizon = response_horizon

    def name(self) -> str:
        return f"ShockCond({self._baseline.name()})"

    def min_history(self) -> int:
        return self._baseline.min_history()

    def estimate(
        self,
        returns: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> np.ndarray:
        """Estimate shock-conditioned covariance matrix.

        1. Get baseline Σ from the wrapped model.
        2. Decompose into Corr and D (diagonal of standard deviations).
        3. Look up the latest shock magnitude before ``as_of_date``.
        4. Inflate each instrument's volatility using IRF coefficients.
        5. Reconstruct Σ^cond = D^cond × Corr × D^cond.
        """
        # 1. Baseline covariance
        cov_baseline = self._baseline.estimate(returns, as_of_date)
        n = cov_baseline.shape[0]
        columns = list(returns.columns)

        # 2. Decompose: Σ = D Corr D
        d_baseline = np.sqrt(np.diag(cov_baseline))
        # Guard against zero/negative diagonal
        d_baseline = np.maximum(d_baseline, 1e-10)
        D_inv = np.diag(1.0 / d_baseline)
        corr = D_inv @ cov_baseline @ D_inv

        # 3. Latest shock magnitude
        shock_mag = self._get_latest_shock(as_of_date)

        # 4. Adjust volatilities
        d_cond = d_baseline.copy()
        for i, col in enumerate(columns):
            if col in self._irf:
                irf = self._irf[col]
                # Sum of squared IRF coefficients up to response_horizon
                h_max = min(self._horizon, len(irf.coefficients))
                beta_sq_sum = np.sum(irf.coefficients[:h_max] ** 2)
                # Volatility scaling: inflate proportional to shock × IRF
                adjustment = 1.0 + self._scale * abs(shock_mag) * beta_sq_sum
                d_cond[i] = d_baseline[i] * np.sqrt(max(adjustment, 1.0))

        # 5. Reconstruct conditioned covariance
        D_cond = np.diag(d_cond)
        cov_cond = D_cond @ corr @ D_cond

        # Ridge regularization
        cov_cond = cov_cond + np.eye(n) * 1e-6

        return cov_cond

    def _get_latest_shock(self, as_of_date: pd.Timestamp) -> float:
        """Find the most recent non-zero shock on or before as_of_date."""
        available = self._shocks.loc[self._shocks.index <= as_of_date]
        if len(available) == 0:
            return 0.0
        # Find the most recent non-zero shock
        nonzero = available[available != 0.0]
        if len(nonzero) == 0:
            return 0.0
        return float(nonzero.iloc[-1])
