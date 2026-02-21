"""Jordà (2005) Local Projection estimator for impulse response functions.

Estimates the response of each instrument to a monetary policy shock
at horizons h = 1, ..., H using the regression:

    Δx_{t+h} = α_h + β_h · s_t + Γ_h · Z_t + ε_{t+h}

where x is the cumulative change in the instrument, s is the shock,
and Z contains control lags.  Standard errors are Newey-West HAC
with bandwidth h+1 to account for the overlapping residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class IRFResult:
    """Stores the LP-estimated IRF for one instrument."""

    instrument: str
    horizons: np.ndarray       # [1, 2, ..., H]
    coefficients: np.ndarray   # β_h for each horizon
    std_errors: np.ndarray     # HAC standard errors
    ci_lower: np.ndarray       # confidence interval lower bound
    ci_upper: np.ndarray       # confidence interval upper bound
    t_stats: np.ndarray        # t-statistics
    p_values: np.ndarray       # p-values
    n_obs: np.ndarray          # number of observations per horizon


class LPEstimator:
    """Jordà (2005) Local Projection estimator.

    Parameters
    ----------
    max_horizon : Maximum impulse response horizon (default 24).
    n_lags : Number of control lags of returns in Z_t (default 4).
    confidence_level : Confidence level for CI bands (default 0.90).
    """

    def __init__(
        self,
        max_horizon: int = 24,
        n_lags: int = 4,
        confidence_level: float = 0.90,
    ) -> None:
        self._max_horizon = max_horizon
        self._n_lags = n_lags
        self._confidence = confidence_level

    def estimate(
        self,
        returns: pd.DataFrame,
        shocks: pd.Series,
        controls: Optional[pd.DataFrame] = None,
    ) -> Dict[str, IRFResult]:
        """Estimate IRFs for all instruments in returns.

        Parameters
        ----------
        returns : DataFrame of daily returns (DatetimeIndex × instruments).
        shocks : Series of shock magnitudes (sparse: NaN or 0 on non-event
            dates).  Must share the same DatetimeIndex as returns.
        controls : Optional additional control variables.

        Returns
        -------
        Dict mapping instrument name -> IRFResult.
        """
        # Align returns and shocks on the same index
        common_idx = returns.index.intersection(shocks.index)
        if len(common_idx) == 0:
            # If no overlap, reindex shocks to returns filling with 0
            shocks_aligned = shocks.reindex(returns.index, fill_value=0.0)
        else:
            shocks_aligned = shocks.reindex(returns.index, fill_value=0.0)

        # Build control matrix: lags of all return columns
        control_df = self._build_controls(returns, controls)

        results: Dict[str, IRFResult] = {}
        for col in returns.columns:
            results[col] = self._estimate_instrument(
                returns[col], shocks_aligned, control_df
            )

        return results

    def _estimate_instrument(
        self,
        y_series: pd.Series,
        shocks: pd.Series,
        controls: pd.DataFrame,
    ) -> IRFResult:
        """Run LP regressions for a single instrument across all horizons."""
        instrument = y_series.name
        H = self._max_horizon

        horizons = np.arange(1, H + 1)
        betas = np.zeros(H)
        ses = np.zeros(H)
        ci_lo = np.zeros(H)
        ci_hi = np.zeros(H)
        t_stats = np.zeros(H)
        p_vals = np.zeros(H)
        n_obs_arr = np.zeros(H, dtype=int)

        from scipy.stats import norm
        z_crit = norm.ppf(0.5 + self._confidence / 2)

        for i, h in enumerate(horizons):
            beta, se, t, p, n = self._estimate_single(
                y_series, shocks, controls, h
            )
            betas[i] = beta
            ses[i] = se
            ci_lo[i] = beta - z_crit * se
            ci_hi[i] = beta + z_crit * se
            t_stats[i] = t
            p_vals[i] = p
            n_obs_arr[i] = n

        return IRFResult(
            instrument=instrument,
            horizons=horizons,
            coefficients=betas,
            std_errors=ses,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            t_stats=t_stats,
            p_values=p_vals,
            n_obs=n_obs_arr,
        )

    def _estimate_single(
        self,
        y_series: pd.Series,
        shocks: pd.Series,
        controls: pd.DataFrame,
        horizon: int,
    ) -> Tuple[float, float, float, float, int]:
        """Run a single OLS regression for one horizon.

        Returns (beta, se, t_stat, p_value, n_obs).
        """
        # Dependent variable: cumulative change from t to t+h
        y_cumul = y_series.rolling(horizon).sum().shift(-horizon)

        # Build regression DataFrame
        reg_df = pd.DataFrame({
            "y": y_cumul,
            "shock": shocks,
        })

        # Add controls
        if controls is not None and len(controls.columns) > 0:
            reg_df = reg_df.join(controls)

        # Drop NaN rows
        reg_df = reg_df.dropna()

        if len(reg_df) < self._n_lags + 5:
            # Insufficient data for this horizon
            return 0.0, np.inf, 0.0, 1.0, 0

        y = reg_df["y"]
        X = reg_df.drop(columns=["y"])
        X = sm.add_constant(X)

        # OLS with Newey-West HAC standard errors
        model = sm.OLS(y, X)
        # Bandwidth = horizon + 1 for overlapping residuals
        result = model.fit(
            cov_type="HAC",
            cov_kwds={"maxlags": max(horizon, 1)},
        )

        # Extract the shock coefficient (index 1, after constant)
        shock_idx = list(X.columns).index("shock")
        beta = result.params.iloc[shock_idx]
        se = result.bse.iloc[shock_idx]
        t_stat = result.tvalues.iloc[shock_idx]
        p_value = result.pvalues.iloc[shock_idx]

        return beta, se, t_stat, p_value, len(reg_df)

    def _build_controls(
        self,
        returns: pd.DataFrame,
        extra_controls: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build control variable matrix from lagged returns."""
        parts = []
        for lag in range(1, self._n_lags + 1):
            lagged = returns.shift(lag)
            lagged.columns = [f"{c}_lag{lag}" for c in returns.columns]
            parts.append(lagged)

        if extra_controls is not None:
            parts.append(extra_controls)

        if parts:
            return pd.concat(parts, axis=1)
        return pd.DataFrame(index=returns.index)
