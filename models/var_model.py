"""Vector Autoregression (VAR) based covariance model.

Fits a VAR(p) model on trailing returns and extracts the residual
covariance matrix as the forward-looking risk estimate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from models.base_model import RiskModel


class VARModel(RiskModel):
    """VAR-based covariance estimator.

    Parameters
    ----------
    lags : Number of autoregressive lags (default 1).
    forecast_horizon : Steps ahead for forecast-error covariance
        (used only when ``covariance_from_residuals=False``).
    covariance_from_residuals : If ``True`` (default), use the residual
        covariance ``sigma_u``.  Otherwise use the forecast-error
        covariance at *forecast_horizon* steps.
    annualization_factor : Scaling factor for annualization.
    window : Trailing window length for fitting (default 504 = ~2 years).
    """

    def __init__(
        self,
        lags: int = 1,
        forecast_horizon: int = 1,
        covariance_from_residuals: bool = True,
        annualization_factor: int = 252,
        window: int = 504,
    ) -> None:
        self._lags = lags
        self._horizon = forecast_horizon
        self._use_residual_cov = covariance_from_residuals
        self._ann = annualization_factor
        self._window = window

    # ------------------------------------------------------------------
    # RiskModel interface
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "var1_cov"

    def min_history(self) -> int:
        return max(self._window, self._lags + 60)

    def estimate(
        self, returns: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> np.ndarray:
        mask = returns.index <= as_of_date
        data = returns.loc[mask].iloc[-self._window :]

        if len(data) < self._lags + 30:
            raise ValueError(
                f"VAR: insufficient data ({len(data)} rows) for "
                f"{self._lags} lag(s)"
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="No frequency information")
            model = VAR(data)
            fitted = model.fit(maxlags=self._lags, ic=None, verbose=False)

        if self._use_residual_cov:
            cov = np.array(fitted.sigma_u) * self._ann
        else:
            fevd = fitted.forecast_cov(steps=self._horizon)
            cov = np.array(fevd[-1]) * self._ann

        return cov
