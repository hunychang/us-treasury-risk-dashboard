"""Walk-forward backtesting engine.

At each rebalance date the engine:
1. Asks the risk model to estimate a covariance matrix (using only data
   available up to that date).
2. Passes the covariance matrix to the optimizer to obtain new weights.
3. Records weights, computes daily portfolio returns between rebalances
   assuming buy-and-hold weight drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from models.base_model import RiskModel
from optimizer.min_variance import MinVarianceOptimizer
from utils.dates import get_rebalance_dates


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Stores the output of a single model's backtest run."""

    model_name: str
    portfolio_returns: pd.Series       # daily portfolio returns
    weights_history: pd.DataFrame      # (n_rebalances x n_assets)
    turnover: pd.Series                # turnover at each rebalance
    rebalance_dates: List[pd.Timestamp] = field(default_factory=list)


# ------------------------------------------------------------------
# Engine
# ------------------------------------------------------------------

class BacktestEngine:
    """Run walk-forward backtests for one or more risk models.

    Parameters
    ----------
    returns : Full return history (in-sample + out-of-sample).
    models : Risk models to evaluate.
    optimizer : Shared optimizer instance (same constraints for all models).
    rebalance_frequency : ``'daily'`` | ``'weekly'`` | ``'monthly'``.
    oos_start : First date of the out-of-sample evaluation period.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        models: List[RiskModel],
        optimizer: MinVarianceOptimizer,
        rebalance_frequency: str = "monthly",
        oos_start: Optional[pd.Timestamp] = None,
    ) -> None:
        self._returns = returns
        self._models = models
        self._optimizer = optimizer
        self._rebalance_freq = rebalance_frequency
        self._oos_start = oos_start or returns.index[0]
        self._asset_names = list(returns.columns)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, BacktestResult]:
        """Execute the backtest for every model.

        Returns
        -------
        Dictionary mapping model name -> BacktestResult.
        """
        oos_returns = self._returns.loc[self._returns.index >= self._oos_start]
        rebal_dates = get_rebalance_dates(
            oos_returns.index, self._rebalance_freq
        )

        results: Dict[str, BacktestResult] = {}
        for model in self._models:
            logger.info(f"Backtesting model: {model.name()}")
            results[model.name()] = self._run_single(model, rebal_dates)

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_single(
        self,
        model: RiskModel,
        rebal_dates: List[pd.Timestamp],
    ) -> BacktestResult:
        n_assets = len(self._asset_names)
        oos_returns = self._returns.loc[self._returns.index >= self._oos_start]

        prev_weights: Optional[np.ndarray] = None
        current_weights = np.ones(n_assets) / n_assets

        all_weights: List[np.ndarray] = []
        all_turnovers: List[float] = []
        valid_rebal_dates: List[pd.Timestamp] = []

        # Pre-allocate daily portfolio return series (filled in below)
        port_rets = pd.Series(
            0.0, index=oos_returns.index, dtype=float, name=model.name()
        )

        for i, rebal_date in enumerate(rebal_dates):
            # 1. Estimate covariance
            try:
                cov = model.estimate(self._returns, rebal_date)
                new_weights = self._optimizer.optimize(cov, prev_weights)
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    f"{model.name()} @ {rebal_date.date()}: {exc}  "
                    "-- keeping previous weights"
                )
                new_weights = current_weights.copy()

            # 2. Turnover
            if prev_weights is not None:
                turnover = float(np.sum(np.abs(new_weights - prev_weights)))
            else:
                turnover = float(
                    np.sum(np.abs(new_weights - np.ones(n_assets) / n_assets))
                )

            all_weights.append(new_weights)
            all_turnovers.append(turnover)
            valid_rebal_dates.append(rebal_date)

            # 3. Holding period
            if i + 1 < len(rebal_dates):
                next_date = rebal_dates[i + 1]
            else:
                next_date = oos_returns.index[-1]

            mask = (oos_returns.index > rebal_date) & (
                oos_returns.index <= next_date
            )
            period_rets = oos_returns.loc[mask]

            # 4. Daily returns with buy-and-hold weight drift
            w = new_weights.copy()
            for idx in period_rets.index:
                row = period_rets.loc[idx].values
                port_rets.loc[idx] = float(np.dot(w, row))
                # Drift weights
                w = w * (1.0 + row)
                w_sum = w.sum()
                if w_sum > 0:
                    w = w / w_sum

            prev_weights = new_weights
            current_weights = new_weights

        weights_df = pd.DataFrame(
            all_weights, index=valid_rebal_dates, columns=self._asset_names
        )
        turnover_series = pd.Series(
            all_turnovers, index=valid_rebal_dates, name="turnover"
        )

        return BacktestResult(
            model_name=model.name(),
            portfolio_returns=port_rets,
            weights_history=weights_df,
            turnover=turnover_series,
            rebalance_dates=valid_rebal_dates,
        )
