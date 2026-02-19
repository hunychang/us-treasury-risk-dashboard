"""Minimum-variance portfolio optimizer.

Solves:  min  w' Sigma w  [+ tc * ||w - w_prev||_1]
         s.t. sum(w) = weight_sum
              w >= 0          (if long_only)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize


class MinVarianceOptimizer:
    """Constrained minimum-variance portfolio solver.

    Parameters
    ----------
    long_only : Enforce non-negative weights.
    weight_sum : Equality constraint on the sum of weights (default 1).
    transaction_cost_bps : Turnover penalty in basis points.  0 = no
        penalty.  The penalty is added to the objective as
        ``tc * ||w - w_prev||_1``.
    """

    def __init__(
        self,
        long_only: bool = True,
        weight_sum: float = 1.0,
        transaction_cost_bps: float = 0.0,
        max_weight: float = 1.0,
    ) -> None:
        self._long_only = long_only
        self._weight_sum = weight_sum
        self._tc = transaction_cost_bps / 10_000  # bps -> decimal
        self._max_weight = max_weight

    def optimize(
        self,
        cov_matrix: np.ndarray,
        prev_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve for minimum-variance portfolio weights.

        Parameters
        ----------
        cov_matrix : (n, n) annualized covariance matrix.
        prev_weights : Previous period weights for the turnover penalty.
            ``None`` on the first rebalance.

        Returns
        -------
        (n,) array of optimal weights.
        """
        n = cov_matrix.shape[0]
        w0 = np.ones(n) / n  # equal-weight initial guess

        def objective(w: np.ndarray) -> float:
            port_var = w @ cov_matrix @ w
            if self._tc > 0 and prev_weights is not None:
                port_var += self._tc * np.sum(np.abs(w - prev_weights))
            return port_var

        def grad(w: np.ndarray) -> np.ndarray:
            g = 2.0 * cov_matrix @ w
            # Note: turnover penalty is non-smooth, so the gradient is
            # approximate when tc > 0.  SLSQP handles this adequately
            # for the small problems we solve here.
            return g

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - self._weight_sum}
        ]

        bounds = (
            [(0.0, self._max_weight)] * n
            if self._long_only
            else [(-self._max_weight, self._max_weight)] * n
        )

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            jac=grad,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-10},
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        # Clip tiny negatives from numerical noise and re-normalize
        weights = np.maximum(result.x, 0.0)
        weight_total = weights.sum()
        if weight_total > 0:
            weights = weights * (self._weight_sum / weight_total)
        else:
            # Fallback: equal weight
            weights = np.ones(n) * (self._weight_sum / n)

        return weights
