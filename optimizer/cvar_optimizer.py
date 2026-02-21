"""CVaR (Conditional Value-at-Risk / Expected Shortfall) optimizer.

Solves the Rockafellar-Uryasev LP formulation:

    min  α + (1 / (S·(1−β))) Σ_s u_s
    s.t. u_s ≥ −w'r_s − α      ∀s
         u_s ≥ 0                ∀s
         Σ w_i = weight_sum
         0 ≤ w_i ≤ max_weight  (if long_only)

where α is the VaR threshold, u_s are auxiliary loss variables, r_s are
simulated return scenarios, β is the confidence level, and S is the
number of scenarios.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linprog


class CVaROptimizer:
    """Scenario-based CVaR minimization.

    Parameters
    ----------
    confidence_level : CVaR confidence (default 0.95 → 95% CVaR).
    n_scenarios : Number of Monte Carlo scenarios (default 5000).
    long_only : Enforce non-negative weights.
    weight_sum : Equality constraint on weight sum (default 1).
    max_weight : Per-asset upper bound.
    transaction_cost_bps : Turnover penalty in basis points.
    random_seed : Seed for scenario generation reproducibility.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        n_scenarios: int = 5000,
        long_only: bool = True,
        weight_sum: float = 1.0,
        max_weight: float = 1.0,
        transaction_cost_bps: float = 0.0,
        random_seed: int = 42,
    ) -> None:
        self._beta = confidence_level
        self._S = n_scenarios
        self._long_only = long_only
        self._weight_sum = weight_sum
        self._max_weight = max_weight
        self._tc = transaction_cost_bps / 10_000
        self._seed = random_seed

    def optimize(
        self,
        cov_matrix: np.ndarray,
        prev_weights: Optional[np.ndarray] = None,
        mean_returns: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve for CVaR-optimal portfolio weights.

        Parameters
        ----------
        cov_matrix : (n, n) annualized covariance matrix.
        prev_weights : Previous weights (for turnover penalty).
        mean_returns : (n,) expected return vector (default zero).

        Returns
        -------
        (n,) array of optimal weights.
        """
        n = cov_matrix.shape[0]
        S = self._S
        beta = self._beta

        # Default: zero expected returns (pure risk minimization)
        if mean_returns is None:
            mean_returns = np.zeros(n)

        # Generate return scenarios from multivariate normal
        rng = np.random.RandomState(self._seed)
        scenarios = rng.multivariate_normal(mean_returns, cov_matrix, size=S)
        # scenarios shape: (S, n)

        # ---------------------------------------------------------------
        # LP formulation:
        # Decision variables: [w (n), alpha (1), u (S)]
        # Total vars: n + 1 + S
        # ---------------------------------------------------------------

        n_vars = n + 1 + S

        # Objective: min 0·w + alpha + (1/(S(1-beta))) * sum(u_s)
        c = np.zeros(n_vars)
        c[n] = 1.0  # alpha coefficient
        c[n + 1:] = 1.0 / (S * (1 - beta))  # u_s coefficients

        # ---------------------------------------------------------------
        # Inequality constraints: u_s >= -w'r_s - alpha  →
        #   -w'r_s - alpha - u_s <= 0
        #   i.e., -r_s · w - alpha - u_s <= 0
        # ---------------------------------------------------------------
        A_ub = np.zeros((S, n_vars))
        b_ub = np.zeros(S)

        for s in range(S):
            A_ub[s, :n] = -scenarios[s]     # -r_s · w
            A_ub[s, n] = -1.0               # -alpha
            A_ub[s, n + 1 + s] = -1.0       # -u_s

        # ---------------------------------------------------------------
        # Equality constraint: sum(w) = weight_sum
        # ---------------------------------------------------------------
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n] = 1.0
        b_eq = np.array([self._weight_sum])

        # ---------------------------------------------------------------
        # Bounds
        # ---------------------------------------------------------------
        bounds = []
        for i in range(n):
            if self._long_only:
                bounds.append((0.0, self._max_weight))
            else:
                bounds.append((-self._max_weight, self._max_weight))
        # alpha is unbounded
        bounds.append((None, None))
        # u_s >= 0
        for _ in range(S):
            bounds.append((0.0, None))

        # ---------------------------------------------------------------
        # Solve LP
        # ---------------------------------------------------------------
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
        )

        if not result.success:
            raise RuntimeError(f"CVaR optimization failed: {result.message}")

        weights = result.x[:n]

        # Clip numerical noise and re-normalize
        if self._long_only:
            weights = np.maximum(weights, 0.0)
        weight_total = weights.sum()
        if weight_total > 0:
            weights = weights * (self._weight_sum / weight_total)
        else:
            weights = np.ones(n) * (self._weight_sum / n)

        return weights
