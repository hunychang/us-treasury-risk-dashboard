"""Optimizer factory for building portfolio optimizers from config."""

from __future__ import annotations

from typing import Union

from config.config_loader import PortfolioConfig
from optimizer.min_variance import MinVarianceOptimizer
from optimizer.cvar_optimizer import CVaROptimizer


def build_optimizer(
    cfg: PortfolioConfig,
) -> Union[MinVarianceOptimizer, CVaROptimizer]:
    """Build the appropriate optimizer from portfolio config.

    Parameters
    ----------
    cfg : Portfolio configuration with objective type and constraints.

    Returns
    -------
    MinVarianceOptimizer or CVaROptimizer.
    """
    if cfg.objective == "cvar":
        return CVaROptimizer(
            confidence_level=cfg.cvar_confidence,
            n_scenarios=cfg.cvar_n_scenarios,
            long_only=cfg.long_only,
            weight_sum=cfg.weight_sum,
            max_weight=cfg.max_weight,
            transaction_cost_bps=cfg.transaction_cost_bps,
        )
    else:
        # Default: minimum_variance
        return MinVarianceOptimizer(
            long_only=cfg.long_only,
            weight_sum=cfg.weight_sum,
            transaction_cost_bps=cfg.transaction_cost_bps,
            max_weight=cfg.max_weight,
        )


__all__ = [
    "MinVarianceOptimizer",
    "CVaROptimizer",
    "build_optimizer",
]
