from typing import Dict, List, Optional

import pandas as pd

from models.rolling_cov import RollingCovarianceModel
from models.ewma import EWMAModel
from models.var_model import VARModel
from models.base_model import RiskModel
from models.shock_conditioned import ShockConditionedModel
from models.irf.local_projection import IRFResult
from config.config_loader import ModelsConfig


def build_models(cfg: ModelsConfig) -> list[RiskModel]:
    """Factory: instantiate all enabled baseline risk models from config."""
    models = []
    if cfg.rolling_cov.enabled:
        models.append(RollingCovarianceModel(
            window=cfg.rolling_cov.window,
            shrinkage=cfg.rolling_cov.shrinkage,
            annualization_factor=cfg.rolling_cov.annualization_factor,
        ))
    if cfg.ewma.enabled:
        models.append(EWMAModel(
            lambda_=cfg.ewma.lambda_,
            annualization_factor=cfg.ewma.annualization_factor,
        ))
    if cfg.var.enabled:
        models.append(VARModel(
            lags=cfg.var.lags,
            forecast_horizon=cfg.var.forecast_horizon,
            covariance_from_residuals=cfg.var.covariance_from_residuals,
            annualization_factor=cfg.var.annualization_factor,
        ))
    return models


def build_conditioned_models(
    baseline_models: List[RiskModel],
    cfg: ModelsConfig,
    irf_results: Dict[str, IRFResult],
    shock_series: pd.Series,
) -> List[RiskModel]:
    """Wrap baseline models with shock conditioning if enabled."""
    if not cfg.shock_conditioned.enabled:
        return []

    conditioned = []
    for model in baseline_models:
        conditioned.append(ShockConditionedModel(
            baseline_model=model,
            irf_results=irf_results,
            shock_series=shock_series,
            scale_factor=cfg.shock_conditioned.scale_factor,
            response_horizon=cfg.shock_conditioned.response_horizon,
        ))
    return conditioned


__all__ = [
    "RiskModel",
    "RollingCovarianceModel",
    "EWMAModel",
    "VARModel",
    "ShockConditionedModel",
    "build_models",
    "build_conditioned_models",
]
