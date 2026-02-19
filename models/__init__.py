from models.rolling_cov import RollingCovarianceModel
from models.ewma import EWMAModel
from models.var_model import VARModel
from models.base_model import RiskModel
from config.config_loader import ModelsConfig


def build_models(cfg: ModelsConfig) -> list[RiskModel]:
    """Factory: instantiate all enabled risk models from config."""
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


__all__ = [
    "RiskModel",
    "RollingCovarianceModel",
    "EWMAModel",
    "VARModel",
    "build_models",
]
