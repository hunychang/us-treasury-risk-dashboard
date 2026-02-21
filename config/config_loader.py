"""Configuration loader with Pydantic validation.

Reads the YAML config file and produces a fully-validated ProjectConfig
object that every other module can depend on.
"""

from __future__ import annotations

import yaml
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-config models
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    source: str = "FRED"
    instruments: List[str] = ["DGS1", "DGS2", "DGS5", "DGS10"]
    frequency: str = "daily"
    start_date: date = date(1995, 1, 1)
    end_date: Optional[date] = None
    return_type: str = "duration_adj"
    interpolation: str = "linear"
    missing_handling: str = "drop"
    instrument_metadata: Dict[str, dict] = Field(default_factory=dict)


class PortfolioConfig(BaseModel):
    objective: str = "minimum_variance"  # minimum_variance | cvar
    long_only: bool = True
    weight_sum: float = 1.0
    leverage: bool = False
    max_weight: float = 0.40
    rebalance_frequency: str = "monthly"  # daily | weekly | monthly
    transaction_cost_bps: float = 0.0
    cvar_confidence: float = 0.95
    cvar_n_scenarios: int = 5000


class RollingCovConfig(BaseModel):
    enabled: bool = True
    window: int = 252
    shrinkage: str = "none"  # none | ledoit_wolf
    annualization_factor: int = 252


class EWMAConfig(BaseModel):
    enabled: bool = True
    lambda_: float = Field(default=0.94, alias="lambda")
    annualization_factor: int = 252

    model_config = {"populate_by_name": True}


class VARConfig(BaseModel):
    enabled: bool = True
    lags: int = 1
    forecast_horizon: int = 1
    covariance_from_residuals: bool = True
    annualization_factor: int = 252


class ShockConditionedConfig(BaseModel):
    enabled: bool = False
    scale_factor: float = 1.0
    response_horizon: int = 12


class ModelsConfig(BaseModel):
    rolling_cov: RollingCovConfig = RollingCovConfig()
    ewma: EWMAConfig = EWMAConfig()
    var: VARConfig = VARConfig()
    shock_conditioned: ShockConditionedConfig = ShockConditionedConfig()


class ShockConfig(BaseModel):
    enabled: bool = False
    source: str = "csv"  # csv | database
    csv_path: str = "data/shock_data/romer_romer_shocks.csv"
    shock_column: str = "rr_shock"
    cumulate: bool = False
    shock_window_months: int = 3


class IRFConfig(BaseModel):
    enabled: bool = False
    max_horizon: int = 24
    n_lags: int = 4
    confidence_level: float = 0.90
    output_dir: str = "output/irf"


class RiskMetricsConfig(BaseModel):
    volatility: bool = True
    sharpe_ratio: bool = True
    max_drawdown: bool = True
    turnover: bool = True
    var_95: bool = True


class EvaluationConfig(BaseModel):
    out_of_sample: bool = True
    oos_start: date = date(2005, 1, 1)
    benchmark: List[str] = ["equal_weight", "60_40_proxy"]
    performance_window: str = "rolling_36m"


class WebConfig(BaseModel):
    framework: str = "streamlit"
    host: str = "localhost"
    port: int = 8501
    interactive_weights: bool = True
    interactive_model_toggle: bool = True
    export_csv: bool = True
    export_png: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    save_weights: bool = True
    save_cov_matrices: bool = False
    save_forecasts: bool = True


class ReproducibilityConfig(BaseModel):
    random_seed: int = 42
    save_config_snapshot: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    """Aggregates every configuration section into one validated object."""

    name: str = "us-treasury-risk-management"
    version: str = "0.1.0"
    data: DataConfig = DataConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    models: ModelsConfig = ModelsConfig()
    shocks: ShockConfig = ShockConfig()
    irf: IRFConfig = IRFConfig()
    risk_metrics: RiskMetricsConfig = RiskMetricsConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    web: WebConfig = WebConfig()
    logging: LoggingConfig = LoggingConfig()
    reproducibility: ReproducibilityConfig = ReproducibilityConfig()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: Optional[str | Path] = None) -> ProjectConfig:
    """Load and validate the YAML configuration file.

    Parameters
    ----------
    path : Path to a YAML config file.  Falls back to
           ``config/default_config.yaml`` relative to this file.

    Returns
    -------
    ProjectConfig  -- fully validated configuration object.
    """
    if path is None:
        path = Path(__file__).parent / "default_config.yaml"
    else:
        path = Path(path)

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)

    # --- Extract top-level sections ------------------------------------------
    project_raw = raw.get("project", {})
    data_raw = raw.get("data", {})

    portfolio_raw = raw.get("portfolio", {})
    # Flatten nested 'constraints' into the portfolio dict
    constraints = portfolio_raw.pop("constraints", {})
    portfolio_raw.update(constraints)

    models_raw = raw.get("models", {})
    shocks_raw = raw.get("shocks", {})
    irf_raw = raw.get("irf", {})
    risk_metrics_raw = raw.get("risk_metrics", {})
    evaluation_raw = raw.get("evaluation", {})
    web_raw = raw.get("web", {})
    logging_raw = raw.get("logging", {})
    repro_raw = raw.get("reproducibility", {})

    # --- Build validated config -----------------------------------------------
    return ProjectConfig(
        name=project_raw.get("name", "us-treasury-risk-management"),
        version=project_raw.get("version", "0.1.0"),
        data=DataConfig(**data_raw),
        portfolio=PortfolioConfig(**portfolio_raw),
        models=ModelsConfig(
            rolling_cov=RollingCovConfig(**models_raw.get("rolling_cov", {})),
            ewma=EWMAConfig(**models_raw.get("ewma", {})),
            var=VARConfig(**models_raw.get("var", {})),
            shock_conditioned=ShockConditionedConfig(
                **models_raw.get("shock_conditioned", {})
            ),
        ),
        shocks=ShockConfig(**shocks_raw),
        irf=IRFConfig(**irf_raw),
        risk_metrics=RiskMetricsConfig(**risk_metrics_raw),
        evaluation=EvaluationConfig(**evaluation_raw),
        web=WebConfig(**web_raw),
        logging=LoggingConfig(**logging_raw),
        reproducibility=ReproducibilityConfig(**repro_raw),
    )
