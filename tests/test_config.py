"""Tests for the configuration loader."""

from __future__ import annotations

from datetime import date

from config.config_loader import load_config, ProjectConfig


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, ProjectConfig)
    assert cfg.data.instruments == [
        "DGS1", "DGS2", "DGS5", "DGS10", "DGS30",
        "T10Y2Y", "T10Y3M",
        "BAMLC0A0CM", "BAMLH0A0HYM2", "BAA10Y",
        "T5YIE", "T10YIE", "VIXCLS",
    ]
    assert cfg.portfolio.long_only is True
    assert cfg.models.ewma.lambda_ == 0.94
    assert cfg.models.rolling_cov.window == 252


def test_config_dates():
    cfg = load_config()
    assert cfg.data.start_date == date(1997, 1, 1)
    assert cfg.evaluation.oos_start == date(2005, 1, 1)


def test_config_portfolio_constraints():
    cfg = load_config()
    assert cfg.portfolio.weight_sum == 1.0
    assert cfg.portfolio.leverage is False
    assert cfg.portfolio.transaction_cost_bps == 5.0


def test_config_models_enabled():
    cfg = load_config()
    assert cfg.models.rolling_cov.enabled is True
    assert cfg.models.ewma.enabled is True
    assert cfg.models.var.enabled is True


def test_config_evaluation():
    cfg = load_config()
    assert "equal_weight" in cfg.evaluation.benchmark
    assert "duration_weighted" in cfg.evaluation.benchmark


def test_config_instrument_metadata():
    cfg = load_config()
    assert "DGS10" in cfg.data.instrument_metadata
    assert cfg.data.instrument_metadata["DGS10"]["type"] == "treasury_yield"
    assert cfg.data.instrument_metadata["DGS10"]["duration"] == 8.5
    assert cfg.data.return_type == "duration_adj"


def test_config_max_weight():
    cfg = load_config()
    assert cfg.portfolio.max_weight == 0.40


def test_config_interpolation_ffill():
    cfg = load_config()
    assert cfg.data.interpolation == "ffill"


def test_config_shrinkage_ledoit_wolf():
    cfg = load_config()
    assert cfg.models.rolling_cov.shrinkage == "ledoit_wolf"


def test_config_var_annualization_factor():
    cfg = load_config()
    assert cfg.models.var.annualization_factor == 252


def test_config_spread_duration_metadata():
    cfg = load_config()
    meta = cfg.data.instrument_metadata
    assert meta["BAMLC0A0CM"]["spread_duration"] == 7.0
    assert meta["BAMLH0A0HYM2"]["spread_duration"] == 4.0
    assert meta["T5YIE"]["spread_duration"] == 5.0
    assert meta["T10Y2Y"]["spread_duration"] == 7.0


def test_config_new_instruments_metadata():
    cfg = load_config()
    meta = cfg.data.instrument_metadata
    assert meta["DGS30"]["type"] == "treasury_yield"
    assert meta["DGS30"]["duration"] == 19.5
    assert meta["T10Y3M"]["type"] == "spread"
    assert meta["T10Y3M"]["spread_duration"] == 8.0
    assert meta["T10YIE"]["type"] == "spread"
    assert meta["T10YIE"]["spread_duration"] == 8.5


def test_config_shock_defaults():
    cfg = load_config()
    assert cfg.shocks.enabled is False
    assert cfg.shocks.source == "csv"
    assert cfg.shocks.shock_column == "rr_shock"
    assert cfg.shocks.shock_window_months == 3


def test_config_irf_defaults():
    cfg = load_config()
    assert cfg.irf.enabled is False
    assert cfg.irf.max_horizon == 24
    assert cfg.irf.n_lags == 4
    assert cfg.irf.confidence_level == 0.90


def test_config_shock_conditioned_defaults():
    cfg = load_config()
    assert cfg.models.shock_conditioned.enabled is False
    assert cfg.models.shock_conditioned.scale_factor == 1.0
    assert cfg.models.shock_conditioned.response_horizon == 12


def test_config_cvar_defaults():
    cfg = load_config()
    assert cfg.portfolio.cvar_confidence == 0.95
    assert cfg.portfolio.cvar_n_scenarios == 5000
