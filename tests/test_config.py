"""Tests for the configuration loader."""

from __future__ import annotations

from datetime import date

from config.config_loader import load_config, ProjectConfig


def test_load_default_config():
    cfg = load_config()
    assert isinstance(cfg, ProjectConfig)
    assert cfg.data.instruments == ["DGS1", "DGS2", "DGS5", "DGS10"]
    assert cfg.portfolio.long_only is True
    assert cfg.models.ewma.lambda_ == 0.94
    assert cfg.models.rolling_cov.window == 252


def test_config_dates():
    cfg = load_config()
    assert cfg.data.start_date == date(1995, 1, 1)
    assert cfg.evaluation.oos_start == date(2005, 1, 1)


def test_config_portfolio_constraints():
    cfg = load_config()
    assert cfg.portfolio.weight_sum == 1.0
    assert cfg.portfolio.leverage is False
    assert cfg.portfolio.transaction_cost_bps == 0.0


def test_config_models_enabled():
    cfg = load_config()
    assert cfg.models.rolling_cov.enabled is True
    assert cfg.models.ewma.enabled is True
    assert cfg.models.var.enabled is True


def test_config_evaluation():
    cfg = load_config()
    assert "equal_weight" in cfg.evaluation.benchmark
    assert "60_40_proxy" in cfg.evaluation.benchmark
