"""Streamlit sidebar controls."""

from __future__ import annotations

from datetime import date

import streamlit as st

from config.config_loader import ProjectConfig


def render_sidebar(cfg: ProjectConfig) -> dict:
    """Render sidebar widgets and return a dict of user-selected settings.

    The returned dict keys match the names used by ``app.py`` to override
    config defaults.
    """
    st.sidebar.header("Risk Models")

    rolling_cov_on = st.sidebar.checkbox(
        "Rolling Covariance", value=cfg.models.rolling_cov.enabled
    )
    ewma_on = st.sidebar.checkbox(
        "EWMA", value=cfg.models.ewma.enabled
    )
    var_on = st.sidebar.checkbox(
        "VAR(1) Residual Cov", value=cfg.models.var.enabled
    )

    st.sidebar.divider()
    st.sidebar.header("Backtest Settings")

    freq_options = ["daily", "weekly", "monthly"]
    rebalance_freq = st.sidebar.selectbox(
        "Rebalance Frequency",
        freq_options,
        index=freq_options.index(cfg.portfolio.rebalance_frequency),
    )

    oos_start = st.sidebar.date_input(
        "Out-of-Sample Start",
        value=cfg.evaluation.oos_start,
        min_value=date(2000, 1, 1),
        max_value=date(2024, 1, 1),
    )

    max_weight = st.sidebar.slider(
        "Max Weight per Asset",
        min_value=0.10,
        max_value=1.0,
        value=cfg.portfolio.max_weight,
        step=0.05,
        help="Maximum allocation to any single instrument.",
    )

    tc_bps = st.sidebar.slider(
        "Transaction Cost (bps)",
        min_value=0,
        max_value=50,
        value=int(cfg.portfolio.transaction_cost_bps),
        step=1,
        help="Round-trip transaction cost in basis points, applied as turnover penalty.",
    )

    st.sidebar.divider()
    st.sidebar.header("Optimizer")
    obj_options = ["minimum_variance", "cvar"]
    objective = st.sidebar.selectbox(
        "Objective",
        obj_options,
        index=0,
        help="Portfolio optimization objective function.",
    )
    cvar_confidence = 0.95
    if objective == "cvar":
        cvar_confidence = st.sidebar.slider(
            "CVaR Confidence",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Confidence level for CVaR (Expected Shortfall).",
        )

    st.sidebar.divider()
    st.sidebar.header("Shock Conditioning")
    shock_enabled = st.sidebar.checkbox(
        "Enable Shock Conditioning",
        value=False,
        help="Adjust covariance matrices using IRF-implied volatility scaling.",
    )
    shock_scale = 1.0
    shock_horizon = 12
    if shock_enabled:
        shock_scale = st.sidebar.slider(
            "Scale Factor",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Multiplier for IRF-based volatility adjustment.",
        )
        shock_horizon = st.sidebar.slider(
            "Response Horizon",
            min_value=1,
            max_value=24,
            value=12,
            step=1,
            help="Number of LP horizons to aggregate for vol scaling.",
        )

    st.sidebar.divider()
    st.sidebar.header("Benchmarks")
    show_equal_weight = st.sidebar.checkbox("Equal Weight (1/N)", value=True)
    show_duration_weighted = st.sidebar.checkbox("DV01 Parity", value=True)
    show_treasuries_only = st.sidebar.checkbox("Treasuries Only (1/4)", value=True)

    return {
        "rolling_cov_on": rolling_cov_on,
        "ewma_on": ewma_on,
        "var_on": var_on,
        "rebalance_freq": rebalance_freq,
        "oos_start": oos_start,
        "max_weight": max_weight,
        "tc_bps": tc_bps,
        "objective": objective,
        "cvar_confidence": cvar_confidence,
        "shock_enabled": shock_enabled,
        "shock_scale": shock_scale,
        "shock_horizon": shock_horizon,
        "show_equal_weight": show_equal_weight,
        "show_duration_weighted": show_duration_weighted,
        "show_treasuries_only": show_treasuries_only,
    }
