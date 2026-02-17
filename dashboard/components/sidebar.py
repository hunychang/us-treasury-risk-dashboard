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
        "VAR", value=cfg.models.var.enabled
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

    st.sidebar.divider()
    st.sidebar.header("Benchmarks")
    show_equal_weight = st.sidebar.checkbox("Equal Weight (1/N)", value=True)
    show_sixty_forty = st.sidebar.checkbox("60/40 Proxy", value=True)

    return {
        "rolling_cov_on": rolling_cov_on,
        "ewma_on": ewma_on,
        "var_on": var_on,
        "rebalance_freq": rebalance_freq,
        "oos_start": oos_start,
        "show_equal_weight": show_equal_weight,
        "show_sixty_forty": show_sixty_forty,
    }
