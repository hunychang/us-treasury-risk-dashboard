"""Streamlit main application — U.S. Treasury Risk Management Dashboard.

Run with:
    streamlit run dashboard/app.py
or:
    python run_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so absolute imports work
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import streamlit as st

from config.config_loader import load_config, ModelsConfig
from data.fred_provider import FREDProvider
from data.cache_manager import CachedProvider
from data.returns import compute_returns
from models import build_models
from models.base_model import RiskModel
from optimizer.min_variance import MinVarianceOptimizer
from backtester.engine import BacktestEngine
from backtester.benchmarks import equal_weight_benchmark, sixty_forty_benchmark
from metrics.performance import metrics_comparison_table
from dashboard.components.sidebar import render_sidebar
from dashboard.components.charts import (
    plot_cumulative_returns,
    plot_weights_over_time,
    plot_drawdowns,
    plot_turnover,
    plot_rolling_vol,
)
from dashboard.components.tables import render_metrics_table
from dashboard.components.export import export_buttons


# ------------------------------------------------------------------
# Cached data loading
# ------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Loading Treasury yield data...")
def _load_data():
    """Load config, fetch yields, compute returns (cached 1 h)."""
    cfg = load_config()
    provider = CachedProvider(FREDProvider())
    yields = provider.get(
        cfg.data.instruments, cfg.data.start_date, cfg.data.end_date
    )
    returns = compute_returns(
        yields,
        method=cfg.data.return_type,
        interpolation=cfg.data.interpolation,
        missing_handling=cfg.data.missing_handling,
    )
    return cfg, returns


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _build_models_from_sidebar(
    sidebar_cfg: dict, models_cfg: ModelsConfig
) -> list[RiskModel]:
    """Override enabled flags according to sidebar toggles, then build."""
    # Create copies so we don't mutate the original config
    mc = models_cfg.model_copy(deep=True)
    mc.rolling_cov.enabled = sidebar_cfg.get("rolling_cov_on", True)
    mc.ewma.enabled = sidebar_cfg.get("ewma_on", True)
    mc.var.enabled = sidebar_cfg.get("var_on", True)
    return build_models(mc)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="U.S. Treasury Risk Dashboard",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("U.S. Treasury Risk Management Dashboard")
    st.caption(
        "Compare minimum-variance portfolios built with different "
        "covariance estimators on U.S. Treasury yields."
    )

    # --- Load data --------------------------------------------------------
    cfg, returns = _load_data()

    # --- Sidebar controls -------------------------------------------------
    sidebar_cfg = render_sidebar(cfg)
    enabled_models = _build_models_from_sidebar(sidebar_cfg, cfg.models)

    if not enabled_models:
        st.warning("Please enable at least one risk model in the sidebar.")
        st.stop()

    # --- Run backtest -----------------------------------------------------
    optimizer = MinVarianceOptimizer(
        long_only=cfg.portfolio.long_only,
        weight_sum=cfg.portfolio.weight_sum,
        transaction_cost_bps=cfg.portfolio.transaction_cost_bps,
    )

    oos_start = pd.Timestamp(sidebar_cfg["oos_start"])
    engine = BacktestEngine(
        returns,
        enabled_models,
        optimizer,
        sidebar_cfg["rebalance_freq"],
        oos_start,
    )

    with st.spinner("Running backtest..."):
        results = engine.run()

    # --- Add benchmarks ---------------------------------------------------
    if sidebar_cfg["show_equal_weight"]:
        results["equal_weight"] = equal_weight_benchmark(returns, oos_start)
    if sidebar_cfg["show_sixty_forty"]:
        results["60_40_proxy"] = sixty_forty_benchmark(returns, oos_start)

    # --- Data freshness indicator -----------------------------------------
    st.sidebar.divider()
    st.sidebar.caption(
        f"Data: {returns.index[0].date()} to {returns.index[-1].date()}  \n"
        f"OOS window: {oos_start.date()} to {returns.index[-1].date()}  \n"
        f"Models: {', '.join(m.name() for m in enabled_models)}"
    )

    # === Dashboard layout =================================================

    # Row 1: Cumulative performance
    st.subheader("Cumulative Performance")
    fig_cum = plot_cumulative_returns(results)
    st.plotly_chart(fig_cum, use_container_width=True)

    # Row 2: Metrics comparison table
    st.subheader("Risk Metrics Comparison")
    table = metrics_comparison_table(results)
    render_metrics_table(table)

    # Row 3: Weights + Drawdown side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Portfolio Weights Over Time")
        model_names = [m.name() for m in enabled_models]
        model_for_weights = st.selectbox(
            "Select model", model_names, key="weight_model"
        )
        if model_for_weights in results:
            fig_w = plot_weights_over_time(results[model_for_weights])
            st.plotly_chart(fig_w, use_container_width=True)

    with col2:
        st.subheader("Drawdown")
        fig_dd = plot_drawdowns(results)
        st.plotly_chart(fig_dd, use_container_width=True)

    # Row 4: Turnover + Rolling vol
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Turnover per Rebalance")
        fig_to = plot_turnover(results)
        st.plotly_chart(fig_to, use_container_width=True)

    with col4:
        st.subheader("Rolling Volatility (3-Year)")
        fig_vol = plot_rolling_vol(results)
        st.plotly_chart(fig_vol, use_container_width=True)

    # Row 5: Exports
    st.divider()
    export_buttons(results, table)


if __name__ == "__main__":
    main()
