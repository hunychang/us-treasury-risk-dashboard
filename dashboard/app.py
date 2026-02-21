"""Streamlit main application — U.S. Treasury & Credit Risk-Factor Dashboard.

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
from data.shock_provider import ShockProvider
from models.rolling_cov import RollingCovarianceModel
from models.ewma import EWMAModel
from models.var_model import VARModel
from models.base_model import RiskModel
from models.shock_conditioned import ShockConditionedModel
from models.irf.storage import IRFStore
from optimizer.min_variance import MinVarianceOptimizer
from optimizer.cvar_optimizer import CVaROptimizer
from backtester.engine import BacktestEngine
from backtester.benchmarks import (
    equal_weight_benchmark,
    duration_weighted_benchmark,
    treasuries_only_benchmark,
)
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
from dashboard.components.assumptions import render_assumptions


# ------------------------------------------------------------------
# Cached data loading
# ------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Loading data from FRED...")
def _load_returns():
    """Fetch yields from FRED and compute duration-adjusted returns (cached 1 h)."""
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
        instrument_metadata=cfg.data.instrument_metadata,
    )
    return returns


# ------------------------------------------------------------------
# Cached backtest execution
# ------------------------------------------------------------------

@st.cache_data(show_spinner="Running backtest...")
def _run_cached_backtest(
    _returns: pd.DataFrame,
    model_names: tuple,
    long_only: bool,
    weight_sum: float,
    tc_bps: float,
    max_weight: float,
    rebalance_freq: str,
    oos_start_str: str,
    rolling_window: int,
    rolling_shrinkage: str,
    rolling_ann: int,
    ewma_lambda: float,
    ewma_ann: int,
    var_lags: int,
    var_horizon: int,
    var_resid_cov: bool,
    var_ann: int,
    objective: str = "minimum_variance",
    cvar_confidence: float = 0.95,
    shock_enabled: bool = False,
    shock_scale: float = 1.0,
    shock_horizon: int = 12,
):
    """Run backtest with Streamlit caching keyed on all configuration params."""
    oos_start = pd.Timestamp(oos_start_str)

    # Build baseline models
    models = []
    if "rolling_cov" in model_names:
        models.append(RollingCovarianceModel(
            window=rolling_window,
            shrinkage=rolling_shrinkage,
            annualization_factor=rolling_ann,
        ))
    if "ewma" in model_names:
        models.append(EWMAModel(
            lambda_=ewma_lambda,
            annualization_factor=ewma_ann,
        ))
    if "var1_cov" in model_names:
        models.append(VARModel(
            lags=var_lags,
            forecast_horizon=var_horizon,
            covariance_from_residuals=var_resid_cov,
            annualization_factor=var_ann,
        ))

    # Add shock-conditioned models if enabled
    if shock_enabled:
        irf_store = IRFStore()
        if irf_store.exists():
            irf_results = irf_store.load()
            cfg = load_config()
            shock_provider = ShockProvider(
                csv_path=cfg.shocks.csv_path,
                shock_column=cfg.shocks.shock_column,
            )
            shocks = shock_provider.load_as_series(
                start_date=cfg.data.start_date, end_date=cfg.data.end_date
            )
            shocks_daily = shock_provider.reindex_to_daily(
                shocks, _returns.index, fill_value=0.0
            )
            conditioned = []
            for model in models:
                conditioned.append(ShockConditionedModel(
                    baseline_model=model,
                    irf_results=irf_results,
                    shock_series=shocks_daily,
                    scale_factor=shock_scale,
                    response_horizon=shock_horizon,
                ))
            models.extend(conditioned)

    # Build optimizer
    if objective == "cvar":
        optimizer = CVaROptimizer(
            confidence_level=cvar_confidence,
            long_only=long_only,
            weight_sum=weight_sum,
            max_weight=max_weight,
            transaction_cost_bps=tc_bps,
        )
    else:
        optimizer = MinVarianceOptimizer(
            long_only=long_only,
            weight_sum=weight_sum,
            transaction_cost_bps=tc_bps,
            max_weight=max_weight,
        )

    engine = BacktestEngine(
        _returns, models, optimizer, rebalance_freq, oos_start,
    )
    return engine.run()


@st.cache_data(show_spinner=False)
def _get_benchmark(
    _returns: pd.DataFrame, oos_start_str: str, benchmark_name: str
):
    """Cached benchmark computation."""
    oos_start = pd.Timestamp(oos_start_str)
    if benchmark_name == "equal_weight":
        return equal_weight_benchmark(_returns, oos_start)
    elif benchmark_name == "duration_weighted":
        return duration_weighted_benchmark(_returns, oos_start)
    elif benchmark_name == "treasuries_only":
        return treasuries_only_benchmark(_returns, oos_start)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="U.S. Treasury & Credit Risk-Factor Dashboard",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("U.S. Treasury & Credit Risk-Factor Dashboard")
    st.caption(
        "Compare minimum-variance risk-factor portfolios built with different "
        "covariance estimators on U.S. Treasury yields, credit spreads, "
        "breakevens, and volatility."
    )

    # --- Load data --------------------------------------------------------
    cfg = load_config()
    returns = _load_returns()

    # --- Sidebar controls -------------------------------------------------
    sidebar_cfg = render_sidebar(cfg)

    # Determine enabled model names
    model_names_list = []
    if sidebar_cfg.get("rolling_cov_on", True):
        model_names_list.append("rolling_cov")
    if sidebar_cfg.get("ewma_on", True):
        model_names_list.append("ewma")
    if sidebar_cfg.get("var_on", True):
        model_names_list.append("var1_cov")

    if not model_names_list:
        st.warning("Please enable at least one risk model in the sidebar.")
        st.stop()

    # --- Assumptions panel (collapsible) ----------------------------------
    render_assumptions(cfg, sidebar_cfg)

    # --- Run backtest (cached) --------------------------------------------
    oos_start = pd.Timestamp(sidebar_cfg["oos_start"])
    tc_bps = sidebar_cfg.get("tc_bps", cfg.portfolio.transaction_cost_bps)

    results = _run_cached_backtest(
        _returns=returns,
        model_names=tuple(model_names_list),
        long_only=cfg.portfolio.long_only,
        weight_sum=cfg.portfolio.weight_sum,
        tc_bps=tc_bps,
        max_weight=sidebar_cfg["max_weight"],
        rebalance_freq=sidebar_cfg["rebalance_freq"],
        oos_start_str=str(oos_start),
        rolling_window=cfg.models.rolling_cov.window,
        rolling_shrinkage=cfg.models.rolling_cov.shrinkage,
        rolling_ann=cfg.models.rolling_cov.annualization_factor,
        ewma_lambda=cfg.models.ewma.lambda_,
        ewma_ann=cfg.models.ewma.annualization_factor,
        var_lags=cfg.models.var.lags,
        var_horizon=cfg.models.var.forecast_horizon,
        var_resid_cov=cfg.models.var.covariance_from_residuals,
        var_ann=cfg.models.var.annualization_factor,
        objective=sidebar_cfg.get("objective", "minimum_variance"),
        cvar_confidence=sidebar_cfg.get("cvar_confidence", 0.95),
        shock_enabled=sidebar_cfg.get("shock_enabled", False),
        shock_scale=sidebar_cfg.get("shock_scale", 1.0),
        shock_horizon=sidebar_cfg.get("shock_horizon", 12),
    )

    # --- Add benchmarks (cached) ------------------------------------------
    oos_str = str(oos_start)
    if sidebar_cfg["show_equal_weight"]:
        results["equal_weight"] = _get_benchmark(returns, oos_str, "equal_weight")
    if sidebar_cfg["show_duration_weighted"]:
        results["duration_weighted"] = _get_benchmark(returns, oos_str, "duration_weighted")
    if sidebar_cfg["show_treasuries_only"]:
        results["treasuries_only"] = _get_benchmark(returns, oos_str, "treasuries_only")

    # --- Data freshness indicator -----------------------------------------
    st.sidebar.divider()
    st.sidebar.caption(
        f"Data: {returns.index[0].date()} to {returns.index[-1].date()}  \n"
        f"OOS window: {oos_start.date()} to {returns.index[-1].date()}  \n"
        f"Models: {', '.join(model_names_list)}"
    )

    # === Dashboard layout =================================================

    # --- Shock & IRF panels (if enabled) ----------------------------------
    if sidebar_cfg.get("shock_enabled", False):
        from dashboard.components.shock_panel import render_shock_panel
        from dashboard.components.irf_panel import render_irf_panel

        shock_tab, irf_tab, perf_tab = st.tabs([
            "Shock Timeline", "Impulse Responses", "Portfolio Performance"
        ])

        with shock_tab:
            st.subheader("Romer & Romer Monetary Policy Shocks")
            try:
                shock_provider = ShockProvider(
                    csv_path=cfg.shocks.csv_path,
                    shock_column=cfg.shocks.shock_column,
                )
                shocks = shock_provider.load_as_series(
                    start_date=cfg.data.start_date, end_date=cfg.data.end_date
                )
                render_shock_panel(shocks, oos_start)
            except Exception as e:
                st.error(f"Could not load shock data: {e}")

        with irf_tab:
            st.subheader("Impulse Response Functions")
            try:
                irf_store = IRFStore()
                if irf_store.exists():
                    irf_results = irf_store.load()
                    render_irf_panel(irf_results)
                else:
                    st.info(
                        "No IRF results found. Run "
                        "`python run_irf_estimation.py` to estimate IRFs."
                    )
            except Exception as e:
                st.error(f"Could not load IRF results: {e}")

        with perf_tab:
            _render_performance_panels(results, model_names_list, returns, table=None)
    else:
        _render_performance_panels(results, model_names_list, returns, table=None)


def _render_performance_panels(results, model_names_list, returns, table=None):
    """Render the standard performance dashboard panels."""
    # Row 1: Cumulative performance
    st.subheader("Cumulative Performance")
    fig_cum = plot_cumulative_returns(results)
    st.plotly_chart(fig_cum, use_container_width=True)

    # Row 2: Metrics comparison table
    st.subheader("Risk Metrics Comparison")
    metrics_table = metrics_comparison_table(results)
    render_metrics_table(metrics_table)

    # Row 3: Weights + Drawdown side by side
    all_model_names = [n for n in results if n not in {"equal_weight", "duration_weighted", "treasuries_only"}]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Portfolio Weights Over Time")
        model_for_weights = st.selectbox(
            "Select model", all_model_names, key="weight_model"
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
    export_buttons(results, metrics_table)


if __name__ == "__main__":
    main()
