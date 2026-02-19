"""Model assumptions and methodology panel for the Streamlit dashboard."""

from __future__ import annotations

import streamlit as st

from config.config_loader import ProjectConfig


def render_assumptions(cfg: ProjectConfig, sidebar_cfg: dict) -> None:
    """Render a collapsible 'Model Assumptions & Methodology' section.

    Parameters
    ----------
    cfg : The loaded project configuration.
    sidebar_cfg : The current sidebar settings dict.
    """
    with st.expander("Model Assumptions & Methodology", expanded=False):
        # --- Portfolio Interpretation ---
        st.markdown("#### Portfolio Interpretation")
        st.info(
            "This dashboard constructs **risk-factor-mimicking portfolios** "
            "using FRED rate/spread/vol data. Returns are approximated using "
            "duration mapping (Treasuries) and spread-duration mapping "
            "(credit/breakevens). These are **not directly investable** -- "
            "actual trading would require Treasury futures/ETFs, credit index "
            "CDS, TIPS, and VIX futures. Treat performance as indicative "
            "factor-portfolio PnL.",
            icon="\u2139\ufe0f",
        )

        st.markdown("#### Data Source & Instruments")
        st.markdown(
            f"- **Source**: {cfg.data.source} (Federal Reserve Economic Data)\n"
            f"- **Instruments**: {len(cfg.data.instruments)} series\n"
            f"  - Treasury yields: DGS1, DGS2, DGS5, DGS10\n"
            f"  - Spreads: T10Y2Y, BAMLC0A0CM, BAMLH0A0HYM2, BAA10Y, T5YIE\n"
            f"  - Volatility index: VIXCLS\n"
            f"- **Date range**: {cfg.data.start_date} to "
            f"{'present' if cfg.data.end_date is None else cfg.data.end_date}"
        )

        st.markdown("#### Return Construction")
        if cfg.data.return_type == "duration_adj":
            st.markdown(
                "- **Treasury yields** (DGS1/2/5/10): Modified-duration "
                "approximation: `dP/P = -D_mod * dy`\n"
                "  - DGS1: D=1.0, DGS2: D=1.9, DGS5: D=4.5, DGS10: D=8.5\n"
                "- **Spreads** (T10Y2Y, credit OAS, breakevens): "
                "Spread-duration-scaled first difference: "
                "`-SD * diff * 0.01`\n"
                "  - T10Y2Y: SD=7.0, BAMLC0A0CM: SD=7.0, "
                "BAMLH0A0HYM2: SD=4.0, BAA10Y: SD=7.0, T5YIE: SD=5.0\n"
                "- **VIX**: Simple percent returns on the index level"
            )
        else:
            st.markdown(f"- **Method**: `{cfg.data.return_type}`")
        st.markdown(
            f"- **Missing data**: forward-fill (no look-ahead), "
            f"then `{cfg.data.missing_handling}`"
        )

        st.markdown("#### Portfolio Constraints")
        max_w = sidebar_cfg.get("max_weight", cfg.portfolio.max_weight)
        tc_bps = sidebar_cfg.get("tc_bps", cfg.portfolio.transaction_cost_bps)
        st.markdown(
            f"- **Objective**: Minimum variance (`min w'Sigma w`)\n"
            f"- **Long only**: {cfg.portfolio.long_only}\n"
            f"- **Weight sum**: {cfg.portfolio.weight_sum}\n"
            f"- **Max weight per asset**: {max_w:.0%}\n"
            f"- **Transaction costs**: {tc_bps} bps per unit turnover"
        )

        st.markdown("#### Covariance Estimation")
        st.markdown(
            f"- **Rolling window**: {cfg.models.rolling_cov.window} days "
            f"(shrinkage: {cfg.models.rolling_cov.shrinkage})\n"
            f"- **EWMA decay** (lambda): {cfg.models.ewma.lambda_} "
            f"(+ ridge regularization)\n"
            f"- **VAR(1)**: Residual covariance from {cfg.models.var.lags}-lag "
            f"vector autoregression (+ ridge regularization)"
        )

        st.markdown("#### Rebalancing")
        freq = sidebar_cfg.get("rebalance_freq", cfg.portfolio.rebalance_frequency)
        st.markdown(
            f"- **Frequency**: {freq}\n"
            f"- **Timing**: Weights at date *t* are computed using data up to "
            f"and including *t*, then applied to returns from *t+1* onward "
            f"(no look-ahead)\n"
            f"- **Weight drift**: Buy-and-hold between rebalances; weights "
            f"drift with realized returns\n"
            f"- **Transaction costs**: {tc_bps} bps per unit turnover"
        )
