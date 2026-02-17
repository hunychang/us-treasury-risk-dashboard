"""CSV / PNG export helpers for the Streamlit dashboard."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from backtester.engine import BacktestResult


def export_buttons(
    results: Dict[str, BacktestResult],
    metrics_table: pd.DataFrame,
) -> None:
    """Render download buttons for metrics, weights, and returns."""

    st.subheader("Export Data")
    col1, col2, col3 = st.columns(3)

    # 1. Metrics table
    with col1:
        csv_metrics = metrics_table.to_csv()
        st.download_button(
            label="Metrics (CSV)",
            data=csv_metrics,
            file_name="risk_metrics.csv",
            mime="text/csv",
        )

    # 2. Portfolio weights (one file per model)
    with col2:
        for name, res in results.items():
            csv_w = res.weights_history.to_csv()
            st.download_button(
                label=f"{name} Weights (CSV)",
                data=csv_w,
                file_name=f"weights_{name}.csv",
                mime="text/csv",
                key=f"dl_weights_{name}",
            )

    # 3. Portfolio returns
    with col3:
        rets_df = pd.DataFrame(
            {name: res.portfolio_returns for name, res in results.items()}
        )
        csv_rets = rets_df.to_csv()
        st.download_button(
            label="Returns (CSV)",
            data=csv_rets,
            file_name="portfolio_returns.csv",
            mime="text/csv",
        )
