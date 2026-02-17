"""Styled metric tables for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_metrics_table(table: pd.DataFrame) -> None:
    """Display a formatted metrics comparison table.

    Parameters
    ----------
    table : DataFrame with models as rows and metrics as columns,
        as produced by :func:`metrics.performance.metrics_comparison_table`.
    """
    fmt_map = {
        "Annualized Return": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.3f}",
        "Max Drawdown": "{:.2%}",
        "Average Turnover": "{:.4f}",
        "VaR 95%": "{:.4f}",
    }

    styled = table.copy()
    for col, fmt in fmt_map.items():
        if col in styled.columns:
            styled[col] = styled[col].map(lambda x: fmt.format(x))

    st.dataframe(styled, use_container_width=True)
