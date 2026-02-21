"""Shock timeline visualization for the Streamlit dashboard."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def plot_shock_timeline(
    shocks: pd.Series,
    oos_start: Optional[pd.Timestamp] = None,
) -> go.Figure:
    """Bar chart of shock magnitudes over time.

    Positive shocks (tightening) are red; negative (easing) are green.
    """
    nonzero = shocks[shocks != 0.0].dropna()
    if len(nonzero) == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No shocks in sample",
            margin=dict(l=50, r=20, t=40, b=50),
        )
        return fig

    colors = ["#d32f2f" if v > 0 else "#388e3c" for v in nonzero.values]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=nonzero.index,
            y=nonzero.values,
            marker_color=colors,
            name="Shock",
            hovertemplate="%{x|%Y-%m-%d}: %{y:+.3f}<extra></extra>",
        )
    )

    # Add OOS start line
    if oos_start is not None:
        fig.add_vline(
            x=oos_start,
            line_dash="dash",
            line_color="gray",
            annotation_text="OOS start",
        )

    fig.update_layout(
        yaxis_title="Shock Magnitude",
        xaxis_title="Date",
        hovermode="x",
        margin=dict(l=50, r=20, t=30, b=50),
    )
    return fig


def render_shock_panel(
    shocks: pd.Series,
    oos_start: Optional[pd.Timestamp] = None,
) -> None:
    """Render the shock timeline panel in Streamlit."""
    nonzero = shocks[shocks != 0.0].dropna()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Shocks", len(nonzero))
    with col2:
        n_tight = (nonzero > 0).sum()
        st.metric("Tightening", int(n_tight))
    with col3:
        n_ease = (nonzero < 0).sum()
        st.metric("Easing", int(n_ease))
    with col4:
        if len(nonzero) > 0:
            latest = nonzero.iloc[-1]
            latest_date = nonzero.index[-1].strftime("%Y-%m-%d")
            st.metric("Latest Shock", f"{latest:+.3f}", help=latest_date)
        else:
            st.metric("Latest Shock", "N/A")

    fig = plot_shock_timeline(shocks, oos_start)
    st.plotly_chart(fig, use_container_width=True)
