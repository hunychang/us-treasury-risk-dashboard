"""IRF (Impulse Response Function) visualization for the Streamlit dashboard."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from models.irf.local_projection import IRFResult


def plot_irf_curves(
    irf_results: Dict[str, IRFResult],
    instruments: List[str],
) -> go.Figure:
    """Plot IRF curves with confidence bands for selected instruments.

    Parameters
    ----------
    irf_results : Dict mapping instrument name -> IRFResult.
    instruments : Which instruments to plot.
    """
    fig = go.Figure()

    for name in instruments:
        if name not in irf_results:
            continue
        irf = irf_results[name]

        # Confidence band (shaded)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([irf.horizons, irf.horizons[::-1]]),
                y=np.concatenate([irf.ci_upper, irf.ci_lower[::-1]]),
                fill="toself",
                fillcolor=f"rgba(100, 100, 200, 0.15)",
                line=dict(width=0),
                showlegend=False,
                name=f"{name} CI",
                hoverinfo="skip",
            )
        )

        # Point estimates
        fig.add_trace(
            go.Scatter(
                x=irf.horizons,
                y=irf.coefficients,
                mode="lines+markers",
                name=name,
                line=dict(width=2),
                marker=dict(size=5),
                hovertemplate=(
                    f"{name}<br>"
                    "h=%{x}<br>"
                    "β=%{y:.4f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)

    fig.update_layout(
        xaxis_title="Horizon (periods)",
        yaxis_title="Response (β)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
        margin=dict(l=50, r=20, t=30, b=50),
    )
    return fig


def render_irf_stats_table(
    irf_results: Dict[str, IRFResult],
    instruments: List[str],
) -> pd.DataFrame:
    """Build a summary table of IRF statistics."""
    rows = []
    for name in instruments:
        if name not in irf_results:
            continue
        irf = irf_results[name]
        for i, h in enumerate(irf.horizons):
            stars = ""
            if irf.p_values[i] < 0.01:
                stars = "***"
            elif irf.p_values[i] < 0.05:
                stars = "**"
            elif irf.p_values[i] < 0.10:
                stars = "*"
            rows.append({
                "Instrument": name,
                "Horizon": int(h),
                "Coefficient": f"{irf.coefficients[i]:.5f}",
                "Std Error": f"{irf.std_errors[i]:.5f}",
                "t-stat": f"{irf.t_stats[i]:.2f}",
                "p-value": f"{irf.p_values[i]:.4f}{stars}",
                "N": int(irf.n_obs[i]),
            })

    return pd.DataFrame(rows)


def render_irf_panel(irf_results: Dict[str, IRFResult]) -> None:
    """Render the IRF panel in Streamlit."""
    if not irf_results:
        st.info("No IRF results available. Run `python run_irf_estimation.py` first.")
        return

    all_instruments = list(irf_results.keys())

    # Instrument selector
    selected = st.multiselect(
        "Select instruments",
        all_instruments,
        default=all_instruments[:4],
        key="irf_instruments",
    )

    if not selected:
        st.warning("Select at least one instrument.")
        return

    # IRF curves
    fig = plot_irf_curves(irf_results, selected)
    st.plotly_chart(fig, use_container_width=True)

    # Stats table
    with st.expander("IRF Statistics Table"):
        table = render_irf_stats_table(irf_results, selected)
        st.dataframe(table, hide_index=True, use_container_width=True)
