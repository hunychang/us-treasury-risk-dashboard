"""Plotly chart builders for the Streamlit dashboard."""

from __future__ import annotations

from typing import Dict

import plotly.graph_objects as go

from backtester.engine import BacktestResult
from metrics.performance import (
    cumulative_returns,
    drawdown_series,
    rolling_volatility,
)

_BENCHMARK_NAMES = {"equal_weight", "duration_weighted", "treasuries_only"}


def _get_line_style(name: str) -> dict:
    """Determine line style based on model name."""
    if name in _BENCHMARK_NAMES:
        return dict(dash="dash", width=1.5)
    elif name.startswith("ShockCond("):
        return dict(dash="dashdot", width=2.5)
    else:
        return dict(dash="solid", width=2)


def plot_cumulative_returns(results: Dict[str, BacktestResult]) -> go.Figure:
    """Overlay cumulative-return paths for all models and benchmarks."""
    fig = go.Figure()
    for name, res in results.items():
        cum = cumulative_returns(res.portfolio_returns)
        line_style = _get_line_style(name)
        fig.add_trace(
            go.Scatter(
                x=cum.index,
                y=cum.values,
                mode="lines",
                name=name,
                line=line_style,
            )
        )
    fig.update_layout(
        yaxis_title="Growth of $1",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=30, b=50),
    )
    return fig


def plot_weights_over_time(result: BacktestResult) -> go.Figure:
    """Stacked area chart of portfolio weights for one model."""
    weights = result.weights_history
    fig = go.Figure()
    for col in weights.columns:
        fig.add_trace(
            go.Scatter(
                x=weights.index,
                y=weights[col],
                mode="lines",
                stackgroup="one",
                name=col,
            )
        )
    fig.update_layout(
        yaxis_title="Weight",
        xaxis_title="Date",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
        margin=dict(l=50, r=20, t=30, b=50),
    )
    return fig


def plot_drawdowns(results: Dict[str, BacktestResult]) -> go.Figure:
    """Drawdown chart for all models — clean lines without overlapping fills."""
    fig = go.Figure()
    for name, res in results.items():
        dd = drawdown_series(res.portfolio_returns)
        line_style = _get_line_style(name)
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values,
                mode="lines",
                name=name,
                line=line_style,
            )
        )
    fig.update_layout(
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left"),
        margin=dict(l=50, r=20, t=30, b=50),
    )
    return fig


def plot_turnover(results: Dict[str, BacktestResult]) -> go.Figure:
    """Bar chart of turnover per rebalance for each model (excl. benchmarks)."""
    fig = go.Figure()
    for name, res in results.items():
        if name in _BENCHMARK_NAMES:
            continue
        to = res.turnover
        if len(to) > 0 and to.sum() > 0:
            fig.add_trace(
                go.Bar(x=to.index, y=to.values, name=name)
            )
    fig.update_layout(
        yaxis_title="Turnover",
        xaxis_title="Date",
        barmode="group",
        margin=dict(l=50, r=20, t=30, b=50),
    )
    return fig


def plot_rolling_vol(
    results: Dict[str, BacktestResult], window: int = 756
) -> go.Figure:
    """Rolling annualized volatility for all models."""
    fig = go.Figure()
    for name, res in results.items():
        rvol = rolling_volatility(res.portfolio_returns, window=window)
        fig.add_trace(
            go.Scatter(
                x=rvol.index,
                y=rvol.values,
                mode="lines",
                name=name,
            )
        )
    fig.update_layout(
        yaxis_title="Annualized Volatility",
        xaxis_title="Date",
        hovermode="x unified",
        margin=dict(l=50, r=20, t=30, b=50),
    )
    return fig
