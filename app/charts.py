"""Plotly chart builders for Page 3 (Operations Overview)."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

BAR_COLOR = "#2563eb"
GRID_COLOR = "rgba(128, 128, 128, 0.18)"


def _base_layout(fig: go.Figure, title: str, xaxis_title: str, yaxis_title: str) -> go.Figure:
    """Apply consistent single-axis, transparent-background styling suitable for light and dark themes."""
    fig.update_layout(
        title={"text": title, "font": {"size": 15}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin={"l": 16, "r": 36, "t": 44, "b": 36},
        height=300,
        xaxis={
            "title": xaxis_title,
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "zeroline": False,
        },
        yaxis={
            "title": yaxis_title,
            "showgrid": True,
            "gridcolor": GRID_COLOR,
            "zeroline": False,
        },
    )
    return fig


def chart_volume_by_intent(by_intent: pd.Series) -> go.Figure:
    """(a) 'What are people asking for?' horizontal bar chart of inquiry volume by intent."""
    fig = go.Figure()
    if by_intent.empty:
        return _base_layout(fig, "What are people asking for?", "Inquiries", "")

    # Reverse order so the most common intent appears at the top of the horizontal bar chart.
    s = by_intent.iloc[::-1]
    labels = [str(idx).replace("_", " ") for idx in s.index]
    values = [int(v) for v in s.values]
    max_val = max(values) if values else 1

    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=BAR_COLOR,
            width=0.55,
            text=[str(v) for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    _base_layout(fig, "What are people asking for?", "Inquiries", "")
    fig.update_xaxes(range=[0, max_val * 1.18])
    fig.update_yaxes(showgrid=False)
    return fig


def chart_response_time_histogram(
    response_hours: pd.Series,
    median_hours: Optional[float],
    p90_hours: Optional[float],
) -> go.Figure:
    """(b) 'How quickly do we first respond?' 4-hour bins up to 96 h (last bin pooled), with median and P90 lines."""
    fig = go.Figure()
    if response_hours.empty:
        return _base_layout(fig, "How quickly do we first respond?", "First-response time (hours)", "Inquiries")

    # 24 bins of 4 hours each covering [0, 96), plus a pooled 96+ h bin at [96, 100].
    clipped = response_hours.clip(lower=0.0, upper=98.0)
    bin_centers = [i * 4 + 2 for i in range(25)]
    counts = [0] * 25
    for val in clipped:
        idx = min(int(val // 4), 24)
        counts[idx] += 1

    tick_vals = [0, 12, 24, 36, 48, 60, 72, 84, 98]
    tick_text = ["0h", "12h", "24h", "36h", "48h", "60h", "72h", "84h", "96h+"]

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=counts,
            width=3.4,
            marker_color=BAR_COLOR,
            hovertemplate="Hours bin center: %{x}h<br>Inquiries: %{y}<extra></extra>",
        )
    )

    if median_hours is not None:
        med_x = min(float(median_hours), 98.0)
        fig.add_vline(
            x=med_x,
            line_dash="solid",
            line_width=2,
            line_color="#d97706",
            annotation_text=f"Median: {median_hours:.1f}h",
            annotation_position="top left",
        )
    if p90_hours is not None:
        p90_x = min(float(p90_hours), 98.0)
        fig.add_vline(
            x=p90_x,
            line_dash="dash",
            line_width=2,
            line_color="#dc2626",
            annotation_text=f"P90: {p90_hours:.1f}h",
            annotation_position="top right",
        )

    _base_layout(fig, "How quickly do we first respond?", "First-response time (4h bins, 96h+ pooled)", "Inquiries")
    fig.update_xaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_text, range=[0, 102], showgrid=False)
    return fig


def chart_weekly_demand(inquiries: pd.DataFrame) -> go.Figure:
    """(c) 'Is demand steady?' weekly arrival counts over the selected period."""
    fig = go.Figure()
    if inquiries.empty:
        return _base_layout(fig, "Is demand steady?", "Week starting", "Inquiries")

    weekly = (
        inquiries.set_index("received_at")
        .resample("W-MON", label="left", closed="left")
        .size()
        .rename("count")
    )
    x_vals = [d.strftime("%d %b") for d in weekly.index]
    y_vals = [int(v) for v in weekly.values]
    max_y = max(y_vals) if y_vals else 1

    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_vals,
            marker_color=BAR_COLOR,
            width=0.55,
            text=[str(v) for v in y_vals],
            textposition="outside",
            cliponaxis=False,
        )
    )
    _base_layout(fig, "Is demand steady?", "Week starting", "Inquiries received")
    fig.update_yaxes(range=[0, max_y * 1.2])
    fig.update_xaxes(showgrid=False)
    return fig


def chart_completeness_distribution(completeness_series: pd.Series) -> go.Figure:
    """(d) 'How complete are submissions on arrival?' histogram of completeness percentages."""
    fig = go.Figure()
    if completeness_series.empty:
        return _base_layout(
            fig, "How complete are submissions on arrival?", "Completeness (% of applicable fields)", "Inquiries"
        )

    pcts = (completeness_series * 100.0).clip(0.0, 100.0)
    fig.add_trace(
        go.Histogram(
            x=pcts,
            xbins={"start": 0, "end": 100, "size": 10},
            marker_color=BAR_COLOR,
            opacity=0.9,
        )
    )
    _base_layout(fig, "How complete are submissions on arrival?", "Completeness (% of applicable fields)", "Inquiries")
    fig.update_xaxes(range=[0, 102], dtick=20, showgrid=False)
    fig.update_layout(bargap=0.12)
    return fig
