"""Reusable Plotly chart builders with consistent colours, denominators and captions."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.styling import GATE_COLOURS, NAVY, PLOTLY_TEMPLATE, STATUS_COLOURS, WHITE


def _render(fig: go.Figure, caption: str | None = None, limitation: str | None = None) -> None:
    # Explicit colours rather than relying on template/theme inheritance: a
    # Plotly figure is otherwise unaffected by the page's light/dark theme,
    # but pinning these keeps chart text legible independent of it.
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=10, t=50, b=10),
        font_color=NAVY,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
    )
    st.plotly_chart(fig, use_container_width=True)
    if caption:
        st.caption(caption)
    if limitation:
        st.caption(f"Limitation: {limitation}")


def status_bar(df: pd.DataFrame, label_col: str, count_col: str, title: str, n_total: int, caption: str | None = None) -> None:
    """Horizontal bar of claim-status classes; preserves zero-count classes."""
    d = df.copy()
    d["pct"] = (d[count_col] / n_total * 100).round(1) if n_total else 0.0
    d["text"] = d.apply(lambda r: f"{int(r[count_col])} ({r['pct']}%)", axis=1)
    colours = [STATUS_COLOURS.get(v, "#8A94A6") for v in d[label_col]]
    fig = go.Figure(go.Bar(
        x=d[count_col], y=d[label_col], orientation="h", text=d["text"], textposition="outside",
        marker_color=colours,
    ))
    fig.update_layout(title=f"{title} (n = {n_total})", xaxis_title="Claims", yaxis_title="")
    _render(fig, caption=caption or f"Counts and share of all {n_total} evaluated claims.")


def category_label_heatmap(df: pd.DataFrame, x_col: str, y_col: str, value_col: str, title: str, caption: str | None = None) -> None:
    """Cell values are counts of value_col per (y_col, x_col) pair, not a sum of value_col itself."""
    pivot = df.pivot_table(index=y_col, columns=x_col, values=value_col, aggfunc="count", fill_value=0)
    fig = px.imshow(
        pivot, text_auto=True, color_continuous_scale="Blues", aspect="auto",
        labels=dict(color="Claims"),
    )
    fig.update_layout(title=title)
    _render(fig, caption=caption)


def stacked_percentage_bar(df: pd.DataFrame, x_col: str, stack_col: str, pct_col: str, n_col: str, title: str, caption: str | None = None) -> None:
    fig = px.bar(
        df, x=x_col, y=pct_col, color=stack_col, barmode="stack",
        hover_data={n_col: True, pct_col: ":.1f"}, title=title,
    )
    fig.update_layout(yaxis_title="% within brand", xaxis_title="")
    _render(fig, caption=caption)


def grouped_bar(df: pd.DataFrame, x_col: str, y_col: str, color_col: str | None, title: str, caption: str | None = None) -> None:
    fig = px.bar(df, x=x_col, y=y_col, color=color_col, barmode="group", title=title)
    _render(fig, caption=caption)


def risk_heatmap(df: pd.DataFrame, likelihood_col: str, impact_col: str, id_col: str, title: str, caption: str | None = None) -> None:
    grid = pd.crosstab(df[impact_col], df[likelihood_col])
    grid = grid.reindex(index=sorted(grid.index, reverse=True), columns=sorted(grid.columns), fill_value=0)
    fig = px.imshow(
        grid, text_auto=True, color_continuous_scale=["#F6F8FB", "#D99A2B", "#C44536"],
        labels=dict(x="Likelihood (1-5)", y="Impact (1-5)", color="Risk count"),
    )
    fig.update_layout(title=title)
    _render(fig, caption=caption or f"{len(df)} risks plotted by residual likelihood x impact.")


def gate_status_bar(df: pd.DataFrame, context: str, title: str) -> tuple[int, int, int]:
    subset = df[df["deployment_context"] == context]
    counts = subset["status"].value_counts()
    go_n, partial_n, nogo_n = int(counts.get("GO", 0)), int(counts.get("PARTIAL", 0)), int(counts.get("NO-GO", 0))
    fig = go.Figure(go.Bar(
        x=["GO", "PARTIAL", "NO-GO"], y=[go_n, partial_n, nogo_n],
        marker_color=[GATE_COLOURS["GO"], GATE_COLOURS["PARTIAL"], GATE_COLOURS["NO-GO"]],
        text=[go_n, partial_n, nogo_n], textposition="outside",
    ))
    fig.update_layout(title=f"{title} (n = {len(subset)} gates)")
    _render(fig)
    return go_n, partial_n, nogo_n
