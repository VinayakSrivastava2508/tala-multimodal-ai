"""Reusable metric-card row for executive KPI display."""

from __future__ import annotations

import streamlit as st


def metric_row(metrics: list[tuple[str, str, str | None]], per_row: int = 5) -> None:
    """Render metrics as (label, value, help_text) tuples in a wrapped grid of columns."""
    for start in range(0, len(metrics), per_row):
        chunk = metrics[start:start + per_row]
        cols = st.columns(len(chunk))
        for col, (label, value, help_text) in zip(cols, chunk):
            col.metric(label, value, help=help_text)
