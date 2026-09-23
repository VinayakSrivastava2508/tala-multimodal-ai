"""Shared colour tokens and CSS for the Executive Decision Cockpit."""

from __future__ import annotations

import streamlit as st

NAVY = "#1B2A4A"
CORAL = "#E4572E"
GREEN = "#2E7D5B"
AMBER = "#D99A2B"
RED = "#C44536"
GREY = "#8A94A6"
BG = "#F6F8FB"
WHITE = "#FFFFFF"

STATUS_COLOURS = {
    "aligned": GREEN,
    "partially_aligned": AMBER,
    "mixed": AMBER,
    "divergent": RED,
    "insufficient_evidence": GREY,
}

BRAND_COLOURS = {
    "TALA": NAVY,
    "Adanola": CORAL,
    "Girlfriend Collective": GREEN,
    "Oner Active": AMBER,
}

GATE_COLOURS = {"GO": GREEN, "PARTIAL": AMBER, "NO-GO": RED}

PLOTLY_TEMPLATE = "plotly_white"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {BG}; }}
        [data-testid="stSidebar"] {{ background-color: {WHITE}; border-right: 1px solid #E3E7EE; }}
        h1, h2, h3 {{ color: {NAVY}; font-family: "Source Sans Pro", sans-serif; }}
        [data-testid="stMetric"] {{
            background-color: {WHITE};
            border: 1px solid #E3E7EE;
            border-radius: 8px;
            padding: 14px 16px;
        }}
        .cockpit-card {{
            background-color: {WHITE};
            border: 1px solid #E3E7EE;
            border-radius: 8px;
            padding: 16px 18px;
            margin-bottom: 12px;
        }}
        .cockpit-notice {{
            background-color: #FFF6EC;
            border-left: 4px solid {AMBER};
            padding: 10px 14px;
            border-radius: 4px;
            margin-bottom: 14px;
        }}
        .cockpit-caption {{ color: {GREY}; font-size: 0.85rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
