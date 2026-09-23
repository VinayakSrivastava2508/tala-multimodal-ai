"""Shared limitation/boundary notice renderer."""

from __future__ import annotations

import streamlit as st


def notice(text: str) -> None:
    st.markdown(f"<div class='cockpit-notice'>{text}</div>", unsafe_allow_html=True)
