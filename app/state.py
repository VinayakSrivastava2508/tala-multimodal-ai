"""Session-state keys and helpers for the Executive Decision Cockpit."""

from __future__ import annotations

import streamlit as st

PAGE_KEY = "cockpit_page"
PENDING_PAGE_KEY = "_cockpit_pending_page"
CLAIM_ID_KEY = "cockpit_selected_claim_id"
CLAIM_FILTERS_KEY = "cockpit_claim_filters"
RAG_QUESTION_KEY = "cockpit_rag_question"
RAG_FILTERS_KEY = "cockpit_rag_filters"
RAG_TRACE_KEY = "cockpit_rag_show_trace"
RAG_HANDOFF_KEY = "cockpit_rag_handoff_claim"
RAG_RESULT_KEY = "cockpit_rag_result"

DEFAULTS = {
    PAGE_KEY: "Executive Overview",
    CLAIM_ID_KEY: None,
    CLAIM_FILTERS_KEY: {},
    RAG_QUESTION_KEY: "",
    RAG_FILTERS_KEY: {},
    RAG_TRACE_KEY: False,
    RAG_HANDOFF_KEY: None,
    RAG_RESULT_KEY: None,
}


def clear_rag_investigation() -> None:
    """Explicit reset for the 'Clear investigation' control -- the only other way
    a persisted RAG_RESULT_KEY may be cleared is a fresh, explicit submission
    that overwrites it (never a display-only control like the trace toggle)."""
    st.session_state[RAG_RESULT_KEY] = None
    st.session_state[RAG_HANDOFF_KEY] = None


def init_state() -> None:
    for key, default in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def go_to_page(page: str) -> None:
    """Request a navigation. PAGE_KEY reflects it immediately; the sidebar radio's
    own (private, differently-keyed) widget state is synced from PENDING_PAGE_KEY
    on the next run by navigation.render_sidebar_nav() -- neither key is ever the
    widget's bound key, so writing here is always safe even though this runs
    after the sidebar has already been rendered this script pass.
    """
    st.session_state[PAGE_KEY] = page
    st.session_state[PENDING_PAGE_KEY] = page


def send_claim_to_rag(claim_id: str) -> None:
    """Store the claim for the RAG view and navigate there without calling Gemini."""
    st.session_state[CLAIM_ID_KEY] = claim_id
    st.session_state[RAG_HANDOFF_KEY] = claim_id
    go_to_page("Multimodal RAG")
