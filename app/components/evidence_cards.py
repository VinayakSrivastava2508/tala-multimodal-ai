"""Executive-facing claim/evidence card renderers, shared by Claim Diagnostic and RAG views."""

from __future__ import annotations

import streamlit as st

from app.styling import STATUS_COLOURS


def claim_summary_card(row: dict) -> None:
    label = row.get("production_fusion_label", "unknown")
    colour = STATUS_COLOURS.get(label, "#8A94A6")
    with st.container(border=True):
        st.markdown(f"**{row.get('claim_id', '')}** &nbsp; :{'' if False else ''}")
        st.markdown(f"<span style='color:{colour}; font-weight:600'>{label}</span>", unsafe_allow_html=True)
        st.write(row.get("claim_text", ""))
        cols = st.columns(4)
        cols[0].metric("Confidence", f"{float(row.get('fusion_confidence', 0) or 0):.2f}")
        cols[1].metric("Evidence units", row.get("evidence_unit_count", "-"))
        cols[2].metric("Independent evidence", row.get("independent_evidence_count", "-"))
        cols[3].metric("Modalities", row.get("modalities_present", "-"))


def evidence_gap_notice(gap_text: str) -> None:
    if gap_text and str(gap_text).strip() and str(gap_text).strip().lower() != "nan":
        st.info(f"Evidence gap (not a contradiction): {gap_text}")
