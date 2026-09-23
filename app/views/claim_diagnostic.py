"""Page 2: Claim Diagnostic -- claim-level filtering, drill-down and RAG handoff."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import data_loader as dl
from app.components.evidence_cards import evidence_gap_notice
from app.state import CLAIM_FILTERS_KEY, send_claim_to_rag


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filters = st.session_state.get(CLAIM_FILTERS_KEY, {}) or {}
    categories = sorted(df["claim_category"].dropna().unique().tolist())
    labels = sorted(df["production_fusion_label"].dropna().unique().tolist())

    col1, col2, col3 = st.columns(3)
    default_cat = filters.get("claim_category", "(any)")
    cat = col1.selectbox("Claim category", ["(any)"] + categories, index=(["(any)"] + categories).index(default_cat) if default_cat in categories else 0)
    label = col2.selectbox("Production fusion label", ["(any)"] + labels)
    modality = col3.selectbox("Modality present", ["(any)", "text", "image", "video", "reference"])

    col4, col5, col6 = st.columns(3)
    indep = col4.selectbox("Independent evidence present", ["(any)", "yes", "no"])
    conflict = col5.selectbox("Conflicting evidence", ["(any)", "yes", "no"])
    sensitive = col6.selectbox("Sensitive to weights", ["(any)", "yes", "no"])

    restriction = st.selectbox("Presentation restriction", ["(any)"] + sorted(df["presentation_restriction"].dropna().unique().tolist()))

    out = df.copy()
    if cat != "(any)":
        out = out[out["claim_category"] == cat]
    if label != "(any)":
        out = out[out["production_fusion_label"] == label]
    if modality != "(any)":
        out = out[out["modalities_present"].fillna("").str.contains(modality)]
    if indep != "(any)":
        want = indep == "yes"
        out = out[(pd.to_numeric(out["independent_evidence_count"], errors="coerce").fillna(0) > 0) == want]
    if conflict != "(any)":
        want = conflict == "yes"
        out = out[out["conflicting_evidence"].astype(bool) == want]
    if sensitive != "(any)":
        want = sensitive == "yes"
        out = out[out["sensitive_to_weights"].astype(bool) == want]
    if restriction != "(any)":
        out = out[out["presentation_restriction"] == restriction]

    st.session_state[CLAIM_FILTERS_KEY] = {"claim_category": cat}
    return out


def render() -> None:
    st.title("Claim Diagnostic")
    master = dl.load_claim_master()

    filtered = _apply_filters(master)
    st.caption(f"Showing {len(filtered)} of {len(master)} claims.")

    if filtered.empty:
        st.warning("No claims match the current filters. Adjust filters above.")
        return

    display_cols = {
        "claim_id": "Claim ID", "claim_text": "Claim (short)", "claim_category": "Category",
        "production_fusion_label": "Label", "fusion_confidence": "Confidence",
        "evidence_unit_count": "Evidence units", "modalities_present": "Modalities",
        "independent_evidence_count": "Independent evidence", "presentation_restriction": "Restriction",
    }
    table = filtered[list(display_cols)].rename(columns=display_cols).copy()
    table["Claim (short)"] = table["Claim (short)"].str.slice(0, 100) + "..."
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered claims (CSV, full evidence IDs retained)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="claim_diagnostic_filtered.csv",
    )

    st.divider()
    st.subheader("Claim detail")
    claim_id = st.selectbox("Select a claim", filtered["claim_id"].tolist())
    row = filtered[filtered["claim_id"] == claim_id].iloc[0]

    st.markdown(f"**Official claim:** {row['claim_text']}")
    if row.get("official_source_url"):
        st.markdown(f"[Official source]({row['official_source_url']})")
    cols = st.columns(5)
    cols[0].metric(
        "Fusion label", row["production_fusion_label"],
        help="The system's evidence classification. It is not a legal or factual verdict.",
    )
    cols[1].metric("Confidence", f"{float(row['fusion_confidence']):.2f}")
    cols[2].metric("Modalities present", row["modalities_present"])
    cols[3].metric(
        "Weight-sensitive", "Yes" if bool(row["sensitive_to_weights"]) else "No",
        help="The classification can change under reasonable changes to evidence weights.",
    )
    cols[4].metric(
        "Presentation restriction", row.get("presentation_restriction", "-"),
        help="How the finding may be communicated based on evidence strength and risk.",
    )

    cols = st.columns(3)
    cols[0].metric("Supporting evidence", row.get("supporting_evidence_count", "-"))
    cols[1].metric("Challenging evidence", row.get("challenging_evidence_count", "-"))
    cols[2].metric("Contextual evidence", row.get("contextual_evidence_count", "-"))
    cols = st.columns(2)
    cols[0].metric(
        "Independent evidence", row.get("independent_evidence_count", "-"),
        help="Evidence not published by the brand itself. Independence does not automatically establish correctness.",
    )
    cols[1].metric("Self-reported evidence", row.get("self_reported_evidence_count", "-"))

    evidence_gap_notice(row.get("evidence_gap", ""))
    st.write(f"**Analytical interpretation:** {row.get('analytical_interpretation', '-')}")
    st.write(f"**Management implication:** {row.get('managerial_implication', '-')}")
    st.write(f"**Recommended action:** {row.get('recommended_management_action', '-')}")
    st.caption(f"Caveat: {row.get('caveat', '-')}")

    if st.button("Investigate this claim in Multimodal RAG", type="primary"):
        send_claim_to_rag(claim_id)
        st.rerun()

    st.divider()
    st.subheader("8.1 Mixed-theme handling")
    st.caption(
        f"{len(master[master['production_fusion_label'] == 'mixed'])} mixed claim record(s) map to "
        f"underlying strategic themes below -- do not treat these as independent problems."
    )
    themes = dl.load_mixed_theme_summary()
    for _, t in themes.iterrows():
        with st.container(border=True):
            st.markdown(f"**{t['theme_name']}** ({int(t['claim_record_count'])} claim record(s))")
            st.write(f"Representative claim: {t['representative_claim']}")
            st.write(f"Observed tension: {t['observed_tension']}")
            st.write(
                f"Supporting evidence: {t.get('supporting_evidence_count', '-')} | "
                f"Challenging evidence: {t.get('challenging_evidence_count', '-')}"
            )
            st.write(f"**Management implication:** {t['management_implication']}")
            st.caption(f"Caveat: {t['caveat']}")
