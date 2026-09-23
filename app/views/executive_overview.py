"""Page 1: Executive Overview -- portfolio-level claim-experience findings."""

from __future__ import annotations

import sys
from contextlib import contextmanager

import pandas as pd
import streamlit as st

from app import data_loader as dl
from app.components import charts
from app.components.metric_cards import metric_row
from app.state import go_to_page

FUSION_CAPTION = (
    "Added modalities increased evidence breadth and confidence calibration; "
    "they did not change the production label distribution, and correctness "
    "was not independently measured."
)


@contextmanager
def _section(name: str):
    """Isolate one subsection so its failure doesn't blank the rest of the page."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001 -- executive-facing pages must never show a raw traceback
        st.error(f"The '{name}' section hit an unexpected error and could not render.")
        print(f"[cockpit] unhandled error in Executive Overview section '{name}': {exc!r}", file=sys.stderr)


def render() -> None:
    st.title("TALA Multimodal AI Decision Cockpit")
    st.caption("Claim–experience divergence, creator strategy and evidence-grounded decision support")

    master = dl.load_claim_master()
    label_summary = dl.load_label_summary()
    n_total = len(master)

    overall = label_summary[label_summary["section"] == "overall_label_distribution"]
    counts = overall.set_index("automated_label")["n_claims"].to_dict()
    all_labels = ["aligned", "partially_aligned", "mixed", "divergent", "insufficient_evidence"]
    for lbl in all_labels:
        counts.setdefault(lbl, 0)

    flags = master["modalities_present"].fillna("").map(dl.modality_flags)
    n_image = int(sum(f["image"] for f in flags))
    n_video = int(sum(f["video"] for f in flags))
    n_reference = int(sum(f["reference"] for f in flags))
    n_independent = int((pd.to_numeric(master["independent_evidence_count"], errors="coerce").fillna(0) > 0).sum())

    theme_summary = dl.load_mixed_theme_summary()

    st.subheader("Portfolio at a glance")
    metric_row([
        ("Total claims", str(n_total), "All evaluated official claim records (analytical_synthesis_claim_master.csv)."),
        ("Aligned", str(int(counts["aligned"])), None),
        ("Mixed", str(int(counts["mixed"])), f"{len(theme_summary)} underlying mixed theme(s) (see mixed_claim_theme_summary.csv)."),
        ("Insufficient evidence", str(int(counts["insufficient_evidence"])), None),
        ("Claims w/ independent evidence", str(n_independent), None),
    ])
    metric_row([
        ("Claims w/ image evidence", str(n_image), None),
        ("Claims w/ video evidence", str(n_video), None),
        ("Claims w/ reference evidence", str(n_reference), None),
        ("Divergent", str(int(counts["divergent"])), None),
        ("Partially aligned", str(int(counts["partially_aligned"])), None),
    ])

    st.divider()
    with _section("7.1 Claim-status overview"):
        st.subheader("7.1 Claim-status overview")
        status_df = pd.DataFrame({"automated_label": all_labels, "n_claims": [counts[l] for l in all_labels]})
        charts.status_bar(status_df, "automated_label", "n_claims", "Claim status distribution", n_total)

    with _section("7.2 Claim-category x fusion-label heatmap"):
        st.subheader("7.2 Claim-category x fusion-label heatmap")
        charts.category_label_heatmap(
            master, "production_fusion_label", "claim_category", "claim_id",
            "Claims by category and production fusion label",
            caption="Cell values are claim counts. Select a category below to filter in Claim Diagnostic.",
        )
        categories = sorted(master["claim_category"].dropna().unique().tolist())
        selected_cat = st.selectbox("Jump to Claim Diagnostic filtered by category", ["(none)"] + categories)
        if selected_cat != "(none)" and st.button("Open in Claim Diagnostic"):
            st.session_state["cockpit_claim_filters"] = {"claim_category": selected_cat}
            go_to_page("Claim Diagnostic")
            st.rerun()

    with _section("7.3 Modality coverage"):
        st.subheader("7.3 Modality coverage")
        coverage_rows = [
            {"coverage": "Text", "n": int(sum(f["text"] for f in flags))},
            {"coverage": "Image", "n": n_image},
            {"coverage": "Video", "n": n_video},
            {"coverage": "Reference", "n": n_reference},
            {"coverage": "Two or more modalities", "n": int(sum(sum(f.values()) >= 2 for f in flags))},
            {"coverage": "Full text+image+video", "n": int(sum(f["text"] and f["image"] and f["video"] for f in flags))},
            {"coverage": "Full text+image+video+reference", "n": int(master["full_multimodal_bundle"].fillna(False).astype(bool).sum())},
        ]
        charts.grouped_bar(pd.DataFrame(coverage_rows), "coverage", "n", None, f"Modality coverage across {n_total} claims")

    with _section("7.4 Fusion contribution across evidence configurations"):
        st.subheader("7.4 Fusion contribution across evidence configurations")
        fusion_eval = dl.load_fusion_contribution()
        if fusion_eval is not None and "evaluation_type" in fusion_eval.columns:
            st.dataframe(
                fusion_eval[[c for c in ["evaluation_type", "expected_property", "observed_property", "passed"] if c in fusion_eval.columns]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No per-configuration fusion evaluation table found; showing production label distribution only.")
        st.warning(FUSION_CAPTION)

    with _section("7.5 Executive findings"):
        st.subheader("7.5 Executive findings")
        findings = dl.load_executive_findings()
        for _, f in findings.head(12).iterrows():
            with st.container(border=True):
                st.markdown(f"**{f['finding_id']}** — {f['finding']}")
                st.caption(f"Evidence confidence: {f.get('evidence_confidence', '-')}")
                st.write(f"**Business implication:** {f.get('business_implication', '-')}")
                st.write(f"**Recommended action:** {f.get('recommended_action', '-')}")
                st.caption(f"Caveat: {f.get('caveat', '-')}")

    with _section("7.6 Evidence gaps"):
        st.subheader("7.6 Evidence gaps")
        gaps = dl.load_evidence_gap_by_category()
        labour_gap = gaps[gaps["claim_category"].str.lower() == "labour"]
        if not labour_gap.empty:
            row = labour_gap.iloc[0]
            st.info(
                f"Labour evidence gap: {int(row['total_claims'])} labour claim(s), "
                f"{int(row['claims_with_independent_evidence'])} with independent evidence, "
                f"dominant missing evidence type: {row['dominant_missing_evidence_type']}."
            )
        st.dataframe(gaps, use_container_width=True, hide_index=True)
        st.caption("Insufficient evidence reflects a gap in the current corpus, not a finding that a claim is false.")
