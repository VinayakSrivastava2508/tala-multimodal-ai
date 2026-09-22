"""Day 3B Streamlit interface: TALA Multimodal Claim-Experience Diagnostic.

Two modes: investigate an existing claim, or ask an open analytical question.
ChromaDB is the only vector store, Gemini is the only generator -- if either
is unavailable, this app shows the error and stops; it never substitutes a
fallback answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import orchestrator
from src.rag.chroma_store import ChromaUnavailableError, get_client
from src.rag.gemini_generator import GeminiConfigError, GeminiGenerationError
from src.rag.schemas import COLLECTION_CLAIMS

st.set_page_config(page_title="TALA Multimodal Claim-Experience Diagnostic", layout="wide")

st.title("TALA Multimodal Claim-Experience Diagnostic")
st.caption("Evidence-grounded analysis across official claims, customer experience, images, video and reference documents.")

# ── Sidebar ──────────────────────────────────────────────────────────────────────

st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", ["Investigate a claim", "Ask an analytical question"])
brand_filter = st.sidebar.selectbox("Brand filter", ["(any)", "TALA", "Adanola", "Girlfriend Collective", "Oner Active"])
modality_filter = st.sidebar.selectbox("Modality filter", ["(any)", "text", "image", "video_frame"])
source_type_filter = st.sidebar.selectbox(
    "Source-type filter",
    ["(any)", "official_claims", "official_reference", "customer_experience", "creator_strategy",
     "official_product_image", "official_video"],
)
evidence_strength_filter = st.sidebar.selectbox("Evidence-strength filter", ["(any)", "strong", "medium"])
n_results = st.sidebar.slider("Number of retrieved evidence items", min_value=3, max_value=20, value=10)
show_trace = st.sidebar.toggle("Show retrieval trace", value=False)


def _build_filters() -> dict:
    filters = {}
    if brand_filter != "(any)":
        filters["brand"] = brand_filter
    if modality_filter != "(any)":
        filters["modality"] = modality_filter
    if source_type_filter != "(any)":
        filters["source_type"] = source_type_filter
    if evidence_strength_filter != "(any)":
        filters["evidence_strength"] = evidence_strength_filter
    return filters


def _evidence_card(entry: dict) -> None:
    meta = entry["metadata"]
    with st.container(border=True):
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**{entry['id']}** &nbsp; `{meta.get('modality', '')}` &nbsp; `{meta.get('source_type', '')}`")
            st.caption(f"Brand: {meta.get('brand', '')} | Evidence strength: {meta.get('evidence_strength', '')}")
            independence = "Independent" if str(meta.get("independent_source")) in ("True", "true") else (
                "Self-reported" if str(meta.get("self_reported")) in ("True", "true") else "Customer/creator-reported")
            st.caption(f"Source independence: **{independence}** | Stance: {meta.get('stance', meta.get('support_or_challenge', ''))}")
            if entry.get("document"):
                st.write(entry["document"][:500])
            elif meta.get("citation_text"):
                st.write(str(meta["citation_text"])[:500])
            source_title = meta.get("source_title")
            source_url = meta.get("source_url")
            if source_title:
                st.caption(f"Source: {source_title}")
            if source_url:
                st.markdown(f"[Source link]({source_url})")
            if meta.get("modality") == "video_frame":
                st.caption(
                    f"Video {meta.get('video_id')} | position: {meta.get('opening_middle_closing_position')} "
                    f"| timestamp: {meta.get('timestamp_seconds')}s | role: {meta.get('video_role')}"
                )
        with cols[1]:
            local_path = meta.get("local_asset_path")
            if local_path and (PROJECT_ROOT / local_path).exists():
                st.image(str(PROJECT_ROOT / local_path), use_container_width=True)
            st.metric("Similarity", f"{entry.get('best_similarity', entry.get('similarity', 0)):.3f}")
            st.caption(f"Final rank: {entry.get('final_rank', '-')}")


def _diagnostic_panel(fused_candidates: list[dict], modalities_searched: list[str], filters: dict, response: dict | None, validation: dict | None) -> None:
    st.subheader("Diagnostic trace")
    st.write("Collections/modalities searched:", modalities_searched)
    st.write("Filters applied:", filters or "(none)")
    st.write(f"Candidates retrieved: {len(fused_candidates)}")
    trace_rows = []
    for c in fused_candidates:
        trace_rows.append({
            "id": c["id"], "collection": c["collection"], "final_rank": c.get("final_rank"),
            "final_score": c.get("final_score"), "rank_fusion_contribution": c["trace"]["rank_fusion_contribution"],
            "direct_link_bonus": c["trace"].get("direct_link_bonus"), "exclusion_reason": c["trace"]["exclusion_reason"],
        })
    st.dataframe(pd.DataFrame(trace_rows), use_container_width=True)
    if response:
        st.write("Gemini model used:", response.get("_model_used"))
    if validation:
        st.write("Citation-validation status:", "PASSED" if validation["passed"] else "FAILED")
        if not validation["passed"]:
            st.error(validation["failures"])


def _render_generation(response: dict, validation: dict) -> None:
    if not validation["passed"]:
        st.error("Citation validation FAILED -- the answer is shown for transparency but did not pass automated checks:")
        for failure in validation["failures"]:
            st.write("-", failure)
        return
    st.markdown("### Generated explanation")
    st.write(response["concise_answer"])
    st.markdown("**Claim assessment**")
    st.write(response["claim_assessment"])
    st.markdown("**Confidence explanation**")
    st.write(response["confidence_explanation"])
    if response.get("caveats"):
        st.markdown("**Caveats**")
        for c in response["caveats"]:
            st.write("-", c)
    if response.get("evidence_gaps"):
        st.markdown("**Evidence gaps**")
        for g in response["evidence_gaps"]:
            st.write("-", g)


try:
    client = get_client()
except ChromaUnavailableError as exc:
    st.error(f"ChromaDB is unavailable: {exc}")
    st.stop()

if mode == "Investigate a claim":
    try:
        collection = client.get_collection(COLLECTION_CLAIMS)
        all_claims = collection.get(include=["metadatas"])
    except Exception as exc:
        st.error(f"Could not load claims from ChromaDB: {exc}")
        st.stop()

    claims_meta = sorted(all_claims["metadatas"], key=lambda m: m["claim_id"])
    label_to_id = {
        f"[{m['claim_category']}] {m['claim_text'][:110]}{'...' if len(m['claim_text']) > 110 else ''}  ({m['claim_id']})": m["claim_id"]
        for m in claims_meta
    }
    selected_label = st.selectbox("Select a claim to investigate", list(label_to_id.keys()))
    selected_claim = label_to_id[selected_label]

    if st.button("Investigate", type="primary"):
        with st.spinner("Retrieving evidence and generating explanation..."):
            try:
                result = orchestrator.investigate_claim(selected_claim, n_results=n_results, filters=_build_filters())
            except (GeminiConfigError, GeminiGenerationError) as exc:
                st.error(f"Answer generation failed: {exc}")
                st.stop()

        meta = result["claim"]["metadata"]
        st.subheader("Claim")
        st.write(result["claim"]["document"])
        cols = st.columns(4)
        cols[0].metric("Category", result["claim_category"])
        cols[1].metric("Fusion label", meta["fusion_label"])
        cols[2].metric("Confidence", f"{meta['confidence']:.2f}")
        cols[3].metric("Presentation restriction", meta["presentation_restriction"])
        if meta.get("conflicting_evidence"):
            st.warning("Evidence conflict flagged for this claim.")

        if result["generation"] and result["validation"]:
            _render_generation(result["generation"], result["validation"])

        st.markdown("### Supporting evidence")
        for e in result["supports"]:
            _evidence_card(e)
        st.markdown("### Challenging evidence")
        for e in result["challenges"]:
            _evidence_card(e)
        st.markdown("### Contextual evidence")
        for e in result["context"]:
            _evidence_card(e)

        if show_trace:
            _diagnostic_panel(result["fused_candidates"], result["modalities_searched"], _build_filters(), result["generation"], result["validation"])

else:
    question = st.text_input("Enter an analytical question about TALA's claim-experience divergence")
    if st.button("Submit", type="primary") and question:
        with st.spinner("Routing question and retrieving evidence..."):
            try:
                result = orchestrator.answer_question(question, n_results=n_results, filters=_build_filters())
            except (GeminiConfigError, GeminiGenerationError) as exc:
                st.error(f"Answer generation failed: {exc}")
                st.stop()

        st.write("Detected intent(s):", result["intents"])
        st.write("Modalities searched:", result["modalities_searched"])

        if result["generation"] and result["validation"]:
            _render_generation(result["generation"], result["validation"])

        st.markdown("### Retrieved evidence")
        for e in result["evidence"]:
            _evidence_card(e)

        if show_trace:
            _diagnostic_panel(result["fused_candidates"], result["modalities_searched"], _build_filters(), result["generation"], result["validation"])
