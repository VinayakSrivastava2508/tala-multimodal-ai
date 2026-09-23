"""Page 4: Multimodal RAG -- preserves existing claim-investigation and open-question flows.

All RAG/model imports are lazy (inside render()) so opening this page's module
never triggers ChromaDB, embedding, or Gemini initialisation until the user
explicitly submits a query.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from app import data_loader as dl
from app.state import (
    CLAIM_ID_KEY,
    RAG_FILTERS_KEY,
    RAG_HANDOFF_KEY,
    RAG_QUESTION_KEY,
    RAG_RESULT_KEY,
    RAG_TRACE_KEY,
    clear_rag_investigation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_filters(brand, modality, source_type, evidence_strength) -> dict:
    filters = {}
    if brand != "(any)":
        filters["brand"] = brand
    if modality != "(any)":
        filters["modality"] = modality
    if source_type != "(any)":
        filters["source_type"] = source_type
    if evidence_strength != "(any)":
        filters["evidence_strength"] = evidence_strength
    return filters


def _evidence_card(entry: dict) -> None:
    meta = entry["metadata"]
    with st.container(border=True):
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**{entry['id']}** &nbsp; `{meta.get('modality', '')}` &nbsp; `{meta.get('source_type', '')}`")
            st.caption(f"Brand: {meta.get('brand', '')} | Evidence strength: {meta.get('evidence_strength', '')}")
            # source_origin/independence_status are the two explicit provenance
            # concepts written by src/rag/corpus_builder.py (Part D of the
            # Executive Cockpit Acceptance Patch). Fall back to a conservative
            # re-derivation from the older boolean flags only if the index
            # hasn't been rebuilt yet with the new fields.
            if "source_origin" in meta and "independence_status" in meta:
                source_origin, independence_status = meta["source_origin"], meta["independence_status"]
            else:
                from src.rag.schemas import classify_evidence_provenance
                source_origin, independence_status = classify_evidence_provenance(
                    meta.get("source_type", ""),
                    str(meta.get("self_reported")) in ("True", "true"),
                    str(meta.get("independent_source")) in ("True", "true"),
                )
            st.caption(
                f"Source: **{source_origin}** | Independence: **{independence_status}** "
                f"| Stance: {meta.get('stance', meta.get('support_or_challenge', ''))}"
            )
            if entry.get("document"):
                st.write(entry["document"][:500])
            elif meta.get("citation_text"):
                st.write(str(meta["citation_text"])[:500])
            if meta.get("source_title"):
                st.caption(f"Source: {meta['source_title']}")
            if meta.get("source_url"):
                st.markdown(f"[Source link]({meta['source_url']})")
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


def _diagnostic_panel(fused_candidates, modalities_searched, filters, response, validation) -> None:
    st.subheader("Diagnostic trace")
    st.write("Collections/modalities searched:", modalities_searched)
    st.write("Filters applied:", filters or "(none)")
    st.write(f"Candidates retrieved: {len(fused_candidates)}")
    trace_rows = [{
        "id": c["id"], "collection": c["collection"], "final_rank": c.get("final_rank"),
        "final_score": c.get("final_score"), "rank_fusion_contribution": c["trace"]["rank_fusion_contribution"],
        "direct_link_bonus": c["trace"].get("direct_link_bonus"), "exclusion_reason": c["trace"]["exclusion_reason"],
    } for c in fused_candidates]
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


def _render_persisted_result(persisted: dict, show_trace: bool) -> None:
    """Render a previously completed investigation/question from session state.

    This never triggers retrieval or generation -- it only reads `persisted`,
    which is written exactly once per successful Investigate/Submit click.
    Called on every run a result exists, independent of whether a button was
    clicked this run, so toggling display controls (e.g. show_trace) cannot
    make the result disappear.
    """
    handoff_ctx = persisted.get("handoff_context")
    if handoff_ctx:
        with st.container(border=True):
            st.markdown("**Arrived from Claim Diagnostic**")
            st.write(f"Claim: {handoff_ctx['claim_text']}")
            cols = st.columns(3)
            cols[0].metric("Category", handoff_ctx["claim_category"])
            cols[1].metric("Production label", handoff_ctx["production_fusion_label"])
            cols[2].metric("Presentation restriction", handoff_ctx["presentation_restriction"])

    if persisted["mode"] == "Investigate a claim":
        st.subheader("Claim")
        st.write(persisted["claim_document"])
        cols = st.columns(4)
        cols[0].metric("Category", persisted["claim_category"])
        cols[1].metric("Fusion label", persisted["fusion_label"])
        cols[2].metric("Confidence", f"{persisted['confidence']:.2f}")
        cols[3].metric("Presentation restriction", persisted["presentation_restriction"])
        if persisted.get("conflicting_evidence"):
            st.warning("Evidence conflict flagged for this claim.")

        if persisted["generation"] and persisted["validation"]:
            _render_generation(persisted["generation"], persisted["validation"])

        st.markdown("### Supporting evidence")
        for e in persisted["supports"]:
            _evidence_card(e)
        st.markdown("### Challenging evidence")
        for e in persisted["challenges"]:
            _evidence_card(e)
        st.markdown("### Contextual evidence")
        for e in persisted["context"]:
            _evidence_card(e)
    else:
        st.write("Detected intent(s):", persisted["intents"])
        st.write("Modalities searched:", persisted["modalities_searched"])

        if persisted["generation"] and persisted["validation"]:
            _render_generation(persisted["generation"], persisted["validation"])

        st.markdown("### Retrieved evidence")
        for e in persisted["evidence"]:
            _evidence_card(e)

    st.caption(f"Answer generated in {persisted['latency_seconds']:.1f}s.")

    if show_trace:
        _diagnostic_panel(
            persisted["fused_candidates"], persisted["modalities_searched"],
            persisted["filters_used"], persisted["generation"], persisted["validation"],
        )


def render() -> None:
    st.title("Multimodal RAG")
    st.caption("Evidence-grounded analysis across official claims, customer experience, images, video and reference documents.")

    persisted = st.session_state.get(RAG_RESULT_KEY)
    handoff_claim = st.session_state.get(RAG_HANDOFF_KEY)
    handoff_row = None
    # Only show the pre-investigation "Arrived from Claim Diagnostic" banner
    # when the current handoff claim hasn't been investigated yet (a *stale*
    # persisted result for a *different* claim must never suppress a fresh
    # handoff banner). Once investigated, the same context travels inside the
    # persisted result itself (see _render_persisted_result), so it survives
    # trace-toggle reruns instead of depending on this banner.
    if handoff_claim and (not persisted or persisted.get("claim_id") != handoff_claim):
        try:
            master = dl.load_claim_master()
            row = master[master["claim_id"] == handoff_claim]
            if not row.empty:
                handoff_row = row.iloc[0]
                with st.container(border=True):
                    st.markdown("**Arrived from Claim Diagnostic**")
                    st.write(f"Claim: {handoff_row['claim_text']}")
                    cols = st.columns(3)
                    cols[0].metric("Category", handoff_row["claim_category"])
                    cols[1].metric("Production label", handoff_row["production_fusion_label"])
                    cols[2].metric("Presentation restriction", handoff_row["presentation_restriction"])
        except dl.MissingTableError as exc:
            st.warning(str(exc))

    # Lazy import: RAG/model modules only load once this page actually renders.
    from src.rag import orchestrator
    from src.rag.chroma_store import ChromaUnavailableError, get_client
    from src.rag.gemini_generator import GeminiConfigError, GeminiGenerationError
    from src.rag.schemas import COLLECTION_CLAIMS

    st.sidebar.header("RAG Controls")
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
    # Display-only: never clears a result, never reruns retrieval, never calls
    # Gemini. It only changes whether _render_persisted_result shows the trace.
    show_trace = st.sidebar.toggle("Show retrieval trace", value=st.session_state.get(RAG_TRACE_KEY, False))
    st.session_state[RAG_TRACE_KEY] = show_trace

    if st.session_state.get(RAG_RESULT_KEY) and st.sidebar.button("Clear investigation"):
        clear_rag_investigation()
        st.rerun()

    filters = _build_filters(brand_filter, modality_filter, source_type_filter, evidence_strength_filter)
    st.session_state[RAG_FILTERS_KEY] = filters

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
        id_to_label = {v: k for k, v in label_to_id.items()}
        preselect = st.session_state.get(CLAIM_ID_KEY)
        options = list(label_to_id.keys())
        default_index = options.index(id_to_label[preselect]) if preselect in id_to_label else 0
        selected_label = st.selectbox("Select a claim to investigate", options, index=default_index)
        selected_claim = label_to_id[selected_label]
        st.session_state[CLAIM_ID_KEY] = selected_claim

        if st.button("Investigate", type="primary"):
            start = time.perf_counter()
            with st.spinner("Retrieving evidence and generating explanation..."):
                try:
                    result = orchestrator.investigate_claim(selected_claim, n_results=n_results, filters=filters)
                except (GeminiConfigError, GeminiGenerationError) as exc:
                    st.error(f"Answer generation failed: {exc}")
                    st.stop()
            elapsed = time.perf_counter() - start

            meta = result["claim"]["metadata"]
            handoff_context = None
            if handoff_row is not None and handoff_claim == selected_claim:
                handoff_context = {
                    "claim_text": handoff_row["claim_text"],
                    "claim_category": handoff_row["claim_category"],
                    "production_fusion_label": handoff_row["production_fusion_label"],
                    "presentation_restriction": handoff_row["presentation_restriction"],
                }
            st.session_state[RAG_RESULT_KEY] = {
                "mode": "Investigate a claim",
                "claim_id": selected_claim,
                "claim_document": result["claim"]["document"],
                "claim_category": result["claim_category"],
                "fusion_label": meta["fusion_label"],
                "confidence": meta["confidence"],
                "presentation_restriction": meta["presentation_restriction"],
                "conflicting_evidence": meta.get("conflicting_evidence"),
                "generation": result["generation"],
                "validation": result["validation"],
                "supports": result["supports"],
                "challenges": result["challenges"],
                "context": result["context"],
                "fused_candidates": result["fused_candidates"],
                "modalities_searched": result["modalities_searched"],
                "filters_used": filters,
                "handoff_context": handoff_context,
                "latency_seconds": elapsed,
            }
            st.rerun()

    else:
        question = st.text_input("Enter an analytical question about TALA's claim-experience divergence", value=st.session_state.get(RAG_QUESTION_KEY, ""))
        st.session_state[RAG_QUESTION_KEY] = question
        if st.button("Submit", type="primary") and question:
            start = time.perf_counter()
            with st.spinner("Routing question and retrieving evidence..."):
                try:
                    result = orchestrator.answer_question(question, n_results=n_results, filters=filters)
                except (GeminiConfigError, GeminiGenerationError) as exc:
                    st.error(f"Answer generation failed: {exc}")
                    st.stop()
            elapsed = time.perf_counter() - start

            st.session_state[RAG_RESULT_KEY] = {
                "mode": "Ask an analytical question",
                "question": question,
                "intents": result["intents"],
                "generation": result["generation"],
                "validation": result["validation"],
                "evidence": result["evidence"],
                "fused_candidates": result["fused_candidates"],
                "modalities_searched": result["modalities_searched"],
                "filters_used": filters,
                "handoff_context": None,
                "latency_seconds": elapsed,
            }
            st.rerun()

    if st.session_state.get(RAG_RESULT_KEY):
        _render_persisted_result(st.session_state[RAG_RESULT_KEY], show_trace)
