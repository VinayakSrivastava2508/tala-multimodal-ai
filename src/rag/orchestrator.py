"""Day 3B: ties query_router + retriever + rank_fusion + gemini_generator +
citation_validator together for the two interaction modes (claim
investigation, open analytical question). Used by both the Streamlit app and
the evaluation script so there is exactly one retrieval/generation pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.rag import citation_validator, gemini_generator, query_router, rank_fusion, retriever
from src.rag.chroma_store import get_client
from src.rag.schemas import COLLECTION_CLAIMS, COLLECTION_TEXT_EVIDENCE, COLLECTION_VISUAL_EVIDENCE


def stance_for_claim(metadata: dict, claim_id: str) -> str:
    """claim_id and stance/support_or_challenge are index-aligned,
    semicolon-joined lists on a shared evidence row -- returns the stance
    token for THIS specific claim, not just the first one."""
    claim_ids = str(metadata.get("claim_id", "")).split(";")
    stances = str(metadata.get("stance", metadata.get("support_or_challenge", ""))).split(";")
    if claim_id in claim_ids:
        idx = claim_ids.index(claim_id)
        if idx < len(stances):
            return stances[idx]
    return "unlinked"


def _split_by_stance(entries: List[dict], claim_id: Optional[str]) -> Dict[str, List[dict]]:
    supports, challenges, context = [], [], []
    for e in entries:
        stance = stance_for_claim(e["metadata"], claim_id) if claim_id else e["metadata"].get("stance", e["metadata"].get("support_or_challenge", ""))
        if stance == "supports":
            supports.append(e)
        elif stance == "challenges":
            challenges.append(e)
        else:
            context.append(e)
    return {"supports": supports, "challenges": challenges, "context": context}


def _get_claim_record(client, claim_id: str) -> Optional[dict]:
    collection = client.get_collection(COLLECTION_CLAIMS)
    result = collection.get(ids=[f"claim::{claim_id}"], include=["documents", "metadatas"])
    if not result["ids"]:
        return None
    return {"id": result["ids"][0], "document": result["documents"][0], "metadata": result["metadatas"][0]}


def investigate_claim(claim_id: str, n_results: int = 10, filters: Optional[Dict[str, str]] = None, generate: bool = True) -> dict:
    client = get_client()
    claim = _get_claim_record(client, claim_id)
    if claim is None:
        raise ValueError(f"Claim '{claim_id}' not found in {COLLECTION_CLAIMS}")

    meta = claim["metadata"]
    claim_category = meta["claim_category"]
    modalities = query_router.route_modalities_for_claim(claim_category)

    direct = retriever.direct_linked_evidence(client, claim_id)
    candidate_lists = {"direct_link": direct}

    text_semantic = retriever.retrieve_text_evidence(client, claim["document"], n_results, filters)
    candidate_lists["text_semantic"] = text_semantic

    if "image" in modalities or "video" in modalities:
        visual_semantic = retriever.retrieve_visual_evidence(client, claim["document"], n_results, filters)
        candidate_lists["visual_semantic"] = visual_semantic

    fused = rank_fusion.fuse_candidates(candidate_lists, max_results=n_results)
    kept = [e for e in fused if e["trace"]["exclusion_reason"] is None]
    split = _split_by_stance(kept, claim_id)

    result = {
        "claim": claim, "claim_category": claim_category, "modalities_searched": modalities,
        "fused_candidates": fused, "supports": split["supports"], "challenges": split["challenges"],
        "context": split["context"], "generation": None, "validation": None,
    }

    if generate:
        evidence_items = kept
        response = gemini_generator.generate_answer(
            question=f"Explain the evidence behind the fusion classification for this claim.",
            evidence_items=evidence_items, claim_text=claim["document"],
            fusion_label=meta["fusion_label"], business_facing_label=meta.get("business_facing_label"),
        )
        validation = citation_validator.validate_response(response, evidence_items, true_fusion_label=meta["fusion_label"])
        result["generation"] = response
        result["validation"] = validation

    return result


def answer_question(question: str, n_results: int = 10, filters: Optional[Dict[str, str]] = None, generate: bool = True) -> dict:
    client = get_client()
    intents = query_router.classify_intents(question)
    modalities = query_router.route_modalities_for_intents(intents)

    candidate_lists = {}
    candidate_lists["text_semantic"] = retriever.retrieve_text_evidence(client, question, n_results, filters)
    if "image" in modalities or "video" in modalities:
        candidate_lists["visual_semantic"] = retriever.retrieve_visual_evidence(client, question, n_results, filters)

    fused = rank_fusion.fuse_candidates(candidate_lists, max_results=n_results)
    kept = [e for e in fused if e["trace"]["exclusion_reason"] is None]

    result = {
        "question": question, "intents": intents, "modalities_searched": modalities,
        "fused_candidates": fused, "evidence": kept, "generation": None, "validation": None,
    }

    if generate:
        response = gemini_generator.generate_answer(question=question, evidence_items=kept)
        validation = citation_validator.validate_response(response, kept, true_fusion_label=None)
        result["generation"] = response
        result["validation"] = validation

    return result
