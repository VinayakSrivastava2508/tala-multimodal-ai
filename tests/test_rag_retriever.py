"""Integration tests for src/rag/retriever.py and src/rag/orchestrator.py
against the REAL built ChromaDB index. Skipped if the index has not been
built in this environment (never fabricates a passing result)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chroma_store import ChromaUnavailableError, get_client
from src.rag.schemas import CHROMA_PERSIST_DIR, COLLECTION_CLAIMS, COLLECTION_TEXT_EVIDENCE, COLLECTION_VISUAL_EVIDENCE

INDEX_BUILT = CHROMA_PERSIST_DIR.exists()
pytestmark = pytest.mark.skipif(not INDEX_BUILT, reason="ChromaDB index not built in this environment -- run scripts/build_chroma_index.py")


def test_missing_chroma_database_raises_clear_error(tmp_path):
    from src.rag import chroma_store
    client = chroma_store.get_client(str(tmp_path / "nonexistent"))
    with pytest.raises(ChromaUnavailableError):
        chroma_store.get_collection_or_raise(client, COLLECTION_CLAIMS)


def test_text_retrieval_returns_results():
    from src.rag import retriever
    client = get_client()
    results = retriever.retrieve_text_evidence(client, "durability of the fabric", n_results=5)
    assert len(results) > 0
    assert all(r["collection"] == COLLECTION_TEXT_EVIDENCE for r in results)


def test_visual_retrieval_returns_results():
    from src.rag import retriever
    client = get_client()
    results = retriever.retrieve_visual_evidence(client, "a person wearing leggings", n_results=5)
    assert len(results) > 0
    assert all(r["collection"] == COLLECTION_VISUAL_EVIDENCE for r in results)


def test_video_frame_retrieval_is_distinguishable_from_image_results():
    from src.rag import retriever
    client = get_client()
    results = retriever.retrieve_visual_evidence(client, "product demonstration video", n_results=15)
    modalities = {r["metadata"]["modality"] for r in results}
    assert modalities <= {"image", "video_frame"}


def test_direct_linked_evidence_returns_only_claim_id_matches():
    from src.rag import retriever
    client = get_client()
    results = retriever.direct_linked_evidence(client, "c_OC_0003_01")
    assert len(results) > 0
    for r in results:
        assert "c_OC_0003_01" in str(r["metadata"]["claim_id"]).split(";")


def test_no_protected_characteristic_fields_in_visual_metadata():
    from src.rag.chroma_store import get_all_metadata
    client = get_client()
    collection = client.get_collection(COLLECTION_VISUAL_EVIDENCE)
    metadata_map = get_all_metadata(collection)
    forbidden_substrings = ("race", "ethnicity", "gender", "age_group", "face_embedding")
    for meta in metadata_map.values():
        for key in meta:
            assert not any(f in key.lower() for f in forbidden_substrings)


def test_no_engagement_prediction_fields_anywhere_in_claims_metadata():
    from src.rag.chroma_store import get_all_metadata
    client = get_client()
    collection = client.get_collection(COLLECTION_CLAIMS)
    metadata_map = get_all_metadata(collection)
    forbidden_substrings = ("predicted_engagement", "engagement_score", "predicted_likes", "predicted_views")
    for meta in metadata_map.values():
        for key in meta:
            assert not any(f in key.lower() for f in forbidden_substrings)


# ── orchestrator-level integration (retrieval only, no live Gemini call) ────────

def test_investigate_claim_routes_labour_claim_away_from_visual_modalities():
    from src.rag.orchestrator import investigate_claim
    result = investigate_claim("c_OC_0002_00", n_results=5, generate=False)
    assert "image" not in result["modalities_searched"]
    assert "video" not in result["modalities_searched"]


def test_investigate_claim_materials_claim_can_include_visual_modalities():
    from src.rag.orchestrator import investigate_claim
    result = investigate_claim("c_OC_0003_01", n_results=5, generate=False)
    assert "image" in result["modalities_searched"]
    assert "video" in result["modalities_searched"]


def test_investigate_unknown_claim_raises_value_error():
    from src.rag.orchestrator import investigate_claim
    with pytest.raises(ValueError):
        investigate_claim("does_not_exist_claim_id", generate=False)


def test_answer_question_returns_streamlit_facing_schema():
    from src.rag.orchestrator import answer_question
    result = answer_question("What do customer reviews say about sizing?", n_results=5, generate=False)
    for key in ("question", "intents", "modalities_searched", "fused_candidates", "evidence"):
        assert key in result
    for e in result["evidence"]:
        assert "id" in e and "metadata" in e
