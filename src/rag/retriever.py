"""Day 3B: retrieval against the three ChromaDB collections. Every candidate
carries a full trace record (collection, distance, filter result, inclusion/
exclusion reason) so the Streamlit diagnostic panel and evaluation scripts can
inspect exactly what was searched and why a result was kept or dropped."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.rag import embedders
from src.rag.chroma_store import ChromaUnavailableError, get_all_metadata, get_client, get_collection_or_raise
from src.rag.schemas import COLLECTION_CLAIMS, COLLECTION_TEXT_EVIDENCE, COLLECTION_VISUAL_EVIDENCE


def _build_where(filters: Optional[Dict[str, str]]):
    if not filters:
        return None
    clauses = [{k: v} for k, v in filters.items() if v]
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def query_text_collection(client, collection_name: str, query_embedding: List[float], n_results: int, filters: Optional[Dict[str, str]] = None) -> List[dict]:
    collection = get_collection_or_raise(client, collection_name)
    where = _build_where(filters)
    kwargs = {"query_embeddings": [query_embedding], "n_results": n_results, "include": ["documents", "metadatas", "distances"]}
    if where:
        kwargs["where"] = where
    result = collection.query(**kwargs)
    if not result["ids"] or not result["ids"][0]:
        return []
    candidates = []
    for rank, (cid, doc, meta, dist) in enumerate(zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])):
        candidates.append({
            "id": cid, "collection": collection_name, "document": doc, "metadata": meta,
            "distance": dist, "similarity": max(0.0, 1.0 - dist), "original_rank": rank,
        })
    return candidates


def query_visual_collection(client, query_embedding: List[float], n_results: int, filters: Optional[Dict[str, str]] = None) -> List[dict]:
    collection = get_collection_or_raise(client, COLLECTION_VISUAL_EVIDENCE)
    where = _build_where(filters)
    kwargs = {"query_embeddings": [query_embedding], "n_results": n_results, "include": ["metadatas", "distances"]}
    if where:
        kwargs["where"] = where
    result = collection.query(**kwargs)
    if not result["ids"] or not result["ids"][0]:
        return []
    candidates = []
    for rank, (cid, meta, dist) in enumerate(zip(result["ids"][0], result["metadatas"][0], result["distances"][0])):
        candidates.append({
            "id": cid, "collection": COLLECTION_VISUAL_EVIDENCE, "document": None, "metadata": meta,
            "distance": dist, "similarity": max(0.0, 1.0 - dist), "original_rank": rank,
        })
    return candidates


def direct_linked_evidence(client, claim_id: str, collections: Optional[List[str]] = None) -> List[dict]:
    """Evidence whose metadata.claim_id (semicolon-joined) contains this exact
    claim_id -- the Day 3A fusion's own evidence links, not a similarity
    search. Chroma's `where` has no substring/$contains operator for this
    Chroma version, so this filters in-memory over the (small) collections."""
    collections = collections or [COLLECTION_TEXT_EVIDENCE, COLLECTION_VISUAL_EVIDENCE]
    results = []
    for name in collections:
        collection = get_collection_or_raise(client, name)
        metadata_by_id = get_all_metadata(collection)
        for doc_id, meta in metadata_by_id.items():
            linked = str(meta.get("claim_id", "")).split(";")
            if claim_id in linked:
                results.append({"id": doc_id, "collection": name, "metadata": meta, "distance": 0.0, "similarity": 1.0, "original_rank": 0, "direct_link": True})
    return results


def retrieve_text_evidence(client, query: str, n_results: int, filters: Optional[Dict[str, str]] = None) -> List[dict]:
    embedding = embedders.embed_query_text(query)
    return query_text_collection(client, COLLECTION_TEXT_EVIDENCE, embedding, n_results, filters)


def retrieve_claims(client, query: str, n_results: int, filters: Optional[Dict[str, str]] = None) -> List[dict]:
    embedding = embedders.embed_query_text(query)
    return query_text_collection(client, COLLECTION_CLAIMS, embedding, n_results, filters)


def retrieve_visual_evidence(client, query: str, n_results: int, filters: Optional[Dict[str, str]] = None) -> List[dict]:
    """Cross-modal retrieval: embeds the query with CLIP's own text tower so
    it lives in the same space as the indexed CLIP image/frame embeddings."""
    embedding = embedders.embed_query_image_text(query)
    if embedding is None:
        return []
    return query_visual_collection(client, embedding, n_results, filters)
