"""Day 3B: thin, explicit wrapper around the ONE local persistent ChromaDB
instance used by this project. No other vector store, no backup store, no
in-memory fallback -- if Chroma cannot be opened, callers must fail loudly."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.rag.schemas import CHROMA_PERSIST_DIR, CLIP_EMBEDDING_DIM, TEXT_EMBEDDING_DIM


class ChromaUnavailableError(RuntimeError):
    """Raised when the local persistent ChromaDB cannot be opened. Retrieval
    must stop with this error -- it must never be silently swallowed."""


def get_client(persist_dir: Optional[str] = None):
    import chromadb  # lazy import
    path = persist_dir or str(CHROMA_PERSIST_DIR)
    try:
        return chromadb.PersistentClient(path=path)
    except Exception as exc:  # pragma: no cover -- environment-dependent
        raise ChromaUnavailableError(f"Could not open local ChromaDB at {path}: {exc}") from exc


def get_or_create_collection(client, name: str):
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def get_collection_or_raise(client, name: str):
    try:
        return client.get_collection(name=name)
    except Exception as exc:
        raise ChromaUnavailableError(f"Collection '{name}' does not exist -- run scripts/build_chroma_index.py first.") from exc


_EXPECTED_DIM = {"text": TEXT_EMBEDDING_DIM, "image": CLIP_EMBEDDING_DIM, "video_frame": CLIP_EMBEDDING_DIM}


def validate_embedding_dimension(embedding, kind: str) -> None:
    expected = _EXPECTED_DIM.get(kind)
    if expected is None:
        return
    if len(embedding) != expected:
        raise ValueError(f"Embedding dimension mismatch for kind='{kind}': got {len(embedding)}, expected {expected}")


def upsert_records(collection, records: List[dict], embeddings: List[List[float]], batch_size: int = 128) -> int:
    """Upsert (ids, documents, metadatas, embeddings) in batches. Chroma
    upsert-by-id is idempotent by construction: re-upserting the same ID
    replaces the row rather than duplicating it."""
    ids = [r["id"] for r in records]
    documents = [r.get("document") or "" for r in records]
    metadatas = [_sanitize_metadata(r["metadata"]) for r in records]

    n = len(ids)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        collection.upsert(
            ids=ids[start:end], embeddings=embeddings[start:end],
            documents=documents[start:end], metadatas=metadatas[start:end],
        )
    return n


def _sanitize_metadata(metadata: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool -- never None or a
    list. Coerces without silently dropping information."""
    out = {}
    for k, v in metadata.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (list, tuple)):
            out[k] = ";".join(str(x) for x in v)
        else:
            out[k] = v
    return out


def remove_stale_records(collection, authoritative_ids: set) -> int:
    """Deletes any row in the collection whose ID is no longer present in the
    authoritative corpus -- keeps the index from silently accumulating
    records for evidence that has since been excluded or removed upstream."""
    existing = collection.get(include=[])
    existing_ids = set(existing.get("ids", []))
    stale = existing_ids - authoritative_ids
    if stale:
        collection.delete(ids=list(stale))
    return len(stale)


def collection_counts(collection) -> int:
    return collection.count()


def get_all_metadata(collection) -> Dict[str, dict]:
    """Returns {id: metadata} for every row currently in the collection."""
    result = collection.get(include=["metadatas"])
    return dict(zip(result["ids"], result["metadatas"]))
