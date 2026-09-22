"""Tests for src/rag/chroma_store.py -- collection creation, embedding-dimension
validation, idempotent upserts, and stale-record removal. Uses a throwaway
Chroma directory under tmp_path, never the real data/vector_db/chroma/."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import chroma_store


def _sample_records(n=3):
    return [
        {"id": f"id_{i}", "document": f"doc {i}", "metadata": {"evidence_strength": "strong", "brand": "TALA"}}
        for i in range(n)
    ]


def test_get_client_opens_persistent_client(tmp_path):
    client = chroma_store.get_client(str(tmp_path / "chroma"))
    assert client is not None


def test_get_collection_or_raise_raises_when_missing(tmp_path):
    client = chroma_store.get_client(str(tmp_path / "chroma"))
    with pytest.raises(chroma_store.ChromaUnavailableError):
        chroma_store.get_collection_or_raise(client, "does_not_exist")


def test_validate_embedding_dimension_accepts_correct_dim():
    chroma_store.validate_embedding_dimension([0.0] * 384, "text")
    chroma_store.validate_embedding_dimension([0.0] * 512, "image")


def test_validate_embedding_dimension_rejects_wrong_dim():
    with pytest.raises(ValueError):
        chroma_store.validate_embedding_dimension([0.0] * 10, "text")


def test_upsert_and_count(tmp_path):
    client = chroma_store.get_client(str(tmp_path / "chroma"))
    collection = chroma_store.get_or_create_collection(client, "test_collection")
    records = _sample_records(3)
    embeddings = [[float(i)] * 384 for i in range(3)]
    n = chroma_store.upsert_records(collection, records, embeddings)
    assert n == 3
    assert chroma_store.collection_counts(collection) == 3


def test_upsert_is_idempotent_on_rerun(tmp_path):
    client = chroma_store.get_client(str(tmp_path / "chroma"))
    collection = chroma_store.get_or_create_collection(client, "test_collection")
    records = _sample_records(3)
    embeddings = [[float(i)] * 384 for i in range(3)]
    chroma_store.upsert_records(collection, records, embeddings)
    chroma_store.upsert_records(collection, records, embeddings)  # rerun, same IDs
    assert chroma_store.collection_counts(collection) == 3  # never doubles


def test_upsert_with_updated_content_replaces_not_duplicates(tmp_path):
    client = chroma_store.get_client(str(tmp_path / "chroma"))
    collection = chroma_store.get_or_create_collection(client, "test_collection")
    records = _sample_records(2)
    embeddings = [[0.1] * 384, [0.2] * 384]
    chroma_store.upsert_records(collection, records, embeddings)

    records[0]["document"] = "updated document text"
    chroma_store.upsert_records(collection, [records[0]], [embeddings[0]])
    assert chroma_store.collection_counts(collection) == 2
    fetched = collection.get(ids=["id_0"], include=["documents"])
    assert fetched["documents"][0] == "updated document text"


def test_remove_stale_records_deletes_ids_not_in_authoritative_set(tmp_path):
    client = chroma_store.get_client(str(tmp_path / "chroma"))
    collection = chroma_store.get_or_create_collection(client, "test_collection")
    records = _sample_records(3)
    embeddings = [[0.1] * 384] * 3
    chroma_store.upsert_records(collection, records, embeddings)

    n_removed = chroma_store.remove_stale_records(collection, authoritative_ids={"id_0", "id_1"})
    assert n_removed == 1
    assert chroma_store.collection_counts(collection) == 2


def test_sanitize_metadata_coerces_none_and_lists():
    sanitized = chroma_store._sanitize_metadata({"a": None, "b": ["x", "y"], "c": 1, "d": True})
    assert sanitized["a"] == ""
    assert sanitized["b"] == "x;y"
    assert sanitized["c"] == 1
    assert sanitized["d"] is True


def test_get_all_metadata_returns_id_to_metadata_map(tmp_path):
    client = chroma_store.get_client(str(tmp_path / "chroma"))
    collection = chroma_store.get_or_create_collection(client, "test_collection")
    records = _sample_records(2)
    embeddings = [[0.1] * 384, [0.2] * 384]
    chroma_store.upsert_records(collection, records, embeddings)
    metadata_map = chroma_store.get_all_metadata(collection)
    assert set(metadata_map.keys()) == {"id_0", "id_1"}
