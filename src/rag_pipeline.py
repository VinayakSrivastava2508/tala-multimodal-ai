"""Multimodal RAG pipeline: corpus ingestion, vector store, and querying."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

CORPUS_NAMES = ["official_claims", "creator_content", "customer_experience"]


def prepare_corpus_documents(
    df: pd.DataFrame,
    text_column: str,
    metadata_columns: Optional[List[str]] = None,
) -> tuple[List[str], List[dict]]:
    """Extract text documents and metadata dicts from a DataFrame.

    Returns (documents: list[str], metadatas: list[dict]).
    """
    documents = df[text_column].fillna("").tolist()
    if metadata_columns:
        metadatas = df[metadata_columns].fillna("").to_dict(orient="records")
    else:
        metadatas = [{} for _ in documents]
    return documents, metadatas


def build_chroma_collection(
    corpus_name: str,
    documents: List[str],
    metadatas: List[dict],
    ids: Optional[List[str]] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    persist_directory: Optional[str] = None,
) -> object:
    """Build or update a ChromaDB collection for a named corpus.

    corpus_name: one of 'official_claims', 'creator_content', 'customer_experience'.
    Returns the ChromaDB collection object.
    """
    import chromadb  # lazy import
    from chromadb.utils import embedding_functions

    persist_dir = persist_directory or str(CHROMA_DIR)
    client = chromadb.PersistentClient(path=persist_dir)

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )

    collection = client.get_or_create_collection(
        name=corpus_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if ids is None:
        ids = [f"{corpus_name}_{i}" for i in range(len(documents))]

    # Add in batches to avoid memory issues
    batch_size = 128
    for start in range(0, len(documents), batch_size):
        end = min(start + batch_size, len(documents))
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
    print(f"[rag_pipeline] Built collection '{corpus_name}' with {len(documents)} documents.")
    return collection


def query_corpus(
    corpus_name: str,
    query_text: str,
    n_results: int = 5,
    persist_directory: Optional[str] = None,
) -> dict:
    """Query a named ChromaDB corpus and return top-n results.

    Returns a dict with keys: ids, documents, distances, metadatas.
    """
    import chromadb  # lazy import
    from chromadb.utils import embedding_functions

    persist_dir = persist_directory or str(CHROMA_DIR)
    client = chromadb.PersistentClient(path=persist_dir)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name=corpus_name, embedding_function=emb_fn)
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results


def cross_corpus_query(
    query_text: str,
    n_results: int = 3,
    corpora: Optional[List[str]] = None,
    persist_directory: Optional[str] = None,
) -> dict:
    """Query all three corpora and return combined results keyed by corpus name.

    Returns dict mapping corpus_name → query result dict.
    """
    target_corpora = corpora or CORPUS_NAMES
    results = {}
    for corpus in target_corpora:
        try:
            results[corpus] = query_corpus(corpus, query_text, n_results, persist_directory)
        except Exception as exc:
            print(f"[rag_pipeline] Could not query corpus '{corpus}': {exc}")
            results[corpus] = None
    return results


def format_rag_context(query_results: dict, corpus_name: str) -> str:
    """Format retrieved documents from a single corpus into a readable string.

    Returns a newline-separated string of retrieved passages with their source IDs.
    """
    if query_results is None:
        return f"[No results from {corpus_name}]"
    docs = query_results.get("documents", [[]])[0]
    ids = query_results.get("ids", [[]])[0]
    lines = []
    for doc_id, doc in zip(ids, docs):
        lines.append(f"[{doc_id}] {doc}")
    return "\n---\n".join(lines)
