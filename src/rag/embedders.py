"""Day 3B: embedding functions for the RAG corpus -- thin wrappers that reuse
the exact same local models already used by the Day 1-3A pipelines
(src/text_features.py, src/image_features.py). No new model is introduced."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from src.image_features import get_clip_text_embedding, get_or_compute_clip_embedding
from src.rag.schemas import CLIP_MODEL_NAME, TEXT_EMBEDDING_MODEL
from src.text_features import get_sentence_embeddings


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Sentence-transformers all-MiniLM-L6-v2, cosine-normalised, 384-dim --
    the same model/config used throughout Day 3A text fusion."""
    if not texts:
        return []
    embeddings = get_sentence_embeddings(texts, model_name=TEXT_EMBEDDING_MODEL, show_progress=False)
    return [row.tolist() for row in embeddings]


def embed_query_text(query: str) -> List[float]:
    return embed_texts([query])[0]


def embed_image(local_path: str, file_hash: Optional[str] = None) -> Optional[List[float]]:
    """CLIP ViT-B/32 image embedding, 512-dim, disk-cached by content hash
    (src/image_features.py::get_or_compute_clip_embedding)."""
    path = Path(local_path)
    hash_key = file_hash or _hash_file(path)
    emb = get_or_compute_clip_embedding(path, hash_key)
    return emb.tolist() if emb is not None else None


def embed_query_image_text(query: str) -> Optional[List[float]]:
    """CLIP text embedding for a natural-language query, so it can be compared
    against CLIP image embeddings in the same space (cross-modal retrieval)."""
    emb = get_clip_text_embedding(query, model_name=CLIP_MODEL_NAME)
    return emb.tolist() if emb is not None else None


def _hash_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
