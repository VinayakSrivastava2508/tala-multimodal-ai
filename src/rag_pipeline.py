"""Deprecated pre-Day-3B scaffold. The real multimodal RAG implementation is
`src/rag/` (schemas, corpus_builder, chroma_store, embedders, query_router,
retriever, rank_fusion, gemini_generator, citation_validator, orchestrator),
built and evaluated by `scripts/build_chroma_index.py` and
`scripts/run_multimodal_rag_evaluation.py`. See
`docs/multimodal_rag_architecture.md` for the full design.

This module is kept only as a thin compatibility import so any existing
external reference to `src.rag_pipeline` does not break; do not add new logic
here -- add it to `src/rag/`.
"""

from __future__ import annotations

from src.rag.chroma_store import get_client  # noqa: F401
from src.rag.orchestrator import answer_question, investigate_claim  # noqa: F401
from src.rag.schemas import (  # noqa: F401
    CHROMA_PERSIST_DIR as CHROMA_DIR,
    COLLECTION_CLAIMS,
    COLLECTION_TEXT_EVIDENCE,
    COLLECTION_VISUAL_EVIDENCE,
)
