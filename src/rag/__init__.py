"""Day 3B: production multimodal RAG system for TALA claim-experience divergence.

ChromaDB (local, persistent) is the only vector store. Gemini is the only
generation model. There is no fallback generator, no backup vector store, and
no engagement-prediction or manual-labelling logic anywhere in this package.
"""
