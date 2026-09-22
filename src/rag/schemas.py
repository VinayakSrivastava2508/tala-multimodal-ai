"""Day 3B: shared constants, deterministic ID builders, and evidence gates for
the multimodal RAG corpus. Every gate here reuses the Day 3A authoritative
rules (configs/fusion_rules.yaml) rather than inventing new thresholds."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PERSIST_DIR = PROJECT_ROOT / "data" / "vector_db" / "chroma"

COLLECTION_CLAIMS = "tala_claims"
COLLECTION_TEXT_EVIDENCE = "tala_text_evidence"
COLLECTION_VISUAL_EVIDENCE = "tala_visual_evidence"
ALL_COLLECTIONS = (COLLECTION_CLAIMS, COLLECTION_TEXT_EVIDENCE, COLLECTION_VISUAL_EVIDENCE)

TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"       # sentence-transformers, 384-dim, reused from src/text_features.py
TEXT_EMBEDDING_DIM = 384
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"  # reused from src/image_features.py
CLIP_EMBEDDING_DIM = 512

# Evidence rows below "medium" are never ingested (Day 3B Part 6 gate). No row
# in the current authoritative corpus is actually "weak" or "unusable", but
# the gate is enforced regardless of current data, not assumed absent.
ELIGIBLE_EVIDENCE_STRENGTHS = ("strong", "medium")

VISUALLY_GROUNDABLE_TIERS = ("potentially_visually_groundable", "conditionally_visually_groundable")


def load_fusion_rules() -> dict:
    with open(PROJECT_ROOT / "configs" / "fusion_rules.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def claim_category_groundability(claim_category: str, fusion_rules: Optional[dict] = None) -> str:
    """Returns one of potentially/conditionally/not_visually_groundable for a
    claim category, reading the authoritative Day 3A mapping -- never a
    locally-invented rule."""
    rules = fusion_rules or load_fusion_rules()
    mapping = rules["visual_groundability"]["claim_category_groundability"]
    return mapping.get(claim_category, "not_visually_groundable")


def is_visually_groundable_category(claim_category: str, fusion_rules: Optional[dict] = None) -> bool:
    return claim_category_groundability(claim_category, fusion_rules) in VISUALLY_GROUNDABLE_TIERS


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Deterministic ID builders -- stable across reruns, no random components ────

def claim_id_for(claim_id: str) -> str:
    return f"claim::{claim_id}"


def text_evidence_id_for(source_type: str, raw_id: str) -> str:
    return f"text::{source_type}::{raw_id}"


def visual_evidence_id_for(modality: str, raw_id: str) -> str:
    return f"visual::{modality}::{raw_id}"
