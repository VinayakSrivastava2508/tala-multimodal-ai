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


# ── Evidence provenance (Executive Cockpit Acceptance Patch, Part D) ───────────
#
# Two explicit, separate concepts instead of one ambiguous combined label:
#   source_origin        -- who published/produced the evidence
#   independence_status  -- whether it corroborates the brand's own claim
#
# official_claim(s), official_video and official_product_image are always
# "Brand official" + "Self-reported" -- unconditionally, regardless of any
# join-derived flags -- because by construction they cannot corroborate
# TALA's own claims independently.
_BRAND_OFFICIAL_UNCONDITIONAL = frozenset({
    "official_claim", "official_claims", "official_video", "official_product_image",
})
# official_reference is "Brand official" + "Self-reported" *unless* a
# genuinely independent underlying source is explicitly verified in the
# evidence-unit join data, per Part D's stated exception for this one type.
_BRAND_OFFICIAL_WITH_INDEPENDENCE_EXCEPTION = frozenset({"official_reference"})
_SOURCE_ORIGIN_MAP = {
    "customer_experience": "Customer",
    "creator_strategy": "Creator",
    "press_reddit_sources": "Press/third party",
    "press": "Press/third party",
    "third_party": "Press/third party",
    "certification": "Certification/assurance source",
    "assurance": "Certification/assurance source",
    "ngo_assessment": "Certification/assurance source",
}


def classify_evidence_provenance(source_type: str, self_reported: bool, independent_source: bool) -> tuple[str, str]:
    """Returns (source_origin, independence_status) for one evidence record.

    source_origin in {"Brand official", "Customer", "Creator",
    "Press/third party", "Certification/assurance source", "Other"}.
    independence_status in {"Self-reported", "Independent",
    "Not independently verified", "Unknown"}.

    Brand-published source types are always Brand official + Self-reported.
    For everything else, independence is read from the existing verified
    self_reported/independent_source flags -- never inferred from keywords,
    and never defaulted to "Independent" just because the source isn't the
    brand itself.
    """
    if not source_type:
        return "Other", "Unknown"
    if source_type in _BRAND_OFFICIAL_UNCONDITIONAL:
        return "Brand official", "Self-reported"
    if source_type in _BRAND_OFFICIAL_WITH_INDEPENDENCE_EXCEPTION:
        return "Brand official", ("Independent" if independent_source else "Self-reported")

    origin = _SOURCE_ORIGIN_MAP.get(source_type, "Other")
    if independent_source:
        independence = "Independent"
    elif self_reported:
        independence = "Self-reported"
    else:
        independence = "Not independently verified"
    return origin, independence
