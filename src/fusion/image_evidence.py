"""Day 3A Part 9: image evidence pipeline.

Reuses existing CLIP-based claim-image matches from
data/processed/claim_visual_evidence_matches.csv (built in
scripts/build_claim_multimodal_candidates.py) rather than recomputing
embeddings -- this module's job is to apply the fusion-specific visual-
groundability gate, evidence-role assignment, and (very limited) stance
contribution on top of that already-computed relevance signal.

No face detection, no protected-characteristic inference, no retained face
embeddings -- this module never touches raw pixel data at all.
"""

from __future__ import annotations

from typing import Dict


def visual_groundability_tier(claim_category: str, config: dict) -> str:
    """Return 'potentially_visually_groundable', 'conditionally_visually_groundable',
    or 'not_visually_groundable' for this claim category (Part 7)."""
    mapping = config["visual_groundability"]["claim_category_groundability"]
    return mapping.get(claim_category, "not_visually_groundable")


def visual_gate(claim_category: str, relevance_score: float, threshold: float, config: dict) -> Dict:
    """Part 6 rule 6/7: image evidence may contribute stance only for directly
    visually groundable claims, and never for labour/emissions/supplier/ethics/
    certification/long-term-durability/wash-performance claims regardless of
    similarity score."""
    tier = visual_groundability_tier(claim_category, config)
    if tier == "not_visually_groundable":
        return {"visually_groundable": False, "visual_gate_passed": False,
                "visual_gate_reason": f"claim_category '{claim_category}' is not visually groundable"}
    if relevance_score < threshold:
        return {"visually_groundable": True, "visual_gate_passed": False,
                "visual_gate_reason": f"relevance {relevance_score} below threshold {threshold}"}
    return {"visually_groundable": True, "visual_gate_passed": True,
            "visual_gate_reason": f"claim_category '{claim_category}' is {tier} and relevance clears threshold"}


def build_image_evidence_unit(
    claim_category: str, relevance_score: float, match_method: str,
    product_category_match: bool, config: dict,
) -> Dict:
    """Returns the image-specific fields for one claim-image evidence unit.
    Images DEFAULT to stance=neutral_context (Part 6 rule 5) -- they only ever
    escalate to 'supports' (never 'challenges': a product photo cannot show
    absence of a defect) when the visual gate passes, and always at reduced
    confidence relative to text evidence."""
    threshold = config["relevance_thresholds"]["image_relevance_min"]
    gate = visual_gate(claim_category, relevance_score, threshold, config)

    if gate["visual_gate_passed"]:
        stance = "supports"
        stance_score = round(min(relevance_score, 0.6), 4)  # capped -- visual support is inherently limited
        permitted_evidence_role = "limited_support_context"
    else:
        stance = "neutral_context"
        stance_score = 0.0
        permitted_evidence_role = "context_only"

    return {
        "visually_groundable": gate["visually_groundable"],
        "visual_gate_passed": gate["visual_gate_passed"],
        "visual_gate_reason": gate["visual_gate_reason"],
        "stance": stance,
        "stance_score": stance_score,
        "permitted_evidence_role": permitted_evidence_role,
        "product_category_match": product_category_match,
        "match_method": match_method,
    }
