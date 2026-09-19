"""Day 3A Part 10: video evidence pipeline.

Reuses existing CLIP-based claim-video matches from
data/processed/claim_visual_evidence_matches.csv and the genuine temporal
features in data/processed/video_level_features.csv. Frame relevance,
temporal relevance, and transcript relevance are tracked as SEPARATE signals
-- only the frame-relevance signal currently exists for this project's video
pool (no transcripts have been recovered; see docs/day2_6b_...md), so
transcript fields are honestly None/empty rather than fabricated.

Part 6 rule 10 / Part 10: a shared brand-content gallery video (recognisable
here by a semicolon-joined, multi-category `product_category` field -- see
src/collectors/product_pages.py's discovery) must remain contextual unless the
claim-to-product/category match is independently defensible. Ordinary product
demonstration footage cannot establish long-term quality, durability, or
responsibility claims by itself, regardless of visual similarity score.
"""

from __future__ import annotations

from typing import Dict, Optional

from src.fusion.image_evidence import visual_gate


def is_shared_gallery_video(product_category_field: str) -> bool:
    """A semicolon-joined product_category (e.g.
    'accessories;dresses_or_lifestyle;leggings;...') means this video was
    discovered as a shared brand-content gallery item featured on many
    product pages across many categories -- not tied to one specific product."""
    return ";" in str(product_category_field)


def build_video_evidence_unit(
    claim_category: str, relevance_score: float, match_method: str,
    video_product_category_field: str, has_transcript: bool, transcript_text: Optional[str],
    config: dict,
) -> Dict:
    """Returns the video-specific fields for one claim-video evidence unit."""
    threshold = config["relevance_thresholds"]["video_relevance_min"]
    gate = visual_gate(claim_category, relevance_score, threshold, config)
    is_shared = is_shared_gallery_video(video_product_category_field)
    is_exact_product_tier = match_method in ("exact_product", "exact_product_handle")

    # Rule 10: a shared gallery video can only escalate past context if the
    # match was via an exact product/handle tier -- category-gated semantic
    # matching alone is not a "verified product/category match" for a video
    # that appears on nearly every product page regardless of content.
    defensible_product_match = is_exact_product_tier or not is_shared

    frame_relevance = relevance_score  # the only relevance signal this project's video pool has
    temporal_relevance = None  # no claim-specific temporal-feature relevance signal implemented
    transcript_relevance = None
    transcript_stance = "not_applicable"

    if has_transcript and transcript_text:
        # Transcript evidence would route through the same rule-based stance
        # classifier as text evidence -- not implemented here because no TALA
        # video in this project's pool has a transcript (documented
        # limitation, not fabricated).
        transcript_relevance = None
        transcript_stance = "unclear"

    if gate["visual_gate_passed"] and defensible_product_match:
        stance = "supports"
        stance_score = round(min(relevance_score, 0.55), 4)  # capped below image's cap -- video is contextual by default
        contextual_only = False
        permitted_evidence_role = "limited_support_context"
    else:
        stance = "neutral_context"
        stance_score = 0.0
        contextual_only = True
        permitted_evidence_role = "context_only"
        if is_shared and gate["visual_gate_passed"] and not is_exact_product_tier:
            gate["visual_gate_reason"] += "; shared brand-content gallery video without exact product match -- kept contextual"

    return {
        "visually_groundable": gate["visually_groundable"],
        "visual_gate_passed": gate["visual_gate_passed"],
        "visual_gate_reason": gate["visual_gate_reason"],
        "is_shared_gallery_video": is_shared,
        "defensible_product_match": defensible_product_match,
        "frame_relevance": frame_relevance,
        "temporal_relevance": temporal_relevance,
        "transcript_relevance": transcript_relevance,
        "transcript_stance": transcript_stance,
        "contextual_only": contextual_only,
        "stance": stance,
        "stance_score": stance_score,
        "permitted_evidence_role": permitted_evidence_role,
        "match_method": match_method,
    }
