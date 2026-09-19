"""Tests for src/fusion/video_evidence.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.video_evidence import build_video_evidence_unit, is_shared_gallery_video

CONFIG = {
    "visual_groundability": {
        "claim_category_groundability": {
            "materials": "conditionally_visually_groundable",
            "labour": "not_visually_groundable",
        }
    },
    "relevance_thresholds": {"video_relevance_min": 0.24},
}


def test_is_shared_gallery_video_detects_semicolon_joined_categories():
    assert is_shared_gallery_video("accessories;leggings;tops") is True
    assert is_shared_gallery_video("leggings") is False


def test_shared_gallery_video_stays_contextual_without_exact_product_match():
    """Part 6 rule 10: generic/shared brand videos cannot be treated as exact
    product evidence without verified product/category matching."""
    unit = build_video_evidence_unit(
        "materials", relevance_score=0.5, match_method="category_gated_semantic_match",
        video_product_category_field="accessories;leggings;tops;shorts", has_transcript=False,
        transcript_text=None, config=CONFIG,
    )
    assert unit["stance"] == "neutral_context"
    assert unit["contextual_only"] is True


def test_exact_product_match_on_shared_gallery_can_escalate():
    unit = build_video_evidence_unit(
        "materials", relevance_score=0.5, match_method="exact_product",
        video_product_category_field="accessories;leggings;tops;shorts", has_transcript=False,
        transcript_text=None, config=CONFIG,
    )
    assert unit["stance"] == "supports"


def test_single_category_video_with_gate_pass_can_support():
    unit = build_video_evidence_unit(
        "materials", relevance_score=0.5, match_method="category_gated_semantic_match",
        video_product_category_field="leggings", has_transcript=False, transcript_text=None, config=CONFIG,
    )
    assert unit["stance"] == "supports"
    assert unit["is_shared_gallery_video"] is False


def test_ordinary_product_video_cannot_establish_labour_claim():
    unit = build_video_evidence_unit(
        "labour", relevance_score=0.99, match_method="exact_product",
        video_product_category_field="leggings", has_transcript=False, transcript_text=None, config=CONFIG,
    )
    assert unit["visual_gate_passed"] is False
    assert unit["stance"] == "neutral_context"


def test_video_never_produces_challenges_stance():
    for relevance in (0.1, 0.5, 0.99):
        unit = build_video_evidence_unit(
            "materials", relevance_score=relevance, match_method="exact_product",
            video_product_category_field="leggings", has_transcript=False, transcript_text=None, config=CONFIG,
        )
        assert unit["stance"] in ("supports", "neutral_context")


def test_no_transcript_reports_unclear_not_fabricated():
    unit = build_video_evidence_unit(
        "materials", relevance_score=0.5, match_method="exact_product",
        video_product_category_field="leggings", has_transcript=False, transcript_text=None, config=CONFIG,
    )
    assert unit["transcript_relevance"] is None
    assert unit["transcript_stance"] == "not_applicable"


def test_video_stance_score_capped_below_image_cap():
    unit = build_video_evidence_unit(
        "materials", relevance_score=0.99, match_method="exact_product",
        video_product_category_field="leggings", has_transcript=False, transcript_text=None, config=CONFIG,
    )
    assert unit["stance_score"] <= 0.55
