"""Tests for src/fusion/image_evidence.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.image_evidence import build_image_evidence_unit, visual_gate, visual_groundability_tier

CONFIG = {
    "visual_groundability": {
        "claim_category_groundability": {
            "materials": "conditionally_visually_groundable",
            "labour": "not_visually_groundable",
            "emissions": "not_visually_groundable",
            "packaging": "potentially_visually_groundable",
        }
    },
    "relevance_thresholds": {"image_relevance_min": 0.27},
}


def test_image_cannot_validate_labour_claim_regardless_of_score():
    gate = visual_gate("labour", relevance_score=0.99, threshold=0.27, config=CONFIG)
    assert gate["visual_gate_passed"] is False
    assert gate["visually_groundable"] is False


def test_image_cannot_validate_emissions_claim():
    gate = visual_gate("emissions", relevance_score=0.99, threshold=0.27, config=CONFIG)
    assert gate["visual_gate_passed"] is False


def test_image_gate_passes_for_materials_above_threshold():
    gate = visual_gate("materials", relevance_score=0.3, threshold=0.27, config=CONFIG)
    assert gate["visual_gate_passed"] is True


def test_image_gate_fails_for_materials_below_threshold():
    gate = visual_gate("materials", relevance_score=0.1, threshold=0.27, config=CONFIG)
    assert gate["visual_gate_passed"] is False
    assert gate["visually_groundable"] is True  # groundable category, just below relevance


def test_images_default_to_neutral_context_when_gate_fails():
    unit = build_image_evidence_unit("labour", relevance_score=0.9, match_method="category_gated_semantic_match",
                                      product_category_match=False, config=CONFIG)
    assert unit["stance"] == "neutral_context"
    assert unit["permitted_evidence_role"] == "context_only"


def test_images_only_ever_support_never_challenge():
    """A product photo cannot demonstrate a defect -- images never produce
    stance='challenges'."""
    for relevance in (0.1, 0.3, 0.5, 0.9):
        unit = build_image_evidence_unit("materials", relevance_score=relevance, match_method="category_gated_semantic_match",
                                          product_category_match=False, config=CONFIG)
        assert unit["stance"] in ("supports", "neutral_context")


def test_image_stance_score_is_capped_below_full_confidence():
    unit = build_image_evidence_unit("materials", relevance_score=0.95, match_method="category_gated_semantic_match",
                                      product_category_match=True, config=CONFIG)
    assert unit["stance_score"] <= 0.6


def test_visual_groundability_tier_unknown_category_defaults_not_groundable():
    assert visual_groundability_tier("unknown_category_xyz", CONFIG) == "not_visually_groundable"
