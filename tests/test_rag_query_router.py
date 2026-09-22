"""Tests for src/rag/query_router.py -- deterministic intent classification
and modality routing (no LLM secretly decides what is searched)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.query_router import (
    classify_intents, groundability_tier_for_claim, route_modalities_for_claim, route_modalities_for_intents,
)

FUSION_RULES = {
    "visual_groundability": {
        "claim_category_groundability": {
            "materials": "conditionally_visually_groundable", "labour": "not_visually_groundable",
            "manufacturing": "not_visually_groundable", "packaging": "potentially_visually_groundable",
            "emissions": "not_visually_groundable", "circularity": "not_visually_groundable", "other": "not_visually_groundable",
        }
    }
}


def test_classify_intents_is_deterministic():
    a = classify_intents("What evidence challenges TALA's durability claims?")
    b = classify_intents("What evidence challenges TALA's durability claims?")
    assert a == b


def test_durability_question_detected():
    intents = classify_intents("What evidence challenges TALA's durability claims?")
    assert "durability_care" in intents


def test_sizing_question_detected():
    intents = classify_intents("What do customer reviews say about sizing?")
    assert "fit_sizing" in intents
    assert "customer_experience" in intents


def test_competitor_question_detected():
    intents = classify_intents("How does TALA's creator partnership intent differ from Adanola?")
    assert "competitor_comparison" in intents
    assert "creator_strategy" in intents


def test_evidence_gap_question_detected():
    intents = classify_intents("Which responsibility claims lack independent validation?")
    assert "evidence_gap_analysis" in intents
    assert "sustainability_responsibility" in intents


def test_unclassifiable_question_defaults_to_claim_validation():
    intents = classify_intents("asdkjaslkdjalksjd")
    assert intents == ["claim_validation"]


def test_route_modalities_always_includes_text():
    for intents in ([], ["claim_validation"], ["customer_experience"]):
        assert "text" in route_modalities_for_intents(intents or ["claim_validation"])


def test_route_modalities_adds_visual_for_eligible_intents():
    modalities = route_modalities_for_intents(["fit_sizing"])
    assert "image" in modalities
    assert "video" in modalities


def test_route_modalities_excludes_visual_for_creator_only_intent():
    modalities = route_modalities_for_intents(["creator_strategy"])
    assert "image" not in modalities
    assert "video" not in modalities


# ── claim-category-based routing (protects labour/sustainability claims) ───────

def test_materials_claim_routes_to_image_and_video():
    modalities = route_modalities_for_claim("materials", FUSION_RULES)
    assert "image" in modalities and "video" in modalities


def test_packaging_claim_routes_to_image_and_video():
    modalities = route_modalities_for_claim("packaging", FUSION_RULES)
    assert "image" in modalities and "video" in modalities


def test_labour_claim_never_routes_to_image_or_video():
    modalities = route_modalities_for_claim("labour", FUSION_RULES)
    assert "image" not in modalities
    assert "video" not in modalities
    assert "text" in modalities and "reference" in modalities


def test_emissions_claim_never_routes_to_image_or_video():
    modalities = route_modalities_for_claim("emissions", FUSION_RULES)
    assert "image" not in modalities and "video" not in modalities


def test_manufacturing_and_circularity_never_route_to_visual():
    for category in ("manufacturing", "circularity", "other"):
        modalities = route_modalities_for_claim(category, FUSION_RULES)
        assert "image" not in modalities and "video" not in modalities


def test_groundability_tier_helper_matches_config():
    assert groundability_tier_for_claim("materials", FUSION_RULES) == "conditionally_visually_groundable"
    assert groundability_tier_for_claim("labour", FUSION_RULES) == "not_visually_groundable"
