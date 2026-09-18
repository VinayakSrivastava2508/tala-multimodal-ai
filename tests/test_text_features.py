"""Tests for deterministic text-feature rules (src/text_features.py)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.text_features import (
    classify_content_intent,
    creator_strategy_indicators,
    customer_experience_features,
    general_text_features,
    official_claim_indicators,
)


def test_hashtag_mention_url_counts():
    feats = general_text_features("Loving my #tala kit @wearetala https://x.co/abc so much! Really?")
    assert feats["hashtag_count"] == 1
    assert feats["mention_count"] == 1
    assert feats["url_count"] == 1
    assert feats["exclamation_mark_count"] == 1
    assert feats["question_mark_count"] == 1


def test_lexical_diversity_bounds():
    feats = general_text_features("the the the the")
    assert 0.0 <= feats["lexical_diversity"] <= 1.0
    assert feats["lexical_diversity"] == 0.25


def test_general_text_features_empty_string_is_safe():
    feats = general_text_features("")
    assert feats["word_count"] == 0
    assert feats["lexical_diversity"] == 0.0


def test_disclosure_detection_hashtag_ad():
    result = creator_strategy_indicators("Big haul today #ad #tala")
    assert result["disclosure_indicator"] is True


def test_disclosure_detection_gifted():
    result = creator_strategy_indicators("Not sponsored, I was gifted these")
    assert result["disclosure_indicator"] is True


def test_disclosure_not_flagged_without_marker():
    result = creator_strategy_indicators("Loving these leggings, so comfy")
    assert result["disclosure_indicator"] is False


def test_discount_code_detection():
    result = creator_strategy_indicators("Use my code LUCY10 for 20% off")
    assert result["discount_code_indicator"] is True


def test_discount_code_not_flagged_without_marker():
    result = creator_strategy_indicators("Just a regular haul video")
    assert result["discount_code_indicator"] is False


def test_call_to_action_and_launch_indicators():
    result = creator_strategy_indicators("New drop just launched -- link in bio to shop now")
    assert result["launch_indicator"] is True
    assert result["call_to_action_indicator"] is True


def test_customer_experience_category_term_detection():
    feats = customer_experience_features("The fabric quality is terrible and sizing runs small")
    assert feats["quality_term_frequency"] >= 1
    assert feats["fit_sizing_term_frequency"] >= 1
    assert feats["durability_term_frequency"] == 0


def test_customer_experience_review_polarity_negative():
    feats = customer_experience_features("Terrible awful quality, worst purchase, so disappointed")
    assert feats["review_polarity"] == "negative"


def test_customer_experience_review_polarity_positive():
    feats = customer_experience_features("Amazing quality, love it, highly recommend, excellent")
    assert feats["review_polarity"] == "positive"


def test_official_claim_percentage_and_numerical_indicator():
    result = official_claim_indicators("We use 87% recycled nylon across our range")
    assert result["numerical_claim_indicator"] is True
    assert result["percentage_indicator"] is True


def test_official_claim_certification_and_material_indicator():
    result = official_claim_indicators("Our fabric is GRS certified and made from recycled polyester")
    assert result["certification_indicator"] is True
    assert result["material_indicator"] is True


def test_official_claim_target_year_indicator():
    result = official_claim_indicators("We aim to be net zero by 2030")
    assert result["target_year_indicator"] is True


def test_official_claim_no_false_positive_on_plain_text():
    result = official_claim_indicators("We love making activewear for everyone")
    assert result["numerical_claim_indicator"] is False
    assert result["certification_indicator"] is False


def test_content_intent_conversion_label():
    result = classify_content_intent("Shop now and use my discount code for a sale")
    assert result["intent_weak_label"] == "conversion"
    assert result["requires_human_validation"] is True


def test_content_intent_unclear_for_empty_text():
    result = classify_content_intent("")
    assert result["intent_weak_label"] == "unclear"
    assert result["intent_confidence"] == "low"


def test_content_intent_mixed_when_multiple_categories_match():
    result = classify_content_intent(
        "New collection launching soon, shop now, and it's made from sustainable recycled fabric"
    )
    assert result["intent_weak_label"] == "mixed"
    assert "product_launch" in result["intent_rule_evidence"] or "conversion" in result["intent_rule_evidence"]
