"""Tests for src/fusion/text_evidence.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.text_evidence import (
    extract_passage,
    nli_stance_if_available,
    reliability_score,
    rule_based_stance,
    source_independence_for_corpus,
)

CONFIG = {
    "stance_thresholds": {"supports_polarity_min": 0.15, "challenges_polarity_max": -0.15, "unclear_subjectivity_min": 0.85},
    "evidence_strength_weight": {"strong": 1.0, "medium": 0.7, "weak": 0.4, "unusable": 0.0},
    "source_reliability_tiers": {"independent": 1.0, "customer_reported": 0.85, "self_reported": 0.5, "unclear": 0.4},
}


def test_relevance_below_threshold_never_produces_a_stance():
    """Relevance is a precondition, never itself treated as support (Part 6)."""
    result = rule_based_stance("This is a great product, love it!", relevance_score=0.1, relevance_threshold=0.35, config=CONFIG)
    assert result["stance"] == "not_applicable"


def test_positive_sentiment_above_threshold_supports():
    result = rule_based_stance("This product is excellent and works perfectly, I love it.", relevance_score=0.5, relevance_threshold=0.35, config=CONFIG)
    assert result["stance"] == "supports"


def test_negative_sentiment_above_threshold_challenges():
    result = rule_based_stance("This product is terrible and broke immediately, awful quality.", relevance_score=0.5, relevance_threshold=0.35, config=CONFIG)
    assert result["stance"] == "challenges"


def test_neutral_sentiment_is_neutral_context():
    result = rule_based_stance("The item arrived on Tuesday in a box.", relevance_score=0.5, relevance_threshold=0.35, config=CONFIG)
    assert result["stance"] == "neutral_context"


def test_source_independence_claim_itself_is_self_reported():
    assert source_independence_for_corpus("official_claims") == "self_reported"


def test_source_independence_customer_corpus_is_customer_reported():
    assert source_independence_for_corpus("customer_experience") == "customer_reported"


def test_source_independence_creator_corpus_is_unclear():
    assert source_independence_for_corpus("creator_strategy") == "unclear"


def test_source_independence_certification_registry_is_independent():
    assert source_independence_for_corpus("certification_registry") == "independent"


def test_extract_passage_truncates_long_text():
    long_text = "word " * 200
    passage = extract_passage(long_text, max_chars=50)
    assert len(passage) <= 53  # + "..."


def test_extract_passage_never_exceeds_source_content():
    text = "Short review text."
    assert extract_passage(text) == text


def test_reliability_score_strong_independent_is_highest():
    r = reliability_score("strong", "independent", CONFIG)
    assert r == 1.0


def test_reliability_score_weak_self_reported_is_low():
    r = reliability_score("weak", "self_reported", CONFIG)
    assert r == round(0.4 * 0.5, 4)


def test_nli_returns_none_when_unavailable(monkeypatch):
    import src.fusion.text_evidence as mod
    monkeypatch.setattr(mod, "nli_available", lambda: False)
    assert nli_stance_if_available("claim", "evidence") is None


def test_final_fusion_label_is_never_an_input_to_stance():
    """Structural check: rule_based_stance's signature has no fusion-label
    parameter -- it cannot accept one, by construction."""
    import inspect
    params = inspect.signature(rule_based_stance).parameters
    assert "automated_label" not in params
    assert "fusion_label" not in params
