"""Tests for schema routing and semantic validation in src/validation.py.

Covers the Day 2 schema-routing fix: Day 2 files (RAG corpora, enriched interim
files) must route to their own explicit schemas rather than being misrouted to
similarly-named Day 1 candidate schemas, and the new semantic checks must catch
real evidence-quality violations without weakening any Day 1 behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import guess_schema, validate


# ── Filename routing ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "filename,expected_schema",
    [
        ("official_claims_corpus.csv", "official_claims_corpus"),
        ("customer_experience_corpus.csv", "customer_experience_corpus"),
        ("creator_strategy_corpus.csv", "creator_strategy_corpus"),
        ("creator_posts_enriched.csv", "creator_posts_enriched"),
        ("platform_strategy_enriched.csv", "platform_strategy_enriched"),
        ("hydrated_sources.csv", "hydrated_sources"),
    ],
)
def test_day2_filenames_route_to_day2_schemas(filename, expected_schema):
    assert guess_schema(filename) == expected_schema


def test_official_claims_corpus_does_not_route_to_official_claims():
    """The bug this fix addresses: a Day2 corpus filename must never fall through
    to the similarly-named Day1 candidate schema via prefix matching."""
    assert guess_schema("official_claims_corpus.csv") != "official_claims"


def test_creator_posts_enriched_does_not_route_to_creator_posts():
    assert guess_schema("creator_posts_enriched.csv") != "creator_posts"


@pytest.mark.parametrize(
    "filename,expected_schema",
    [
        ("official_claims_candidates.csv", "official_claims"),
        ("creator_posts_candidates.csv", "creator_posts"),
        ("customer_reviews_candidates.csv", "customer_reviews"),
        ("competitor_platforms_candidates.csv", "competitor_platforms"),
        ("press_reddit_sources_candidates.csv", "press_reddit_sources"),
        ("social_posts_template.csv", "social_posts"),
        ("claim_assessments_template.csv", "claim_assessments"),
    ],
)
def test_day1_filenames_still_route_via_legacy_prefix_fallback(filename, expected_schema):
    """Day 1 routing must be unchanged by the new exact-match precedence step."""
    assert guess_schema(filename) == expected_schema


def test_unrecognised_filename_returns_none():
    assert guess_schema("some_unrelated_export.csv") is None


# ── RAG corpus semantic validation ────────────────────────────────────────────

def _base_corpus_row(**overrides) -> dict:
    row = {
        "document_id": "oc_0001",
        "corpus": "official_claims",
        "brand": "TALA",
        "source_platform": "wearetala.com",
        "source_url": "https://www.wearetala.com/pages/purpose",
        "document_title": "Purpose",
        "publication_date": "2026-01-01",
        "extracted_text": "TALA uses recycled materials across its core ranges.",
        "evidence_strength": "strong",
        "rag_usable": True,
        "retrieved_at": "2026-09-18T00:00:00+00:00",
        "provenance_note": "claim_category=materials; extraction=trafilatura; strength_source=source_page",
    }
    row.update(overrides)
    return row


def test_valid_official_claims_corpus_row_passes():
    df = pd.DataFrame([_base_corpus_row()])
    passed, errors = validate(df, "official_claims_corpus")
    assert passed, errors


def test_weak_rag_usable_true_fails():
    df = pd.DataFrame([_base_corpus_row(evidence_strength="weak", rag_usable=True)])
    passed, errors = validate(df, "official_claims_corpus")
    assert not passed
    assert any("weak/unusable but rag_usable=True" in e for e in errors)


def test_unusable_rag_usable_true_fails():
    df = pd.DataFrame([_base_corpus_row(evidence_strength="unusable", rag_usable=True)])
    passed, errors = validate(df, "official_claims_corpus")
    assert not passed


def test_duplicate_document_id_fails():
    df = pd.DataFrame([_base_corpus_row(document_id="oc_0001"), _base_corpus_row(document_id="oc_0001")])
    passed, errors = validate(df, "official_claims_corpus")
    assert not passed
    assert any(e.startswith("DUPLICATE document_id") for e in errors)


def test_blank_document_id_fails():
    df = pd.DataFrame([_base_corpus_row(document_id="")])
    passed, errors = validate(df, "official_claims_corpus")
    assert not passed
    assert any("document_id blank" in e for e in errors)


def test_non_url_source_url_fails():
    df = pd.DataFrame([_base_corpus_row(source_url="wearetala.com/pages/purpose")])
    passed, errors = validate(df, "official_claims_corpus")
    assert not passed
    assert any("not URL-like" in e for e in errors)


def test_extracted_text_empty_while_rag_usable_true_fails():
    df = pd.DataFrame([_base_corpus_row(extracted_text="")])
    passed, errors = validate(df, "official_claims_corpus")
    assert not passed
    assert any("extracted_text empty while rag_usable=True" in e for e in errors)


def test_wrong_corpus_name_fails_via_generic_values_check():
    df = pd.DataFrame([_base_corpus_row(corpus="customer_experience")])
    passed, errors = validate(df, "official_claims_corpus")
    assert not passed
    assert any("INVALID VALUES in 'corpus'" in e for e in errors)


def test_invalid_evidence_strength_value_fails():
    df = pd.DataFrame([_base_corpus_row(evidence_strength="very_strong", rag_usable=False)])
    passed, errors = validate(df, "official_claims_corpus")
    assert not passed
    assert any("INVALID VALUES in 'evidence_strength'" in e for e in errors)


def test_strong_claim_without_source_page_provenance_fails():
    """A short extracted sentence must not be silently scored 'strong' without
    provenance tying that score to a successfully hydrated source page — this is
    the guard against reverting to length-based per-sentence scoring."""
    df = pd.DataFrame([_base_corpus_row(provenance_note="claim_category=materials; extraction=trafilatura")])
    passed, errors = validate(df, "official_claims_corpus")
    assert not passed
    assert any("without provenance tying that strength" in e for e in errors)


def test_customer_experience_corpus_valid_row_passes():
    row = _base_corpus_row(
        document_id="ce_rev_0001",
        corpus="customer_experience",
        source_platform="trustpilot.com",
        source_url="https://www.trustpilot.com/review/wearetala.com",
        provenance_note="category=quality; sentiment=positive",
    )
    df = pd.DataFrame([row])
    passed, errors = validate(df, "customer_experience_corpus")
    assert passed, errors


def test_creator_strategy_corpus_weak_row_with_no_text_is_allowed_when_not_rag_usable():
    """A weak, non-rag-usable row with genuinely empty extracted_text (e.g. an empty
    caption) must not be forced to fail merely for being short."""
    row = _base_corpus_row(
        document_id="cs_0001",
        corpus="creator_strategy",
        source_platform="youtube",
        source_url="https://www.youtube.com/watch?v=abc123",
        extracted_text="",
        evidence_strength="weak",
        rag_usable=False,
        provenance_note="partnership_type=organic; source=ddg_search",
    )
    df = pd.DataFrame([row])
    passed, errors = validate(df, "creator_strategy_corpus")
    assert passed, errors


# ── creator_posts_enriched semantic validation ────────────────────────────────

def _base_creator_row(**overrides) -> dict:
    row = {
        "brand": "TALA",
        "platform": "youtube",
        "source_url": "https://www.youtube.com/watch?v=abc123",
        "post_url": "https://www.youtube.com/watch?v=abc123",
        "direct_post_id": "abc123",
        "creator_handle": "youtube.com/channel/xyz",
        "creator_identity_confidence": "high",
        "url_validation_status": "valid",
        "brand_link_verified": True,
        "partnership_type": "organic",
        "usable_as_creator_post": True,
        "evidence_strength": "medium",
    }
    row.update(overrides)
    return row


def test_valid_verified_creator_post_passes():
    df = pd.DataFrame([_base_creator_row()])
    passed, errors = validate(df, "creator_posts_enriched")
    assert passed, errors


def test_verified_creator_post_with_non_social_url_fails():
    df = pd.DataFrame([_base_creator_row(
        post_url=None,
        source_url="https://www.instagram.com/popular/oner-active-ambassador/",
    )])
    passed, errors = validate(df, "creator_posts_enriched")
    assert not passed
    assert any("without a direct Instagram/TikTok/YouTube post URL" in e for e in errors)


def test_verified_creator_post_on_search_result_url_fails():
    df = pd.DataFrame([_base_creator_row(
        post_url=None,
        source_url="https://duckduckgo.com/html/?q=TALA+activewear+review",
    )])
    passed, errors = validate(df, "creator_posts_enriched")
    assert not passed
    assert any("search-engine result URL" in e for e in errors)


def test_verified_creator_post_missing_handle_fails():
    df = pd.DataFrame([_base_creator_row(creator_handle=None)])
    passed, errors = validate(df, "creator_posts_enriched")
    assert not passed
    assert any("creator_handle blank" in e for e in errors)


def test_unverified_creator_post_missing_handle_does_not_fail():
    """usable_as_creator_post=False rows are not held to the verified-post rules."""
    df = pd.DataFrame([_base_creator_row(usable_as_creator_post=False, creator_handle=None)])
    passed, errors = validate(df, "creator_posts_enriched")
    assert passed, errors


def test_missing_follower_and_engagement_metrics_do_not_fail():
    """The schema deliberately has no required follower/like/view/comment fields —
    those columns aren't even part of creator_posts_enriched's required set."""
    df = pd.DataFrame([_base_creator_row()])
    # sanity: none of these optional metric columns are present at all
    for col in ("follower_count", "like_count", "view_count", "comment_count"):
        assert col not in df.columns
    passed, errors = validate(df, "creator_posts_enriched")
    assert passed, errors


def test_invalid_partnership_type_fails():
    df = pd.DataFrame([_base_creator_row(partnership_type="sponsored_giveaway")])
    passed, errors = validate(df, "creator_posts_enriched")
    assert not passed
    assert any("INVALID VALUES in 'partnership_type'" in e for e in errors)
