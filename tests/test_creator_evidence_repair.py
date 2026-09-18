"""Tests for Day 2 Task 1B creator-evidence repair: URL classification, identity
resolution, brand-link verification, partnership classification, evidence-strength
rules, and the repair script's row-level and dedup logic.

No live network calls: fetch_youtube_oembed's `get` parameter is always a mock here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.creator_identity import (
    classify_partnership,
    classify_url,
    compute_creator_evidence_strength,
    extract_tiktok_handle,
    fetch_youtube_oembed,
    resolve_youtube_identity,
    verify_brand_link,
)

from scripts.repair_day2_creator_evidence import dedupe, repair_row


# ── URL classification ────────────────────────────────────────────────────────

def test_tiktok_direct_video_url_classified_valid():
    info = classify_url("https://www.tiktok.com/@lilypebbles/video/7331658061460933920")
    assert info["url_validation_status"] == "valid"
    assert info["url_class"] == "tiktok_video"
    assert info["direct_post_id"] == "7331658061460933920"


def test_youtube_watch_url_classified_valid():
    info = classify_url("https://www.youtube.com/watch?v=abc123XYZ")
    assert info["url_validation_status"] == "valid"
    assert info["direct_post_id"] == "abc123XYZ"


def test_instagram_post_shortcode_extraction():
    info = classify_url("https://www.instagram.com/p/Cabc123/")
    assert info["url_validation_status"] == "valid"
    assert info["url_class"] == "instagram_p"
    assert info["direct_post_id"] == "Cabc123"


def test_instagram_reel_shortcode_extraction():
    info = classify_url("https://www.instagram.com/reel/Cxyz789/")
    assert info["url_validation_status"] == "valid"
    assert info["direct_post_id"] == "Cxyz789"


def test_instagram_programme_url_rejected():
    info = classify_url("https://www.instagram.com/popular/oner-active-ambassador/")
    assert info["url_validation_status"] == "invalid"
    assert info["invalidation_reason"] == "instagram_programme_or_campaign_page"


def test_instagram_profile_page_rejected():
    info = classify_url("https://www.instagram.com/wearetala/")
    assert info["url_validation_status"] == "invalid"
    assert info["invalidation_reason"] == "instagram_non_post_page"


def test_youtube_search_result_rejected():
    info = classify_url("https://www.youtube.com/results?search_query=tala+haul")
    assert info["url_validation_status"] == "invalid"


def test_search_engine_result_rejected():
    info = classify_url("https://duckduckgo.com/html/?q=tala+activewear")
    assert info["url_validation_status"] == "invalid"
    assert info["invalidation_reason"] == "search_engine_result_page"


def test_youtube_channel_page_retained_as_lead_not_valid():
    info = classify_url("https://www.youtube.com/channel/UCabc123")
    assert info["url_validation_status"] == "lead_only"


# ── TikTok handle extraction ──────────────────────────────────────────────────

def test_tiktok_handle_extracted_from_url():
    handle = extract_tiktok_handle("https://www.tiktok.com/@lilypebbles/video/7331658061460933920")
    assert handle == "lilypebbles"


def test_tiktok_handle_none_for_non_video_url():
    assert extract_tiktok_handle("https://www.tiktok.com/@lilypebbles") is None


# ── YouTube oEmbed (mocked, no live network) ──────────────────────────────────

def _mock_oembed_response(author_name="Jane Creator", author_url="https://www.youtube.com/@janecreator"):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "title": "My honest TALA review", "author_name": author_name, "author_url": author_url,
    }
    return resp


def test_youtube_oembed_author_parsed_from_mocked_response(tmp_path):
    mock_get = MagicMock(return_value=_mock_oembed_response())
    data = fetch_youtube_oembed(
        "https://www.youtube.com/watch?v=abc123", get=mock_get, cache_dir=tmp_path, delay=0,
    )
    assert data["author_name"] == "Jane Creator"
    mock_get.assert_called_once()


def test_resolve_youtube_identity_uses_oembed_author(tmp_path):
    mock_get = MagicMock(return_value=_mock_oembed_response(author_name="Jane Creator"))
    identity = resolve_youtube_identity(
        "https://www.youtube.com/watch?v=abc123", get=mock_get, cache_dir=tmp_path, delay=0,
    )
    assert identity["creator_name"] == "Jane Creator"
    assert identity["creator_handle"] == "janecreator"
    assert identity["creator_identity_confidence"] == "high"
    assert identity["creator_identity_source"] == "youtube_oembed"


def test_resolve_youtube_identity_falls_back_when_oembed_fails(tmp_path):
    resp = MagicMock()
    resp.status_code = 404
    mock_get = MagicMock(return_value=resp)
    identity = resolve_youtube_identity(
        "https://www.youtube.com/watch?v=deadvideo", fallback_channel_name="Existing Channel",
        get=mock_get, cache_dir=tmp_path, delay=0,
    )
    assert identity["creator_identity_source"] == "existing_channel_metadata"
    assert identity["creator_identity_confidence"] == "medium"


def test_resolve_youtube_identity_unresolved_with_no_fallback(tmp_path):
    resp = MagicMock()
    resp.status_code = 404
    mock_get = MagicMock(return_value=resp)
    identity = resolve_youtube_identity(
        "https://www.youtube.com/watch?v=deadvideo", get=mock_get, cache_dir=tmp_path, delay=0,
    )
    assert identity["creator_identity_confidence"] == "unresolved"
    assert identity["creator_handle"] == ""


# ── Brand-link verification ───────────────────────────────────────────────────

def test_brand_link_verified_via_title_match():
    result = verify_brand_link(brand="TALA", video_title="TALA Activewear Try-On Haul | Honest Review")
    assert result["brand_link_verified"] is True
    assert result["brand_link_confidence"] == "high"
    assert result["brand_link_source"] == "video_title"


def test_brand_link_verified_via_hashtag():
    result = verify_brand_link(brand="Adanola", caption_or_description="Big haul today #adanola #tryonhaul")
    assert result["brand_link_verified"] is True
    assert result["brand_link_confidence"] == "high"


def test_inherited_brand_field_alone_does_not_verify_link():
    """The `brand` column comes from the search query that discovered the row, not
    from any evidence in the post itself. verify_brand_link never looks at a `brand`
    field value as text evidence -- only at real content fields."""
    result = verify_brand_link(
        brand="TALA", video_title="", caption_or_description="", channel_name="", creator_name="",
    )
    assert result["brand_link_verified"] is False
    assert result["brand_link_confidence"] == "unresolved"


def test_brand_link_unverified_when_only_in_channel_name():
    """Brand name appearing only in channel_name (likely the brand's own account)
    is flagged, not accepted as third-party evidence."""
    result = verify_brand_link(brand="TALA", channel_name="TALA Official")
    assert result["brand_link_verified"] is False
    assert result["brand_link_confidence"] == "low"


# ── Partnership classification ────────────────────────────────────────────────

def test_partnership_paid_requires_explicit_ad_disclosure():
    result = classify_partnership(caption_or_description="Adanola try on haul #ad #adanola")
    assert result["partnership_type"] == "paid_sponsorship"


def test_partnership_not_paid_from_mere_brand_mention():
    """Wearing/mentioning a brand alone must not be classified as paid sponsorship."""
    result = classify_partnership(caption_or_description="Loving my new TALA leggings, review inside")
    assert result["partnership_type"] != "paid_sponsorship"


def test_partnership_gifted_from_hashtag():
    result = classify_partnership(caption_or_description="Not an ad, I was #gifted these TALA pieces")
    assert result["partnership_type"] == "gifted"


def test_partnership_unclear_when_no_text_available():
    result = classify_partnership(caption_or_description="", video_title="")
    assert result["partnership_type"] == "unclear"
    assert result["partnership_inferred"] is False


# ── Evidence-strength rules ────────────────────────────────────────────────────

def test_evidence_strength_strong_requires_all_four_conditions():
    assert compute_creator_evidence_strength("valid", "high", True, True) == "strong"


def test_evidence_strength_medium_without_content_text():
    assert compute_creator_evidence_strength("valid", "high", True, False) == "medium"


def test_evidence_strength_weak_when_identity_unresolved():
    assert compute_creator_evidence_strength("valid", "unresolved", True, True) == "weak"


def test_evidence_strength_weak_when_brand_link_unverified():
    assert compute_creator_evidence_strength("valid", "high", False, True) == "weak"


def test_evidence_strength_unusable_for_invalid_url():
    assert compute_creator_evidence_strength("invalid", "high", True, True) == "unusable"


def test_evidence_strength_never_blank():
    for status in ("valid", "invalid", "lead_only"):
        for conf in ("high", "medium", "low", "unresolved"):
            for verified in (True, False):
                for content in (True, False):
                    result = compute_creator_evidence_strength(status, conf, verified, content)
                    assert result in ("strong", "medium", "weak", "unusable")


# ── Row-level repair (usable_as_creator_post gating) ───────────────────────────

def _row(**overrides) -> pd.Series:
    base = {
        "brand": "TALA", "platform": "youtube",
        "source_url": "https://www.youtube.com/watch?v=abc123",
        "post_url": "https://www.youtube.com/watch?v=abc123",
        "video_title": "TALA Activewear Try-On Haul", "caption_or_description": "",
        "channel_name": "", "creator_name": "", "personalised_code": "", "disclosure_type": "",
        "creator_handle": "",
    }
    base.update(overrides)
    return pd.Series(base)


def test_unresolved_creator_forces_usable_false(tmp_path):
    resp = MagicMock()
    resp.status_code = 404
    mock_get = MagicMock(return_value=resp)
    row = _row(video_title="", caption_or_description="")  # no brand text at all, no fallback channel
    out = repair_row(row, yt_get=mock_get, yt_cache_dir=tmp_path)
    assert out["creator_identity_confidence"] == "unresolved"
    assert out["usable_as_creator_post"] is False
    assert out["invalidation_reason"] == "creator_identity_unresolved"


def test_unverified_brand_link_forces_usable_false(tmp_path):
    mock_get = MagicMock(return_value=_mock_oembed_response())
    row = _row(video_title="", caption_or_description="")  # identity resolves, but no brand evidence
    out = repair_row(row, yt_get=mock_get, yt_cache_dir=tmp_path)
    assert out["creator_identity_confidence"] == "high"
    assert out["brand_link_verified"] is False
    assert out["usable_as_creator_post"] is False
    assert out["invalidation_reason"] == "brand_link_unverified"


def test_fully_verified_row_is_usable(tmp_path):
    mock_get = MagicMock(return_value=_mock_oembed_response())
    row = _row(video_title="TALA Activewear Try-On Haul | Honest Review")
    out = repair_row(row, yt_get=mock_get, yt_cache_dir=tmp_path)
    assert out["usable_as_creator_post"] is True
    assert out["evidence_strength"] in ("strong", "medium")
    assert out["creator_handle"]


def test_weak_record_cannot_be_usable(tmp_path):
    """A row with a valid URL but only low-confidence signals must never end up
    usable_as_creator_post=True -- only strong/medium may."""
    mock_get = MagicMock(return_value=_mock_oembed_response())
    row = _row(video_title="", caption_or_description="", channel_name="TALA")  # brand only via channel_name -> low
    out = repair_row(row, yt_get=mock_get, yt_cache_dir=tmp_path)
    assert out["evidence_strength"] == "weak"
    assert out["usable_as_creator_post"] is False


def test_instagram_programme_url_never_usable(tmp_path):
    row = _row(
        platform="instagram", source_url="https://www.instagram.com/popular/oner-active-ambassador/",
        post_url="", video_title="Oner Active Ambassador Program",
    )
    out = repair_row(row)
    assert out["url_validation_status"] == "invalid"
    assert out["usable_as_creator_post"] is False
    assert out["evidence_strength"] == "unusable"


# ── Deduplication ──────────────────────────────────────────────────────────────

def test_duplicate_direct_post_ids_are_consolidated():
    df = pd.DataFrame([
        {
            "platform": "youtube", "direct_post_id": "abc123", "evidence_strength": "medium",
            "source_id": "creator_enriched_0001", "creator_handle": "",
        },
        {
            "platform": "youtube", "direct_post_id": "abc123", "evidence_strength": "strong",
            "source_id": "creator_enriched_0044", "creator_handle": "janecreator",
        },
    ])
    out, n_merged, log = dedupe(df)
    assert len(out) == 1
    assert n_merged == 1
    assert out.iloc[0]["evidence_strength"] == "strong"
    assert out.iloc[0]["creator_handle"] == "janecreator"
    assert "creator_enriched_0001" in out.iloc[0]["source_record_ids"]
    assert "creator_enriched_0044" in out.iloc[0]["source_record_ids"]


def test_rows_without_direct_post_id_are_not_deduplicated_together():
    df = pd.DataFrame([
        {"platform": "instagram", "direct_post_id": "", "evidence_strength": "unusable", "source_id": "a"},
        {"platform": "instagram", "direct_post_id": "", "evidence_strength": "unusable", "source_id": "b"},
    ])
    out, n_merged, _ = dedupe(df)
    assert len(out) == 2
    assert n_merged == 0
