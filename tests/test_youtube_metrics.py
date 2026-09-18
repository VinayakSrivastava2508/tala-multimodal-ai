"""Tests for Day 2 Task 2B YouTube metadata enrichment and engagement math
(src/collectors/youtube_metrics.py). No live network calls -- all API calls are
mocked via an injectable `get`."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.youtube_metrics import (
    batch_ids,
    compute_engagement_features,
    extract_video_id,
    fetch_channels_metadata,
    fetch_videos_metadata,
    parse_channel_item,
    parse_video_item,
    within_channel_normalisation,
)


# ── video ID extraction ────────────────────────────────────────────────────────

def test_extract_video_id_from_watch_url():
    assert extract_video_id("https://www.youtube.com/watch?v=abc123XYZ") == "abc123XYZ"


def test_extract_video_id_from_watch_url_with_extra_params():
    assert extract_video_id("https://www.youtube.com/watch?v=abc123XYZ&pp=ygUbI2Zvbw") == "abc123XYZ"


def test_extract_video_id_from_shorts_url():
    assert extract_video_id("https://www.youtube.com/shorts/abc123XYZ") == "abc123XYZ"


def test_extract_video_id_from_youtu_be():
    assert extract_video_id("https://youtu.be/abc123XYZ") == "abc123XYZ"


def test_extract_video_id_returns_none_for_non_youtube_url():
    assert extract_video_id("https://www.tiktok.com/@x/video/123") is None


def test_extract_video_id_returns_none_for_empty():
    assert extract_video_id("") is None


# ── batching at 50 ─────────────────────────────────────────────────────────────

def test_batch_ids_splits_at_fifty():
    ids = [f"id{i}" for i in range(120)]
    batches = batch_ids(ids, batch_size=50)
    assert [len(b) for b in batches] == [50, 50, 20]
    assert sum(len(b) for b in batches) == 120


def test_batch_ids_deduplicates_preserving_order():
    batches = batch_ids(["a", "b", "a", "c"], batch_size=50)
    assert batches == [["a", "b", "c"]]


def test_batch_ids_under_limit_single_batch():
    batches = batch_ids(["a", "b", "c"], batch_size=50)
    assert len(batches) == 1


# ── parsing: missing statistics remain missing (never 0) ──────────────────────

def test_parse_video_item_full_statistics():
    item = {
        "id": "abc123", "snippet": {"channelId": "UC1", "channelTitle": "Jane", "title": "T", "description": "D",
                                     "publishedAt": "2024-01-01T00:00:00Z", "categoryId": "22"},
        "statistics": {"viewCount": "1000", "likeCount": "50", "commentCount": "5"},
        "contentDetails": {"duration": "PT5M"},
        "status": {"privacyStatus": "public"},
    }
    record = parse_video_item(item, now_iso="2026-01-01T00:00:00Z")
    assert record["view_count"] == 1000
    assert record["like_count"] == 50
    assert record["comment_count"] == 5
    assert record["api_retrieval_status"] == "success"


def test_parse_video_item_missing_like_count_stays_none_not_zero():
    """A creator can hide the like count -- statistics.likeCount is then simply
    absent from the API response. This must become None, never 0."""
    item = {
        "id": "abc123", "snippet": {"channelId": "UC1"},
        "statistics": {"viewCount": "1000", "commentCount": "5"},  # likeCount absent
        "contentDetails": {}, "status": {},
    }
    record = parse_video_item(item, now_iso="2026-01-01T00:00:00Z")
    assert record["view_count"] == 1000
    assert record["like_count"] is None
    assert record["comment_count"] == 5


def test_parse_video_item_no_statistics_object_at_all():
    item = {"id": "abc123", "snippet": {}, "contentDetails": {}, "status": {}}
    record = parse_video_item(item, now_iso="2026-01-01T00:00:00Z")
    assert record["view_count"] is None
    assert record["statistics_available"] is False


def test_parse_channel_item_hidden_subscriber_count():
    item = {"id": "UC1", "snippet": {"title": "Jane"}, "statistics": {"hiddenSubscriberCount": True, "videoCount": "10"}}
    record = parse_channel_item(item, now_iso="2026-01-01T00:00:00Z")
    assert record["hidden_subscriber_count"] is True
    assert record["channel_subscriber_count_current"] is None


def test_parse_channel_item_visible_subscriber_count():
    item = {"id": "UC1", "snippet": {"title": "Jane"}, "statistics": {"subscriberCount": "5000", "hiddenSubscriberCount": False}}
    record = parse_channel_item(item, now_iso="2026-01-01T00:00:00Z")
    assert record["channel_subscriber_count_current"] == 5000
    assert record["hidden_subscriber_count"] is False


# ── fetch_videos_metadata (mocked API, batching, caching) ─────────────────────

def _mock_videos_response(ids_present):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "items": [
            {"id": vid, "snippet": {"channelId": "UC1", "title": "T", "description": "D"},
             "statistics": {"viewCount": "100", "likeCount": "10", "commentCount": "1"},
             "contentDetails": {"duration": "PT1M"}, "status": {"privacyStatus": "public"}}
            for vid in ids_present
        ]
    }
    return resp


def test_fetch_videos_metadata_marks_missing_video_not_found(tmp_path):
    mock_get = MagicMock(return_value=_mock_videos_response(["vid1"]))  # vid2 absent from response
    results = fetch_videos_metadata(["vid1", "vid2"], api_key="FAKE", get=mock_get, cache_dir=tmp_path)
    assert results["vid1"]["api_retrieval_status"] == "success"
    assert results["vid2"]["api_retrieval_status"] == "not_found"
    assert results["vid2"]["view_count"] is None


def test_fetch_videos_metadata_uses_cache_on_second_call(tmp_path):
    mock_get = MagicMock(return_value=_mock_videos_response(["vid1"]))
    fetch_videos_metadata(["vid1"], api_key="FAKE", get=mock_get, cache_dir=tmp_path)
    assert mock_get.call_count == 1
    fetch_videos_metadata(["vid1"], api_key="FAKE", get=mock_get, cache_dir=tmp_path)
    assert mock_get.call_count == 1  # second call served entirely from cache


def test_fetch_videos_metadata_batches_over_fifty_ids(tmp_path):
    ids = [f"vid{i}" for i in range(75)]
    mock_get = MagicMock(side_effect=lambda url, params, timeout: _mock_videos_response(params["id"].split(",")))
    results = fetch_videos_metadata(ids, api_key="FAKE", get=mock_get, cache_dir=tmp_path)
    assert mock_get.call_count == 2  # 50 + 25
    assert len(results) == 75


def test_fetch_channels_metadata_mocked(tmp_path):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"items": [{"id": "UC1", "snippet": {"title": "Jane"}, "statistics": {"subscriberCount": "500"}}]}
    mock_get = MagicMock(return_value=resp)
    results = fetch_channels_metadata(["UC1"], api_key="FAKE", get=mock_get, cache_dir=tmp_path)
    assert results["UC1"]["channel_subscriber_count_current"] == 500


# ── engagement feature math ────────────────────────────────────────────────────

def test_engagement_count_sums_likes_and_comments():
    out = compute_engagement_features(1000, 50, 5, None)
    assert out["engagement_count"] == 55


def test_engagement_count_none_when_like_missing():
    out = compute_engagement_features(1000, None, 5, None)
    assert out["engagement_count"] is None


def test_zero_view_count_does_not_divide_by_zero():
    out = compute_engagement_features(0, 50, 5, None)
    assert out["interaction_rate_by_views"] is None
    assert out["like_rate_by_views"] is None
    assert out["comment_rate_by_views"] is None


def test_missing_view_count_does_not_divide_by_zero():
    out = compute_engagement_features(None, 50, 5, None)
    assert out["interaction_rate_by_views"] is None


def test_interaction_rate_calculation():
    out = compute_engagement_features(1000, 50, 5, None)
    assert out["interaction_rate_by_views"] == pytest.approx(55 / 1000, rel=1e-6)
    assert out["like_rate_by_views"] == pytest.approx(50 / 1000, rel=1e-6)
    assert out["comment_rate_by_views"] == pytest.approx(5 / 1000, rel=1e-6)


def test_engagement_rate_by_subscribers_requires_positive_subscriber_count():
    out = compute_engagement_features(1000, 50, 5, 0)
    assert out["engagement_rate_by_current_subscribers"] is None
    out2 = compute_engagement_features(1000, 50, 5, 2000)
    assert out2["engagement_rate_by_current_subscribers"] == pytest.approx(55 / 2000, rel=1e-6)


def test_video_age_and_views_per_day():
    out = compute_engagement_features(
        1000, 50, 5, None,
        published_at="2026-01-01T00:00:00+00:00", retrieved_at="2026-01-11T00:00:00+00:00",
    )
    assert out["video_age_days"] == 10
    assert out["views_per_day"] == pytest.approx(100.0, rel=1e-6)


def test_missing_published_at_leaves_age_and_views_per_day_none():
    out = compute_engagement_features(1000, 50, 5, None, published_at="", retrieved_at="2026-01-11T00:00:00+00:00")
    assert out["video_age_days"] is None
    assert out["views_per_day"] is None


# ── within-channel normalisation eligibility ───────────────────────────────────

def test_channels_with_fewer_than_three_posts_excluded():
    records = [
        {"channel_id": "A", "engagement_count": 10}, {"channel_id": "A", "engagement_count": 20},  # only 2 posts
        {"channel_id": "B", "engagement_count": 5}, {"channel_id": "B", "engagement_count": 15}, {"channel_id": "B", "engagement_count": 25},  # 3 posts
    ]
    out = within_channel_normalisation(records, min_posts=3)
    a_rows = [r for r in out if r["channel_id"] == "A"]
    b_rows = [r for r in out if r["channel_id"] == "B"]
    assert all(r["within_channel_normalisation_eligible"] is False for r in a_rows)
    assert all(r["within_channel_engagement_zscore"] is None for r in a_rows)
    assert all(r["within_channel_normalisation_eligible"] is True for r in b_rows)
    assert all(r["within_channel_engagement_zscore"] is not None for r in b_rows)


def test_within_channel_zscore_and_percentile_values():
    records = [
        {"channel_id": "B", "engagement_count": 10}, {"channel_id": "B", "engagement_count": 20},
        {"channel_id": "B", "engagement_count": 30},
    ]
    out = within_channel_normalisation(records, min_posts=3)
    percentiles = sorted(r["within_channel_engagement_percentile"] for r in out)
    assert percentiles == pytest.approx([1 / 3, 2 / 3, 1.0], abs=1e-3)


def test_within_channel_normalisation_never_uses_brand_level_grouping():
    """Two different channels within the same brand must be normalised
    independently -- never pooled as if they were one 'account'."""
    records = [
        {"channel_id": "A", "brand": "TALA", "engagement_count": 1000},
        {"channel_id": "A", "brand": "TALA", "engagement_count": 1000},
        {"channel_id": "A", "brand": "TALA", "engagement_count": 1000},
        {"channel_id": "B", "brand": "TALA", "engagement_count": 5},
    ]
    out = within_channel_normalisation(records, min_posts=3)
    b_row = next(r for r in out if r["channel_id"] == "B")
    assert b_row["within_channel_normalisation_eligible"] is False  # only 1 post on channel B
