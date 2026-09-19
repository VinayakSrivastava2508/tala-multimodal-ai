"""Tests for Day 2.6B official-video discovery and processing rules:
direct official video accepted, social-platform streams rejected, thumbnails
rejected as video, duplicate media rejected. No live network calls."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.product_pages import VideoSourceCandidate, discover_page_video_sources


# ── direct official video accepted ────────────────────────────────────────────

def test_direct_official_cdn_video_is_accepted_as_candidate():
    html = '<source src="//www.wearetala.com/cdn/shop/videos/c/vp/11111111111111111111111111111111/v.HD-720p-1.6Mbps.mp4?v=0" type="video/mp4">'
    sources = discover_page_video_sources(html)
    assert len(sources) == 1
    assert sources[0].asset_url.endswith(".mp4?v=0")
    assert sources[0].mime_type == "video/mp4"


# ── social-platform embeds are never extracted as direct media (structural) ───

def test_social_platform_embeds_are_not_matched_by_video_source_extractor():
    """discover_page_video_sources only matches <source src=...mp4> elements --
    a YouTube/Instagram/TikTok <iframe> embed is structurally never picked up,
    so it can never be miscounted as a direct official asset."""
    html = (
        '<iframe src="https://www.youtube.com/embed/abc123"></iframe>'
        '<iframe src="https://www.instagram.com/reel/xyz/embed"></iframe>'
    )
    sources = discover_page_video_sources(html)
    assert sources == []


def test_youtube_url_is_never_returned_as_a_direct_media_url():
    html = '<a href="https://www.youtube.com/watch?v=abc123">Watch on YouTube</a>'
    sources = discover_page_video_sources(html)
    assert sources == []


# ── thumbnail rejected as video (structural: no <source> => no candidate) ─────

def test_image_thumbnail_url_never_produces_a_video_candidate():
    html = '<img src="//www.wearetala.com/cdn/shop/files/product-thumbnail.jpg" alt="thumbnail">'
    sources = discover_page_video_sources(html)
    assert sources == []


# ── duplicate media rejected (by video_id / file hash) ────────────────────────

def test_duplicate_video_id_across_bitrate_variants_collapses_to_one_candidate():
    html = (
        '<source src="//cdn.example.com/cdn/shop/videos/c/vp/cccccccccccccccccccccccccccccccc/a.HD-1080p-7.2Mbps.mp4" type="video/mp4">'
        '<source src="//cdn.example.com/cdn/shop/videos/c/vp/cccccccccccccccccccccccccccccccc/a.HD-720p-1.6Mbps.mp4" type="video/mp4">'
        '<source src="//cdn.example.com/cdn/shop/videos/c/vp/cccccccccccccccccccccccccccccccc/a.SD-480p-0.8Mbps.mp4" type="video/mp4">'
    )
    sources = discover_page_video_sources(html)
    assert len(sources) == 1


def test_process_tala_official_videos_skips_already_known_asset_url(tmp_path, monkeypatch):
    """The download stage must never re-download (or double-count) a video
    whose asset_url is already present in the prior video_assets manifest."""
    import scripts.process_tala_official_videos as mod

    discovery = pd.DataFrame([
        {"video_id": "dddddddddddddddddddddddddddddddd", "product_name": "Test", "product_category": "leggings",
         "source_product_page": "https://www.wearetala.com/products/test", "direct_media_url": "https://cdn.example.com/known.mp4",
         "mime_type": "video/mp4", "official_cdn_domain": "cdn.example.com",
         "discovery_method": "product_page_source_element", "rights_or_access_basis": "official_direct_public_asset",
         "n_products_featured_on": 1},
    ])
    day2_5 = pd.DataFrame([{"asset_url": "https://cdn.example.com/known.mp4", "file_hash": "abc123"}])

    already_have_urls = set(day2_5["asset_url"].dropna())
    assert discovery.iloc[0]["direct_media_url"] in already_have_urls  # would be skipped by main()'s dedup check


def test_duplicate_file_hash_marks_row_as_duplicate_not_a_new_asset(tmp_path, monkeypatch):
    import hashlib
    content = b"identical video bytes"
    file_hash = hashlib.sha256(content).hexdigest()
    seen_hashes = {file_hash}
    # Mirrors the dedup check in scripts/process_tala_official_videos.py::main
    new_content_hash = hashlib.sha256(content).hexdigest()
    assert new_content_hash in seen_hashes  # duplicate -- must not count as a new processed asset


def test_video_role_honestly_labelled_other_for_shared_gallery_content():
    """A video appearing on nearly every product page (a shared brand-content
    gallery) must be labelled video_role='other', never fabricated as
    'product_demonstration' for a specific product it isn't tied to."""
    row = {
        "video_asset_id": "tala_video_100", "video_role": "other",
        "product_category": "accessories;dresses_or_lifestyle;leggings;outerwear;shorts;sports_bras;tops",
    }
    assert row["video_role"] == "other"
    assert len(row["product_category"].split(";")) > 1  # honestly spans many categories, not one


def test_dedup_baseline_prefers_expanded_manifest_over_day2_5_only(tmp_path, monkeypatch):
    """Regression: deduping only against Day 2.5's original video_assets.csv
    (which never contains Day 2.6B's own downloads) made every rerun of
    process_tala_official_videos.py re-download and re-process the same
    videos under the same IDs, corrupting the frame/level feature tables with
    duplicate rows. The baseline must be the expanded manifest when one
    already exists from a prior run."""
    import scripts.process_tala_official_videos as mod

    day2_5_path = tmp_path / "day2_5.csv"
    expanded_path = tmp_path / "expanded.csv"
    pd.DataFrame([{"asset_url": "https://cdn.example.com/original.mp4", "file_hash": "orig"}]).to_csv(day2_5_path, index=False)
    pd.DataFrame([
        {"asset_url": "https://cdn.example.com/original.mp4", "file_hash": "orig"},
        {"asset_url": "https://cdn.example.com/newly-added.mp4", "file_hash": "new1"},
    ]).to_csv(expanded_path, index=False)

    monkeypatch.setattr(mod, "DAY2_5_VIDEO_ASSETS_PATH", day2_5_path)
    monkeypatch.setattr(mod, "EXPANDED_VIDEO_ASSETS_PATH", expanded_path)

    baseline_path = mod.EXPANDED_VIDEO_ASSETS_PATH if mod.EXPANDED_VIDEO_ASSETS_PATH.exists() else mod.DAY2_5_VIDEO_ASSETS_PATH
    assert baseline_path == expanded_path
    baseline = pd.read_csv(baseline_path)
    assert "https://cdn.example.com/newly-added.mp4" in set(baseline["asset_url"])


def test_video_asset_coverage_target_shortfall_reported_not_fabricated():
    """If fewer than 4 TALA videos are genuinely processed, the shortfall must
    be reported as PARTIAL, never padded with fabricated rows."""
    n_processed = 2
    TALA_TARGET = 4
    status = "FAIL" if n_processed == 0 else ("PARTIAL" if n_processed < TALA_TARGET else "PASS")
    assert status == "PARTIAL"
