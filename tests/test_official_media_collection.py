"""Tests for src/collectors/official_media.py (Day 2.5 Image Package collection).
No live network calls -- requests.get is stubbed."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.official_media import (
    ImageCandidate,
    dedupe_candidates,
    download_image,
    fetch_shopify_products,
)


def _fake_response(json_data=None, content=b"", status_code=200):
    return SimpleNamespace(
        status_code=status_code,
        json=lambda: json_data if json_data is not None else {},
        content=content,
    )


# ── fetch_shopify_products ────────────────────────────────────────────────────

def test_fetch_shopify_products_extracts_candidates_with_provenance():
    payload = {
        "products": [
            {
                "title": "Test Vest", "product_type": "Vest Tops", "handle": "test-vest",
                "images": [
                    {"src": "https://cdn.example.com/a.jpg", "width": 1000, "height": 1500},
                    {"src": "https://cdn.example.com/b.jpg", "width": 1000, "height": 1500},
                ],
            }
        ]
    }
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(url)
        if params and params.get("page", 1) > 1:
            return _fake_response({"products": []})
        return _fake_response(payload)

    candidates = fetch_shopify_products("https://brand.example.com", "TestBrand", get=fake_get, max_pages=2, page_size=50, delay=0)
    assert len(candidates) == 2
    assert candidates[0].image_role == "official_catalog"
    assert candidates[1].image_role == "product_detail"
    assert candidates[0].source_page_url == "https://brand.example.com/products/test-vest"
    assert candidates[0].brand == "TestBrand"


def test_fetch_shopify_products_returns_empty_on_non_200():
    def fake_get(url, headers=None, params=None, timeout=None):
        return _fake_response(status_code=404)
    assert fetch_shopify_products("https://brand.example.com", "TestBrand", get=fake_get, delay=0) == []


# ── dedupe_candidates ─────────────────────────────────────────────────────────

def test_dedupe_candidates_strips_query_string_and_removes_duplicates():
    candidates = [
        ImageCandidate(source_url="https://cdn.example.com/a.jpg?v=1"),
        ImageCandidate(source_url="https://cdn.example.com/a.jpg?v=2"),
        ImageCandidate(source_url="https://cdn.example.com/b.jpg"),
    ]
    out = dedupe_candidates(candidates)
    assert len(out) == 2
    assert {c.source_url for c in out} == {"https://cdn.example.com/a.jpg?v=1", "https://cdn.example.com/b.jpg"}


# ── download_image: rights basis, provenance, honest failure ─────────────────

def test_download_image_records_rights_basis_and_provenance(tmp_path):
    candidate = ImageCandidate(
        brand="TestBrand", product_name="Test Vest", product_category="Vest Tops",
        image_role="official_catalog", source_url="https://cdn.example.com/a.jpg",
        source_page_url="https://brand.example.com/products/test-vest", width=1000, height=1500,
    )
    fake_bytes = b"\xff\xd8\xff" + b"0" * 100  # not a real JPEG but non-empty content is enough here

    def fake_get(url, headers=None, timeout=None):
        return _fake_response(content=fake_bytes)

    row = download_image(candidate, "test_001", get=fake_get, media_dir=tmp_path, delay=0)
    assert row["processing_status"] == "downloaded"
    assert row["rights_or_access_basis"] == "official_direct_public_asset"
    assert row["file_hash"] != ""
    assert "product_page=https://brand.example.com/products/test-vest" in row["provenance_note"]
    assert Path(row["local_path"]).exists()


def test_download_image_never_raises_on_failed_download(tmp_path):
    candidate = ImageCandidate(source_url="https://cdn.example.com/missing.jpg", brand="TestBrand")

    def fake_get(url, headers=None, timeout=None):
        return _fake_response(status_code=404)

    row = download_image(candidate, "test_fail", get=fake_get, media_dir=tmp_path, delay=0)
    assert row["processing_status"] == "failed"
    assert row["evidence_strength"] == "unusable"
    assert row["local_path"] == ""
    assert "download_failed" in row["provenance_note"]


def test_download_image_never_labels_a_failed_download_as_strong_evidence(tmp_path):
    candidate = ImageCandidate(source_url="https://cdn.example.com/missing.jpg", brand="TestBrand")

    def fake_get(url, headers=None, timeout=None):
        raise ConnectionError("simulated network failure")

    row = download_image(candidate, "test_exc", get=fake_get, media_dir=tmp_path, delay=0)
    assert row["evidence_strength"] != "strong"
    assert row["processing_status"] == "failed"


# ── image_assets.csv manifest-level invariants (no fabricated official status) ─

def test_downloaded_image_row_never_uses_creator_thumbnail_role_for_official_catalog():
    """A row discovered via the official Shopify catalog route must never be
    mislabelled as a creator thumbnail -- catalog vs. creator-sourced imagery
    must stay distinguishable downstream."""
    candidate = ImageCandidate(image_role="official_catalog", source_url="https://cdn.example.com/a.jpg")
    assert candidate.image_role != "creator_thumbnail"
