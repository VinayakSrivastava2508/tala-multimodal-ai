"""Tests for src/collectors/product_pages.py (Day 2.6B targeted TALA
product-page enumeration and reference-content extraction). No live network
calls."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.product_pages import (
    ProductCandidate,
    _TYPE_TO_CATEGORY,
    discover_page_video_sources,
    enumerate_diverse_products,
    extract_care_fields,
    extract_material_composition,
    extract_model_info,
    extract_product_sections,
    is_explicit_care_instruction,
)


# ── care text requires explicit instruction; generic prose rejected ──────────

def test_explicit_care_instruction_accepted():
    text = "Delicate machine wash at 30C. Do not bleach. Do not tumble dry. Do not iron."
    assert is_explicit_care_instruction(text)


def test_generic_care_prose_rejected():
    text = "All TALA items are made to last. By caring for our clothes we prolong their life."
    assert not is_explicit_care_instruction(text)


def test_extract_care_fields_empty_for_generic_prose():
    text = "Please take good care of your garments so they last a long time."
    fields = extract_care_fields(text)
    assert fields["washing_method"] == ""
    assert fields["care_temperature"] == ""


def test_extract_care_fields_populates_from_explicit_instruction():
    text = "Delicate machine wash at 30C. Wash with similar colours. Do not bleach. Do not tumble dry. Do not iron."
    fields = extract_care_fields(text)
    assert fields["care_temperature"] == "30C"
    assert fields["washing_method"] == "machine_wash"
    assert fields["drying_method"] == "do_not_tumble_dry"
    assert fields["ironing_guidance"] == "do_not_iron"
    assert fields["bleaching_guidance"] == "do_not_bleach"


def test_extract_care_fields_hand_wash_variant():
    text = "Hand wash only. Do not tumble dry. Do not bleach."
    fields = extract_care_fields(text)
    assert fields["washing_method"] == "hand_wash"


# ── product/category provenance retained ──────────────────────────────────────

def test_extract_product_sections_finds_wear_care_aware():
    html = (
        '<div class="product-information__description product-information__description--wear">'
        '<div class="metafield-rich_text_field"><p>Flattering wrap waist. Model Info: Naomi / UK Size 8 / Wearing TALA Size XS / 5\'7"</p></div></div>'
        '<div class="product-information__description product-information__description--care">'
        '<div class="metafield-rich_text_field"><ul><li>Delicate machine wash at 30C</li><li>Do not bleach</li></ul></div></div>'
        '<div class="product-information__description product-information__description--aware">'
        '<div class="metafield-rich_text_field"><p>This item is made from 78% recycled Polyamide.</p></div></div>'
    )
    sections = extract_product_sections(html)
    assert "wear" in sections and "Flattering" in sections["wear"]
    assert "care" in sections and "machine wash" in sections["care"]
    assert "aware" in sections and "78%" in sections["aware"]


def test_extract_model_info_returns_line():
    wear_text = "Flattering wrap waist.\nModel Info: Naomi / UK Size 8 / Wearing TALA Size XS / 5'7\""
    info = extract_model_info(wear_text)
    assert "Naomi" in info
    assert "UK Size 8" in info


def test_extract_model_info_empty_when_absent():
    assert extract_model_info("Flattering wrap waist, no model line here.") == ""


def test_extract_material_composition_percentage_pattern():
    aware_text = "This item is made from 78% recycled Polyamide certified by GRS."
    composition = extract_material_composition(aware_text)
    assert "78%" in composition
    assert "Polyamide" in composition


def test_extract_material_composition_empty_when_absent():
    assert extract_material_composition("No material info here at all.") == ""


# ── category mapping / diverse enumeration ─────────────────────────────────────

def test_category_map_covers_all_seven_target_categories():
    expected = {"leggings", "shorts", "sports_bras", "tops", "outerwear", "dresses_or_lifestyle", "accessories"}
    from src.collectors.product_pages import CATEGORY_MAP
    assert set(CATEGORY_MAP.keys()) == expected


def test_enumerate_diverse_products_selects_per_category_cap(monkeypatch):
    fake_products = []
    for i in range(5):
        fake_products.append({
            "title": f"Legging {i}", "handle": f"legging-{i}", "product_type": "Leggings",
            "variants": [{"available": True, "option1": "M"}],
        })
    for i in range(5):
        fake_products.append({
            "title": f"Bra {i}", "handle": f"bra-{i}", "product_type": "Sports Bras",
            "variants": [{"available": True, "option1": "S"}],
        })

    def fake_get(url, headers=None, params=None, timeout=None):
        class R:
            status_code = 200
            def json(self):
                return {"products": fake_products if params.get("page", 1) == 1 else []}
        return R()

    monkeypatch.setattr("src.collectors.product_pages.can_fetch_robots", lambda url, ua=None: True)
    candidates = enumerate_diverse_products("https://www.wearetala.com", get=fake_get, per_category=3)
    leggings = [c for c in candidates if c.category == "leggings"]
    bras = [c for c in candidates if c.category == "sports_bras"]
    assert len(leggings) == 3
    assert len(bras) == 3
    for c in candidates:
        assert c.canonical_url.startswith("https://www.wearetala.com/products/")


def test_product_candidate_retains_provenance_fields():
    c = ProductCandidate(
        product_name="Test Legging", product_handle="test-legging", canonical_url="https://www.wearetala.com/products/test-legging",
        category="leggings", availability="in_stock", discovery_method="products_json",
    )
    assert c.product_name and c.product_handle and c.canonical_url and c.category and c.discovery_method


# ── per-page video-source discovery: distinct-ID grouping ─────────────────────

def test_discover_page_video_sources_groups_by_video_id_not_bitrate():
    html = (
        '<source src="//cdn.example.com/cdn/shop/videos/c/vp/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/x.HD-1080p-7.2Mbps-1.mp4?v=0" type="video/mp4">'
        '<source src="//cdn.example.com/cdn/shop/videos/c/vp/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/x.HD-720p-1.6Mbps-2.mp4?v=0" type="video/mp4">'
        '<source src="//cdn.example.com/cdn/shop/videos/c/vp/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/y.HD-1080p-7.2Mbps-3.mp4?v=0" type="video/mp4">'
    )
    sources = discover_page_video_sources(html)
    assert len(sources) == 2  # two distinct video IDs, not three source tags
    ids = {s.video_id for s in sources}
    assert ids == {"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}


def test_discover_page_video_sources_prefers_lowest_bitrate():
    html = (
        '<source src="//cdn.example.com/cdn/shop/videos/c/vp/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/x.HD-1080p-7.2Mbps-1.mp4?v=0" type="video/mp4">'
        '<source src="//cdn.example.com/cdn/shop/videos/c/vp/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/x.HD-720p-1.6Mbps-2.mp4?v=0" type="video/mp4">'
    )
    sources = discover_page_video_sources(html)
    assert len(sources) == 1
    assert "1.6Mbps" in sources[0].asset_url


def test_discover_page_video_sources_empty_for_no_video():
    assert discover_page_video_sources("<html><body>No video here.</body></html>") == []
