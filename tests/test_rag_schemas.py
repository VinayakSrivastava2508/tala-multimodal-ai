"""Tests for src/rag/schemas.py -- deterministic IDs and groundability gates."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.schemas import (
    claim_category_groundability, claim_id_for, is_visually_groundable_category,
    sha256_text, text_evidence_id_for, visual_evidence_id_for,
)


def test_claim_id_is_deterministic():
    assert claim_id_for("c_OC_0003_01") == claim_id_for("c_OC_0003_01")
    assert claim_id_for("c_OC_0003_01") != claim_id_for("c_OC_0003_02")


def test_text_evidence_id_is_deterministic_and_namespaced():
    a = text_evidence_id_for("customer_experience", "ce_rev_0001")
    b = text_evidence_id_for("customer_experience", "ce_rev_0001")
    c = text_evidence_id_for("official_claims", "ce_rev_0001")
    assert a == b
    assert a != c  # same raw id, different source_type -> different ID


def test_visual_evidence_id_is_deterministic_and_namespaced():
    a = visual_evidence_id_for("image", "tala_001")
    b = visual_evidence_id_for("video_frame", "tala_001")
    assert a != b


def test_sha256_text_is_deterministic():
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("hello!")


def test_materials_is_conditionally_groundable():
    assert claim_category_groundability("materials") == "conditionally_visually_groundable"
    assert is_visually_groundable_category("materials") is True


def test_packaging_is_potentially_groundable():
    assert claim_category_groundability("packaging") == "potentially_visually_groundable"
    assert is_visually_groundable_category("packaging") is True


def test_labour_sustainability_categories_never_visually_groundable():
    for category in ("labour", "manufacturing", "emissions", "circularity", "other"):
        assert claim_category_groundability(category) == "not_visually_groundable"
        assert is_visually_groundable_category(category) is False


def test_unknown_category_defaults_to_not_groundable():
    assert is_visually_groundable_category("some_unmapped_category") is False
