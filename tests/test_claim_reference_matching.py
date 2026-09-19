"""Tests for the Day 2.6A Part I claim-reference matching hierarchy
(scripts/build_claim_multimodal_candidates.py::match_reference_evidence).
No live sentence-transformers model calls -- compute_semantic_similarity is
stubbed where used."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.build_claim_multimodal_candidates as mod


def _claims_df(rows):
    cols = ["claim_id", "brand", "claim_text", "claim_category"]
    return pd.DataFrame(rows, columns=cols)


def _chunks_df(rows):
    cols = ["chunk_id", "reference_id", "brand", "reference_type", "chunk_text", "materials", "certifications"]
    return pd.DataFrame(rows, columns=cols)


# ── category gating: irrelevant reference_type is rejected before any scoring ──

def test_return_policy_cannot_support_a_materials_claim(monkeypatch):
    """A generic return policy cannot support a materials claim -- the charter's
    own example. The return_policy chunk must never be offered as a match for
    a 'materials' category claim, regardless of textual similarity."""
    monkeypatch.setattr(mod, "compute_semantic_similarity", lambda q, c: np.array([[0.99]]))  # even if similarity were high
    claims = _claims_df([{"claim_id": "c1", "brand": "TALA", "claim_text": "Made from recycled nylon.", "claim_category": "materials"}])
    chunks = _chunks_df([
        {"chunk_id": "ch1", "reference_id": "ref1", "brand": "TALA", "reference_type": "return_policy",
         "chunk_text": "You may return items within 30 days.", "materials": "", "certifications": ""},
    ])
    matches, details = mod.match_reference_evidence(claims, chunks)
    assert matches["c1"] == []
    assert details == []


def test_size_guide_cannot_support_an_emissions_claim(monkeypatch):
    monkeypatch.setattr(mod, "compute_semantic_similarity", lambda q, c: np.array([[0.99]]))
    claims = _claims_df([{"claim_id": "c1", "brand": "TALA", "claim_text": "We are reducing our carbon emissions by 2030.", "claim_category": "emissions"}])
    chunks = _chunks_df([
        {"chunk_id": "ch1", "reference_id": "ref1", "brand": "TALA", "reference_type": "size_guide",
         "chunk_text": "UK 8: chest 82cm.", "materials": "", "certifications": ""},
    ])
    matches, details = mod.match_reference_evidence(claims, chunks)
    assert matches["c1"] == []


def test_unmapped_claim_category_is_never_gated_open():
    """A claim category with no entry in CLAIM_CATEGORY_TO_REFERENCE_TYPES (e.g.
    'other') must never match any reference chunk -- there is no known
    plausible reference_type for it, so the gate defaults closed, not open."""
    claims = _claims_df([{"claim_id": "c1", "brand": "TALA", "claim_text": "Something unrelated.", "claim_category": "other"}])
    chunks = _chunks_df([
        {"chunk_id": "ch1", "reference_id": "ref1", "brand": "TALA", "reference_type": "sustainability_page",
         "chunk_text": "Something unrelated to match against.", "materials": "", "certifications": ""},
    ])
    matches, details = mod.match_reference_evidence(claims, chunks)
    assert matches["c1"] == []


# ── material/certification keyword match (tier 3) ──────────────────────────────

def test_material_keyword_overlap_produces_material_or_certification_match():
    claims = _claims_df([{"claim_id": "c1", "brand": "TALA", "claim_text": "Made from recycled polyester.", "claim_category": "materials"}])
    chunks = _chunks_df([
        {"chunk_id": "ch1", "reference_id": "ref1", "brand": "TALA", "reference_type": "material_specification",
         "chunk_text": "This fabric contains recycled polyester and elastane.",
         "materials": "recycled polyester;elastane", "certifications": ""},
    ])
    matches, details = mod.match_reference_evidence(claims, chunks)
    assert matches["c1"] == ["ref1"]
    assert details[0]["match_method"] == "material_or_certification_match"
    assert details[0]["category_gate_passed"] is True
    assert details[0]["requires_human_validation"] is True
    assert "recycled polyester" in details[0]["match_explanation"]


# ── category-gated semantic match (tier 5) ──────────────────────────────────────

def test_semantic_match_used_when_no_keyword_overlap(monkeypatch):
    monkeypatch.setattr(mod, "compute_semantic_similarity", lambda q, c: np.array([[0.5]]))
    claims = _claims_df([{"claim_id": "c1", "brand": "TALA", "claim_text": "Our garments are ethically made.", "claim_category": "labour"}])
    chunks = _chunks_df([
        {"chunk_id": "ch1", "reference_id": "ref1", "brand": "TALA", "reference_type": "labour_or_ethics_policy",
         "chunk_text": "We audit our factories for fair labour practices.", "materials": "", "certifications": ""},
    ])
    matches, details = mod.match_reference_evidence(claims, chunks)
    assert matches["c1"] == ["ref1"]
    assert details[0]["match_method"] == "category_gated_semantic_match"
    assert details[0]["match_score"] == 0.5


def test_semantic_match_rejected_below_threshold(monkeypatch):
    monkeypatch.setattr(mod, "compute_semantic_similarity", lambda q, c: np.array([[0.1]]))
    claims = _claims_df([{"claim_id": "c1", "brand": "TALA", "claim_text": "Our garments are ethically made.", "claim_category": "labour"}])
    chunks = _chunks_df([
        {"chunk_id": "ch1", "reference_id": "ref1", "brand": "TALA", "reference_type": "labour_or_ethics_policy",
         "chunk_text": "Unrelated filler text.", "materials": "", "certifications": ""},
    ])
    matches, details = mod.match_reference_evidence(claims, chunks)
    assert matches["c1"] == []


# ── brand isolation ─────────────────────────────────────────────────────────────

def test_match_never_crosses_brands():
    claims = _claims_df([{"claim_id": "c1", "brand": "TALA", "claim_text": "Made from recycled polyester.", "claim_category": "materials"}])
    chunks = _chunks_df([
        {"chunk_id": "ch1", "reference_id": "ref1", "brand": "Adanola", "reference_type": "material_specification",
         "chunk_text": "Made from recycled polyester.", "materials": "recycled polyester", "certifications": ""},
    ])
    matches, details = mod.match_reference_evidence(claims, chunks)
    assert matches["c1"] == []


# ── empty pools ──────────────────────────────────────────────────────────────────

def test_empty_chunks_returns_empty_matches_for_every_claim():
    claims = _claims_df([{"claim_id": "c1", "brand": "TALA", "claim_text": "x", "claim_category": "materials"}])
    matches, details = mod.match_reference_evidence(claims, pd.DataFrame())
    assert matches["c1"] == []
    assert details == []


# ── provenance retained on every match detail row ──────────────────────────────

def test_match_detail_rows_retain_full_provenance():
    claims = _claims_df([{"claim_id": "c1", "brand": "TALA", "claim_text": "Made from recycled polyester.", "claim_category": "materials"}])
    chunks = _chunks_df([
        {"chunk_id": "ch1", "reference_id": "ref1", "brand": "TALA", "reference_type": "material_specification",
         "chunk_text": "Contains recycled polyester.", "materials": "recycled polyester", "certifications": ""},
    ])
    _, details = mod.match_reference_evidence(claims, chunks)
    row = details[0]
    for field in ("claim_id", "reference_id", "chunk_id", "match_method", "match_score",
                  "category_gate_passed", "match_explanation", "requires_human_validation"):
        assert field in row


# ── weak/unusable reference rows are excluded upstream from RAG eligibility ────
# (the matching function itself only ever sees chunks built from rows already
# filtered to reference_status in {collected, already_available} by
# scripts/process_reference_package.py -- this test pins that contract at the
# chunk-selection boundary, not inside match_reference_evidence.)

def test_process_reference_package_only_processes_collected_or_already_available():
    import scripts.process_reference_package as proc
    assert proc.PROCESSABLE_STATUSES == ("collected", "already_available")
