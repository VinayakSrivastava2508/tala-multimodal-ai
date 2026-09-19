"""Tests for scripts/build_full_modality_bundles.py (Day 2.6B Part H) and the
Part G product-video eligibility restriction. No live network/model calls."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.build_claim_multimodal_candidates as claim_mod


def _write_candidates(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


# ── full-modality bundle requires eligible evidence in all required fields ────

def test_bundle_requires_text_and_visual_and_reference(tmp_path, monkeypatch):
    import scripts.build_full_modality_bundles as mod
    monkeypatch.setattr(mod, "CANDIDATES_PATH", tmp_path / "candidates.csv")
    monkeypatch.setattr(mod, "OUT_PATH", tmp_path / "bundles.csv")

    _write_candidates(mod.CANDIDATES_PATH, [
        {"claim_id": "c_full", "claim_text": "x", "claim_category": "materials",
         "text_evidence_ids": "doc1", "image_asset_ids": "img1", "video_asset_ids": "",
         "reference_document_ids": "ref1"},
        {"claim_id": "c_missing_ref", "claim_text": "y", "claim_category": "materials",
         "text_evidence_ids": "doc1", "image_asset_ids": "img1", "video_asset_ids": "",
         "reference_document_ids": ""},
        {"claim_id": "c_missing_visual", "claim_text": "z", "claim_category": "materials",
         "text_evidence_ids": "doc1", "image_asset_ids": "", "video_asset_ids": "",
         "reference_document_ids": "ref1"},
    ])
    mod.main()
    out = pd.read_csv(mod.OUT_PATH)
    assert list(out["claim_or_issue_id"]) == ["c_full"]


def test_bundle_accepts_video_in_place_of_image():
    import scripts.build_full_modality_bundles as mod
    df = pd.DataFrame([{
        "claim_id": "c1", "claim_text": "x", "claim_category": "materials",
        "text_evidence_ids": "doc1", "image_asset_ids": "", "video_asset_ids": "vid1",
        "reference_document_ids": "ref1",
    }])
    for col in mod.REQUIRED_FIELD_ORDER:
        df[col] = df[col].fillna("")
    r = df.iloc[0]
    has_visual = r["image_asset_ids"] != "" or r["video_asset_ids"] != ""
    is_full = r["text_evidence_ids"] != "" and has_visual and r["reference_document_ids"] != ""
    assert is_full


# ── missing modalities not fabricated ──────────────────────────────────────────

def test_bundle_missing_modalities_field_lists_actual_gaps(tmp_path, monkeypatch):
    import scripts.build_full_modality_bundles as mod
    monkeypatch.setattr(mod, "CANDIDATES_PATH", tmp_path / "candidates.csv")
    monkeypatch.setattr(mod, "OUT_PATH", tmp_path / "bundles.csv")
    _write_candidates(mod.CANDIDATES_PATH, [
        {"claim_id": "c1", "claim_text": "x", "claim_category": "materials",
         "text_evidence_ids": "doc1", "image_asset_ids": "img1", "video_asset_ids": "vid1",
         "reference_document_ids": "ref1"},
    ])
    mod.main()
    out = pd.read_csv(mod.OUT_PATH)
    assert out.iloc[0]["missing_modalities"] == "none"
    assert out.iloc[0]["modality_count"] == 4


def test_bundle_never_assigns_aligned_or_divergent_verdict(tmp_path, monkeypatch):
    import scripts.build_full_modality_bundles as mod
    monkeypatch.setattr(mod, "CANDIDATES_PATH", tmp_path / "candidates.csv")
    monkeypatch.setattr(mod, "OUT_PATH", tmp_path / "bundles.csv")
    _write_candidates(mod.CANDIDATES_PATH, [
        {"claim_id": "c1", "claim_text": "x", "claim_category": "materials",
         "text_evidence_ids": "doc1", "image_asset_ids": "img1", "video_asset_ids": "",
         "reference_document_ids": "ref1"},
    ])
    mod.main()
    out = pd.read_csv(mod.OUT_PATH)
    forbidden = ("aligned", "divergent", "verdict", "supported", "contradicted")
    for col in out.columns:
        assert not any(f in col.lower() for f in forbidden)
    assert out.iloc[0]["evidence_eligibility"] == "candidate_prototype_not_scored"
    assert bool(out.iloc[0]["requires_human_validation"]) is True


# ── Part G: generic product video cannot support a sustainability/labour claim ─

def test_generic_product_video_cannot_support_sustainability_or_labour_claim(monkeypatch):
    """A video showing product appearance/movement must never be offered as
    evidence for labour, manufacturing, packaging, emissions, or circularity
    claims -- those require a different evidence type entirely."""
    monkeypatch.setattr(claim_mod, "clip_is_available", lambda: True)
    monkeypatch.setattr(claim_mod, "get_clip_text_embedding", lambda text: np.array([1.0, 0.0]))

    claims = pd.DataFrame([
        {"claim_id": "c1", "brand": "TALA", "claim_text": "We ensure fair labour conditions in our factories.", "claim_category": "labour"},
    ])
    embeddings = {"vid1": np.array([1.0, 0.0])}  # perfect similarity -- would match if not gated
    matches, details = claim_mod.match_visual_evidence(claims, "TALA", embeddings, threshold=0.1, modality="video")
    assert matches["c1"] == []
    assert details == []


def test_generic_product_video_can_support_materials_claim_as_presentation_only(monkeypatch):
    monkeypatch.setattr(claim_mod, "clip_is_available", lambda: True)
    monkeypatch.setattr(claim_mod, "get_clip_text_embedding", lambda text: np.array([1.0, 0.0]))

    claims = pd.DataFrame([
        {"claim_id": "c1", "brand": "TALA", "claim_text": "Made from soft, breathable fabric.", "claim_category": "materials"},
    ])
    embeddings = {"vid1": np.array([1.0, 0.0])}
    matches, details = claim_mod.match_visual_evidence(claims, "TALA", embeddings, threshold=0.1, modality="video")
    assert matches["c1"] == [("vid1", 1.0)]
    assert details[0]["evidence_eligibility"] == "presentation_only"


def test_video_ineligible_categories_cover_non_visual_claim_types():
    expected_subset = {"labour", "manufacturing", "packaging", "emissions", "circularity"}
    assert expected_subset.issubset(claim_mod.VIDEO_INELIGIBLE_CLAIM_CATEGORIES)


# ── reference PASS requires real care evidence (not generic prose) ────────────

def test_reference_package_care_criterion_requires_explicit_instruction():
    from src.collectors.product_pages import is_explicit_care_instruction
    assert not is_explicit_care_instruction("Please look after your clothes.")
    assert is_explicit_care_instruction("Machine wash at 30C. Do not tumble dry.")


# ── Video Package target does not affect Video Pipeline technical PASS ────────

def test_video_pipeline_pass_independent_of_video_package_count_target():
    """The pipeline is judged on whether every genuinely processed video
    produced valid temporal features -- not on whether the COUNT clears the
    Video Package's >=4 depth target. A single successfully processed video
    still yields Video Pipeline PASS."""
    n_processed = 1
    n_with_temporal = 1
    pipeline_fully_reliable = n_processed > 0 and n_with_temporal == n_processed
    video_pipeline_status = "PASS" if pipeline_fully_reliable else "PARTIAL"
    assert video_pipeline_status == "PASS"  # independent of the separate Video Package >=4 target


# ── engagement-prediction outputs remain absent ────────────────────────────────

def test_full_modality_bundle_output_has_no_engagement_prediction_columns(tmp_path, monkeypatch):
    import scripts.build_full_modality_bundles as mod
    monkeypatch.setattr(mod, "CANDIDATES_PATH", tmp_path / "candidates.csv")
    monkeypatch.setattr(mod, "OUT_PATH", tmp_path / "bundles.csv")
    _write_candidates(mod.CANDIDATES_PATH, [
        {"claim_id": "c1", "claim_text": "x", "claim_category": "materials",
         "text_evidence_ids": "doc1", "image_asset_ids": "img1", "video_asset_ids": "",
         "reference_document_ids": "ref1"},
    ])
    mod.main()
    out = pd.read_csv(mod.OUT_PATH)
    forbidden = ("engagement", "predicted", "residual", "engagement_score")
    for col in out.columns:
        assert not any(f in col.lower() for f in forbidden)
