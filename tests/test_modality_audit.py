"""Tests for scripts/audit_assignment_modalities.py -- the strict Day 2.5
acceptance audit. Uses small synthetic DataFrames, not the live project data,
so these tests pin down the RULES (thumbnail != video, lead != processed,
no fabricated completeness) rather than today's specific numbers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_assignment_modalities import (
    audit_images,
    build_claim_evidence_bundle_summary,
    build_modality_audit,
    build_video_asset_coverage,
    build_video_feature_summary,
    build_video_role_distribution,
    evaluate_primary_fusion_readiness,
)


def _video_assets_df(rows):
    cols = ["video_asset_id", "brand", "video_role", "rights_or_access_basis",
            "processing_status", "local_path"]
    return pd.DataFrame(rows, columns=cols)


def _video_level_df(rows):
    cols = ["video_asset_id", "n_sampled_frames", "duration_seconds",
            "mean_inter_frame_perceptual_distance", "transcript_text"]
    return pd.DataFrame(rows, columns=cols)


# ── video coverage: leads never counted as processed ──────────────────────────

def test_metadata_only_leads_are_never_counted_as_processed(tmp_path, monkeypatch):
    import scripts.audit_assignment_modalities as mod
    monkeypatch.setattr(mod, "TABLES", tmp_path)

    video_assets = _video_assets_df([
        {"video_asset_id": "lead_1", "brand": "TALA", "video_role": "other",
         "rights_or_access_basis": "platform_metadata_only", "processing_status": "metadata_only", "local_path": ""},
        {"video_asset_id": "lead_2", "brand": "TALA", "video_role": "other",
         "rights_or_access_basis": "platform_metadata_only", "processing_status": "metadata_only", "local_path": ""},
    ])
    video_level = _video_level_df([])  # no genuinely processed videos

    out = build_video_asset_coverage(video_assets, video_level)
    total = out[out["brand"] == "TOTAL"].iloc[0]
    assert total["genuinely_processed_videos"] == 0
    assert total["video_leads_metadata_only"] == 2


def test_single_frame_thumbnail_row_does_not_count_as_processed(tmp_path, monkeypatch):
    """A video_assets row marked 'processed' but with only 1 sampled frame in
    video_level_features (i.e. a thumbnail, not genuine temporal content) must
    not be counted -- coverage requires n_sampled_frames >= 2."""
    import scripts.audit_assignment_modalities as mod
    monkeypatch.setattr(mod, "TABLES", tmp_path)

    video_assets = _video_assets_df([
        {"video_asset_id": "v1", "brand": "TALA", "video_role": "product_demonstration",
         "rights_or_access_basis": "official_direct_public_asset", "processing_status": "processed",
         "local_path": "data/media/v1.mp4"},
    ])
    video_level = _video_level_df([
        {"video_asset_id": "v1", "n_sampled_frames": 1, "duration_seconds": 10.0,
         "mean_inter_frame_perceptual_distance": None, "transcript_text": None},
    ])
    out = build_video_asset_coverage(video_assets, video_level)
    total = out[out["brand"] == "TOTAL"].iloc[0]
    assert total["genuinely_processed_videos"] == 0


def test_genuinely_processed_multi_frame_video_is_counted(tmp_path, monkeypatch):
    import scripts.audit_assignment_modalities as mod
    monkeypatch.setattr(mod, "TABLES", tmp_path)

    video_assets = _video_assets_df([
        {"video_asset_id": "v1", "brand": "TALA", "video_role": "product_demonstration",
         "rights_or_access_basis": "official_direct_public_asset", "processing_status": "processed",
         "local_path": "data/media/v1.mp4"},
    ])
    video_level = _video_level_df([
        {"video_asset_id": "v1", "n_sampled_frames": 5, "duration_seconds": 30.0,
         "mean_inter_frame_perceptual_distance": 0.12, "transcript_text": None},
    ])
    out = build_video_asset_coverage(video_assets, video_level)
    total = out[out["brand"] == "TOTAL"].iloc[0]
    assert total["genuinely_processed_videos"] == 1


# ── video_feature_summary: temporal-feature requirement ───────────────────────

def test_video_feature_summary_reports_zero_when_no_processed_videos(tmp_path, monkeypatch):
    import scripts.audit_assignment_modalities as mod
    monkeypatch.setattr(mod, "TABLES", tmp_path)
    out = build_video_feature_summary(pd.DataFrame(), pd.DataFrame())
    assert out.iloc[0]["n_processed_videos"] == 0
    assert out.iloc[0]["n_with_temporal_features"] == 0


def test_video_feature_summary_counts_temporal_features_only_for_valid_rows(tmp_path, monkeypatch):
    import scripts.audit_assignment_modalities as mod
    monkeypatch.setattr(mod, "TABLES", tmp_path)
    video_level = _video_level_df([
        {"video_asset_id": "v1", "n_sampled_frames": 5, "duration_seconds": 30.0,
         "mean_inter_frame_perceptual_distance": 0.12, "transcript_text": None},
        {"video_asset_id": "v2", "n_sampled_frames": 1, "duration_seconds": 5.0,
         "mean_inter_frame_perceptual_distance": None, "transcript_text": None},
    ])
    out = build_video_feature_summary(video_level, pd.DataFrame())
    row = out.iloc[0]
    assert row["n_videos_with_ge2_frames"] == 1
    assert row["n_with_temporal_features"] == 1


# ── image audit: verification requirements ─────────────────────────────────────

def _image_assets_df(rows):
    cols = ["asset_id", "brand", "image_role", "source_url", "source_page_url",
            "local_path", "rights_or_access_basis", "width", "height", "file_hash", "processing_status"]
    return pd.DataFrame(rows, columns=cols)


def test_image_audit_excludes_rows_missing_required_fields():
    df = _image_assets_df([
        {"asset_id": "a1", "brand": "TALA", "image_role": "official_catalog",
         "source_url": "https://x/a.jpg", "source_page_url": "https://x/p", "local_path": "p1.jpg",
         "rights_or_access_basis": "official_direct_public_asset", "width": 1000, "height": 1500,
         "file_hash": "abc", "processing_status": "downloaded"},
        {"asset_id": "a2", "brand": "TALA", "image_role": "official_catalog",
         "source_url": "https://x/b.jpg", "source_page_url": "https://x/p2", "local_path": "",
         "rights_or_access_basis": "official_direct_public_asset", "width": None, "height": None,
         "file_hash": "", "processing_status": "failed"},
    ])
    audit = audit_images(df)
    assert audit["verified"] == 1
    assert audit["available"] == 2


def test_image_audit_excludes_creator_thumbnails_from_proxy_count_confusion():
    df = _image_assets_df([
        {"asset_id": "a1", "brand": "TALA", "image_role": "creator_thumbnail",
         "source_url": "https://x/a.jpg", "source_page_url": "https://x/p", "local_path": "p1.jpg",
         "rights_or_access_basis": "platform_metadata_only", "width": 400, "height": 400,
         "file_hash": "abc", "processing_status": "downloaded"},
    ])
    audit = audit_images(df)
    assert audit["proxy_only"] == 1


def test_claim_candidates_empty_string_columns_survive_csv_round_trip(tmp_path):
    """Regression: pandas reads a saved empty string back as NaN, which would
    silently flip `!= ""` presence checks to True for every missing-modality
    claim. The audit must restore '' before counting, never trust NaN as
    'has evidence'."""
    path = tmp_path / "claim_multimodal_evidence_candidates.csv"
    pd.DataFrame([
        {"claim_id": "c1", "text_evidence_ids": "", "image_asset_ids": "", "video_asset_ids": "", "reference_document_ids": ""},
        {"claim_id": "c2", "text_evidence_ids": "doc_1", "image_asset_ids": "", "video_asset_ids": "", "reference_document_ids": ""},
    ]).to_csv(path, index=False)

    df = pd.read_csv(path)
    assert df["text_evidence_ids"].isna().any()  # confirms the round-trip bug is real

    for col in ("text_evidence_ids", "image_asset_ids", "video_asset_ids", "reference_document_ids"):
        df[col] = df[col].fillna("")
    n_all3 = int(((df["text_evidence_ids"] != "") & (df["image_asset_ids"] != "") & (df["video_asset_ids"] != "")).sum())
    assert n_all3 == 0


def test_image_audit_detects_duplicate_urls_and_hashes():
    df = _image_assets_df([
        {"asset_id": "a1", "brand": "TALA", "image_role": "official_catalog",
         "source_url": "https://x/a.jpg", "source_page_url": "https://x/p", "local_path": "p1.jpg",
         "rights_or_access_basis": "official_direct_public_asset", "width": 1000, "height": 1500,
         "file_hash": "same_hash", "processing_status": "downloaded"},
        {"asset_id": "a2", "brand": "TALA", "image_role": "official_catalog",
         "source_url": "https://x/a.jpg", "source_page_url": "https://x/p", "local_path": "p2.jpg",
         "rights_or_access_basis": "official_direct_public_asset", "width": 1000, "height": 1500,
         "file_hash": "same_hash", "processing_status": "downloaded"},
    ])
    audit = audit_images(df)
    assert audit["dup_urls"] == 1
    assert audit["dup_hashes"] == 1


# ── claim-evidence bundle summary: no all-or-nothing collapse ─────────────────

def _claim_candidates_df(rows):
    cols = ["claim_id", "text_evidence_ids", "image_asset_ids", "video_asset_ids", "reference_document_ids"]
    return pd.DataFrame(rows, columns=cols)


def test_bundle_summary_text_and_reference_counts_as_ge1_and_ge2_not_zero():
    """The exact scenario the correction calls out: text + reference evidence,
    no image/video. Must be counted as processed (>=1) and as >=2 eligible
    modalities -- never collapsed to zero because it isn't fully multimodal."""
    df = _claim_candidates_df([
        {"claim_id": "c1", "text_evidence_ids": "doc_1", "image_asset_ids": "", "video_asset_ids": "", "reference_document_ids": "ref_1"},
    ])
    summary = build_claim_evidence_bundle_summary(df)
    row = summary.iloc[0]
    assert row["bundles_with_ge1_verified_modality"] == 1
    assert row["bundles_with_ge2_eligible_modalities"] == 1
    assert row["bundles_with_text_image_video"] == 0
    assert row["bundles_with_insufficient_evidence"] == 0


def test_bundle_summary_separates_every_combination_independently():
    df = _claim_candidates_df([
        {"claim_id": "c_none", "text_evidence_ids": "", "image_asset_ids": "", "video_asset_ids": "", "reference_document_ids": ""},
        {"claim_id": "c_text_only", "text_evidence_ids": "doc_1", "image_asset_ids": "", "video_asset_ids": "", "reference_document_ids": ""},
        {"claim_id": "c_full3", "text_evidence_ids": "doc_1", "image_asset_ids": "img_1", "video_asset_ids": "vid_1", "reference_document_ids": ""},
        {"claim_id": "c_full4", "text_evidence_ids": "doc_1", "image_asset_ids": "img_1", "video_asset_ids": "vid_1", "reference_document_ids": "ref_1"},
    ])
    summary = build_claim_evidence_bundle_summary(df)
    row = summary.iloc[0]
    assert row["n_claims"] == 4
    assert row["bundles_with_insufficient_evidence"] == 1
    assert row["bundles_with_ge1_verified_modality"] == 3
    assert row["bundles_with_text_image_video"] == 2  # c_full3 and c_full4 both have text+image+video
    assert row["bundles_with_text_image_video_reference"] == 1  # only c_full4


def test_bundle_summary_empty_claims_returns_zeroed_row():
    summary = build_claim_evidence_bundle_summary(pd.DataFrame())
    row = summary.iloc[0]
    assert row["n_claims"] == 0
    assert row["bundles_with_ge1_verified_modality"] == 0


# ── primary fusion readiness: minimum defensible bar, not "all 37 claims" ─────

def _bundle_summary_row(**overrides):
    base = {
        "n_claims": 37, "bundles_with_ge1_verified_modality": 21, "bundles_with_ge2_eligible_modalities": 0,
        "bundles_with_text_evidence": 21, "bundles_with_image_evidence": 0, "bundles_with_eligible_video_evidence": 0,
        "bundles_with_reference_evidence": 0, "bundles_with_text_image_video": 0,
        "bundles_with_text_image_video_reference": 0, "bundles_with_insufficient_evidence": 16,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_fusion_readiness_is_nogo_when_reference_package_empty():
    image_audit = {"verified": 24}
    readiness = evaluate_primary_fusion_readiness(image_audit, n_with_temporal=1, n_reference=0, bundle_summary=_bundle_summary_row())
    overall = readiness[readiness["criterion"] == "OVERALL"].iloc[0]
    assert overall["status"] == "NO-GO"
    assert "Reference Package" in overall["detail"]


def test_fusion_readiness_does_not_require_all_claims_multimodal():
    """The minimum bar is >=5 claims with >=2 modalities and >=1 with all
    three -- not all 37 claims fully multimodal."""
    image_audit = {"verified": 24}
    bundle_summary = _bundle_summary_row(bundles_with_ge2_eligible_modalities=5, bundles_with_text_image_video=1)
    readiness = evaluate_primary_fusion_readiness(image_audit, n_with_temporal=1, n_reference=1, bundle_summary=bundle_summary)
    overall = readiness[readiness["criterion"] == "OVERALL"].iloc[0]
    assert overall["status"] == "GO"


def test_fusion_readiness_video_pipeline_gate_ignores_asset_count():
    """Even a single genuinely processed video with real temporal features
    satisfies the video-pipeline criterion -- count is not the gate."""
    image_audit = {"verified": 24}
    bundle_summary = _bundle_summary_row(bundles_with_ge2_eligible_modalities=5, bundles_with_text_image_video=1)
    readiness = evaluate_primary_fusion_readiness(image_audit, n_with_temporal=1, n_reference=1, bundle_summary=bundle_summary)
    video_row = readiness[readiness["criterion"].str.contains("video pipeline")].iloc[0]
    assert video_row["status"] == "MET"


# ── Video Package status: competitor absence must never fail the package ──────

def test_video_package_status_ignores_competitor_brand_count(tmp_path, monkeypatch):
    """Correction: do not classify the Video Package as failed merely because
    every competitor lacks an authorised local video. Status must key off TALA
    depth (the primary-objective brand), not brand-count breadth."""
    monkeypatch.setattr("scripts.audit_assignment_modalities.TABLES", tmp_path)

    video_assets = _video_assets_df([
        {"video_asset_id": "tala_v1", "brand": "TALA", "video_role": "product_demonstration",
         "rights_or_access_basis": "official_direct_public_asset", "processing_status": "processed", "local_path": "x.mp4"},
        {"video_asset_id": "lead_adanola", "brand": "Adanola", "video_role": "other",
         "rights_or_access_basis": "platform_metadata_only", "processing_status": "metadata_only", "local_path": ""},
        {"video_asset_id": "lead_gc", "brand": "Girlfriend Collective", "video_role": "other",
         "rights_or_access_basis": "platform_metadata_only", "processing_status": "metadata_only", "local_path": ""},
        {"video_asset_id": "lead_oner", "brand": "Oner Active", "video_role": "other",
         "rights_or_access_basis": "platform_metadata_only", "processing_status": "metadata_only", "local_path": ""},
    ])
    video_level = _video_level_df([
        {"video_asset_id": "tala_v1", "n_sampled_frames": 9, "duration_seconds": 44.7,
         "mean_inter_frame_perceptual_distance": 0.1, "transcript_text": None},
    ])
    video_coverage = build_video_asset_coverage(video_assets, video_level)
    video_feature_summary = build_video_feature_summary(video_level, pd.DataFrame())
    image_audit = {"verified": 24, "available": 24, "directly_processed": 24, "proxy_only": 0, "missing": ""}
    text_corpora = {"a": pd.DataFrame({"x": [1] * 10})}
    claim_candidates = pd.DataFrame()
    bundle_summary = build_claim_evidence_bundle_summary(claim_candidates)

    audit = build_modality_audit(text_corpora, image_audit, video_coverage, video_feature_summary, claim_candidates, bundle_summary)
    video_pkg = audit[audit["package"] == "Video Package"].iloc[0]

    # 1 TALA processed video, 0 competitor processed -- must be PARTIAL (below the
    # TALA depth target), never FAIL, and the remediation must not blame
    # competitor absence as a blocking failure.
    assert video_pkg["assignment_status"] == "PARTIAL"
    assert "does not block" in video_pkg["remediation_action"].lower()
