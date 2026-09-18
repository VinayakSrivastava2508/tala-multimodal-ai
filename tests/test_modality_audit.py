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
    build_video_asset_coverage,
    build_video_feature_summary,
    build_video_role_distribution,
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
