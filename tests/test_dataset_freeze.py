"""Tests for the Day 2 Task 2 dataset freeze (scripts/freeze_day2_datasets.py) and
the shared modelling-table guards in src/validation.py. No live network calls."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.freeze_day2_datasets import _sha256_of_file, freeze_file
from src.validation import assert_evidence_strength_allowed, assert_no_duplicate_ids


# ── checksum reproducibility ──────────────────────────────────────────────────

def test_sha256_is_reproducible_for_identical_content(tmp_path):
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    content = "col1,col2\n1,2\n3,4\n"
    p1.write_text(content, encoding="utf-8")
    p2.write_text(content, encoding="utf-8")
    assert _sha256_of_file(p1) == _sha256_of_file(p2)


def test_sha256_differs_for_different_content(tmp_path):
    p1 = tmp_path / "a.csv"
    p2 = tmp_path / "b.csv"
    p1.write_text("col1,col2\n1,2\n", encoding="utf-8")
    p2.write_text("col1,col2\n1,3\n", encoding="utf-8")
    assert _sha256_of_file(p1) != _sha256_of_file(p2)


def test_sha256_same_file_reread_is_stable(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text("x,y\n1,2\n", encoding="utf-8")
    assert _sha256_of_file(p) == _sha256_of_file(p)


# ── freeze_file manifest construction ─────────────────────────────────────────

def test_freeze_file_reports_missing_file(tmp_path):
    fake_rel_path = "data/does_not_exist/nope.csv"
    spec = {"id_field": "document_id", "critical_fields": ["brand"], "has_strength": True}
    result = freeze_file(fake_rel_path, spec)
    assert isinstance(result, dict)
    assert result.get("exists") is False


def test_freeze_file_computes_unique_and_duplicate_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rel = "sample.csv"
    df = pd.DataFrame({
        "document_id": ["a", "a", "b"],
        "brand": ["TALA", "TALA", "Adanola"],
        "evidence_strength": ["strong", "strong", "medium"],
        "source_url": ["http://x", "http://x", "http://y"],
        "rag_usable": [True, True, True],
    })
    import scripts.freeze_day2_datasets as fd
    monkeypatch.setattr(fd, "PROJECT_ROOT", tmp_path)
    df.to_csv(tmp_path / rel, index=False)

    spec = {"id_field": "document_id", "critical_fields": ["brand", "source_url", "evidence_strength", "rag_usable"], "has_strength": True}
    manifest_row, snapshot_entry = fd.freeze_file(rel, spec)
    assert manifest_row["row_count"] == 3
    assert manifest_row["unique_id_count"] == 2
    assert manifest_row["duplicate_id_count"] == 1
    assert manifest_row["sha256"] == _sha256_of_file(tmp_path / rel)


# ── modelling-table guards ────────────────────────────────────────────────────

def test_assert_no_duplicate_ids_raises_on_duplicates():
    df = pd.DataFrame({"post_id": ["a", "a", "b"]})
    with pytest.raises(ValueError, match="duplicate"):
        assert_no_duplicate_ids(df, "post_id")


def test_assert_no_duplicate_ids_passes_on_unique():
    df = pd.DataFrame({"post_id": ["a", "b", "c"]})
    assert_no_duplicate_ids(df, "post_id")  # should not raise


def test_assert_evidence_strength_allowed_raises_on_weak():
    df = pd.DataFrame({"evidence_strength": ["strong", "weak", "medium"]})
    with pytest.raises(ValueError, match="weak/unusable"):
        assert_evidence_strength_allowed(df, ("strong", "medium"))


def test_assert_evidence_strength_allowed_raises_on_unusable():
    df = pd.DataFrame({"evidence_strength": ["strong", "unusable"]})
    with pytest.raises(ValueError):
        assert_evidence_strength_allowed(df, ("strong", "medium"))


def test_assert_evidence_strength_allowed_passes_on_all_usable():
    df = pd.DataFrame({"evidence_strength": ["strong", "medium", "strong"]})
    assert_evidence_strength_allowed(df, ("strong", "medium"))  # should not raise


def test_assert_evidence_strength_allowed_skips_missing_column():
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    assert_evidence_strength_allowed(df, ("strong", "medium"))  # should not raise, column absent


# ── missing metrics remain missing (no fabricated zeros) ──────────────────────

def test_missing_engagement_metrics_are_not_filled_with_zero():
    """Contract test on the actual frozen creator feature table (Day 2 Task 2B):
    a row whose YouTube API call did not succeed must have NULL view/like/comment
    counts -- never a fabricated 0 standing in for 'not retrieved'."""
    features_path = PROJECT_ROOT / "data" / "processed" / "creator_multimodal_features.csv"
    if not features_path.exists():
        pytest.skip("creator_multimodal_features.csv not built yet -- run build_multimodal_features.py")
    df = pd.read_csv(features_path)
    for col in ("view_count", "like_count", "comment_count", "channel_subscriber_count_current"):
        assert col in df.columns
    not_retrieved = df["api_retrieval_status"] != "success"
    if not_retrieved.any():
        for col in ("view_count", "like_count", "comment_count"):
            assert df.loc[not_retrieved, col].isna().all(), (
                f"{col} has a non-null value for a row where the API call did not succeed"
            )
