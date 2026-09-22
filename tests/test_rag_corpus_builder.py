"""Tests for src/rag/corpus_builder.py -- eligibility gates and deterministic
IDs. Uses small synthetic CSVs written to tmp_path; never touches real data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import corpus_builder


def test_corpus_text_records_excludes_non_rag_usable_rows(tmp_path):
    df = pd.DataFrame([
        {"document_id": "oc_1", "corpus": "official_claims", "brand": "TALA", "source_platform": "web",
         "source_url": "https://x", "document_title": "t", "publication_date": "", "extracted_text": "Usable text.",
         "evidence_strength": "strong", "rag_usable": True, "retrieved_at": ""},
        {"document_id": "oc_2", "corpus": "official_claims", "brand": "TALA", "source_platform": "web",
         "source_url": "https://x", "document_title": "t", "publication_date": "", "extracted_text": "Not usable.",
         "evidence_strength": "weak", "rag_usable": False, "retrieved_at": ""},
    ])
    path = tmp_path / "corpus.csv"
    df.to_csv(path, index=False)

    records = corpus_builder._corpus_text_records(path, "official_claims", pd.DataFrame(columns=["modality", "source_type", "evidence_id", "claim_id", "stance", "source_independence", "evidence_strength"]))
    assert len(records) == 1
    assert records[0]["metadata"]["evidence_id"] == "oc_1"


def test_corpus_text_records_excludes_weak_evidence_strength(tmp_path):
    df = pd.DataFrame([
        {"document_id": "oc_1", "corpus": "official_claims", "brand": "TALA", "source_platform": "web",
         "source_url": "https://x", "document_title": "t", "publication_date": "", "extracted_text": "Weak text.",
         "evidence_strength": "weak", "rag_usable": True, "retrieved_at": ""},
        {"document_id": "oc_2", "corpus": "official_claims", "brand": "TALA", "source_platform": "web",
         "source_url": "https://x", "document_title": "t", "publication_date": "", "extracted_text": "Unusable text.",
         "evidence_strength": "unusable", "rag_usable": True, "retrieved_at": ""},
    ])
    path = tmp_path / "corpus.csv"
    df.to_csv(path, index=False)

    records = corpus_builder._corpus_text_records(path, "official_claims", pd.DataFrame(columns=["modality", "source_type", "evidence_id", "claim_id", "stance", "source_independence", "evidence_strength"]))
    assert records == []


def test_corpus_text_records_excludes_empty_text(tmp_path):
    df = pd.DataFrame([
        {"document_id": "oc_1", "corpus": "official_claims", "brand": "TALA", "source_platform": "web",
         "source_url": "https://x", "document_title": "t", "publication_date": "", "extracted_text": "   ",
         "evidence_strength": "strong", "rag_usable": True, "retrieved_at": ""},
    ])
    path = tmp_path / "corpus.csv"
    df.to_csv(path, index=False)
    records = corpus_builder._corpus_text_records(path, "official_claims", pd.DataFrame(columns=["modality", "source_type", "evidence_id", "claim_id", "stance", "source_independence", "evidence_strength"]))
    assert records == []


def test_corpus_text_records_deduplicates_by_document_id(tmp_path):
    df = pd.DataFrame([
        {"document_id": "oc_1", "corpus": "official_claims", "brand": "TALA", "source_platform": "web",
         "source_url": "https://x", "document_title": "t", "publication_date": "", "extracted_text": "Text A.",
         "evidence_strength": "strong", "rag_usable": True, "retrieved_at": ""},
        {"document_id": "oc_1", "corpus": "official_claims", "brand": "TALA", "source_platform": "web",
         "source_url": "https://x", "document_title": "t", "publication_date": "", "extracted_text": "Text A duplicate.",
         "evidence_strength": "strong", "rag_usable": True, "retrieved_at": ""},
    ])
    path = tmp_path / "corpus.csv"
    df.to_csv(path, index=False)
    records = corpus_builder._corpus_text_records(path, "official_claims", pd.DataFrame(columns=["modality", "source_type", "evidence_id", "claim_id", "stance", "source_independence", "evidence_strength"]))
    assert len(records) == 1


def test_join_field_drops_nan_and_empty():
    assert corpus_builder._join_field(["a", "", "nan", "b"]) == "a;b"
    assert corpus_builder._join_field([]) == ""


def test_claim_links_groups_by_evidence_id():
    units = pd.DataFrame([
        {"modality": "text", "source_type": "customer_review", "evidence_id": "ce_1", "claim_id": "c1", "stance": "supports", "source_independence": "customer_reported", "evidence_strength": "strong"},
        {"modality": "text", "source_type": "customer_review", "evidence_id": "ce_1", "claim_id": "c2", "stance": "challenges", "source_independence": "customer_reported", "evidence_strength": "strong"},
        {"modality": "text", "source_type": "official_claim", "evidence_id": "c3", "claim_id": "c3", "stance": "not_applicable", "source_independence": "self_reported", "evidence_strength": "strong"},
    ])
    links = corpus_builder._claim_links(units, "text")
    assert links["ce_1"]["claim_ids"] == ["c1", "c2"]
    assert links["ce_1"]["stances"] == ["supports", "challenges"]
    assert "c3" not in links  # official_claim self-referencing rows are excluded


# ── real-data smoke tests (skip if the pipeline hasn't been run) ───────────────

FUSION_RESULTS = PROJECT_ROOT / "data" / "processed" / "fusion" / "claim_fusion_results.csv"


@pytest.mark.skipif(not FUSION_RESULTS.exists(), reason="Day 3A fusion outputs not present in this environment")
def test_build_claims_records_produces_one_record_per_real_claim():
    records = corpus_builder.build_claims_records()
    fusion_results = pd.read_csv(FUSION_RESULTS)
    assert len(records) == len(fusion_results)
    ids = {r["id"] for r in records}
    assert len(ids) == len(records)  # deterministic IDs, no collisions


@pytest.mark.skipif(not FUSION_RESULTS.exists(), reason="Day 3A fusion outputs not present in this environment")
def test_build_visual_evidence_records_never_includes_missing_assets():
    records = corpus_builder.build_visual_evidence_records()
    for r in records:
        assert Path(r["local_path"]).exists()


@pytest.mark.skipif(not FUSION_RESULTS.exists(), reason="Day 3A fusion outputs not present in this environment")
def test_build_visual_evidence_records_excludes_metadata_only_videos():
    records = corpus_builder.build_visual_evidence_records()
    video_assets = pd.read_csv(PROJECT_ROOT / "data" / "interim" / "day2_6" / "video_assets_expanded.csv")
    metadata_only_ids = set(video_assets.loc[video_assets["processing_status"] == "metadata_only", "video_asset_id"].astype(str))
    for r in records:
        if r["modality"] == "video_frame":
            assert r["metadata"]["video_id"] not in metadata_only_ids
