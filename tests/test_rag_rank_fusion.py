"""Tests for src/rag/rank_fusion.py -- deterministic reciprocal-rank fusion."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.rank_fusion import fuse_candidates


def _candidate(id_, collection, rank, similarity, metadata=None, direct_link=False):
    return {
        "id": id_, "collection": collection, "document": f"doc {id_}",
        "metadata": metadata or {"evidence_strength": "strong", "source_independence": "customer_reported", "source_type": "customer_experience"},
        "distance": 1 - similarity, "similarity": similarity, "original_rank": rank, "direct_link": direct_link,
    }


def test_fusion_is_deterministic():
    lists = {"a": [_candidate("x", "col", 0, 0.9), _candidate("y", "col", 1, 0.5)]}
    r1 = fuse_candidates(lists, max_results=5)
    r2 = fuse_candidates(lists, max_results=5)
    assert [e["id"] for e in r1] == [e["id"] for e in r2]


def test_higher_similarity_ranks_first_when_only_one_list():
    lists = {"a": [_candidate("low", "col", 1, 0.3), _candidate("high", "col", 0, 0.9)]}
    result = fuse_candidates(lists, max_results=5)
    kept = [e for e in result if e["trace"]["exclusion_reason"] is None]
    assert kept[0]["id"] == "high"


def test_direct_link_bonus_promotes_a_candidate():
    lists = {
        "semantic": [_candidate("semantic_only", "col", 0, 0.95)],
        "direct": [_candidate("direct_only", "col", 5, 0.1, direct_link=True)],
    }
    result = fuse_candidates(lists, max_results=5)
    kept = {e["id"]: e for e in result if e["trace"]["exclusion_reason"] is None}
    assert kept["direct_only"]["final_score"] > 0
    # the direct-linked candidate should not simply vanish despite a low similarity rank
    assert "direct_only" in kept


def test_deduplicates_candidate_appearing_in_multiple_lists():
    lists = {
        "a": [_candidate("dup", "col", 0, 0.8)],
        "b": [_candidate("dup", "col", 2, 0.6)],
    }
    result = fuse_candidates(lists, max_results=5)
    assert len([e for e in result if e["id"] == "dup"]) == 1
    assert len(result[0]["trace"]["sources"]) == 2


def test_max_results_caps_kept_candidates_and_marks_exclusion_reason():
    lists = {"a": [_candidate(f"id{i}", "col", i, 1.0 - i * 0.05) for i in range(10)]}
    result = fuse_candidates(lists, max_results=3)
    kept = [e for e in result if e["trace"]["exclusion_reason"] is None]
    excluded = [e for e in result if e["trace"]["exclusion_reason"] is not None]
    assert len(kept) == 3
    assert len(excluded) == 7
    assert all(e["trace"]["exclusion_reason"] == "max_results_exceeded" for e in excluded)


def test_source_diversity_cap_excludes_overrepresented_source_type():
    metas = [{"evidence_strength": "strong", "source_independence": "customer_reported", "source_type": "customer_experience"} for _ in range(5)]
    lists = {"a": [_candidate(f"id{i}", "col", i, 0.9 - i * 0.01, metadata=metas[i]) for i in range(5)]}
    result = fuse_candidates(lists, max_results=10, source_diversity_cap=2)
    kept = [e for e in result if e["trace"]["exclusion_reason"] is None]
    assert len(kept) == 2
    excluded_reasons = {e["trace"]["exclusion_reason"] for e in result if e["trace"]["exclusion_reason"]}
    assert any("source_diversity_cap_exceeded" in r for r in excluded_reasons)


def test_evidence_strength_and_independence_affect_score():
    strong_independent = _candidate("strong", "col", 0, 0.5, metadata={"evidence_strength": "strong", "source_independence": "independent", "source_type": "official_reference"})
    weak_self_reported = _candidate("weak", "col", 0, 0.5, metadata={"evidence_strength": "medium", "source_independence": "self_reported", "source_type": "official_claims"})
    lists = {"a": [strong_independent], "b": [weak_self_reported]}
    result = fuse_candidates(lists, max_results=5)
    scores = {e["id"]: e["final_score"] for e in result}
    assert scores["strong"] > scores["weak"]


def test_trace_retains_original_rank_and_collection():
    lists = {"a": [_candidate("x", "tala_text_evidence", 2, 0.7)]}
    result = fuse_candidates(lists, max_results=5)
    source = result[0]["trace"]["sources"][0]
    assert source["original_rank"] == 2
    assert source["collection"] == "tala_text_evidence"
