"""Tests for src/fusion/evidence_fusion.py -- decision-level fusion."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.evidence_fusion import BUSINESS_FACING_LABELS, fuse_all_claims, fuse_claim

CONFIG = {
    "minimum_evidence": {
        "min_eligible_evidence_count": 1, "min_reliability_for_single_evidence": 0.5,
    },
    "label_decision": {
        "aligned_min_support_ratio": 0.7, "partial_min_support_ratio": 0.34,
        "divergent_min_challenge_reliability": 0.5,
    },
    "confidence": {
        "base": 0.5, "per_eligible_evidence_unit": 0.05, "max_evidence_bonus": 0.3,
        "independent_evidence_bonus": 0.1, "conflicting_evidence_penalty": 0.15,
        "single_evidence_penalty": 0.15, "cap_min": 0.05, "cap_max": 0.95,
    },
    "presentation_restrictions": {
        "aligned": "may_present_as_automated_finding", "partially_aligned": "may_present_as_automated_finding",
        "mixed": "present_with_strong_caveat", "divergent": "present_with_strong_caveat",
        "insufficient_evidence": "may_present_as_automated_finding",
    },
}

BASE_COLS = ["claim_id", "claim_text", "claim_category", "modality", "stance", "stance_score",
             "reliability_score", "source_independence", "visual_gate_passed"]


def _units(rows):
    for r in rows:
        r.setdefault("claim_text", "x")
        r.setdefault("claim_category", "materials")
        r.setdefault("reliability_score", 0.8)
        r.setdefault("visual_gate_passed", True)
    return pd.DataFrame(rows, columns=BASE_COLS)


# ── no evidence -> insufficient_evidence ──────────────────────────────────────

def test_no_evidence_is_insufficient():
    units = _units([{"claim_id": "c1", "modality": "text", "stance": "not_applicable", "stance_score": 0.0, "source_independence": "unclear"}])
    result = fuse_claim("c1", units, CONFIG)
    assert result["automated_label"] == "insufficient_evidence"


def test_only_context_only_evidence_is_insufficient():
    units = _units([{"claim_id": "c1", "modality": "image", "stance": "neutral_context", "stance_score": 0.0, "source_independence": "unclear"}])
    result = fuse_claim("c1", units, CONFIG)
    assert result["automated_label"] == "insufficient_evidence"


# ── official claim cannot validate itself / self-reported-only ────────────────

def test_self_reported_only_support_is_partially_aligned_not_aligned():
    units = _units([{"claim_id": "c1", "modality": "reference", "stance": "supports", "stance_score": 0.4, "source_independence": "self_reported"}])
    result = fuse_claim("c1", units, CONFIG)
    assert result["automated_label"] == "partially_aligned"
    assert result["self_reported_only"] is True


def test_self_reported_never_produces_aligned_alone():
    for i in range(5):
        units = _units([{"claim_id": "c1", "modality": "reference", "stance": "supports", "stance_score": 0.4, "source_independence": "self_reported"}] * (i + 1))
        result = fuse_claim("c1", units, CONFIG)
        assert result["automated_label"] != "aligned"


# ── official disclosure separated from independent support ────────────────────

def test_official_disclosure_and_independent_support_are_counted_separately():
    units = _units([
        {"claim_id": "c1", "modality": "reference", "stance": "supports", "stance_score": 0.4, "source_independence": "self_reported"},
        {"claim_id": "c1", "modality": "reference", "stance": "supports", "stance_score": 0.7, "source_independence": "independent"},
    ])
    result = fuse_claim("c1", units, CONFIG)
    assert result["official_disclosure_support"] == 1
    assert result["independent_support"] == 1


# ── support + challenge -> mixed ────────────────────────────────────────────────

def test_support_and_credible_challenge_is_mixed():
    units = _units([
        {"claim_id": "c1", "modality": "text", "stance": "supports", "stance_score": 0.5, "source_independence": "customer_reported", "reliability_score": 0.8},
        {"claim_id": "c1", "modality": "text", "stance": "challenges", "stance_score": -0.5, "source_independence": "customer_reported", "reliability_score": 0.8},
    ])
    result = fuse_claim("c1", units, CONFIG)
    assert result["automated_label"] == "mixed"
    assert result["evidence_conflict"] is True


# ── credible challenge only -> divergent ────────────────────────────────────────

def test_credible_challenge_only_is_divergent():
    units = _units([
        {"claim_id": "c1", "modality": "text", "stance": "challenges", "stance_score": -0.5, "source_independence": "customer_reported", "reliability_score": 0.8},
    ])
    result = fuse_claim("c1", units, CONFIG)
    assert result["automated_label"] == "divergent"


def test_lone_low_reliability_support_does_not_produce_aligned():
    units = _units([
        {"claim_id": "c1", "modality": "text", "stance": "supports", "stance_score": 0.2, "source_independence": "customer_reported", "reliability_score": 0.1},
    ])
    result = fuse_claim("c1", units, CONFIG)
    assert result["automated_label"] == "insufficient_evidence"
    assert result["eligible_evidence_count"] == 0


def test_low_reliability_challenge_does_not_trigger_divergent():
    units = _units([
        {"claim_id": "c1", "modality": "text", "stance": "challenges", "stance_score": -0.2, "source_independence": "customer_reported", "reliability_score": 0.1},
    ])
    result = fuse_claim("c1", units, CONFIG)
    assert result["automated_label"] != "divergent"


# ── divergent is always provisional and restricted ─────────────────────────────

def test_divergent_is_provisional_and_restricted():
    units = _units([
        {"claim_id": "c1", "modality": "text", "stance": "challenges", "stance_score": -0.5, "source_independence": "customer_reported", "reliability_score": 0.8},
    ])
    result = fuse_claim("c1", units, CONFIG)
    assert result["automated_label"] == "divergent"
    assert result["provisional_finding"] is True
    assert result["external_validation_recommended"] is True
    assert result["presentation_restriction"] == "present_with_strong_caveat"


def test_divergent_never_presented_as_verified_contradiction():
    units = _units([
        {"claim_id": "c1", "modality": "text", "stance": "challenges", "stance_score": -0.5, "source_independence": "customer_reported", "reliability_score": 0.8},
    ])
    result = fuse_claim("c1", units, CONFIG)
    assert result["business_facing_label"] == "potential divergence"
    assert "verified" not in result["business_facing_label"].lower()
    assert "confirmed" not in result["business_facing_label"].lower()


# ── irrelevant modality cannot force a label change ────────────────────────────

def test_irrelevant_image_does_not_change_label():
    base_units = _units([
        {"claim_id": "c1", "modality": "text", "stance": "supports", "stance_score": 0.5, "source_independence": "customer_reported", "reliability_score": 0.8},
    ])
    result_base = fuse_claim("c1", base_units, CONFIG)

    with_context = _units([
        {"claim_id": "c1", "modality": "text", "stance": "supports", "stance_score": 0.5, "source_independence": "customer_reported", "reliability_score": 0.8},
        {"claim_id": "c1", "modality": "image", "stance": "neutral_context", "stance_score": 0.0, "source_independence": "unclear", "visual_gate_passed": False},
    ])
    result_with_context = fuse_claim("c1", with_context, CONFIG)
    assert result_base["automated_label"] == result_with_context["automated_label"]


def test_gated_out_image_never_counted_as_eligible():
    units = _units([
        {"claim_id": "c1", "modality": "image", "stance": "supports", "stance_score": 0.5, "source_independence": "unclear", "visual_gate_passed": False},
    ])
    result = fuse_claim("c1", units, CONFIG)
    assert result["eligible_evidence_count"] == 0
    assert result["automated_label"] == "insufficient_evidence"


# ── missing modalities disclosed ────────────────────────────────────────────────

def test_missing_modalities_are_recorded():
    units = _units([{"claim_id": "c1", "modality": "text", "stance": "supports", "stance_score": 0.5, "source_independence": "customer_reported"}])
    result = fuse_claim("c1", units, CONFIG)
    assert "image" in result["missing_modalities"]
    assert "video" in result["missing_modalities"]
    assert "reference" in result["missing_modalities"]
    assert "text" not in result["missing_modalities"]


# ── every real claim in the universe gets a result ─────────────────────────────

def test_fuse_all_claims_gives_every_claim_a_result_even_with_no_evidence():
    units = _units([{"claim_id": "c1", "modality": "text", "stance": "supports", "stance_score": 0.5, "source_independence": "customer_reported"}])
    out = fuse_all_claims(units, CONFIG, claims_universe=["c1", "c2", "c3"])
    assert set(out["claim_id"]) == {"c1", "c2", "c3"}
    c2_result = out[out["claim_id"] == "c2"].iloc[0]
    assert c2_result["automated_label"] == "insufficient_evidence"
    assert c2_result["evaluation_status"] == "automated_not_human_validated"


def test_every_result_has_required_evaluation_fields():
    units = _units([{"claim_id": "c1", "modality": "text", "stance": "supports", "stance_score": 0.5, "source_independence": "customer_reported"}])
    result = fuse_claim("c1", units, CONFIG)
    for field in ("evaluation_status", "provisional_finding", "external_validation_recommended", "presentation_restriction"):
        assert field in result
    assert result["evaluation_status"] == "automated_not_human_validated"
    assert result["provisional_finding"] is True


def test_business_facing_labels_cover_all_internal_labels():
    from src.fusion.evidence_fusion import INTERNAL_LABELS
    assert set(BUSINESS_FACING_LABELS.keys()) == set(INTERNAL_LABELS)


def test_no_output_field_claims_human_validation():
    units = _units([{"claim_id": "c1", "modality": "text", "stance": "supports", "stance_score": 0.5, "source_independence": "customer_reported"}])
    result = fuse_claim("c1", units, CONFIG)
    for key, value in result.items():
        if isinstance(value, str):
            assert "human_validated" not in value.lower() or "not_human_validated" in value.lower()
