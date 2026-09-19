"""Tests for scripts/run_fusion_sensitivity.py's perturbation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_fusion_sensitivity import (
    apply_relevance_threshold, apply_stance_thresholds, perturb_min_evidence,
    perturb_relevance_thresholds, perturb_reliability_weights, perturb_stance_thresholds,
    remove_image_video_context, remove_medium_strength, remove_self_reported,
)

BASE_CONFIG = {
    "source_reliability_tiers": {"independent": 1.0, "customer_reported": 0.85, "self_reported": 0.5, "unclear": 0.4},
    "relevance_thresholds": {"text_relevance_min": 0.35, "image_relevance_min": 0.27, "video_relevance_min": 0.24, "reference_relevance_min": 0.35},
    "stance_thresholds": {"supports_polarity_min": 0.15, "challenges_polarity_max": -0.15, "unclear_subjectivity_min": 0.85},
    "minimum_evidence": {"min_eligible_evidence_count": 1},
}


def test_perturb_reliability_weights_shifts_and_clamps():
    c = perturb_reliability_weights(BASE_CONFIG, 0.9)
    assert c["source_reliability_tiers"]["independent"] == 1.0  # clamped at 1.0
    assert c["source_reliability_tiers"]["unclear"] == 1.0
    assert BASE_CONFIG["source_reliability_tiers"]["independent"] == 1.0  # original untouched


def test_perturb_relevance_thresholds_never_goes_negative():
    c = perturb_relevance_thresholds(BASE_CONFIG, -1.0)
    for v in c["relevance_thresholds"].values():
        assert v >= 0.0


def test_perturb_stance_thresholds_widens_or_narrows_band():
    c = perturb_stance_thresholds(BASE_CONFIG, 0.05)
    assert round(c["stance_thresholds"]["supports_polarity_min"], 4) == 0.20
    assert round(c["stance_thresholds"]["challenges_polarity_max"], 4) == -0.20


def test_perturb_min_evidence_sets_value():
    c = perturb_min_evidence(BASE_CONFIG, 2)
    assert c["minimum_evidence"]["min_eligible_evidence_count"] == 2


def _units():
    return pd.DataFrame([
        {"claim_id": "c1", "modality": "text", "stance": "supports", "stance_score": 0.2,
         "relevance_score": 0.5, "source_independence": "customer_reported", "evidence_strength": "medium"},
        {"claim_id": "c1", "modality": "reference", "stance": "supports", "stance_score": 0.4,
         "relevance_score": 0.4, "source_independence": "self_reported", "evidence_strength": "strong"},
        {"claim_id": "c1", "modality": "image", "stance": "neutral_context", "stance_score": 0.0,
         "relevance_score": 0.3, "source_independence": "unclear", "evidence_strength": "weak"},
    ])


def test_apply_relevance_threshold_downgrades_below_threshold_units():
    units = _units()
    strict = {"relevance_thresholds": {"text_relevance_min": 0.9, "image_relevance_min": 0.27,
                                        "video_relevance_min": 0.24, "reference_relevance_min": 0.35}}
    out = apply_relevance_threshold(units, strict)
    text_row = out[out["modality"] == "text"].iloc[0]
    assert text_row["stance"] == "not_applicable"
    assert text_row["stance_score"] == 0.0


def test_apply_stance_thresholds_only_affects_text():
    units = _units()
    widened = {"stance_thresholds": {"supports_polarity_min": 0.5, "challenges_polarity_max": -0.5}}
    out = apply_stance_thresholds(units, widened)
    text_row = out[out["modality"] == "text"].iloc[0]
    assert text_row["stance"] == "neutral_context"  # 0.2 no longer clears the raised 0.5 bar
    ref_row = out[out["modality"] == "reference"].iloc[0]
    assert ref_row["stance"] == "supports"  # non-text left untouched


def test_remove_self_reported_drops_only_self_reported_rows():
    out = remove_self_reported(_units())
    assert "self_reported" not in set(out["source_independence"])
    assert len(out) == 2


def test_remove_medium_strength_drops_only_medium():
    out = remove_medium_strength(_units())
    assert "medium" not in set(out["evidence_strength"])
    assert len(out) == 2


def test_remove_image_video_context_drops_only_contextual_visual_units():
    out = remove_image_video_context(_units())
    assert not ((out["modality"] == "image") & (out["stance"] == "neutral_context")).any()
    assert len(out) == 2
