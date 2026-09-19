"""Tests for src/fusion/automated_evaluation.py -- Day 3A Part 15 methods A and E."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.automated_evaluation import run_provenance_checks, run_synthetic_evaluation, synthetic_test_cases
from src.validation import guess_schema, validate

CONFIG = {
    "minimum_evidence": {"min_eligible_evidence_count": 1, "min_reliability_for_single_evidence": 0.5},
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


def test_synthetic_cases_never_use_real_claim_ids():
    cases = synthetic_test_cases()
    for units in cases.values():
        assert all(str(cid).startswith("synthetic_") for cid in units["claim_id"])


def test_synthetic_evaluation_returns_a_row_per_scenario():
    rows = run_synthetic_evaluation(CONFIG)
    test_case_ids = {r["test_case_id"] for r in rows}
    expected = {
        "no_evidence", "weak_evidence_only", "support_plus_challenge", "credible_challenge_only",
        "official_disclosure_only", "irrelevant_image_added", "irrelevant_video_added",
        "missing_modality", "absence_from_document", "certification_confirmation", "claim_validates_itself",
    }
    assert expected.issubset(test_case_ids)


def test_synthetic_evaluation_all_pass_with_correct_decision_logic():
    rows = run_synthetic_evaluation(CONFIG)
    failed = [r for r in rows if not r["passed"]]
    assert failed == [], f"unexpected synthetic evaluation failures: {failed}"


def test_synthetic_evaluation_rows_never_leak_into_real_claim_id_field():
    rows = run_synthetic_evaluation(CONFIG)
    for r in rows:
        assert r["claim_id"] == ""  # synthetic results are keyed by test_case_id, not claim_id
        assert r["evaluation_type"] == "synthetic_test_case"


def test_provenance_checks_flag_missing_evidence_id():
    units = pd.DataFrame([
        {"claim_id": "c1", "modality": "text", "evidence_id": "unknown_doc", "evidence_strength": "medium",
         "stance": "supports", "evidence_passage": "some text", "visual_gate_passed": True},
    ])
    fusion_results = pd.DataFrame([
        {"claim_id": "c1", "missing_modalities": "image;video;reference", "evaluation_status": "automated_not_human_validated"},
    ])
    known = {"text": {"known_doc_1", "known_doc_2"}}
    rows = run_provenance_checks(units, fusion_results, ["c1"], known)
    failures = [r for r in rows if not r["passed"]]
    assert any("evidence_id_exists" in r["expected_property"] for r in failures)


def test_provenance_checks_accept_nan_missing_modalities_as_disclosed_empty():
    units = pd.DataFrame(columns=["claim_id", "modality", "evidence_id", "evidence_strength", "stance", "evidence_passage", "visual_gate_passed"])
    fusion_results = pd.DataFrame([
        {"claim_id": "c1", "missing_modalities": float("nan"), "evaluation_status": "automated_not_human_validated"},
    ])
    rows = run_provenance_checks(units, fusion_results, ["c1"], {})
    disclosure_rows = [r for r in rows if r["expected_property"] == "missing_modalities_disclosed"]
    assert disclosure_rows and all(r["passed"] for r in disclosure_rows)


def test_provenance_checks_flag_visual_gate_violation():
    units = pd.DataFrame([
        {"claim_id": "c1", "modality": "image", "evidence_id": "img_1", "evidence_strength": "medium",
         "stance": "supports", "evidence_passage": "", "visual_gate_passed": False},
    ])
    fusion_results = pd.DataFrame([{"claim_id": "c1", "missing_modalities": "", "evaluation_status": "automated_not_human_validated"}])
    rows = run_provenance_checks(units, fusion_results, ["c1"], {})
    gate_rows = [r for r in rows if r["expected_property"] == "visual_gate_enforced"]
    assert gate_rows and not gate_rows[0]["passed"]


def test_provenance_checks_flag_missing_human_validation_disclosure():
    units = pd.DataFrame(columns=["claim_id", "modality", "evidence_id", "evidence_strength", "stance", "evidence_passage", "visual_gate_passed"])
    fusion_results = pd.DataFrame([{"claim_id": "c1", "missing_modalities": "", "evaluation_status": "manually_reviewed"}])
    rows = run_provenance_checks(units, fusion_results, ["c1"], {})
    disclosure_rows = [r for r in rows if r["expected_property"] == "no_human_validation_claimed"]
    assert disclosure_rows and not disclosure_rows[0]["passed"]


AUTOMATED_EVALUATION_OUTPUT = PROJECT_ROOT / "outputs" / "tables" / "automated_fusion_evaluation.csv"


def test_automated_fusion_evaluation_output_matches_its_schema():
    if not AUTOMATED_EVALUATION_OUTPUT.exists():
        return  # not yet generated in this environment -- run scripts/run_fusion_sensitivity.py
    assert guess_schema(AUTOMATED_EVALUATION_OUTPUT.name) == "automated_fusion_evaluation"
    df = pd.read_csv(AUTOMATED_EVALUATION_OUTPUT)
    passed, errors = validate(df, "automated_fusion_evaluation")
    assert passed, errors
