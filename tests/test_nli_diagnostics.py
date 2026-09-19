"""Tests for the Day 3A NLI disagreement diagnostic (src/fusion/nli_diagnostics.py,
src/fusion/text_evidence.py's NLI helpers). Skips if sentence-transformers /
the NLI checkpoint is unavailable in this environment -- never fabricates a
result when the model cannot be loaded."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.text_evidence import nli_available

pytestmark = pytest.mark.skipif(not nli_available(), reason="sentence-transformers unavailable in this environment")


# ── id2label / entailment-contradiction-neutral mapping ────────────────────────

def test_nli_label_indices_are_read_from_model_config_not_assumed():
    from src.fusion.text_evidence import nli_label_indices
    label2id = nli_label_indices()
    assert set(label2id.keys()) == {"entailment", "contradiction", "neutral"}
    assert len(set(label2id.values())) == 3  # three distinct indices, no collision


def test_nli_label_indices_match_known_checkpoint_ordering():
    """Pins the specific ordering of cross-encoder/nli-deberta-v3-xsmall so a
    silent checkpoint change is caught, without assuming this ordering
    elsewhere in the codebase."""
    from src.fusion.text_evidence import nli_label_indices
    label2id = nli_label_indices()
    assert label2id == {"contradiction": 0, "entailment": 1, "neutral": 2}


# ── premise/hypothesis orientation ──────────────────────────────────────────────

def test_production_orientation_constants():
    from src.fusion.text_evidence import NLI_HYPOTHESIS_FIELD, NLI_PREMISE_FIELD
    assert NLI_PREMISE_FIELD == "evidence_text"
    assert NLI_HYPOTHESIS_FIELD == "claim_text"


def test_orientation_choice_materially_affects_predictions():
    """Regression: confirms the wiring (not just the constants) -- across the
    synthetic calibration set, swapping premise/hypothesis order must change
    the predicted label for at least some cases (NLI is not symmetric in
    premise/hypothesis), and the two orientations must not score identically."""
    from src.fusion.nli_diagnostics import run_calibration

    forward_rows = {r["case_id"]: r["predicted_label"] for r in run_calibration("evidence_premise_claim_hypothesis")}
    reverse_rows = {r["case_id"]: r["predicted_label"] for r in run_calibration("claim_premise_evidence_hypothesis")}
    n_differ = sum(1 for cid in forward_rows if forward_rows[cid] != reverse_rows[cid])
    assert n_differ > 0, "premise/hypothesis orientation had zero effect on any synthetic case"


# ── synthetic calibration: entailment / contradiction / neutral ────────────────

def test_synthetic_entailment_case_predicted_correctly_or_reported():
    from src.fusion.nli_diagnostics import ORIENTATIONS, run_calibration
    rows = run_calibration("evidence_premise_claim_hypothesis")
    entailment_rows = [r for r in rows if r["expected_label"] == "entailment"]
    assert len(entailment_rows) == 6
    # At least a majority of clear entailment cases must be recognised as such --
    # a total failure here would indicate a real defect, not a limitation.
    n_correct = sum(r["correct"] for r in entailment_rows)
    assert n_correct >= 3, f"only {n_correct}/6 clear entailment cases recognised"


def test_synthetic_contradiction_case_predicted_correctly_or_reported():
    from src.fusion.nli_diagnostics import run_calibration
    rows = run_calibration("evidence_premise_claim_hypothesis")
    contradiction_rows = [r for r in rows if r["expected_label"] == "contradiction"]
    assert len(contradiction_rows) == 6
    n_correct = sum(r["correct"] for r in contradiction_rows)
    assert n_correct >= 3, f"only {n_correct}/6 clear contradiction cases recognised"


def test_synthetic_neutral_case_predicted_correctly():
    from src.fusion.nli_diagnostics import run_calibration
    rows = run_calibration("evidence_premise_claim_hypothesis")
    neutral_rows = [r for r in rows if r["expected_label"] == "neutral"]
    assert len(neutral_rows) == 6
    n_correct = sum(r["correct"] for r in neutral_rows)
    assert n_correct == 6, "unrelated evidence must never be misread as entailment/contradiction"


def test_synthetic_calibration_covers_taxonomy_categories():
    from src.fusion.nli_diagnostics import SYNTHETIC_CALIBRATION_CASES
    categories = {c["category"] for c in SYNTHETIC_CALIBRATION_CASES}
    assert categories == {"fit", "sizing", "durability", "returns", "materials", "certification"}
    assert len(SYNTHETIC_CALIBRATION_CASES) >= 18


def test_synthetic_cases_are_not_copied_into_real_evidence():
    import pandas as pd
    from src.fusion.nli_diagnostics import SYNTHETIC_CALIBRATION_CASES

    units_path = PROJECT_ROOT / "data" / "processed" / "fusion" / "claim_evidence_units.csv"
    if not units_path.exists():
        pytest.skip("claim_evidence_units.csv not generated in this environment")
    real_evidence_texts = set(pd.read_csv(units_path)["evidence_text"].fillna("").astype(str))
    for case in SYNTHETIC_CALIBRATION_CASES:
        assert case["evidence_text"] not in real_evidence_texts


# ── low-confidence abstention ────────────────────────────────────────────────────

def test_nli_label_from_probs_respects_threshold():
    from src.fusion.text_evidence import nli_label_from_probs
    # low-confidence probabilities (all near 1/3) must fall back to neutral, never
    # a confident-looking entailment/contradiction label.
    low_confidence_probs = {"entailment": 0.34, "contradiction": 0.33, "neutral": 0.33}
    assert nli_label_from_probs(low_confidence_probs, threshold=0.5) == "neutral"


def test_categorize_disagreement_flags_low_confidence():
    from src.fusion.nli_diagnostics import categorize_disagreement
    category = categorize_disagreement(
        rule_label="supports", nli_label="neutral", nli_confidence=0.4, confidence_threshold=0.6,
        category_gate_passed=True, truncation_occurred=False, negation_present=False,
        compound_claim=False, orientation_agrees_alt=False, nli_result_missing=False,
    )
    assert category == "low_nli_confidence"


def test_categorize_disagreement_returns_empty_string_on_agreement():
    from src.fusion.nli_diagnostics import categorize_disagreement
    category = categorize_disagreement(
        rule_label="supports", nli_label="entailment", nli_confidence=0.9, confidence_threshold=0.6,
        category_gate_passed=True, truncation_occurred=False, negation_present=False,
        compound_claim=False, orientation_agrees_alt=False, nli_result_missing=False,
    )
    assert category == ""


def test_categorize_disagreement_never_returns_a_disallowed_category():
    from src.fusion.nli_diagnostics import categorize_disagreement
    allowed = {
        "label_mapping_error", "orientation_error", "low_nli_confidence", "rule_keyword_false_positive",
        "rule_keyword_false_negative", "compound_claim", "negation_scope", "neutral_vs_unclear_mapping",
        "category_mismatch", "truncated_input", "genuine_method_disagreement", "unresolved", "",
    }
    scenarios = [
        dict(rule_label="challenges", nli_label="neutral", nli_confidence=0.9, confidence_threshold=0.6,
             category_gate_passed=True, truncation_occurred=True, negation_present=False,
             compound_claim=False, orientation_agrees_alt=False, nli_result_missing=False),
        dict(rule_label="neutral_context", nli_label="entailment", nli_confidence=0.9, confidence_threshold=0.6,
             category_gate_passed=False, truncation_occurred=False, negation_present=False,
             compound_claim=False, orientation_agrees_alt=False, nli_result_missing=False),
        dict(rule_label="unclear", nli_label="entailment", nli_confidence=0.9, confidence_threshold=0.6,
             category_gate_passed=True, truncation_occurred=False, negation_present=False,
             compound_claim=False, orientation_agrees_alt=False, nli_result_missing=False),
        dict(rule_label="supports", nli_label="neutral", nli_confidence=0.9, confidence_threshold=0.6,
             category_gate_passed=True, truncation_occurred=False, negation_present=True,
             compound_claim=True, orientation_agrees_alt=False, nli_result_missing=False),
        dict(rule_label="supports", nli_label="contradiction", nli_confidence=0.9, confidence_threshold=0.6,
             category_gate_passed=True, truncation_occurred=False, negation_present=False,
             compound_claim=False, orientation_agrees_alt=True, nli_result_missing=False),
        dict(rule_label="supports", nli_label="contradiction", nli_confidence=0.9, confidence_threshold=0.6,
             category_gate_passed=True, truncation_occurred=False, negation_present=False,
             compound_claim=False, orientation_agrees_alt=False, nli_result_missing=True),
        dict(rule_label="supports", nli_label="contradiction", nli_confidence=0.9, confidence_threshold=0.6,
             category_gate_passed=True, truncation_occurred=False, negation_present=False,
             compound_claim=False, orientation_agrees_alt=False, nli_result_missing=False),
    ]
    for kwargs in scenarios:
        assert categorize_disagreement(**kwargs) in allowed


# ── NLI-disabled fusion behaviour / auxiliary-only guarantee ────────────────────

def test_nli_has_no_call_site_in_primary_fusion_pipeline():
    """Regression: nli_stance_if_available must remain an auxiliary evaluator
    only -- if it is ever wired into run_primary_multimodal_fusion.py or
    evidence_fusion.py, it must start affecting fusion decisions silently."""
    fusion_script = (PROJECT_ROOT / "scripts" / "run_primary_multimodal_fusion.py").read_text(encoding="utf-8")
    evidence_fusion_module = (PROJECT_ROOT / "src" / "fusion" / "evidence_fusion.py").read_text(encoding="utf-8")
    assert "nli_stance_if_available" not in fusion_script
    assert "nli_stance_if_available" not in evidence_fusion_module


def test_fuse_claim_result_identical_regardless_of_nli(monkeypatch):
    """NLI-disabled fusion behaviour: forcing nli_available() to False must not
    change a claim's fused result, because fuse_claim never calls it."""
    import pandas as pd
    from src.fusion.evidence_fusion import fuse_claim
    import src.fusion.text_evidence as text_evidence_mod

    config = {
        "minimum_evidence": {"min_eligible_evidence_count": 1, "min_reliability_for_single_evidence": 0.5},
        "label_decision": {"aligned_min_support_ratio": 0.7, "partial_min_support_ratio": 0.34, "divergent_min_challenge_reliability": 0.5},
        "confidence": {"base": 0.5, "per_eligible_evidence_unit": 0.05, "max_evidence_bonus": 0.3,
                       "independent_evidence_bonus": 0.1, "conflicting_evidence_penalty": 0.15,
                       "single_evidence_penalty": 0.15, "cap_min": 0.05, "cap_max": 0.95},
        "presentation_restrictions": {"aligned": "may_present_as_automated_finding", "partially_aligned": "may_present_as_automated_finding",
                                       "mixed": "present_with_strong_caveat", "divergent": "present_with_strong_caveat",
                                       "insufficient_evidence": "may_present_as_automated_finding"},
    }
    units = pd.DataFrame([{
        "claim_id": "c1", "claim_text": "x", "claim_category": "materials", "modality": "text",
        "stance": "supports", "stance_score": 0.5, "reliability_score": 0.8,
        "source_independence": "customer_reported", "visual_gate_passed": True,
    }])

    result_with_nli = fuse_claim("c1", units, config)
    monkeypatch.setattr(text_evidence_mod, "nli_available", lambda: False)
    result_without_nli = fuse_claim("c1", units, config)
    assert result_with_nli == result_without_nli


# ── no fabricated agreement score ────────────────────────────────────────────────

def test_no_fabricated_agreement_when_model_unavailable(monkeypatch):
    import src.fusion.text_evidence as text_evidence_mod
    monkeypatch.setattr(text_evidence_mod, "nli_available", lambda: False)
    assert text_evidence_mod.nli_stance_if_available("claim", "evidence") is None


def test_disagreement_analysis_output_never_reports_perfect_fabricated_agreement():
    import pandas as pd
    path = PROJECT_ROOT / "outputs" / "tables" / "nli_rule_agreement_by_scope.csv"
    if not path.exists():
        pytest.skip("nli_rule_agreement_by_scope.csv not generated in this environment")
    df = pd.read_csv(path)
    # A real disagreement analysis on genuine mixed-sentiment customer text should
    # never land at exactly 1.0 -- that would indicate a hardcoded/fabricated result.
    assert (df["raw_agreement"] < 1.0).all()
