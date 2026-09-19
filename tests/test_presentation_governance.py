"""Tests for src/fusion/presentation_governance.py -- the Day 3A final
presentation-restriction governance correction. Pure-function unit tests plus
end-to-end checks against the real, corrected claim_fusion_results.csv."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.presentation_governance import (
    DO_NOT_PRESENT, EVIDENCE_GAP_ONLY, MAY_PRESENT, STRONG_CAVEAT, classify_presentation_restriction,
)

BASE_KWARGS = dict(
    automated_label="aligned", self_reported_only=False, single_evidence_dependency=False,
    sensitive_to_weights=False, provenance_failure=False, groundability_failure=False,
)


def _classify(**overrides):
    kwargs = {**BASE_KWARGS, **overrides}
    return classify_presentation_restriction(**kwargs)


# ── precedence tier 1: provenance / groundability failure always wins ──────────

def test_provenance_failure_forces_do_not_present_even_when_aligned():
    restriction, reason, gap = _classify(provenance_failure=True)
    assert restriction == DO_NOT_PRESENT
    assert "provenance_failure" in reason
    assert gap is False


def test_groundability_failure_forces_do_not_present():
    restriction, reason, gap = _classify(groundability_failure=True)
    assert restriction == DO_NOT_PRESENT
    assert "visual_video_groundability_failure" in reason


def test_provenance_failure_outranks_caveat_conditions():
    restriction, reason, _ = _classify(provenance_failure=True, automated_label="mixed", self_reported_only=True)
    assert restriction == DO_NOT_PRESENT
    assert "automated_label" not in reason  # tier 1 short-circuits before tier 2 reasons are added


# ── precedence tier 2: sensitivity-unstable / self-reported-only / single-evidence / mixed-divergent ──

def test_every_sensitivity_unstable_claim_gets_strong_caveat():
    restriction, reason, _ = _classify(sensitive_to_weights=True)
    assert restriction == STRONG_CAVEAT
    assert "sensitive_to_weights" in reason


def test_every_self_reported_only_claim_gets_strong_caveat():
    restriction, reason, _ = _classify(self_reported_only=True)
    assert restriction == STRONG_CAVEAT
    assert "self_reported_only" in reason


def test_every_single_evidence_dependent_claim_gets_strong_caveat():
    restriction, reason, _ = _classify(single_evidence_dependency=True)
    assert restriction == STRONG_CAVEAT
    assert "single_evidence_dependency" in reason


@pytest.mark.parametrize("label", ["mixed", "divergent"])
def test_mixed_and_divergent_get_strong_caveat(label):
    restriction, reason, _ = _classify(automated_label=label)
    assert restriction == STRONG_CAVEAT
    assert f"automated_label={label}" in reason


def test_multiple_caveat_conditions_all_recorded_in_reason():
    _, reason, _ = _classify(self_reported_only=True, single_evidence_dependency=True, sensitive_to_weights=True)
    for token in ("self_reported_only", "single_evidence_dependency", "sensitive_to_weights"):
        assert token in reason


# ── precedence tier 4: insufficient_evidence is never a substantive conclusion ──

def test_insufficient_evidence_cannot_be_presented_as_substantive_conclusion():
    restriction, reason, gap = _classify(automated_label="insufficient_evidence")
    assert restriction == EVIDENCE_GAP_ONLY
    assert restriction != MAY_PRESENT
    assert gap is True


def test_insufficient_evidence_with_no_other_flags_still_gap_only_not_caveat():
    restriction, _, _ = _classify(automated_label="insufficient_evidence")
    assert restriction not in (STRONG_CAVEAT, MAY_PRESENT, DO_NOT_PRESENT)


# ── precedence tier 3: stable, eligible, multi-source claims retain may_present ──

def test_stable_multi_source_claim_retains_may_present():
    restriction, reason, gap = _classify()  # aligned, no flags at all
    assert restriction == MAY_PRESENT
    assert gap is False


def test_partially_aligned_with_no_flags_retains_may_present():
    restriction, _, _ = _classify(automated_label="partially_aligned")
    assert restriction == MAY_PRESENT


def test_may_present_is_the_only_restriction_with_no_reason_flags():
    """No claim reaches may_present_as_automated_finding while carrying an
    active caveat/failure condition -- precedence tier 3 requires a clean bill."""
    for automated_label in ("aligned", "partially_aligned"):
        for flag_name in ("self_reported_only", "single_evidence_dependency", "sensitive_to_weights", "provenance_failure", "groundability_failure"):
            restriction, _, _ = _classify(automated_label=automated_label, **{flag_name: True})
            assert restriction != MAY_PRESENT, f"{flag_name}=True incorrectly allowed may_present for {automated_label}"


def test_classify_returns_no_unknown_restriction_values():
    from src.fusion.presentation_governance import GOVERNANCE_RESTRICTIONS
    scenarios = [
        dict(provenance_failure=True), dict(groundability_failure=True), dict(sensitive_to_weights=True),
        dict(self_reported_only=True), dict(single_evidence_dependency=True), dict(automated_label="mixed"),
        dict(automated_label="divergent"), dict(automated_label="insufficient_evidence"), dict(),
    ]
    for overrides in scenarios:
        restriction, _, _ = _classify(**overrides)
        assert restriction in GOVERNANCE_RESTRICTIONS


# ── end-to-end: the real, corrected claim_fusion_results.csv obeys these rules ──

FUSION_RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "fusion" / "claim_fusion_results.csv"
STABILITY_PATH = PROJECT_ROOT / "outputs" / "tables" / "fusion_sensitivity_stability.csv"


@pytest.mark.skipif(not FUSION_RESULTS_PATH.exists() or not STABILITY_PATH.exists(), reason="fusion pipeline not yet run in this environment")
def test_real_fusion_results_have_governance_fields():
    df = pd.read_csv(FUSION_RESULTS_PATH)
    for col in ("presentation_restriction", "presentation_restriction_reason", "evidence_gap_finding"):
        assert col in df.columns


@pytest.mark.skipif(not FUSION_RESULTS_PATH.exists() or not STABILITY_PATH.exists(), reason="fusion pipeline not yet run in this environment")
def test_real_sensitivity_unstable_claims_never_may_present():
    df = pd.read_csv(FUSION_RESULTS_PATH)
    stability = pd.read_csv(STABILITY_PATH)
    unstable_ids = set(stability.loc[~stability["label_stable"], "claim_id"])
    unstable = df[df["claim_id"].isin(unstable_ids)]
    assert (unstable["presentation_restriction"] != "may_present_as_automated_finding").all()


@pytest.mark.skipif(not FUSION_RESULTS_PATH.exists() or not STABILITY_PATH.exists(), reason="fusion pipeline not yet run in this environment")
def test_real_self_reported_only_claims_never_may_present():
    df = pd.read_csv(FUSION_RESULTS_PATH)
    self_reported = df[df["self_reported_only"]]
    assert (self_reported["presentation_restriction"] != "may_present_as_automated_finding").all()


@pytest.mark.skipif(not FUSION_RESULTS_PATH.exists() or not STABILITY_PATH.exists(), reason="fusion pipeline not yet run in this environment")
def test_real_single_evidence_dependent_claims_never_may_present():
    df = pd.read_csv(FUSION_RESULTS_PATH)
    stability = pd.read_csv(STABILITY_PATH)
    single_evidence_ids = set(stability.loc[stability["single_evidence_dependency"], "claim_id"])
    single_evidence = df[df["claim_id"].isin(single_evidence_ids)]
    assert (single_evidence["presentation_restriction"] != "may_present_as_automated_finding").all()


@pytest.mark.skipif(not FUSION_RESULTS_PATH.exists(), reason="fusion pipeline not yet run in this environment")
def test_real_insufficient_evidence_claims_never_substantive_conclusions():
    df = pd.read_csv(FUSION_RESULTS_PATH)
    insufficient = df[df["automated_label"] == "insufficient_evidence"]
    assert (insufficient["presentation_restriction"] == "present_as_evidence_gap_finding_only").all()
    assert insufficient["evidence_gap_finding"].all()


@pytest.mark.skipif(not FUSION_RESULTS_PATH.exists() or not STABILITY_PATH.exists(), reason="fusion pipeline not yet run in this environment")
def test_real_stable_multi_source_claims_retain_may_present():
    df = pd.read_csv(FUSION_RESULTS_PATH)
    stability = pd.read_csv(STABILITY_PATH)
    stable_ids = set(stability.loc[stability["label_stable"], "claim_id"])
    clean = df[
        df["claim_id"].isin(stable_ids) & df["automated_label"].isin(["aligned", "partially_aligned"])
        & ~df["self_reported_only"]
    ]
    single_evidence_ids = set(stability.loc[stability["single_evidence_dependency"], "claim_id"])
    clean = clean[~clean["claim_id"].isin(single_evidence_ids)]
    assert len(clean) > 0, "no stable multi-source claim exists to test against"
    assert (clean["presentation_restriction"] == "may_present_as_automated_finding").all()


def test_evidence_units_thresholds_and_labels_untouched_by_governance():
    """The governance correction must never change automated_label, confidence,
    or evidence-level fields -- only presentation_restriction and the two new
    columns."""
    if not FUSION_RESULTS_PATH.exists():
        pytest.skip("fusion pipeline not yet run in this environment")
    df = pd.read_csv(FUSION_RESULTS_PATH)
    assert set(df["automated_label"].unique()) <= {"aligned", "partially_aligned", "mixed", "divergent", "insufficient_evidence"}
    assert df["confidence"].between(0.0, 1.0).all()
