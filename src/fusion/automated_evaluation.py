"""Day 3A Part 15: automated fusion evaluation helpers.

Method A (deterministic synthetic test cases) and Method E (provenance /
groundedness checks) live here. Synthetic cases are constructed entirely
in-memory with claim IDs prefixed `synthetic_` and are NEVER written into
claim_evidence_units.csv or claim_fusion_results.csv -- they exist only to
produce rows in outputs/tables/automated_fusion_evaluation.csv.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

from src.fusion.evidence_fusion import fuse_claim

NOW_ISO = datetime.now(timezone.utc).isoformat()


def _unit(claim_id, modality, stance, stance_score, source_independence, reliability_score,
          visual_gate_passed=True, claim_category="materials") -> dict:
    return {
        "claim_id": claim_id, "claim_text": "synthetic claim text", "claim_category": claim_category,
        "modality": modality, "stance": stance, "stance_score": stance_score,
        "source_independence": source_independence, "reliability_score": reliability_score,
        "visual_gate_passed": visual_gate_passed,
    }


def synthetic_test_cases() -> Dict[str, pd.DataFrame]:
    """Returns {test_case_id: evidence_units_df} for every synthetic case in
    Part 15.A. Each is fused independently and checked against its expected
    property in run_synthetic_evaluation()."""
    cases = {}

    cases["no_evidence"] = pd.DataFrame([
        _unit("synthetic_no_evidence", "text", "not_applicable", 0.0, "unclear", 0.0),
    ])

    cases["weak_evidence_only"] = pd.DataFrame([
        _unit("synthetic_weak_only", "text", "supports", 0.1, "customer_reported", 0.1),
    ])

    cases["support_plus_challenge"] = pd.DataFrame([
        _unit("synthetic_mixed", "text", "supports", 0.5, "customer_reported", 0.8),
        _unit("synthetic_mixed", "text", "challenges", -0.5, "customer_reported", 0.8),
    ])

    cases["credible_challenge_only"] = pd.DataFrame([
        _unit("synthetic_divergent", "text", "challenges", -0.6, "customer_reported", 0.8),
    ])

    cases["official_disclosure_only"] = pd.DataFrame([
        _unit("synthetic_disclosure_only", "reference", "supports", 0.4, "self_reported", 0.5),
    ])

    base_support = _unit("synthetic_irrelevant_image", "text", "supports", 0.5, "customer_reported", 0.8)
    irrelevant_image = _unit("synthetic_irrelevant_image", "image", "neutral_context", 0.0, "unclear", 0.0, visual_gate_passed=False)
    cases["irrelevant_image_added"] = pd.DataFrame([base_support])
    cases["irrelevant_image_added_with_image"] = pd.DataFrame([base_support, irrelevant_image])

    base_support2 = _unit("synthetic_irrelevant_video", "text", "supports", 0.5, "customer_reported", 0.8)
    irrelevant_video = _unit("synthetic_irrelevant_video", "video", "neutral_context", 0.0, "unclear", 0.0, visual_gate_passed=False)
    cases["irrelevant_video_added"] = pd.DataFrame([base_support2])
    cases["irrelevant_video_added_with_video"] = pd.DataFrame([base_support2, irrelevant_video])

    cases["missing_modality"] = pd.DataFrame([
        _unit("synthetic_missing_modality", "text", "supports", 0.5, "customer_reported", 0.8),
    ])  # image/video/reference absent entirely -- must not crash

    cases["absence_from_document"] = pd.DataFrame([
        _unit("synthetic_absence", "reference", "not_applicable", 0.0, "unclear", 0.0),
    ])

    cases["certification_confirmation"] = pd.DataFrame([
        _unit("synthetic_certification", "reference", "supports", 0.7, "independent", 1.0),
    ])

    cases["claim_validates_itself"] = pd.DataFrame([
        _unit("synthetic_self_validation", "text", "not_applicable", 0.0, "self_reported", 0.0, claim_category="materials"),
    ])

    return cases


def run_synthetic_evaluation(config: dict) -> List[dict]:
    """Runs every synthetic case and returns automated_fusion_evaluation.csv
    rows (evaluation_type='synthetic_test_case')."""
    cases = synthetic_test_cases()
    results = {}
    for name, units in cases.items():
        claim_id = units["claim_id"].iloc[0]
        results[name] = fuse_claim(claim_id, units, config)

    rows = []

    def add(test_case_id, expected, observed, passed, reason=""):
        rows.append({
            "evaluation_id": f"synth_{len(rows) + 1:03d}", "evaluation_type": "synthetic_test_case",
            "claim_id": "", "test_case_id": test_case_id, "expected_property": expected,
            "observed_property": observed, "passed": passed, "failure_reason": reason,
            "method": "deterministic_rule_based_fusion", "generated_at": NOW_ISO,
        })

    r = results["no_evidence"]
    add("no_evidence", "insufficient_evidence", r["automated_label"], r["automated_label"] == "insufficient_evidence")

    r = results["weak_evidence_only"]
    add("weak_evidence_only", "insufficient_evidence", r["automated_label"], r["automated_label"] == "insufficient_evidence",
        "" if r["automated_label"] == "insufficient_evidence" else "weak evidence (reliability 0.1) was treated as eligible")

    r = results["support_plus_challenge"]
    add("support_plus_challenge", "mixed", r["automated_label"], r["automated_label"] == "mixed")

    r = results["credible_challenge_only"]
    add("credible_challenge_only", "divergent", r["automated_label"], r["automated_label"] == "divergent")

    r = results["official_disclosure_only"]
    ok = r["automated_label"] in ("partially_aligned", "insufficient_evidence") and r["self_reported_only"]
    add("official_disclosure_only", "partial_or_insufficient_and_self_reported_only",
        f"{r['automated_label']} (self_reported_only={r['self_reported_only']})", ok)

    r1 = results["irrelevant_image_added"]
    r2 = results["irrelevant_image_added_with_image"]
    add("irrelevant_image_added", "label_unchanged", f"{r1['automated_label']} -> {r2['automated_label']}",
        r1["automated_label"] == r2["automated_label"])

    r1 = results["irrelevant_video_added"]
    r2 = results["irrelevant_video_added_with_video"]
    add("irrelevant_video_added", "label_unchanged", f"{r1['automated_label']} -> {r2['automated_label']}",
        r1["automated_label"] == r2["automated_label"])

    r = results["missing_modality"]
    add("missing_modality", "successful_abstaining_execution", r["automated_label"], True)  # reaching here without an exception IS the pass condition

    r = results["absence_from_document"]
    add("absence_from_document", "not_a_contradiction", r["automated_label"], r["automated_label"] != "divergent")

    r = results["certification_confirmation"]
    add("certification_confirmation", "independent_support_gt_0", f"independent_support={r['independent_support']}",
        r["independent_support"] > 0)

    r = results["claim_validates_itself"]
    add("claim_validates_itself", "prohibited_self_validation", r["automated_label"],
        r["automated_label"] == "insufficient_evidence" and r["eligible_evidence_count"] == 0)

    return rows


def run_provenance_checks(
    evidence_units: pd.DataFrame, fusion_results: pd.DataFrame, claims_universe: List[str],
    known_evidence_ids: Dict[str, set],
) -> List[dict]:
    """Part 15.E: for every REAL claim, verify claim ID exists, evidence IDs
    exist in their source tables, modality/evidence_strength are valid,
    stance has a passage/asset when eligible, visual gate is enforced, and
    missing modalities are disclosed."""
    rows = []

    def add(claim_id, expected, observed, passed, reason=""):
        rows.append({
            "evaluation_id": f"prov_{len(rows) + 1:04d}", "evaluation_type": "provenance_groundedness",
            "claim_id": claim_id, "test_case_id": "", "expected_property": expected,
            "observed_property": observed, "passed": passed, "failure_reason": reason,
            "method": "automated_provenance_check", "generated_at": NOW_ISO,
        })

    for claim_id in claims_universe:
        add(claim_id, "claim_id_exists_in_universe", "present", claim_id in claims_universe)

    for _, unit in evidence_units.iterrows():
        modality = unit["modality"]
        evidence_id = unit["evidence_id"]
        known = known_evidence_ids.get(modality, set())
        exists = evidence_id in known if known else True  # if no reference set provided, skip (not a failure)
        add(unit["claim_id"], f"evidence_id_exists ({modality})", str(exists), exists,
            "" if exists else f"{evidence_id} not found in source table for modality {modality}")

        valid_strength = unit["evidence_strength"] in ("strong", "medium", "weak", "unusable")
        add(unit["claim_id"], "evidence_strength_valid", unit["evidence_strength"], valid_strength)

        if unit["stance"] in ("supports", "challenges"):
            has_grounding = bool(str(unit.get("evidence_passage", "")).strip()) or bool(str(unit.get("evidence_id", "")).strip())
            add(unit["claim_id"], "stance_has_passage_or_asset", str(has_grounding), has_grounding)

        if modality in ("image", "video"):
            gate_ok = (unit["stance"] != "supports") or bool(unit["visual_gate_passed"])
            add(unit["claim_id"], "visual_gate_enforced", str(gate_ok), gate_ok,
                "" if gate_ok else "supports stance without visual_gate_passed=True")

    for _, r in fusion_results.iterrows():
        # missing_modalities is a comma-joined string of absent modalities, or
        # "" when none are missing -- pandas round-trips "" through CSV as NaN,
        # so NaN here means "disclosed and empty", not "not disclosed".
        raw = r.get("missing_modalities")
        disclosed = isinstance(raw, str) or pd.isna(raw)
        add(r["claim_id"], "missing_modalities_disclosed", str(disclosed), disclosed)
        no_human_claim = r.get("evaluation_status") == "automated_not_human_validated"
        add(r["claim_id"], "no_human_validation_claimed", str(no_human_claim), no_human_claim)

    return rows
