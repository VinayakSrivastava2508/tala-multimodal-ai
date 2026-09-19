"""Day 3A Part 12: decision-level multimodal evidence fusion.

Transparent, rule-based aggregation of claim_evidence_units.csv rows into one
automated (never human-validated) result per claim. No weight here was fit
against a human-labelled ground truth -- none exists for this academic
prototype (Part 3) -- every threshold is read from configs/fusion_rules.yaml
so it can be inspected and perturbed by scripts/run_fusion_sensitivity.py.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

INTERNAL_LABELS = ("aligned", "partially_aligned", "mixed", "divergent", "insufficient_evidence")

BUSINESS_FACING_LABELS = {
    "aligned": "automated alignment signal",
    "partially_aligned": "automated partial-alignment signal",
    "mixed": "automated mixed-evidence signal",
    "divergent": "potential divergence",
    "insufficient_evidence": "insufficient evidence",
}

ALL_MODALITIES = ("text", "image", "video", "reference")


def _is_eligible(row: pd.Series) -> bool:
    """An evidence unit is 'eligible' (can move the needle) only if it has a
    stance of supports/challenges AND (for image/video) passed the visual
    gate. Context-only / neutral_context / not_applicable units never count
    toward evidence_conflict or the label decision -- they may still appear
    in multimodal_context reporting."""
    if row["stance"] not in ("supports", "challenges"):
        return False
    if row["modality"] in ("image", "video") and not bool(row.get("visual_gate_passed", False)):
        return False
    return True


def fuse_claim(claim_id: str, claim_units: pd.DataFrame, config: dict) -> Dict:
    """Fuse all evidence units for ONE claim into a single automated result."""
    units = claim_units.copy()
    units["_eligible"] = units.apply(_is_eligible, axis=1)
    eligible = units[units["_eligible"]]

    # A single eligible evidence unit must clear a minimum reliability bar to
    # move the label -- otherwise one weak/low-independence unit alone could
    # produce "aligned"/"divergent" as confidently as strong evidence would.
    min_reliability_solo = config["minimum_evidence"]["min_reliability_for_single_evidence"]
    if len(eligible) == 1 and float(eligible.iloc[0]["reliability_score"]) < min_reliability_solo:
        units.loc[eligible.index, "_eligible"] = False
        eligible = units[units["_eligible"]]

    supports = eligible[eligible["stance"] == "supports"]
    challenges = eligible[eligible["stance"] == "challenges"]

    official_disclosure_support = int((supports["source_independence"] == "self_reported").sum())
    independent_support = int((supports["source_independence"] == "independent").sum())
    customer_support = supports[supports["source_independence"].isin(["customer_reported", "independent"])]
    customer_challenge = challenges[challenges["source_independence"].isin(["customer_reported", "independent"])]

    min_challenge_reliability = config["label_decision"]["divergent_min_challenge_reliability"]
    credible_challenges = customer_challenge[customer_challenge["reliability_score"] >= min_challenge_reliability]
    credible_support_exists = len(customer_support) > 0
    credible_challenge_exists = len(credible_challenges) > 0

    all_non_self_reported_eligible = eligible[eligible["source_independence"] != "self_reported"]
    self_reported_only = len(eligible) > 0 and len(all_non_self_reported_eligible) == 0

    missing_modalities = [m for m in ALL_MODALITIES if m not in set(units["modality"])]
    context_units = units[~units["_eligible"]]
    multimodal_context = len(context_units)
    evidence_count = len(units)
    eligible_evidence_count = len(eligible)

    aligned_ratio = config["label_decision"]["aligned_min_support_ratio"]
    partial_ratio = config["label_decision"]["partial_min_support_ratio"]

    self_reported_only_flag = False
    if eligible_evidence_count == 0:
        automated_label = "insufficient_evidence"
    elif self_reported_only:
        automated_label = "partially_aligned"
        self_reported_only_flag = True
    elif credible_support_exists and credible_challenge_exists:
        automated_label = "mixed"
    elif credible_challenge_exists and not credible_support_exists:
        automated_label = "divergent"
    elif credible_support_exists and not credible_challenge_exists:
        total_customer_indep = len(customer_support) + len(customer_challenge)
        support_ratio = len(customer_support) / total_customer_indep if total_customer_indep else 0.0
        automated_label = "aligned" if support_ratio >= aligned_ratio else "partially_aligned" if support_ratio >= partial_ratio else "partially_aligned"
    else:
        automated_label = "insufficient_evidence"

    evidence_conflict = credible_support_exists and credible_challenge_exists

    conf_cfg = config["confidence"]
    confidence = conf_cfg["base"]
    confidence += min(eligible_evidence_count * conf_cfg["per_eligible_evidence_unit"], conf_cfg["max_evidence_bonus"])
    if independent_support > 0:
        confidence += conf_cfg["independent_evidence_bonus"]
    if evidence_conflict:
        confidence -= conf_cfg["conflicting_evidence_penalty"]
    if eligible_evidence_count == 1:
        confidence -= conf_cfg["single_evidence_penalty"]
    confidence = round(max(conf_cfg["cap_min"], min(conf_cfg["cap_max"], confidence)), 4)

    presentation_restriction = config["presentation_restrictions"][automated_label]
    provisional_finding = True  # every real claim result -- Part 3
    external_validation_recommended = automated_label in ("mixed", "divergent") or confidence < 0.5

    claim_text = units["claim_text"].iloc[0] if len(units) else ""
    claim_category = units["claim_category"].iloc[0] if len(units) else ""

    return {
        "claim_id": claim_id, "claim_text": claim_text, "claim_category": claim_category,
        "official_disclosure_support": official_disclosure_support,
        "customer_experience_alignment": int(len(customer_support)) - int(len(customer_challenge)),
        "independent_support": independent_support,
        "multimodal_context": multimodal_context,
        "evidence_conflict": bool(evidence_conflict),
        "missing_modalities": ";".join(missing_modalities),
        "evidence_count": evidence_count,
        "eligible_evidence_count": eligible_evidence_count,
        "self_reported_only": self_reported_only_flag,
        "confidence": confidence,
        "automated_label": automated_label,
        "business_facing_label": BUSINESS_FACING_LABELS[automated_label],
        "evaluation_status": "automated_not_human_validated",
        "provisional_finding": provisional_finding,
        "external_validation_recommended": external_validation_recommended,
        "presentation_restriction": presentation_restriction,
    }


def fuse_all_claims(evidence_units: pd.DataFrame, config: dict, claims_universe: List[str] | None = None) -> pd.DataFrame:
    """Fuse every claim present in evidence_units. If claims_universe is given,
    claims with NO evidence units at all still receive an insufficient_evidence
    result (every real claim gets a result -- Part 20 acceptance criterion 1)."""
    results = []
    seen = set()
    for claim_id, group in evidence_units.groupby("claim_id"):
        results.append(fuse_claim(claim_id, group, config))
        seen.add(claim_id)

    if claims_universe:
        for claim_id in claims_universe:
            if claim_id not in seen:
                results.append({
                    "claim_id": claim_id, "claim_text": "", "claim_category": "",
                    "official_disclosure_support": 0, "customer_experience_alignment": 0,
                    "independent_support": 0, "multimodal_context": 0, "evidence_conflict": False,
                    "missing_modalities": ";".join(ALL_MODALITIES), "evidence_count": 0,
                    "eligible_evidence_count": 0, "self_reported_only": False,
                    "confidence": config["confidence"]["cap_min"], "automated_label": "insufficient_evidence",
                    "business_facing_label": BUSINESS_FACING_LABELS["insufficient_evidence"],
                    "evaluation_status": "automated_not_human_validated", "provisional_finding": True,
                    "external_validation_recommended": False,
                    "presentation_restriction": config["presentation_restrictions"]["insufficient_evidence"],
                })
    return pd.DataFrame(results)
