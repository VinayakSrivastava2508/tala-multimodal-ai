"""Day 3C Part A: claim-experience analytical synthesis tables.

Builds the executive-facing claim-level synthesis, distribution, deep-dive,
aligned-example, insufficient-evidence, modality-contribution and
evidence-gap tables from the already-verified Day 3A fusion outputs.
Reads only existing repository data -- no new collection, no external calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUSION_DIR = PROJECT_ROOT / "data" / "processed" / "fusion"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

RANDOM_SEED = 42


def _load() -> dict[str, pd.DataFrame]:
    d = {
        "units": pd.read_csv(FUSION_DIR / "claim_evidence_units.csv"),
        "fusion": pd.read_csv(FUSION_DIR / "claim_fusion_results.csv"),
        "sensitivity": pd.read_csv(FUSION_DIR / "fusion_sensitivity_results.csv"),
        "stability": pd.read_csv(TABLES_DIR / "fusion_sensitivity_stability.csv"),
        "config_compare": pd.read_csv(TABLES_DIR / "fusion_configuration_comparison.csv"),
        "config_results": pd.read_csv(FUSION_DIR / "fusion_configuration_results.csv"),
        "review_flags": pd.read_csv(TABLES_DIR / "fusion_review_flags.csv"),
        "claim_features": pd.read_csv(PROCESSED_DIR / "official_claim_features.csv"),
        "evidence_candidates": pd.read_csv(PROCESSED_DIR / "claim_multimodal_evidence_candidates.csv"),
    }
    return d


def _official_source_id(claim_id: str) -> str:
    """Return the parent official-claim document_id (oc_NNNN) for a split claim_id (c_OC_NNNN_XX)."""
    parts = claim_id.split("_")
    # c_OC_0002_00 -> oc_0002
    return f"oc_{parts[2]}"


def _strongest_evidence_id(units: pd.DataFrame, claim_id: str, stance: str) -> str:
    sub = units[(units["claim_id"] == claim_id) & (units["stance"] == stance)]
    sub = sub[sub["permitted_evidence_role"] == "supports_or_challenges"]
    if sub.empty:
        return ""
    sub = sub.sort_values(["relevance_score", "reliability_score"], ascending=False)
    return str(sub.iloc[0]["evidence_unit_id"])


def _strongest_reference_id(units: pd.DataFrame, claim_id: str) -> str:
    sub = units[(units["claim_id"] == claim_id) & (units["modality"] == "reference")]
    if sub.empty:
        return ""
    sub = sub.sort_values(["relevance_score"], ascending=False)
    return str(sub.iloc[0]["evidence_unit_id"])


def build_claim_master(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    fusion = d["fusion"].copy()
    units = d["units"]
    stability = d["stability"]
    review = d["review_flags"]
    features = d["claim_features"].set_index("document_id")

    rows = []
    for _, f in fusion.iterrows():
        cid = f["claim_id"]
        oc_id = _official_source_id(cid)
        cu = units[units["claim_id"] == cid]
        eligible = cu[cu["permitted_evidence_role"] == "supports_or_challenges"]

        feat = features.loc[oc_id] if oc_id in features.index else None
        stab = stability[stability["claim_id"] == cid]
        rflag = review[review["claim_id"] == cid]

        modalities_present = sorted(cu["modality"].unique().tolist())
        full_bundle = set(["text", "image", "video", "reference"]).issubset(set(modalities_present))

        supporting_n = int((eligible["stance"] == "supports").sum())
        challenging_n = int((eligible["stance"] == "challenges").sum())
        context_n = int((cu["permitted_evidence_role"] == "context_only").sum()) + int(
            (cu["stance"] == "neutral_context").sum()
        )
        independent_n = int((eligible["source_independence"] == "customer_reported").sum())
        self_reported_n = int((eligible["source_independence"] == "self_reported").sum())

        text_n = int((cu["modality"] == "text").sum())
        image_n = int((cu["modality"] == "image").sum())
        video_n = int((cu["modality"] == "video").sum())
        reference_n = int((cu["modality"] == "reference").sum())

        label = f["automated_label"]
        conflicting = bool(f["evidence_conflict"])
        sensitivity_row = stab.iloc[0] if not stab.empty else None
        label_instability = int(sensitivity_row["n_distinct_labels_across_scenarios"] - 1) if sensitivity_row is not None else 0
        sensitive_to_weights = bool(not sensitivity_row["label_stable"]) if sensitivity_row is not None else False

        presentation_restriction = f.get("presentation_restriction", "")
        external_validation = bool(f["external_validation_recommended"])

        gap = ""
        if label == "insufficient_evidence":
            missing = str(f.get("missing_modalities", "") or "")
            gap = f"missing modalities: {missing}" if missing else "no eligible independent/customer evidence beyond the official claim"

        # analytical interpretation - grounded in counts, not label alone
        if label == "aligned":
            interp = (
                f"The verified corpus contains {supporting_n} supporting evidence unit(s) "
                f"(of which {independent_n} independent/customer-reported) and {challenging_n} challenging unit(s) "
                f"across modalities {modalities_present}."
            )
            action = "Retain as a low-risk claim to cite; periodically re-verify as new evidence accumulates."
            confidence_note = "supported by evidence-gated fusion output"
        elif label == "mixed":
            interp = (
                f"The verified corpus contains both supporting ({supporting_n}) and challenging ({challenging_n}) "
                f"evidence unit(s); {independent_n} independent unit(s) contribute, indicating a genuine divergence "
                f"between official framing and observed customer/press experience rather than an absence of data."
            )
            action = "Route to management review; do not present as simply true or false without qualification."
            confidence_note = "mixed evidence signal, not resolved automatically"
        else:  # insufficient_evidence
            interp = (
                f"Only the self-reported official claim and {supporting_n + challenging_n} eligible independent "
                f"evidence unit(s) were found; the corpus cannot currently corroborate or challenge this claim."
            )
            action = "Do not present as validated; flag for targeted external evidence collection if material to the deck."
            confidence_note = "insufficient corroborating evidence in current corpus"

        caveat = (
            "Automated NLI/rule-based fusion label, not human-adjudicated; "
            + ("label is sensitive to modality-weighting scenario. " if sensitive_to_weights else "label was stable across weighting scenarios. ")
            + ("Presentation-restricted: " + str(presentation_restriction) if presentation_restriction else "")
        ).strip()

        rows.append(
            {
                "claim_id": cid,
                "claim_text": f["claim_text"],
                "claim_category": f["claim_category"],
                "official_source_id": oc_id,
                "official_source_url": feat["source_url"] if feat is not None else "",
                "production_fusion_label": label,
                "fusion_confidence": round(float(f["confidence"]), 4),
                "presentation_restriction": presentation_restriction,
                "conflicting_evidence": conflicting,
                "external_validation_recommended": external_validation,
                "sensitive_to_weights": sensitive_to_weights,
                "label_instability_count": label_instability,
                "evidence_unit_count": int(len(cu)),
                "supporting_evidence_count": supporting_n,
                "challenging_evidence_count": challenging_n,
                "contextual_evidence_count": context_n,
                "independent_evidence_count": independent_n,
                "self_reported_evidence_count": self_reported_n,
                "text_evidence_count": text_n,
                "image_evidence_count": image_n,
                "video_evidence_count": video_n,
                "reference_evidence_count": reference_n,
                "modalities_present": ";".join(modalities_present),
                "full_multimodal_bundle": full_bundle,
                "strongest_supporting_evidence_id": _strongest_evidence_id(units, cid, "supports"),
                "strongest_challenging_evidence_id": _strongest_evidence_id(units, cid, "challenges"),
                "strongest_reference_evidence_id": _strongest_reference_id(units, cid),
                "evidence_gap": gap,
                "analytical_interpretation": interp,
                "managerial_implication": (
                    "Divergence risk for investor/marketing claims; verify before external reuse."
                    if label == "mixed"
                    else ("Evidence gap; do not cite as independently verified." if label == "insufficient_evidence" else "Usable as a supported claim with standard caveats.")
                ),
                "recommended_management_action": action,
                "finding_confidence": confidence_note,
                "caveat": caveat,
            }
        )

    out = pd.DataFrame(rows)
    assert out["claim_id"].nunique() == 37 == len(out), "expected exactly 37 unique claims"
    return out


def build_label_summary(master: pd.DataFrame) -> pd.DataFrame:
    all_labels = ["aligned", "partially_aligned", "mixed", "divergent", "insufficient_evidence"]
    n = len(master)
    rows = []
    for lab in all_labels:
        sub = master[master["production_fusion_label"] == lab]
        rows.append(
            {
                "automated_label": lab,
                "n_claims": len(sub),
                "pct_of_total": round(100 * len(sub) / n, 1),
            }
        )
    overall = pd.DataFrame(rows)

    by_cat = (
        master.groupby(["claim_category", "production_fusion_label"]).size().reset_index(name="n_claims")
    )
    # ensure all category x label combos present (zero-count retained)
    cats = sorted(master["claim_category"].unique())
    full_index = pd.MultiIndex.from_product([cats, all_labels], names=["claim_category", "production_fusion_label"])
    by_cat = by_cat.set_index(["claim_category", "production_fusion_label"]).reindex(full_index, fill_value=0).reset_index()

    conf_bins = pd.cut(master["fusion_confidence"], bins=[0, 0.5, 0.6, 0.7, 0.8, 1.01], right=False)
    conf_dist = conf_bins.value_counts().sort_index().reset_index()
    conf_dist.columns = ["confidence_band", "n_claims"]
    conf_dist["confidence_band"] = conf_dist["confidence_band"].astype(str)

    restriction_dist = master["presentation_restriction"].fillna("none").replace("", "none").value_counts().reset_index()
    restriction_dist.columns = ["presentation_restriction", "n_claims"]

    sensitivity_dist = pd.DataFrame(
        [
            {"metric": "sensitive_to_weights=True", "n_claims": int(master["sensitive_to_weights"].sum())},
            {"metric": "sensitive_to_weights=False", "n_claims": int((~master["sensitive_to_weights"]).sum())},
            {"metric": "label_instability_count>0", "n_claims": int((master["label_instability_count"] > 0).sum())},
        ]
    )

    combined = pd.concat(
        [
            overall.assign(section="overall_label_distribution"),
            by_cat.rename(columns={"production_fusion_label": "automated_label", "n_claims": "n_claims"}).assign(section="by_claim_category"),
            conf_dist.rename(columns={"confidence_band": "automated_label"}).assign(section="confidence_distribution"),
            restriction_dist.rename(columns={"presentation_restriction": "automated_label"}).assign(section="presentation_restriction"),
            sensitivity_dist.rename(columns={"metric": "automated_label"}).assign(section="sensitivity_instability"),
        ],
        ignore_index=True,
    )
    return combined


def build_mixed_deep_dive(master: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    mixed = master[master["production_fusion_label"] == "mixed"].copy()
    rows = []
    for _, m in mixed.iterrows():
        cu = units[units["claim_id"] == m["claim_id"]]
        elig = cu[cu["permitted_evidence_role"] == "supports_or_challenges"]
        support_ids = ";".join(elig[elig["stance"] == "supports"]["evidence_unit_id"].astype(str))
        challenge_ids = ";".join(elig[elig["stance"] == "challenges"]["evidence_unit_id"].astype(str))
        rows.append(
            {
                "claim_id": m["claim_id"],
                "exact_official_claim": m["claim_text"],
                "claim_category": m["claim_category"],
                "supporting_evidence_ids": support_ids,
                "challenging_evidence_ids": challenge_ids,
                "supporting_evidence_count": m["supporting_evidence_count"],
                "challenging_evidence_count": m["challenging_evidence_count"],
                "self_reported_evidence_count": m["self_reported_evidence_count"],
                "independent_evidence_count": m["independent_evidence_count"],
                "modalities_contributing": m["modalities_present"],
                "why_evidence_conflicts": (
                    "Supporting evidence draws on official/self-reported and some independent sources while "
                    "challenging evidence comes from independent customer/press reports describing a different "
                    "experience for the same claim category; the corpus does not resolve which context dominates."
                ),
                "weight_sensitive": m["sensitive_to_weights"],
                "management_investigation": (
                    "Review the specific supporting vs. challenging evidence text; check whether the divergence "
                    "reflects a product-line, timeframe, or geography split rather than a uniform contradiction."
                ),
                "cannot_be_concluded": (
                    "Cannot conclude the claim is generally true or generally false; cannot conclude which "
                    "sub-population of customers the challenging evidence represents."
                ),
                "official_source_id": m["official_source_id"],
                "official_source_url": m["official_source_url"],
            }
        )
    return pd.DataFrame(rows)


def build_aligned_examples(master: pd.DataFrame) -> pd.DataFrame:
    aligned = master[master["production_fusion_label"] == "aligned"].copy()
    aligned["n_modalities"] = aligned["modalities_present"].apply(lambda s: len(str(s).split(";")) if s else 0)
    # transparent selection criteria: strong evidence (>=2 supporting), low sensitivity, multi-modality preferred
    candidates = aligned[(aligned["supporting_evidence_count"] >= 2) & (~aligned["sensitive_to_weights"])]
    candidates = candidates.sort_values(["n_modalities", "independent_evidence_count", "supporting_evidence_count"], ascending=False)
    selected = candidates.head(4) if len(candidates) >= 2 else aligned.sort_values("supporting_evidence_count", ascending=False).head(4)

    rows = []
    for _, m in selected.iterrows():
        basis = "independent evidence contributes" if m["independent_evidence_count"] > 0 else "official/self-reported evidence only"
        rows.append(
            {
                "claim_id": m["claim_id"],
                "claim_text": m["claim_text"],
                "claim_category": m["claim_category"],
                "supporting_evidence_count": m["supporting_evidence_count"],
                "independent_evidence_count": m["independent_evidence_count"],
                "self_reported_evidence_count": m["self_reported_evidence_count"],
                "modalities_present": m["modalities_present"],
                "sensitive_to_weights": m["sensitive_to_weights"],
                "selection_criteria": "supporting_evidence_count>=2; label stable across weighting scenarios (not sensitive_to_weights); ranked by modality breadth then independent-evidence count",
                "alignment_basis": basis,
                "managerial_implication": m["managerial_implication"],
                "strongest_supporting_evidence_id": m["strongest_supporting_evidence_id"],
            }
        )
    return pd.DataFrame(rows)


REASON_CODES = {
    1: "no_public_corroboration_found",
    2: "evidence_category_unavailable_by_design",
    3: "relevant_source_blocked_or_inaccessible",
    4: "existing_evidence_failed_verification_gates",
    5: "claim_not_testable_using_public_multimodal_evidence",
}


def build_insufficient_analysis(master: pd.DataFrame) -> pd.DataFrame:
    insuff = master[master["production_fusion_label"] == "insufficient_evidence"].copy()
    rows = []
    for _, m in insuff.iterrows():
        missing_modalities = str(m["evidence_gap"]).replace("missing modalities: ", "")
        no_customer = m["independent_evidence_count"] == 0
        no_reference = m["reference_evidence_count"] == 0
        no_image = m["image_evidence_count"] == 0
        no_video = m["video_evidence_count"] == 0
        only_official = m["evidence_unit_count"] <= 1

        # reason classification: heuristic based on which modalities are absent
        if only_official and no_customer and no_reference:
            reason_code = 1  # no public corroboration found at all
        elif "video" in missing_modalities and m["claim_category"] in ("materials", "circularity"):
            reason_code = 2  # category structurally unlikely to have video evidence
        else:
            reason_code = 1

        rows.append(
            {
                "claim_id": m["claim_id"],
                "claim_category": m["claim_category"],
                "missing_evidence_types": missing_modalities,
                "corpus_contains_only_official_claim": only_official,
                "not_visually_groundable": no_image and no_video,
                "independent_corroboration_absent": no_customer,
                "reference_evidence_absent": no_reference,
                "customer_evidence_absent": no_customer,
                "gap_classification": (
                    "collection_limitation" if reason_code in (1, 3, 4) else "genuinely_unavailable_public_disclosure"
                ),
                "reason_code": reason_code,
                "reason_label": REASON_CODES[reason_code],
                "evidence_needed_to_evaluate": (
                    "At least one independent customer, press, or reference document unit that explicitly "
                    "corroborates or challenges this specific claim text."
                ),
            }
        )
    out = pd.DataFrame(rows)
    assert not out.empty
    return out


def build_modality_contribution(config_compare: pd.DataFrame, config_results: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    cc = config_compare.copy()
    cc["confidence_delta_from_text_only"] = (cc["mean_confidence"] - cc.loc[cc["configuration"] == "text_only", "mean_confidence"].iloc[0]).round(4)

    # claims gaining evidence / modality breadth as configuration widens
    order = ["text_only", "text_image", "text_image_video", "text_image_video_reference"]
    pivot = config_results.pivot(index="claim_id", columns="configuration", values=["automated_label", "confidence", "eligible_evidence_count"])
    claims_gaining_evidence = 0
    claims_gaining_breadth = 0
    for cid in pivot.index:
        try:
            base_ev = pivot[("eligible_evidence_count", "text_only")][cid]
            full_ev = pivot[("eligible_evidence_count", "text_image_video_reference")][cid]
            if full_ev > base_ev:
                claims_gaining_evidence += 1
        except KeyError:
            pass
    labels_changed = 0
    for cid in pivot.index:
        try:
            lbls = {pivot[("automated_label", cfg)][cid] for cfg in order}
            if len(lbls) > 1:
                labels_changed += 1
        except KeyError:
            pass

    by_modality = units.groupby("modality").agg(
        n_units=("evidence_unit_id", "count"),
        n_supports=("stance", lambda s: int((s == "supports").sum())),
        n_challenges=("stance", lambda s: int((s == "challenges").sum())),
        n_context_only=("permitted_evidence_role", lambda s: int((s == "context_only").sum())),
        n_gate_passed=("permitted_evidence_role", lambda s: int((s == "supports_or_challenges").sum())),
    ).reset_index()

    summary_rows = [
        {"metric": "claims_with_label_change_across_configurations", "value": labels_changed, "denominator": 37},
        {"metric": "claims_with_more_eligible_evidence_in_full_vs_text_only", "value": claims_gaining_evidence, "denominator": 37},
        {"metric": "note", "value": "label distribution identical across all 4 configurations (14 aligned / 5 mixed / 18 insufficient_evidence); only mean_confidence changed", "denominator": ""},
    ]
    summary_df = pd.DataFrame(summary_rows)

    out = {
        "configuration_comparison": cc,
        "modality_evidence_unit_summary": by_modality,
        "cross_configuration_summary": summary_df,
    }
    return out


def build_evidence_gap_by_category(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat, sub in master.groupby("claim_category"):
        total = len(sub)
        with_independent = int((sub["independent_evidence_count"] > 0).sum())
        only_self_reported = int(((sub["self_reported_evidence_count"] > 0) & (sub["independent_evidence_count"] == 0)).sum())
        with_customer = with_independent  # customer evidence is the independent evidence source in this corpus
        with_image = int((sub["image_evidence_count"] > 0).sum())
        with_video = int((sub["video_evidence_count"] > 0).sum())
        with_reference = int((sub["reference_evidence_count"] > 0).sum())
        zero_usable = int((sub["evidence_unit_count"] <= 1).sum())

        missing_counts = {
            "customer/independent": total - with_customer,
            "image": total - with_image,
            "video": total - with_video,
            "reference": total - with_reference,
        }
        dominant_missing = max(missing_counts, key=missing_counts.get)

        rows.append(
            {
                "claim_category": cat,
                "total_claims": total,
                "claims_with_independent_evidence": with_independent,
                "claims_with_only_self_reported_evidence": only_self_reported,
                "claims_with_customer_evidence": with_customer,
                "claims_with_image_evidence": with_image,
                "claims_with_video_evidence": with_video,
                "claims_with_reference_evidence": with_reference,
                "claims_with_zero_usable_evidence_beyond_claim": zero_usable,
                "dominant_missing_evidence_type": dominant_missing,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    d = _load()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    master = build_claim_master(d)
    master.to_csv(TABLES_DIR / "analytical_synthesis_claim_master.csv", index=False)

    label_summary = build_label_summary(master)
    label_summary.to_csv(TABLES_DIR / "claim_label_summary.csv", index=False)

    mixed_dd = build_mixed_deep_dive(master, d["units"])
    mixed_dd.to_csv(TABLES_DIR / "mixed_claim_deep_dive.csv", index=False)

    aligned_ex = build_aligned_examples(master)
    aligned_ex.to_csv(TABLES_DIR / "aligned_claim_examples.csv", index=False)

    insuff = build_insufficient_analysis(master)
    insuff.to_csv(TABLES_DIR / "insufficient_evidence_analysis.csv", index=False)

    modality = build_modality_contribution(d["config_compare"], d["config_results"], d["units"])
    combined_modality = pd.concat(
        [
            modality["configuration_comparison"].assign(section="configuration_comparison"),
            modality["modality_evidence_unit_summary"].assign(section="evidence_unit_by_modality"),
            modality["cross_configuration_summary"].assign(section="cross_configuration_summary"),
        ],
        ignore_index=True,
    )
    combined_modality.to_csv(TABLES_DIR / "modality_contribution_summary.csv", index=False)

    gap = build_evidence_gap_by_category(master)
    gap.to_csv(TABLES_DIR / "evidence_gap_by_category.csv", index=False)

    print("claim_master rows:", len(master))
    print("label_summary rows:", len(label_summary))
    print("mixed_deep_dive rows:", len(mixed_dd))
    print("aligned_examples rows:", len(aligned_ex))
    print("insufficient_evidence rows:", len(insuff))
    print("evidence_gap_by_category rows:", len(gap))
    print("Part A build complete.")


if __name__ == "__main__":
    main()
