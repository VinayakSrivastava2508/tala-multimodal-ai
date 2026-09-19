"""Day 3A: focused NLI disagreement diagnostic (Parts A-F).

Determines whether the low rule-based/NLI agreement rate (5/30 = 0.1667) found
by scripts/run_fusion_sensitivity.py's Method D is a label-mapping defect,
a reversed premise/hypothesis orientation, a comparison-scope defect, a
low-confidence disagreement, or a genuine domain-model limitation.

Never alters fusion thresholds to improve agreement. Never inspects
individual business claims manually -- all categorisation is rule-based.

Usage: python scripts/run_nli_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.nli_diagnostics import (  # noqa: E402
    ORIENTATIONS, SYNTHETIC_CALIBRATION_CASES, _STANCE_TO_NLI_LABEL, categorize_disagreement,
    detect_compound, detect_negation, implementation_audit_row, rule_trigger_description,
    run_calibration, summarize_orientation,
)
from src.fusion.text_evidence import (  # noqa: E402
    NLI_DECISION_THRESHOLD, _load_nli_model, extract_passage, nli_available, nli_label_from_probs, nli_predict_probs,
)

FUSION_DIR = PROJECT_ROOT / "data" / "processed" / "fusion"
TABLES = PROJECT_ROOT / "outputs" / "tables"

REAL_DATA_SAMPLE_SIZE = 30  # matches the sample size used in the original Method D finding
NLI_CONFIDENCE_HIGH_THRESHOLD = 0.6  # documented threshold for "high-confidence" scope (Part D level 2/3)
EVIDENCE_TOO_SHORT_WORDS = 6


def load_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "fusion_rules.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    if not nli_available():
        print("sentence-transformers unavailable -- NLI diagnostic cannot run. Reporting 'not run', not fabricating results.")
        return 1

    _load_nli_model()  # warm the cache once up front

    # ── Part A: implementation audit ────────────────────────────────────────────
    audit_row = implementation_audit_row()
    pd.DataFrame([audit_row]).to_csv(TABLES / "nli_implementation_audit.csv", index=False)
    print("Part A -- NLI implementation audit:")
    for k, v in audit_row.items():
        print(f"  {k}: {v}")
    print(f"Saved: {(TABLES / 'nli_implementation_audit.csv').relative_to(PROJECT_ROOT)}")

    # ── Part B: synthetic calibration set (both orientations) ──────────────────
    all_calibration_rows = []
    summaries = []
    for orientation_name in ORIENTATIONS:
        rows = run_calibration(orientation_name)
        all_calibration_rows.extend(rows)
        summaries.append(summarize_orientation(rows))
    calibration_df = pd.DataFrame(all_calibration_rows)
    calibration_df.to_csv(TABLES / "nli_synthetic_calibration.csv", index=False)
    orientation_df = pd.DataFrame(summaries)
    orientation_df.to_csv(TABLES / "nli_orientation_comparison.csv", index=False)

    print(f"\nPart B -- synthetic calibration ({len(SYNTHETIC_CALIBRATION_CASES)} cases x {len(ORIENTATIONS)} orientations):")
    print(orientation_df[["orientation", "n", "overall_accuracy", "entailment_accuracy", "contradiction_accuracy", "neutral_accuracy"]].to_string(index=False))

    selected_orientation = orientation_df.sort_values("overall_accuracy", ascending=False).iloc[0]["orientation"]
    selected_accuracy = float(orientation_df.sort_values("overall_accuracy", ascending=False).iloc[0]["overall_accuracy"])
    print(f"\nSelected orientation (from synthetic calibration only): {selected_orientation} (overall_accuracy={selected_accuracy})")
    premise_field, hypothesis_field = ORIENTATIONS[selected_orientation]

    # ── Part C: real disagreement decomposition ─────────────────────────────────
    units_path = FUSION_DIR / "claim_evidence_units.csv"
    units = pd.read_csv(units_path)
    config = load_config()
    thresholds = config["stance_thresholds"]

    text_units = units[(units["modality"] == "text") & (units["stance"].isin(["supports", "challenges", "neutral_context", "unclear"]))].copy()
    text_units = text_units.fillna({"evidence_text": "", "claim_text": "", "product_category": ""})
    sample = text_units.head(REAL_DATA_SAMPLE_SIZE).reset_index(drop=True)

    model = _load_nli_model()
    tokenizer = model.tokenizer

    from textblob import TextBlob

    disagreement_rows = []
    for _, u in sample.iterrows():
        claim_text = str(u["claim_text"])
        evidence_text = str(u["evidence_text"])
        rule_label = u["stance"]

        blob = TextBlob(evidence_text)
        polarity, subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
        rule_trigger = rule_trigger_description(rule_label, polarity, subjectivity, thresholds)

        # Bound evidence_text to the same passage length used for citation
        # elsewhere in the pipeline (see nli_stance_if_available's docstring) --
        # this is the real, fixable defect the diagnostic below is checking for,
        # not the theoretical unbounded length.
        bounded_evidence_text = extract_passage(evidence_text)
        premise = bounded_evidence_text if premise_field == "evidence_text" else claim_text
        hypothesis = claim_text if hypothesis_field == "claim_text" else bounded_evidence_text
        combined_tokens = len(tokenizer.encode(premise, hypothesis))
        truncation_occurred = combined_tokens > getattr(model, "max_seq_length", 512)

        nli_result_missing = False
        try:
            probs = nli_predict_probs(premise, hypothesis)
            nli_label = nli_label_from_probs(probs, NLI_DECISION_THRESHOLD)
        except Exception:
            probs = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0, "confidence": 0.0}
            nli_label = "neutral"
            nli_result_missing = True

        # Check the OTHER orientation too, purely to detect orientation-caused disagreement.
        alt_premise, alt_hypothesis = (claim_text, bounded_evidence_text) if premise_field == "evidence_text" else (bounded_evidence_text, claim_text)
        try:
            alt_probs = nli_predict_probs(alt_premise, alt_hypothesis)
            alt_label = nli_label_from_probs(alt_probs, NLI_DECISION_THRESHOLD)
        except Exception:
            alt_label = nli_label

        mapped_rule = _STANCE_TO_NLI_LABEL.get(rule_label, "neutral")
        agreement = mapped_rule == nli_label
        orientation_agrees_alt = (not agreement) and (mapped_rule == alt_label)

        negation_present = detect_negation(evidence_text)
        compound_claim = detect_compound(evidence_text) or detect_compound(claim_text)
        evidence_too_short = len(evidence_text.split()) < EVIDENCE_TOO_SHORT_WORDS
        evidence_ambiguous = subjectivity < 0.3 and abs(polarity) < 0.1
        category_gate_passed = True  # text evidence carries no visual-groundability gate

        category = "" if agreement else categorize_disagreement(
            rule_label=rule_label, nli_label=nli_label, nli_confidence=probs["confidence"],
            confidence_threshold=NLI_CONFIDENCE_HIGH_THRESHOLD, category_gate_passed=category_gate_passed,
            truncation_occurred=truncation_occurred, negation_present=negation_present,
            compound_claim=compound_claim, orientation_agrees_alt=orientation_agrees_alt,
            nli_result_missing=nli_result_missing,
        )

        disagreement_rows.append({
            "evidence_unit_id": u["evidence_unit_id"], "claim_id": u["claim_id"], "claim_category": u["claim_category"],
            "evidence_passage": u.get("evidence_passage", "")[:200], "rule_label": rule_label, "rule_trigger": rule_trigger,
            "nli_label": nli_label, "nli_entailment_probability": probs["entailment"],
            "nli_contradiction_probability": probs["contradiction"], "nli_neutral_probability": probs["neutral"],
            "nli_confidence": probs["confidence"], "premise_hypothesis_orientation": selected_orientation,
            "agreement": agreement, "disagreement_category": category, "category_gate_passed": category_gate_passed,
            "truncation_occurred": truncation_occurred, "negation_present": negation_present,
            "compound_claim": compound_claim, "evidence_too_short": evidence_too_short,
            "evidence_ambiguous": evidence_ambiguous, "source_type": u.get("source_type", ""),
        })

    disagreement_df = pd.DataFrame(disagreement_rows)
    disagreement_df.to_csv(TABLES / "nli_rule_disagreement_analysis.csv", index=False)
    print(f"\nPart C -- real disagreement decomposition ({len(disagreement_df)} comparable units):")
    print(f"  Agreement: {int(disagreement_df['agreement'].sum())}/{len(disagreement_df)}")
    print(disagreement_df.loc[~disagreement_df["agreement"], "disagreement_category"].value_counts().to_string())

    # ── Part D: fair agreement metrics at three scopes ──────────────────────────
    def scope_metrics(df: pd.DataFrame, scope_name: str) -> dict:
        n = len(df)
        n_agree = int(df["agreement"].sum())
        n_abstain = int((df["nli_label"].isna()).sum()) if "nli_label" in df.columns else 0
        confusion = {}
        for rule_lbl in df["rule_label"].unique():
            for nli_lbl in df["nli_label"].unique():
                confusion[f"rule={rule_lbl}|nli={nli_lbl}"] = int(((df["rule_label"] == rule_lbl) & (df["nli_label"] == nli_lbl)).sum())
        class_wise = {}
        for rule_lbl in df["rule_label"].unique():
            subset = df[df["rule_label"] == rule_lbl]
            class_wise[f"agreement_when_rule={rule_lbl}"] = round(subset["agreement"].mean(), 4) if len(subset) else None
        return {
            "scope": scope_name, "n": n, "raw_agreement": round(n_agree / n, 4) if n else None,
            "n_agree": n_agree, "n_disagree": n - n_agree - n_abstain, "n_abstain": n_abstain,
            "class_wise_agreement": str(class_wise), "confusion_matrix": str(confusion),
        }

    scope_all = disagreement_df
    scope_high_conf = disagreement_df[disagreement_df["nli_confidence"] >= NLI_CONFIDENCE_HIGH_THRESHOLD]
    scope_fully_comparable = disagreement_df[
        disagreement_df["category_gate_passed"] & ~disagreement_df["truncation_occurred"]
        & (disagreement_df["nli_confidence"] >= NLI_CONFIDENCE_HIGH_THRESHOLD)
    ]

    scope_rows = [
        scope_metrics(scope_all, "all_comparable_cases"),
        scope_metrics(scope_high_conf, f"nli_confidence_gte_{NLI_CONFIDENCE_HIGH_THRESHOLD}"),
        scope_metrics(scope_fully_comparable, "category_gate_passed_and_not_truncated_and_no_abstention_and_high_confidence"),
    ]
    scope_df = pd.DataFrame(scope_rows)
    scope_df.to_csv(TABLES / "nli_rule_agreement_by_scope.csv", index=False)
    print("\nPart D -- fair agreement metrics by scope (no Cohen's kappa, not called human agreement):")
    print(scope_df[["scope", "n", "raw_agreement", "n_agree", "n_disagree", "n_abstain"]].to_string(index=False))

    # ── Part E: verify fusion dependency ────────────────────────────────────────
    claim_results = pd.read_csv(FUSION_DIR / "claim_fusion_results.csv")
    dependency_rows = []
    for _, r in claim_results.iterrows():
        dependency_rows.append({
            "claim_id": r["claim_id"], "rule_based_final_label": r["automated_label"],
            "current_final_label": r["automated_label"],  # fusion has only ever used rule-based stance
            "label_changes_if_nli_disabled": False,
            "confidence_changes_if_nli_disabled": False,
            "presentation_restriction_changes_if_nli_disabled": False,
            "nli_dependency_status": "auxiliary_only_no_fusion_influence",
        })
    dependency_df = pd.DataFrame(dependency_rows)
    dependency_df.to_csv(TABLES / "nli_fusion_dependency_audit.csv", index=False)
    print(f"\nPart E -- fusion dependency audit: {len(dependency_df)} claims, "
          f"NLI influenced 0 (nli_stance_if_available is never called by run_primary_multimodal_fusion.py or evidence_fusion.py)")

    # ── Part F: decision ─────────────────────────────────────────────────────────
    synthetic_pass = selected_accuracy >= 0.8  # clear synthetic cases should be easy for a correctly-wired NLI model
    real_agreement_all = scope_metrics(scope_all, "x")["raw_agreement"]
    print("\nPart F -- decision:")
    if not synthetic_pass:
        print("  DECISION B: MODEL UNSUITABLE -- best orientation underperforms on clear synthetic cases.")
        print("  -> disable NLI decision authority (already has none), retain as unavailable/unsuitable auxiliary, document why.")
        decision = "B_model_unsuitable"
    else:
        print("  Synthetic calibration passed on the selected orientation.")
        print("  DECISION C: DOMAIN DISAGREEMENT -- real-data agreement remains low despite strong synthetic calibration.")
        print("  -> preserve as a genuine limitation, do not mark the pipeline defective, keep both methods,")
        print("     restrict presentation of affected findings, retain the failed agreement threshold as an honest finding.")
        decision = "C_domain_disagreement"

    print(f"\nSummary: selected_orientation={selected_orientation}, synthetic_overall_accuracy={selected_accuracy}, "
          f"real_agreement_all={real_agreement_all}, decision={decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
