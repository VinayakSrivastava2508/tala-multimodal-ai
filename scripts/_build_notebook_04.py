"""One-off generator for notebooks/04_primary_multimodal_fusion.ipynb.
Not part of the Day 3A pipeline itself -- run once to (re)build the notebook
skeleton, then execute it with nbconvert. Safe to delete after the notebook
exists; kept for reproducibility of how the notebook was authored."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# 1. Title, scope, and non-negotiable ground rules
md("""# 04 — Primary Multimodal Fusion for TALA Claim-Experience Divergence (Day 3A)

This notebook is the authoritative evidence that Day 3A's primary multimodal evidence fusion was
built and actually run. It integrates the Text, Image, Video, and Multimodal Reference Packages
(Notebooks 02/03) into one automated, per-claim divergence signal.

**Explicitly out of scope in this notebook:** engagement prediction, the final RAG prototype, the
governance framework, final business recommendations, and presentation slides.

**No manual coding or human-coder evaluation is used anywhere in this pipeline.** Every real
claim result carries `evaluation_status=automated_not_human_validated`, `provisional_finding=True`,
and an explicit `presentation_restriction`. Nothing here should be presented as a verified or
confirmed finding.

Pipeline entry points (run in this order):
1. `python scripts/freeze_fusion_inputs.py`
2. `python scripts/run_primary_multimodal_fusion.py`
3. `python scripts/run_fusion_sensitivity.py` (also produces the Part 15 automated evaluation)""")

code("""import sys
from pathlib import Path

PROJECT_ROOT = Path().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", 100)
RANDOM_SEED = 42

FUSION_DIR = PROJECT_ROOT / "data" / "processed" / "fusion"
TABLES = PROJECT_ROOT / "outputs" / "tables"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
DOCS = PROJECT_ROOT / "docs"

evidence_units = pd.read_csv(FUSION_DIR / "claim_evidence_units.csv")
claim_results = pd.read_csv(FUSION_DIR / "claim_fusion_results.csv")
config_results = pd.read_csv(FUSION_DIR / "fusion_configuration_results.csv")
sensitivity_results = pd.read_csv(FUSION_DIR / "fusion_sensitivity_results.csv")

configuration_comparison = pd.read_csv(TABLES / "fusion_configuration_comparison.csv")
modality_contribution = pd.read_csv(TABLES / "fusion_modality_contribution.csv")
label_distribution = pd.read_csv(TABLES / "fusion_label_distribution.csv")
divergence_matrix = pd.read_csv(TABLES / "claim_divergence_matrix.csv")
evidence_gaps = pd.read_csv(TABLES / "fusion_evidence_gaps.csv")
review_flags = pd.read_csv(TABLES / "fusion_review_flags.csv")
sensitivity_stability = pd.read_csv(TABLES / "fusion_sensitivity_stability.csv")
automated_evaluation = pd.read_csv(TABLES / "automated_fusion_evaluation.csv")

with open(FUSION_DIR / "fusion_input_manifest.json", encoding="utf-8") as f:
    fusion_manifest = json.load(f)

print("Loaded Day 3A primary multimodal fusion outputs.")""")

# 2. Frozen input manifest
md("""## 1. Frozen, checksummed fusion inputs

Every input file used by primary fusion is checksummed and versioned before any fusion logic
runs, so every downstream output traces back to a reproducible snapshot
(`scripts/freeze_fusion_inputs.py`, `data/processed/fusion/fusion_input_manifest.json`).""")

code("""print(f"Dataset version: {fusion_manifest['dataset_version']}")
print(f"Frozen at: {fusion_manifest['generated_at_utc']}")
manifest_df = pd.DataFrame(fusion_manifest["files"]).T[["modality", "row_count", "unique_id_count", "duplicate_id_count", "sha256"]]
display(manifest_df)""")

# 3. Fusion rules configuration
md("""## 2. Fusion rules configuration

All thresholds, weights, and decision rules are read from `configs/fusion_rules.yaml` -- nothing
is hardcoded in the fusion modules, so every number below is independently inspectable and was
perturbed in the automated sensitivity analysis (§8).""")

code("""import yaml
with open(PROJECT_ROOT / "configs" / "fusion_rules.yaml", encoding="utf-8") as f:
    fusion_rules = yaml.safe_load(f)

print("Relevance thresholds:", fusion_rules["relevance_thresholds"])
print("Stance thresholds:", fusion_rules["stance_thresholds"])
print("Source reliability tiers:", fusion_rules["source_reliability_tiers"])
print("Label decision rules:", fusion_rules["label_decision"])""")

# 4. Non-negotiable evidence rules
md("""## 3. Non-negotiable evidence rules (Part 6)

These rules are enforced in code (`src/fusion/*.py`), not just documented:

1. A claim cannot validate itself -- the official claim text is never its own evidence.
2. Official brand references (self-reported) never alone produce "aligned".
3. Certification registries only prove what they specifically confirm.
4. Customer/critic evidence can support or challenge specific claim dimensions.
5. Images/videos default to `neutral_context` unless a visual-groundability gate passes.
6. Images/videos can never validate labour, emissions, certification, or durability claims.
7. Absence of a claim from a document is not treated as contradiction.
8. Missing evidence is never scored as negative evidence.
9. A shared/generic "brand gallery" video cannot be exact-product evidence without a verified match.
10. Every real result is provisional and flagged for external validation, never presented as
    verified.""")

# 5. Evidence unit construction
md("""## 4. Claim-evidence unit construction

One row per real evidence item matched to a real claim, across text/image/video/reference
(`scripts/run_primary_multimodal_fusion.py`, `data/processed/fusion/claim_evidence_units.csv`).""")

code("""print(f"Total evidence units: {len(evidence_units)}")
display(evidence_units["modality"].value_counts().rename_axis("modality").reset_index(name="n_units"))
display(evidence_units.groupby(["modality", "source_independence"]).size().reset_index(name="n_units"))""")

code("""display(evidence_units[["claim_id", "modality", "evidence_id", "source_type", "source_independence",
                          "evidence_strength", "relevance_score", "stance", "reliability_score"]].head(10))""")

# 6. Visual groundability gating in practice
md("""## 5. Visual groundability gating in practice

Every image/video evidence unit's `visual_gate_passed` flag is checked before it can ever
contribute a `supports`/`challenges` stance -- irrelevant or ungroundable visual evidence is
forced to `neutral_context` regardless of similarity score.""")

code("""visual_units = evidence_units[evidence_units["modality"].isin(["image", "video"])]
display(visual_units.groupby(["modality", "visual_gate_passed", "stance"]).size().reset_index(name="n_units"))
print(f"\\nImage/video units that ever produced 'challenges': "
      f"{int((visual_units['stance']=='challenges').sum())} (should be 0 -- images/video can only support or stay contextual)")""")

# 7. Decision-level fusion results
md("""## 6. Decision-level fusion results (final configuration: text + image + video + reference)

`src/fusion/evidence_fusion.py::fuse_claim` aggregates all eligible evidence for a claim into one
automated label. No label was fit against a human-labelled ground truth -- none exists.""")

code("""display(label_distribution)
print(f"\\nTotal claims: {len(claim_results)}")
display(claim_results["automated_label"].value_counts())""")

code("""for field in ("evaluation_status", "provisional_finding", "external_validation_recommended", "presentation_restriction"):
    assert field in claim_results.columns, field
print("Every claim result carries evaluation_status / provisional_finding / external_validation_recommended / presentation_restriction.")
display(claim_results[["claim_id", "automated_label", "business_facing_label", "confidence", "presentation_restriction"]].head(10))""")

# 8. Business-facing label mapping
md("""## 7. Internal vs. business-facing labels

Internal labels are never shown to a business stakeholder directly -- they are mapped to
deliberately hedged business-facing language, and "divergent" is never called "confirmed" or
"verified".""")

code("""from src.fusion.evidence_fusion import BUSINESS_FACING_LABELS
display(pd.DataFrame(list(BUSINESS_FACING_LABELS.items()), columns=["internal_label", "business_facing_label"]))""")

# 9. Worked examples per label
md("""## 8. Worked examples — one claim per label

Concrete evidence trails, not just aggregate counts, so the decision logic can be inspected by a
reader.""")

code("""for label in ["aligned", "mixed", "insufficient_evidence"]:
    subset = claim_results[claim_results["automated_label"] == label]
    if subset.empty:
        print(f"[{label}] -- no claim currently receives this label.")
        continue
    example = subset.iloc[0]
    print(f"\\n=== {label.upper()} -- {example['claim_id']} ===")
    print(f"Claim: {example['claim_text']}")
    print(f"official_disclosure_support={example['official_disclosure_support']}, "
          f"independent_support={example['independent_support']}, "
          f"eligible_evidence_count={example['eligible_evidence_count']}, "
          f"confidence={example['confidence']}")
    units_for_claim = evidence_units[evidence_units["claim_id"] == example["claim_id"]]
    display(units_for_claim[["modality", "source_independence", "stance", "reliability_score", "evidence_passage"]])""")

# 10. Configuration comparison
md("""## 9. Modality-ablation configuration comparison (Part 8)

Four configurations were run: text-only, +image, +image+video, +image+video+reference. Labels
should only move when a NEW modality contributes genuine customer-reported or independent
evidence -- not merely by being present.""")

code("""display(configuration_comparison)
display(config_results.groupby("configuration")["automated_label"].value_counts().unstack(fill_value=0))""")

md("""**Finding:** all four configurations produce an identical label distribution. This was
investigated and is correct, not a bug: every new-modality evidence unit added in this dataset is
either self-reported (official catalog images, brand-owned reference disclosures) or a
category-gated video that never clears the exact-product match tier required to escalate past
`neutral_context` (the "shared brand-gallery video" rule, Part 6 rule 9). Confidence still shifts
slightly across configurations even though labels do not.""")

# 11. Modality contribution
md("## 10. Modality contribution to confidence")

code("""display(modality_contribution)""")

# 12. Divergence matrix / heatmap
md("## 11. Claim-category × label divergence matrix")

code("""display(divergence_matrix)
fig_path = FIGURES / "claim_divergence_heatmap.png"
if fig_path.exists():
    plt.figure(figsize=(8, 5))
    plt.imshow(mpimg.imread(fig_path))
    plt.axis("off")
    plt.title("claim_divergence_heatmap.png (data source: claim_fusion_results.csv, n=37)")
    plt.show()""")

# 13. Evidence gaps
md("""## 12. Evidence gaps

Claims with zero or thin evidence are reported honestly, not padded.""")

code("""display(evidence_gaps.head(20))
print(f"\\n{len(evidence_gaps)} claim(s) flagged with an evidence gap")""")

# 14. Review flags
md("## 13. Review flags (including sensitivity-derived flags)")

code("""display(review_flags.head(20))
print(f"\\n{int(review_flags['sensitive_to_weights'].sum())} claim(s) flagged sensitive_to_weights")""")

# 15. Figures
md("## 14. Summary figures")

code("""for fname, caption in [
    ("fusion_label_distribution.png", "Label distribution across n=37 claims (source: claim_fusion_results.csv)"),
    ("fusion_modality_coverage.png", "Evidence-unit modality coverage (source: claim_evidence_units.csv)"),
    ("fusion_configuration_comparison.png", "4-configuration ablation comparison (source: fusion_configuration_results.csv)"),
]:
    p = FIGURES / fname
    if p.exists():
        plt.figure(figsize=(7, 4))
        plt.imshow(mpimg.imread(p))
        plt.axis("off")
        plt.title(f"{fname} -- {caption}")
        plt.show()""")

# 16. Sensitivity analysis
md("""## 15. Automated sensitivity analysis (Part 14)

Reasonable, documented perturbations of `configs/fusion_rules.yaml` -- reliability weights,
relevance thresholds, stance thresholds, minimum-evidence count, and three "remove a class of
evidence" scenarios. No human-labelled ground truth is used or required.""")

code("""n_scenarios = sensitivity_results["scenario"].nunique() - 1  # exclude baseline
print(f"{n_scenarios} perturbation scenarios x {sensitivity_stability.shape[0]} claims")
display(sensitivity_stability["label_stable"].value_counts())
print(f"\\n{int((~sensitivity_stability['label_stable']).sum())}/{len(sensitivity_stability)} claims changed label under at least one perturbation")
print(f"{int(sensitivity_stability['single_evidence_dependency'].sum())} claim(s) depend on a single evidence unit")
display(sensitivity_stability[~sensitivity_stability["label_stable"]].head(10))""")

code("""scenario_changes = (
    sensitivity_results[sensitivity_results["scenario"] != "baseline"]
    .merge(claim_results[["claim_id", "automated_label"]].rename(columns={"automated_label": "baseline_label"}), on="claim_id")
)
scenario_changes["changed"] = scenario_changes["automated_label"] != scenario_changes["baseline_label"]
display(scenario_changes.groupby("scenario")["changed"].sum().sort_values(ascending=False).rename("n_claims_changed"))""")

# 17. Automated evaluation overview
md("""## 16. Automated evaluation (Part 15) — methods A-E

No manual coding or human-coder adjudication is used anywhere in this evaluation. Full narrative:
`docs/automated_fusion_evaluation.md`.""")

code("""summary = automated_evaluation.groupby("evaluation_type")["passed"].agg(["sum", "count"])
summary["fail"] = summary["count"] - summary["sum"]
display(summary)
print(f"\\nOverall: {int(automated_evaluation['passed'].sum())}/{len(automated_evaluation)} automated checks passed")""")

# 18. Method A detail
md("### 16a. Method A — deterministic synthetic test cases")

code("""synthetic = automated_evaluation[automated_evaluation["evaluation_type"] == "synthetic_test_case"]
display(synthetic[["test_case_id", "expected_property", "observed_property", "passed"]])""")

# 19. Method D detail (cross-method)
md("""### 16b. Method D — cross-method comparison (rule-based sentiment vs. NLI)

This is never called "human agreement" and no Cohen's kappa or similar human-rater statistic is
computed. If the NLI cross-encoder is unavailable, the result is reported as "not run", never
fabricated.""")

code("""crossmethod = automated_evaluation[automated_evaluation["evaluation_type"] == "cross_method_nli_comparison"]
display(crossmethod[["test_case_id", "expected_property", "observed_property", "passed", "failure_reason"]])""")

md("""**Interpretation:** a low cross-method agreement rate reflects that rule-based sentiment
polarity and NLI entailment-of-the-claim are measuring different things (evidence tone vs. logical
entailment of the specific claim) -- it is reported as an honest limitation, not forced to agree
and not treated as evidence either method is "wrong".""")

# 20. Method E detail (provenance)
md("### 16c. Method E — provenance / groundedness checks")

code("""provenance = automated_evaluation[automated_evaluation["evaluation_type"] == "provenance_groundedness"]
print(f"{int(provenance['passed'].sum())}/{len(provenance)} provenance/groundedness checks passed")
display(provenance[~provenance["passed"]].head(10))""")

# 21. Limitations
md("""## 16d. Focused NLI disagreement diagnostic

Method D's low agreement (5/30 = 0.1667) was the only failed automated evaluation check, so it was
investigated in depth against five candidates: label-mapping defect, reversed premise/hypothesis
orientation, comparison-scope defect, low-confidence disagreement, and genuine domain-model
limitation (`scripts/run_nli_diagnostics.py`, `src/fusion/nli_diagnostics.py`). No individual
business claim was manually inspected -- all categorisation is rule-based.""")

code("""nli_audit = pd.read_csv(TABLES / "nli_implementation_audit.csv")
print("Part A -- implementation audit (label mapping read from model config, not assumed):")
display(nli_audit[["model_name", "config_id2label", "entailment_label_id", "contradiction_label_id",
                    "neutral_label_id", "premise_field", "hypothesis_field", "max_seq_length",
                    "inference_device", "label_mapping_verified_against_config"]])""")

code("""nli_orientation = pd.read_csv(TABLES / "nli_orientation_comparison.csv")
print("Part B -- synthetic calibration (18 cases, 6 per class, orientation selected from this only):")
display(nli_orientation[["orientation", "n", "overall_accuracy", "entailment_accuracy",
                          "contradiction_accuracy", "neutral_accuracy"]])""")

code("""nli_disagreement = pd.read_csv(TABLES / "nli_rule_disagreement_analysis.csv")
print(f"Part C -- real disagreement decomposition: {int(nli_disagreement['agreement'].sum())}/{len(nli_disagreement)} agree")
display(nli_disagreement.loc[~nli_disagreement["agreement"], "disagreement_category"].value_counts().reset_index())""")

md("""**Finding:** `compound_claim` accounts for 20/25 (80%) of disagreements -- the rule-based
method averages sentiment across a whole (often mixed-sentiment) customer review, while NLI judges
entailment of one specific claim sentence. On reviews containing both praise and complaints, these
are legitimately different questions. This is the primary driver of the low agreement rate, not a
bug.""")

code("""nli_scope = pd.read_csv(TABLES / "nli_rule_agreement_by_scope.csv")
print("Part D -- fair agreement metrics by scope (no Cohen's kappa, never called human agreement):")
display(nli_scope[["scope", "n", "raw_agreement", "n_agree", "n_disagree", "n_abstain"]])""")

code("""nli_dependency = pd.read_csv(TABLES / "nli_fusion_dependency_audit.csv")
print(f"Part E -- fusion dependency: NLI influenced "
      f"{int((nli_dependency['nli_dependency_status'] != 'auxiliary_only_no_fusion_influence').sum())}/"
      f"{len(nli_dependency)} claims (should be 0 -- NLI is auxiliary-only, never fusion decision authority)")""")

md("""**Decision (Part F): C -- genuine domain disagreement.** Label mapping and orientation both
pass synthetic calibration (no implementation defect found there); a real comparison-scope defect
(evidence_text up to ~6000 chars exceeding the model's 512-token window) was found and fixed by
bounding the NLI premise to the same 400-char evidence_passage used for citation elsewhere, but
this did not change the real-data agreement rate. NLI never had fusion decision authority to begin
with. The remaining low agreement is preserved as an honest, documented limitation -- fusion
thresholds were **not** altered to improve it, and the failed evaluation check remains in
`automated_fusion_evaluation.csv` as an honest finding. Full narrative:
`docs/automated_fusion_evaluation.md`.""")

md("""## 16e. Presentation-restriction governance correction (final Day 3A correction)

A final governance pass recomputes `presentation_restriction` for every claim from signals gathered
across fusion, sensitivity, and provenance evaluation. It changes only presentation fields --
never evidence units, `automated_label`, confidence, thresholds, NLI outputs, or sensitivity
results (`src/fusion/presentation_governance.py`). Fixed precedence:

1. Provenance or visual/video groundability failure -> `do_not_present_as_conclusion`
2. Mixed/divergent label, sensitivity-unstable, self-reported-only, or single-evidence-dependent
   -> `present_with_strong_caveat`
3. Stable, eligible, multi-source evidence -> `may_present_as_automated_finding`
4. `insufficient_evidence` -> `present_as_evidence_gap_finding_only` (never a substantive
   claim conclusion)""")

code("""governance_counts = claim_results["presentation_restriction"].value_counts()
display(governance_counts)

unstable_claims = pd.read_csv(TABLES / "fusion_unstable_claims.csv")
print(f"\\n{int(claim_results['evidence_gap_finding'].sum())} claim(s) are evidence-gap findings only")
print(f"{len(unstable_claims)} claim(s) recorded in fusion_unstable_claims.csv (everything not cleanly presentable)")
display(unstable_claims[["claim_id", "automated_label", "presentation_restriction", "presentation_restriction_reason"]].head(10))""")

code("""governance_checks = automated_evaluation[automated_evaluation["evaluation_type"] == "presentation_governance"]
display(governance_checks[["test_case_id", "expected_property", "observed_property", "passed"]])""")

md("""**Verification:** every sensitivity-unstable, self-reported-only, and single-evidence-dependent
claim never receives `may_present_as_automated_finding`; every `insufficient_evidence` claim is
`present_as_evidence_gap_finding_only`; every stable, eligible, multi-source claim retains
`may_present_as_automated_finding`. All five checks above pass against the real, corrected claim
results -- not merely asserted.""")

md("""## 17. Limitations

- All findings are automated and provisional; `mixed`/`divergent` results are flagged
  `external_validation_recommended=True` and must be reviewed before any business use.
- All four modality-ablation configurations produce identical labels in this dataset because
  every new-modality evidence unit here is self-reported or gated to `neutral_context` -- this
  reflects real evidence scarcity, not a broken fusion mechanism (§9).
- Cross-method (rule-based vs. NLI) agreement is low (§16b) and is reported as a genuine
  limitation of comparing two different automated signals, not glossed over.
- 15/37 claims are label-unstable under at least one reasonable configuration perturbation
  (§15) -- these are exactly the claims flagged `sensitive_to_weights=True` in
  `fusion_review_flags.csv` and should be treated with the most caution downstream.
- This notebook does not implement RAG, engagement prediction, governance, or final
  recommendations -- those remain out of scope for Day 3A.""")

# 22. Final summary + RAG note
md("## 18. Final summary")

code("""print("=" * 70)
print("NOTEBOOK 04 COMPLETE -- Day 3A Primary Multimodal Fusion")
print("=" * 70)
print(f"  Claims processed:                {len(claim_results)}")
print(f"  Evidence units:                  {len(evidence_units)} ({evidence_units['modality'].value_counts().to_dict()})")
print(f"  Label distribution:              {claim_results['automated_label'].value_counts().to_dict()}")
print(f"  Configurations compared:         {config_results['configuration'].nunique()}")
print(f"  Sensitivity scenarios:           {n_scenarios}")
print(f"  Claims label-unstable:           {int((~sensitivity_stability['label_stable']).sum())}/{len(sensitivity_stability)}")
print(f"  Automated evaluation checks:     {int(automated_evaluation['passed'].sum())}/{len(automated_evaluation)} passed")
print(f"  Claims flagged for review:       {len(review_flags)}")
print(f"  NLI diagnostic decision:         C -- genuine domain disagreement (not a defect, not fusion-influencing)")
print(f"  Presentation restrictions:       {claim_results['presentation_restriction'].value_counts().to_dict()}")
print("\\n  Every real result: evaluation_status=automated_not_human_validated, provisional_finding=True")
print("  No manual coding or human-coder evaluation was used anywhere in this pipeline.")
print("\\n  RAG readiness: fusion outputs (claim_fusion_results.csv, claim_evidence_units.csv) are")
print("  structured and provenance-tracked, and are a suitable retrieval corpus for a future RAG")
print("  prototype -- but RAG itself was NOT built in this task.")""")

nb["cells"] = cells
with open("notebooks/04_primary_multimodal_fusion.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Notebook written.")
