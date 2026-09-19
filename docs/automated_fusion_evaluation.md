# Automated Fusion Evaluation (Day 3A, Part 15)

All evaluation in this document is automated. No manual coding, human-coder
adjudication, or Cohen's kappa is used anywhere in this project. Every check
below is deterministic and reproducible from code and configuration alone.

## Method A -- Deterministic synthetic test cases

Synthetic evidence units (`synthetic_*` claim IDs) are constructed in-memory
only, fused through the same `fuse_claim` logic used for real claims, and never
written to `claim_evidence_units.csv` or `claim_fusion_results.csv`. They check
that the decision rules behave as specified (no self-validation, absence is not
contradiction, irrelevant modality evidence cannot change a label, etc.).

## Method B -- Modality ablation

Reuses `fusion_configuration_results.csv` (text-only through full 4-modality
configuration) to check that labels only move when a new modality contributes
customer-reported or independent evidence, not merely by being present.

## Method C -- Sensitivity reuse

Reuses `fusion_sensitivity_results.csv` / `fusion_sensitivity_stability.csv`.
22/37 claims are stable under every perturbation tested.

## Method D -- Cross-method comparison (rule-based vs. NLI)

Compares the deterministic rule-based sentiment stance against an optional
zero-shot NLI cross-encoder (`cross-encoder/nli-deberta-v3-xsmall`) on a sample
of real text evidence units, purely as a second automated signal. This is never
described as human agreement, and no Cohen's kappa or similar human-rater
statistic is computed. If the NLI model is unavailable in this environment, the
result is explicitly recorded as 'not run', never fabricated.

## Method E -- Provenance / groundedness checks

Every real claim and evidence unit is checked for: claim ID membership in the
claims universe, evidence ID presence in its source table, valid evidence_strength
value, a passage or asset backing any supports/challenges stance, visual-gate
enforcement for image/video, disclosure of missing modalities, and that no output
field claims human validation.

## Results summary

| evaluation_type | passed | total |
|---|---|---|
| cross_method_nli_comparison | 0 | 1 |
| modality_ablation | 1 | 1 |
| presentation_governance | 5 | 5 |
| provenance_groundedness | 427 | 427 |
| sensitivity_reuse | 1 | 1 |
| synthetic_test_case | 11 | 11 |

## Limitations

- Method D is a cross-method automated check, not ground truth; disagreement
  between rule-based and NLI stance does not itself mean either is wrong.
- All findings labelled `mixed` or `divergent` remain provisional and are
  flagged `external_validation_recommended=True` regardless of these checks.
- This evaluation cannot substitute for domain-expert review before any
  business decision is made from these outputs.

## NLI disagreement diagnostic (focused follow-up)

Method D's low agreement rate was the only failed automated evaluation check, so
it was investigated against five candidates: label-mapping defect, reversed
premise/hypothesis orientation, comparison-scope defect, low-confidence
disagreement, and genuine domain-model limitation
(`scripts/run_nli_diagnostics.py`, `src/fusion/nli_diagnostics.py`). No
individual business claim was manually inspected -- all categorisation below
is rule-based.

**Implementation audit (Part A).** Model: `cross-encoder/nli-deberta-v3-xsmall`; `config.id2label`={0: 'contradiction', 1: 'entailment', 2: 'neutral'} -- read from the model's own
configuration at runtime, never assumed. No label-mapping defect was found;
the lookup was nonetheless hardened to always read from config instead of a
hardcoded position (`src/fusion/text_evidence.py::nli_label_indices`).

**Comparison-scope defect found and fixed.** Some `evidence_text` values run
to ~6000 characters, far exceeding the model's 512-token window, while the
rule-based TextBlob classifier scores the full untruncated text. Fixed by
bounding the NLI premise to the same 400-character `evidence_passage` used for
citation elsewhere (`nli_stance_if_available`). This did not change the
aggregate agreement rate (the tokenizer already truncated safely at inference
time either way) but removes a real input-scope inconsistency.

**Orientation (Part B).** Selected orientation: `evidence_premise_claim_hypothesis` (overall accuracy 0.8333 on an 18-case synthetic set --
6 entailment / 6 contradiction / 6 neutral across fit, sizing, durability,
returns, materials, certification, none copied from real claim/evidence text).
Selection used only the synthetic results, never the real agreement rate. No
orientation defect was found -- the production orientation already in use
scored highest.

**Real disagreement decomposition (Part C).** 5/30 comparable
real text evidence units agree. Automatic categorisation of the disagreements:

| disagreement_category | count |
|---|---|
| compound_claim | 20 |
| low_nli_confidence | 2 |
| rule_keyword_false_positive | 2 |
| orientation_error | 1 |

The dominant category explains the low agreement as a genuine difference in
what the two methods measure (whole-document sentiment vs. single-claim
entailment on compound, mixed-sentiment real text), not a code defect.

**Fair agreement metrics by scope (Part D, no Cohen's kappa, never called
human agreement):**

| scope | n | raw_agreement |
|---|---|---|
| all_comparable_cases | 30 | 0.1667 |
| nli_confidence_gte_0.6 | 28 | 0.1786 |
| category_gate_passed_and_not_truncated_and_no_abstention_and_high_confidence | 28 | 0.1786 |

**Fusion dependency (Part E).** NLI influenced 0/37
claims' fusion label, confidence, or presentation restriction (0 expected --
`nli_stance_if_available` is called only from this script's Method D check,
never from `run_primary_multimodal_fusion.py` or `evidence_fusion.py`).

**Decision (Part F): C -- genuine domain disagreement.** Label mapping and
orientation both pass synthetic calibration; the comparison-scope defect found
was fixed but did not change the outcome; NLI never had fusion decision
authority. The remaining low real-data agreement is preserved as an honest,
documented limitation -- fusion thresholds were **not** altered to improve it,
and the failed cross-method check remains in `automated_fusion_evaluation.csv`
as an honest finding rather than being hidden or forced to agree.

## Presentation-restriction governance correction (final Day 3A correction)

A final governance pass recomputes `presentation_restriction` for every
claim from signals gathered across fusion, sensitivity, and provenance
evaluation -- it changes only presentation fields, never evidence units,
`automated_label`, confidence, thresholds, NLI outputs, or sensitivity
results. Fixed precedence (`src/fusion/presentation_governance.py`):

1. Provenance or visual/video groundability failure -> `do_not_present_as_conclusion`
2. Mixed/divergent label, sensitivity-unstable, self-reported-only, or
   single-evidence-dependent -> `present_with_strong_caveat`
3. Stable, eligible, multi-source evidence -> `may_present_as_automated_finding`
4. `insufficient_evidence` -> `present_as_evidence_gap_finding_only`
   (never presented as a substantive claim conclusion)

| presentation_restriction | n claims |
|---|---|
| present_as_evidence_gap_finding_only | 18 |
| present_with_strong_caveat | 16 |
| may_present_as_automated_finding | 3 |

18 claim(s) are evidence-gap findings only. 34 claim(s) are recorded in `outputs/tables/fusion_unstable_claims.csv` (everything not cleanly presentable, with its reason).

Five automated regression checks (`evaluation_type=presentation_governance`
above) verify this precedence against the real, corrected claim results:
every sensitivity-unstable, self-reported-only, and single-evidence-dependent
claim never receives `may_present_as_automated_finding`; every
`insufficient_evidence` claim is `present_as_evidence_gap_finding_only`; and
every stable, eligible, multi-source claim retains
`may_present_as_automated_finding`.
