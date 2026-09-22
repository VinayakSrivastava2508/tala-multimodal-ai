"""One-off generator for notebooks/06_analytical_synthesis_and_creator_strategy.ipynb
(Day 3C). Run once to (re)build the notebook skeleton, then execute with nbconvert."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# 1. Objective and decision questions
md("""# 06 — Executive Analytical Synthesis and Creator-Strategy Comparison (Day 3C)

## 1. Objective and decision questions

**Primary objective:** what does the verified multimodal evidence reveal about divergence
between TALA's official quality/responsibility claims and externally observable customer
experience?

**Secondary objective (professor feedback):** since TALA is prioritising a creator-led
strategy, how does its creator/partnership strategy differ from Adanola, Girlfriend
Collective and Oner Active across platforms -- what can TALA learn, and what defensible
moat hypotheses emerge?

This notebook uses **only existing, already-verified repository outputs** -- no new
collection, no external API calls, no engagement prediction, no causal inference, no
manual labelling. See `CLAUDE.md` §9 and `docs/project_charter.md` for the binding scope.""")

code("""import sys
from pathlib import Path

PROJECT_ROOT = Path().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_columns", 30)

TABLES = PROJECT_ROOT / "outputs" / "tables"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
print("Project root:", PROJECT_ROOT)""")

# 2. Data sources and boundaries
md("""## 2. Data sources and boundaries

Inputs (all already verified in Day 1-3B):
- `data/processed/fusion/claim_evidence_units.csv`, `claim_fusion_results.csv`,
  `fusion_configuration_results.csv`, `fusion_sensitivity_results.csv`
- `data/processed/official_claim_features.csv`, `claim_multimodal_evidence_candidates.csv`
- `data/processed/creator_multimodal_features.csv`, `platform_strategy_features.csv`

Known evidence boundaries carried forward from Day 2/3A/3B (see
`docs/creator_strategy_comparison.md` for the full list):
- Creator evidence is overwhelmingly YouTube-based (60/61 verified records); 1 TikTok record.
- Creator records were collected via a targeted evidence-recovery process, not a
  statistically representative sample.
- Cross-platform creator-**performance** comparison is not permitted; engagement prediction
  is out of scope.
- Partnership/intent labels are automated, evidence-gated classifications, not
  human-labelled ground truth.""")

code("""units = pd.read_csv(PROJECT_ROOT / "data/processed/fusion/claim_evidence_units.csv")
fusion = pd.read_csv(PROJECT_ROOT / "data/processed/fusion/claim_fusion_results.csv")
creators = pd.read_csv(PROJECT_ROOT / "data/processed/creator_multimodal_features.csv")
platform = pd.read_csv(PROJECT_ROOT / "data/processed/platform_strategy_features.csv")
print("claim_evidence_units:", units.shape)
print("claim_fusion_results:", fusion.shape)
print("creator_multimodal_features:", creators.shape, "| brands:", creators['brand'].value_counts().to_dict())
print("platform_strategy_features:", platform.shape)""")

# 3. Claim-level synthesis
md("""## 3. Claim-level synthesis

The authoritative, claim-level synthesis table joining Day 3A fusion labels to
evidence-unit counts by modality and role, for all 37 TALA claims.""")

code("""import subprocess
subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_analytical_synthesis.py")], check=True)
subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_mixed_claim_themes.py")], check=True)

master = pd.read_csv(TABLES / "analytical_synthesis_claim_master.csv")
assert len(master) == 37 and master["claim_id"].nunique() == 37
master[["claim_id", "claim_category", "production_fusion_label", "fusion_confidence",
        "supporting_evidence_count", "challenging_evidence_count", "independent_evidence_count"]].head(10)""")

# 4. Fusion-label distribution
md("""## 4. Fusion-label distribution""")

code("""label_summary = pd.read_csv(TABLES / "claim_label_summary.csv")
display(label_summary[label_summary["section"] == "overall_label_distribution"])
display(label_summary[label_summary["section"] == "by_claim_category"].pivot(
    index="claim_category", columns="automated_label", values="n_claims"))
from IPython.display import Image
Image(filename=str(FIGURES / "claim_label_distribution.png"))""")

# 5. Mixed-claim deep dive
md("""## 5. Mixed-claim deep dive

Every claim RECORD labelled `mixed` -- both supporting and challenging evidence exist; the
narrative explains the conflict rather than forcing a binary conclusion. **Record-level
vs. theme-level:** five claim records were classified as mixed, but they consolidate into
two underlying strategic tensions (deterministic text-match grouping, not five independent
areas of contradiction).""")

code("""mixed_dd = pd.read_csv(TABLES / "mixed_claim_deep_dive.csv")
mixed_themes = pd.read_csv(TABLES / "mixed_claim_theme_summary.csv")
print(f"Claim-record level: {len(mixed_dd)}/{len(master)} mixed")
print(f"Theme level: {len(mixed_themes)} distinct mixed themes")
mixed_dd[["claim_id", "theme_id", "exact_official_claim", "supporting_evidence_count",
          "challenging_evidence_count", "why_evidence_conflicts"]]""")

code("""mixed_themes[["theme_id", "theme_name", "claim_record_count", "claim_ids", "observed_tension"]]""")

# 6. Insufficient-evidence analysis
md("""## 6. Insufficient-evidence analysis

Insufficient evidence is a **gap finding**, never interpreted as contradiction.""")

code("""insuff = pd.read_csv(TABLES / "insufficient_evidence_analysis.csv")
print(f"{len(insuff)} insufficient-evidence claims (of {len(master)} total)")
insuff["reason_label"].value_counts()""")

code("""aligned_ex = pd.read_csv(TABLES / "aligned_claim_examples.csv")
aligned_ex[["claim_id", "claim_text", "selection_criteria", "alignment_basis"]]""")

# 7. Modality contribution
md("""## 7. Modality contribution

Comparing the four fusion configurations (text-only -> text+image -> text+image+video ->
full multimodal) shows label **stability** with confidence **drift** as modalities are added.""")

code("""modality = pd.read_csv(TABLES / "modality_contribution_summary.csv")
display(modality[modality["section"] == "configuration_comparison"])
display(modality[modality["section"] == "cross_configuration_summary"])
Image(filename=str(FIGURES / "modality_confidence_contribution.png"))""")

code("""gap = pd.read_csv(TABLES / "evidence_gap_by_category.csv")
gap""")

# 8. Creator evidence coverage
md("""## 8. Creator evidence coverage (Part B)

**Analytical scope of Part B: cross-brand YouTube creator-strategy comparison combined
with an official-channel platform comparison.** 60 of 61 verified creator records are
YouTube, 1 is TikTok, 0 are Instagram -- this is not a comprehensive cross-platform
creator comparison, not an Instagram creator analysis, and not a representative sample of
each brand's complete creator strategy.""")

code("""subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_creator_strategy_comparison.py")], check=True)
coverage = pd.read_csv(TABLES / "creator_strategy_evidence_coverage.csv")
coverage""")

# 9. Partnership comparison
md("""## 9. Partnership comparison""")

code("""partnership_mix = pd.read_csv(TABLES / "creator_partnership_mix.csv")
display(partnership_mix.pivot(index="brand", columns="partnership_type", values="pct_within_brand"))
Image(filename=str(FIGURES / "creator_partnership_mix_by_brand.png"))""")

# 10. Content-intent comparison
md("""## 10. Content-intent comparison""")

code("""intent_mix = pd.read_csv(TABLES / "creator_content_intent_mix.csv")
display(intent_mix.pivot(index="brand", columns="content_intent", values="pct_within_brand"))
Image(filename=str(FIGURES / "creator_intent_mix_by_brand.png"))""")

# 11. Partnership x intent
md("""## 11. Partnership x intent

The central question: **what job does each partnership type appear to perform**, not
which partnership performs best.""")

code("""matrix = pd.read_csv(TABLES / "creator_partnership_intent_matrix.csv")
non_organic = matrix[matrix["scope"] == "non_organic_only"]
display(non_organic.pivot(index="partnership_type", columns="content_intent", values="count"))
Image(filename=str(FIGURES / "nonorganic_partnership_intent_matrix.png"))""")

# 12. Platform roles
md("""## 12. Platform roles""")

code("""platform_role = pd.read_csv(TABLES / "brand_platform_strategy_comparison.csv")
platform_role
""")

code("""Image(filename=str(FIGURES / "brand_platform_strategy_matrix.png"))""")

# 13. Competitor lessons
md("""## 13. Competitor lessons""")

code("""subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_strategic_interpretation.py")], check=True)
lessons = pd.read_csv(TABLES / "tala_competitor_lessons.csv")
lessons[["lesson_id", "comparator_brand", "observed_competitor_practice", "lesson_category", "potential_tala_action"]]""")

# 14. Moat hypotheses
md("""## 14. Moat hypotheses

Status is always one of `plausible / weak / unsupported / contradicted` -- never
"proven moat".""")

code("""moat = pd.read_csv(TABLES / "creator_moat_hypotheses.csv")
moat[["hypothesis_id", "brand", "observed_pattern", "verified_evidence_count", "status"]]""")

code("""pattern_metrics = pd.read_csv(TABLES / "creator_strategic_pattern_metrics.csv")
pattern_metrics
Image(filename=str(FIGURES / "tala_competitor_strategy_heatmap.png"))""")

# 15. Executive findings
md("""## 15. Executive findings""")

code("""subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_executive_findings.py")], check=True)
findings = pd.read_csv(TABLES / "executive_finding_register.csv")
print(f"{len(findings)} executive findings")
findings[["finding_id", "analytical_stream", "finding", "observed_or_inferred"]]""")

code("""prof = pd.read_csv(TABLES / "professor_feedback_response.csv")
prof[["question_id", "question", "answer"]]""")

# 16. Limitations
md("""## 16. Limitations

- Automated NLI/rule-based fusion labels are **not human-adjudicated**.
- Creator/partnership/intent labels are automated, evidence-gated classifications.
- Creator sample is targeted evidence-recovery, not statistically representative;
  brand sample sizes range n=8 (Oner Active) to n=24 (TALA).
- No Instagram creator records verified for any brand.
- Cross-platform creator-performance comparison and engagement prediction are out of scope.
- Moat hypotheses are hypotheses, not proven competitive advantages.
- Absence of a partnership/intent category in the corpus does not prove the brand never
  uses it.""")

# 17. Final conclusions
md("""## 17. Final conclusions

See `docs/executive_analytical_synthesis.md` and `docs/creator_strategy_comparison.md`
for the full deck-ready narrative. In summary: the verified corpus places 14/37 (37.8%)
of TALA's claims as `aligned`, 5/37 (13.5%) as `mixed` at the claim-record level --
consolidating into 2 underlying strategic themes (brand mission; tier-3
supplier/material innovation), not five independent contradictions -- and 18/37 (48.6%)
as `insufficient_evidence` (a corpus gap, not a contradiction). Multimodal evidence
changed confidence but not labels across the four fusion configurations. Within the
observed, targeted creator sample (a cross-brand YouTube creator-strategy comparison
combined with an official-channel platform comparison, not a comprehensive
cross-platform read), Oner Active has the highest observed organic share (87.5%, n=8),
but its sample is too shallow for a confident conclusion; TALA has an organic plurality
within a larger sample (62.5%, n=24) and is the only brand with verified
founder/employee and ambassador creator records -- a plausible-but-weak moat hypothesis,
not a proven one.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

out_path = "notebooks/06_analytical_synthesis_and_creator_strategy.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote", out_path, "with", len(cells), "cells")
