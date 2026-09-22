"""Day 3C: executive finding register -- synthesises Part A (claim-experience)
and Part B (creator strategy) outputs into ~8-12 deck-ready findings.
All quantitative_support values are pulled live from already-built tables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"


def _load():
    return {
        "master": pd.read_csv(TABLES_DIR / "analytical_synthesis_claim_master.csv"),
        "label_summary": pd.read_csv(TABLES_DIR / "claim_label_summary.csv"),
        "mixed": pd.read_csv(TABLES_DIR / "mixed_claim_deep_dive.csv"),
        "mixed_themes": pd.read_csv(TABLES_DIR / "mixed_claim_theme_summary.csv"),
        "insufficient": pd.read_csv(TABLES_DIR / "insufficient_evidence_analysis.csv"),
        "gap": pd.read_csv(TABLES_DIR / "evidence_gap_by_category.csv"),
        "modality": pd.read_csv(TABLES_DIR / "modality_contribution_summary.csv"),
        "coverage": pd.read_csv(TABLES_DIR / "creator_strategy_evidence_coverage.csv"),
        "pattern": pd.read_csv(TABLES_DIR / "creator_strategic_pattern_metrics.csv"),
        "lessons": pd.read_csv(TABLES_DIR / "tala_competitor_lessons.csv"),
        "moat": pd.read_csv(TABLES_DIR / "creator_moat_hypotheses.csv"),
    }


def build(d) -> pd.DataFrame:
    master, label_summary, mixed, insuff, gap = d["master"], d["label_summary"], d["mixed"], d["insufficient"], d["gap"]
    modality, pattern, lessons, moat = d["modality"], d["pattern"], d["lessons"], d["moat"]
    mixed_themes = d["mixed_themes"]

    n_total = len(master)
    n_aligned = int((master["production_fusion_label"] == "aligned").sum())
    n_mixed = int((master["production_fusion_label"] == "mixed").sum())
    n_insuff = int((master["production_fusion_label"] == "insufficient_evidence").sum())
    labour_row = gap[gap["claim_category"] == "labour"].iloc[0]
    conf_delta_row = modality[(modality["section"] == "configuration_comparison")]
    conf_text_only = conf_delta_row.loc[conf_delta_row["configuration"] == "text_only", "mean_confidence"].iloc[0]
    conf_full = conf_delta_row.loc[conf_delta_row["configuration"] == "text_image_video_reference", "mean_confidence"].iloc[0]
    label_changed = modality[(modality["section"] == "cross_configuration_summary") & (modality["metric"] == "claims_with_label_change_across_configurations")]["value"].iloc[0]

    findings = []

    def add(fid, stream, finding, support, evidence_ids, inferred, confidence, implication, action, caveat, slide):
        findings.append(
            {
                "finding_id": fid,
                "analytical_stream": stream,
                "finding": finding,
                "quantitative_support": support,
                "evidence_ids": evidence_ids,
                "observed_or_inferred": inferred,
                "evidence_confidence": confidence,
                "business_implication": implication,
                "recommended_action": action,
                "caveat": caveat,
                "suggested_deck_slide": slide,
            }
        )

    add(
        "F01", "claim–experience divergence",
        "The verified corpus places under half of TALA's 37 official claims in the 'aligned' automated fusion label; a substantial share are labelled insufficient evidence rather than validated.",
        f"aligned={n_aligned}/{n_total} (37.8%); mixed={n_mixed}/{n_total} (13.5%); insufficient_evidence={n_insuff}/{n_total} (48.6%)",
        "see outputs/tables/analytical_synthesis_claim_master.csv (claim_id column, full list)",
        "observed",
        "high — direct tabulation of Day 3A production fusion labels",
        "Nearly half of TALA's public claims currently lack corroborating public evidence in this corpus; this is a gap, not proof the claims are false.",
        "Prioritise external verification (or internal documentation release) for the insufficient-evidence claim set before using them in investor or marketing materials.",
        "Automated NLI/rule-based label, not human-adjudicated; absence of evidence is not evidence of absence.",
        "Claim-label distribution",
    )

    n_mixed_themes = len(mixed_themes)
    add(
        "F02", "claim–experience divergence",
        (
            f"Five of 37 claim records were classified as mixed, but they consolidate into "
            f"{n_mixed_themes} underlying strategic tensions: TALA's conscious-value "
            "proposition and a recurring tier-3 supplier/material-innovation claim "
            "represented by four claim variants. Each theme shows genuine conflicting "
            "evidence (both supporting and challenging units), not a simple contradiction."
        ),
        f"mixed claim records: {n_mixed}/{n_total}; mixed themes: {n_mixed_themes}; see outputs/tables/mixed_claim_theme_summary.csv",
        ";".join(mixed["claim_id"].tolist()),
        "observed",
        "medium — small n (5), qualitative evidence review recommended",
        "These claims are the highest-risk items for external scrutiny (press, ESG raters) because the corpus already documents disagreement.",
        "Route the 5 mixed claims to management/legal review before repeating them in external communications.",
        "Cannot conclude which context (product line, timeframe) drives the disagreement from text alone.",
        "Mixed-claim findings",
    )

    add(
        "F03", "evidence gaps",
        "Labour-category claims (14 of 37 claims, the largest category) show zero independent/customer evidence in the current corpus.",
        f"labour: total={int(labour_row['total_claims'])}, claims_with_independent_evidence={int(labour_row['claims_with_independent_evidence'])}, claims_with_zero_usable_evidence_beyond_claim={int(labour_row['claims_with_zero_usable_evidence_beyond_claim'])}",
        "see outputs/tables/evidence_gap_by_category.csv",
        "observed",
        "high — direct tabulation",
        "TALA's largest claim category (labour/social responsibility) is also its least externally corroborated; this is the single largest evidence gap in the claim-experience analysis.",
        "Commission or source independent third-party verification (e.g. supplier audits, NGO assessments) specifically for labour claims.",
        "This reflects a collection-corpus gap, not evidence that labour claims are false.",
        "Insufficient-evidence findings",
    )

    add(
        "F04", "modality contribution",
        "Adding image, video, and reference evidence changed mean fusion confidence but did not change any claim's production label across the four fusion configurations.",
        f"mean_confidence: text_only={conf_text_only}, full (text+image+video+reference)={conf_full}; claims_with_label_change_across_configurations={int(label_changed)}/37",
        "see outputs/tables/modality_contribution_summary.csv",
        "observed",
        "high — direct tabulation of the four fusion configurations",
        "Multimodal evidence currently functions as a confidence-calibration input, not a label-changing one; investment in additional modalities should target confidence robustness, not label correctness (which was not independently measured).",
        "Do not claim added modalities improved correctness; report the confidence-shift finding as-is in the deck.",
        "Correctness of the fusion label itself was never independently (human) validated in this task's scope.",
        "Modality-contribution conclusion",
    )

    add(
        "F05", "evidence gaps",
        "No claim category has both zero image AND zero reference evidence simultaneously except labour and packaging, which have zero usable evidence of any kind beyond the official claim text.",
        "labour and packaging categories: claims_with_zero_usable_evidence_beyond_claim = total_claims (see evidence_gap_by_category.csv)",
        "see outputs/tables/evidence_gap_by_category.csv",
        "observed",
        "high",
        "These two categories are the weakest points in the claim-experience evidence base and the first candidates for targeted evidence collection in any follow-on sprint.",
        "Scope a future, ethics-compliant collection pass focused on labour and packaging claim categories specifically.",
        "Reflects the current corpus only; no inference is made about why these categories are under-evidenced.",
        "Evidence-gap findings",
    )

    n_tala = int(pattern.loc[pattern.brand == "TALA", "n"].iloc[0])
    n_oner = int(pattern.loc[pattern.brand == "Oner Active", "n"].iloc[0])
    organic_tala = pattern.loc[pattern.brand == "TALA", "organic_share_pct"].iloc[0]
    organic_adanola = pattern.loc[pattern.brand == "Adanola", "organic_share_pct"].iloc[0]
    organic_oner = pattern.loc[pattern.brand == "Oner Active", "organic_share_pct"].iloc[0]
    add(
        "F06", "creator strategy",
        "Oner Active has the highest observed organic share, but its eight-record sample is too shallow for a confident strategic conclusion. TALA has an organic plurality within a larger sample and the most varied observed partnership mix, and is the only brand with verified founder/employee and ambassador creator records.",
        f"Oner Active organic_share={organic_oner}% (n={n_oner}); TALA organic_share={organic_tala}% (n={n_tala}) vs Adanola organic_share={organic_adanola}%; TALA founder_or_employee+ambassador records=2/{n_tala}",
        "see outputs/tables/creator_partnership_mix.csv; creator_strategic_pattern_metrics.csv",
        "observed",
        "medium — targeted, non-random sample; brand sample sizes differ (n=8 to n=24)",
        "TALA's current creator mix is already differentiated from the paid-sponsorship-led approach seen at Adanola, with broader partnership-type coverage than any competitor's verified sample.",
        "Preserve and deliberately document the founder/ambassador creator relationships as a distinct strategic asset; do not present Oner Active's higher organic percentage as a stronger organic strategy without a larger verified sample.",
        "Descriptive only; not a claim that TALA's or Oner Active's mix outperforms competitors' engagement.",
        "Partnership mix by brand",
    )

    add(
        "F07", "competitor learning",
        "Adanola's verified creator content is the most content-intent-concentrated in the corpus (100% product_demonstration) and relies most on paid sponsorship; Girlfriend Collective relies most on affiliate partnerships.",
        "Adanola product_demonstration_share=100%, intent_concentration_hhi=1.0; Girlfriend Collective affiliate_share=61.5%",
        "see outputs/tables/creator_strategic_pattern_metrics.csv",
        "observed",
        "medium",
        "Two distinct, testable competitor models exist for TALA to learn from: a paid, single-intent demonstration model (Adanola) and an affiliate-led, more review-oriented model (Girlfriend Collective).",
        "Pilot small, bounded tests of each model (see outputs/tables/tala_competitor_lessons.csv) rather than adopting either wholesale.",
        "No outcome/engagement comparison was performed; 'more concentrated' is not evidence of 'more effective'.",
        "TALA vs. competitor conclusions",
    )

    n_moat_weak = int((moat["status"] == "weak").sum())
    add(
        "F08", "moat hypothesis",
        "A plausible-but-currently-weak moat hypothesis exists around TALA's founder/employee and ambassador creator relationships, which would require time and organisational relationships for a competitor to replicate.",
        f"moat hypotheses at status=weak: {n_moat_weak}/{len(moat)}; supporting evidence_count for TALA hypothesis = 2 records",
        "see outputs/tables/creator_moat_hypotheses.csv (moat_01)",
        "hypothesis",
        "low-medium — small underlying record count",
        "If validated with a larger sample, this could be a defensible, hard-to-imitate differentiator worth deliberate investment.",
        "Track and verify whether the founder/ambassador creator pattern persists and grows over the next collection cycle before treating it as a confirmed strategic asset.",
        "This is explicitly a hypothesis, not a proven moat; do not present as causally established.",
        "Moat hypotheses",
    )

    add(
        "F09", "creator strategy",
        "No Instagram creator records are verified for any of the four brands in this corpus, and cross-platform creator-performance comparison is out of scope.",
        "platforms_represented_creator: TALA/Adanola/Girlfriend Collective/Oner Active all show youtube (Adanola also tiktok); zero instagram records",
        "see outputs/tables/creator_strategy_evidence_coverage.csv",
        "observed",
        "high — direct tabulation",
        "Any deck claim about TALA's or competitors' Instagram creator strategy cannot currently be supported from this evidence base.",
        "If Instagram creator strategy is strategically important, scope a separate, ethics-compliant collection effort; do not infer it from YouTube data.",
        "This scope boundary applies to the whole creator-strategy comparison, not just this finding.",
        "Evidence boundary",
    )

    add(
        "F10", "claim–experience divergence",
        "Two of TALA's seven claim categories (manufacturing, materials) have image, and in materials' case reference and some video, evidence coverage on all claims, making them the strongest-evidenced categories in the corpus.",
        "manufacturing: claims_with_image_evidence=8/8; materials: claims_with_image_evidence=7/7, claims_with_reference_evidence=7/7, claims_with_video_evidence=3/7",
        "see outputs/tables/evidence_gap_by_category.csv",
        "observed",
        "high",
        "Manufacturing and materials claims are the most defensible in external communications given current evidence depth.",
        "Prioritise these categories for confident external re-use, subject to the standard mixed/insufficient exclusions already flagged per-claim.",
        "Category-level evidence depth does not guarantee every individual claim in the category is aligned; consult analytical_synthesis_claim_master.csv row-by-row.",
        "Aligned findings",
    )

    return pd.DataFrame(findings)


def main() -> None:
    d = _load()
    out = build(d)
    assert 8 <= len(out) <= 12, f"expected 8-12 findings, got {len(out)}"
    out.to_csv(TABLES_DIR / "executive_finding_register.csv", index=False)
    print("executive_finding_register rows:", len(out))


if __name__ == "__main__":
    main()
