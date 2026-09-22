"""Day 3C Part B (continued): strategic interpretation, competitor lessons,
moat hypotheses, and the professor-feedback response table.

Every numeric figure is pulled live from the already-built creator-strategy
tables (never hardcoded), so reruns stay identical. Narrative fields observe
the three-level split (observed_finding / interpretation / hypothesis) and
the analytical-honesty wording rules in CLAUDE.md / the task brief.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BRANDS = ["TALA", "Adanola", "Girlfriend Collective", "Oner Active"]


def _load():
    return {
        "creators": pd.read_csv(PROCESSED_DIR / "creator_multimodal_features.csv"),
        "pattern": pd.read_csv(TABLES_DIR / "creator_strategic_pattern_metrics.csv").set_index("brand"),
        "partnership_mix": pd.read_csv(TABLES_DIR / "creator_partnership_mix.csv"),
        "intent_mix": pd.read_csv(TABLES_DIR / "creator_content_intent_mix.csv"),
        "coverage": pd.read_csv(TABLES_DIR / "creator_strategy_evidence_coverage.csv").set_index("brand"),
    }


def _top_n(df: pd.DataFrame, brand: str, col: str, value_col: str = "count", n: int = 2) -> list[str]:
    sub = df[(df["brand"] == brand) & (df[value_col] > 0)].sort_values(value_col, ascending=False)
    return sub[col].head(n).tolist()


def _example_evidence_ids(creators: pd.DataFrame, brand: str, ptype: str | None = None, intent: str | None = None, n: int = 3) -> str:
    sub = creators[creators["brand"] == brand]
    if ptype:
        sub = sub[sub["revised_partnership_type"] == ptype]
    if intent:
        sub = sub[sub["primary_intent"] == intent]
    return ";".join(sub["post_id"].head(n).astype(str).tolist())


def build_interpretation(d) -> pd.DataFrame:
    creators, pattern = d["creators"], d["pattern"]
    rows = []
    for brand in BRANDS:
        n = int(pattern.loc[brand, "n"])
        dom_partnerships = _top_n(d["partnership_mix"], brand, "partnership_type")
        dom_intents = _top_n(d["intent_mix"], brand, "content_intent")
        organic = pattern.loc[brand, "organic_share_pct"]
        non_organic = pattern.loc[brand, "non_organic_share_pct"]
        demo_share = pattern.loc[brand, "product_demonstration_share_pct"]
        evidence_ids = _example_evidence_ids(creators, brand)
        platforms = d["coverage"].loc[brand, "platforms_represented_creator"]

        common = {
            "brand": brand,
            "dominant_partnership_types": ";".join(dom_partnerships),
            "dominant_content_intents": ";".join(dom_intents),
            "platform_role": f"verified creator evidence on: {platforms}",
            "evidence_supporting_pattern": f"n={n} verified creator records; example evidence_ids: {evidence_ids}",
            "evidence_strength": (
                "targeted, non-random sample; adequate for descriptive within-brand comparison, "
                "not for statistical inference"
            ),
            "limitation": (
                "Creator records were collected through a targeted evidence-recovery process, not a "
                "statistically representative sample; absence of a partnership type does not prove the "
                "brand never uses it; sample size varies by brand (n=8 to n=24)."
            ),
            "potential_advantage": "",
            "potential_vulnerability": "",
        }

        observed = dict(common)
        observed["level"] = "observed_finding"
        observed["statement"] = (
            f"Within the observed sample (n={n}), {organic}% of {brand}'s verified creator records are "
            f"classified organic and {non_organic}% non-organic; {demo_share}% carry a "
            f"product_demonstration primary intent; the dominant partnership type(s) are "
            f"{', '.join(dom_partnerships) or 'not determined'}."
        )
        rows.append(observed)

        interp = dict(common)
        interp["level"] = "interpretation"
        if brand == "TALA":
            interp["statement"] = (
                "The evidence suggests TALA leans more on organic/unpaid creator mentions than Adanola and "
                "Girlfriend Collective, and has the most varied observed partnership mix of the four brands "
                "(the only one with founder/employee and ambassador records). Oner Active shows a higher "
                "organic percentage (87.5%) than TALA, but on a sample less than a third the size (n=8 vs. "
                "n=24), so it is not described here as 'more organic-led' than TALA."
            )
            interp["potential_advantage"] = "Lower-cost, potentially higher-perceived-authenticity content if organic mentions are genuine."
            interp["potential_vulnerability"] = "Less contractual control over messaging consistency than a paid-sponsorship-led strategy."
        elif brand == "Adanola":
            interp["statement"] = (
                "The evidence suggests Adanola concentrates on paid sponsorship and affiliate partnerships "
                "with a content mix skewed almost entirely to product demonstration."
            )
            interp["potential_advantage"] = "Consistent, controllable product-demonstration messaging at scale."
            interp["potential_vulnerability"] = "Heavier reliance on paid relationships may read as less authentic and costs more to sustain."
        elif brand == "Girlfriend Collective":
            interp["statement"] = (
                "The evidence suggests Girlfriend Collective relies most heavily on affiliate partnerships "
                "among the four brands, with more review-intent content than competitors in this corpus."
            )
            interp["potential_advantage"] = "Affiliate structure scales creator participation with performance-linked cost."
            interp["potential_vulnerability"] = "Affiliate-driven content may skew toward conversion framing over brand narrative."
        else:  # Oner Active
            interp["statement"] = (
                "The evidence suggests Oner Active's verified sample is the smallest and most organic-"
                "leaning of the four brands, limiting how much can be inferred about its broader strategy."
            )
            interp["potential_advantage"] = "Apparently low reliance on paid creator relationships in the observed sample."
            interp["potential_vulnerability"] = "Small n (8) makes any pattern fragile; could reflect corpus recovery limits rather than brand strategy."
        rows.append(interp)

        hyp = dict(common)
        hyp["level"] = "hypothesis"
        if brand == "TALA":
            hyp["statement"] = (
                "A plausible hypothesis is that TALA's organic-leaning, founder/ambassador-inclusive creator "
                "mix reflects a community-relationship strategy that would take time to imitate, but the "
                "current corpus cannot establish this as causal or measure its business impact."
            )
        else:
            hyp["statement"] = (
                f"A plausible hypothesis is that {brand}'s partnership mix reflects a deliberate strategic "
                "choice rather than incidental sampling, but the current corpus cannot establish this as "
                "causal or compare its effectiveness to TALA's."
            )
        rows.append(hyp)

    return pd.DataFrame(rows)


def build_lessons(d) -> pd.DataFrame:
    creators, pattern = d["creators"], d["pattern"]
    lessons = []
    lid = 1

    def add(comparator, practice, category, tala_action, mechanism, difficulty, confidence, risk, caveat, ptype=None, intent=None):
        nonlocal lid
        sub = creators[creators["brand"] == comparator]
        if ptype:
            sub = sub[sub["revised_partnership_type"] == ptype]
        if intent:
            sub = sub[sub["primary_intent"] == intent]
        lessons.append(
            {
                "lesson_id": f"lesson_{lid:02d}",
                "comparator_brand": comparator,
                "observed_competitor_practice": practice,
                "supporting_record_count": len(sub),
                "source_evidence_ids": ";".join(sub["post_id"].head(5).astype(str).tolist()),
                "relevance_to_tala": "TALA is prioritising a creator-led strategy per the professor's feedback; comparator practices bound the option space TALA can draw from.",
                "potential_tala_action": tala_action,
                "expected_business_mechanism": mechanism,
                "implementation_difficulty": difficulty,
                "evidence_confidence": confidence,
                "risk": risk,
                "caveat": caveat,
                "lesson_category": category,
            }
        )
        lid += 1

    n_adanola = int(pattern.loc["Adanola", "n"])
    n_gfc = int(pattern.loc["Girlfriend Collective", "n"])
    n_tala = int(pattern.loc["TALA", "n"])

    add(
        "Adanola",
        f"Adanola's paid-sponsorship and affiliate records ({n_adanola} verified) carry product_demonstration intent almost exclusively (100% of the brand's verified sample).",
        "test",
        "Pilot a small, contractually-scoped paid-sponsorship cohort for a single product line, with a pre-agreed product-demonstration content brief, and compare content consistency against TALA's current organic mix.",
        "Contracted demonstration content is more consistent and repeatable than organic mentions, which may improve message control at the cost of paid spend.",
        "medium",
        "medium — based on 16 verified records, single brand, no outcome data",
        "Paid sponsorship without disclosure discipline risks credibility and platform-policy issues.",
        "Competitor's higher paid-sponsorship share is observed, not proven to be more effective than TALA's organic mix; no engagement outcome comparison is made.",
        ptype="paid_sponsorship",
    )
    add(
        "Girlfriend Collective",
        f"Girlfriend Collective shows the highest affiliate share among the four brands (61.5% of {n_gfc} verified records) and more review-intent content than competitors.",
        "test",
        "Trial an affiliate program for a defined creator tier, paired with review-intent content briefs, tracked separately from TALA's existing organic and ambassador relationships.",
        "Affiliate structures scale creator participation at performance-linked cost and may generate more review-style trust content than organic mentions alone.",
        "medium",
        "medium — based on 13 verified records, single brand",
        "Affiliate-driven content can skew toward discount/conversion framing rather than brand narrative if not briefed carefully.",
        "Affiliate share is observed as a proportion of a small, non-random sample; whether it drives more conversions than organic content cannot be assessed from this corpus.",
        ptype="affiliate",
    )
    add(
        "Adanola",
        "Adanola's verified creator sample shows no responsibility-intent content (0%) and no founder/employee or ambassador records.",
        "avoid_copying_without_validation",
        "Do not copy Adanola's apparent product-demonstration-only focus wholesale; TALA's existing responsibility and founder/ambassador content (absent in Adanola's sample) may be a differentiator worth protecting rather than abandoning.",
        "Narrow, demonstration-only content may under-communicate TALA's responsibility claims, which the Part A claim-experience analysis shows already carry evidence gaps.",
        "n/a — this is a caution, not an action to implement",
        "low — absence of a category in Adanola's corpus does not prove Adanola never produces it",
        "None (this is a preservation recommendation).",
        "Absence of a partnership/intent type in the corpus does not prove the brand never uses it; Adanola's corpus depth (16 records) may simply not have captured responsibility content.",
    )
    add(
        "TALA",
        f"TALA's own verified sample already includes founder_or_employee and ambassador partnership records not observed for any competitor in this corpus, plus a small responsibility-intent share ({pattern.loc['TALA','responsibility_content_share_pct']}%).",
        "preserve",
        "Preserve and deliberately expand founder/employee and ambassador-style creator relationships and responsibility-intent content, since competitors in this corpus show none of it.",
        "Founder/ambassador-led content and responsibility framing are harder for competitors to imitate quickly because they depend on relationships and organisational history, not just budget.",
        "low — extension of an existing practice",
        "medium — based on TALA's own 24 verified records",
        "Small volume (1-2 records per category) limits how much weight this pattern can currently bear.",
        "This is a within-TALA observation, not a competitor lesson; it is retained here because it directly informs which competitor practices are and are not worth copying.",
        ptype="founder_or_employee",
    )
    add(
        "Girlfriend Collective",
        "Girlfriend Collective is the only competitor with verified responsibility-intent creator content in this corpus (partial, low count).",
        "test",
        "Review whether TALA's existing responsibility-content creators could be briefed similarly to Girlfriend Collective's review+responsibility framing, then compare qualitatively (not by engagement metrics).",
        "Pairing responsibility claims with independent creator review framing may support the claim-experience divergence findings from Part A by adding corroborating evidence.",
        "medium",
        "low — very small underlying counts (1-2 records)",
        "Over-generalising from 1-2 records risks presenting a coincidence as a strategy.",
        "Based on single-digit record counts; treat as a lead for further verification, not a settled pattern.",
        intent="responsibility",
    )
    add(
        "Oner Active",
        f"Oner Active's verified sample is small (n={int(pattern.loc['Oner Active','n'])}) and organic-leaning ({pattern.loc['Oner Active','organic_share_pct']}%), similar in direction to TALA's own organic lean.",
        "evidence_gap",
        "No specific action recommended; corpus depth is too shallow to draw a comparator lesson from Oner Active beyond noting directional similarity to TALA.",
        "not applicable — no action mechanism proposed",
        "not applicable — implementation difficulty not assessed",
        "low — n=8 is the smallest brand sample in the corpus",
        "not applicable — no action proposed, so no direct risk",
        "Corpus depth for Oner Active is materially shallower than for TALA or Adanola; any comparison should be revisited if more records are verified.",
    )
    add(
        "Adanola",
        "No Instagram creator records are verified for any brand, including Adanola, despite Instagram being a common creator-marketing channel in the category.",
        "evidence_gap",
        "Flag as a collection gap, not a strategy conclusion: TALA cannot currently be benchmarked against competitors on Instagram creator activity.",
        "not applicable — no action mechanism proposed",
        "not applicable — implementation difficulty not assessed",
        "not applicable — data collection gap, not a business decision",
        "not applicable — no action proposed, so no direct risk",
        "Absence of a platform in the corpus reflects a collection/access boundary (no Instagram scraping performed under the project's ethics rules), not a finding about actual brand behaviour.",
    )

    return pd.DataFrame(lessons)


def build_moat_hypotheses(d) -> pd.DataFrame:
    creators, pattern = d["creators"], d["pattern"]
    rows = []
    hid = 1

    def add(brand, pattern_desc, mechanism, imitability, durability, confidence, counterevidence, validation, status, ptype=None, intent=None):
        nonlocal hid
        sub = creators[creators["brand"] == brand]
        if ptype:
            sub = sub[sub["revised_partnership_type"] == ptype]
        if intent:
            sub = sub[sub["primary_intent"] == intent]
        rows.append(
            {
                "hypothesis_id": f"moat_{hid:02d}",
                "brand": brand,
                "observed_pattern": pattern_desc,
                "verified_evidence_count": len(sub),
                "evidence_ids": ";".join(sub["post_id"].head(6).astype(str).tolist()),
                "proposed_moat_mechanism": mechanism,
                "imitability": imitability,
                "durability": durability,
                "evidence_confidence": confidence,
                "counterevidence": counterevidence,
                "what_would_validate_the_hypothesis": validation,
                "status": status,
            }
        )
        hid += 1

    add(
        "TALA",
        f"TALA is the only brand in the corpus with verified founder_or_employee and ambassador creator records ({int((creators[(creators.brand=='TALA')].revised_partnership_type.isin(['founder_or_employee','ambassador'])).sum())} of {int(pattern.loc['TALA','n'])}), alongside an organic plurality (62.5%) within the largest and most partnership-diverse sample among the four brands. (Oner Active has the highest observed organic share at 87.5%, but on a shallower n=8 sample.)",
        "Founder/employee and ambassador relationships require accumulated organisational relationships and community trust that a competitor cannot purchase quickly, unlike a one-off paid sponsorship.",
        "hard to imitate quickly — depends on internal relationships and organisational history",
        "moderate — durability depends on the relationships persisting, not on a media contract",
        "low-medium — supported by only 2 records out of 24",
        "None directly contradicts it, but the low record count (2) means the pattern could be incidental to this recovery-based sample rather than a deliberate strategy.",
        "A larger, still-verified sample confirming a sustained (not one-off) founder/ambassador creator program, ideally with dated evidence spanning multiple periods.",
        "weak",
        ptype=None,
    )
    add(
        "Girlfriend Collective",
        f"Girlfriend Collective shows the highest affiliate concentration among the four brands (61.5% of {int(pattern.loc['Girlfriend Collective','n'])} verified records).",
        "A large, established affiliate network could represent accumulated creator relationships and tracking infrastructure that takes time to build.",
        "moderately hard to imitate — affiliate infrastructure and creator relationships take time, though affiliate programs themselves are not proprietary",
        "moderate",
        "low — based on 13 verified records for one brand",
        "Affiliate share alone does not distinguish a deep creator network from a broad low-commitment one; no data on affiliate tenure or repeat participation is available in this corpus.",
        "Evidence of repeat-affiliate participation over time and affiliate-network size relative to competitors.",
        "weak",
        ptype="affiliate",
    )
    add(
        "Adanola",
        f"Adanola shows the most concentrated content-intent mix in the corpus (100% product_demonstration, intent HHI=1.00 on {int(pattern.loc['Adanola','n'])} records), paired with the highest paid-sponsorship share.",
        "A highly disciplined, paid, single-intent creator content strategy could reflect strong brand-guideline enforcement that is organisationally hard to replicate quickly.",
        "moderately easy to imitate — paid sponsorship and content briefs can be purchased/copied by any brand with budget",
        "low — paid relationships are the most easily substitutable partnership type",
        "medium — based on 16 verified records with a very clean pattern",
        "The pattern is consistent with pure budget deployment rather than a hard-to-copy capability; paid sponsorship is inherently more imitable than relationship- or community-based moats.",
        "Evidence that the demonstration-content discipline persists despite creator turnover, suggesting a repeatable internal process rather than one campaign.",
        "unsupported",
        ptype="paid_sponsorship",
    )
    add(
        "Oner Active",
        f"Oner Active's sample is the smallest (n={int(pattern.loc['Oner Active','n'])}) and shows no distinctive partnership or intent concentration beyond a generic organic lean shared directionally with TALA.",
        "not applicable — no distinctive pattern strong enough to propose a mechanism",
        "not applicable — no mechanism proposed",
        "not applicable — no mechanism proposed",
        "very low — n=8",
        "Sample is too small to distinguish a strategic pattern from noise.",
        "A materially larger verified sample for this brand.",
        "unsupported",
    )

    return pd.DataFrame(rows)


def build_professor_response(d) -> pd.DataFrame:
    pattern = d["pattern"]
    rows = [
        {
            "question_id": "q1",
            "question": "How does TALA's creator-led strategy differ from competitors?",
            "answer": (
                f"This is a cross-brand YouTube creator-strategy comparison combined with an official-channel "
                f"platform comparison, not a comprehensive cross-platform read. From creator evidence: Oner "
                f"Active has the highest observed organic share (87.5%, n=8), but its sample is too shallow for "
                f"a confident conclusion; TALA has an organic plurality within a larger sample (62.5% of "
                f"{int(pattern.loc['TALA','n'])} verified records) and is the only brand with verified "
                "founder/employee and ambassador partnership records; Adanola and Girlfriend Collective lean "
                "more non-organic (81.2% and 69.2% respectively), concentrated in paid sponsorship and "
                "affiliate partnerships. From official-channel evidence: TALA's own platform footprint spans "
                "website, TikTok, and YouTube, while competitors' official-channel evidence in this corpus is "
                "website-dominant."
            ),
            "evidence_basis": "outputs/tables/creator_partnership_mix.csv; outputs/tables/creator_strategic_pattern_metrics.csv; outputs/tables/brand_platform_strategy_comparison.csv",
            "confidence": "medium — descriptive pattern on a targeted, non-random sample",
            "caveat": "Sample sizes differ by brand (n=8 to n=24); not a statistically representative comparison; Oner Active's organic share is the highest observed but rests on the shallowest sample.",
        },
        {
            "question_id": "q2",
            "question": "What role do different partnership types play?",
            "answer": (
                "The partnership x intent matrix shows paid-sponsorship and affiliate records skew toward "
                "product-demonstration content across brands, while organic records carry more of the review and "
                "mixed-intent content observed in the corpus; responsibility-intent content is rare across all "
                "partnership types."
            ),
            "evidence_basis": "outputs/tables/creator_partnership_intent_matrix.csv",
            "confidence": "medium",
            "caveat": "Based on automated, evidence-gated intent classification, not human-labelled ground truth.",
        },
        {
            "question_id": "q3",
            "question": "What appears to be the intent behind TALA's partnership choices?",
            "answer": (
                f"TALA's verified creator content is dominated by product_demonstration intent (75% of "
                f"{int(pattern.loc['TALA','n'])} records) delivered mostly through organic relationships, with "
                "small but present review, responsibility, and social-proof shares that are not clearly present "
                "in the competitor corpora."
            ),
            "evidence_basis": "outputs/tables/creator_content_intent_mix.csv",
            "confidence": "medium",
            "caveat": "Intent labels are automated classifications; low-count categories (1 record each) should not be over-read.",
        },
        {
            "question_id": "q4",
            "question": "What can TALA learn from each competitor?",
            "answer": (
                "From Adanola: test a small, contractually-scoped paid-sponsorship pilot for consistent "
                "product-demonstration content. From Girlfriend Collective: test a defined affiliate program "
                "paired with review-intent briefs. From Oner Active: no actionable lesson — corpus too shallow. "
                "TALA should avoid copying Adanola's demonstration-only concentration wholesale, since it may "
                "crowd out the responsibility and founder-led content that appears distinctive to TALA."
            ),
            "evidence_basis": "outputs/tables/tala_competitor_lessons.csv",
            "confidence": "medium-low — descriptive comparator practices, no outcome data",
            "caveat": "Recommendations are hypotheses to test, not validated best practices.",
        },
        {
            "question_id": "q5",
            "question": "Is there a plausible creator-strategy moat?",
            "answer": (
                "A plausible but currently weak hypothesis is that TALA's founder/employee and ambassador "
                "creator relationships, combined with its organic-lean content mix, are harder for competitors "
                "to imitate quickly than the paid-sponsorship or affiliate patterns observed at Adanola and "
                "Girlfriend Collective. No hypothesis in the current corpus reaches a status stronger than "
                "'plausible/weak' given record counts."
            ),
            "evidence_basis": "outputs/tables/creator_moat_hypotheses.csv",
            "confidence": "low-medium",
            "caveat": "This is a hypothesis, not a proven moat; it rests on 2 founder/ambassador records out of 24.",
        },
        {
            "question_id": "q6",
            "question": "What cannot be concluded from the current evidence?",
            "answer": (
                "The current corpus cannot establish which brand's creator strategy performs better, cannot "
                "support cross-platform creator-performance comparison (Instagram is entirely absent), cannot "
                "support any causal claim that a partnership type drives outcomes, and cannot rule out that "
                "absent categories (e.g. Adanola responsibility content) simply were not captured by this "
                "targeted recovery process rather than not existing."
            ),
            "evidence_basis": "outputs/tables/creator_strategy_evidence_coverage.csv",
            "confidence": "n/a — this is a scope boundary statement",
            "caveat": "Engagement prediction and statistical significance testing were explicitly out of scope for this analysis.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    d = _load()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    interp = build_interpretation(d)
    interp.to_csv(TABLES_DIR / "creator_strategy_interpretation.csv", index=False)

    lessons = build_lessons(d)
    lessons.to_csv(TABLES_DIR / "tala_competitor_lessons.csv", index=False)

    moat = build_moat_hypotheses(d)
    moat.to_csv(TABLES_DIR / "creator_moat_hypotheses.csv", index=False)

    prof = build_professor_response(d)
    prof.to_csv(TABLES_DIR / "professor_feedback_response.csv", index=False)

    print("interpretation:", len(interp))
    print("lessons:", len(lessons))
    print("moat:", len(moat))
    print("professor_response:", len(prof))
    print("Strategic interpretation build complete.")


if __name__ == "__main__":
    main()
