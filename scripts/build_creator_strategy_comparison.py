"""Day 3C Part B: creator-strategy and platform comparison across TALA and competitors.

Descriptive only. Uses only verified creator_multimodal_features.csv and
platform_strategy_features.csv. No engagement prediction, no significance
tests, no cross-platform creator-performance ranking.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

BRANDS = ["TALA", "Adanola", "Girlfriend Collective", "Oner Active"]

PARTNERSHIP_TYPES = [
    "organic", "affiliate", "paid_sponsorship", "gifted",
    "ambassador", "founder_or_employee", "unclear",
]

INTENT_TYPES = [
    "product_demonstration", "review", "responsibility",
    "social_proof", "mixed", "other", "unclear",
]


def _load():
    return {
        "creators": pd.read_csv(PROCESSED_DIR / "creator_multimodal_features.csv"),
        "platform": pd.read_csv(PROCESSED_DIR / "platform_strategy_features.csv"),
    }


def build_evidence_coverage(creators: pd.DataFrame, platform: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for brand in BRANDS:
        c = creators[creators["brand"] == brand]
        p = platform[platform["brand"] == brand]
        rows.append(
            {
                "brand": brand,
                "verified_creator_records": len(c),
                "unique_creators_or_channels": c["channel_id"].nunique(),
                "platforms_represented_creator": ";".join(sorted(c["platform"].unique())),
                "official_platform_records": len(p),
                "platforms_represented_official": ";".join(sorted(p["platform"].unique())) if len(p) else "",
                "records_with_verified_partnership_type": int((c["revised_partnership_type"] != "unclear").sum()),
                "records_with_classified_intent": int(c["primary_intent"].notna().sum()),
                "records_with_usable_media": int((c["media_retrieval_status"] == "success").sum()),
                "records_with_engagement_metrics": int(c["view_count"].notna().sum()),
                "coverage_limitation": (
                    "Creator sample overwhelmingly YouTube across the whole verified corpus (60/61 records); "
                    f"this brand contributes {int((c['platform']!='youtube').sum())} non-YouTube record(s) "
                    f"({', '.join(sorted(c.loc[c['platform']!='youtube','platform'].unique())) or 'none'}); "
                    "no Instagram creator records verified for any brand."
                ),
                "analysis_permitted": "within-platform (YouTube) creator strategy comparison; official cross-platform comparison using platform_strategy_features",
                "analysis_prohibited": "cross-platform creator-performance comparison; engagement prediction; causal claims",
            }
        )
    return pd.DataFrame(rows)


def build_partnership_mix(creators: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for brand in BRANDS:
        c = creators[creators["brand"] == brand]
        n = len(c)
        for ptype in PARTNERSHIP_TYPES:
            count = int((c["revised_partnership_type"] == ptype).sum())
            rows.append(
                {
                    "brand": brand,
                    "partnership_type": ptype,
                    "count": count,
                    "pct_within_brand": round(100 * count / n, 1) if n else 0.0,
                    "brand_denominator_n": n,
                }
            )
    return pd.DataFrame(rows)


def build_intent_mix(creators: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for brand in BRANDS:
        c = creators[creators["brand"] == brand]
        n = len(c)
        for intent in INTENT_TYPES:
            count = int((c["primary_intent"] == intent).sum())
            rows.append(
                {
                    "brand": brand,
                    "content_intent": intent,
                    "count": count,
                    "pct_within_brand": round(100 * count / n, 1) if n else 0.0,
                    "brand_denominator_n": n,
                }
            )
    return pd.DataFrame(rows)


def build_partnership_intent_matrix(creators: pd.DataFrame) -> pd.DataFrame:
    frames = []

    def matrix_for(df: pd.DataFrame, scope: str, brand: str | None = None) -> pd.DataFrame:
        n_total = len(df)
        out_rows = []
        for ptype in PARTNERSHIP_TYPES:
            sub = df[df["revised_partnership_type"] == ptype]
            n_p = len(sub)
            for intent in INTENT_TYPES:
                count = int((sub["primary_intent"] == intent).sum())
                out_rows.append(
                    {
                        "scope": scope,
                        "brand": brand or "all",
                        "partnership_type": ptype,
                        "content_intent": intent,
                        "count": count,
                        "pct_within_partnership_type": round(100 * count / n_p, 1) if n_p else 0.0,
                        "partnership_type_n": n_p,
                        "scope_total_n": n_total,
                    }
                )
        return pd.DataFrame(out_rows)

    frames.append(matrix_for(creators, "overall"))
    for brand in BRANDS:
        frames.append(matrix_for(creators[creators["brand"] == brand], "by_brand", brand))

    non_organic = creators[creators["revised_partnership_type"] != "organic"]
    frames.append(matrix_for(non_organic, "non_organic_only"))
    for brand in BRANDS:
        frames.append(matrix_for(non_organic[non_organic["brand"] == brand], "non_organic_by_brand", brand))

    return pd.concat(frames, ignore_index=True)


def build_platform_role_comparison(platform: pd.DataFrame, creators: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for brand in BRANDS:
        p = platform[platform["brand"] == brand]
        c = creators[creators["brand"] == brand]
        seen_platforms = set(p["platform"].unique()) | set(c["platform"].unique())
        for plat in sorted(seen_platforms) or ["none_verified"]:
            pp = p[p["platform"] == plat]
            cc = c[c["platform"] == plat]
            evidence_count = len(pp) + len(cc)
            roles = pp["channel_role"].value_counts().to_dict() if len(pp) else {}
            responsibility_yes = int((pp["responsibility_messaging"] == "yes").sum())
            commerce_yes = int((pp["commerce_integration"] == "yes").sum())
            creator_led = int((pp["creator_usage"] == "confirmed_creator_program").sum())
            rows.append(
                {
                    "brand": brand,
                    "platform": plat,
                    "platform_present": True,
                    "observed_content_role": ";".join(f"{k}:{v}" for k, v in roles.items()) if roles else "no official_platform_features records",
                    "creator_led_vs_brand_led": (
                        f"creator_led_confirmed:{creator_led}/{len(pp)}" if len(pp) else f"creator_records_observed:{len(cc)}"
                    ),
                    "product_demonstration_role": int((cc["primary_intent"] == "product_demonstration").sum()) if len(cc) else 0,
                    "responsibility_communication_role": responsibility_yes,
                    "community_social_proof_role": int((cc["primary_intent"] == "social_proof").sum()) if len(cc) else 0,
                    "commerce_or_affiliate_role": commerce_yes,
                    "evidence_count": evidence_count,
                    "evidence_limitation": (
                        "Creator-record evidence for this platform" if len(cc) else "Official-platform snapshot evidence only, no verified creator records for this platform/brand"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _hhi(counts: pd.Series) -> float:
    n = counts.sum()
    if n == 0:
        return 0.0
    shares = counts / n
    return round(float((shares ** 2).sum()), 4)


def build_strategic_pattern_metrics(creators: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for brand in BRANDS:
        c = creators[creators["brand"] == brand]
        n = len(c)
        if n == 0:
            continue
        partnership_counts = c["revised_partnership_type"].value_counts()
        intent_counts = c["primary_intent"].value_counts()
        platform_counts = c["platform"].value_counts()

        organic_share = round(100 * (c["revised_partnership_type"] == "organic").sum() / n, 1)
        non_organic_share = round(100 - organic_share, 1)
        affiliate_share = round(100 * (c["revised_partnership_type"] == "affiliate").sum() / n, 1)
        product_demo_share = round(100 * (c["primary_intent"] == "product_demonstration").sum() / n, 1)
        review_share = round(100 * (c["primary_intent"] == "review").sum() / n, 1)
        responsibility_share = round(100 * (c["primary_intent"] == "responsibility").sum() / n, 1)
        founder_share = round(100 * (c["revised_partnership_type"] == "founder_or_employee").sum() / n, 1)

        rows.append(
            {
                "brand": brand,
                "n": n,
                "partnership_type_concentration_hhi": _hhi(partnership_counts),
                "intent_concentration_hhi": _hhi(intent_counts),
                "platform_concentration_hhi": _hhi(platform_counts),
                "organic_share_pct": organic_share,
                "non_organic_share_pct": non_organic_share,
                "affiliate_share_pct": affiliate_share,
                "product_demonstration_share_pct": product_demo_share,
                "review_share_pct": review_share,
                "responsibility_content_share_pct": responsibility_share,
                "founder_or_employee_share_pct": founder_share,
                "hhi_definition": (
                    "Herfindahl-Hirschman Index = sum of squared within-brand shares (0-1 scale); "
                    "higher = more concentrated in fewer categories. Denominator = brand's n creator records."
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    d = _load()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    coverage = build_evidence_coverage(d["creators"], d["platform"])
    coverage.to_csv(TABLES_DIR / "creator_strategy_evidence_coverage.csv", index=False)

    partnership_mix = build_partnership_mix(d["creators"])
    partnership_mix.to_csv(TABLES_DIR / "creator_partnership_mix.csv", index=False)

    intent_mix = build_intent_mix(d["creators"])
    intent_mix.to_csv(TABLES_DIR / "creator_content_intent_mix.csv", index=False)

    matrix = build_partnership_intent_matrix(d["creators"])
    matrix.to_csv(TABLES_DIR / "creator_partnership_intent_matrix.csv", index=False)

    platform_role = build_platform_role_comparison(d["platform"], d["creators"])
    platform_role.to_csv(TABLES_DIR / "brand_platform_strategy_comparison.csv", index=False)

    pattern_metrics = build_strategic_pattern_metrics(d["creators"])
    pattern_metrics.to_csv(TABLES_DIR / "creator_strategic_pattern_metrics.csv", index=False)

    print("coverage:", len(coverage))
    print("partnership_mix:", len(partnership_mix))
    print("intent_mix:", len(intent_mix))
    print("matrix:", len(matrix))
    print("platform_role:", len(platform_role))
    print("pattern_metrics:", len(pattern_metrics))
    print("Part B core tables build complete.")


if __name__ == "__main__":
    main()
