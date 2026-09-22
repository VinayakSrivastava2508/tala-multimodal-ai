"""Day 3C: executive-quality figures for the analytical synthesis and
creator-strategy comparison. Reads only already-built outputs/tables CSVs.
No 3D charts, no truncated labels, denominators/n shown, consistent palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
PALETTE = sns.color_palette("Set2")
BRAND_ORDER = ["TALA", "Adanola", "Girlfriend Collective", "Oner Active"]
BRAND_COLORS = dict(zip(BRAND_ORDER, sns.color_palette("Set2", 4)))


def save(fig, name):
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("[fig]", path)


def fig_claim_label_distribution():
    df = pd.read_csv(TABLES_DIR / "claim_label_summary.csv")
    sub = df[df["section"] == "overall_label_distribution"].copy()
    sub = sub.sort_values("n_claims", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(sub["automated_label"], sub["n_claims"], color=PALETTE[: len(sub)])
    for b, n, pct in zip(bars, sub["n_claims"], sub["pct_of_total"]):
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, f"n={n} ({pct}%)", va="center", fontsize=10)
    ax.set_xlabel("Number of TALA claims (of 37 total)")
    ax.set_title("Claim-label distribution — 37 TALA official claims\nSource: analytical_synthesis_claim_master.csv (Day 3A fusion labels)")
    ax.set_xlim(0, sub["n_claims"].max() * 1.35)
    save(fig, "claim_label_distribution.png")


def fig_claim_category_heatmap():
    df = pd.read_csv(TABLES_DIR / "claim_label_summary.csv")
    sub = df[df["section"] == "by_claim_category"].copy()
    sub = sub.rename(columns={"claim_category": "cat"})
    pivot = sub.pivot(index="cat", columns="automated_label", values="n_claims").fillna(0)
    order = ["aligned", "partially_aligned", "mixed", "divergent", "insufficient_evidence"]
    pivot = pivot[[c for c in order if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "n claims"}, ax=ax, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Fusion label")
    ax.set_ylabel("Claim category")
    ax.set_title("Claim label by category (counts, zero cells retained)\nSource: claim_label_summary.csv")
    save(fig, "claim_category_label_heatmap.png")


def fig_claim_modality_coverage():
    df = pd.read_csv(TABLES_DIR / "analytical_synthesis_claim_master.csv")
    cols = ["text_evidence_count", "image_evidence_count", "video_evidence_count", "reference_evidence_count"]
    covered = {c.replace("_evidence_count", ""): int((df[c] > 0).sum()) for c in cols}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(covered.keys())
    vals = list(covered.values())
    bars = ax.bar(names, vals, color=PALETTE[: len(names)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v}/37", ha="center", fontsize=10)
    ax.set_ylabel("Claims with >=1 evidence unit in this modality (of 37)")
    ax.set_title("Claim-level modality coverage\nSource: analytical_synthesis_claim_master.csv")
    ax.set_ylim(0, 40)
    save(fig, "claim_modality_coverage.png")


def fig_modality_confidence_contribution():
    df = pd.read_csv(TABLES_DIR / "modality_contribution_summary.csv")
    sub = df[df["section"] == "configuration_comparison"].copy()
    order = ["text_only", "text_image", "text_image_video", "text_image_video_reference"]
    sub["configuration"] = pd.Categorical(sub["configuration"], categories=order, ordered=True)
    sub = sub.sort_values("configuration")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(sub["configuration"].astype(str), sub["mean_confidence"], color=PALETTE[2])
    for b, v in zip(bars, sub["mean_confidence"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("Mean fusion confidence (n=37 claims each configuration)")
    ax.set_title("Mean confidence by fusion configuration — labels unchanged across all 4\nSource: modality_contribution_summary.csv (fusion_configuration_comparison)")
    ax.set_ylim(0.5, 0.6)
    save(fig, "modality_confidence_contribution.png")


def fig_evidence_gap_by_category():
    df = pd.read_csv(TABLES_DIR / "evidence_gap_by_category.csv")
    df = df.sort_values("total_claims", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(df))
    ax.barh(y, df["total_claims"], color="#d9d9d9", label="total claims")
    ax.barh(y, df["claims_with_independent_evidence"], color=PALETTE[0], label="with independent evidence")
    ax.set_yticks(y)
    ax.set_yticklabels(df["claim_category"])
    for yi, tot, ind in zip(y, df["total_claims"], df["claims_with_independent_evidence"]):
        ax.text(tot + 0.15, yi, f"{ind}/{tot}", va="center", fontsize=9)
    ax.set_xlabel("Number of claims")
    ax.set_title("Independent-evidence coverage by claim category\nSource: evidence_gap_by_category.csv")
    ax.legend(loc="lower right")
    save(fig, "evidence_gap_by_category.png")


def fig_creator_partnership_mix_by_brand():
    df = pd.read_csv(TABLES_DIR / "creator_partnership_mix.csv")
    pivot = df.pivot(index="brand", columns="partnership_type", values="pct_within_brand").reindex(BRAND_ORDER)
    n_map = df.drop_duplicates("brand").set_index("brand")["brand_denominator_n"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_ylabel("% of brand's verified creator records")
    labels = [f"{b}\n(n={n_map[b]})" for b in pivot.index]
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title("Creator partnership mix by brand — descriptive, targeted sample\nSource: creator_partnership_mix.csv")
    ax.legend(title="Partnership type", bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, "creator_partnership_mix_by_brand.png")


def fig_creator_intent_mix_by_brand():
    df = pd.read_csv(TABLES_DIR / "creator_content_intent_mix.csv")
    pivot = df.pivot(index="brand", columns="content_intent", values="pct_within_brand").reindex(BRAND_ORDER)
    n_map = df.drop_duplicates("brand").set_index("brand")["brand_denominator_n"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set3")
    ax.set_ylabel("% of brand's verified creator records")
    labels = [f"{b}\n(n={n_map[b]})" for b in pivot.index]
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title("Creator content-intent mix by brand — descriptive, targeted sample\nSource: creator_content_intent_mix.csv")
    ax.legend(title="Content intent", bbox_to_anchor=(1.02, 1), loc="upper left")
    save(fig, "creator_intent_mix_by_brand.png")


def fig_nonorganic_partnership_intent_matrix():
    df = pd.read_csv(TABLES_DIR / "creator_partnership_intent_matrix.csv")
    sub = df[df["scope"] == "non_organic_only"]
    pivot = sub.pivot(index="partnership_type", columns="content_intent", values="count").fillna(0)
    pivot = pivot.loc[(pivot.sum(axis=1) > 0), :]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "n records"}, ax=ax, linewidths=0.5, linecolor="white")
    total_n = int(sub["scope_total_n"].iloc[0]) if len(sub) else 0
    ax.set_title(f"Non-organic partnership x intent (all brands combined, n={total_n})\nSource: creator_partnership_intent_matrix.csv — descriptive, targeted sample")
    ax.set_xlabel("Content intent")
    ax.set_ylabel("Partnership type")
    save(fig, "nonorganic_partnership_intent_matrix.png")


def fig_brand_platform_strategy_matrix():
    df = pd.read_csv(TABLES_DIR / "brand_platform_strategy_comparison.csv")
    pivot = df.pivot_table(index="brand", columns="platform", values="evidence_count", aggfunc="sum").reindex(BRAND_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Purples", cbar_kws={"label": "evidence records"}, ax=ax, linewidths=0.5, linecolor="white")
    ax.set_title("Platform-role evidence count by brand — descriptive, targeted sample\nSource: brand_platform_strategy_comparison.csv")
    ax.set_xlabel("Platform")
    ax.set_ylabel("Brand")
    save(fig, "brand_platform_strategy_matrix.png")


def fig_competitor_strategy_heatmap():
    # Replaces the radar chart: descriptive measures are on different scales /
    # denominators (HHI, %, n), so a radar would be misleading. Heatmap instead.
    df = pd.read_csv(TABLES_DIR / "creator_strategic_pattern_metrics.csv").set_index("brand").reindex(BRAND_ORDER)
    cols = [
        "organic_share_pct", "non_organic_share_pct", "affiliate_share_pct",
        "product_demonstration_share_pct", "review_share_pct",
        "responsibility_content_share_pct", "founder_or_employee_share_pct",
    ]
    sub = df[cols]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(sub, annot=True, fmt=".1f", cmap="RdYlGn", cbar_kws={"label": "% within brand"}, ax=ax, linewidths=0.5, linecolor="white")
    n_labels = [f"{b} (n={int(df.loc[b,'n'])})" for b in sub.index]
    ax.set_yticklabels(n_labels, rotation=0)
    ax.set_title("Descriptive creator-strategy measures by brand — targeted sample, not comparable ranking\nSource: creator_strategic_pattern_metrics.csv")
    ax.set_xlabel("Descriptive measure (%)")
    save(fig, "tala_competitor_strategy_heatmap.png")


def main():
    fig_claim_label_distribution()
    fig_claim_category_heatmap()
    fig_claim_modality_coverage()
    fig_modality_confidence_contribution()
    fig_evidence_gap_by_category()
    fig_creator_partnership_mix_by_brand()
    fig_creator_intent_mix_by_brand()
    fig_nonorganic_partnership_intent_matrix()
    fig_brand_platform_strategy_matrix()
    fig_competitor_strategy_heatmap()
    print("All 10 synthesis figures written.")


if __name__ == "__main__":
    main()
