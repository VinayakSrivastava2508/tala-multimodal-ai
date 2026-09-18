"""Day 1 data cleaning, auditing, and EDA output generation.

Input:  data/raw/day1/auto/*.csv
Output: data/interim/day1/*_cleaned.csv
        outputs/tables/day1_*.csv
        outputs/figures/rows_by_*.png

Run: python scripts/clean_day1_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parents[1]
RAW_DIR    = ROOT / "data" / "raw" / "day1" / "auto"
INTERIM    = ROOT / "data" / "interim" / "day1"
TABLES     = ROOT / "outputs" / "tables"
FIGURES    = ROOT / "outputs" / "figures"

for d in (INTERIM, TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

TODAY = pd.Timestamp.today().strftime("%Y-%m-%d")

# ── Constants ─────────────────────────────────────────────────────────────────

BRAND_NORM = {
    "tala": "TALA",
    "wear tala": "TALA",
    "wearetala": "TALA",
    "adanola": "Adanola",
    "girlfriend collective": "Girlfriend Collective",
    "girlfriend-collective": "Girlfriend Collective",
    "oner active": "Oner Active",
    "oneractive": "Oner Active",
}

PLATFORM_NORM = {
    "uk.trustpilot.com": "trustpilot.com",
    "ie.trustpilot.com": "trustpilot.com",
    "ca.trustpilot.com": "trustpilot.com",
    "au.trustpilot.com": "trustpilot.com",
    "us.trustpilot.com": "trustpilot.com",
    "de.trustpilot.com": "trustpilot.com",
    "fr.trustpilot.com": "trustpilot.com",
    "news.google.com":   "google_news",
    "www.reddit.com":    "reddit.com",
    "old.reddit.com":    "reddit.com",
    "www.youtube.com":   "youtube.com",
    "youtu.be":          "youtube.com",
    "www.instagram.com": "instagram.com",
    "www.tiktok.com":    "tiktok.com",
}

EVIDENCE_NORM = {
    "official":            "official",
    "customer_experience": "customer_experience",
    "creator_strategy":    "creator_strategy",
    "brand_positioning":   "brand_positioning",
    "competitor_benchmark":"competitor_benchmark",
    "press":               "press",
    "community":           "community",
    "assessment":          "assessment",
}

# Providers that extracted actual article/page text
TEXT_EXTRACTED_PROVIDERS = {"trafilatura", "apify_trustpilot"}

# Providers that only have titles/snippets
SNIPPET_PROVIDERS = {"duckduckgo", "google_news_rss", "apify_google"}

# Minimum character length for a row to be considered substantive
MIN_TEXT_LEN = 80


# ── Helper functions ──────────────────────────────────────────────────────────

def norm_brand(s: str) -> str:
    """Normalise brand name to canonical form."""
    key = str(s).strip().lower()
    return BRAND_NORM.get(key, str(s).strip())


def norm_platform(s: str) -> str:
    """Normalise source_platform to canonical domain."""
    key = str(s).strip().lower()
    return PLATFORM_NORM.get(key, key)


def norm_evidence(s: str) -> str:
    """Normalise evidence_type to schema-valid value."""
    key = str(s).strip().lower()
    return EVIDENCE_NORM.get(key, key)


def norm_bool(val) -> bool:
    """Coerce any truthy/string value to Python bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, float):
        import math
        return not math.isnan(val) and bool(val)
    return str(val).strip().lower() in ("true", "yes", "1", "t")


def get_text_field(df: pd.DataFrame, schema: str) -> str:
    """Return the primary text column name for a schema."""
    return {
        "official_claims":      "claim_text",
        "customer_reviews":     "review_text",
        "press_reddit_sources": "summary",
        "creator_posts":        "caption",
        "competitor_platforms": "context_text",
    }.get(schema, "")


def audit_row(row: pd.Series, schema: str, text_col: str) -> tuple[bool, list[str], bool, bool, bool, bool]:
    """Compute audit flags for a single row.

    Returns: (needs_manual_review, reasons, usable_analysis, usable_rag,
               usable_creator, usable_quality)
    """
    reasons: list[str] = []
    provider = str(row.get("provider_name", "")).lower()
    source_url = str(row.get("source_url", "")).strip()
    platform = str(row.get("source_platform", "")).lower()
    evidence = str(row.get("evidence_type", "")).lower()

    # --- URL checks ---
    if not source_url or source_url in ("nan", "none", ""):
        reasons.append("missing_source_url")
    elif "google.com/url" in source_url or "news.google.com" in source_url:
        reasons.append("google_redirect_url")

    # --- Text length ---
    text = str(row.get(text_col, "")).strip() if text_col else ""
    if len(text) < MIN_TEXT_LEN:
        reasons.append(f"short_text_{len(text)}chars")

    # --- Snippet-only providers ---
    if provider in SNIPPET_PROVIDERS:
        reasons.append(f"snippet_only_{provider}")

    # --- Google News RSS: title+summary only, no article body ---
    if provider == "google_news_rss":
        reasons.append("press_lead_no_article_body")

    # --- Social platform without verified creator context ---
    if platform in ("instagram.com", "tiktok.com") and provider not in TEXT_EXTRACTED_PROVIDERS:
        reasons.append("social_unverified_creator_context")

    needs_review = len(reasons) > 0

    # --- Usability flags ---
    # usable_for_analysis: has substantive text even if needs review
    usable_analysis = (
        len(text) >= MIN_TEXT_LEN
        and source_url not in ("", "nan", "none")
    )

    # usable_for_rag: same bar + provider either extracted text or has a clean URL
    usable_rag = usable_analysis and (
        provider in TEXT_EXTRACTED_PROVIDERS
        or (provider in SNIPPET_PROVIDERS and "google.com/url" not in source_url)
    )

    # usable_for_creator_strategy
    usable_creator = evidence in ("creator_strategy", "brand_positioning") and usable_analysis

    # usable_for_quality_responsibility
    usable_quality = evidence in ("customer_experience", "official") and usable_analysis

    return needs_review, reasons, usable_analysis, usable_rag, usable_creator, usable_quality


def clean_df(
    df: pd.DataFrame,
    schema: str,
) -> pd.DataFrame:
    """Apply all normalisation and audit steps to a DataFrame."""

    # 1. Normalise core fields
    df["brand"] = df["brand"].apply(norm_brand)
    df["source_platform"] = df["source_platform"].fillna("unknown").apply(norm_platform)
    df["evidence_type"] = df["evidence_type"].fillna("").apply(norm_evidence)
    df["citation_ready"] = df["citation_ready"].apply(norm_bool)
    df["collection_date"] = df["collection_date"].fillna(TODAY)
    df["synthetic"] = df["synthetic"].apply(norm_bool) if "synthetic" in df.columns else False

    # 2. Remove exact duplicates on source_url + text
    text_col = get_text_field(df, schema)
    n_before = len(df)
    if text_col and text_col in df.columns:
        df = df.drop_duplicates(
            subset=["source_url", text_col],
            keep="first",
        ).reset_index(drop=True)
    else:
        df = df.drop_duplicates(subset=["source_url"], keep="first").reset_index(drop=True)
    n_dupes = n_before - len(df)

    # 3. Add audit columns
    audit_results = df.apply(
        lambda row: audit_row(row, schema, text_col), axis=1
    )

    df["needs_manual_review"]          = [r[0] for r in audit_results]
    df["review_reason"]                = ["; ".join(r[1]) for r in audit_results]
    df["usable_for_analysis"]          = [r[2] for r in audit_results]
    df["usable_for_rag"]               = [r[3] for r in audit_results]
    df["usable_for_creator_strategy"]  = [r[4] for r in audit_results]
    df["usable_for_quality_responsibility"] = [r[5] for r in audit_results]

    print(f"  {schema}: {n_before} raw -> {len(df)} cleaned ({n_dupes} dupes removed)")
    print(f"    needs_manual_review: {df['needs_manual_review'].sum()}")
    print(f"    usable_for_analysis: {df['usable_for_analysis'].sum()}")

    return df, n_dupes


# ── Load, clean, save ─────────────────────────────────────────────────────────

FILE_MAP = {
    "official_claims":      ("official_claims_candidates.csv",     "official_claims_cleaned.csv"),
    "customer_reviews":     ("customer_reviews_candidates.csv",    "customer_reviews_cleaned.csv"),
    "press_reddit_sources": ("press_reddit_candidates.csv",        "press_reddit_sources_cleaned.csv"),
    "creator_posts":        ("creator_posts_candidates.csv",       "creator_posts_cleaned.csv"),
    "competitor_platforms": ("competitor_platforms_candidates.csv","competitor_platforms_cleaned.csv"),
}

print("=" * 60)
print("Day 1 Data Cleaning Pipeline")
print("=" * 60)

summary_rows = []
cleaned_dfs: dict[str, pd.DataFrame] = {}

for schema, (raw_name, clean_name) in FILE_MAP.items():
    raw_path = RAW_DIR / raw_name
    if not raw_path.exists():
        print(f"  [SKIP] {raw_name} not found")
        continue

    df_raw = pd.read_csv(raw_path, low_memory=False)
    df_clean, n_dupes = clean_df(df_raw.copy(), schema)

    out_path = INTERIM / clean_name
    df_clean.to_csv(out_path, index=False)
    cleaned_dfs[schema] = df_clean

    summary_rows.append({
        "file":                       clean_name,
        "raw_rows":                   len(df_raw),
        "cleaned_rows":               len(df_clean),
        "duplicate_rows_removed":     n_dupes,
        "rows_missing_source_url":    int((df_clean["source_url"].isna() | (df_clean["source_url"].astype(str).str.strip() == "")).sum()),
        "rows_needing_manual_review": int(df_clean["needs_manual_review"].sum()),
        "rows_usable_for_analysis":   int(df_clean["usable_for_analysis"].sum()),
    })

# ── Combine all for cross-cutting tables ─────────────────────────────────────

all_dfs = []
for schema, df in cleaned_dfs.items():
    df2 = df.copy()
    df2["_schema"] = schema
    all_dfs.append(df2[["brand", "source_platform", "evidence_type",
                          "provider_name", "needs_manual_review",
                          "usable_for_analysis", "_schema"]])

combined = pd.concat(all_dfs, ignore_index=True)

# ── Table 1: Collection summary ───────────────────────────────────────────────

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(TABLES / "day1_collection_summary.csv", index=False)
print()
print("Collection summary saved.")

# ── Table 2: Brand / platform / evidence matrix ───────────────────────────────

matrix = (
    combined.groupby(["brand", "source_platform", "evidence_type"])
    .size()
    .reset_index(name="row_count")
    .sort_values(["brand", "row_count"], ascending=[True, False])
)
matrix.to_csv(TABLES / "day1_brand_platform_matrix.csv", index=False)
print("Brand/platform matrix saved.")

# ── Table 3: Provisional label distribution ───────────────────────────────────

def assign_provisional_label(row: pd.Series) -> str:
    provider = str(row.get("provider_name", "")).lower()
    schema = str(row.get("_schema", ""))
    evidence = str(row.get("evidence_type", ""))
    platform = str(row.get("source_platform", ""))

    if schema == "press_reddit_sources" and provider == "google_news_rss":
        return "press_lead_rss"
    if schema == "customer_reviews" and "trustpilot" in platform:
        return "review_snippet_trustpilot"
    if schema == "customer_reviews":
        return "review_snippet_other"
    if schema == "official_claims":
        return "official_claim_candidate"
    if schema == "creator_posts" and platform == "youtube.com":
        return "creator_content_youtube"
    if schema == "creator_posts":
        return "creator_content_pointer"
    if schema == "competitor_platforms" and provider == "trafilatura":
        return "competitor_page_extracted"
    if schema == "competitor_platforms":
        return "competitor_page_pointer"
    return "other"

combined["provisional_label"] = combined.apply(assign_provisional_label, axis=1)

label_dist = (
    combined["provisional_label"]
    .value_counts()
    .reset_index()
)
label_dist.columns = ["provisional_label", "row_count"]
label_dist["percent"] = (label_dist["row_count"] / label_dist["row_count"].sum() * 100).round(1)
label_dist.to_csv(TABLES / "day1_label_distribution.csv", index=False)
print("Label distribution saved.")

# ── Charts ────────────────────────────────────────────────────────────────────

CHART_STYLE = {
    "figure.figsize": (9, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.family": "DejaVu Sans",
}

def save_bar(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    with plt.rc_context(CHART_STYLE):
        fig, ax = plt.subplots()
        bars = ax.barh(series.index[::-1], series.values[::-1], color="#4C72B0", edgecolor="white")
        ax.set_title(title, pad=12)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(int(w)), va="center", ha="left", fontsize=9)
        plt.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Chart saved: {path.name}")


# Chart 1: rows by brand
rows_by_brand = combined.groupby("brand").size().sort_values()
save_bar(
    rows_by_brand,
    title="Day 1 — Rows by Brand\n(all schemas, Google News RSS + DDG)",
    xlabel="Brand",
    ylabel="Row count",
    path=FIGURES / "rows_by_brand.png",
)

# Chart 2: rows by evidence type
rows_by_evidence = combined.groupby("evidence_type").size().sort_values()
save_bar(
    rows_by_evidence,
    title="Day 1 — Rows by Evidence Type",
    xlabel="Evidence type",
    ylabel="Row count",
    path=FIGURES / "rows_by_evidence_type.png",
)

# Chart 3: rows by source platform (top 12)
rows_by_platform = combined.groupby("source_platform").size().sort_values().tail(12)
save_bar(
    rows_by_platform,
    title="Day 1 — Rows by Source Platform (top 12)",
    xlabel="Source platform",
    ylabel="Row count",
    path=FIGURES / "rows_by_source_platform.png",
)

# ── Final summary print ───────────────────────────────────────────────────────

print()
print("=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)
print(f"{'File':<45} {'Raw':>5} {'Clean':>6} {'Dupes':>6} {'NeedsRev':>9} {'Usable':>7}")
print("-" * 60)
for row in summary_rows:
    print(f"  {row['file']:<43} {row['raw_rows']:>5} {row['cleaned_rows']:>6} "
          f"{row['duplicate_rows_removed']:>6} {row['rows_needing_manual_review']:>9} "
          f"{row['rows_usable_for_analysis']:>7}")
print("-" * 60)
total_raw   = sum(r["raw_rows"] for r in summary_rows)
total_clean = sum(r["cleaned_rows"] for r in summary_rows)
total_dupes = sum(r["duplicate_rows_removed"] for r in summary_rows)
total_rev   = sum(r["rows_needing_manual_review"] for r in summary_rows)
total_use   = sum(r["rows_usable_for_analysis"] for r in summary_rows)
print(f"  {'TOTAL':<43} {total_raw:>5} {total_clean:>6} {total_dupes:>6} {total_rev:>9} {total_use:>7}")
print()

print("Provisional label distribution:")
for _, r in label_dist.iterrows():
    print(f"  {r['provisional_label']:<35} {r['row_count']:>4} rows  ({r['percent']}%)")

print()
print("Files written:")
print(f"  data/interim/day1/  ({len(FILE_MAP)} cleaned CSVs)")
print(f"  outputs/tables/day1_collection_summary.csv")
print(f"  outputs/tables/day1_brand_platform_matrix.csv")
print(f"  outputs/tables/day1_label_distribution.csv")
print(f"  outputs/figures/rows_by_brand.png")
print(f"  outputs/figures/rows_by_evidence_type.png")
print(f"  outputs/figures/rows_by_source_platform.png")
print("=" * 60)
