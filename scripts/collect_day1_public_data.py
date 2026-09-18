"""Day 1 automated public data collection pipeline.

Priority order: GDELT -> Google News RSS -> Official pages -> Reddit JSON
               -> YouTube API -> Apify Trustpilot -> DuckDuckGo -> source queue

Run: python scripts/collect_day1_public_data.py
Outputs: data/raw/day1/auto/*.csv  (7 files)
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load .env before any imports that read env vars
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import yaml

from src.collectors.utils import (
    assign_ids, deduplicate, log_failure, make_source_queue_row, save_candidates,
)
from src.collectors.apify_client import ApifyClient
from src.collectors.official import collect_brand_official_pages, collect_competitor_pages
from src.collectors.gdelt_press import collect_gdelt, collect_google_news, collect_ddg_news
from src.collectors.reddit import collect_reddit_community, collect_reddit_reviews
from src.collectors.reviews import collect_all_reviews
from src.collectors.youtube_creator import collect_youtube_creators
from src.collectors.search import collect_ddg_press, collect_ddg_creator, collect_ddg_competitor, collect_ddg_reviews

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "day1" / "auto"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES_CFG = ROOT / "configs" / "sources.yaml"
with open(SOURCES_CFG, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

BRANDS = CFG["brands"]
FOCAL_BRAND = "tala"
COMPETITORS = ["adanola", "girlfriend_collective", "oner_active"]
ALL_BRANDS = [FOCAL_BRAND] + COMPETITORS

TARGETS = {
    "official_claims":        20,
    "creator_posts":          20,
    "customer_reviews":       40,
    "competitor_platforms":   30,
    "press_reddit_sources":   25,
    "source_queue":            0,  # no minimum — advisory only
}

# ── API detection ──────────────────────────────────────────────────────────────

APIFY_CLIENT = ApifyClient.from_env()
YOUTUBE_KEY = os.getenv("YOUTUBE_API_KEY", "").strip()

print("=" * 60)
print("TALA Day 1 Public Data Collection")
print("=" * 60)
print(f"  Apify token  : {'[OK]' if APIFY_CLIENT else '[SKIP] not configured'}")
print(f"  YouTube API  : {'[OK]' if YOUTUBE_KEY else '[SKIP] not configured'}")
print(f"  Output dir   : {OUT_DIR}")
print()

# ── Accumulators ──────────────────────────────────────────────────────────────

official_rows:     List[Dict] = []
creator_rows:      List[Dict] = []
review_rows:       List[Dict] = []
competitor_rows:   List[Dict] = []
press_rows:        List[Dict] = []
source_queue:      List[Dict] = []
failures:          List[Dict] = []

# ── 1. Official claims ────────────────────────────────────────────────────────

print("-" * 60)
print("Step 1: Official brand pages")
print("-" * 60)

for slug in ALL_BRANDS:
    brand = BRANDS[slug]
    name = brand["name"]
    pages = brand.get("official_pages", [])
    print(f"  {name} ({len(pages)} pages)")

    if slug == FOCAL_BRAND:
        rows = collect_brand_official_pages(slug, name, pages, failures)
        official_rows.extend(rows)
    else:
        rows = collect_competitor_pages(slug, name, pages, failures)
        competitor_rows.extend(rows)

print(f"  -> official: {len(official_rows)} rows, competitor: {len(competitor_rows)} rows")

# ── 1b. Official claims via DuckDuckGo (JS-rendered fallback) ────────────────

print()
print("-" * 60)
print("Step 1b: DDG official claim search (Shopify JS-rendered fallback)")
print("-" * 60)

tala_claim_queries = [
    "TALA activewear sustainability claims site:weartala.com",
    "TALA activewear about community values",
    "TALA weartala.com responsibility ethics",
    "TALA Grace Beverley brand mission statement",
]
print("  TALA official claim DDG search")
ddg_official = collect_ddg_press("TALA", tala_claim_queries, max_per_query=5, failures=failures)
# Convert press rows to official_claims rows for TALA brand claims
from src.collectors.utils import make_claim_row
for pr in ddg_official:
    if "weartala.com" in pr.get("source_url", "") or any(
        k in (pr.get("summary", "") + pr.get("title", "")).lower()
        for k in ("sustainability", "mission", "responsibility", "community", "ethical")
    ):
        claim_row = make_claim_row(
            brand="TALA",
            claim_text=pr.get("summary", "") or pr.get("title", ""),
            source_url=pr.get("source_url", ""),
            claim_category="brand_values",
            source_type="press",
            provider_name="duckduckgo",
            source_method="web_search",
        )
        official_rows.append(claim_row)

print(f"  -> official claims after DDG: {len(official_rows)} rows")

# ── 2. Press: GDELT ───────────────────────────────────────────────────────────

print()
print("-" * 60)
print("Step 2: GDELT press articles")
print("-" * 60)

for slug in ALL_BRANDS:
    brand = BRANDS[slug]
    name = brand["name"]
    queries = brand.get("gdelt_queries", [])
    print(f"  {name}")
    rows = collect_gdelt(name, queries, max_per_query=25, failures=failures)
    press_rows.extend(rows)

print(f"  -> press total so far: {len(press_rows)} rows")

# ── 3. Press: Google News RSS ─────────────────────────────────────────────────

print()
print("-" * 60)
print("Step 3: Google News RSS")
print("-" * 60)

for slug in ALL_BRANDS:
    brand = BRANDS[slug]
    name = brand["name"]
    queries = brand.get("news_queries", [])
    print(f"  {name}")
    rows = collect_google_news(name, queries, max_per_query=20, failures=failures)
    press_rows.extend(rows)

print(f"  -> press total so far: {len(press_rows)} rows")

# ── 4. Reddit community ───────────────────────────────────────────────────────

print()
print("-" * 60)
print("Step 4: Reddit community posts")
print("-" * 60)

global_subreddits = CFG.get("reddit", {}).get("subreddits", [])

for slug in ALL_BRANDS:
    brand = BRANDS[slug]
    name = brand["name"]
    queries = brand.get("reddit_queries", [])
    print(f"  {name}")
    subs = global_subreddits if slug == FOCAL_BRAND else []
    rows = collect_reddit_community(name, queries, subreddits=subs, limit=25, failures=failures)
    press_rows.extend(rows)

print(f"  -> press+community total: {len(press_rows)} rows")

# ── 5. Reddit reviews (customer experience) ───────────────────────────────────

print()
print("-" * 60)
print("Step 5: Reddit review discussions")
print("-" * 60)

tala_review_queries = CFG.get("reddit", {}).get("quality_queries", [])
print(f"  TALA quality queries: {len(tala_review_queries)}")
reddit_review_rows = collect_reddit_reviews(
    "TALA", tala_review_queries, limit=25, failures=failures
)
review_rows.extend(reddit_review_rows)
print(f"  -> review rows: {len(reddit_review_rows)}")

# Also get review discussions for competitors
for slug in COMPETITORS:
    brand = BRANDS[slug]
    name = brand["name"]
    r_queries = [f"{name} quality", f"{name} review", f"{name} sizing"]
    rows = collect_reddit_reviews(name, r_queries, limit=15, failures=failures)
    review_rows.extend(rows)

print(f"  -> review total: {len(review_rows)} rows")

# ── 5b. DDG review snippets ───────────────────────────────────────────────────

print()
print("-" * 60)
print("Step 5b: DuckDuckGo review search")
print("-" * 60)

for slug in ALL_BRANDS:
    brand = BRANDS[slug]
    name = brand["name"]
    queries = [
        f"{name} activewear review",
        f"{name} quality customer experience",
        f"site:trustpilot.com {name}",
    ]
    print(f"  {name}")
    ddg_rev = collect_ddg_reviews(name, queries, max_per_query=6, failures=failures)
    review_rows.extend(ddg_rev)

print(f"  -> review total after DDG: {len(review_rows)} rows")

# ── 6. Apify Trustpilot ───────────────────────────────────────────────────────

print()
print("-" * 60)
print("Step 6: Apify Trustpilot reviews")
print("-" * 60)

if APIFY_CLIENT:
    for slug in ALL_BRANDS:
        brand = BRANDS[slug]
        name = brand["name"]
        tp_url = brand.get("trustpilot_url", "")
        if not tp_url:
            continue
        domain = tp_url.replace("https://www.trustpilot.com/review/", "").strip("/")
        count = CFG.get("apify", {}).get("trustpilot_review_count", 50)
        print(f"  {name}: {domain}")
        # Pass already-collected Reddit reviews only for TALA to avoid double-counting
        rr = reddit_review_rows if slug == FOCAL_BRAND else None
        rows = collect_all_reviews(
            name, domain,
            reddit_rows=rr,
            apify_client=APIFY_CLIENT,
            count=count,
            failures=failures,
        )
        # Extend but don't double-add reddit rows
        tp_only = [r for r in rows if r.get("platform") == "trustpilot"]
        review_rows.extend(tp_only)
        print(f"    -> {len(tp_only)} Trustpilot rows")
else:
    print("  [SKIP] Apify token not configured — Trustpilot collection skipped")
    for slug in ALL_BRANDS:
        brand = BRANDS[slug]
        source_queue.append(make_source_queue_row(
            brand=brand["name"],
            target_schema="customer_reviews",
            source_url=brand.get("trustpilot_url", ""),
            reason="Requires Apify token for automated Trustpilot scraping",
            evidence_type="customer_experience",
        ))

print(f"  -> review total: {len(review_rows)} rows")

# ── 7. YouTube creator content ────────────────────────────────────────────────

print()
print("-" * 60)
print("Step 7: YouTube creator search")
print("-" * 60)

yt_queries = CFG.get("youtube", {}).get("queries", [])

if YOUTUBE_KEY:
    for slug in ALL_BRANDS:
        brand = BRANDS[slug]
        name = brand["name"]
        q = brand.get("creator_queries", [])
        print(f"  {name}")
        rows = collect_youtube_creators(
            name, q, api_key=YOUTUBE_KEY,
            max_per_query=25, failures=failures,
        )
        creator_rows.extend(rows)

    # Global cross-brand queries
    print("  Cross-brand YouTube queries")
    global_yt_rows = collect_youtube_creators(
        "TALA", yt_queries[:4], api_key=YOUTUBE_KEY,
        max_per_query=15, failures=failures,
    )
    creator_rows.extend(global_yt_rows)
else:
    print("  [SKIP] YouTube API key not configured — adding to source queue")
    for slug in ALL_BRANDS:
        brand = BRANDS[slug]
        for q in brand.get("creator_queries", [])[:1]:
            source_queue.append(make_source_queue_row(
                brand=brand["name"],
                target_schema="creator_posts",
                source_url=f"https://www.youtube.com/results?search_query={q.replace(' ','+')}",
                reason="YouTube API key required for automated collection",
                evidence_type="creator_strategy",
            ))

print(f"  -> creator total: {len(creator_rows)} rows")

# ── 8. DuckDuckGo supplementary ───────────────────────────────────────────────

print()
print("-" * 60)
print("Step 8: DuckDuckGo supplementary search")
print("-" * 60)

# Press fallback (DDG news already covered in Step 2-3 via GDELT/GNews)
# Use DDG text search only to top-up if press rows are low
if len(press_rows) < TARGETS["press_reddit_sources"]:
    print("  DDG press top-up (below target)")
    for slug in [FOCAL_BRAND]:
        brand = BRANDS[slug]
        name = brand["name"]
        ddg_p = collect_ddg_press(name, brand.get("news_queries", [])[:3],
                                  max_per_query=6, failures=failures)
        press_rows.extend(ddg_p)

# Creator supplements if YouTube is skipped
if not YOUTUBE_KEY:
    print("  DDG creator fallback (no YouTube key)")
    for slug in ALL_BRANDS:
        brand = BRANDS[slug]
        name = brand["name"]
        ddg_c = collect_ddg_creator(name, brand.get("creator_queries", [])[:2],
                                    max_per_query=6, failures=failures)
        creator_rows.extend(ddg_c)

# Competitor DDG supplement
print("  DDG competitor search")
for slug in COMPETITORS:
    brand = BRANDS[slug]
    name = brand["name"]
    ddg_comp = collect_ddg_competitor(
        name,
        [f"{name} activewear", f"{name} sustainability"],
        max_per_query=5, failures=failures,
    )
    competitor_rows.extend(ddg_comp)

print(f"  -> press: {len(press_rows)}, creator: {len(creator_rows)}, competitor: {len(competitor_rows)}")

# ── 9. Apify Google Search ───────────────────────────────────────────────────

print()
print("-" * 60)
print("Step 9: Apify Google Search (creator + press)")
print("-" * 60)

if APIFY_CLIENT:
    from src.collectors.apify_client import run_google_search_scraper
    from src.collectors.utils import make_press_row, platform_from_url

    google_queries = CFG.get("apify", {}).get("google_queries", [])
    results = run_google_search_scraper(APIFY_CLIENT, google_queries,
                                       results_per_query=10)
    for item in (results or []):
        # Each item may have organicResults list
        organic = item.get("organicResults", []) if isinstance(item, dict) else []
        for r in organic:
            url = r.get("url", "")
            if not url:
                continue
            title = r.get("title", "")
            snippet = r.get("description", "") or r.get("snippet", "")
            ev_type = "creator_strategy" if any(
                k in title.lower() for k in ("haul", "review", "try on", "ambassador")
            ) else "press"
            if ev_type == "creator_strategy":
                from src.collectors.utils import make_creator_row
                row = make_creator_row(
                    brand="TALA",
                    source_url=url,
                    caption=snippet,
                    platform=platform_from_url(url),
                    video_title=title,
                    provider_name="apify_google",
                    source_method="actor",
                )
                creator_rows.append(row)
            else:
                row = make_press_row(
                    brand="TALA",
                    title=title,
                    source_url=url,
                    publication=platform_from_url(url),
                    evidence_type="press",
                    summary=snippet,
                    provider_name="apify_google",
                    source_method="actor",
                )
                press_rows.append(row)
    print(f"  -> processed Apify Google results")
else:
    print("  [SKIP] Apify not configured")

# ── 10. Deduplicate and assign IDs ────────────────────────────────────────────

print()
print("-" * 60)
print("Step 10: Deduplicating and assigning IDs")
print("-" * 60)

official_rows   = assign_ids(deduplicate(official_rows,   "claim_text"), "OC")
creator_rows    = assign_ids(deduplicate(creator_rows,    "video_title"), "CP")
review_rows     = assign_ids(deduplicate(review_rows,     "review_text"), "CR")
competitor_rows = assign_ids(deduplicate(competitor_rows, "context_text"), "COMP")
press_rows      = assign_ids(deduplicate(press_rows,      "title"), "PR")
source_queue    = assign_ids(deduplicate(source_queue,    "source_url"), "SQ")

# ── 11. Save CSVs ────────────────────────────────────────────────────────────

print()
print("-" * 60)
print("Step 11: Saving CSVs")
print("-" * 60)

output_map = {
    "official_claims_candidates.csv":      official_rows,
    "creator_posts_candidates.csv":        creator_rows,
    "customer_reviews_candidates.csv":     review_rows,
    "competitor_platforms_candidates.csv": competitor_rows,
    "press_reddit_candidates.csv":         press_rows,
    "source_queue_for_manual_review.csv":  source_queue,
}

saved_counts = {}
for filename, rows in output_map.items():
    path = OUT_DIR / filename
    n = save_candidates(rows, path)
    saved_counts[filename] = n
    print(f"  {filename}: {n} rows")

# Save failures log
failures_path = OUT_DIR / "collection_failures.csv"
if failures:
    fields = list(failures[0].keys())
    with open(failures_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)
print(f"  collection_failures.csv: {len(failures)} entries")

# ── 12. Summary report ────────────────────────────────────────────────────────

schema_counts = {
    "official_claims":       saved_counts.get("official_claims_candidates.csv", 0),
    "creator_posts":         saved_counts.get("creator_posts_candidates.csv", 0),
    "customer_reviews":      saved_counts.get("customer_reviews_candidates.csv", 0),
    "competitor_platforms":  saved_counts.get("competitor_platforms_candidates.csv", 0),
    "press_reddit_sources":  saved_counts.get("press_reddit_candidates.csv", 0),
    "source_queue":          saved_counts.get("source_queue_for_manual_review.csv", 0),
}

print()
print("=" * 60)
print("COLLECTION SUMMARY")
print("=" * 60)
print(f"{'Schema':<30} {'Got':>6} {'Target':>8} {'Status':>8}")
print("-" * 60)

all_targets_met = True
for schema, target in TARGETS.items():
    got = schema_counts.get(schema, 0)
    if target == 0:
        status = "advisory"
    elif got >= target:
        status = "[OK]"
    else:
        status = "!!"
        all_targets_met = False
    print(f"  {schema:<28} {got:>6} {target:>8} {status:>8}")

print("-" * 60)
print(f"  Failures logged: {len(failures)}")
print(f"  Apify used:      {'yes' if APIFY_CLIENT else 'no'}")
print(f"  YouTube used:    {'yes' if YOUTUBE_KEY else 'no'}")
print()

if all_targets_met:
    print("[OK] All minimum row targets met.")
else:
    print("!! Some targets not met — review failures and source queue.")
    print("   Run: python scripts/validate_data.py")

print()
print("Next steps for Day 1 EDA:")
print("  1. python scripts/validate_data.py")
print("  2. Open notebooks/01_data_compilation_and_eda.ipynb")
print("  3. Review source_queue_for_manual_review.csv for manual gaps")
print("  4. Spot-check 5+ rows per schema for quality before analysis")
print("=" * 60)
