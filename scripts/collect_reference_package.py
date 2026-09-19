"""Day 2.6A Part B/C: build the Multimodal Reference Package manifest.

Discovers and fetches reference documents for TALA (Priority 1: official
domains; Priority 2: authoritative third-party registries), and, for
strategic benchmarking only, a lighter-weight pass over the three competitor
brands (Priority 3). Writes
data/interim/day2_6/multimodal_reference_assets.csv and
outputs/tables/day2_6_tala_requirement_status.csv (the 13-item TALA coverage
checklist from the charter).

Usage: python scripts/collect_reference_package.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.api_clients import ddg_text_search  # noqa: E402
from src.collectors.reference_documents import (  # noqa: E402
    AUTHORITATIVE_REGISTRY_DOMAINS,
    COMPETITOR_DOMAINS,
    TALA_DOMAINS,
    ReferenceCandidate,
    candidates_from_hydrated_sources,
    dedupe_candidates,
    discover_common_slugs,
    discover_sitemap,
    discover_via_ddgs_then_verify,
    fetch_reference_document,
)

HYDRATED_SOURCES_PATH = PROJECT_ROOT / "data" / "interim" / "day2" / "hydrated_sources.csv"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "day2_6"
OUT_PATH = OUT_DIR / "multimodal_reference_assets.csv"
MEDIA_DIR = PROJECT_ROOT / "data" / "media" / "reference_documents"
TABLES = PROJECT_ROOT / "outputs" / "tables"

TALA_BASE_URLS = ["https://www.wearetala.com", "https://eu.tala.co.uk", "https://talaactivewear.com"]

# 13-item required-coverage checklist (Part C). Each maps to the reference_type(s)
# that would satisfy it, plus a short DDGS fallback query fragment.
TALA_REQUIREMENTS = [
    ("Sustainability or responsibility policy", ["sustainability_page"], "sustainability responsibility"),
    ("Material/fabric composition", ["material_specification"], "fabric material composition"),
    ("Product specifications", ["product_specification"], "product specification technical details"),
    ("Size chart or measurement guidance", ["size_guide"], "size guide size chart measurements"),
    ("Fit guidance", ["size_guide", "product_specification"], "fit guide how it fits"),
    ("Garment-care or wash guidance", ["care_guide"], "care guide washing instructions"),
    ("Return/refund policy", ["return_policy"], "refund policy return policy"),
    ("Quality/warranty policy, if one exists", ["quality_or_warranty_policy"], "warranty guarantee quality promise"),
    ("Supplier or manufacturing disclosure", ["supplier_disclosure"], "factory supplier manufacturing"),
    ("Packaging policy", ["packaging_policy"], "packaging policy"),
    ("Certification evidence or explicit certification status", ["certification"], "certification bluesign oeko-tex b corp"),
    ("Emissions or environmental-impact disclosure", ["emissions_disclosure"], "carbon emissions environmental impact"),
    ("Public brand/visual guidance, if available", ["brand_style_reference"], "lookbook style guide brand guidelines"),
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    hydrated = pd.read_csv(HYDRATED_SOURCES_PATH) if HYDRATED_SOURCES_PATH.exists() else pd.DataFrame()

    all_rows = []
    seen_hashes: set[str] = set()
    ref_counters: dict[str, int] = {}

    def next_id(brand: str) -> str:
        ref_counters[brand] = ref_counters.get(brand, 0) + 1
        slug = brand.lower().replace(" ", "_")
        return f"ref_{slug}_{ref_counters[brand]:03d}"

    def fetch_and_record(candidate: ReferenceCandidate) -> dict:
        rid = next_id(candidate.brand)
        result = fetch_reference_document(candidate, rid, get=requests.get, media_dir=MEDIA_DIR)
        row = result.to_dict()
        if row["file_hash"] and row["file_hash"] in seen_hashes:
            row["reference_status"] = "duplicate"
            row["processing_status"] = "skipped"
        elif row["file_hash"]:
            seen_hashes.add(row["file_hash"])
        all_rows.append(row)
        return row

    # ── Priority 1: TALA official domains -- reuse existing hydrated sources first ──
    print("[TALA] Priority 1: reusing already-hydrated official/registry sources ...")
    tala_seed_candidates = candidates_from_hydrated_sources(hydrated, "TALA")
    tala_seed_candidates = dedupe_candidates(tala_seed_candidates)
    print(f"  {len(tala_seed_candidates)} seed candidate(s) from Day 2 hydration")
    for c in tala_seed_candidates:
        fetch_and_record(c)

    # ── Priority 1/2 continued: probe common slugs + sitemap on TALA's own domains ──
    print("\n[TALA] Priority 1/2: probing common policy/page slugs + sitemap ...")
    tala_fresh_candidates: list[ReferenceCandidate] = []
    for base_url in TALA_BASE_URLS:
        tala_fresh_candidates.extend(discover_common_slugs(base_url, "TALA"))
        tala_fresh_candidates.extend(discover_sitemap(base_url, "TALA"))
    tala_fresh_candidates = dedupe_candidates(tala_fresh_candidates)
    existing_urls = {r["source_url"].split("?")[0].rstrip("/") for r in all_rows}
    tala_fresh_candidates = [c for c in tala_fresh_candidates if c.source_url.split("?")[0].rstrip("/") not in existing_urls]
    print(f"  {len(tala_fresh_candidates)} new candidate(s) discovered")
    for c in tala_fresh_candidates:
        fetch_and_record(c)

    # ── TALA requirement checklist: DDGS fallback for anything still unmet ─────────
    print("\n[TALA] Requirement checklist -- DDGS fallback for unmet items ...")
    requirement_status_rows = []
    for label, satisfying_types, query_fragment in TALA_REQUIREMENTS:
        satisfied = [
            r for r in all_rows
            if r["brand"] == "TALA" and r["reference_type"] in satisfying_types
            and r["reference_status"] in ("collected", "already_available")
            and r["evidence_strength"] in ("strong", "medium")
        ]
        checked_pages = [r["source_url"] for r in all_rows if r["brand"] == "TALA"]
        if satisfied:
            requirement_status_rows.append({
                "requirement": label, "status": "found", "reference_ids": ";".join(s["reference_id"] for s in satisfied),
                "reference_type": satisfying_types[0], "pages_checked": len(checked_pages),
                "discovery_method": satisfied[0]["discovery_method"], "notes": "",
            })
            continue

        domains = TALA_DOMAINS if satisfying_types[0] != "certification" else list(AUTHORITATIVE_REGISTRY_DOMAINS.keys())
        candidate = discover_via_ddgs_then_verify("TALA", query_fragment, domains, ddg_text_search, get=requests.get)
        if candidate is not None:
            row = fetch_and_record(candidate)
            # The DDGS-discovered page must ALSO classify into one of this
            # requirement's satisfying_types -- discovery via a query hint does
            # not guarantee the landed page is actually about that topic (e.g. a
            # "size guide" query can resolve to an unrelated page). Accepting it
            # regardless of classified reference_type would over-claim coverage.
            type_matches = row["reference_type"] in satisfying_types
            if (row["reference_status"] in ("collected", "already_available")
                    and row["evidence_strength"] in ("strong", "medium") and type_matches):
                requirement_status_rows.append({
                    "requirement": label, "status": "found", "reference_ids": row["reference_id"],
                    "reference_type": row["reference_type"], "pages_checked": len(checked_pages) + 1,
                    "discovery_method": "ddgs_discovery_then_direct_fetch", "notes": "",
                })
                continue
            elif row["reference_status"] in ("collected", "already_available") and not type_matches:
                print(f"    [{label}] DDGS landed on {row['reference_id']} but it classified as "
                      f"'{row['reference_type']}', not {satisfying_types} -- not counted as found "
                      f"(kept in manifest as its own genuine document)")

        requirement_status_rows.append({
            "requirement": label, "status": "not_found", "reference_ids": "",
            "reference_type": satisfying_types[0], "pages_checked": len(checked_pages),
            "discovery_method": "", "notes": "no verified official/authoritative page located -- not fabricated",
        })

    # ── Priority 3: competitor brands, secondary comparison only (lighter probe) ──
    print("\n[Competitors] Priority 3: secondary-comparison probe (common slugs only) ...")
    for brand, domains in COMPETITOR_DOMAINS.items():
        base_url = f"https://{domains[0]}"
        candidates = discover_common_slugs(base_url, brand)
        candidates = dedupe_candidates(candidates)
        print(f"  [{brand}] {len(candidates)} candidate(s) discovered")
        for c in candidates:
            fetch_and_record(c)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH.relative_to(PROJECT_ROOT)} ({len(df)} rows)")
    if not df.empty:
        print(df.groupby(["brand", "reference_status"]).size().to_string())

    req_df = pd.DataFrame(requirement_status_rows)
    req_path = TABLES / "day2_6_tala_requirement_status.csv"
    TABLES.mkdir(parents=True, exist_ok=True)
    req_df.to_csv(req_path, index=False)
    n_found = int((req_df["status"] == "found").sum()) if not req_df.empty else 0
    print(f"\nSaved: {req_path.relative_to(PROJECT_ROOT)} ({n_found}/{len(TALA_REQUIREMENTS)} TALA requirements found)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
