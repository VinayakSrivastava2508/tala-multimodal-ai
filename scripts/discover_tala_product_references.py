"""Day 2.6B Part A/B/C: targeted TALA product-page enumeration and
product-level reference-content extraction (care, materials, fit, size,
product specification), plus (Part E) per-page official video-source
discovery, sharing the single page fetch across both purposes.

Usage: python scripts/discover_tala_product_references.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.product_pages import (  # noqa: E402
    TALA_PRODUCT_DOMAINS,
    enumerate_diverse_products,
    extract_care_fields,
    extract_material_composition,
    extract_model_info,
    extract_product_sections,
    fetch_catalog,
    fetch_product_page_html,
    discover_page_video_sources,
    is_explicit_care_instruction,
)

OUT_DIR = PROJECT_ROOT / "data" / "interim" / "day2_6"
FUNNEL_PATH = PROJECT_ROOT / "outputs" / "tables" / "day2_6b_product_page_funnel.csv"
EXTRACTS_PATH = OUT_DIR / "tala_product_reference_extracts.csv"
VIDEO_DISCOVERY_PATH = OUT_DIR / "tala_official_video_discovery.csv"

PER_CATEGORY_TARGET = 3
NOW_ISO = datetime.now(timezone.utc).isoformat()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FUNNEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Enumeration] scanning {TALA_PRODUCT_DOMAINS[0]} catalog for diverse product pages ...")
    candidates = enumerate_diverse_products(TALA_PRODUCT_DOMAINS[0], per_category=PER_CATEGORY_TARGET)
    print(f"  {len(candidates)} product(s) selected across {len(set(c.category for c in candidates))} categories")
    for cat in sorted(set(c.category for c in candidates)):
        n = sum(1 for c in candidates if c.category == cat)
        print(f"    {cat}: {n}")

    # Variant/size lookup: re-use the already-fetched catalog rather than a
    # second network call per product.
    catalog = fetch_catalog(TALA_PRODUCT_DOMAINS[0])
    handle_to_product = {p.get("handle"): p for p in catalog}

    funnel_rows = []
    extract_rows = []
    video_rows = []
    ref_counter = 0
    video_id_seen_on: dict[str, list[str]] = {}

    for cand in candidates:
        result = fetch_product_page_html(cand.canonical_url)
        funnel_row = {
            "product_name": cand.product_name, "product_handle": cand.product_handle,
            "canonical_url": cand.canonical_url, "category": cand.category, "collection": cand.collection,
            "availability": cand.availability, "discovery_method": cand.discovery_method,
            "retrieved_at": NOW_ISO, "http_status": None, "page_extraction_status": "failed",
        }
        if result is None:
            funnel_row["page_extraction_status"] = "blocked_or_failed"
            funnel_rows.append(funnel_row)
            print(f"  [{cand.product_handle}] fetch failed/blocked")
            continue

        status, html = result
        funnel_row["http_status"] = status
        if status != 200:
            funnel_row["page_extraction_status"] = f"http_{status}"
            funnel_rows.append(funnel_row)
            continue

        sections = extract_product_sections(html)
        if not sections:
            funnel_row["page_extraction_status"] = "no_sections_found"
            funnel_rows.append(funnel_row)
            print(f"  [{cand.product_handle}] no Wear/Care/Aware sections found")
        else:
            funnel_row["page_extraction_status"] = "extracted"
            funnel_rows.append(funnel_row)

            def next_id() -> str:
                nonlocal ref_counter
                ref_counter += 1
                return f"tpr_{ref_counter:04d}"

            def base_row(subtype: str, section_title: str, text: str) -> dict:
                return {
                    "product_reference_id": next_id(), "product_name": cand.product_name,
                    "product_category": cand.category, "source_url": cand.canonical_url,
                    "section_title": section_title, "reference_subtype": subtype,
                    "extracted_text": text, "care_temperature": "", "washing_method": "",
                    "drying_method": "", "ironing_guidance": "", "bleaching_guidance": "",
                    "material_composition": "", "fit_guidance": "", "size_guidance": "",
                    "extraction_method": "bs4_class_selector", "evidence_strength": "strong" if len(text) >= 150 else "medium",
                    "retrieved_at": NOW_ISO,
                    "provenance_note": f"product_handle={cand.product_handle}; product_type_raw={cand.product_type_raw}",
                }

            wear_text = sections.get("wear", "")
            care_text = sections.get("care", "")
            aware_text = sections.get("aware", "")

            if wear_text:
                row = base_row("product_specification", "Wear", wear_text)
                extract_rows.append(row)

            if care_text:
                if is_explicit_care_instruction(care_text):
                    row = base_row("care_guidance", "Care", care_text)
                    row.update(extract_care_fields(care_text))
                    extract_rows.append(row)
                else:
                    print(f"  [{cand.product_handle}] Care tab present but no explicit instruction -- not counted as care evidence")

            if aware_text:
                row = base_row("material_composition", "Aware", aware_text)
                row["material_composition"] = extract_material_composition(aware_text, wear_text)
                extract_rows.append(row)

            model_info = extract_model_info(wear_text)
            if model_info:
                row = base_row("fit_guidance", "Wear (model info)", model_info)
                row["fit_guidance"] = model_info
                extract_rows.append(row)

            product = handle_to_product.get(cand.product_handle, {})
            sizes = sorted({o for v in product.get("variants", []) for o in [v.get("option1")] if o})
            if sizes:
                text = f"Available sizes: {', '.join(sizes)}"
                row = base_row("size_guidance", "Variants", text)
                row["size_guidance"] = ", ".join(sizes)
                extract_rows.append(row)

            # Part E: per-page video-source discovery (shares this same fetch).
            page_videos = discover_page_video_sources(html)
            for v in page_videos:
                video_id_seen_on.setdefault(v.video_id, []).append(cand.product_handle)
                video_rows.append({
                    "video_id": v.video_id, "product_name": cand.product_name,
                    "product_category": cand.category, "source_product_page": cand.canonical_url,
                    "direct_media_url": v.asset_url, "mime_type": v.mime_type,
                    "official_cdn_domain": "cdn.shopify.com" if "cdn" in v.asset_url or "shopify" in v.asset_url else "wearetala.com",
                    "discovery_method": "product_page_source_element", "rights_or_access_basis": "official_direct_public_asset",
                })

    funnel_df = pd.DataFrame(funnel_rows)
    extracts_df = pd.DataFrame(extract_rows)
    funnel_df.to_csv(FUNNEL_PATH, index=False)
    extracts_df.to_csv(EXTRACTS_PATH, index=False)

    # Consolidate video candidates: one row per distinct video_id, listing every
    # product page it was seen on (a video reused across many pages is one
    # asset, not many).
    video_df = pd.DataFrame(video_rows)
    if not video_df.empty:
        video_df = video_df.drop_duplicates(subset=["video_id", "source_product_page"])
        agg = video_df.groupby("video_id").agg({
            "product_name": lambda s: ";".join(sorted(set(s))),
            "product_category": lambda s: ";".join(sorted(set(s))),
            "source_product_page": lambda s: ";".join(sorted(set(s))),
            "direct_media_url": "first", "mime_type": "first",
            "official_cdn_domain": "first", "discovery_method": "first",
            "rights_or_access_basis": "first",
        }).reset_index()
        agg["n_products_featured_on"] = agg["source_product_page"].apply(lambda s: len(s.split(";")))
        video_df = agg
    video_df.to_csv(VIDEO_DISCOVERY_PATH, index=False)

    print(f"\nSaved: {FUNNEL_PATH.relative_to(PROJECT_ROOT)} ({len(funnel_df)} product pages)")
    print(f"Saved: {EXTRACTS_PATH.relative_to(PROJECT_ROOT)} ({len(extracts_df)} extract rows)")
    print(f"Saved: {VIDEO_DISCOVERY_PATH.relative_to(PROJECT_ROOT)} ({len(video_df)} distinct video(s) discovered)")
    if not extracts_df.empty:
        print("\nExtracts by subtype:")
        print(extracts_df["reference_subtype"].value_counts().to_string())
        n_care = int((extracts_df["reference_subtype"] == "care_guidance").sum())
        n_care_categories = extracts_df.loc[extracts_df["reference_subtype"] == "care_guidance", "product_category"].nunique()
        print(f"\nCare records recovered: {n_care} across {n_care_categories} categories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
