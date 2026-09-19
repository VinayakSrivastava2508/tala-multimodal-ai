"""Day 2.6B Part D: merge verified product-reference extracts into the
Multimodal Reference Package.

Boilerplate care instructions that are IDENTICAL across multiple products are
consolidated into ONE reference row (one unique instruction), with every
product it applies to retained in provenance -- never counted as N
independent documents for the same instruction.

Usage: python scripts/merge_product_references_into_package.py
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXTRACTS_PATH = PROJECT_ROOT / "data" / "interim" / "day2_6" / "tala_product_reference_extracts.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "interim" / "day2_6" / "multimodal_reference_assets.csv"
TABLES_OUT = PROJECT_ROOT / "data" / "processed" / "reference_tables.csv"
CARE_COVERAGE_OUT = PROJECT_ROOT / "outputs" / "tables" / "day2_6b_care_guidance_coverage.csv"

NOW_ISO = datetime.now(timezone.utc).isoformat()

SUBTYPE_TO_REFERENCE_TYPE = {
    "care_guidance": "care_guide",
    "product_specification": "product_specification",
    "material_composition": "material_specification",
    "fit_guidance": "size_guide",
    "size_guidance": "size_guide",
}

CARE_FIELDS = ["care_temperature", "washing_method", "drying_method", "ironing_guidance", "bleaching_guidance"]


def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def next_reference_id(existing_ids: pd.Series) -> callable:
    tala_ids = existing_ids[existing_ids.str.startswith("ref_tala_", na=False)]
    max_n = 0
    for rid in tala_ids:
        try:
            max_n = max(max_n, int(rid.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    counter = [max_n]

    def _next() -> str:
        counter[0] += 1
        return f"ref_tala_{counter[0]:03d}"
    return _next


def main() -> int:
    if not EXTRACTS_PATH.exists():
        print(f"{EXTRACTS_PATH} not found -- run scripts/discover_tala_product_references.py first.")
        return 1

    extracts = pd.read_csv(EXTRACTS_PATH)
    manifest = pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else pd.DataFrame()
    gen_id = next_reference_id(manifest.get("reference_id", pd.Series(dtype=str)))

    new_rows = []

    # ── Care instructions: dedupe identical (temp, wash, dry, iron, bleach) tuples ──
    care = extracts[extracts["reference_subtype"] == "care_guidance"].copy()
    for field in CARE_FIELDS:
        care[field] = care[field].fillna("")
    if not care.empty:
        grouped = care.groupby(CARE_FIELDS, dropna=False)
        for care_key, group in grouped:
            products = sorted(group["product_name"].unique())
            categories = sorted(group["product_category"].unique())
            urls = sorted(group["source_url"].unique())
            rep_text = group.iloc[0]["extracted_text"]
            reference_id = gen_id()
            new_rows.append({
                "reference_id": reference_id, "brand": "TALA", "reference_type": "care_guide",
                "document_title": f"TALA garment-care instructions ({len(products)} product(s))",
                "source_url": urls[0], "canonical_url": urls[0], "source_domain": "www.wearetala.com",
                "publication_date": "", "effective_date": "", "retrieved_at": NOW_ISO,
                "mime_type": "text/html", "local_path": "", "extracted_text": rep_text,
                "text_length": len(rep_text), "page_count": None, "has_tables": False, "has_images": False,
                "extraction_method": "bs4_class_selector", "evidence_strength": "strong",
                "rights_or_access_basis": "official_direct_public_asset", "processing_status": "downloaded",
                "file_hash": hashlib.sha256(rep_text.encode()).hexdigest(),
                "provenance_note": (
                    f"applies to {len(products)} product(s) across {len(categories)} categor(y/ies): "
                    f"{', '.join(products)}; source pages: {'; '.join(urls)}"
                ),
                "discovery_method": "product_page_section_extraction", "official_source": True,
                "certification_authority": "", "claim_relevance": "materials;manufacturing",
                "reference_status": "collected", "failure_reason": "",
                "_care_key": care_key, "_products": products, "_categories": categories, "_urls": urls,
            })
        n_unique_care = len(new_rows)
        print(f"Care instructions: {len(care)} product-level records -> {n_unique_care} unique instruction(s) "
              f"(consolidated, not counted as {len(care)} independent documents)")

    # ── Material composition, product specification, size/fit guidance ────────────
    for subtype in ("material_composition", "product_specification", "fit_guidance", "size_guidance"):
        rows = extracts[extracts["reference_subtype"] == subtype]
        for _, r in rows.iterrows():
            text = r["extracted_text"]
            reference_id = gen_id()
            new_rows.append({
                "reference_id": reference_id, "brand": "TALA", "reference_type": SUBTYPE_TO_REFERENCE_TYPE[subtype],
                "document_title": f"{r['product_name']} -- {r['section_title']}",
                "source_url": r["source_url"], "canonical_url": r["source_url"], "source_domain": "www.wearetala.com",
                "publication_date": "", "effective_date": "", "retrieved_at": NOW_ISO,
                "mime_type": "text/html", "local_path": "", "extracted_text": text,
                "text_length": len(str(text)), "page_count": None, "has_tables": False, "has_images": False,
                "extraction_method": "bs4_class_selector", "evidence_strength": r["evidence_strength"],
                "rights_or_access_basis": "official_direct_public_asset", "processing_status": "downloaded",
                "file_hash": hashlib.sha256(str(text).encode()).hexdigest(),
                "provenance_note": f"product={r['product_name']}; category={r['product_category']}; section={r['section_title']}",
                "discovery_method": "product_page_section_extraction", "official_source": True,
                "certification_authority": "", "claim_relevance": r["product_category"],
                "reference_status": "collected", "failure_reason": "",
            })

    new_df = pd.DataFrame(new_rows)
    care_meta_cols = ["_care_key", "_products", "_categories", "_urls"]
    if any(c in new_df.columns for c in care_meta_cols):
        care_meta = new_df[["reference_id"] + [c for c in care_meta_cols if c in new_df.columns]].copy()
        care_meta = care_meta.dropna(subset=["_care_key"])
    else:
        care_meta = pd.DataFrame()
    new_df = new_df.drop(columns=[c for c in care_meta_cols if c in new_df.columns])

    # Idempotency: replace any previous run's product-page-derived rows rather than
    # appending on top of them -- rerunning this script (e.g. via the Day 2.6B
    # orchestrator) must never duplicate the same 79 merged rows a second time.
    if not manifest.empty and "discovery_method" in manifest.columns:
        manifest = manifest[manifest["discovery_method"] != "product_page_section_extraction"]
    combined = pd.concat([manifest, new_df], ignore_index=True) if not manifest.empty else new_df
    combined.to_csv(MANIFEST_PATH, index=False)

    print(f"\nSaved: {_relpath(MANIFEST_PATH)} ({len(combined)} total rows, {len(new_df)} newly merged)")
    print(new_df["reference_type"].value_counts().to_string())

    # Save the care-consolidation detail table for the coverage audit (Part D/J).
    if not care_meta.empty:
        care_meta_path = CARE_COVERAGE_OUT
        rows = []
        for _, r in care_meta.iterrows():
            rows.append({
                "reference_id": r["reference_id"],
                "care_temperature": r["_care_key"][0], "washing_method": r["_care_key"][1],
                "drying_method": r["_care_key"][2], "ironing_guidance": r["_care_key"][3],
                "bleaching_guidance": r["_care_key"][4],
                "n_products": len(r["_products"]), "n_categories": len(r["_categories"]),
                "products": ";".join(r["_products"]), "categories": ";".join(r["_categories"]),
            })
        pd.DataFrame(rows).to_csv(care_meta_path, index=False)
        print(f"Saved: {_relpath(care_meta_path)} ({len(rows)} unique care instruction(s))")

        # A genuine structured table: which products each unique care instruction
        # applies to (Part D: "create structured tables when appropriate").
        tables_path = TABLES_OUT
        try:
            existing_tables = pd.read_csv(tables_path) if tables_path.exists() else pd.DataFrame()
        except pd.errors.EmptyDataError:
            existing_tables = pd.DataFrame()
        care_table_rows = []
        for i, r in care_meta.iterrows():
            headers = ["care_temperature", "washing_method", "drying_method", "ironing_guidance", "bleaching_guidance", "applies_to_products"]
            row_text = ";".join([
                r["_care_key"][0] or "n/a", r["_care_key"][1] or "n/a", r["_care_key"][2] or "n/a",
                r["_care_key"][3] or "n/a", r["_care_key"][4] or "n/a", ", ".join(r["_products"]),
            ])
            care_table_rows.append({
                "table_id": f"{r['reference_id']}_table_01", "reference_id": r["reference_id"],
                "page_number_or_section": "", "table_type": "other",
                "headers": ";".join(headers), "rows_or_normalised_text": row_text,
                "units": "", "source_url": r["_urls"][0], "extraction_method": "manual_consolidation",
                "evidence_strength": "strong",
                "provenance_note": f"care-instruction-to-product applicability table, {len(r['_products'])} product(s)",
            })
        care_tables_df = pd.DataFrame(care_table_rows)
        # Idempotency: drop any previously-appended manual-consolidation tables
        # before appending this run's fresh set, mirroring the manifest fix above.
        if not existing_tables.empty and "extraction_method" in existing_tables.columns:
            existing_tables = existing_tables[existing_tables["extraction_method"] != "manual_consolidation"]
        combined_tables = pd.concat([existing_tables, care_tables_df], ignore_index=True) if not existing_tables.empty else care_tables_df
        combined_tables.to_csv(tables_path, index=False)
        print(f"Saved: {_relpath(tables_path)} ({len(combined_tables)} total tables, +{len(care_table_rows)} care-applicability tables)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
