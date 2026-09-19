"""Day 2.6A Parts E/F/G: process the Multimodal Reference Package manifest into
chunks, tables, and document images.

Reads data/interim/day2_6/multimodal_reference_assets.csv (already fetched by
scripts/collect_reference_package.py) and writes:
  data/processed/reference_document_chunks.csv
  data/processed/reference_tables.csv
  data/processed/reference_images.csv

Only 'collected' or 'already_available' rows with non-empty extracted_text are
processed -- not_found/blocked/irrelevant/duplicate/extraction_failed rows are
skipped (they carry no usable text).

Usage: python scripts/process_reference_package.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.hydration import BROWSER_HEADERS  # noqa: E402
from src.reference_features import build_chunks, extract_html_images, extract_html_tables  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "data" / "interim" / "day2_6" / "multimodal_reference_assets.csv"
CHUNKS_OUT = PROJECT_ROOT / "data" / "processed" / "reference_document_chunks.csv"
TABLES_OUT = PROJECT_ROOT / "data" / "processed" / "reference_tables.csv"
IMAGES_OUT = PROJECT_ROOT / "data" / "processed" / "reference_images.csv"

PROCESSABLE_STATUSES = ("collected", "already_available")


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"{MANIFEST_PATH} not found -- run scripts/collect_reference_package.py first.")
        return 1

    df = pd.read_csv(MANIFEST_PATH)
    processable = df[
        df["reference_status"].isin(PROCESSABLE_STATUSES)
        & df["extracted_text"].notna() & (df["extracted_text"].astype(str).str.strip() != "")
    ]
    print(f"{len(processable)}/{len(df)} reference_assets rows are processable "
          f"(reference_status in {PROCESSABLE_STATUSES}, non-empty extracted_text)")

    all_chunks, all_tables, all_images = [], [], []

    for _, r in processable.iterrows():
        chunks = build_chunks(
            reference_id=r["reference_id"], brand=r["brand"], reference_type=r["reference_type"],
            document_title=str(r.get("document_title", "")), extracted_text=str(r["extracted_text"]),
            source_url=r["source_url"], evidence_strength=r["evidence_strength"], retrieved_at=r["retrieved_at"],
        )
        all_chunks.extend(chunks)

        if r.get("mime_type") == "text/html" and bool(r.get("has_tables")):
            try:
                resp = requests.get(r["source_url"], headers=BROWSER_HEADERS, timeout=15)
                if resp.status_code == 200:
                    all_tables.extend(extract_html_tables(resp.text, r["reference_id"], r["source_url"], r["evidence_strength"]))
                    all_images.extend(extract_html_images(resp.text, r["reference_id"], r["source_url"], r["evidence_strength"]))
            except Exception as exc:
                print(f"  [{r['reference_id']}] table/image re-fetch failed: {str(exc)[:80]}")

    chunks_df = pd.DataFrame(all_chunks)
    tables_df = pd.DataFrame(all_tables)
    images_df = pd.DataFrame(all_images)

    chunks_df.to_csv(CHUNKS_OUT, index=False)
    tables_df.to_csv(TABLES_OUT, index=False)
    images_df.to_csv(IMAGES_OUT, index=False)

    print(f"\nSaved: {CHUNKS_OUT.relative_to(PROJECT_ROOT)} ({len(chunks_df)} chunks)")
    print(f"Saved: {TABLES_OUT.relative_to(PROJECT_ROOT)} ({len(tables_df)} tables)")
    print(f"Saved: {IMAGES_OUT.relative_to(PROJECT_ROOT)} ({len(images_df)} document images)")

    if not chunks_df.empty:
        print("\nChunks by reference_type:")
        print(chunks_df.groupby("reference_type").size().to_string())
    if not tables_df.empty:
        print("\nTables by table_type:")
        print(tables_df.groupby("table_type").size().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
