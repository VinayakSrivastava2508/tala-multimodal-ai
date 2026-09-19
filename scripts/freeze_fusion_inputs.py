"""Day 3A Part 4: freeze the exact input state used for primary multimodal
evidence fusion, so every fusion output traces back to a checksummed,
reproducible snapshot -- never copying binary media into git.

Usage: python scripts/freeze_fusion_inputs.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "fusion"
MANIFEST_PATH = OUT_DIR / "fusion_input_manifest.json"
FILES_PATH = OUT_DIR / "fusion_input_files.csv"
DATASET_VERSION = "day3_primary_fusion_v1"
NOW_ISO = datetime.now(timezone.utc).isoformat()

# path (relative to project root) -> (unique_id_field, modality)
FUSION_INPUT_FILES: dict = {
    "data/corpora/official_claims_corpus.csv": ("document_id", "text"),
    "data/corpora/customer_experience_corpus.csv": ("document_id", "text"),
    "data/processed/claim_multimodal_evidence_candidates.csv": ("evidence_bundle_id", "multi"),
    "data/processed/full_modality_prototype_bundles.csv": ("bundle_id", "multi"),
    "data/interim/day2_5/image_assets.csv": ("asset_id", "image"),
    "data/interim/day2_6/video_assets_expanded.csv": ("video_asset_id", "video"),
    "data/processed/video_frame_features.csv": ("frame_id", "video"),
    "data/processed/video_level_features.csv": ("video_asset_id", "video"),
    "data/interim/day2_6/multimodal_reference_assets.csv": ("reference_id", "reference"),
    "data/processed/reference_document_chunks.csv": ("chunk_id", "reference"),
    "data/processed/reference_tables.csv": ("table_id", "reference"),
    "data/processed/reference_images.csv": ("reference_image_id", "reference"),
}


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_file(rel_path: str, id_field: str, modality: str) -> dict | None:
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    checksum = _sha256_of_file(path)

    if id_field in df.columns:
        unique_ids = int(df[id_field].dropna().nunique())
        duplicate_ids = int(df[id_field].dropna().duplicated().sum())
    else:
        unique_ids, duplicate_ids = 0, 0

    strength_dist = {}
    if "evidence_strength" in df.columns:
        strength_dist = {str(k): int(v) for k, v in df["evidence_strength"].value_counts(dropna=False).to_dict().items()}

    return {
        "path": rel_path, "modality": modality, "row_count": len(df), "column_count": len(df.columns),
        "sha256": checksum, "unique_id_count": unique_ids, "duplicate_id_count": duplicate_ids,
        "evidence_strength_distribution": strength_dist, "generated_at_utc": NOW_ISO,
        "validation_status": "PRESENT",
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    snapshot = {"dataset_version": DATASET_VERSION, "generated_at_utc": NOW_ISO, "files": {}}

    print(f"Freezing Day 3A fusion inputs as {DATASET_VERSION} ...")
    any_missing = False
    for rel_path, (id_field, modality) in FUSION_INPUT_FILES.items():
        entry = freeze_file(rel_path, id_field, modality)
        if entry is None:
            print(f"  [MISSING] {rel_path}")
            any_missing = True
            manifest_rows.append({"path": rel_path, "modality": modality, "row_count": 0, "validation_status": "MISSING"})
            continue
        manifest_rows.append(entry)
        snapshot["files"][rel_path] = entry
        print(f"  [OK] {rel_path}  rows={entry['row_count']}  unique_ids={entry['unique_id_count']}  "
              f"dup_ids={entry['duplicate_id_count']}  sha256={entry['sha256'][:12]}...")

    pd.DataFrame(manifest_rows).to_csv(FILES_PATH, index=False)
    MANIFEST_PATH.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nManifest: {FILES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Snapshot: {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Dataset version: {DATASET_VERSION}")
    if any_missing:
        print("\nWARNING: one or more fusion input files are missing -- run the Day 2.5/2.6A/2.6B pipelines first.")
    return 1 if any_missing else 0


if __name__ == "__main__":
    sys.exit(main())
