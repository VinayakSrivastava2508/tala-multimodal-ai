"""Day 2 Task 2 Part A: freeze the validated Day 2 analytical datasets.

Snapshots the six validated Day 2 outputs (no raw DDG/search leads) as a versioned,
checksummed manifest so every downstream feature table traces back to an exact,
reproducible input state.

Usage: python scripts/freeze_day2_datasets.py
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

from src.validation import guess_schema, validate  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "day2_dataset_manifest.csv"
SNAPSHOT_PATH = PROCESSED_DIR / "day2_dataset_snapshot.json"
DATASET_VERSION = "day2_verified_v1"
NOW_ISO = datetime.now(timezone.utc).isoformat()

# path (relative to project root) -> (unique_id_field, critical_fields, has_evidence_strength)
ANALYTICAL_FILES: dict = {
    "data/interim/day2/creator_posts_enriched.csv": {
        "id_field": "source_id",
        "critical_fields": ["brand", "platform", "evidence_strength", "usable_as_creator_post", "brand_link_verified"],
        "has_strength": True,
    },
    "data/interim/day2/platform_strategy_enriched.csv": {
        "id_field": "source_id",
        "critical_fields": ["brand", "platform", "evidence_url", "evidence_strength"],
        "has_strength": True,
    },
    "data/interim/day2/hydrated_sources.csv": {
        "id_field": "source_id",
        "critical_fields": ["source_url", "hydration_status", "evidence_strength", "rag_usable"],
        "has_strength": True,
    },
    "data/corpora/official_claims_corpus.csv": {
        "id_field": "document_id",
        "critical_fields": ["brand", "source_url", "evidence_strength", "rag_usable"],
        "has_strength": True,
    },
    "data/corpora/customer_experience_corpus.csv": {
        "id_field": "document_id",
        "critical_fields": ["brand", "source_url", "evidence_strength", "rag_usable"],
        "has_strength": True,
    },
    "data/corpora/creator_strategy_corpus.csv": {
        "id_field": "document_id",
        "critical_fields": ["brand", "source_url", "evidence_strength", "rag_usable"],
        "has_strength": True,
    },
}


def _sha256_of_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze_file(rel_path: str, spec: dict) -> dict:
    """Build one manifest row + snapshot entry for a single analytical file."""
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        return {"path": rel_path, "exists": False}

    df = pd.read_csv(path, low_memory=False)
    checksum = _sha256_of_file(path)

    id_field = spec["id_field"]
    if id_field in df.columns:
        unique_ids = int(df[id_field].dropna().nunique())
        duplicate_ids = int(df[id_field].dropna().duplicated().sum())
    else:
        unique_ids, duplicate_ids = 0, 0

    missingness = {}
    for col in spec["critical_fields"]:
        if col in df.columns:
            missingness[col] = round(100 * float(df[col].isna().mean()), 2)
        else:
            missingness[col] = None  # column absent entirely

    strength_dist = {}
    if spec["has_strength"] and "evidence_strength" in df.columns:
        strength_dist = df["evidence_strength"].value_counts(dropna=False).to_dict()
        strength_dist = {str(k): int(v) for k, v in strength_dist.items()}

    schema_name = guess_schema(path.name)
    if schema_name:
        passed, errors = validate(df, schema_name)
        validation_status = "PASS" if passed else "FAIL"
    else:
        validation_status = "NO_SCHEMA"
        errors = []

    manifest_row = {
        "path": rel_path,
        "row_count": len(df),
        "column_count": len(df.columns),
        "sha256": checksum,
        "unique_id_count": unique_ids,
        "duplicate_id_count": duplicate_ids,
        "missingness_json": json.dumps(missingness),
        "evidence_strength_distribution_json": json.dumps(strength_dist),
        "generated_at_utc": NOW_ISO,
        "source_script": "scripts/hydrate_day2_sources.py + scripts/repair_day2_creator_evidence.py",
        "validation_status": validation_status,
        "dataset_version": DATASET_VERSION,
    }
    snapshot_entry = {
        **manifest_row,
        "missingness_pct": missingness,
        "evidence_strength_distribution": strength_dist,
        "validation_errors": errors if validation_status == "FAIL" else [],
    }
    return manifest_row, snapshot_entry


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    snapshot = {
        "dataset_version": DATASET_VERSION,
        "generated_at_utc": NOW_ISO,
        "source_script": "scripts/freeze_day2_datasets.py",
        "files": {},
    }

    print(f"Freezing Day 2 analytical datasets as {DATASET_VERSION} ...")
    any_fail = False
    for rel_path, spec in ANALYTICAL_FILES.items():
        result = freeze_file(rel_path, spec)
        if isinstance(result, dict) and result.get("exists") is False:
            print(f"  [MISSING] {rel_path} -- run the Day 2 pipeline first")
            any_fail = True
            continue
        manifest_row, snapshot_entry = result
        manifest_rows.append(manifest_row)
        snapshot["files"][rel_path] = snapshot_entry
        status = manifest_row["validation_status"]
        print(f"  [{status}] {rel_path}  rows={manifest_row['row_count']}  "
              f"unique_ids={manifest_row['unique_id_count']}  dup_ids={manifest_row['duplicate_id_count']}  "
              f"sha256={manifest_row['sha256'][:12]}...")
        if status == "FAIL":
            any_fail = True

    pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False)
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nManifest:  {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Snapshot:  {SNAPSHOT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Dataset version: {DATASET_VERSION}")
    if any_fail:
        print("\nWARNING: one or more frozen files failed validation or are missing -- "
              "review before building features on top of this freeze.")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
