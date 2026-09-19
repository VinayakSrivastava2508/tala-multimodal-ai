"""Day 2.6B orchestrator: targeted TALA product-reference and official-video
recovery.

Stages (network stages skipped by default if their output already exists --
pass --refresh to re-run from scratch):
  1. discover_tala_product_references.py -- product-page enumeration +
     care/spec/fit/size extraction + per-page video-source discovery (network).
  2. merge_product_references_into_package.py -- merge extracts into the
     Reference Package, consolidating boilerplate care instructions (offline).
  3. process_reference_package.py -- rebuild chunks/tables/images for the
     whole (now larger) Reference Package (offline, but re-fetches HTML for
     any row with has_tables=True to extract real tables/images).
  4. process_tala_official_videos.py -- download + genuinely process eligible
     official TALA videos (network for the download step only).
  5. build_claim_multimodal_candidates.py -- rebuild claim-evidence candidates
     with product-aware matching and the expanded video pool (offline).
  6. build_full_modality_bundles.py -- assemble full-modality prototype
     bundles (offline).
  7. audit_assignment_modalities.py -- regenerate all audit tables including
     the Day 2.6B readiness gates (offline).

Ordering note: stage 3 (process_reference_package.py) rebuilds
reference_tables.csv by re-fetching each has_tables=True row's source_url, but
now explicitly preserves any pre-existing extraction_method='manual_consolidation'
rows (the Day 2.6B care-applicability tables stage 2 appends directly) across
reruns -- stage 2 must still run before stage 3 so those rows exist to preserve.

Does NOT implement final fusion, RAG, engagement prediction, or slides.

Usage:
    python scripts/run_day2_6b_product_video_recovery.py [--refresh]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

PRODUCT_EXTRACTS_PATH = PROJECT_ROOT / "data" / "interim" / "day2_6" / "tala_product_reference_extracts.csv"
VIDEO_DISCOVERY_PATH = PROJECT_ROOT / "data" / "interim" / "day2_6" / "tala_official_video_discovery.csv"

STAGES_NETWORK_DISCOVERY = [
    ("TALA product-page discovery + extraction", "scripts/discover_tala_product_references.py", PRODUCT_EXTRACTS_PATH),
]
STAGES_OFFLINE_MERGE = [
    ("Merge product references into Reference Package", "scripts/merge_product_references_into_package.py"),
    ("Reference Package processing (chunks/tables/images)", "scripts/process_reference_package.py"),
]
STAGE_VIDEO_PROCESSING = ("Download + process official TALA videos", "scripts/process_tala_official_videos.py")
STAGES_FINAL = [
    ("Claim-evidence candidate layer (product-aware rebuild)", "scripts/build_claim_multimodal_candidates.py"),
    ("Full-modality prototype bundles", "scripts/build_full_modality_bundles.py"),
    ("Assignment-modality audit", "scripts/audit_assignment_modalities.py"),
]


def run_stage(label: str, rel_script: str) -> int:
    print(f"\n{'=' * 70}\n[{label}] running {rel_script}\n{'=' * 70}")
    result = subprocess.run([PYTHON, str(PROJECT_ROOT / rel_script)], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[{label}] FAILED (exit code {result.returncode})")
    return result.returncode


def main() -> int:
    refresh = "--refresh" in sys.argv
    exit_codes = []

    for label, rel_script, output_path in STAGES_NETWORK_DISCOVERY:
        if output_path.exists() and not refresh:
            print(f"\n[{label}] {output_path.relative_to(PROJECT_ROOT)} already exists -- skipping "
                  f"(pass --refresh to re-discover from the network)")
        else:
            exit_codes.append(run_stage(label, rel_script))

    for label, rel_script in STAGES_OFFLINE_MERGE:
        exit_codes.append(run_stage(label, rel_script))

    label, rel_script = STAGE_VIDEO_PROCESSING
    if VIDEO_DISCOVERY_PATH.exists():
        exit_codes.append(run_stage(label, rel_script))
    else:
        print(f"\n[{label}] {VIDEO_DISCOVERY_PATH.relative_to(PROJECT_ROOT)} not found -- run discovery first")
        exit_codes.append(1)

    for label, rel_script in STAGES_FINAL:
        exit_codes.append(run_stage(label, rel_script))

    print(f"\n{'=' * 70}")
    if any(code != 0 for code in exit_codes):
        print("Day 2.6B product/video recovery pipeline: ONE OR MORE STAGES FAILED. See output above.")
        return 1
    print("Day 2.6B product/video recovery pipeline: all stages completed. "
          "See outputs/tables/day2_6b_readiness_gates.csv and outputs/tables/assignment_modality_audit.csv "
          "for the authoritative status.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
