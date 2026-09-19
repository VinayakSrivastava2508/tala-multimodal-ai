"""Day 2.6A orchestrator: build and validate the Multimodal Reference Package.

Stages:
  1. collect_reference_package.py  -- discovery + fetch (network; skipped by
     default if data/interim/day2_6/multimodal_reference_assets.csv already
     exists -- pass --refresh to re-collect).
  2. process_reference_package.py  -- chunking/table/image extraction (offline;
     always re-run so it picks up any newly collected documents).
  3. build_claim_multimodal_candidates.py -- rebuild claim-evidence candidates
     now that the reference corpus is populated (offline; always re-run).
  4. audit_assignment_modalities.py -- regenerate the authoritative audit
     tables, including the Day 2.6A reference-package tables (offline).

Does NOT touch video collection, fusion, RAG, or engagement prediction.

Usage:
    python scripts/run_day2_6a_reference_package.py [--refresh]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

REFERENCE_MANIFEST_PATH = PROJECT_ROOT / "data" / "interim" / "day2_6" / "multimodal_reference_assets.csv"

STAGES_NETWORK = [
    ("Reference Package collection", "scripts/collect_reference_package.py", REFERENCE_MANIFEST_PATH),
]
STAGES_OFFLINE = [
    ("Reference Package processing (chunks/tables/images)", "scripts/process_reference_package.py"),
    ("Claim-evidence candidate layer (rebuild with reference evidence)", "scripts/build_claim_multimodal_candidates.py"),
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
    for label, rel_script, output_path in STAGES_NETWORK:
        if output_path.exists() and not refresh:
            print(f"\n[{label}] {output_path.relative_to(PROJECT_ROOT)} already exists -- skipping "
                  f"(pass --refresh to re-collect from the network)")
            continue
        exit_codes.append(run_stage(label, rel_script))

    for label, rel_script in STAGES_OFFLINE:
        exit_codes.append(run_stage(label, rel_script))

    print(f"\n{'=' * 70}")
    if any(code != 0 for code in exit_codes):
        print("Day 2.6A reference-package pipeline: ONE OR MORE STAGES FAILED. See output above.")
        return 1
    print("Day 2.6A reference-package pipeline: all stages completed. "
          "See outputs/tables/day2_6_reference_package_readiness.csv and "
          "outputs/tables/assignment_modality_audit.csv for the authoritative status.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
