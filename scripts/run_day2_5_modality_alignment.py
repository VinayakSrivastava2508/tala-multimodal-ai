"""Day 2.5 orchestrator: run the full modality-realignment pipeline in order.

Stages:
  1. collect_official_product_media.py  -- Image Package (network; skipped by
     default if data/interim/day2_5/image_assets.csv already exists -- pass
     --refresh to re-collect from the brand websites).
  2. build_video_asset_manifest.py      -- Video Package discovery (network;
     same --refresh behaviour as stage 1).
  3. process_video_assets.py            -- genuine local video processing
     (offline; always re-run so it picks up any newly downloaded videos).
  4. build_claim_multimodal_candidates.py -- claim-evidence candidate layer
     (offline; always re-run).
  5. audit_assignment_modalities.py     -- regenerate the authoritative audit
     tables from current data (offline; always re-run).

Each stage is a real subprocess call to the existing standalone script (not a
re-implementation) so `python scripts/<stage>.py` on its own always matches
what the orchestrator does.

Usage:
    python scripts/run_day2_5_modality_alignment.py [--refresh]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

IMAGE_ASSETS_PATH = PROJECT_ROOT / "data" / "interim" / "day2_5" / "image_assets.csv"
VIDEO_ASSETS_PATH = PROJECT_ROOT / "data" / "interim" / "day2_5" / "video_assets.csv"

STAGES_NETWORK = [
    ("Image Package collection", "scripts/collect_official_product_media.py", IMAGE_ASSETS_PATH),
    ("Video Package discovery", "scripts/build_video_asset_manifest.py", VIDEO_ASSETS_PATH),
]
STAGES_OFFLINE = [
    ("Genuine video processing", "scripts/process_video_assets.py"),
    ("Claim-evidence candidate layer", "scripts/build_claim_multimodal_candidates.py"),
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
        print("Day 2.5 modality alignment pipeline: ONE OR MORE STAGES FAILED. See output above.")
        return 1
    print("Day 2.5 modality alignment pipeline: all stages completed. "
          "See outputs/tables/assignment_modality_audit.csv for the authoritative PASS/PARTIAL/FAIL status.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
