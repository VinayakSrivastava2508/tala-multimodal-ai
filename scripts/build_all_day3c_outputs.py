"""Orchestrates the full Day 3C build pipeline in the required order.

Order matters: build_mixed_claim_themes.py reads and enriches files written
by build_analytical_synthesis.py, and build_executive_findings.py reads
outputs from every earlier stage.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    "build_analytical_synthesis.py",
    "build_mixed_claim_themes.py",
    "build_creator_strategy_comparison.py",
    "build_strategic_interpretation.py",
    "build_executive_findings.py",
    "build_synthesis_figures.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n=== {step} ===")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / step)], check=True)
    print("\nDay 3C full build pipeline complete.")


if __name__ == "__main__":
    main()
