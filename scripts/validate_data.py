"""
scripts/validate_data.py

Run from the project root:
    python scripts/validate_data.py

Scans data/raw/ (including data/raw/templates/) for all CSV files,
infers the schema from each filename, and runs:
  1. Schema validation (required columns, allowed values, type ranges)
  2. Provenance checks (source_url, collection_date, brand, citation_ready)

Prints a detailed report and a summary table.
Exit code 0 = all non-empty files passed or are empty templates.
Exit code 1 = one or more files have hard failures.
"""

import sys
from pathlib import Path

# Make sure src/ is importable when run from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import validate_all_files, print_summary_report  # noqa: E402


def main() -> int:
    # Scan data/raw/ (original), plus data/interim/ and data/corpora/ when present
    scan_dirs = [PROJECT_ROOT / "data" / "raw"]
    for extra in ("data/interim", "data/corpora"):
        p = PROJECT_ROOT / extra
        if p.exists():
            scan_dirs.append(p)

    all_results: dict = {}
    for data_dir in scan_dirs:
        print(f"\nScanning: {data_dir}")
        print("-" * 70)
        results = validate_all_files(data_dir=data_dir, verbose=True)
        all_results.update(results)

    if not all_results:
        print("No CSV files found.")
        return 0

    print_summary_report(all_results)

    failures = [
        path for path, r in all_results.items()
        if r.get("rows", 0) > 0 and not r.get("passed")
    ]
    if failures:
        print(f"Action required: fix the {len(failures)} file(s) marked FAIL above.\n")
        return 1

    print("All files passed validation (empty templates are fine -- fill them in).\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
