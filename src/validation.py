"""Schema validation for DataFrames — checks columns, types, values, and provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Map CSV filename patterns to schema names
FILENAME_TO_SCHEMA: Dict[str, str] = {
    "social_posts":           "social_posts",
    "creator_posts":          "creator_posts",
    "customer_reviews":       "customer_reviews",
    "official_claims":        "official_claims",
    "competitor_platforms":   "competitor_platforms",
    "press_reddit_sources":   "press_reddit_sources",
    "claim_assessments":      "claim_assessments",
}

# Provenance fields every dataset must carry
PROVENANCE_FIELDS = ["source_url", "source_platform", "collection_date", "collected_by", "evidence_type"]


def load_schema() -> dict:
    """Return parsed schema.yaml as a dict."""
    with open(PROJECT_ROOT / "configs" / "schema.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def guess_schema(filename: str) -> str | None:
    """Guess the schema name from a CSV filename. Returns None if no match."""
    stem = Path(filename).stem.lower()
    for pattern, schema in FILENAME_TO_SCHEMA.items():
        if stem.startswith(pattern):
            return schema
    return None


# ── Core field validation ──────────────────────────────────────────────────────

def validate(df: pd.DataFrame, schema_name: str) -> Tuple[bool, list[str]]:
    """Validate a DataFrame against a named schema from configs/schema.yaml.

    Returns (passed: bool, errors: list[str]).
    Warnings (lines starting with WARNING) do not cause passed=False.
    """
    schema = load_schema()
    if schema_name not in schema["schemas"]:
        return False, [f"Unknown schema: '{schema_name}'. Valid: {list(schema['schemas'].keys())}"]

    fields = schema["schemas"][schema_name]["fields"]
    errors: list[str] = []

    for col, spec in fields.items():
        if col not in df.columns:
            errors.append(f"MISSING COLUMN: '{col}'")
            continue

        if spec.get("required") and df[col].isnull().any():
            n = int(df[col].isnull().sum())
            errors.append(f"NULL in required column '{col}': {n} row(s)")

        if "values" in spec and df[col].notna().any():
            allowed = set(spec["values"])
            invalid = set(df[col].dropna().unique()) - allowed
            if invalid:
                errors.append(f"INVALID VALUES in '{col}': {sorted(invalid)} — allowed: {sorted(allowed)}")

        if "min" in spec and pd.api.types.is_numeric_dtype(df[col]):
            n = int((df[col] < spec["min"]).sum())
            if n:
                errors.append(f"BELOW MIN in '{col}': {n} row(s) < {spec['min']}")

        if "max" in spec and pd.api.types.is_numeric_dtype(df[col]):
            n = int((df[col] > spec["max"]).sum())
            if n:
                errors.append(f"ABOVE MAX in '{col}': {n} row(s) > {spec['max']}")

    if "synthetic" in df.columns:
        n = int(df["synthetic"].fillna(False).sum())
        if n:
            errors.append(f"WARNING: {n} row(s) marked synthetic=True — exclude from final analysis")

    passed = all(not e.startswith(("MISSING", "NULL", "INVALID", "BELOW", "ABOVE")) for e in errors)
    return passed, errors


# ── Provenance checks ──────────────────────────────────────────────────────────

def check_source_url(df: pd.DataFrame) -> list[str]:
    """Return list of row indices where source_url is missing or blank."""
    if "source_url" not in df.columns:
        return ["source_url column missing entirely"]
    blank = df[df["source_url"].isna() | (df["source_url"].astype(str).str.strip() == "")]
    return blank.index.tolist()


def check_collection_date(df: pd.DataFrame) -> list[str]:
    """Return list of row indices where collection_date is missing."""
    if "collection_date" not in df.columns:
        return ["collection_date column missing entirely"]
    missing = df[df["collection_date"].isna()]
    return missing.index.tolist()


def check_brand(df: pd.DataFrame) -> list[str]:
    """Return list of row indices where brand is missing or blank."""
    if "brand" not in df.columns:
        return ["brand column missing entirely"]
    blank = df[df["brand"].isna() | (df["brand"].astype(str).str.strip() == "")]
    return blank.index.tolist()


def check_citation_ready(df: pd.DataFrame) -> dict:
    """Return summary of citation_ready status: total, ready, not_ready."""
    if "citation_ready" not in df.columns:
        return {"total": len(df), "ready": 0, "not_ready": len(df), "column_missing": True}
    ready = int(df["citation_ready"].fillna(False).sum())
    return {"total": len(df), "ready": ready, "not_ready": len(df) - ready, "column_missing": False}


def validate_provenance(df: pd.DataFrame, label: str = "") -> list[str]:
    """Run all provenance checks and return a list of warning strings.

    Does not raise — returns issues as strings for the caller to print.
    """
    issues = []
    label = label or "dataset"

    bad_url = check_source_url(df)
    if bad_url:
        issues.append(f"[{label}] source_url missing/blank: {len(bad_url)} row(s) → indices {bad_url[:5]}{'...' if len(bad_url) > 5 else ''}")

    bad_date = check_collection_date(df)
    if bad_date:
        issues.append(f"[{label}] collection_date missing: {len(bad_date)} row(s)")

    bad_brand = check_brand(df)
    if bad_brand:
        issues.append(f"[{label}] brand missing/blank: {len(bad_brand)} row(s)")

    citation = check_citation_ready(df)
    if citation.get("column_missing"):
        issues.append(f"[{label}] citation_ready column absent — add and populate before final submission")
    elif citation["not_ready"] > 0:
        issues.append(f"[{label}] citation_ready=False: {citation['not_ready']}/{citation['total']} row(s) not yet verified")

    return issues


# ── Reporting ──────────────────────────────────────────────────────────────────

def validate_and_report(
    df: pd.DataFrame,
    schema_name: str,
    dataset_label: str = "",
    raise_on_fail: bool = False,
) -> pd.DataFrame:
    """Validate df against schema, print a report, and return df unchanged.

    If raise_on_fail=True, raises ValueError on hard failures (not warnings).
    """
    label = dataset_label or schema_name
    passed, errors = validate(df, schema_name)
    prov_issues = validate_provenance(df, label)

    print(f"\n{'-' * 60}")
    print(f"  {label}  ({len(df)} rows)")
    print(f"{'-' * 60}")

    if not errors and not prov_issues:
        print("  [OK] PASS -- schema and provenance checks clear.")
    else:
        for e in errors:
            tag = "  [WARN]" if e.startswith("WARNING") else "  [FAIL]"
            print(f"{tag} {e}")
        for p in prov_issues:
            print(f"  [PROV] {p}")

    status = "PASS" if (passed and not prov_issues) else ("WARN" if passed else "FAIL")
    print(f"  --> Status: {status}")

    if not passed and raise_on_fail:
        raise ValueError(f"Schema validation failed for '{label}'. Fix FAIL errors above.")

    return df


def validate_all_files(
    data_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Scan a directory for CSV files, infer schemas, and validate each.

    Returns a dict mapping filename → {'schema', 'rows', 'passed', 'errors', 'prov_issues'}.
    """
    data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / "data" / "raw"
    results = {}

    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        print(f"[validation] No CSV files found in {data_dir}")
        return results

    for csv_path in csv_files:
        schema_name = guess_schema(csv_path.name)
        rel_path = csv_path.relative_to(PROJECT_ROOT)

        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:
            results[str(rel_path)] = {"schema": schema_name, "rows": 0, "passed": False,
                                       "errors": [f"Could not read file: {exc}"], "prov_issues": []}
            continue

        if df.empty:
            results[str(rel_path)] = {"schema": schema_name, "rows": 0, "passed": True,
                                       "errors": [], "prov_issues": [], "note": "Empty template — no rows to validate"}
            if verbose:
                print(f"  [EMPTY] {rel_path} — template ready, 0 rows")
            continue

        if schema_name:
            passed, errors = validate(df, schema_name)
        else:
            passed, errors = True, [f"WARNING: No schema matched for '{csv_path.name}' — skipping schema check"]

        prov_issues = validate_provenance(df, csv_path.name)

        results[str(rel_path)] = {
            "schema": schema_name,
            "rows": len(df),
            "passed": passed and not any("FAIL" in p for p in prov_issues),
            "errors": errors,
            "prov_issues": prov_issues,
        }

        if verbose:
            status = "OK" if results[str(rel_path)]["passed"] else "!!"
            print(f"  [{status}] {rel_path} ({len(df)} rows, schema={schema_name or 'unknown'})")
            for e in errors:
                print(f"       {e}")
            for p in prov_issues:
                print(f"       {p}")

    return results


def print_summary_report(results: dict) -> None:
    """Print a clean one-line-per-file summary from validate_all_files output."""
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)

    total = len(results)
    empty = sum(1 for r in results.values() if r.get("rows", -1) == 0)
    passed = sum(1 for r in results.values() if r.get("passed") and r.get("rows", 0) > 0)

    for path, r in results.items():
        rows = r.get("rows", "?")
        schema = r.get("schema") or "unknown"
        note = r.get("note", "")
        if rows == 0:
            status = "EMPTY"
        elif r.get("passed"):
            status = "PASS "
        else:
            status = "FAIL "
        print(f"  {status}  {path:<55} ({rows} rows, {schema}){' — ' + note if note else ''}")

    print("-" * 70)
    with_data = total - empty
    print(f"  Files checked : {total}  ({empty} empty templates, {with_data} with data)")
    print(f"  Passed        : {passed} / {with_data}")
    print(f"  Needs fixes   : {with_data - passed}")
    print("=" * 70 + "\n")
