"""Schema validation for DataFrames loaded from data/raw/."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_schema() -> dict:
    """Return parsed schema.yaml."""
    with open(PROJECT_ROOT / "configs" / "schema.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate(df: pd.DataFrame, schema_name: str) -> Tuple[bool, list[str]]:
    """Validate a DataFrame against a named schema from configs/schema.yaml.

    Returns (passed: bool, errors: list[str]).
    """
    schema = load_schema()
    if schema_name not in schema["schemas"]:
        return False, [f"Unknown schema: {schema_name}"]

    fields = schema["schemas"][schema_name]["fields"]
    errors: list[str] = []

    # Check required columns exist
    for col, spec in fields.items():
        if col not in df.columns:
            errors.append(f"Missing column: '{col}'")
            continue

        if spec.get("required") and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            errors.append(f"Column '{col}' has {null_count} null values (required)")

        if "values" in spec and df[col].notna().any():
            allowed = set(spec["values"])
            actual = set(df[col].dropna().unique())
            invalid = actual - allowed
            if invalid:
                errors.append(f"Column '{col}' contains invalid values: {invalid}. Allowed: {allowed}")

        if "min" in spec and pd.api.types.is_numeric_dtype(df[col]):
            below = (df[col] < spec["min"]).sum()
            if below:
                errors.append(f"Column '{col}' has {below} values below minimum {spec['min']}")

        if "max" in spec and pd.api.types.is_numeric_dtype(df[col]):
            above = (df[col] > spec["max"]).sum()
            if above:
                errors.append(f"Column '{col}' has {above} values above maximum {spec['max']}")

    # Check for synthetic flag
    if "synthetic" in df.columns and df["synthetic"].any():
        synthetic_count = df["synthetic"].sum()
        errors.append(
            f"WARNING: {synthetic_count} rows are marked synthetic=True. "
            "Do not include in final analysis without clear labelling."
        )

    passed = len([e for e in errors if not e.startswith("WARNING")]) == 0
    return passed, errors


def validate_and_report(df: pd.DataFrame, schema_name: str, dataset_label: str = "") -> pd.DataFrame:
    """Validate df, print a report, and return the df unchanged.

    Raises ValueError if hard validation fails.
    """
    passed, errors = validate(df, schema_name)
    label = dataset_label or schema_name
    print(f"\n── Validation: {label} ({len(df)} rows) ──")
    if passed and not errors:
        print("  PASS — no issues found.")
    else:
        for e in errors:
            prefix = "  [WARN]" if e.startswith("WARNING") else "  [FAIL]"
            print(f"{prefix} {e}")
        if not passed:
            raise ValueError(f"Schema validation failed for '{label}'. Fix errors above before proceeding.")
    return df


def check_source_ids(df: pd.DataFrame, source_log_path: str | None = None) -> list[str]:
    """Check that all source_id values in df have corresponding entries in the source log.

    Returns list of unresolved source_ids. Prints a warning if source log not found.
    """
    if "source_id" not in df.columns:
        return []

    source_ids = df["source_id"].dropna().unique().tolist()

    if source_log_path is None:
        source_log_path = PROJECT_ROOT / "docs" / "source_log_template.md"

    if not Path(source_log_path).exists():
        print("[validation] Source log not found; skipping source_id check.")
        return []

    with open(source_log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    unresolved = [sid for sid in source_ids if sid not in log_content]
    if unresolved:
        print(f"[validation] Unresolved source_ids (add to source_log_template.md): {unresolved}")
    return unresolved
