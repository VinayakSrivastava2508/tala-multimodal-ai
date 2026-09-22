"""Direct validation for the 19 Day 3C output tables in outputs/tables/.

scripts/validate_data.py only scans data/raw, data/interim, and data/corpora --
it never touches outputs/, so the 18 (now 19) Day 3C schemas registered in
configs/schema.yaml were never actually exercised by that run. This script
closes that gap: it reuses src/validation.py's existing schema-routing
(guess_schema) and column/type/value validation (validate()) rather than
re-implementing them, and adds the Day 3C-specific semantic checks listed in
the acceptance-patch brief (claim counts, mixed-theme consistency, evidence-id
resolution, brand canonicalisation, denominator correctness, moat-language
and scope-boundary checks).

Exits non-zero if any of the 19 tables fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from validation import validate, guess_schema  # noqa: E402

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FUSION_DIR = PROJECT_ROOT / "data" / "processed" / "fusion"

CANONICAL_BRANDS = {"TALA", "Adanola", "Girlfriend Collective", "Oner Active"}

# The 19 Day 3C tables (18 original + mixed_claim_theme_summary), mapped to the
# schema name they must route to (exact-filename routing, checked explicitly).
DAY3C_TABLES: dict[str, str] = {
    "analytical_synthesis_claim_master.csv": "analytical_synthesis_claim_master",
    "claim_label_summary.csv":                "claim_label_summary",
    "mixed_claim_deep_dive.csv":              "mixed_claim_deep_dive",
    "mixed_claim_theme_summary.csv":          "mixed_claim_theme_summary",
    "aligned_claim_examples.csv":              "aligned_claim_examples",
    "insufficient_evidence_analysis.csv":    "insufficient_evidence_analysis",
    "modality_contribution_summary.csv":      "modality_contribution_summary",
    "evidence_gap_by_category.csv":            "evidence_gap_by_category",
    "creator_strategy_evidence_coverage.csv": "creator_strategy_evidence_coverage",
    "creator_partnership_mix.csv":            "creator_partnership_mix",
    "creator_content_intent_mix.csv":          "creator_content_intent_mix",
    "creator_partnership_intent_matrix.csv":  "creator_partnership_intent_matrix",
    "brand_platform_strategy_comparison.csv": "brand_platform_strategy_comparison",
    "creator_strategic_pattern_metrics.csv":  "creator_strategic_pattern_metrics",
    "creator_strategy_interpretation.csv":    "creator_strategy_interpretation",
    "tala_competitor_lessons.csv":              "tala_competitor_lessons",
    "creator_moat_hypotheses.csv":              "creator_moat_hypotheses",
    "professor_feedback_response.csv":          "professor_feedback_response",
    "executive_finding_register.csv":            "executive_finding_register",
}

assert len(DAY3C_TABLES) == 19, f"expected 19 Day 3C tables, got {len(DAY3C_TABLES)}"

PROTECTED_TERMS = ["race", "ethnicity", "gender", "age_group", "religion", "disability", "sexual_orientation"]
ENGAGEMENT_PREDICTION_TERMS = ["predicted_engagement", "engagement_residual", "out_of_fold", "engagement_prediction", "engagement_rank", "performance_score"]
INACCURATE_ORGANIC_PHRASES = ["tala is the most organic-led", "tala is most organic-led", "most organic-led brand"]
INACCURATE_MIXED_PHRASES = ["five independent mixed", "five separate mixed", "five separate areas of contradiction"]
FORBIDDEN_MOAT_PHRASE = "proven moat"


def _pct_columns_in_range(df: pd.DataFrame) -> list[str]:
    errors = []
    for col in df.columns:
        if "pct" in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
            bad = df[(df[col] < 0) | (df[col] > 100)]
            if len(bad):
                errors.append(f"{col}: {len(bad)} value(s) outside [0, 100]")
    return errors


def _count_columns_nonnegative_int(df: pd.DataFrame) -> list[str]:
    errors = []
    for col in df.columns:
        low = col.lower()
        if (low == "count" or low.endswith("_count") or low.startswith("n_") or low == "n") and pd.api.types.is_numeric_dtype(df[col]):
            bad = df[df[col] < 0]
            if len(bad):
                errors.append(f"{col}: {len(bad)} negative value(s)")
            non_int = df[col].dropna().apply(lambda v: float(v) != int(v))
            if non_int.any():
                errors.append(f"{col}: {int(non_int.sum())} non-integer value(s)")
    return errors


def _no_forbidden_columns(df: pd.DataFrame, terms: list[str], label: str) -> list[str]:
    errors = []
    cols_lower = [c.lower() for c in df.columns]
    for term in terms:
        if any(term in c for c in cols_lower):
            errors.append(f"contains forbidden {label} column pattern '{term}'")
    return errors


def _no_forbidden_text(df: pd.DataFrame, phrases: list[str], allow_negation: bool = True) -> list[str]:
    """Flag any cell containing a forbidden phrase, unless it's a negated usage
    (e.g. 'not a proven moat', 'never...most organic-led')."""
    errors = []
    blob_cells = df.fillna("").astype(str)
    for col in blob_cells.columns:
        for idx, val in blob_cells[col].items():
            low = val.lower()
            for phrase in phrases:
                if phrase in low:
                    window_start = max(0, low.find(phrase) - 25)
                    window = low[window_start: low.find(phrase) + len(phrase)]
                    if allow_negation and any(neg in window for neg in ["not ", "never ", "cannot ", "no hypothesis", "isn't", "n't described"]):
                        continue
                    errors.append(f"{col}[{idx}]: forbidden phrase '{phrase}' found without negation")
    return errors


def check_file(filename: str, expected_schema: str) -> list[str]:
    errors: list[str] = []
    path = TABLES_DIR / filename

    if not path.exists():
        return [f"FILE MISSING: {path}"]

    routed = guess_schema(filename)
    if routed != expected_schema:
        errors.append(f"ROUTING: '{filename}' routed to schema '{routed}', expected '{expected_schema}'")

    df = pd.read_csv(path)

    ok, schema_errors = validate(df, expected_schema)
    hard_fail_prefixes = ("MISSING", "NULL", "INVALID", "BELOW", "ABOVE")
    errors.extend([e for e in schema_errors if e.startswith(hard_fail_prefixes)])

    errors.extend(f"PERCENTAGE RANGE: {e}" for e in _pct_columns_in_range(df))
    errors.extend(f"COUNT: {e}" for e in _count_columns_nonnegative_int(df))
    errors.extend(f"PROTECTED CHARACTERISTIC: {e}" for e in _no_forbidden_columns(df, PROTECTED_TERMS, "protected-characteristic"))
    errors.extend(f"ENGAGEMENT PREDICTION: {e}" for e in _no_forbidden_columns(df, ENGAGEMENT_PREDICTION_TERMS, "engagement-prediction"))
    errors.extend(f"MOAT LANGUAGE: {e}" for e in _no_forbidden_text(df, [FORBIDDEN_MOAT_PHRASE]))
    errors.extend(f"ORGANIC-LED ACCURACY: {e}" for e in _no_forbidden_text(df, INACCURATE_ORGANIC_PHRASES, allow_negation=False))
    errors.extend(f"MIXED-COUNT ACCURACY: {e}" for e in _no_forbidden_text(df, INACCURATE_MIXED_PHRASES, allow_negation=False))

    if "brand" in df.columns:
        # creator_partnership_intent_matrix.csv legitimately uses the sentinel
        # 'all' for its overall/non_organic_only scope rows (all brands combined,
        # not a per-brand row) -- not a canonicalisation failure.
        allowed_brands = CANONICAL_BRANDS | ({"all"} if filename == "creator_partnership_intent_matrix.csv" else set())
        bad_brands = set(df["brand"].dropna()) - allowed_brands
        if bad_brands:
            errors.append(f"BRAND CANONICALISATION: non-canonical brand values {bad_brands}")

    for id_col in ["claim_id", "hypothesis_id", "lesson_id", "finding_id", "theme_id"]:
        if id_col in df.columns and filename not in ("mixed_claim_deep_dive.csv", "creator_partnership_intent_matrix.csv"):
            # theme_id/claim_id can legitimately repeat as a foreign key in some
            # tables (e.g. mixed_claim_deep_dive.theme_id); only enforce
            # uniqueness where the column is that table's own primary id.
            pass
    if filename == "analytical_synthesis_claim_master.csv":
        if df["claim_id"].duplicated().any():
            errors.append("DUPLICATE PRIMARY ID: claim_id duplicated in analytical_synthesis_claim_master.csv")
        if df["claim_id"].nunique() != 37:
            errors.append(f"CLAIM COUNT: expected exactly 37 unique claims, found {df['claim_id'].nunique()}")
        n_mixed = int((df["production_fusion_label"] == "mixed").sum())
        if n_mixed != 5:
            errors.append(f"MIXED CLAIM COUNT: expected exactly 5 mixed claim records, found {n_mixed}")

    if filename == "mixed_claim_theme_summary.csv":
        if df["theme_id"].duplicated().any():
            errors.append("DUPLICATE PRIMARY ID: theme_id duplicated")
        if len(df) != 2:
            errors.append(f"MIXED THEME COUNT: expected exactly 2 mixed themes, found {len(df)}")
        total = df["claim_record_count"].sum()
        if total != 5:
            errors.append(f"MIXED-THEME CLAIM SUM: theme claim_record_count sums to {total}, expected 5")
        all_ids = [cid for ids in df["claim_ids"] for cid in str(ids).split(";")]
        if len(all_ids) != len(set(all_ids)):
            errors.append("THEME MEMBERSHIP DUPLICATION: a claim_id appears in more than one theme")
        if len(all_ids) != total:
            errors.append("THEME MEMBERSHIP: claim_ids count does not match claim_record_count sum")

    if filename in ("creator_partnership_mix.csv", "creator_content_intent_mix.csv"):
        denom_col = "brand_denominator_n"
        cat_col = "partnership_type" if "partnership_type" in df.columns else "content_intent"
        for brand, sub in df.groupby("brand"):
            pct_sum = (100 * sub["count"] / sub[denom_col]).sum() if "pct_within_brand" not in sub.columns else sub["pct_within_brand"].sum()
            if not (99.0 <= pct_sum <= 101.0):
                errors.append(f"DENOMINATOR/PERCENTAGE: {brand} {cat_col} shares sum to {pct_sum:.1f}, expected ~100")
        # zero-count categories retained: every brand must have every category present
        n_brands = df["brand"].nunique()
        n_cats = df[cat_col].nunique()
        if len(df) != n_brands * n_cats:
            errors.append(f"ZERO-COUNT RETENTION: expected {n_brands * n_cats} rows (brands x categories), found {len(df)} -- a zero-count category may have been dropped")

    if filename == "executive_finding_register.csv":
        for col in ["evidence_ids", "caveat"]:
            if col in df.columns:
                blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
                if blank.any():
                    errors.append(f"EXECUTIVE FINDINGS: {int(blank.sum())} row(s) missing {col}")

    if filename == "creator_moat_hypotheses.csv":
        allowed = {"plausible", "weak", "unsupported", "contradicted"}
        bad = set(df["status"].dropna()) - allowed
        if bad:
            errors.append(f"MOAT STATUS: disallowed status value(s) {bad}")

    if filename == "mixed_claim_deep_dive.csv":
        units = pd.read_csv(FUSION_DIR / "claim_evidence_units.csv")
        valid_ids = set(units["evidence_unit_id"].astype(str))
        for col in ["supporting_evidence_ids", "challenging_evidence_ids"]:
            if col in df.columns:
                for val in df[col].dropna():
                    for eid in str(val).split(";"):
                        if eid and eid not in valid_ids:
                            errors.append(f"EVIDENCE ID RESOLUTION: '{eid}' in {col} not found in claim_evidence_units.csv")

    if filename == "mixed_claim_theme_summary.csv":
        units = pd.read_csv(FUSION_DIR / "claim_evidence_units.csv")
        valid_ids = set(units["evidence_unit_id"].astype(str))
        for col in ["supporting_evidence_ids", "challenging_evidence_ids"]:
            for val in df[col].dropna():
                for eid in str(val).split(";"):
                    if eid and eid not in valid_ids:
                        errors.append(f"EVIDENCE ID RESOLUTION: '{eid}' in {col} not found in claim_evidence_units.csv")

    if filename == "analytical_synthesis_claim_master.csv":
        units = pd.read_csv(FUSION_DIR / "claim_evidence_units.csv")
        valid_ids = set(units["evidence_unit_id"].astype(str)) | {""}
        for col in ["strongest_supporting_evidence_id", "strongest_challenging_evidence_id", "strongest_reference_evidence_id"]:
            unresolved = set(df[col].fillna("").astype(str)) - valid_ids
            if unresolved:
                errors.append(f"EVIDENCE ID RESOLUTION: {col} has unresolved ids {unresolved}")

    return errors


def main() -> int:
    print("Day 3C synthesis outputs")
    results: dict[str, list[str]] = {}
    for filename, schema_name in DAY3C_TABLES.items():
        results[filename] = check_file(filename, schema_name)

    n_total = len(results)
    n_passed = sum(1 for errs in results.values() if not errs)
    n_failed = n_total - n_passed

    for filename, errs in results.items():
        status = "PASS" if not errs else "FAIL"
        print(f"  {status}   {filename}")
        for e in errs:
            print(f"         - {e}")

    print(f"Files checked: {n_total}")
    print(f"Passed: {n_passed}/{n_total}")
    print(f"Failed: {n_failed}")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
