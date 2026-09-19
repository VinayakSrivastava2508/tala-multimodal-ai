"""Schema validation for DataFrames — checks columns, types, values, and provenance."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Deterministic exact-filename-stem -> schema routing. Checked BEFORE the legacy
# substring inference below, so a Day 2 file whose stem happens to start with a Day 1
# schema name (e.g. "official_claims_corpus" starts with "official_claims") is never
# misrouted to the Day 1 schema.
EXACT_FILENAME_SCHEMA: Dict[str, str] = {
    "official_claims_corpus":      "official_claims_corpus",
    "customer_experience_corpus":  "customer_experience_corpus",
    "creator_strategy_corpus":      "creator_strategy_corpus",
    "creator_posts_enriched":      "creator_posts_enriched",
    "platform_strategy_enriched":  "platform_strategy_enriched",
    "hydrated_sources":              "hydrated_sources",
    "image_assets":                    "image_assets",
    "video_assets":                    "video_assets",
    "video_frame_features":            "video_frame_features",
    "video_level_features":            "video_level_features",
    "multimodal_reference_assets":    "multimodal_reference_assets",
    "claim_multimodal_evidence_candidates": "claim_multimodal_evidence_candidates",
    "reference_document_chunks":      "reference_document_chunks",
    "reference_tables":                "reference_tables",
    "reference_images":                "reference_images",
}

# Legacy filename PREFIX inference for Day 1 files. Only consulted when no exact
# match is found above.
FILENAME_TO_SCHEMA: Dict[str, str] = {
    "social_posts":           "social_posts",
    "creator_posts":          "creator_posts",
    "customer_reviews":       "customer_reviews",
    "official_claims":        "official_claims",
    "competitor_platforms":   "competitor_platforms",
    "press_reddit_sources":   "press_reddit_sources",
    "claim_assessments":      "claim_assessments",
    "image_assets":                    "image_assets",
    "video_assets":                    "video_assets",
    "video_frame_features":            "video_frame_features",
    "video_level_features":            "video_level_features",
    "multimodal_reference_assets":    "multimodal_reference_assets",
    "claim_multimodal_evidence_candidates": "claim_multimodal_evidence_candidates",
    "reference_document_chunks":      "reference_document_chunks",
    "reference_tables":                "reference_tables",
    "reference_images":                "reference_images",
}

# Provenance fields every dataset must carry
PROVENANCE_FIELDS = ["source_url", "source_platform", "collection_date", "collected_by", "evidence_type"]

RAG_CORPUS_SCHEMAS: Dict[str, str] = {
    "official_claims_corpus":      "official_claims",
    "customer_experience_corpus":  "customer_experience",
    "creator_strategy_corpus":      "creator_strategy",
}

CREATOR_PARTNERSHIP_VALUES = {
    "unknown", "unclear", "organic", "gifted", "paid_sponsorship", "ambassador", "affiliate",
}

DIRECT_SOCIAL_URL_RX = re.compile(
    r"instagram\.com/(?:p|reel)/[A-Za-z0-9_-]+"
    r"|tiktok\.com/@[^/]+/video/\d+"
    r"|youtube\.com/watch\?v=[\w-]+"
    r"|youtube\.com/shorts/[\w-]+"
    r"|youtu\.be/[\w-]+",
    re.I,
)

SEARCH_ENGINE_URL_RX = re.compile(r"google\.|duckduckgo\.", re.I)


def load_schema() -> dict:
    """Return parsed schema.yaml as a dict."""
    with open(PROJECT_ROOT / "configs" / "schema.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def guess_schema(filename: str) -> str | None:
    """Return the schema name for a CSV filename: exact stem match first, then
    legacy Day 1 prefix inference as a fallback. Returns None if no match.
    """
    stem = Path(filename).stem.lower()
    if stem in EXACT_FILENAME_SCHEMA:
        return EXACT_FILENAME_SCHEMA[stem]
    for pattern, schema in FILENAME_TO_SCHEMA.items():
        if stem.startswith(pattern):
            return schema
    return None


# ── Core field validation ──────────────────────────────────────────────────────

FAIL_PREFIXES = ("MISSING", "NULL", "INVALID", "BELOW", "ABOVE", "DUPLICATE", "SEMANTIC")


def assert_no_duplicate_ids(df: pd.DataFrame, id_col: str, label: str = "") -> None:
    """Raise ValueError if `id_col` contains any duplicate value. Used as a hard
    gate before writing a modelling table -- a feature table must have exactly one
    row per record id."""
    dup_n = int(df[id_col].duplicated().sum())
    if dup_n:
        raise ValueError(f"{label or id_col}: {dup_n} duplicate {id_col} value(s) -- refusing to write")


def assert_evidence_strength_allowed(
    df: pd.DataFrame, allowed: tuple = ("strong", "medium"), col: str = "evidence_strength", label: str = "",
) -> None:
    """Raise ValueError if any row's `col` is outside `allowed`. Used as a hard
    gate so weak/unusable evidence can never enter a modelling/feature table."""
    if col not in df.columns:
        return
    bad = ~df[col].isin(allowed)
    if bad.any():
        raise ValueError(
            f"{label or col}: {int(bad.sum())} row(s) have {col} outside {allowed} -- "
            "weak/unusable evidence must never enter the modelling table"
        )


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

    if schema_name in RAG_CORPUS_SCHEMAS:
        errors.extend(validate_rag_corpus(df, schema_name))
    elif schema_name == "creator_posts_enriched":
        errors.extend(validate_creator_posts_enriched(df))

    if "synthetic" in df.columns:
        n = int(df["synthetic"].fillna(False).sum())
        if n:
            errors.append(f"WARNING: {n} row(s) marked synthetic=True — exclude from final analysis")

    passed = all(not e.startswith(FAIL_PREFIXES) for e in errors)
    return passed, errors


# ── Semantic validation: RAG corpora ─────────────────────────────────────────────

def _blank_mask(series: pd.Series) -> pd.Series:
    """Return boolean mask of null-or-blank-string values in a Series."""
    return series.isna() | (series.astype(str).str.strip() == "")


def _bool_mask(series: pd.Series) -> pd.Series:
    """Coerce a bool-like column (bool dtype or 'True'/'False' strings) to bool."""
    return series.map(lambda v: str(v).strip().lower() == "true")


def validate_rag_corpus(df: pd.DataFrame, schema_name: str) -> list[str]:
    """Cross-field semantic checks for the three RAG corpus schemas.

    Enforces: unique non-blank document_id, corpus name matches the schema,
    source_url is URL-like, extracted_text is non-empty when rag_usable=True,
    weak/unusable rows cannot be rag_usable=True, and (official_claims_corpus only)
    each row must carry claim text via claim_text or extracted_text.
    """
    errors: list[str] = []

    if "document_id" in df.columns:
        blank = _blank_mask(df["document_id"])
        if blank.any():
            errors.append(f"SEMANTIC: document_id blank: {int(blank.sum())} row(s)")
        dup_n = int(df["document_id"].dropna().duplicated().sum())
        if dup_n:
            errors.append(f"DUPLICATE document_id: {dup_n} row(s) share an id with an earlier row")

    if "source_url" in df.columns:
        bad = df["source_url"].isna() | ~df["source_url"].astype(str).str.match(r"^https?://", na=False)
        n = int(bad.sum())
        if n:
            errors.append(f"SEMANTIC: source_url missing or not URL-like (http/https): {n} row(s)")

    rag_bool = _bool_mask(df["rag_usable"]) if "rag_usable" in df.columns else None
    strength = df["evidence_strength"] if "evidence_strength" in df.columns else None

    if rag_bool is not None and "extracted_text" in df.columns:
        empty_text = _blank_mask(df["extracted_text"]) | (df["extracted_text"].astype(str).str.lower() == "nan")
        bad = empty_text & rag_bool
        n = int(bad.sum())
        if n:
            errors.append(f"SEMANTIC: extracted_text empty while rag_usable=True: {n} row(s)")

    if rag_bool is not None and strength is not None:
        bad = strength.isin(["weak", "unusable"]) & rag_bool
        n = int(bad.sum())
        if n:
            errors.append(f"SEMANTIC: evidence_strength weak/unusable but rag_usable=True: {n} row(s)")

    if schema_name == "official_claims_corpus":
        text_cols = [c for c in ("claim_text", "extracted_text") if c in df.columns]
        if text_cols:
            has_any = pd.Series(False, index=df.index)
            for c in text_cols:
                has_any = has_any | (~_blank_mask(df[c]) & (df[c].astype(str).str.lower() != "nan"))
            n = int((~has_any).sum())
            if n:
                errors.append(
                    f"SEMANTIC: official_claims_corpus row has neither claim_text nor extracted_text: {n} row(s)"
                )
        if strength is not None and "provenance_note" in df.columns:
            strong_rows = strength == "strong"
            unsupported = strong_rows & ~df["provenance_note"].astype(str).str.contains(
                "strength_source=source_page", na=False
            )
            n = int(unsupported.sum())
            if n:
                errors.append(
                    "SEMANTIC: official_claims_corpus rows marked 'strong' without provenance tying that "
                    f"strength to a successfully hydrated source page: {n} row(s)"
                )

    return errors


# ── Semantic validation: creator_posts_enriched ──────────────────────────────────

def validate_creator_posts_enriched(df: pd.DataFrame) -> list[str]:
    """Cross-field semantic checks for the creator_posts_enriched schema.

    A row counts as a VERIFIED creator post only if usable_as_creator_post=True AND
    it resolves (via post_url, falling back to source_url) to a direct Instagram,
    TikTok, or YouTube post/video URL — never a search-result, profile, or campaign
    page. Missing follower/engagement metrics never fail validation.
    """
    errors: list[str] = []
    if "usable_as_creator_post" not in df.columns:
        return errors

    usable = _bool_mask(df["usable_as_creator_post"])

    url_cols = [c for c in ("post_url", "source_url") if c in df.columns]

    def _effective_url(row: pd.Series) -> str:
        for c in url_cols:
            val = row.get(c)
            if pd.notna(val) and str(val).strip():
                return str(val)
        return ""

    urls = df.apply(_effective_url, axis=1) if url_cols else pd.Series([""] * len(df), index=df.index)

    not_direct = ~urls.apply(lambda u: bool(DIRECT_SOCIAL_URL_RX.search(u)))
    bad_url = usable & not_direct
    n = int(bad_url.sum())
    if n:
        errors.append(
            f"SEMANTIC: usable_as_creator_post=True without a direct Instagram/TikTok/YouTube post URL: {n} row(s)"
        )

    is_search = usable & urls.str.contains(SEARCH_ENGINE_URL_RX, na=False)
    n = int(is_search.sum())
    if n:
        errors.append(f"SEMANTIC: usable_as_creator_post=True on a search-engine result URL: {n} row(s)")

    if "creator_handle" in df.columns:
        blank_handle = usable & _blank_mask(df["creator_handle"])
        n = int(blank_handle.sum())
        if n:
            errors.append(f"SEMANTIC: creator_handle blank for usable_as_creator_post=True: {n} row(s)")

    for col in ("brand", "platform"):
        if col in df.columns:
            n = int(_blank_mask(df[col]).sum())
            if n:
                errors.append(f"SEMANTIC: {col} blank: {n} row(s)")

    if "evidence_strength" in df.columns:
        blank_strength = usable & df["evidence_strength"].isna()
        n = int(blank_strength.sum())
        if n:
            errors.append(f"SEMANTIC: evidence_strength missing for usable_as_creator_post=True: {n} row(s)")

    if "brand_link_verified" in df.columns:
        verified_bool = _bool_mask(df["brand_link_verified"])
        unverified = usable & ~verified_bool
        n = int(unverified.sum())
        if n:
            errors.append(f"SEMANTIC: brand_link_verified is not True for usable_as_creator_post=True: {n} row(s)")

    if "direct_post_id" in df.columns and "platform" in df.columns:
        key = df["platform"].astype(str).str.lower() + "::" + df["direct_post_id"].astype(str)
        usable_keys = key[usable & ~_blank_mask(df["direct_post_id"])]
        dup_n = int(usable_keys.duplicated().sum())
        if dup_n:
            errors.append(
                f"DUPLICATE platform+direct_post_id among usable_as_creator_post=True rows: {dup_n} row(s)"
            )

    return errors


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
        issues.append(f"[{label}] source_url missing/blank: {len(bad_url)} row(s) -- indices {bad_url[:5]}{'...' if len(bad_url) > 5 else ''}")

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

    # "archive" directories hold pre-repair backup snapshots (see
    # scripts/repair_day2_creator_evidence.py) -- historical copies of a file's OLD
    # shape, kept for audit purposes. They're intentionally excluded from schema
    # validation: they don't reflect current pipeline output and re-validating them
    # against a schema built for the CURRENT shape would be comparing the file to a
    # standard it predates, not a real data-quality issue.
    csv_files = sorted(
        p for p in data_dir.rglob("*.csv") if "archive" not in p.relative_to(PROJECT_ROOT).parts
    )
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
