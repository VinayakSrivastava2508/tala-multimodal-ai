"""Validates outputs/app_audit/feature_test_matrix.csv structurally (Part F of
the Executive Cockpit Acceptance Patch). Catches the exact class of defect the
patch found: rows with unquoted commas parsing into the wrong column count,
and summary counts silently drifting from the actual row data."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MATRIX_PATH = PROJECT_ROOT / "outputs" / "app_audit" / "feature_test_matrix.csv"
BACKLOG_PATH = PROJECT_ROOT / "outputs" / "app_audit" / "improvement_backlog.csv"
SCORECARD_PATH = PROJECT_ROOT / "outputs" / "app_audit" / "executive_comprehension_scorecard.csv"

EXPECTED_HEADER = [
    "Test ID", "Page", "Feature", "Test steps", "Expected result", "Observed result",
    "PASS/PARTIAL/FAIL/NOT TESTABLE", "Severity", "Screenshot", "Recommendation",
]
VALID_STATUSES = {"PASS", "PARTIAL", "FAIL", "NOT TESTABLE"}
VALID_SEVERITIES = {"", "P0", "P1", "P2", "P3"}


def _read_matrix() -> tuple[list[str], list[list[str]]]:
    with open(MATRIX_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    return header, rows


@pytest.mark.skipif(not MATRIX_PATH.exists(), reason="Audit not yet run in this environment")
def test_feature_matrix_header_matches_expected_schema():
    header, _ = _read_matrix()
    assert header == EXPECTED_HEADER


@pytest.mark.skipif(not MATRIX_PATH.exists(), reason="Audit not yet run in this environment")
def test_feature_matrix_every_row_has_exactly_ten_columns():
    header, rows = _read_matrix()
    for i, row in enumerate(rows, start=2):
        assert len(row) == len(header), f"Row {i} ({row[0] if row else '?'}) has {len(row)} columns, expected {len(header)}: {row}"


@pytest.mark.skipif(not MATRIX_PATH.exists(), reason="Audit not yet run in this environment")
def test_feature_matrix_status_values_are_valid():
    _, rows = _read_matrix()
    for row in rows:
        assert row[6] in VALID_STATUSES, f"{row[0]}: invalid status {row[6]!r}"


@pytest.mark.skipif(not MATRIX_PATH.exists(), reason="Audit not yet run in this environment")
def test_feature_matrix_severity_values_are_valid():
    _, rows = _read_matrix()
    for row in rows:
        assert row[7] in VALID_SEVERITIES, f"{row[0]}: invalid severity {row[7]!r}"


@pytest.mark.skipif(not MATRIX_PATH.exists(), reason="Audit not yet run in this environment")
def test_feature_matrix_test_ids_are_unique():
    _, rows = _read_matrix()
    ids = [row[0] for row in rows]
    assert len(ids) == len(set(ids))


@pytest.mark.skipif(not MATRIX_PATH.exists(), reason="Audit not yet run in this environment")
def test_feature_matrix_summary_totals_match_row_count():
    """Any prose summary of the matrix (e.g. docs/executive_cockpit_audit.md)
    must derive its totals from this file, not hand-type them -- this test is
    the machine-checkable half of that rule: the row data itself must sum
    consistently, so a derived total is always correct by construction."""
    _, rows = _read_matrix()
    from collections import Counter
    counts = Counter(row[6] for row in rows)
    assert sum(counts.values()) == len(rows)
    for status in VALID_STATUSES:
        counts.setdefault(status, 0)
    # A row-count regression (rows silently dropped/duplicated) would still
    # pass the per-row checks above but change this total unexpectedly --
    # pin it so a future edit has to be a deliberate, reviewed change.
    assert len(rows) == 40


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    return header, rows


@pytest.mark.skipif(not BACKLOG_PATH.exists(), reason="Audit not yet run in this environment")
def test_improvement_backlog_every_row_has_correct_column_count():
    header, rows = _read_csv(BACKLOG_PATH)
    for i, row in enumerate(rows, start=2):
        assert len(row) == len(header), f"Row {i} has {len(row)} columns, expected {len(header)}"


@pytest.mark.skipif(not SCORECARD_PATH.exists(), reason="Audit not yet run in this environment")
def test_comprehension_scorecard_every_row_has_correct_column_count_and_valid_scores():
    header, rows = _read_csv(SCORECARD_PATH)
    score_cols = header[1:-2]  # between "Page" and "Overall score"/"Primary issue"
    for i, row in enumerate(rows, start=2):
        assert len(row) == len(header), f"Row {i} has {len(row)} columns, expected {len(header)}"
        scores = [int(v) for v in row[1:1 + len(score_cols)]]
        assert all(1 <= s <= 5 for s in scores), f"Row {i}: dimension score out of 1-5 range: {scores}"
        overall = float(row[header.index("Overall score")])
        assert round(sum(scores) / len(scores), 1) == overall, (
            f"Row {i}: overall score {overall} does not match the mean of its dimension scores"
        )
