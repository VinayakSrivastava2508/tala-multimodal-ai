"""Tests for scripts/validate_synthesis_outputs.py -- the dedicated Day 3C
output-table validator (acceptance-patch item 4).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.validate_synthesis_outputs import (  # noqa: E402
    DAY3C_TABLES,
    check_file,
    TABLES_DIR,
)
from validation import guess_schema  # noqa: E402


def test_validator_reports_19_files_all_passing():
    """The real, current outputs/tables/ directory must pass all 19 checks."""
    for filename, schema_name in DAY3C_TABLES.items():
        errors = check_file(filename, schema_name)
        assert not errors, f"{filename} failed: {errors}"
    assert len(DAY3C_TABLES) == 19


def test_all_19_filenames_route_to_expected_schema():
    for filename, schema_name in DAY3C_TABLES.items():
        assert guess_schema(filename) == schema_name, f"{filename} did not route to {schema_name}"


@pytest.fixture
def scratch_table(tmp_path):
    """Copy a real Day 3C table into a temp dir so it can be corrupted and
    checked without touching the real outputs/tables/ files."""

    def _copy(filename: str) -> Path:
        dest = tmp_path / filename
        shutil.copy(TABLES_DIR / filename, dest)
        return dest

    return _copy


def _check_corrupted(monkeypatch, tmp_path, filename, schema_name, mutate_fn):
    """Run check_file against a corrupted copy by monkeypatching TABLES_DIR."""
    import scripts.validate_synthesis_outputs as mod

    df = pd.read_csv(TABLES_DIR / filename)
    df = mutate_fn(df)
    corrupted_dir = tmp_path / "corrupted"
    corrupted_dir.mkdir(exist_ok=True)
    df.to_csv(corrupted_dir / filename, index=False)

    monkeypatch.setattr(mod, "TABLES_DIR", corrupted_dir)
    return mod.check_file(filename, schema_name)


def test_missing_required_column_fails(monkeypatch, tmp_path):
    errors = _check_corrupted(
        monkeypatch, tmp_path, "analytical_synthesis_claim_master.csv", "analytical_synthesis_claim_master",
        lambda df: df.drop(columns=["claim_text"]),
    )
    assert any("MISSING COLUMN" in e for e in errors)


def test_duplicate_ids_fail(monkeypatch, tmp_path):
    def mutate(df):
        dup_row = df.iloc[[0]].copy()
        return pd.concat([df, dup_row], ignore_index=True)

    errors = _check_corrupted(monkeypatch, tmp_path, "analytical_synthesis_claim_master.csv", "analytical_synthesis_claim_master", mutate)
    assert any("DUPLICATE PRIMARY ID" in e for e in errors)


def test_unresolved_evidence_id_fails(monkeypatch, tmp_path):
    def mutate(df):
        df = df.copy()
        df.loc[0, "strongest_supporting_evidence_id"] = "eu_does_not_exist_12345"
        return df

    errors = _check_corrupted(monkeypatch, tmp_path, "analytical_synthesis_claim_master.csv", "analytical_synthesis_claim_master", mutate)
    assert any("EVIDENCE ID RESOLUTION" in e for e in errors)


def test_incorrect_percentage_range_fails(monkeypatch, tmp_path):
    def mutate(df):
        df = df.copy()
        df.loc[0, "pct_within_brand"] = 250.0
        return df

    errors = _check_corrupted(monkeypatch, tmp_path, "creator_partnership_mix.csv", "creator_partnership_mix", mutate)
    assert any("PERCENTAGE RANGE" in e for e in errors)


def test_incorrect_claim_count_fails(monkeypatch, tmp_path):
    def mutate(df):
        return df.iloc[:-1]  # drop one row -> only 36 claims

    errors = _check_corrupted(monkeypatch, tmp_path, "analytical_synthesis_claim_master.csv", "analytical_synthesis_claim_master", mutate)
    assert any("CLAIM COUNT" in e for e in errors)


def test_incorrect_mixed_theme_count_fails(monkeypatch, tmp_path):
    def mutate(df):
        extra = df.iloc[[0]].copy()
        extra["theme_id"] = "theme_03"
        extra["claim_ids"] = "c_OC_9999_00"
        return pd.concat([df, extra], ignore_index=True)

    errors = _check_corrupted(monkeypatch, tmp_path, "mixed_claim_theme_summary.csv", "mixed_claim_theme_summary", mutate)
    assert any("MIXED THEME COUNT" in e for e in errors)


def test_theme_membership_duplication_fails(monkeypatch, tmp_path):
    def mutate(df):
        df = df.copy()
        # put the same claim_id into both theme rows
        first_claim = df.iloc[0]["claim_ids"].split(";")[0]
        df.loc[1, "claim_ids"] = df.loc[1, "claim_ids"] + ";" + first_claim
        df.loc[1, "claim_record_count"] = len(df.loc[1, "claim_ids"].split(";"))
        return df

    errors = _check_corrupted(monkeypatch, tmp_path, "mixed_claim_theme_summary.csv", "mixed_claim_theme_summary", mutate)
    assert any("THEME MEMBERSHIP DUPLICATION" in e for e in errors)


def test_inaccurate_tala_most_organic_led_assertion_fails(monkeypatch, tmp_path):
    def mutate(df):
        df = df.copy()
        df.loc[0, "observed_pattern"] = "TALA is the most organic-led brand in the corpus."
        return df

    errors = _check_corrupted(monkeypatch, tmp_path, "creator_moat_hypotheses.csv", "creator_moat_hypotheses", mutate)
    assert any("ORGANIC-LED ACCURACY" in e for e in errors)


def test_legitimate_negation_of_organic_led_does_not_fail():
    """A correctly-worded negation (as actually used in the real tables) must
    not be flagged -- only an unqualified affirmative claim should fail."""
    df = pd.read_csv(TABLES_DIR / "creator_strategy_interpretation.csv")
    blob = " ".join(df.fillna("").astype(str).values.flatten()).lower()
    assert "most organic-led" not in blob or "not described here as 'more organic-led'" in blob


def test_no_proven_moat_language_in_real_tables():
    errors = check_file("creator_moat_hypotheses.csv", "creator_moat_hypotheses")
    assert not any("MOAT LANGUAGE" in e for e in errors)
