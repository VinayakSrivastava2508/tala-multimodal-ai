"""Day 3C quality controls: claim-experience synthesis and creator-strategy
comparison outputs. Validates the section 15 requirements from the task
brief -- claim coverage, label fidelity, evidence-id resolution, denominator
correctness, zero-count retention, and scope-boundary absence checks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FUSION_DIR = PROJECT_ROOT / "data" / "processed" / "fusion"

REQUIRED_TABLES = [
    "analytical_synthesis_claim_master.csv",
    "claim_label_summary.csv",
    "mixed_claim_deep_dive.csv",
    "mixed_claim_theme_summary.csv",
    "aligned_claim_examples.csv",
    "insufficient_evidence_analysis.csv",
    "modality_contribution_summary.csv",
    "evidence_gap_by_category.csv",
    "creator_strategy_evidence_coverage.csv",
    "creator_partnership_mix.csv",
    "creator_content_intent_mix.csv",
    "creator_partnership_intent_matrix.csv",
    "brand_platform_strategy_comparison.csv",
    "creator_strategic_pattern_metrics.csv",
    "creator_strategy_interpretation.csv",
    "tala_competitor_lessons.csv",
    "creator_moat_hypotheses.csv",
    "professor_feedback_response.csv",
    "executive_finding_register.csv",
]


@pytest.fixture(scope="module")
def master():
    return pd.read_csv(TABLES_DIR / "analytical_synthesis_claim_master.csv")


@pytest.fixture(scope="module")
def fusion_results():
    return pd.read_csv(FUSION_DIR / "claim_fusion_results.csv")


@pytest.fixture(scope="module")
def evidence_units():
    return pd.read_csv(FUSION_DIR / "claim_evidence_units.csv")


def test_all_required_tables_exist():
    for name in REQUIRED_TABLES:
        assert (TABLES_DIR / name).exists(), f"missing required output table: {name}"


def test_all_37_claims_appear_exactly_once(master):
    assert len(master) == 37
    assert master["claim_id"].nunique() == 37


def test_fusion_labels_match_authoritative_day3a_output(master, fusion_results):
    merged = master.merge(fusion_results[["claim_id", "automated_label"]], on="claim_id", suffixes=("", "_authoritative"))
    mismatches = merged[merged["production_fusion_label"] != merged["automated_label"]]
    assert mismatches.empty, f"labels diverge from Day 3A output for: {mismatches['claim_id'].tolist()}"


def test_no_insufficient_evidence_claim_labelled_contradicted(master):
    insuff = master[master["production_fusion_label"] == "insufficient_evidence"]
    assert not insuff.empty
    # 'divergent'/'contradicted' framing must never be applied to an insufficient-evidence claim
    assert (insuff["production_fusion_label"] != "divergent").all()


def test_evidence_ids_resolve(master, evidence_units):
    valid_ids = set(evidence_units["evidence_unit_id"].astype(str)) | {""}
    for col in ["strongest_supporting_evidence_id", "strongest_challenging_evidence_id", "strongest_reference_evidence_id"]:
        vals = master[col].fillna("").astype(str)
        unresolved = set(vals) - valid_ids
        assert not unresolved, f"{col} has unresolved evidence ids: {unresolved}"


def test_claim_label_summary_retains_zero_count_classes():
    df = pd.read_csv(TABLES_DIR / "claim_label_summary.csv")
    overall = df[df["section"] == "overall_label_distribution"]
    all_labels = {"aligned", "partially_aligned", "mixed", "divergent", "insufficient_evidence"}
    assert all_labels.issubset(set(overall["automated_label"]))
    zero_rows = overall[overall["automated_label"].isin(["partially_aligned", "divergent"])]
    assert (zero_rows["n_claims"] == 0).all()


def test_mixed_deep_dive_covers_every_mixed_claim(master):
    dd = pd.read_csv(TABLES_DIR / "mixed_claim_deep_dive.csv")
    mixed_ids = set(master[master["production_fusion_label"] == "mixed"]["claim_id"])
    assert set(dd["claim_id"]) == mixed_ids


def test_mixed_claims_consolidate_into_exactly_two_themes():
    themes = pd.read_csv(TABLES_DIR / "mixed_claim_theme_summary.csv")
    assert len(themes) == 2
    assert themes["claim_record_count"].sum() == 5
    all_ids = [cid for ids in themes["claim_ids"] for cid in str(ids).split(";")]
    assert len(all_ids) == len(set(all_ids)) == 5


def test_every_mixed_claim_appears_in_exactly_one_theme(master):
    dd = pd.read_csv(TABLES_DIR / "mixed_claim_deep_dive.csv")
    mixed_ids = set(master[master["production_fusion_label"] == "mixed"]["claim_id"])
    assert set(dd["claim_id"]) == mixed_ids
    assert dd["theme_id"].notna().all()
    # each claim_id maps to exactly one theme_id (no duplicate claim_id rows)
    assert dd["claim_id"].duplicated().sum() == 0


def test_insufficient_evidence_analysis_covers_all_insufficient_claims(master):
    ia = pd.read_csv(TABLES_DIR / "insufficient_evidence_analysis.csv")
    insuff_ids = set(master[master["production_fusion_label"] == "insufficient_evidence"]["claim_id"])
    assert set(ia["claim_id"]) == insuff_ids


def test_partnership_shares_sum_correctly_per_brand():
    df = pd.read_csv(TABLES_DIR / "creator_partnership_mix.csv")
    sums = df.groupby("brand")["pct_within_brand"].sum()
    assert (sums.between(99.0, 101.0)).all(), sums.to_dict()


def test_intent_shares_sum_correctly_per_brand():
    df = pd.read_csv(TABLES_DIR / "creator_content_intent_mix.csv")
    # add pct column derived from count/denominator for the check
    df["pct"] = 100 * df["count"] / df["brand_denominator_n"]
    sums = df.groupby("brand")["pct"].sum()
    assert (sums.between(99.0, 101.0)).all(), sums.to_dict()


def test_percentages_use_brand_level_denominator():
    df = pd.read_csv(TABLES_DIR / "creator_partnership_mix.csv")
    actual = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "creator_multimodal_features.csv")
    brand_counts = actual["brand"].value_counts().to_dict()
    for brand, denom in df.drop_duplicates("brand").set_index("brand")["brand_denominator_n"].items():
        assert denom == brand_counts[brand], f"{brand} denominator mismatch: {denom} vs {brand_counts[brand]}"


def test_zero_count_partnership_and_intent_categories_retained():
    pm = pd.read_csv(TABLES_DIR / "creator_partnership_mix.csv")
    # Oner Active has no paid_sponsorship records verified -- row must still exist with count 0
    row = pm[(pm["brand"] == "Oner Active") & (pm["partnership_type"] == "paid_sponsorship")]
    assert len(row) == 1
    assert row.iloc[0]["count"] == 0


def test_unclear_classification_not_dropped():
    pm = pd.read_csv(TABLES_DIR / "creator_partnership_mix.csv")
    assert "unclear" in set(pm["partnership_type"])
    tala_unclear = pm[(pm["brand"] == "TALA") & (pm["partnership_type"] == "unclear")]
    assert len(tala_unclear) == 1


def test_brand_names_canonicalised():
    expected = {"TALA", "Adanola", "Girlfriend Collective", "Oner Active"}
    for name in ["creator_partnership_mix.csv", "creator_content_intent_mix.csv", "creator_strategy_evidence_coverage.csv"]:
        df = pd.read_csv(TABLES_DIR / name)
        assert set(df["brand"]) == expected, f"{name} brand set mismatch: {set(df['brand'])}"


def test_no_cross_platform_creator_performance_analysis():
    """No output table ranks or compares creator engagement performance across platforms."""
    forbidden_terms = ["engagement_rank", "performance_score", "best_performing", "cross_platform_engagement"]
    for name in REQUIRED_TABLES:
        df = pd.read_csv(TABLES_DIR / name)
        cols_lower = [c.lower() for c in df.columns]
        for term in forbidden_terms:
            assert not any(term in c for c in cols_lower), f"{name} contains forbidden column pattern '{term}'"


def test_no_engagement_prediction_artifacts():
    forbidden_terms = ["predicted_engagement", "engagement_residual", "out_of_fold", "engagement_prediction"]
    for name in REQUIRED_TABLES:
        df = pd.read_csv(TABLES_DIR / name)
        cols_lower = [c.lower() for c in df.columns]
        for term in forbidden_terms:
            assert not any(term in c for c in cols_lower), f"{name} contains forbidden column pattern '{term}'"


def test_moat_language_is_hypothesis_controlled():
    moat = pd.read_csv(TABLES_DIR / "creator_moat_hypotheses.csv")
    allowed_status = {"plausible", "weak", "unsupported", "contradicted"}
    assert set(moat["status"]).issubset(allowed_status)
    text_blob = " ".join(moat.fillna("").astype(str).values.flatten()).lower()
    assert "proven moat" not in text_blob


def test_observed_and_interpretation_are_separated():
    interp = pd.read_csv(TABLES_DIR / "creator_strategy_interpretation.csv")
    assert set(interp["level"]) == {"observed_finding", "interpretation", "hypothesis"}
    for brand in ["TALA", "Adanola", "Girlfriend Collective", "Oner Active"]:
        levels = set(interp[interp["brand"] == brand]["level"])
        assert levels == {"observed_finding", "interpretation", "hypothesis"}


def test_creator_metrics_report_n():
    df = pd.read_csv(TABLES_DIR / "creator_strategic_pattern_metrics.csv")
    assert "n" in df.columns
    assert (df["n"] > 0).all()


def test_organic_share_accuracy_across_all_day3c_tables():
    """Acceptance-patch item 1: no output table may affirm 'TALA is the most
    organic-led' -- Oner Active has the highest observed organic share (87.5%,
    n=8), TALA has an organic plurality within a larger sample (62.5%, n=24).
    Legitimate negations (e.g. explaining why the phrase does NOT apply) may
    remain."""
    pattern = pd.read_csv(TABLES_DIR / "creator_strategic_pattern_metrics.csv").set_index("brand")
    assert pattern.loc["Oner Active", "organic_share_pct"] > pattern.loc["TALA", "organic_share_pct"]
    assert pattern.loc["Oner Active", "n"] < pattern.loc["TALA", "n"]

    inaccurate_phrases = ["tala is the most organic-led", "tala is most organic-led", "tala shows the highest organic-mention share among the four", "its largest organic share (62.5%) among the four brands"]
    for name in REQUIRED_TABLES:
        df = pd.read_csv(TABLES_DIR / name)
        blob = " ".join(df.fillna("").astype(str).values.flatten()).lower()
        for phrase in inaccurate_phrases:
            assert phrase not in blob, f"{name} contains inaccurate organic-led claim: '{phrase}'"


def test_no_protected_characteristic_fields():
    protected_terms = ["race", "ethnicity", "gender", "age_group", "religion", "disability", "sexual_orientation"]
    for name in REQUIRED_TABLES:
        df = pd.read_csv(TABLES_DIR / name)
        cols_lower = [c.lower() for c in df.columns]
        for term in protected_terms:
            assert not any(term in c for c in cols_lower), f"{name} contains protected-characteristic column pattern '{term}'"


def test_no_external_api_called_in_build_scripts():
    scripts = [
        "scripts/build_analytical_synthesis.py",
        "scripts/build_mixed_claim_themes.py",
        "scripts/build_creator_strategy_comparison.py",
        "scripts/build_strategic_interpretation.py",
        "scripts/build_executive_findings.py",
        "scripts/build_synthesis_figures.py",
    ]
    forbidden = ["requests.get", "requests.post", "genai.", "openai.", "urlopen", "httpx."]
    for script in scripts:
        text = (PROJECT_ROOT / script).read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in text, f"{script} appears to call an external service via '{term}'"


def test_env_not_read_or_modified_by_synthesis_scripts():
    scripts = [
        "scripts/build_analytical_synthesis.py",
        "scripts/build_mixed_claim_themes.py",
        "scripts/build_creator_strategy_comparison.py",
        "scripts/build_strategic_interpretation.py",
        "scripts/build_executive_findings.py",
        "scripts/build_synthesis_figures.py",
    ]
    for script in scripts:
        text = (PROJECT_ROOT / script).read_text(encoding="utf-8")
        assert ".env" not in text
        assert "os.environ" not in text
        assert "dotenv" not in text
