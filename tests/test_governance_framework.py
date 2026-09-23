"""Day 3D quality controls: AI governance, ethics and ESG framework outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

GOVERNANCE_TABLES = [
    "ai_governance_risk_register.csv",
    "ai_governance_control_matrix.csv",
    "claim_decision_policy.csv",
    "ai_governance_raci.csv",
    "ai_data_lifecycle_policy.csv",
    "esg_claim_assurance_matrix.csv",
    "regulatory_standards_map.csv",
    "genai_vendor_control_matrix.csv",
    "ai_governance_kpi_kri.csv",
    "ai_incident_escalation_matrix.csv",
    "ai_governance_implementation_roadmap.csv",
    "ai_governance_go_no_go.csv",
]


@pytest.fixture(scope="module")
def risks():
    return pd.read_csv(TABLES_DIR / "ai_governance_risk_register.csv")


@pytest.fixture(scope="module")
def controls():
    return pd.read_csv(TABLES_DIR / "ai_governance_control_matrix.csv")


def _risk_level(score: int) -> str:
    if score >= 15:
        return "Critical"
    if score >= 10:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


def test_all_governance_tables_exist():
    for name in GOVERNANCE_TABLES:
        assert (TABLES_DIR / name).exists(), f"missing governance output: {name}"


def test_risk_scores_equal_likelihood_times_impact(risks):
    assert (risks["inherent_risk_score"] == risks["likelihood_1_5"] * risks["impact_1_5"]).all()
    assert (risks["residual_risk_score"] == risks["residual_likelihood_1_5"] * risks["residual_impact_1_5"]).all()


def test_risk_levels_map_correctly_to_scores(risks):
    assert (risks["inherent_risk_score"].map(_risk_level) == risks["inherent_risk_level"]).all()
    assert (risks["residual_risk_score"].map(_risk_level) == risks["residual_risk_level"]).all()


def test_residual_risk_never_exceeds_inherent_risk(risks):
    assert (risks["residual_risk_score"] <= risks["inherent_risk_score"]).all()


def test_every_high_critical_risk_has_preventive_and_detective_control(risks, controls):
    high_crit = risks[risks["inherent_risk_level"].isin(["High", "Critical"])]["risk_id"].tolist()
    assert len(high_crit) > 0
    for rid in high_crit:
        matched = controls[controls["risk_ids"].astype(str).str.contains(rid, regex=False)]
        types = set(matched["control_type"])
        assert {"Preventive", "Detective"}.issubset(types), f"{rid} missing coverage: has {types}"


def test_every_control_has_an_owner(controls):
    blank = controls["control_owner"].isna() | (controls["control_owner"].astype(str).str.strip() == "")
    assert not blank.any()


def test_every_raci_activity_has_exactly_one_accountable_role():
    df = pd.read_csv(TABLES_DIR / "ai_governance_raci.csv")
    role_cols = [c for c in df.columns if c not in ("activity_id", "activity")]
    n_accountable = (df[role_cols] == "A").sum(axis=1)
    assert (n_accountable == 1).all()


def test_every_regulatory_row_has_official_url_or_tbc():
    df = pd.read_csv(TABLES_DIR / "regulatory_standards_map.csv")
    ok = df["official_url"].astype(str).str.startswith(("http://", "https://")) | (df["official_url"] == "TBC")
    assert ok.all()


def test_voluntary_standards_not_labelled_mandatory():
    df = pd.read_csv(TABLES_DIR / "regulatory_standards_map.csv").set_index("requirement_id")
    for req_id in ["REG07", "REG08", "REG09"]:  # NIST AI RMF, ISO/IEC 42001, OECD AI Principles
        assert df.loc[req_id, "mandatory_status"].strip().lower() == "voluntary"


def test_eu_ai_act_row_does_not_assert_definitive_classification():
    df = pd.read_csv(TABLES_DIR / "regulatory_standards_map.csv")
    row = df[df["instrument"].str.contains("AI Act", case=False)]
    assert len(row) == 1
    assert row.iloc[0]["legal_review_required"] == True  # noqa: E712
    caveat = row.iloc[0]["caveat"].lower()
    assert "no definitive classification" in caveat or "legal review" in caveat


def test_proposed_controls_not_labelled_implemented(controls):
    proposed = controls[controls["control_status"] == "Proposed"]
    already_language = proposed[proposed["control_description"].astype(str).str.contains(r"\balready\b", case=False, regex=True)]
    assert already_language.empty


def test_implemented_controls_have_supporting_system_evidence(controls):
    implemented = controls[controls["control_status"] == "Implemented"]
    blank_evidence = implemented["system_evidence"].isna() | (implemented["system_evidence"].astype(str).str.strip() == "")
    assert not blank_evidence.any()


def test_aligned_does_not_mean_independently_proven():
    df = pd.read_csv(TABLES_DIR / "claim_decision_policy.csv").set_index("decision_label")
    text = df.loc["aligned", "what_it_does_not_mean"].lower()
    assert "not" in text and "independently proven" in text


def test_insufficient_evidence_is_not_treated_as_false():
    df = pd.read_csv(TABLES_DIR / "claim_decision_policy.csv").set_index("decision_label")
    row = df.loc["insufficient_evidence"]
    assert "not" in row["what_it_does_not_mean"].lower() and "false" in row["what_it_does_not_mean"].lower()
    assert "false" in row["prohibited_wording"].lower()


def test_mixed_claims_require_escalation():
    df = pd.read_csv(TABLES_DIR / "claim_decision_policy.csv").set_index("decision_label")
    esc = df.loc["mixed", "escalation_requirement"].lower()
    assert "escalate" in esc


def test_divergent_and_mixed_require_human_review_before_use():
    df = pd.read_csv(TABLES_DIR / "claim_decision_policy.csv").set_index("decision_label")
    for label in ["mixed", "divergent"]:
        reviewer = df.loc[label, "required_reviewer"].lower()
        assert "legal counsel" in reviewer


def test_no_customer_facing_autonomous_use_is_permitted():
    """No claim_decision_policy row permits unreviewed, fully autonomous external use."""
    df = pd.read_csv(TABLES_DIR / "claim_decision_policy.csv")
    for _, row in df.iterrows():
        status = row["external_communication_status"].lower()
        assert "not permitted" in status or "conditional" in status or "review" in status


def test_labour_esg_gaps_trigger_esg_review():
    df = pd.read_csv(TABLES_DIR / "claim_decision_policy.csv").set_index("decision_label")
    esc = df.loc["insufficient_evidence", "escalation_requirement"].lower()
    assert "sustainability" in esc or "esg" in esc

    esg = pd.read_csv(TABLES_DIR / "esg_claim_assurance_matrix.csv")
    labour_row = esg[esg["claim_category"] == "Labour"]
    assert len(labour_row) == 1
    assert "esg" in labour_row.iloc[0]["review_owner"].lower() or "sustainability" in labour_row.iloc[0]["review_owner"].lower()


def test_every_roadmap_item_has_a_completion_criterion():
    df = pd.read_csv(TABLES_DIR / "ai_governance_implementation_roadmap.csv")
    blank = df["completion_criterion"].isna() | (df["completion_criterion"].astype(str).str.strip() == "")
    assert not blank.any()


def test_external_deployment_cannot_be_go_without_all_mandatory_gates():
    df = pd.read_csv(TABLES_DIR / "ai_governance_go_no_go.csv")
    mandatory_gates = {"Privacy", "Media rights", "Security", "Human oversight", "ESG assurance", "Citation integrity"}
    external = df[df["deployment_context"] == "external_customer_facing_deployment_readiness"]
    any_go = (external["status"] == "GO").any()
    mandatory_rows = external[external["gate"].isin(mandatory_gates)]
    all_mandatory_go = len(mandatory_rows) == len(mandatory_gates) and (mandatory_rows["status"] == "GO").all()
    if any_go:
        assert all_mandatory_go, "external deployment marked GO on some gate without all mandatory gates satisfied"


def test_academic_demonstration_is_go_and_external_is_not(risks):
    df = pd.read_csv(TABLES_DIR / "ai_governance_go_no_go.csv")
    academic = df[df["deployment_context"] == "academic_demonstration_readiness"]
    external = df[df["deployment_context"] == "external_customer_facing_deployment_readiness"]
    assert (academic["status"] == "GO").all()
    assert (external["status"] == "NO-GO").all()


def test_no_protected_characteristic_inference_in_any_governance_table():
    protected_terms = ["race", "ethnicity", "gender", "age_group", "religion", "disability", "sexual_orientation"]
    for name in GOVERNANCE_TABLES:
        df = pd.read_csv(TABLES_DIR / name)
        cols_lower = [c.lower() for c in df.columns]
        for term in protected_terms:
            assert not any(term in c for c in cols_lower), f"{name} has forbidden column '{term}'"


def test_no_face_embeddings_mentioned_without_prohibition():
    genai = pd.read_csv(TABLES_DIR / "genai_vendor_control_matrix.csv")
    control_matrix = pd.read_csv(TABLES_DIR / "ai_governance_control_matrix.csv")
    face_rows = control_matrix[control_matrix["control_name"].str.contains("face", case=False)]
    assert len(face_rows) >= 1
    assert face_rows["control_description"].str.contains("no face", case=False).any()


def test_no_fallback_generator_documented_accurately():
    genai = pd.read_csv(TABLES_DIR / "genai_vendor_control_matrix.csv")
    row = genai[genai["control_area"].str.contains("fallback", case=False)]
    assert len(row) == 1
    desc = row.iloc[0]["control_description"].lower()
    assert "no fallback" in desc
    assert "stop" in desc or "stops" in desc
    assert row.iloc[0]["status"] == "Implemented"


def test_no_engagement_prediction_in_any_governance_table():
    engagement_terms = ["predicted_engagement", "engagement_residual", "out_of_fold", "engagement_prediction"]
    for name in GOVERNANCE_TABLES:
        df = pd.read_csv(TABLES_DIR / name)
        cols_lower = [c.lower() for c in df.columns]
        for term in engagement_terms:
            assert not any(term in c for c in cols_lower), f"{name} has forbidden column '{term}'"


def test_no_env_access_in_governance_build_scripts():
    scripts = [
        "scripts/build_governance_framework.py",
        "scripts/build_governance_figures.py",
        "scripts/validate_governance_outputs.py",
    ]
    for script in scripts:
        text = (PROJECT_ROOT / script).read_text(encoding="utf-8")
        # These scripts document existing .env-based controls in descriptive
        # table text (e.g. "GEMINI_API_KEY read from .env via os.environ");
        # what must never appear is actual code that reads or writes it.
        assert "dotenv" not in text
        assert 'open(".env"' not in text and "open('.env'" not in text
        assert "os.getenv(" not in text
        assert "os.environ[" not in text and "os.environ.get(" not in text


def test_no_external_api_called_in_governance_build_scripts():
    scripts = [
        "scripts/build_governance_framework.py",
        "scripts/build_governance_figures.py",
        "scripts/validate_governance_outputs.py",
    ]
    # Specific enough to avoid matching local variable names like "genai" (the
    # governance framework's own genai_vendor_control_matrix DataFrame variable).
    forbidden = ["requests.get(", "requests.post(", "genai.Client", "genai.configure", "openai.", "urlopen(", "httpx.get(", "httpx.post("]
    for script in scripts:
        text = (PROJECT_ROOT / script).read_text(encoding="utf-8")
        for term in forbidden:
            assert term not in text, f"{script} calls an external service via '{term}'"


def test_governance_does_not_change_existing_fusion_labels():
    """Day 3D must not alter Day 3A/3C analytical results."""
    master = pd.read_csv(TABLES_DIR / "analytical_synthesis_claim_master.csv")
    fusion = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "fusion" / "claim_fusion_results.csv")
    merged = master.merge(fusion[["claim_id", "automated_label"]], on="claim_id", suffixes=("", "_authoritative"))
    assert (merged["production_fusion_label"] == merged["automated_label"]).all()
    assert (master["production_fusion_label"] == "aligned").sum() == 14
    assert (master["production_fusion_label"] == "mixed").sum() == 5
    assert (master["production_fusion_label"] == "insufficient_evidence").sum() == 18
