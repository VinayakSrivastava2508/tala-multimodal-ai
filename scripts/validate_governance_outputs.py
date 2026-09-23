"""Direct validation for the 12 Day 3D AI-governance/ethics/ESG output tables
in outputs/tables/. Mirrors the pattern in scripts/validate_synthesis_outputs.py:
reuses src/validation.py's schema-routing and column/type/value validation,
then adds the governance-specific semantic checks from the task brief.

Exits non-zero if any table fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from validation import validate, guess_schema  # noqa: E402

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

GOVERNANCE_TABLES: dict[str, str] = {
    "ai_governance_risk_register.csv": "ai_governance_risk_register",
    "ai_governance_control_matrix.csv": "ai_governance_control_matrix",
    "claim_decision_policy.csv": "claim_decision_policy",
    "ai_governance_raci.csv": "ai_governance_raci",
    "ai_data_lifecycle_policy.csv": "ai_data_lifecycle_policy",
    "esg_claim_assurance_matrix.csv": "esg_claim_assurance_matrix",
    "regulatory_standards_map.csv": "regulatory_standards_map",
    "genai_vendor_control_matrix.csv": "genai_vendor_control_matrix",
    "ai_governance_kpi_kri.csv": "ai_governance_kpi_kri",
    "ai_incident_escalation_matrix.csv": "ai_incident_escalation_matrix",
    "ai_governance_implementation_roadmap.csv": "ai_governance_implementation_roadmap",
    "ai_governance_go_no_go.csv": "ai_governance_go_no_go",
}

assert len(GOVERNANCE_TABLES) == 12

MANDATORY_INSTRUMENTS = {
    # requirement_id -> True if REGULATION/LEGISLATION (mandatory), False if
    # voluntary framework/standard. Used to catch a voluntary standard being
    # mislabelled mandatory.
    "REG07": False,  # NIST AI RMF -- voluntary
    "REG08": False,  # ISO/IEC 42001 -- voluntary
    "REG09": False,  # OECD AI Principles -- voluntary
}


def _risk_level(score: int) -> str:
    if score >= 15:
        return "Critical"
    if score >= 10:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


def check_file(filename: str, schema_name: str) -> list[str]:
    errors: list[str] = []
    path = TABLES_DIR / filename
    if not path.exists():
        return [f"FILE MISSING: {path}"]

    routed = guess_schema(filename)
    if routed != schema_name:
        errors.append(f"ROUTING: '{filename}' routed to '{routed}', expected '{schema_name}'")

    df = pd.read_csv(path)
    ok, schema_errors = validate(df, schema_name)
    hard_fail_prefixes = ("MISSING", "NULL", "INVALID", "BELOW", "ABOVE")
    errors.extend([e for e in schema_errors if e.startswith(hard_fail_prefixes)])

    if filename == "ai_governance_risk_register.csv":
        if df["risk_id"].duplicated().any():
            errors.append("DUPLICATE PRIMARY ID: risk_id duplicated")
        bad_inherent = df[df["inherent_risk_score"] != df["likelihood_1_5"] * df["impact_1_5"]]
        if len(bad_inherent):
            errors.append(f"RISK SCORE: {len(bad_inherent)} row(s) where inherent_risk_score != likelihood x impact")
        bad_residual = df[df["residual_risk_score"] != df["residual_likelihood_1_5"] * df["residual_impact_1_5"]]
        if len(bad_residual):
            errors.append(f"RISK SCORE: {len(bad_residual)} row(s) where residual_risk_score != residual_likelihood x residual_impact")
        bad_level = df[df["inherent_risk_score"].map(_risk_level) != df["inherent_risk_level"]]
        if len(bad_level):
            errors.append(f"RISK LEVEL MAPPING: {len(bad_level)} row(s) with inherent_risk_level inconsistent with score")
        bad_resid_level = df[df["residual_risk_score"].map(_risk_level) != df["residual_risk_level"]]
        if len(bad_resid_level):
            errors.append(f"RISK LEVEL MAPPING: {len(bad_resid_level)} row(s) with residual_risk_level inconsistent with score")
        for col in ["likelihood_1_5", "impact_1_5", "residual_likelihood_1_5", "residual_impact_1_5"]:
            out_of_range = df[(df[col] < 1) | (df[col] > 5)]
            if len(out_of_range):
                errors.append(f"SCALE RANGE: {col} has {len(out_of_range)} value(s) outside 1-5")
        residual_worse = df[df["residual_risk_score"] > df["inherent_risk_score"]]
        if len(residual_worse):
            errors.append(f"RESIDUAL > INHERENT: {len(residual_worse)} row(s) where controls apparently increase risk")
        empty_owner = df[df["risk_owner"].isna() | (df["risk_owner"].astype(str).str.strip() == "")]
        if len(empty_owner):
            errors.append(f"MISSING OWNER: {len(empty_owner)} risk row(s) without risk_owner")

    if filename == "ai_governance_control_matrix.csv":
        if df["control_id"].duplicated().any():
            errors.append("DUPLICATE PRIMARY ID: control_id duplicated")
        empty_owner = df[df["control_owner"].isna() | (df["control_owner"].astype(str).str.strip() == "")]
        if len(empty_owner):
            errors.append(f"MISSING OWNER: {len(empty_owner)} control row(s) without control_owner")
        # every High/Critical risk must have >=1 preventive and >=1 detective control
        risk_reg_path = TABLES_DIR / "ai_governance_risk_register.csv"
        if risk_reg_path.exists():
            risks = pd.read_csv(risk_reg_path)
            high_crit = risks[risks["inherent_risk_level"].isin(["High", "Critical"])]["risk_id"].tolist()
            for rid in high_crit:
                matched = df[df["risk_ids"].astype(str).str.contains(rid, regex=False)]
                types = set(matched["control_type"])
                if not {"Preventive", "Detective"}.issubset(types):
                    errors.append(f"CONTROL COVERAGE: high/critical risk {rid} missing {({'Preventive', 'Detective'} - types)}")
        implemented_but_proposed_language = df[
            (df["control_status"] == "Implemented")
            & df["system_evidence"].astype(str).str.contains("not present in repository|does not exist|no scheduler exists|no dashboard", case=False, regex=True)
        ]
        if len(implemented_but_proposed_language):
            errors.append(f"STATUS MISMATCH: {len(implemented_but_proposed_language)} row(s) marked Implemented but evidence text suggests it is not built")

    if filename == "claim_decision_policy.csv":
        aligned_row = df[df["decision_label"] == "aligned"]
        if len(aligned_row):
            text = aligned_row.iloc[0]["what_it_does_not_mean"].lower()
            if "not" not in text or "proven" not in text and "independently" not in text:
                errors.append("ALIGNED MEANING: 'aligned' row does not clearly state it is not independently proven")
        insuff_row = df[df["decision_label"] == "insufficient_evidence"]
        if len(insuff_row):
            text = (insuff_row.iloc[0]["what_it_does_not_mean"] + insuff_row.iloc[0]["prohibited_wording"]).lower()
            if "false" not in text:
                errors.append("INSUFFICIENT EVIDENCE: row does not address 'not false' framing")
        mixed_row = df[df["decision_label"] == "mixed"]
        if len(mixed_row):
            esc = mixed_row.iloc[0]["escalation_requirement"].lower()
            if "always escalate" not in esc and "escalate" not in esc:
                errors.append("MIXED ESCALATION: mixed row does not require escalation")

    if filename == "ai_governance_raci.csv":
        role_cols = [c for c in df.columns if c not in ("activity_id", "activity")]
        n_accountable = (df[role_cols] == "A").sum(axis=1)
        bad = df[n_accountable != 1]
        if len(bad):
            errors.append(f"RACI ACCOUNTABILITY: {len(bad)} activity row(s) without exactly one 'A'")

    if filename == "regulatory_standards_map.csv":
        missing_url = df[
            df["official_url"].isna()
            | (~df["official_url"].astype(str).str.strip().isin(["TBC"]) & ~df["official_url"].astype(str).str.startswith(("http://", "https://")))
        ]
        if len(missing_url):
            errors.append(f"OFFICIAL URL: {len(missing_url)} row(s) without an official_url or 'TBC'")
        for req_id, is_mandatory in MANDATORY_INSTRUMENTS.items():
            row = df[df["requirement_id"] == req_id]
            if len(row) and not is_mandatory and row.iloc[0]["mandatory_status"].strip().lower() == "mandatory":
                errors.append(f"MANDATORY MISLABEL: {req_id} is a voluntary standard but labelled mandatory")

    if filename == "ai_governance_control_matrix.csv":
        proposed_marked_implemented_text = df[df["control_status"] == "Proposed"]
        already_exists_language = proposed_marked_implemented_text[
            proposed_marked_implemented_text["control_description"].astype(str).str.contains(r"\balready\b", case=False, regex=True)
        ]
        if len(already_exists_language):
            errors.append(f"STATUS WORDING: {len(already_exists_language)} 'Proposed' control(s) whose description implies it already exists")

    if filename == "ai_governance_implementation_roadmap.csv":
        empty_criterion = df[df["completion_criterion"].isna() | (df["completion_criterion"].astype(str).str.strip() == "")]
        if len(empty_criterion):
            errors.append(f"COMPLETION CRITERION: {len(empty_criterion)} roadmap row(s) missing completion_criterion")

    if filename == "ai_governance_go_no_go.csv":
        if not (df["status"].isin(["GO", "PARTIAL", "NO-GO"])).all():
            errors.append("GATE STATUS: an invalid status value was found")
        # External deployment cannot be GO overall unless every mandatory gate
        # is GO. Mandatory gates for external/customer-facing deployment:
        mandatory_external_gates = {
            "Privacy", "Media rights", "Security", "Human oversight", "ESG assurance", "Citation integrity",
        }
        external = df[df["deployment_context"] == "external_customer_facing_deployment_readiness"]
        any_external_go = (external["status"] == "GO").any()
        mandatory_rows = external[external["gate"].isin(mandatory_external_gates)]
        all_mandatory_go = len(mandatory_rows) == len(mandatory_external_gates) and (mandatory_rows["status"] == "GO").all()
        if any_external_go and not all_mandatory_go:
            errors.append("EXTERNAL DEPLOYMENT GATE: at least one external gate is GO while a mandatory gate is not GO -- external deployment cannot be GO without all mandatory gates satisfied")

    protected_terms = ["race", "ethnicity", "gender", "age_group", "religion", "disability", "sexual_orientation"]
    engagement_terms = ["predicted_engagement", "engagement_residual", "out_of_fold", "engagement_prediction", "engagement_rank"]
    cols_lower = [c.lower() for c in df.columns]
    for term in protected_terms:
        if any(term in c for c in cols_lower):
            errors.append(f"PROTECTED CHARACTERISTIC: column pattern '{term}' found")
    for term in engagement_terms:
        if any(term in c for c in cols_lower):
            errors.append(f"ENGAGEMENT PREDICTION: column pattern '{term}' found")

    text_blob = " ".join(df.fillna("").astype(str).values.flatten()).lower()
    if "fallback model" in text_blob or "fallback generator" in text_blob:
        if "no fallback" not in text_blob and "there is no fallback" not in text_blob:
            errors.append("FALLBACK GENERATOR: text mentions a fallback model/generator without negating it")
    if "face embedding" in text_blob or "face recognition" in text_blob:
        allowed = ["no face", "prohibition", "never", "not implemented"]
        if not any(a in text_blob for a in allowed):
            errors.append("FACE EMBEDDINGS: mentioned without a prohibition/negation")

    return errors


def main() -> int:
    print("Day 3D governance outputs")
    results: dict[str, list[str]] = {}
    for filename, schema_name in GOVERNANCE_TABLES.items():
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
