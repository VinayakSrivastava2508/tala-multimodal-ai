"""Central cached data-loading layer for the Executive Decision Cockpit.

Loads only validated, precomputed CSVs from outputs/tables/ and data/
directories. Never reruns analytical pipelines, never loads .env, and never
imports ML/RAG modules -- those are lazily imported inside the RAG view only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

STRING_ID_COLUMNS = {
    "claim_id", "official_source_id", "evidence_id", "source_id",
    "strongest_supporting_evidence_id", "strongest_challenging_evidence_id",
    "strongest_reference_evidence_id", "supporting_evidence_ids",
    "challenging_evidence_ids", "evidence_ids", "claim_ids", "theme_id",
    "risk_id", "control_id", "finding_id", "lesson_id", "hypothesis_id",
}

BOOL_COLUMNS = {
    "conflicting_evidence", "external_validation_recommended", "sensitive_to_weights",
    "full_multimodal_bundle", "independent_evidence_present", "weight_sensitive",
    "platform_present", "passed",
}


class MissingTableError(RuntimeError):
    """Raised when a required output table is absent or malformed."""


def _table_path(name: str) -> Path:
    return TABLES_DIR / f"{name}.csv"


@st.cache_data(show_spinner=False)
def load_table(name: str, required_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Load one validated CSV from outputs/tables/ and return a typed DataFrame."""
    path = _table_path(name)
    if not path.exists():
        raise MissingTableError(
            f"Required analytical output '{name}.csv' was not found at {path}. "
            "Run the Day 3 synthesis/governance pipelines before opening this page."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise MissingTableError(
            f"'{name}.csv' is missing required column(s): {missing}. "
            "The authoritative schema has changed -- the cockpit was not updated to match."
        )
    return _typed(df)


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in STRING_ID_COLUMNS:
            continue
        if col in BOOL_COLUMNS:
            df[col] = df[col].map({"True": True, "False": False, "true": True, "false": False}).fillna(df[col])
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() > 0 and numeric.notna().sum() >= df[col].notna().sum() * 0.9:
            df[col] = numeric
    return df


@st.cache_data(show_spinner=False)
def load_claim_master() -> pd.DataFrame:
    return load_table(
        "analytical_synthesis_claim_master",
        required_columns=(
            "claim_id", "claim_text", "claim_category", "production_fusion_label",
            "fusion_confidence", "presentation_restriction", "conflicting_evidence",
            "sensitive_to_weights", "evidence_unit_count", "modalities_present",
            "independent_evidence_count",
        ),
    )


@st.cache_data(show_spinner=False)
def load_label_summary() -> pd.DataFrame:
    return load_table("claim_label_summary", required_columns=("automated_label", "n_claims"))


@st.cache_data(show_spinner=False)
def load_mixed_theme_summary() -> pd.DataFrame:
    return load_table("mixed_claim_theme_summary", required_columns=("theme_id", "theme_name", "claim_record_count"))


@st.cache_data(show_spinner=False)
def load_evidence_bundle_summary() -> pd.DataFrame:
    return load_table("claim_evidence_bundle_summary")


@st.cache_data(show_spinner=False)
def load_evidence_gap_by_category() -> pd.DataFrame:
    return load_table("evidence_gap_by_category")


@st.cache_data(show_spinner=False)
def load_executive_findings() -> pd.DataFrame:
    return load_table("executive_finding_register", required_columns=("finding_id", "finding"))


@st.cache_data(show_spinner=False)
def load_fusion_contribution() -> pd.DataFrame | None:
    """Load the multi-configuration fusion evaluation, if present."""
    path = _table_path("automated_fusion_evaluation")
    if not path.exists():
        return None
    return load_table("automated_fusion_evaluation")


@st.cache_data(show_spinner=False)
def load_creator_partnership_mix() -> pd.DataFrame:
    return load_table("creator_partnership_mix", required_columns=("brand", "partnership_type", "count", "brand_denominator_n"))


@st.cache_data(show_spinner=False)
def load_creator_content_intent_mix() -> pd.DataFrame:
    return load_table("creator_content_intent_mix", required_columns=("brand", "content_intent", "count"))


@st.cache_data(show_spinner=False)
def load_creator_partnership_intent_matrix() -> pd.DataFrame:
    return load_table("creator_partnership_intent_matrix")


@st.cache_data(show_spinner=False)
def load_brand_platform_strategy_comparison() -> pd.DataFrame:
    return load_table("brand_platform_strategy_comparison")


@st.cache_data(show_spinner=False)
def load_creator_strategy_interpretation() -> pd.DataFrame:
    return load_table("creator_strategy_interpretation")


@st.cache_data(show_spinner=False)
def load_competitor_lessons() -> pd.DataFrame:
    return load_table("tala_competitor_lessons")


@st.cache_data(show_spinner=False)
def load_moat_hypotheses() -> pd.DataFrame:
    return load_table("creator_moat_hypotheses")


@st.cache_data(show_spinner=False)
def load_creator_strategy_evidence_coverage() -> pd.DataFrame:
    return load_table("creator_strategy_evidence_coverage")


@st.cache_data(show_spinner=False)
def load_go_no_go() -> pd.DataFrame:
    return load_table("ai_governance_go_no_go", required_columns=("gate", "deployment_context", "status"))


@st.cache_data(show_spinner=False)
def load_risk_register() -> pd.DataFrame:
    return load_table(
        "ai_governance_risk_register",
        required_columns=("risk_id", "risk_scenario", "residual_likelihood_1_5", "residual_impact_1_5", "residual_risk_level"),
    )


@st.cache_data(show_spinner=False)
def load_control_matrix() -> pd.DataFrame:
    return load_table("ai_governance_control_matrix", required_columns=("control_id", "control_type", "control_status"))


@st.cache_data(show_spinner=False)
def load_claim_decision_policy() -> pd.DataFrame:
    return load_table("claim_decision_policy", required_columns=("decision_label", "meaning"))


@st.cache_data(show_spinner=False)
def load_kpi_kri() -> pd.DataFrame:
    return load_table("ai_governance_kpi_kri")


@st.cache_data(show_spinner=False)
def load_implementation_roadmap() -> pd.DataFrame:
    return load_table("ai_governance_implementation_roadmap", required_columns=("phase", "initiative"))


@st.cache_data(show_spinner=False)
def load_modality_audit() -> pd.DataFrame:
    return load_table("assignment_modality_audit")


@st.cache_data(show_spinner=False)
def load_esg_claim_assurance_matrix() -> pd.DataFrame | None:
    path = _table_path("esg_claim_assurance_matrix")
    if not path.exists():
        return None
    return load_table("esg_claim_assurance_matrix")


def modality_flags(modalities_present: str) -> dict[str, bool]:
    """Return per-modality booleans parsed from a ';'-joined modalities_present cell."""
    parts = {p.strip() for p in str(modalities_present or "").split(";") if p.strip()}
    return {
        "text": "text" in parts,
        "image": "image" in parts,
        "video": "video" in parts,
        "reference": "reference" in parts,
    }
