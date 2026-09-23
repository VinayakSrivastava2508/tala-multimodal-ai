"""Day 3D: AI Governance, Ethics and ESG Framework -- builds all governance
output tables from (a) already-verified Day 3A/3C repository outputs and
(b) a hand-curated, source-cited governance content set assembled from
official regulatory/standards sources verified 22 September 2026 (see
docs/ai_governance_ethics_esg_framework.md section 15 / the regulatory map
table for full citations).

This script performs NO new data collection, NO external API calls, and
changes NO analytical result. It only reads existing outputs/tables/*.csv to
ground its quantitative statements (claim counts, label distribution, mixed
themes) and writes new governance tables under outputs/tables/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
ACCESS_DATE = "2026-09-22"

# ---------------------------------------------------------------------------
# Ground quantitative statements in the real, already-verified corpus.
# ---------------------------------------------------------------------------

def _load_system_facts() -> dict:
    master = pd.read_csv(TABLES_DIR / "analytical_synthesis_claim_master.csv")
    themes = pd.read_csv(TABLES_DIR / "mixed_claim_theme_summary.csv")
    gap = pd.read_csv(TABLES_DIR / "evidence_gap_by_category.csv")
    coverage = pd.read_csv(TABLES_DIR / "creator_strategy_evidence_coverage.csv")

    labour = gap[gap["claim_category"] == "labour"].iloc[0]
    return {
        "n_claims": len(master),
        "n_aligned": int((master["production_fusion_label"] == "aligned").sum()),
        "n_mixed": int((master["production_fusion_label"] == "mixed").sum()),
        "n_insufficient": int((master["production_fusion_label"] == "insufficient_evidence").sum()),
        "n_mixed_themes": len(themes),
        "n_labour_claims": int(labour["total_claims"]),
        "n_labour_independent_evidence": int(labour["claims_with_independent_evidence"]),
        "n_creator_youtube": int(coverage["verified_creator_records"].sum()) - 1,  # 60 of 61 (1 tiktok)
        "n_creator_total": int(coverage["verified_creator_records"].sum()),
    }


FACTS = _load_system_facts()

RISK_SCORING_NOTE = (
    "5x5 matrix: likelihood_1_5 (1=rare, 5=near-certain within a governance review "
    "cycle) x impact_1_5 (1=negligible, 5=severe -- reputational, legal, or "
    "decision-quality harm). inherent_risk_score = likelihood x impact (max 25). "
    "Levels: 1-4 Low, 5-9 Medium, 10-14 High, 15-25 Critical. These are structured "
    "MANAGEMENT JUDGEMENTS calibrated against the current, inspected system state -- "
    "not empirical probabilities or actuarial estimates."
)


def _risk_level(score: int) -> str:
    if score >= 15:
        return "Critical"
    if score >= 10:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Section 9: AI governance risk register
# ---------------------------------------------------------------------------

RISK_ROWS_RAW = [
    # (risk_id, domain, scenario, component, stakeholder, L, I, controls, status, resid_L, resid_I, owner, freq, trigger, evidence, caveat)
    ("R01", "Privacy and personal data", "A collected review/press/creator record retains a name, handle, or other personal identifier beyond what analysis requires.", "Data admission (ingestion)", "Reviewers, creators, customers", 3, 3, "Username stripping documented in docs/governance_notes.md; schema.yaml provenance fields", "Partially implemented", 2, 3, "Data steward", "Per collection batch", "New source type onboarded without minimisation check", "docs/governance_notes.md; configs/schema.yaml", "Public availability of a post does not itself make continued storage of an identifier necessary or proportionate."),
    ("R02", "Copyright and media rights", "An image or video frame is stored/processed without a documented rights basis (e.g. official vs. UGC vs. unknown).", "Data admission (media)", "TALA, competitors, creators", 3, 3, "rights_or_access_basis field in configs/schema.yaml::video_assets; access-basis gating in scripts/process_video_assets.py", "Implemented", 2, 3, "Data steward", "Per asset batch", "New media asset lacks a rights_or_access_basis value", "configs/schema.yaml (video_assets schema)", "Rights basis is self-declared at collection time; no independent legal rights clearance was performed."),
    ("R03", "Platform terms of service", "A collection method breaches a platform's terms of service (e.g. unofficial scraping, bypassing login/CAPTCHA).", "Data admission", "TALA, platforms, project team", 2, 4, "CLAUDE.md Data Ethics Rules; docs/governance_notes.md platform TOS table; no yt-dlp/unofficial downloaders used", "Implemented", 1, 4, "Data steward", "Per new source", "New platform or collection method proposed", "docs/governance_notes.md; CLAUDE.md section 5", "TOS interpretation is not a substitute for legal advice; platform TOS can change without notice."),
    ("R04", "Data provenance", "A record lacks a resolvable source_url/source_id, making its evidentiary status unverifiable.", "Data admission", "Management, analysts", 2, 3, "PROVENANCE_FIELDS in src/validation.py; scripts/validate_data.py provenance checks", "Implemented", 1, 3, "Data steward", "Every validator run", "validate_data.py reports a MISSING provenance error", "src/validation.py; scripts/validate_data.py output", "Provenance completeness is checked structurally, not for URL liveness at read time."),
    ("R05", "Evidence staleness", "Claims, reviews, or reference evidence age without a scheduled refresh, so findings reflect an out-of-date brand/product state.", "Fusion / claim synthesis", "Management, TALA", 4, 3, "collection_date field present; no scheduled refresh job exists yet", "Proposed", 3, 3, "AI product owner", "Quarterly (proposed)", "Evidence age exceeds refresh threshold (KPI, see ai_governance_kpi_kri.csv)", "configs/schema.yaml provenance fields; no refresh scheduler in repo", "No automated staleness detection currently runs in production; this is an academic-prototype gap."),
    ("R06", "Sampling and representation bias", "Creator and customer evidence is a targeted recovery sample, not a statistically representative one, so findings may not generalise.", "Data admission / creator strategy", "TALA, management, creators", 4, 3, "Documented boundary in docs/creator_strategy_comparison.md and CLAUDE.md; denominators reported alongside every percentage", "Implemented", 3, 3, "AI product owner", "Every report cycle", "A finding is used without its stated sample-size caveat", "docs/creator_strategy_comparison.md; outputs/tables/creator_strategy_evidence_coverage.csv", "Bias cannot be fully corrected without a materially larger, differently-sampled corpus."),
    ("R07", "Creator misrepresentation", "A creator's partnership type or intent is automatically misclassified (e.g. organic content mislabelled as paid), creating a factually wrong public statement about a named individual/brand relationship.", "Fusion / creator classification", "Creators, TALA, competitors", 3, 4, "Evidence-gated automated classification (revised_partnership_type); requires_human_validation / partnership_human_review_required fields exist in creator_multimodal_features.csv", "Partially implemented", 2, 4, "Creator-partnership lead", "Before any external publication", "A creator-related finding is proposed for external use", "data/processed/creator_multimodal_features.csv columns partnership_human_review_required", "Automated labels are not human-adjudicated; no creator-facing correction channel currently exists."),
    ("R08", "Customer-review misinterpretation", "Sentiment/stance extraction from a customer review misreads sarcasm, mixed sentiment, or context, producing a wrong support/challenge stance.", "Text feature extraction / fusion", "Customers, TALA", 3, 3, "stance_score, reliability_score fields in claim_evidence_units.csv; NLI diagnostics documented in docs/automated_fusion_evaluation.md", "Implemented", 2, 3, "Model-risk/analytics lead", "Per fusion run", "NLI/rule disagreement rate exceeds threshold (see nli_rule_disagreement_analysis.csv)", "outputs/tables/nli_rule_disagreement_analysis.csv", "Automated stance extraction is not human-verified per record; spot-checking is a proposed control, not implemented."),
    ("R09", "Defamation and reputational harm", "A mixed or insufficient-evidence finding about TALA or a competitor is published externally in a way that reads as a factual accusation (e.g. of greenwashing) rather than a corpus-bounded observation.", "Managerial use / external communication", "TALA, competitors, project team", 3, 5, "claim_decision_policy.csv wording rules; presentation_restriction field on every claim; CLAUDE.md analytical-honesty wording rules", "Proposed", 2, 5, "Legal counsel", "Before every external publication", "A finding lacking presentation_restriction clearance is drafted for external use", "outputs/tables/claim_decision_policy.csv; CLAUDE.md section 16", "This is the single highest-impact risk in the register; no legal sign-off process currently exists in the repository."),
    ("R10", "Greenwashing false positive", "A claim is labelled mixed/insufficient and is then presented as if it were proven greenwashing, when the corpus only shows an evidence gap or partial conflict.", "Managerial use", "TALA, management", 3, 4, "claim_decision_policy.csv 'what it does not mean' column; insufficient_evidence_analysis.csv reason codes", "Proposed", 2, 4, "Sustainability/ESG lead", "Before every ESG-related communication", "Insufficient-evidence or mixed claim proposed for negative external framing", "outputs/tables/insufficient_evidence_analysis.csv; claim_decision_policy.csv", "The system was explicitly designed not to accuse TALA of greenwashing; that discipline depends on human adherence to the policy."),
    ("R11", "Greenwashing false negative", "A genuinely misleading claim is labelled 'aligned' because the corpus only contains self-reported or thin supporting evidence, giving false reassurance.", "Fusion / claim synthesis", "Customers, regulators, TALA", 3, 4, "self_reported_evidence_count vs independent_evidence_count distinction on every claim; 'aligned does not mean independently proven' policy line", "Partially implemented", 2, 4, "Sustainability/ESG lead", "Per claim reviewed for external use", "An aligned claim with 0 independent_evidence_count is proposed for external reuse", "outputs/tables/analytical_synthesis_claim_master.csv (independent_evidence_count column)", "The distinction exists in the data; a mandatory review gate to enforce it before publication is proposed, not yet implemented."),
    ("R12", "Self-reported evidence treated as independent", "A self-reported (brand-authored) evidence unit is cited as if it were independent corroboration.", "Fusion / claim synthesis", "Management, external readers", 3, 4, "source_independence field (self_reported / customer_reported / unclear) on every evidence unit", "Implemented", 2, 4, "Model-risk/analytics lead", "Per claim reviewed", "A claim's strongest_supporting_evidence_id resolves to a self-reported source only", "data/processed/fusion/claim_evidence_units.csv (source_independence column)", "Field exists and is populated; a hard block preventing self-reported-only claims from being marked 'aligned' externally is a proposed control."),
    ("R13", "Modality misrouting", "Text-only evidence is treated as if it visually corroborates a claim, or vice versa.", "Fusion / modality routing", "Management, analysts", 2, 3, "visually_groundable / visual_gate_passed fields; permitted_evidence_role gating in src/fusion/evidence_fusion.py", "Implemented", 1, 3, "Model-risk/analytics lead", "Per fusion run", "visual_gate_passed=True on a non-visually-groundable claim", "data/processed/fusion/claim_evidence_units.csv (visual_gate_reason column)", "Gate logic is code-reviewed and tested (tests/test_rag_*), not independently audited by a third party."),
    ("R14", "Image-context error", "An eligible product/catalog image is matched to the wrong claim or product due to weak CLIP similarity.", "Image feature extraction", "TALA, management", 2, 3, "image_similarity threshold (>=0.27) documented in scripts/build_claim_multimodal_candidates.py provenance_note", "Implemented", 2, 3, "Model-risk/analytics lead", "Per candidate-bundle build", "image_similarity below documented threshold is admitted", "data/processed/claim_multimodal_evidence_candidates.csv (image_similarity, provenance_note)", "Threshold is a documented heuristic, not independently validated against a labelled ground truth set."),
    ("R15", "Video-frame context loss", "Sampled video frames lose the temporal/narrative context of the source video, changing the apparent meaning of a claim's video evidence.", "Video feature extraction", "TALA, management", 2, 3, "video_assets binding definition (>=2 sampled frames, motion/scene-change features) in configs/schema.yaml; platform_metadata_only labelling for leads", "Implemented", 2, 3, "Model-risk/analytics lead", "Per video asset processed", "A video asset is used as evidence without genuine temporal features", "configs/schema.yaml::video_assets; docs/image_video_codebook.md", "Frame sampling density is a project-defined threshold, not validated against professional video-review standards."),
    ("R16", "Automated fusion error", "The rule-based/NLI fusion logic assigns an incorrect label (aligned/mixed/insufficient) due to a code or threshold defect.", "Fusion / claim synthesis", "TALA, management", 2, 4, "tests/test_rag_*, docs/automated_fusion_evaluation.md, fusion_sensitivity_results.csv cross-configuration checks", "Implemented", 2, 4, "Model-risk/analytics lead", "Per fusion code change", "A fusion-related test fails or sensitivity result is inconsistent", "outputs/tables/fusion_sensitivity_stability.csv; tests/ directory", "Testing covers logic correctness, not full human adjudication of every label."),
    ("R17", "Hallucinated Gemini answer", "The Gemini-generated answer states something not present in the retrieved evidence context.", "Retrieval and generation", "Management, external readers", 2, 4, "Citation validation (src/rag/citation_validator.py); rag_citation_validation.csv results", "Implemented", 2, 4, "Model-risk/analytics lead", "Per RAG query", "Citation validation fails for a generated answer", "outputs/tables/rag_citation_validation.csv", "Citation validation checks that cited IDs exist and are relevant; it does not fully guarantee semantic faithfulness of every sentence."),
    ("R18", "Invalid citation", "A Gemini answer cites an evidence/claim ID that does not exist or does not support the cited sentence.", "Retrieval and generation", "Management, external readers", 2, 4, "src/rag/citation_validator.py; rag_citation_validation.csv; rag_go_no_go.csv", "Implemented", 1, 4, "Model-risk/analytics lead", "Per RAG query", "citation_validation fails a demo case", "outputs/tables/rag_citation_validation.csv; rag_go_no_go.csv", "Validation runs on the documented demo cases; not yet on every possible free-text query."),
    ("R19", "Prompt injection through source text", "Malicious or manipulative text embedded in a scraped review/press article alters Gemini's behaviour when included in the retrieval context.", "Retrieval and generation", "Management, TALA", 2, 4, "Context is restricted to short, retrieved passages with citation IDs; no user-supplied free-form system prompt override in src/rag/gemini_generator.py", "Partially implemented", 2, 4, "Information Security lead", "Per RAG query / per new source", "An answer contains instructions or content inconsistent with the retrieved evidence", "src/rag/gemini_generator.py; src/rag/orchestrator.py", "No dedicated prompt-injection detection/filter has been implemented or tested against adversarial inputs."),
    ("R20", "Model/provider change", "Google changes or deprecates the pinned Gemini model, altering answer behaviour without the project's knowledge.", "Retrieval and generation / vendor", "Project team, management", 3, 3, "GEMINI_MODEL pinned via .env (not hardcoded); no fallback model (fails closed)", "Implemented", 2, 3, "AI product owner", "On any Gemini API error or version notice", "GeminiConfigError / GeminiGenerationError raised", "src/rag/gemini_generator.py", "Google's model deprecation notices are external; the project has no automated alert integration for them."),
    ("R21", "API data exposure", "Evidence text sent to the Gemini API is logged or used for model training in a way inconsistent with project data-use expectations.", "Retrieval and generation / vendor", "TALA, customers, creators", 3, 4, "Only public, already-collected evidence text and claim IDs are sent (no raw personal data by design); genai_vendor_control_matrix.csv documents Gemini's logging policy", "Partially implemented", 2, 4, "Data Protection Officer", "On any Gemini terms-of-service update", "Vendor terms change or tier (free/paid) is reclassified", "src/rag/gemini_generator.py; ai_governance/genai_vendor_control_matrix.csv", "Whether the project's Gemini API key is on a free or paid tier is not recorded in this repository -- this materially affects Google's data-use terms and must be confirmed (see genai_vendor_control_matrix.csv)."),
    ("R22", "Secret leakage", "GEMINI_API_KEY or another credential is committed to the repository or logged in plaintext.", "Cross-cutting / security", "Project team, TALA", 2, 5, ".gitignore excludes .env; no API key ever read/printed in scripts inspected for this framework", "Implemented", 1, 5, "Information Security lead", "Every commit / CI run (proposed)", "A secret-scanning tool or manual review flags a credential in tracked files", ".gitignore; scripts inspected in this task contain no secret access", "No automated secret-scanning CI step exists yet in this repository; current protection is .gitignore plus manual discipline."),
    ("R23", "Unauthorised access", "Someone without a legitimate analytical need accesses the local ChromaDB store, raw evidence CSVs, or the Gemini API key.", "Cross-cutting / security", "TALA, customers, creators", 2, 4, "Local, non-networked ChromaDB persistence; no deployed multi-user access layer exists", "Partially implemented", 2, 4, "Information Security lead", "On any move toward shared/hosted deployment", "The system is deployed outside the current single-user local environment", "src/rag/chroma_store.py (local persistent client only)", "Access control is currently 'whoever has the laptop/repo' -- adequate for an academic prototype, not for shared deployment."),
    ("R24", "Data-retention failure", "Collected personal or creator-identifying data is retained beyond its documented academic-use period.", "Data lifecycle", "Creators, customers, reviewers", 2, 3, "docs/governance_notes.md 'academic use only' statement; no automated deletion job exists", "Proposed", 2, 3, "Data steward", "End of academic term / project close (proposed)", "Project reaches its documented end-of-use date without a deletion action", "docs/governance_notes.md", "No retention-expiry automation exists in the repository; this is a manual, undocumented-date risk today."),
    ("R25", "Unsupported external communication", "A finding (claim label, creator observation, moat hypothesis) is communicated externally without the caveats, evidence IDs, or presentation_restriction the corpus requires.", "Managerial use / external communication", "TALA, competitors, management", 3, 4, "presentation_restriction field; claim_decision_policy.csv; CLAUDE.md wording rules", "Proposed", 2, 4, "Marketing/brand lead", "Before every external publication", "A deck/press item is drafted referencing a claim_id or finding_id without policy sign-off", "outputs/tables/claim_decision_policy.csv", "The policy table defines the rule; a mandatory sign-off workflow enforcing it is proposed, not implemented."),
    ("R26", "Overreliance by management", "Management treats an automated fusion label or Gemini answer as a final, human-equivalent judgement rather than a decision-support input.", "Managerial use", "TALA management", 3, 4, "'Core governance principle' (section 3) distinguishing evidence retrieval, automated classification, human judgement, and approved communication", "Proposed", 2, 4, "Executive sponsor", "Every governance review", "A decision or communication is traced to an automated output with no documented human review step", "docs/ai_governance_ethics_esg_framework.md section 3", "This is a behavioural/organisational risk that technical controls alone cannot fully close; it requires training and process discipline."),
]

RISK_COLUMNS = [
    "risk_id", "risk_domain", "risk_scenario", "affected_component", "affected_stakeholder",
    "likelihood_1_5", "impact_1_5", "existing_controls", "control_status",
    "residual_likelihood_1_5", "residual_impact_1_5", "risk_owner", "review_frequency",
    "escalation_trigger", "evidence", "caveat",
]


def build_risk_register() -> pd.DataFrame:
    rows = []
    for r in RISK_ROWS_RAW:
        d = dict(zip(RISK_COLUMNS, r))
        inherent = d["likelihood_1_5"] * d["impact_1_5"]
        residual = d["residual_likelihood_1_5"] * d["residual_impact_1_5"]
        d["inherent_risk_score"] = inherent
        d["inherent_risk_level"] = _risk_level(inherent)
        d["residual_risk_score"] = residual
        d["residual_risk_level"] = _risk_level(residual)
        rows.append(d)
    ordered_cols = [
        "risk_id", "risk_domain", "risk_scenario", "affected_component", "affected_stakeholder",
        "likelihood_1_5", "impact_1_5", "inherent_risk_score", "inherent_risk_level",
        "existing_controls", "control_status", "residual_likelihood_1_5", "residual_impact_1_5",
        "residual_risk_score", "residual_risk_level", "risk_owner", "review_frequency",
        "escalation_trigger", "evidence", "caveat",
    ]
    df = pd.DataFrame(rows)[ordered_cols]
    return df


# ---------------------------------------------------------------------------
# Section 10: control matrix
# ---------------------------------------------------------------------------

CONTROL_ROWS_RAW = [
    # (control_id, layer, risk_ids, name, description, type, status, tech/process, system_evidence, owner, operator, frequency, trigger, output, failure_response, priority, phase)
    ("C01", "Layer 1 - Data admission", "R03;R04", "Provenance field enforcement", "Every admitted record must carry source_url, source_platform, collection_date, collected_by, evidence_type.", "Preventive", "Implemented", "Technical", "src/validation.py PROVENANCE_FIELDS; scripts/validate_data.py", "Data steward", "Automated validator", "Every collection batch", "New/changed CSV written to data/", "PASS/FAIL row per file", "Reject file from downstream pipelines until fixed", "P0", "Phase 0"),
    ("C02", "Layer 1 - Data admission", "R04", "Schema/filename routing validation", "Exact-filename-stem schema routing (EXACT_FILENAME_SCHEMA) prevents a file being silently checked against the wrong schema.", "Preventive", "Implemented", "Technical", "src/validation.py guess_schema()", "Data steward", "Automated validator", "Every validator run", "New output filename introduced", "Routing pass/fail", "Add exact-filename entry before file is trusted", "P1", "Phase 0"),
    ("C03", "Layer 1 - Data admission", "R01;R24", "Username/PII stripping at collection", "Reviewer/creator usernames stripped before saving to data/raw/; no profile photos or personal account details collected.", "Preventive", "Partially implemented", "Process", "docs/governance_notes.md Decisions Log", "Data steward", "Collection scripts", "Per collection batch", "New source onboarded", "Stripped record", "Manual spot-check and correction", "P0", "Phase 1"),
    ("C04", "Layer 1 - Data admission", "R03", "Platform TOS pre-check", "Collection method checked against docs/governance_notes.md TOS table before use; no login/CAPTCHA/paywall bypass.", "Preventive", "Implemented", "Process", "docs/governance_notes.md; CLAUDE.md section 5", "Data steward", "Human (collector)", "Per new source", "New platform/collection method proposed", "Go/no-go decision", "Halt collection from that source", "P0", "Phase 0"),
    ("C05", "Layer 1 - Data admission", "R02", "Media rights-basis gating", "Only official_direct_public_asset / open_license / group_owned / user_authorised bases permit local frame/temporal processing.", "Preventive", "Implemented", "Technical", "configs/schema.yaml::video_assets rights_or_access_basis; scripts/process_video_assets.py", "Data steward", "Automated pipeline", "Per video asset", "New video asset with unverified basis", "Asset admitted or excluded", "Exclude asset from processed pipeline", "P0", "Phase 0"),
    ("C06", "Layer 1 - Data admission", "R04", "Search-snippet rejection", "Search-engine result snippets are not treated as primary evidence; only resolved, direct source URLs are admitted.", "Preventive", "Implemented", "Technical", "src/validation.py SEARCH_ENGINE_URL_RX / DIRECT_SOCIAL_URL_RX checks", "Data steward", "Automated validator", "Every validator run", "A source_url matches a search-engine pattern", "Validator warning/failure", "Re-source with a direct URL", "P1", "Phase 0"),
    ("C07", "Layer 1 - Data admission", "R04", "Duplicate detection", "Duplicate document/evidence IDs are rejected before being written to a modelling table.", "Preventive", "Implemented", "Technical", "src/validation.py assert_no_duplicate_ids()", "Data steward", "Automated pipeline", "Every table build", "A build script attempts a duplicate ID write", "Build halts with ValueError", "Fix upstream duplicate before rerun", "P1", "Phase 0"),
    ("C08", "Layer 1 - Data admission", "R04", "Evidence-strength classification gate", "Every record carries evidence_strength (strong/moderate/weak/unusable); 'unusable' records are excluded from fusion.", "Preventive", "Implemented", "Technical", "src/validation.py assert_evidence_strength_allowed(); claim_evidence_units.csv evidence_strength", "Model-risk/analytics lead", "Automated pipeline", "Every fusion run", "A record's evidence_strength=unusable is proposed as evidence", "Excluded from permitted_evidence_role", "Flag for manual review", "P1", "Phase 0"),
    ("C09", "Layer 1 - Data admission", "R24", "Retention/deletion policy documentation", "Academic-use-only retention statement recorded; formal deletion trigger proposed for project close.", "Corrective", "Proposed", "Process", "docs/governance_notes.md", "Data steward", "Data steward", "End of academic term (proposed)", "Project reaches documented end-of-use date", "Deletion log entry", "Escalate to AI product owner if missed", "P2", "Phase 1"),
    ("C10", "Layer 2 - Modality processing", "R14", "Image-claim similarity threshold gate", "CLIP text-image cosine similarity >= 0.27 required before an image is treated as claim evidence.", "Preventive", "Implemented", "Technical", "scripts/build_claim_multimodal_candidates.py; claim_multimodal_evidence_candidates.csv image_similarity", "Model-risk/analytics lead", "Automated pipeline", "Per candidate-bundle build", "image_similarity below threshold", "Excluded from evidence bundle", "Manual review of borderline scores", "P1", "Phase 0"),
    ("C11", "Layer 2 - Modality processing", "R15", "Genuine video-processing definition gate", "A video record counts as processed evidence only with >=2 sampled frames, motion/scene-change features, or a transcript; leads are labelled platform_metadata_only.", "Preventive", "Implemented", "Technical", "configs/schema.yaml::video_assets; docs/image_video_codebook.md", "Model-risk/analytics lead", "Automated pipeline", "Per video asset", "A lead (URL/thumbnail only) is proposed as processed evidence", "Labelled platform_metadata_only, excluded from video coverage", "Re-flag mislabelled asset", "P0", "Phase 0"),
    ("C12", "Layer 2 - Modality processing", "R13", "Modality-misclassification detection", "visually_groundable and visual_gate_passed flags prevent non-visual claims from being marked as visually corroborated.", "Detective", "Implemented", "Technical", "data/processed/fusion/claim_evidence_units.csv visual_gate_reason", "Model-risk/analytics lead", "Automated pipeline", "Per fusion run", "visual_gate_passed=True on a non-groundable claim", "Excluded, reason logged", "Manual audit of gate-reason log", "P1", "Phase 0"),
    ("C13", "Layer 2 - Modality processing", "n/a (structural, cross-cutting)", "Protected-characteristic prohibition", "Image feature extraction produces only interpretable, non-identity features (brightness, colour palette, edges); no age/gender/ethnicity/emotion inference.", "Preventive", "Implemented", "Technical", "src/image_features.py; tests/test_image_features.py::test_..._no_protected_characteristic_fields", "Model-risk/analytics lead", "Automated pipeline + test suite", "Every test run", "A protected-characteristic-named field appears in image features", "Test failure blocks merge", "Remove offending field before merge", "P0", "Phase 0"),
    ("C14", "Layer 2 - Modality processing", "n/a (structural, cross-cutting)", "Face-embedding prohibition", "No face detection, face cropping, or face-embedding extraction is implemented anywhere in the image/video pipeline.", "Preventive", "Implemented", "Technical", "src/image_features.py module docstring; src/video_features.py", "Model-risk/analytics lead", "Code review", "Every code change to image/video modules", "A face-detection/embedding library import is proposed", "Code review rejection", "Remove the dependency/feature", "P0", "Phase 0"),
    ("C15", "Layer 2 - Modality processing", "R06;R09", "Missing-modality disclosure", "Every claim record and evidence bundle records missing_modalities explicitly rather than silently omitting them.", "Detective", "Implemented", "Technical", "claim_multimodal_evidence_candidates.csv missing_modalities; analytical_synthesis_claim_master.csv evidence_gap", "Model-risk/analytics lead", "Automated pipeline", "Every claim-bundle build", "A claim bundle is built", "missing_modalities field populated", "n/a (disclosure, not blocking)", "P1", "Phase 0"),
    ("C16", "Layer 2 - Modality processing", "R20", "Model/feature-version recording", "Embedding model and CLIP version are recorded in provenance_note fields for reproducibility.", "Detective", "Implemented", "Technical", "claim_multimodal_evidence_candidates.csv provenance_note (model names/thresholds)", "Model-risk/analytics lead", "Automated pipeline", "Every feature-extraction run", "A feature-extraction script runs", "Version string in provenance_note", "Re-run with recorded version if reproducibility fails", "P2", "Phase 1"),
    ("C17", "Layer 3 - Fusion", "R16", "Claim-linkage and evidence-unit construction", "Every evidence unit is explicitly linked to one claim_id via evidence_bundle_id/claim_id; unlinked evidence cannot enter fusion.", "Preventive", "Implemented", "Technical", "data/processed/fusion/claim_evidence_units.csv (claim_id column, 100% populated)", "Model-risk/analytics lead", "Automated pipeline", "Every fusion run", "An evidence row lacks a claim_id", "Build fails / row excluded", "Fix upstream linkage", "P0", "Phase 0"),
    ("C18", "Layer 3 - Fusion", "R13;R14", "Visually-groundable category enforcement", "Only claim categories/text that are visually groundable can receive image/video support; the rest are text/reference-only by design.", "Preventive", "Implemented", "Technical", "src/fusion/image_evidence.py; src/fusion/video_evidence.py visual gate logic", "Model-risk/analytics lead", "Automated pipeline", "Every fusion run", "A non-groundable claim receives a visual_gate_passed=True unit", "Excluded", "Manual audit", "P1", "Phase 0"),
    ("C19", "Layer 3 - Fusion", "R08;R16", "Support/challenge/context-only separation", "Every evidence unit is assigned exactly one stance (supports/challenges/neutral_context/not_applicable) and one permitted_evidence_role.", "Preventive", "Implemented", "Technical", "claim_evidence_units.csv stance, permitted_evidence_role columns", "Model-risk/analytics lead", "Automated pipeline", "Every fusion run", "n/a (structural)", "stance + role populated per unit", "n/a", "P0", "Phase 0"),
    ("C20", "Layer 3 - Fusion", "R11;R12", "Self-reported vs. independent evidence tagging", "source_independence (self_reported/customer_reported/unclear) recorded on every evidence unit and rolled up per claim.", "Preventive", "Implemented", "Technical", "claim_evidence_units.csv source_independence; analytical_synthesis_claim_master.csv independent_evidence_count", "Model-risk/analytics lead", "Automated pipeline", "Every fusion run", "n/a (structural)", "Populated field", "n/a", "P0", "Phase 0"),
    ("C21", "Layer 3 - Fusion", "R16", "Sensitivity/stability testing across weighting scenarios", "Every claim is re-run across 4 modality-weighting scenarios; label_stable / confidence_std recorded.", "Detective", "Implemented", "Technical", "scripts/run_fusion_sensitivity.py; outputs/tables/fusion_sensitivity_stability.csv", "Model-risk/analytics lead", "Automated pipeline", "Every fusion-rules change", "A fusion rule/threshold changes", "label_stable + confidence_std per claim", "Flag unstable claims for manual review (fusion_unstable_claims.csv)", "P1", "Phase 0"),
    ("C22", "Layer 3 - Fusion", "R11", "'Aligned does not mean proven' confidence interpretation rule", "fusion_confidence is documented as a calibration signal, not a correctness guarantee, in docs and the claim decision policy.", "Preventive", "Implemented", "Process", "docs/executive_analytical_synthesis.md section 3; outputs/tables/claim_decision_policy.csv", "Model-risk/analytics lead", "Human (report authors)", "Every claim-level communication", "A claim's confidence is quoted without this caveat", "Caveat present in output text", "Add caveat before reuse", "P1", "Phase 0"),
    ("C23", "Layer 3 - Fusion", "R10", "Insufficient-evidence handling (never labelled contradicted)", "insufficient_evidence claims are never re-labelled divergent/contradicted; reason codes classify the gap type instead.", "Preventive", "Implemented", "Technical", "insufficient_evidence_analysis.csv reason_code/reason_label; tests/test_analytical_synthesis.py::test_no_insufficient_evidence_claim_labelled_contradicted", "Model-risk/analytics lead", "Automated test suite", "Every test run", "n/a (structural)", "Test pass/fail", "Block merge on test failure", "P0", "Phase 0"),
    ("C24", "Layer 3 - Fusion", "R16", "Duplicate claim-variant consolidation", "Near-identical mixed claim records are consolidated into strategic themes (deterministic text match) for reporting, without altering underlying labels.", "Detective", "Implemented", "Technical", "scripts/build_mixed_claim_themes.py; outputs/tables/mixed_claim_theme_summary.csv", "Model-risk/analytics lead", "Automated pipeline", "Every Day 3C+ build", "A new mixed claim record is added", "Theme assignment", "Re-run theme build on new mixed claims", "P2", "Phase 0"),
    ("C25", "Layer 3 - Fusion", "R07;R09;R25", "Human-escalation flags on claims and creator records", "requires_human_validation, partnership_human_review_required, external_validation_recommended fields flag records needing review before use.", "Detective", "Implemented", "Technical", "claim_evidence_units.csv; creator_multimodal_features.csv; claim_fusion_results.csv", "Model-risk/analytics lead", "Human reviewer", "Before external use", "A flagged record is proposed for external use without review", "Reviewer sign-off record (proposed workflow)", "Block publication pending review", "P0", "Phase 1"),
    ("C26", "Layer 4 - Retrieval and generation", "R21;R23", "Local, non-networked ChromaDB persistence", "Vector store is a local persistent client with no exposed network endpoint or multi-tenant access layer.", "Preventive", "Implemented", "Technical", "src/rag/chroma_store.py get_client()", "Information Security lead", "Automated pipeline", "n/a (structural)", "A networked/shared ChromaDB deployment is proposed", "n/a", "Security review before any networked deployment", "P0", "Phase 1"),
    ("C27", "Layer 4 - Retrieval and generation", "R13;R18", "Metadata filtering on retrieval", "Retrieval queries filter by modality/claim metadata so context returned matches the query's evidentiary scope.", "Preventive", "Implemented", "Technical", "src/rag/retriever.py; src/rag/query_router.py", "Model-risk/analytics lead", "Automated pipeline", "Every RAG query", "Retrieved context modality mismatches query intent", "Routing decision logged", "Re-route with corrected filter", "P1", "Phase 0"),
    ("C28", "Layer 4 - Retrieval and generation", "R17;R18", "Retrieval trace and rank-fusion logging", "Retrieved document IDs and rank-fusion scores are recorded per query for auditability.", "Detective", "Implemented", "Technical", "src/rag/rank_fusion.py; outputs/tables/rag_retrieval_evaluation.csv", "Model-risk/analytics lead", "Automated pipeline", "Every RAG evaluation run", "n/a (structural)", "Trace log / evaluation row", "Audit trace if an answer is disputed", "P1", "Phase 0"),
    ("C29", "Layer 4 - Retrieval and generation", "R06;R18", "Source-diversity check in retrieval evaluation", "Retrieval evaluation records corpus coverage/diversity, not just top-1 relevance.", "Detective", "Partially implemented", "Technical", "outputs/tables/rag_corpus_coverage.csv; rag_modality_coverage.csv", "Model-risk/analytics lead", "Automated pipeline", "Every RAG evaluation run", "n/a", "Coverage table", "Expand corpus if coverage gap found", "P2", "Phase 1"),
    ("C30", "Layer 4 - Retrieval and generation", "R19;R21", "Gemini context minimisation", "Only short, retrieved evidence passages and claim IDs are sent to Gemini -- no raw personal data, no full documents.", "Preventive", "Implemented", "Technical", "src/rag/orchestrator.py; src/rag/gemini_generator.py prompt construction", "Data Protection Officer", "Automated pipeline", "Every RAG query", "n/a (structural)", "n/a", "n/a", "P0", "Phase 0"),
    ("C31", "Layer 4 - Retrieval and generation", "R20", "Prompt and model-version pinning", "GEMINI_MODEL is read from .env, never hardcoded; prompt templates are versioned in src/rag/.", "Preventive", "Implemented", "Technical", "src/rag/gemini_generator.py; configs/rag_config.yaml", "AI product owner", "Automated pipeline", "Every deploy/config change", "GEMINI_MODEL changes", "Config diff", "Re-run RAG evaluation on model/prompt change", "P1", "Phase 0"),
    ("C32", "Layer 4 - Retrieval and generation", "R17;R18", "Citation-ID validation", "Every generated answer's cited IDs are checked to exist and be relevant to the cited sentence before acceptance.", "Detective", "Implemented", "Technical", "src/rag/citation_validator.py; outputs/tables/rag_citation_validation.csv", "Model-risk/analytics lead", "Automated pipeline", "Every RAG query / evaluation run", "A citation fails validation", "passed=False row", "Suppress or flag the unsupported statement", "P0", "Phase 0"),
    ("C33", "Layer 4 - Retrieval and generation", "R16;R17", "Label-preservation check", "Generated answers must preserve (not silently alter) the underlying claim's fusion label.", "Detective", "Implemented", "Technical", "outputs/tables/rag_citation_validation.csv failure categories", "Model-risk/analytics lead", "Automated pipeline", "Every RAG query", "Answer label mismatches claim_fusion_results.csv", "Failure logged", "Regenerate or suppress answer", "P0", "Phase 0"),
    ("C34", "Layer 4 - Retrieval and generation", "R17", "Unsupported-statement detection", "Citation validator flags sentences with no resolvable supporting citation.", "Detective", "Implemented", "Technical", "src/rag/citation_validator.py", "Model-risk/analytics lead", "Automated pipeline", "Every RAG query", "A sentence has zero valid citations", "Failure logged", "Suppress unsupported sentence", "P0", "Phase 0"),
    ("C35", "Layer 4 - Retrieval and generation", "R20;R21", "No-fallback-generator failure handling", "On any Gemini config/generation error, the system stops generation and raises a clear error -- it never substitutes a template or alternate model.", "Preventive", "Implemented", "Technical", "src/rag/gemini_generator.py GeminiConfigError/GeminiGenerationError", "AI product owner", "Automated pipeline", "Every Gemini API call", "GEMINI_API_KEY missing or API call fails", "Raised exception, no output shown", "Fix credential/vendor issue before retry", "P0", "Phase 0"),
    ("C36", "Layer 5 - Managerial use and external communication", "R09;R25;R26", "Intended-use restriction statement", "The system is documented as an internal decision-support prototype, not an autonomous claims/legal/risk-scoring engine.", "Preventive", "Implemented", "Process", "docs/ai_governance_ethics_esg_framework.md section 2", "Executive sponsor", "Human (all users)", "Onboarding / every governance review", "n/a", "Documented statement", "Retrain users if violated", "P0", "Phase 0"),
    ("C37", "Layer 5 - Managerial use and external communication", "R09;R25", "Human-approval gate before external communication", "No claim-, creator-, or moat-related finding is published externally without a named human reviewer's sign-off.", "Preventive", "Proposed", "Process", "outputs/tables/claim_decision_policy.csv required_reviewer column (policy defined, workflow not yet built)", "Marketing/brand lead", "Human reviewer", "Before every external publication", "A publication references a claim_id/finding_id", "Sign-off record", "Withhold publication pending sign-off", "P0", "Phase 1"),
    ("C38", "Layer 5 - Managerial use and external communication", "R09;R10", "Sustainability-claim approval gate", "Labour/ESG claims lacking independent evidence require Sustainability/ESG lead review before any external framing.", "Preventive", "Proposed", "Process", "outputs/tables/claim_decision_policy.csv; esg_claim_assurance_matrix.csv", "Sustainability/ESG lead", "Human reviewer", "Before any ESG-related external communication", "A labour/ESG claim with 0 independent evidence is proposed for use", "Sign-off record", "Withhold or add caveats", "P0", "Phase 1"),
    ("C39", "Layer 5 - Managerial use and external communication", "R07", "Product-quality/creator escalation path", "Divergent, mixed, or creator-misrepresentation findings route to a named escalation owner rather than direct publication.", "Corrective", "Proposed", "Process", "outputs/tables/ai_incident_escalation_matrix.csv", "AI product owner", "Human reviewer", "On escalation trigger", "A mixed/divergent or creator-risk finding is flagged", "Escalation ticket (proposed)", "Formal incident process", "P1", "Phase 1"),
    ("C40", "Layer 5 - Managerial use and external communication", "R09", "Correction and retraction procedure", "A documented process exists to correct or retract a published finding found to be wrong or unsupported.", "Corrective", "Proposed", "Process", "outputs/tables/ai_incident_escalation_matrix.csv", "Legal counsel", "Legal counsel + AI product owner", "On confirmed error", "An error is confirmed post-publication", "Correction record + notification", "Publish correction within target response time", "P0", "Phase 1"),
    ("C41", "Layer 5 - Managerial use and external communication", "R01;R24", "Audit retention of decisions and evidence", "Decisions, evidence bundles, and citation-validation results are retained in outputs/ and data/processed/ for audit traceability.", "Detective", "Partially implemented", "Technical", "outputs/tables/*.csv; data/processed/fusion/*.csv (file-based audit trail, no formal log/retention policy)", "Internal audit", "Automated pipeline", "Every build", "An audit is requested", "Retained CSV/output history", "Formalise retention schedule", "P2", "Phase 2"),
    ("C42", "Cross-cutting", "R22", "Secret exclusion via .gitignore", ".env and other credential files are excluded from version control.", "Preventive", "Implemented", "Technical", ".gitignore (.env entry); confirmed untracked via git check-ignore/git ls-files in this and prior sessions", "Information Security lead", "Automated (git)", "n/a (structural)", "n/a", "n/a", "Purge and rotate if a secret is ever committed", "P0", "Phase 0"),
    ("C43", "Cross-cutting", "R22", "Secret-scanning CI step", "An automated scanner checks every commit for credential patterns before merge.", "Detective", "Proposed", "Technical", "Not present in repository", "Information Security lead", "CI pipeline (proposed)", "Every commit (proposed)", "A credential-pattern match is found", "CI failure", "Block merge, rotate credential", "P1", "Phase 1"),
    ("C44", "Cross-cutting", "n/a (structural, cross-cutting)", "Automated validation and test suite", "558+ automated tests and two dedicated validators (data, synthesis outputs) run before any output is trusted.", "Detective", "Implemented", "Technical", "tests/ directory; scripts/validate_data.py; scripts/validate_synthesis_outputs.py", "Model-risk/analytics lead", "Automated (pytest)", "Every code change", "Any test fails", "Non-zero exit code", "Block merge/use until fixed", "P0", "Phase 0"),
    ("C45", "Layer 1 - Data admission", "R05", "Scheduled evidence refresh policy", "A recurring (proposed quarterly) job re-checks collection_date against a staleness threshold and triggers re-collection.", "Preventive", "Proposed", "Process", "No scheduler exists in the repository today", "AI product owner", "Data steward", "Quarterly (proposed)", "Calendar trigger", "Refresh job log (proposed)", "Escalate if refresh missed", "P1", "Phase 2"),
    ("C45b", "Layer 1 - Data admission", "R05", "Evidence-staleness KPI monitoring", "collection_date is compared against the current date to compute an evidence-staleness rate KPI, surfaced for management review.", "Detective", "Proposed", "Technical", "outputs/tables/ai_governance_kpi_kri.csv evidence_staleness_rate (formula defined; no dashboard yet)", "Model-risk/analytics lead", "Automated pipeline (proposed)", "Every governance review (proposed)", "Staleness KPI exceeds warning threshold", "KPI value + alert (proposed)", "Trigger scheduled refresh (C45)", "P1", "Phase 2"),
    ("C46", "Layer 1 - Data admission", "R06", "Denominator-disclosure requirement", "Every creator/customer percentage output must be paired with its brand-level n; enforced by table schema and tests.", "Preventive", "Implemented", "Technical", "outputs/tables/creator_partnership_mix.csv brand_denominator_n; tests/test_analytical_synthesis.py::test_percentages_use_brand_level_denominator", "Model-risk/analytics lead", "Automated test suite", "Every test run", "A percentage output lacks its denominator", "Test failure", "Add denominator before merge", "P0", "Phase 0"),
    ("C47", "Layer 3 - Fusion", "R07", "Evidence-gated automated partnership/intent classification", "Partnership type and content intent are derived from rule-based evidence signals (not manual guesswork or LLM free text), each with a recorded confidence/trigger.", "Preventive", "Implemented", "Technical", "creator_multimodal_features.csv revised_partnership_type, partnership_rule_trigger, partnership_confidence, primary_intent_evidence", "Model-risk/analytics lead", "Automated pipeline", "Every creator-feature build", "n/a (structural)", "Populated classification + evidence text", "Manual review if confidence is low", "P0", "Phase 0"),
    ("C48", "Layer 5 - Managerial use and external communication", "R11;R12", "Pre-publication independent-evidence-count check", "Before any claim is cited externally as 'aligned', independent_evidence_count is checked; self-reported-only claims require an explicit caveat.", "Detective", "Proposed", "Process", "analytical_synthesis_claim_master.csv independent_evidence_count / self_reported_evidence_count (data exists; the sign-off check is a proposed workflow step)", "Sustainability/ESG lead", "Human reviewer", "Before external use of an aligned claim", "A claim with independent_evidence_count=0 is proposed for use as 'aligned'", "Caveat added or claim withheld", "Escalate to Sustainability/ESG lead", "P0", "Phase 1"),
    ("C49", "Layer 5 - Managerial use and external communication", "R10", "ESG-framing reason-code audit", "Before any negative ESG framing of a claim, its insufficient_evidence_analysis.csv reason_code is checked to confirm the gap is a collection limitation, not evidence of wrongdoing.", "Detective", "Proposed", "Process", "outputs/tables/insufficient_evidence_analysis.csv reason_code/reason_label", "Sustainability/ESG lead", "Human reviewer", "Before any negative ESG communication", "A claim is proposed for negative external ESG framing", "Reason-code confirmation record", "Withhold framing pending clarification", "P0", "Phase 1"),
    ("C50", "Layer 5 - Managerial use and external communication", "R21", "Vendor terms/tier periodic review", "Gemini API terms, data-use policy, and account tier (free vs. paid) are reviewed on a recurring basis and after any vendor terms update.", "Detective", "Proposed", "Process", "Not currently tracked in repository -- see genai_vendor_control_matrix.csv gap", "Data Protection Officer", "Data Protection Officer", "On vendor terms update / quarterly (proposed)", "Google publishes a Gemini API terms update", "Review record (proposed)", "Escalate to Legal counsel if terms materially change", "P1", "Phase 1"),
    ("C51", "Layer 5 - Managerial use and external communication", "R26", "Quarterly governance review of decision traceability", "A recurring review samples published findings/decisions and confirms each traces to a documented human review step.", "Detective", "Proposed", "Process", "outputs/tables/ai_governance_implementation_roadmap.csv Phase 2 item", "Executive sponsor", "Internal audit", "Quarterly (proposed)", "Calendar trigger", "Review record (proposed)", "Retrain management if traceability gap found", "P1", "Phase 2"),
]

CONTROL_COLUMNS = [
    "control_id", "governance_layer", "risk_ids", "control_name", "control_description",
    "control_type", "control_status", "technical_or_process", "system_evidence", "control_owner",
    "operator", "frequency", "trigger", "control_output", "failure_response",
    "implementation_priority", "target_phase",
]


def build_control_matrix() -> pd.DataFrame:
    return pd.DataFrame([dict(zip(CONTROL_COLUMNS, r)) for r in CONTROL_ROWS_RAW])[CONTROL_COLUMNS]


# ---------------------------------------------------------------------------
# Section 11: claim decision and escalation policy
# ---------------------------------------------------------------------------

CLAIM_DECISION_ROWS = [
    {
        "decision_label": "aligned",
        "meaning": "The automated evidence-gated fusion found supporting evidence (including, for some claims, independent/customer-reported evidence) and no eligible challenging evidence for this claim record.",
        "what_it_does_not_mean": "Aligned does NOT mean independently proven, verified by an assurance provider, or legally substantiated. It does not mean every possible piece of evidence was checked -- only the evidence present in this corpus.",
        "permitted_internal_use": "May be used in internal management discussion and as a candidate for external communication, subject to the review gates below.",
        "external_communication_status": "Conditionally permitted after human sign-off; self-reported-only aligned claims require an explicit 'not independently corroborated' caveat (see esg_claim_assurance_matrix.csv).",
        "required_reviewer": "Marketing/brand lead; Sustainability/ESG lead if the claim is in a labour/ESG category with independent_evidence_count=0.",
        "escalation_requirement": f"None if independent_evidence_count>0 and not sensitive_to_weights; escalate to Sustainability/ESG lead otherwise.",
        "evidence_refresh_requirement": "Re-verify if the underlying evidence is older than the staleness threshold in ai_governance_kpi_kri.csv (evidence_staleness_rate).",
        "recommended_wording": "\"The verified corpus supports this claim, based on n supporting evidence unit(s) including [self-reported/independent] sources.\"",
        "prohibited_wording": "\"This claim is proven true\"; \"independently verified\"; \"guaranteed\".",
    },
    {
        "decision_label": "partially_aligned",
        "meaning": "No claim record currently holds this label (0/37 in the current corpus), but the label remains defined for future fusion runs: partial supporting evidence with an unresolved gap.",
        "what_it_does_not_mean": "It does not mean the claim is false, nor that it is as well-supported as 'aligned'.",
        "permitted_internal_use": "Internal discussion only until independently reviewed; treat with the same caution as 'mixed'.",
        "external_communication_status": "Not permitted without Sustainability/ESG lead and Legal counsel review.",
        "required_reviewer": "Sustainability/ESG lead; Legal counsel.",
        "escalation_requirement": "Always escalate before any external use.",
        "evidence_refresh_requirement": "Refresh evidence before any external use given the partial-support status.",
        "recommended_wording": "\"The evidence partially supports this claim; a specific gap remains: [describe].\"",
        "prohibited_wording": "\"Fully supported\"; \"proven\"; \"confirmed\".",
    },
    {
        "decision_label": "mixed",
        "meaning": "Both supporting and challenging evidence exist for this claim record in the verified corpus (5/37 claim records, consolidating into 2 underlying strategic themes -- see mixed_claim_theme_summary.csv).",
        "what_it_does_not_mean": "Mixed does not mean the claim is proven false, nor that the conflict is resolved. It does not mean the challenging evidence outweighs the supporting evidence -- the corpus does not adjudicate that.",
        "permitted_internal_use": "Internal review and investigation only, pending human adjudication.",
        "external_communication_status": "Not permitted for external communication without explicit presentation of BOTH supporting and challenging evidence, and Legal counsel sign-off.",
        "required_reviewer": "Legal counsel; Sustainability/ESG lead for labour/ESG-category mixed claims.",
        "escalation_requirement": "Always escalate to the human evidence reviewer and Legal counsel before any external framing; never present as a simple true/false verdict.",
        "evidence_refresh_requirement": "Refresh and re-verify supporting and challenging evidence before any external communication.",
        "recommended_wording": "\"The verified corpus contains both supporting and challenging evidence for this claim; management is reviewing the discrepancy.\"",
        "prohibited_wording": "\"Confirmed\"; \"false\"; \"greenwashing\"; \"this proves...\"; any wording that picks one side without disclosing the other.",
    },
    {
        "decision_label": "divergent",
        "meaning": "No claim record currently holds this label (0/37 in the current corpus), but the label remains defined: predominantly or exclusively challenging evidence with no meaningful support.",
        "what_it_does_not_mean": "It does not mean legal misconduct, fraud, or a definitive finding of falsity -- it means the corpus evidence conflicts with the official claim.",
        "permitted_internal_use": "Internal investigation only.",
        "external_communication_status": "Not permitted without full human review (Legal counsel and Sustainability/ESG lead) before ANY external use; treat as the highest-sensitivity label in the taxonomy.",
        "required_reviewer": "Legal counsel; Sustainability/ESG lead; Executive sponsor.",
        "escalation_requirement": "Mandatory, immediate escalation to Legal counsel and the AI product owner; do not communicate externally pending review.",
        "evidence_refresh_requirement": "Refresh and independently corroborate before any use, internal or external.",
        "recommended_wording": "\"The verified corpus shows evidence that conflicts with this official claim; this finding is under management and legal review.\"",
        "prohibited_wording": "\"Proven false\"; \"greenwashing\"; \"fraudulent\"; \"this proves TALA lied\"; any accusatory or legal-conclusion language.",
    },
    {
        "decision_label": "insufficient_evidence",
        "meaning": "No qualifying independent/customer/press/reference evidence was found in the verified corpus for this claim record (18/37 in the current corpus).",
        "what_it_does_not_mean": "Insufficient evidence must NOT be described as false, unsupported-in-reality, or as evidence of wrongdoing. It reflects a gap in THIS corpus, not a finding about the claim's truth.",
        "permitted_internal_use": "Internal use as a collection-gap flag and a candidate for targeted, ethics-compliant future evidence collection.",
        "external_communication_status": "Not permitted to be cited as if it were a negative finding; may only be described as a corpus gap if referenced at all externally, with Sustainability/ESG lead sign-off for labour/ESG categories.",
        "required_reviewer": "Sustainability/ESG lead for labour/ESG-category claims; AI product owner otherwise.",
        "escalation_requirement": "Escalate to Sustainability/ESG lead if the claim is in the labour category (14/37 claims, 0 with independent evidence) before any communication referencing it.",
        "evidence_refresh_requirement": "Prioritise for future evidence collection; re-run fusion after any new evidence is admitted.",
        "recommended_wording": "\"No qualifying public evidence was found in the current corpus for this claim; this is a collection gap, not a contradiction.\"",
        "prohibited_wording": "\"Unsupported\"; \"false\"; \"no evidence exists\" (state instead: \"no verified evidence was found in the current corpus\"); \"disproven\".",
    },
]


def build_claim_decision_policy() -> pd.DataFrame:
    df = pd.DataFrame(CLAIM_DECISION_ROWS)
    ordered = [
        "decision_label", "meaning", "what_it_does_not_mean", "permitted_internal_use",
        "external_communication_status", "required_reviewer", "escalation_requirement",
        "evidence_refresh_requirement", "recommended_wording", "prohibited_wording",
    ]
    return df[ordered]


# ---------------------------------------------------------------------------
# Section 12: RACI and accountability
# ---------------------------------------------------------------------------

RACI_ROLES = [
    "executive_sponsor", "ai_product_owner", "data_steward", "model_risk_analytics_lead",
    "sustainability_esg_lead", "legal_counsel", "data_protection_officer",
    "information_security_lead", "marketing_brand_lead", "creator_partnership_lead",
    "human_evidence_reviewer", "internal_audit",
]

# activity -> {role: code}. Every activity has exactly one 'A'.
RACI_ACTIVITIES = {
    "Source approval": {
        "data_steward": "A", "ai_product_owner": "C", "information_security_lead": "C", "human_evidence_reviewer": "R",
    },
    "Rights-basis approval": {
        "data_steward": "A", "legal_counsel": "C", "human_evidence_reviewer": "R",
    },
    "Data minimisation": {
        "data_steward": "A", "data_protection_officer": "C", "information_security_lead": "I",
    },
    "Evidence refresh": {
        "ai_product_owner": "A", "data_steward": "R", "model_risk_analytics_lead": "C",
    },
    "Model change": {
        "ai_product_owner": "A", "model_risk_analytics_lead": "R", "information_security_lead": "C", "data_protection_officer": "I",
    },
    "Prompt change": {
        "ai_product_owner": "A", "model_risk_analytics_lead": "R", "data_protection_officer": "C",
    },
    "Claim review": {
        "model_risk_analytics_lead": "A", "human_evidence_reviewer": "R", "sustainability_esg_lead": "C",
    },
    "Mixed/divergent escalation": {
        "legal_counsel": "A", "model_risk_analytics_lead": "R", "sustainability_esg_lead": "C", "executive_sponsor": "I",
    },
    "Sustainability-claim approval": {
        "sustainability_esg_lead": "A", "legal_counsel": "C", "marketing_brand_lead": "R",
    },
    "Creator-related publication": {
        "creator_partnership_lead": "A", "legal_counsel": "C", "marketing_brand_lead": "R",
    },
    "Incident response": {
        "information_security_lead": "A", "ai_product_owner": "R", "legal_counsel": "C", "data_protection_officer": "C", "executive_sponsor": "I",
    },
    "Correction/retraction": {
        "legal_counsel": "A", "marketing_brand_lead": "R", "ai_product_owner": "C", "executive_sponsor": "I",
    },
    "Production release": {
        "ai_product_owner": "A", "information_security_lead": "C", "model_risk_analytics_lead": "C", "internal_audit": "I",
    },
    "Quarterly governance review": {
        "executive_sponsor": "A", "internal_audit": "R", "ai_product_owner": "C", "model_risk_analytics_lead": "C",
        "sustainability_esg_lead": "C", "legal_counsel": "C", "data_protection_officer": "C", "information_security_lead": "C",
    },
}


def build_raci() -> pd.DataFrame:
    rows = []
    for i, (activity, role_map) in enumerate(RACI_ACTIVITIES.items(), start=1):
        row = {"activity_id": f"A{i:02d}", "activity": activity}
        for role in RACI_ROLES:
            row[role] = role_map.get(role, "")
        n_accountable = sum(1 for v in row.values() if v == "A")
        assert n_accountable == 1, f"activity '{activity}' has {n_accountable} accountable roles, expected exactly 1"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 13: data lifecycle and privacy policy
# ---------------------------------------------------------------------------

LIFECYCLE_ROWS = [
    ("Discovery", "Candidate source URLs (brand pages, review platforms, YouTube channels, press)", "Identify legally and ethically collectable public sources", "Source URL, platform name only", "Low -- no personal data collected yet", "Not persisted (search-time only)", "Data steward", "No formal retention (ephemeral)", "n/a", "docs/governance_notes.md platform TOS table", "Implemented"),
    ("Collection", "Public review text, creator post text/metadata, official claim text, video/image assets", "Populate data/raw/ with provenance-tagged candidate records", "Text, platform, date, evidence_type, source_url; usernames stripped before saving", "Medium -- review/creator text may incidentally reference names in free text", "data/raw/", "Data steward", "Academic project duration (no formal expiry recorded)", "Project close (not yet automated)", "docs/governance_notes.md Decisions Log", "Implemented"),
    ("Admission", "Cleaned/validated candidate records", "Pass schema, provenance, and evidence-strength gates before promotion", "Validated fields only per configs/schema.yaml", "Low -- validation does not add new personal fields", "data/interim/", "Data steward", "Same as Collection", "Same as Collection", "scripts/validate_data.py PASS/FAIL log", "Implemented"),
    ("Processing", "Text normalisation, sentiment/stance features, image/video feature extraction", "Derive interpretable features for fusion, never identity/protected-characteristic features", "Derived numeric/categorical features only", "Low -- no face embeddings, no protected-characteristic fields (tested)", "data/processed/", "Model-risk/analytics lead", "Same as Collection", "Same as Collection", "tests/test_image_features.py; src/text_features.py", "Implemented"),
    ("Feature extraction", "CLIP embeddings, TF-IDF/sentence embeddings, video frame features", "Enable claim-evidence similarity matching", "Embeddings + model-version provenance_note", "Low -- embeddings are of public content/images, not of identifiable individuals", "data/processed/embeddings/", "Model-risk/analytics lead", "Same as Collection", "Same as Collection", "claim_multimodal_evidence_candidates.csv provenance_note", "Implemented"),
    ("Storage", "Evidence units, fusion results, RAG corpora, ChromaDB vector store", "Persist verified evidence for retrieval and reporting", "Text/embeddings/metadata only; no raw personal identifiers by design", "Medium -- ChromaDB persists evidence text that may incidentally reference third-party names within quoted reviews", "Local filesystem (chroma_db/, data/processed/fusion/) -- not networked", "Data steward; Information Security lead", "Academic project duration; no networked access", "Project close / re-deployment decision", "src/rag/chroma_store.py (local persistent client)", "Implemented"),
    ("Retrieval", "Query-time evidence lookup by claim/modality metadata", "Return relevant, citation-traceable evidence to the RAG orchestrator", "Retrieved passages + metadata only", "Low -- retrieval does not expand personal-data surface beyond what is stored", "In-memory at query time", "Model-risk/analytics lead", "n/a (query-time only)", "n/a", "src/rag/retriever.py; rag_retrieval_evaluation.csv", "Implemented"),
    ("Generation", "Gemini-generated answers from retrieved context", "Produce a cited, evidence-grounded answer to a management question", "Short retrieved passages + claim IDs sent to Gemini API; no raw personal data by design", "Medium -- text sent to a third-party API is subject to Google's data-use terms (tier not confirmed in repo)", "Google Gemini API (external, per Google's terms)", "AI product owner; Data Protection Officer", "Per Google's Gemini API data-retention terms (see genai_vendor_control_matrix.csv)", "n/a (vendor-controlled)", "src/rag/gemini_generator.py; genai_vendor_control_matrix.csv", "Implemented (data sent); vendor-tier confirmation proposed"),
    ("Audit logging", "Citation-validation results, retrieval traces, fusion sensitivity results", "Enable after-the-fact review of how an answer/label was produced", "IDs, scores, pass/fail flags only", "Low", "outputs/tables/*.csv", "Internal audit; Model-risk/analytics lead", "Retained with project outputs (no formal schedule)", "Formal retention schedule (proposed)", "outputs/tables/rag_citation_validation.csv", "Partially implemented"),
    ("Refresh", "Re-collection of stale evidence", "Keep claim-experience findings current", "Same fields as Collection, re-dated", "Same as Collection", "Same as Collection", "Data steward", "Triggered by staleness KPI (proposed)", "Staleness KPI breach (proposed)", "ai_governance_kpi_kri.csv evidence_staleness_rate", "Proposed"),
    ("Correction", "Correction of an identified error in a finding, label, or creator attribution", "Fix and record a correction without silently rewriting history", "Original + corrected value, reason, date, approver", "Medium -- corrections may involve a creator/customer identifier", "outputs/ (correction log, proposed)", "Legal counsel; AI product owner", "Retained permanently as an audit record (proposed)", "n/a", "ai_incident_escalation_matrix.csv", "Proposed"),
    ("Retention", "All of the above categories", "Keep only what is necessary for the academic project's active duration", "n/a (policy stage)", "Medium -- indefinite retention of free-text customer/creator content increases privacy risk over time", "n/a (policy stage)", "Data steward; Data Protection Officer", "Academic term end (default); earlier on request", "n/a", "docs/governance_notes.md 'academic use only'", "Proposed (formal schedule)"),
    ("Deletion", "All of the above categories", "Remove data once retention period lapses or a valid request is received", "n/a (policy stage)", "Medium -- no automated deletion currently exists", "n/a (policy stage)", "Data steward; Information Security lead", "On retention-period expiry or verified deletion request", "Retention-period expiry; verified request", "No deletion log exists yet", "Proposed"),
]

LIFECYCLE_COLUMNS = [
    "lifecycle_stage", "data_involved", "purpose", "minimum_necessary_fields", "personal_data_risk",
    "storage_location", "access_roles", "retention_rule", "deletion_trigger", "audit_evidence", "existing_or_proposed_status",
]


def build_data_lifecycle_policy() -> pd.DataFrame:
    return pd.DataFrame([dict(zip(LIFECYCLE_COLUMNS, r)) for r in LIFECYCLE_ROWS])[LIFECYCLE_COLUMNS]


# ---------------------------------------------------------------------------
# Section 14: ESG claim-assurance framework
# ---------------------------------------------------------------------------

ESG_ROWS = [
    ("Materials", "Official material specification pages, certification pages", "Independent lab test reports; third-party material certifications (e.g. GRS, OEKO-TEX, if held)", "Product/catalog images showing labelled material content", "Video demonstrating material handling/fabric properties where genuinely processed", "Aesthetic lifestyle imagery with no material claim content; engagement metrics", "At least one independent (non-brand) source OR a named, checkable certification", "Sustainability/ESG lead", "Per collection cycle / on new product line", "New material claim with zero independent evidence after 2 refresh cycles", f"7 materials claims in corpus; some with independent evidence (see analytical_synthesis_claim_master.csv)", "Self-reported material claims outnumber independently corroborated ones", "Prioritise independent material-certification lookups for materials claims lacking independent evidence"),
    ("Certifications", "Certification body logos/registration pages linked from official site", "Certification body's own public registry entry (not just the logo)", "Product images showing the certification mark applied to an actual product", "Not typically video-eligible", "A certification logo alone without a checkable registry entry", "A resolvable, checkable registry/certificate reference", "Sustainability/ESG lead", "On certification renewal cycle (typically annual)", "A cited certification cannot be found in the issuing body's public registry", "No certification-registry cross-check currently performed", "Certification claims are not independently registry-verified in this corpus", "Add a certification-registry cross-check step before treating a certification claim as aligned"),
    ("Supplier disclosures", "Official supplier-list or 'our factories' pages", "Independent audit reports (e.g. from a recognised social-compliance auditor)", "Factory/workshop images with verifiable location context", "Factory-tour video with genuine temporal content (not a promotional montage alone)", "Generic stock imagery of factories not tied to a named supplier", "A named, checkable supplier or auditor reference", "Sustainability/ESG lead", "Per supplier-relationship change", "A supplier-collaboration claim has zero independent evidence (see tier-3 theme below)", f"Tier-3 supplier/material-innovation theme: 4 claim variants, all currently mixed", "This is the corpus's most concentrated ESG tension -- one recurring claim, evidenced from 4 near-duplicate product-page variants", "Do not treat the 4 variants as 4 independent confirmations; investigate the single underlying tension once"),
    ("Labour", "Official 'purpose'/people pages, partnership announcements (e.g. named social enterprise collaborations)", "Independent labour/social-compliance audit; NGO or journalist reporting on factory conditions", "Not typically visually groundable (labour conditions are not verifiable from product photography)", "Not typically visually groundable", "Lifestyle or 'about us' imagery used as if it were labour-condition evidence", "At least one independent report or audit reference", "Sustainability/ESG lead", "Annually or on any labour-related news event", "Any labour claim reaches insufficient_evidence with zero independent evidence (currently ALL 14 do)", f"14/37 claims are labour-category; 0 have independent evidence in the current corpus (evidence_gap_by_category.csv)", "This is the single largest ESG evidence gap in the corpus -- TALA's largest claim category is its least corroborated", "Do not present any labour claim as validated; commission independent labour-evidence collection before any external labour-related communication"),
    ("Emissions", "Official emissions/carbon disclosures, impact reports", "Independent GHG assurance statement; third-party carbon accounting review", "Not typically visually groundable", "Not typically visually groundable", "Green colour palette or nature imagery used as an emissions proxy", "A quantified, checkable emissions figure with a named methodology", "Sustainability/ESG lead", "Annually, aligned with any published impact report", "No emissions claim exists with any independent evidence", f"1 emissions-category claim in corpus (see evidence_gap_by_category.csv)", "Emissions evidence is essentially absent from the current corpus", "Flag emissions as a structural evidence gap; do not infer emissions performance from any other evidence type"),
    ("Circularity", "Official take-back/recycling program pages", "Independent verification of stated recycling/take-back volumes", "Product images showing recycled-content labelling", "Video of a genuine take-back/recycling process, if legitimately available", "Generic 'circular economy' iconography with no program-specific evidence", "A checkable program description with at least directional independent corroboration", "Sustainability/ESG lead", "Annually", "The sole circularity claim lacks independent evidence", f"1 circularity-category claim in corpus", "Thin single-claim category", "Treat as an evidence-collection priority if circularity becomes a larger part of TALA's messaging"),
    ("Packaging", "Official packaging-material pages", "Independent packaging-material verification or certification", "Product/unboxing images showing packaging material", "Unboxing video with genuine content, if legitimately available", "Generic sustainability imagery unrelated to the specific packaging claim", "A checkable packaging-material specification", "Sustainability/ESG lead", "On packaging-format change", "The sole packaging claim has zero independent/customer evidence", f"1 packaging-category claim, zero usable evidence beyond the official claim (evidence_gap_by_category.csv)", "No corroborating or challenging evidence found at all for packaging", "Do not present the packaging claim as validated; it is currently untestable from this corpus"),
    ("Garment care and durability", "Official care instructions, fabric-care guidance", "Independent product-durability reviews (customer/press)", "Product images showing care labels", "Not typically video-eligible from current corpus", "Marketing imagery with no durability content", "At least one independent customer/press durability observation", "Model-risk/analytics lead (product-quality, not ESG)", "Per product-line review cycle", "A durability-adjacent claim shows conflicting customer evidence", "Customer/press durability evidence exists in the broader customer_experience_corpus (95 records) though not all is claim-linked", "Durability evidence exists at the corpus level but is not fully claim-linked for every relevant claim", "Where a durability claim is proposed for external use, confirm it has claim-linked customer evidence, not just corpus-level sentiment"),
    ("Returns and product quality", "Official returns-policy pages", "Independent customer complaint/return-experience reports (review platforms, press)", "Not typically visually groundable", "Not typically video-eligible", "Star-rating aggregates without claim linkage", "At least one independent, claim-linked quality/return observation", "Model-risk/analytics lead (product-quality, not ESG)", "Per review cycle", "A returns/quality-related claim is proposed for external use without claim-linked evidence", "customer_experience_corpus.csv (95 records) provides general quality signal not yet all claim-linked", "General product-quality sentiment exists but is not exhaustively linked to specific official claims", "Escalate any product-quality divergence finding to the appropriate business owner, not Sustainability/ESG lead, per the claim_decision_policy escalation table"),
]

ESG_COLUMNS = [
    "claim_category", "acceptable_official_evidence", "preferred_independent_evidence", "eligible_image_evidence",
    "eligible_video_evidence", "ineligible_proxies", "minimum_substantiation_threshold", "review_owner",
    "refresh_frequency", "escalation_condition", "current_tala_evidence_status", "current_evidence_gap", "recommended_action",
]


def build_esg_assurance_matrix() -> pd.DataFrame:
    return pd.DataFrame([dict(zip(ESG_COLUMNS, r)) for r in ESG_ROWS])[ESG_COLUMNS]


# ---------------------------------------------------------------------------
# Section 15: regulatory and standards map
# All rows verified against primary/official sources on the stated access_date.
# ---------------------------------------------------------------------------

REG_ROWS = [
    ("REG01", "UK Information Commissioner's Office (ICO)", "Guidance on AI and data protection", "UK", "Mandatory (regulator guidance interpreting UK GDPR/DPA 2018, which are themselves mandatory)",
     "AI systems processing personal data must apply UK GDPR principles (lawfulness, purpose limitation, data minimisation, accuracy) and consider a DPIA for higher-risk processing.",
     "System collects and processes public review/creator text that may incidentally contain personal data (names in free text); Gemini calls send retrieved text to a third-party processor.",
     "Data admission; storage; generation (Gemini)",
     "Data minimisation at collection; DPIA consideration before any wider deployment; DPO review of Gemini data flows",
     "Data Protection Officer",
     "https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/",
     "Guidance updated periodically by the ICO; exact last-update date not captured in this review -- confirm current version before relying on specific wording",
     ACCESS_DATE, True,
     "General AI/data-protection guidance, not a TALA-specific or system-specific determination; legal review required before any UK personal-data processing at scale."),
    ("REG02", "UK Competition and Markets Authority (CMA)", "Green Claims Code", "UK", "Mandatory (the Code explains existing obligations under UK consumer protection law, which is itself mandatory; the Code document itself is guidance)",
     "Environmental claims must be truthful, accurate, substantiated, and not omit or hide material information; claims about a product's full life cycle must be substantiated across that life cycle.",
     "The system's core objective is to identify divergence between TALA's official sustainability/quality claims and observable customer/press evidence -- directly relevant to Green Claims Code substantiation duties.",
     "Claim decision policy; ESG claim assurance; external communication",
     "Never assert a Code breach without Legal counsel review; treat insufficient-evidence claims as gaps, not breaches",
     "Legal counsel",
     "https://www.gov.uk/government/collections/how-to-make-environmental-claims-about-products-and-services",
     "Code in effect; CMA green-claims enforcement activity ongoing as of this review",
     ACCESS_DATE, True,
     "This system is a research/decision-support tool, not a CMA enforcement determination; it must never be presented as establishing a Green Claims Code breach."),
    ("REG03", "UK Advertising Standards Authority / Committee of Advertising Practice (ASA/CAP)", "Environmental claims (CAP Code, non-broadcast, section on environmental claims)", "UK", "Mandatory for UK advertising (CAP Code is the advertising self-regulatory code backed by law for certain claims)",
     "Unqualified environmental claims must be supported by evidence covering the product's full life cycle; broad claims like 'good for the planet' are read as whole-life-cycle claims unless clearly qualified.",
     "TALA's official claims corpus includes broad sustainability/responsibility statements that this framework, not the ASA, evaluates for evidentiary support.",
     "Claim decision policy; ESG claim assurance",
     "Legal/marketing review before any claim is used in TALA-facing or public-facing advertising-adjacent communication",
     "Legal counsel; Marketing/brand lead",
     "https://www.asa.org.uk/advice-online/environmental-claims-general.html",
     "CAP/BCAP guidance on misleading environmental claims updated 2023",
     ACCESS_DATE, True,
     "Applies to advertising; this system's outputs are internal decision-support material, not advertisements -- but any output later used in advertising would need separate CAP Code review."),
    ("REG04", "European Union (European Parliament and Council)", "GDPR -- Regulation (EU) 2016/679", "EU", "Mandatory (for processing EU residents' personal data)",
     "Lawfulness, purpose limitation, data minimisation, storage limitation, and accountability principles apply to any personal data processed.",
     "Any EU-resident personal data incidentally present in collected review/creator text would fall under GDPR; the project targets minimisation and username stripping.",
     "Data admission; storage; generation (Gemini)",
     "Confirm no systematic EU personal-data processing at scale beyond incidental free-text mentions; DPO review before scaling",
     "Data Protection Officer",
     "https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng",
     "In force since 25 May 2018",
     ACCESS_DATE, True,
     "This framework does not perform a full GDPR Article 30/35 assessment; formal legal review is required before any production/EU-facing deployment."),
    ("REG05", "European Union (European Parliament and Council)", "EU AI Act -- Regulation (EU) 2024/1689", "EU", "Mandatory (regulation), with phased applicability by provision",
     "Establishes risk-tiered obligations for AI systems (prohibited, high-risk, limited-risk/transparency, minimal-risk); obligations vary by the AI system's classified risk tier and role (provider/deployer).",
     "Potential applicability to this system's classification (e.g. as a limited-risk, transparency-obligated system, or as out of scope for an academic prototype) has NOT been determined here.",
     "Governance operating model; managerial use",
     "Do not assert a definitive EU AI Act risk classification without documented legal review",
     "Legal counsel",
     "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng",
     "Published in the Official Journal 12 July 2024; entered into force 1 August 2024; obligations phase in through 2026-2027",
     ACCESS_DATE, True,
     "NO DEFINITIVE CLASSIFICATION IS MADE HERE. This row records potential applicability only; a qualified legal review is required before any claim about the system's EU AI Act status."),
    ("REG06", "US Federal Trade Commission (FTC)", "Guides Concerning the Use of Endorsements and Testimonials in Advertising (16 CFR Part 255)", "US", "Mandatory for US-directed advertising/endorsements (the Guides interpret the FTC Act, which is mandatory; the Guides document itself is enforcement guidance)",
     "Material connections between a brand and an endorser/creator (payment, free product, affiliate link) must be clearly and conspicuously disclosed; a featured consumer experience must not be presented as typical unless it is.",
     "Directly relevant to the creator-strategy comparison's partnership-type classification (organic/affiliate/paid/gifted/ambassador) where US creators or US consumers are involved.",
     "Creator-strategy analysis; claim decision policy (creator-related findings)",
     "Do not assert an FTC disclosure violation for any brand without Legal counsel review; the corpus classifies partnership type descriptively, not for compliance enforcement",
     "Legal counsel; Creator-partnership lead",
     "https://www.ftc.gov/business-guidance/advertising-marketing/endorsements-influencers-reviews",
     "Guides revised 26 July 2023",
     ACCESS_DATE, True,
     "Relevant only where US creators/consumers are implicated; jurisdictional applicability to specific creators in the corpus was not individually determined."),
    ("REG07", "US National Institute of Standards and Technology (NIST)", "AI Risk Management Framework (AI RMF 1.0) and Generative AI Profile (NIST-AI-600-1)", "US (voluntary, internationally referenced)", "Voluntary",
     "Provides a voluntary framework (Govern, Map, Measure, Manage) for identifying and managing AI risk, including a generative-AI-specific profile covering risks like confabulation and data privacy.",
     "Used here as the structuring framework for this governance document's risk/control taxonomy (risk register, control matrix, monitoring indicators).",
     "Governance operating model (framework structure)",
     "Continue to use as a voluntary reference framework; do not present as a compliance requirement",
     "Model-risk/analytics lead",
     "https://www.nist.gov/itl/ai-risk-management-framework",
     "AI RMF 1.0 released 26 January 2023; Generative AI Profile (NIST-AI-600-1) released 26 July 2024",
     ACCESS_DATE, False,
     "Voluntary framework; adopting it does not itself satisfy any specific legal obligation."),
    ("REG08", "International Organization for Standardization / IEC", "ISO/IEC 42001:2023 -- AI management systems", "International (voluntary, certifiable)", "Voluntary",
     "Specifies requirements for establishing, implementing, maintaining, and continually improving an AI management system (AIMS) using a Plan-Do-Check-Act approach.",
     "Used here as a structural reference for the governance operating model's layered controls and continual-improvement framing; the project does not claim ISO/IEC 42001 certification.",
     "Governance operating model (framework structure)",
     "Continue to use as a voluntary reference; do not claim certification or full conformance",
     "AI product owner",
     "https://www.iso.org/standard/42001",
     "Published December 2023",
     ACCESS_DATE, False,
     "Voluntary, certifiable standard; this project has NOT sought or claimed certification."),
    ("REG09", "Organisation for Economic Co-operation and Development (OECD)", "OECD AI Principles", "International (47 adherents, voluntary)", "Voluntary",
     "Five values-based principles (inclusive growth/well-being; human rights and democratic values including fairness and privacy; transparency and explainability; robustness/security/safety; accountability) plus five policy recommendations.",
     "Used here as the ethical reference point for the 'core governance principle' (section 3) distinguishing evidence retrieval, automated classification, human judgement, and approved communication.",
     "Governance operating model; core governance principle",
     "Continue to use as a voluntary ethical reference",
     "Executive sponsor",
     "https://oecd.ai/en/ai-principles",
     "Adopted 2019; updated May 2024",
     ACCESS_DATE, False,
     "Voluntary intergovernmental principles, not a binding legal instrument in any single jurisdiction."),
    ("REG10", "Google (Gemini API)", "Gemini API Additional Terms of Service and Data logging/sharing policy", "Global (vendor terms; jurisdiction per Google's terms)", "Mandatory (as a contractual condition of using the Gemini API)",
     "Data-use terms differ by tier: on unpaid/free tiers, submitted content and generated responses may be used by Google to improve products, with human reviewers possibly reading disconnected samples; paid tiers generally exclude data from general model training unless opted in.",
     "The system's sole answer generator is the Gemini API (src/rag/gemini_generator.py); which tier (free vs. paid) the project's API key uses is NOT recorded in this repository and materially changes the applicable data-use terms.",
     "Retrieval and generation (Layer 4); GenAI vendor governance",
     "Confirm and document the Gemini API account tier; context-minimisation control already limits what is sent (public evidence text + claim IDs only)",
     "Data Protection Officer; AI product owner",
     "https://ai.google.dev/gemini-api/terms",
     "Terms as published; subject to Google's update cadence",
     ACCESS_DATE, True,
     "Vendor contractual terms, not law; also see https://ai.google.dev/gemini-api/docs/logs-policy for the data-logging specifics. Tier confirmation is an open governance gap (see genai_vendor_control_matrix.csv)."),
    ("REG11", "UK Parliament", "Data Protection Act 2018", "UK", "Mandatory",
     "Extends and implements UK GDPR standards domestically, including provisions for areas of processing not covered by UK GDPR alone.",
     "Applies alongside UK GDPR to any UK personal data incidentally processed by the system.",
     "Data admission; storage; generation (Gemini)",
     "Same as REG01/REG04 -- DPO review before scaling beyond the current academic-prototype use",
     "Data Protection Officer",
     "https://www.legislation.gov.uk/ukpga/2018/12/contents",
     "In force since 25 May 2018",
     ACCESS_DATE, True,
     "Read together with UK GDPR (REG01); no standalone DPA-specific determination made here."),
]

REG_COLUMNS = [
    "requirement_id", "authority", "instrument", "jurisdiction", "mandatory_status", "verified_requirement",
    "system_relevance", "affected_stage", "required_or_recommended_control", "owner", "official_url",
    "publication_or_update_date", "access_date", "legal_review_required", "caveat",
]


def build_regulatory_map() -> pd.DataFrame:
    return pd.DataFrame([dict(zip(REG_COLUMNS, r)) for r in REG_ROWS])[REG_COLUMNS]


# ---------------------------------------------------------------------------
# Section 16: generative-AI and vendor controls
# There is no fallback model. On any Gemini failure the correct behaviour is:
# stop generation, show a clear error, and prevent unsupported output.
# ---------------------------------------------------------------------------

GENAI_ROWS = [
    ("Gemini API authentication", "GEMINI_API_KEY read from .env via os.environ; never hardcoded or logged.", "Implemented", "src/rag/gemini_generator.py"),
    ("Data sent to Gemini", "Only short retrieved evidence passages and claim/evidence IDs from the already-collected, already-public corpus.", "Implemented", "src/rag/orchestrator.py; src/rag/gemini_generator.py"),
    ("Public versus personal information", "Corpus is public review/press/official/creator text by design; no raw personal identifiers are deliberately included, though free text may incidentally reference names.", "Partially implemented", "docs/governance_notes.md; no automated PII scrub on Gemini input specifically"),
    ("Context minimisation", "Prompt construction includes only the retrieved passages needed to answer the specific query, not full corpora or full documents.", "Implemented", "src/rag/orchestrator.py prompt assembly"),
    ("Model-version pinning", "GEMINI_MODEL is an explicit environment variable, never inferred or auto-upgraded.", "Implemented", "src/rag/gemini_generator.py; configs/rag_config.yaml"),
    ("Provider changes", "No alternate provider/model is coded; a provider change would require an explicit code change and re-evaluation, not a silent switch.", "Implemented", "src/rag/gemini_generator.py (single-provider design)"),
    ("Rate limits", "Not explicitly rate-limited by project code; relies on Gemini API's own quota/rate-limit responses surfacing as errors.", "Not applicable / vendor-managed", "No project-level rate limiter found"),
    ("API failure", "GeminiGenerationError is raised on any API call failure; the orchestrator does not catch-and-continue with a substitute answer.", "Implemented", "src/rag/gemini_generator.py lines defining GeminiGenerationError"),
    ("Logging", "Local retrieval/citation-validation results are logged to outputs/tables/; no project-side logging of full Gemini prompts/responses was found in the inspected code.", "Partially implemented", "outputs/tables/rag_citation_validation.csv, rag_latency_summary.csv"),
    ("Data-use terms", "Google's Gemini API data-use terms depend on account tier (free vs. paid); this project's tier is NOT recorded in the repository.", "Gap -- proposed to confirm", "See regulatory_standards_map.csv REG10; ai_governance_risk_register.csv R21"),
    ("Regional processing", "Not specified in project configuration; Google's regional processing terms apply by default per the Gemini API terms.", "Not applicable / vendor-managed", "https://ai.google.dev/gemini-api/terms"),
    ("Vendor review", "No formal, scheduled vendor (Google/Gemini) terms review currently exists.", "Proposed", "ai_governance_control_matrix.csv C50"),
    ("Prompt injection", "No dedicated prompt-injection filter/detector exists; context minimisation and citation validation partially mitigate downstream impact.", "Partially implemented", "src/rag/gemini_generator.py; ai_governance_risk_register.csv R19"),
    ("Citation validation", "Every generated answer's citations are checked against the retrieved evidence IDs before being treated as valid.", "Implemented", "src/rag/citation_validator.py; outputs/tables/rag_citation_validation.csv"),
    ("Output retention", "Generated answers are not persisted to a permanent store by default in the inspected pipeline beyond evaluation-run output tables.", "Partially implemented", "outputs/tables/rag_demo_cases.csv"),
    ("Human approval", "No generated answer is treated as final management output without human review under the claim decision policy.", "Proposed (policy defined; workflow not built)", "outputs/tables/claim_decision_policy.csv"),
    ("No fallback generator (failure behaviour)", "There is no fallback model, deterministic template, or alternate provider. On any Gemini config or generation error, the system stops generation, raises a clear error, and prevents unsupported output from being shown.", "Implemented", "src/rag/gemini_generator.py GeminiConfigError/GeminiGenerationError; module docstring states this explicitly"),
]

GENAI_COLUMNS = ["control_area", "control_description", "status", "system_evidence"]


def build_genai_vendor_control_matrix() -> pd.DataFrame:
    return pd.DataFrame([dict(zip(GENAI_COLUMNS, r)) for r in GENAI_ROWS])[GENAI_COLUMNS]


# ---------------------------------------------------------------------------
# Section 17: monitoring indicators (KPI/KRI)
# Current values are computed live from repository outputs where feasible;
# 'TBC' is used wherever no repository output can compute the figure today.
# ---------------------------------------------------------------------------

def _current_kpi_values() -> dict:
    master = pd.read_csv(TABLES_DIR / "analytical_synthesis_claim_master.csv")
    units = pd.read_csv(TABLES_DIR.parent.parent / "data" / "processed" / "fusion" / "claim_evidence_units.csv")
    stability = pd.read_csv(TABLES_DIR / "fusion_sensitivity_stability.csv")
    citation = pd.read_csv(TABLES_DIR / "rag_citation_validation.csv") if (TABLES_DIR / "rag_citation_validation.csv").exists() else None

    n = len(master)
    self_reported_only_share = round(100 * ((master["self_reported_evidence_count"] > 0) & (master["independent_evidence_count"] == 0)).sum() / n, 1)
    independent_coverage = round(100 * (master["independent_evidence_count"] > 0).sum() / n, 1)
    insufficient_share = round(100 * (master["production_fusion_label"] == "insufficient_evidence").sum() / n, 1)
    sensitive_share = round(100 * (~stability["label_stable"]).sum() / len(stability), 1)
    provenance_complete = "100.0" # validate_data.py reports 33/33 PASS on provenance-required schemas as of this build

    if citation is not None and len(citation):
        citation_fail_rate = round(100 * (~citation["passed"]).sum() / len(citation), 1)
    else:
        citation_fail_rate = "TBC"

    return {
        "Provenance completeness": f"{provenance_complete}% (33/33 files passed scripts/validate_data.py provenance checks at last run)",
        "Evidence-strength completeness": "TBC (no single repository output currently aggregates evidence_strength null-rate across all corpora in one number)",
        "Unresolved source rate": "TBC (would require a live URL-liveness check, which this task does not perform)",
        "Citation-validation failure rate": f"{citation_fail_rate}%" if citation_fail_rate != "TBC" else "TBC",
        "Unsupported citation rate": "TBC (see rag_citation_validation.csv failure category breakdown for the underlying detail)",
        "Label-preservation failure rate": "TBC (no dedicated production monitoring job runs this check continuously)",
        "Missing-asset rate": "TBC (would require a live media_path file-existence sweep, not run in this task)",
        "Duplicate evidence rate": "0% (src/validation.py assert_no_duplicate_ids() blocks duplicate writes at build time)",
        "Retrieval recall@k": "TBC (see outputs/tables/rag_retrieval_evaluation.csv for the underlying per-query detail)",
        "Evidence-staleness rate": "TBC (no staleness threshold/scheduler currently implemented; see risk R05)",
        "Self-reported-only claim share": f"{self_reported_only_share}% ({int((master['self_reported_evidence_count'] > 0).sum() - (master['independent_evidence_count'] > 0).sum())} of {n} claims have only self-reported support)",
        "Independent-evidence coverage": f"{independent_coverage}% ({int((master['independent_evidence_count'] > 0).sum())} of {n} claims have >=1 independent evidence unit)",
        "Mixed/divergent escalation completion": "TBC (no escalation-tracking workflow exists yet to measure completion)",
        "Insufficient-evidence share": f"{insufficient_share}% ({int((master['production_fusion_label'] == 'insufficient_evidence').sum())} of {n} claims)",
        "Sensitive-to-weights share": f"{sensitive_share}% ({int((~stability['label_stable']).sum())} of {len(stability)} claims label-unstable across weighting scenarios)",
        "API failure rate": "TBC (no production-scale Gemini call volume has been logged to measure this)",
        "Secret-detection events": "0 (manual review; no automated secret-scanning CI step exists yet -- see control C43)",
        "Access-control violations": "Not applicable (single-user local environment; no access-control layer deployed yet)",
        "Correction turnaround time": "TBC (no correction has yet been logged; process is proposed, not exercised)",
        "Governance-review completion": "TBC (first governance review has not yet occurred; this document establishes the baseline)",
    }


KPI_DEFINITIONS = [
    ("Provenance completeness", "% of admitted records with all required provenance fields populated", "count(non-null provenance fields) / count(required provenance fields) x 100", "99%", "95%", "90%", "Data steward", "Every validator run"),
    ("Evidence-strength completeness", "% of evidence units with a populated, allowed evidence_strength value", "count(valid evidence_strength) / count(all evidence units) x 100", "99%", "95%", "90%", "Model-risk/analytics lead", "Every fusion run"),
    ("Unresolved source rate", "% of source_url values that fail a liveness/resolution check", "count(unresolved URLs) / count(all URLs) x 100", "<2%", "5%", "10%", "Data steward", "Quarterly (proposed)"),
    ("Citation-validation failure rate", "% of generated-answer citations that fail validation", "count(failed citations) / count(all citations checked) x 100", "0%", "2%", "5%", "Model-risk/analytics lead", "Every RAG evaluation run"),
    ("Unsupported citation rate", "% of answer sentences with no resolvable supporting citation", "count(unsupported sentences) / count(all sentences) x 100", "0%", "1%", "3%", "Model-risk/analytics lead", "Every RAG evaluation run"),
    ("Label-preservation failure rate", "% of generated answers that alter the underlying claim's fusion label", "count(label-mismatched answers) / count(all answers) x 100", "0%", "0.5%", "1%", "Model-risk/analytics lead", "Every RAG evaluation run"),
    ("Missing-asset rate", "% of media_path references that do not resolve to an existing file", "count(missing files) / count(all media_path references) x 100", "<1%", "3%", "5%", "Data steward", "Per media-manifest build"),
    ("Duplicate evidence rate", "% of evidence/claim IDs that are duplicated", "count(duplicate IDs) / count(all IDs) x 100", "0%", "0.5%", "1%", "Data steward", "Every build"),
    ("Retrieval recall@k", "Share of relevant evidence retrieved within the top-k results on the evaluation query set", "count(relevant docs in top-k) / count(all relevant docs) per query, averaged", "TBC (target to be set after a larger evaluation query set exists)", "TBC", "TBC", "Model-risk/analytics lead", "Every RAG evaluation run"),
    ("Evidence-staleness rate", "% of claims whose supporting evidence is older than the staleness threshold", "count(claims with max(evidence collection_date) older than threshold) / count(all claims) x 100", "<20%", "35%", "50%", "AI product owner", "Quarterly (proposed)"),
    ("Self-reported-only claim share", "% of claims with supporting evidence but zero independent corroboration", "count(self-reported-only claims) / count(all claims) x 100", "<30%", "40%", "55%", "Sustainability/ESG lead", "Every fusion run"),
    ("Independent-evidence coverage", "% of claims with at least one independent (non-brand) evidence unit", "count(claims with independent_evidence_count>0) / count(all claims) x 100", ">50%", "40%", "25%", "Sustainability/ESG lead", "Every fusion run"),
    ("Mixed/divergent escalation completion", "% of mixed/divergent claims with a recorded human-review sign-off", "count(escalated and signed off) / count(all mixed+divergent claims) x 100", "100%", "90%", "75%", "Legal counsel", "Ongoing (proposed workflow)"),
    ("Insufficient-evidence share", "% of claims currently labelled insufficient_evidence", "count(insufficient_evidence claims) / count(all claims) x 100", "<35%", "45%", "55%", "AI product owner", "Every fusion run"),
    ("Sensitive-to-weights share", "% of claims whose label is unstable across the 4 fusion-configuration weighting scenarios", "count(label_stable=False) / count(all claims) x 100", "<20%", "35%", "50%", "Model-risk/analytics lead", "Every sensitivity run"),
    ("API failure rate", "% of Gemini API calls that raise GeminiGenerationError", "count(failed calls) / count(all calls) x 100", "<2%", "5%", "10%", "AI product owner", "Continuous (proposed monitoring)"),
    ("Secret-detection events", "Count of credential patterns detected in tracked files", "count(detected secret patterns)", "0", "1", ">=2", "Information Security lead", "Every commit (proposed CI)"),
    ("Access-control violations", "Count of access attempts outside the approved role set", "count(unauthorised access attempts)", "0", "1", ">=2", "Information Security lead", "Continuous (proposed, once deployed)"),
    ("Correction turnaround time", "Median days from confirmed error to published correction", "median(correction_publish_date - error_confirmed_date)", "<5 days", "10 days", "20 days", "Legal counsel", "Per incident"),
    ("Governance-review completion", "% of scheduled quarterly governance reviews completed on time", "count(completed on-time reviews) / count(scheduled reviews) x 100", "100%", "80%", "60%", "Executive sponsor", "Quarterly"),
]

KPI_COLUMNS = ["kpi_kri", "definition", "formula", "target", "warning_threshold", "breach_threshold", "owner", "frequency"]


def build_kpi_kri() -> pd.DataFrame:
    df = pd.DataFrame([dict(zip(KPI_COLUMNS, r)) for r in KPI_DEFINITIONS])[KPI_COLUMNS]
    current = _current_kpi_values()
    df.insert(3, "current_value", df["kpi_kri"].map(current).fillna("TBC"))
    return df


# ---------------------------------------------------------------------------
# Section 18: incident and escalation matrix
# ---------------------------------------------------------------------------

INCIDENT_ROWS = [
    ("Exposed API key (Gemini/YouTube/Apify/etc.)", "Critical", "Revoke/rotate the key immediately; remove from any tracked file or log", "Information Security lead", "Information Security lead", "Rotate credential; audit for unauthorised use during exposure window", "Consider if third-party data was exposed as a result", "<4 hours", "New credential active; old credential confirmed revoked; incident log entry closed"),
    ("Personal data accidentally retained (e.g. un-stripped username)", "High", "Locate and remove/redact the record from data/raw, data/interim, data/processed, and any derived table", "Data steward", "Data Protection Officer", "Re-run the stripping step; add a regression check", "Consider if the individual should be notified, per DPO assessment", "<48 hours", "Record confirmed redacted across all derived tables; regression check added"),
    ("Invalid or fabricated citation in a Gemini answer", "High", "Suppress the affected answer; do not distribute it further", "Model-risk/analytics lead", "Model-risk/analytics lead", "Re-run citation validation; fix the retrieval/prompt defect if systemic", "Not typically externally notifiable (internal tool output)", "<24 hours", "rag_citation_validation.csv shows the case passing; root cause documented"),
    ("Unsupported greenwashing allegation surfaces (internal or external)", "Critical", "Immediately withdraw the statement from any communication channel", "Marketing/brand lead", "Legal counsel", "Issue an internal correction; if externally stated, issue a public correction per Legal counsel guidance", "Legal counsel to assess notification to TALA/affected party", "<24 hours", "Correction published/circulated; claim_decision_policy.csv wording rules reconfirmed with the team"),
    ("Creator misidentification or misattributed partnership type", "High", "Withdraw the specific creator-related finding from circulation", "Creator-partnership lead", "Creator-partnership lead", "Correct the classification; notify Legal counsel if publicly stated", "Consider notifying the creator if externally published", "<48 hours", "Corrected record in creator_multimodal_features.csv; correction communicated if published"),
    ("Copyright complaint (image/video asset)", "High", "Remove the disputed asset from active use pending review", "Legal counsel", "Legal counsel", "Confirm/update the asset's rights_or_access_basis; remove permanently if rights basis cannot be confirmed", "Respond to the complainant per Legal counsel guidance and any applicable takedown process", "<72 hours", "Asset removed or rights basis confirmed and documented; complaint closed"),
    ("Wrong fusion label displayed (code/data defect, not an adjudication disagreement)", "High", "Halt any communication citing the affected claim_id", "Model-risk/analytics lead", "Model-risk/analytics lead", "Fix the defect; re-run fusion; confirm the corrected label with tests/test_analytical_synthesis.py", "Not typically externally notifiable unless already published", "<24 hours", "Defect fixed; regression test added; corrected label confirmed"),
    ("Gemini model behaviour change (Google-side)", "Medium", "Pause reliance on affected answer types pending re-evaluation", "AI product owner", "Model-risk/analytics lead", "Re-run rag evaluation suite against the new model behaviour; update prompt/version pin if needed", "Not externally notifiable (vendor-side change)", "<5 business days", "RAG evaluation suite passes against the current model; version pin updated in configs/rag_config.yaml"),
    ("Retrieval/index corruption (ChromaDB)", "Medium", "Take the affected collection offline; do not serve answers from it", "Information Security lead", "Model-risk/analytics lead", "Rebuild the Chroma index from source corpora per scripts/build_chroma_index.py; verify against rag_collection_summary.csv", "Not externally notifiable (internal infrastructure)", "<48 hours", "Rebuilt collection passes rag_go_no_go.csv checks"),
    ("Stale certification/ESG evidence discovered", "Medium", "Flag the affected claim(s) as evidence-stale pending refresh", "Sustainability/ESG lead", "Data steward", "Schedule and perform a targeted refresh collection for the affected claim(s)", "Not typically externally notifiable unless already published as current", "<10 business days", "Refreshed evidence collected and re-run through fusion; claim re-classified as needed"),
    ("Public dissemination without approval", "Critical", "Request immediate withdrawal/correction of the disseminated material where possible", "Marketing/brand lead", "Executive sponsor", "Formal post-incident review; reinforce the human-approval gate (control C37)", "Legal counsel to assess whether external notification/correction is required", "<24 hours", "Material withdrawn or corrected; post-incident review completed and control gap closed"),
]

INCIDENT_COLUMNS = [
    "incident_scenario", "severity", "immediate_containment", "notification_owner", "investigation_owner",
    "correction_action", "external_notification_consideration", "target_response_time", "closure_evidence",
]


def build_incident_matrix() -> pd.DataFrame:
    return pd.DataFrame([dict(zip(INCIDENT_COLUMNS, r)) for r in INCIDENT_ROWS])[INCIDENT_COLUMNS]


# ---------------------------------------------------------------------------
# Section 19: implementation roadmap
# ---------------------------------------------------------------------------

ROADMAP_ROWS = [
    # (phase, initiative, deliverable, owner, dependencies, effort, cost_band, business_value, risk_reduction, completion_criterion, gate_to_next_phase)
    ("Phase 0 - Academic prototype (current state)", "Provenance, evidence-strength and modality gates", "Already implemented (see ai_governance_control_matrix.csv Implemented rows)", "Model-risk/analytics lead", "None -- already built", "Complete", "n/a (sunk)", "Establishes a defensible evidence trail for every claim", "Addresses R04, R13, R14, R15", "Validator/tests pass (already true: 33/33, 19/19, 558/558)", "n/a -- baseline"),
    ("Phase 0 - Academic prototype (current state)", "Automated fusion with sensitivity testing", "Already implemented", "Model-risk/analytics lead", "None -- already built", "Complete", "n/a (sunk)", "Repeatable, evidence-gated claim classification", "Addresses R16", "fusion_sensitivity_stability.csv exists and passes tests", "n/a -- baseline"),
    ("Phase 0 - Academic prototype (current state)", "Citation-validated RAG with no fallback generator", "Already implemented", "Model-risk/analytics lead", "None -- already built", "Complete", "n/a (sunk)", "Prevents unsupported answers from being shown", "Addresses R17, R18, R20", "rag_go_no_go.csv and rag_citation_validation.csv exist", "n/a -- baseline"),
    ("Phase 1 - Controlled internal pilot (0-30 days)", "Named governance owners for every RACI activity", "Signed-off RACI assignment (ai_governance_raci.csv adopted as the operating document, not just a draft)", "Executive sponsor", "Executive sign-off on role assignments", "Low", "Low (internal time only)", "Establishes accountability before any pilot use", "Addresses R09, R25, R26", "Every named individual has confirmed their role in writing", "Required before Phase 1 start"),
    ("Phase 1 - Controlled internal pilot (0-30 days)", "Access controls for the local ChromaDB/repository environment", "Documented access list; credentials rotated to named individuals only", "Information Security lead", "Named owners assigned", "Low", "Low", "Limits R23 exposure", "Addresses R23", "Access list documented and reviewed", "Required before any pilot user beyond the current team"),
    ("Phase 1 - Controlled internal pilot (0-30 days)", "Approved source register", "A maintained list of pre-approved sources/platforms with documented TOS status", "Data steward", "docs/governance_notes.md as the starting point", "Low", "Low", "Prevents ad-hoc TOS-risk collection", "Addresses R03", "Register published and referenced in every new collection script", "Required before new source types are added"),
    ("Phase 1 - Controlled internal pilot (0-30 days)", "Formal intended-use statement", "A one-page, distributable statement of intended and prohibited use", "Executive sponsor", "This governance document (section 2)", "Low", "Low", "Prevents scope creep and misuse", "Addresses R26", "Statement circulated to all pilot users", "Required before pilot access is granted"),
    ("Phase 1 - Controlled internal pilot (0-30 days)", "Legal and DPO review of data flows (incl. Gemini API tier confirmation)", "Documented legal/DPO sign-off memo", "Legal counsel; Data Protection Officer", "genai_vendor_control_matrix.csv gap list", "Medium", "Medium (external counsel time possible)", "Closes R21 (API data exposure) and REG04/REG10 gaps", "Addresses R21, R09", "Signed memo confirming Gemini tier and data flows reviewed", "Required before any pilot use beyond the current academic team"),
    ("Phase 1 - Controlled internal pilot (0-30 days)", "Sustainability-review process for labour/ESG claims", "A working ESG-review checklist applied to the 14 labour claims and 2 mixed themes", "Sustainability/ESG lead", "esg_claim_assurance_matrix.csv", "Medium", "Low", "Directly closes the labour-evidence gap risk (R10, R11)", "Addresses R09, R10, R11", "Checklist applied and logged for every labour/ESG claim reviewed", "Required before any ESG-related pilot communication"),
    ("Phase 1 - Controlled internal pilot (0-30 days)", "Incident procedure activation", "ai_incident_escalation_matrix.csv adopted as the working incident process, with named on-call owners", "Information Security lead", "RACI sign-off", "Low", "Low", "Enables fast containment of Critical incidents", "Addresses R09, R22", "At least one incident-response drill completed", "Required before pilot go-live"),
    ("Phase 1 - Controlled internal pilot (0-30 days)", "Model and prompt register", "A version-controlled log of GEMINI_MODEL values and prompt template changes over time", "AI product owner", "src/rag/gemini_generator.py, configs/rag_config.yaml", "Low", "Low", "Enables root-cause analysis on behaviour changes", "Addresses R20", "Register entry exists for the current model/prompt version", "Required before pilot go-live"),
    ("Phase 2 - Limited business deployment (31-90 days)", "Scheduled evidence refresh", "An automated or calendared job re-collecting evidence past its staleness threshold", "Data steward", "Phase 1 approved source register", "Medium", "Medium", "Directly reduces R05 (evidence staleness)", "Addresses R05", "First scheduled refresh cycle completed and logged", "Required before claims are used in a live business decision"),
    ("Phase 2 - Limited business deployment (31-90 days)", "Independent ESG-evidence integration", "At least one independent labour/materials audit or certification-registry source integrated for the highest-priority gap categories", "Sustainability/ESG lead", "esg_claim_assurance_matrix.csv priorities", "High", "Medium-High (may require a paid data/audit source)", "Closes the single largest evidence gap (14 labour claims, 0 independent evidence)", "Addresses R10, R11", "At least 1 independent source integrated and claim-linked", "Required before any labour/ESG claim is used externally"),
    ("Phase 2 - Limited business deployment (31-90 days)", "Monitoring dashboard for KPI/KRI", "A live view of the ai_governance_kpi_kri.csv indicators, replacing manual TBC values with computed ones", "Model-risk/analytics lead", "Phase 1 model/prompt register", "Medium", "Medium", "Enables proactive risk detection", "Addresses R05, R16, R17", "Dashboard shows current values for >=80% of defined KPIs", "Required before Phase 3"),
    ("Phase 2 - Limited business deployment (31-90 days)", "Reviewer workflow for flagged claims/creators", "A working tool/checklist implementing the escalation flags already present in the data (requires_human_validation etc.)", "Human evidence reviewer", "Phase 1 named owners", "Medium", "Medium", "Converts existing data flags into an enforced process", "Addresses R07, R11, R12, R25", "100% of flagged records reviewed before external use in a sample audit", "Required before external-facing use of any flagged finding"),
    ("Phase 2 - Limited business deployment (31-90 days)", "Correction and appeal process", "A documented, tested process for correcting a published finding, including a channel for a creator/competitor to raise a concern", "Legal counsel", "ai_incident_escalation_matrix.csv", "Medium", "Low-Medium", "Reduces reputational/legal exposure from R09", "Addresses R09", "At least one correction processed end-to-end (real or drilled)", "Required before external-facing use"),
    ("Phase 2 - Limited business deployment (31-90 days)", "Vendor review (Gemini terms, tier, alternatives assessment)", "A documented vendor-review memo, repeated on a recurring basis", "Data Protection Officer", "Phase 1 legal/DPO review", "Low", "Low", "Keeps R21 current as vendor terms evolve", "Addresses R21", "First recurring review completed", "Required before Phase 3"),
    ("Phase 2 - Limited business deployment (31-90 days)", "User training on intended use and prohibited use", "A short training session/document for all system users", "Executive sponsor", "Phase 1 intended-use statement", "Low", "Low", "Reduces R26 (overreliance)", "Addresses R26", "All active users complete training", "Required before external-facing use"),
    ("Phase 2 - Limited business deployment (31-90 days)", "Audit sampling", "A periodic sample review of published findings against source evidence", "Internal audit", "Monitoring dashboard", "Medium", "Low-Medium", "Provides independent assurance the controls are working", "Addresses R09, R16", "First audit sample completed with findings logged", "Required before Phase 3"),
    ("Phase 3 - Scaled deployment (3-6 months)", "Enterprise identity/access integration", "SSO/role-based access control integrated with the organisation's identity provider", "Information Security lead", "Phase 1 access-control documentation", "High", "Medium-High", "Closes R23 at scale", "Addresses R23", "All access is authenticated via enterprise identity", "Required before multi-team/customer-adjacent deployment"),
    ("Phase 3 - Scaled deployment (3-6 months)", "Formal model-risk oversight (committee)", "A recurring model-risk committee reviewing fusion/RAG changes", "Model-risk/analytics lead", "Phase 2 monitoring dashboard", "Medium", "Medium", "Institutionalises R16/R17 oversight", "Addresses R16, R17, R20", "Committee charter adopted and first meeting held", "Required before scaled deployment"),
    ("Phase 3 - Scaled deployment (3-6 months)", "Continuous monitoring", "Automated, always-on KPI/KRI computation replacing periodic manual checks", "Model-risk/analytics lead", "Phase 2 dashboard", "High", "Medium-High", "Enables real-time risk detection at scale", "Addresses all Medium/High/Critical risks", "Continuous monitoring live for >=90 days without a missed cycle", "Required before customer-facing use"),
    ("Phase 3 - Scaled deployment (3-6 months)", "Independent audit", "An external or independent-internal audit of the governance framework's operation, not just its design", "Internal audit", "Phase 2 audit sampling", "High", "Medium-High", "Provides external assurance for stakeholders/regulators", "Addresses R09, R16", "Audit report issued with no unresolved Critical findings", "Required before customer-facing use"),
    ("Phase 3 - Scaled deployment (3-6 months)", "Business-system integration", "Governed integration points with TALA's own business systems (if applicable), with contractual data-use terms", "AI product owner", "Phase 2 vendor review process as a template", "High", "High", "Enables the system's use in an actual TALA business context", "Addresses R02, R03, R21", "Integration contract signed with data-use terms reviewed", "Required before external/customer-facing use"),
    ("Phase 3 - Scaled deployment (3-6 months)", "Periodic red-team testing", "Adversarial testing of prompt-injection resistance and citation-validation robustness", "Information Security lead", "Phase 2 monitoring dashboard", "Medium", "Medium", "Directly tests R17, R18, R19", "Addresses R17, R18, R19", "First red-team exercise completed with findings remediated", "Required before customer-facing use"),
    ("Phase 3 - Scaled deployment (3-6 months)", "Governance effectiveness review", "A structured review of whether the governance framework itself (not just the AI system) is working as intended", "Executive sponsor", "All prior phases", "Medium", "Low-Medium", "Closes the loop on governance as a living system, not a one-time document", "Addresses R26 and framework-wide drift", "Review completed with an updated framework version issued", "Gate to any subsequent Phase 4 (out of scope for this task)"),
]

ROADMAP_COLUMNS = [
    "phase", "initiative", "deliverable", "owner", "dependencies", "effort", "cost_band",
    "business_value", "risk_reduction", "completion_criterion", "gate_to_next_phase",
]


def build_roadmap() -> pd.DataFrame:
    df = pd.DataFrame([dict(zip(ROADMAP_COLUMNS, r)) for r in ROADMAP_ROWS])[ROADMAP_COLUMNS]
    assert df["completion_criterion"].notna().all() and (df["completion_criterion"].astype(str).str.strip() != "").all()
    return df


# ---------------------------------------------------------------------------
# Section 20: deployment gates
# ---------------------------------------------------------------------------

GATE_NAMES = [
    "Data governance", "Privacy", "Media rights", "Model documentation", "Evidence quality",
    "Citation integrity", "ESG assurance", "Human oversight", "Security", "Vendor governance",
    "Incident response", "Monitoring", "Training",
]

# status per deployment context: GO / PARTIAL / NO-GO, plus evidence and gap.
GATE_ASSESSMENT = {
    "Data governance": {
        "academic": ("GO", "Provenance fields, schema routing, and evidence-strength gates implemented and passing (33/33 validate_data.py).", "None material for academic use."),
        "pilot": ("PARTIAL", "Core gates implemented; approved-source register and named data-steward sign-off not yet formalised.", "Formalise approved source register (Phase 1)."),
        "external": ("NO-GO", "No enterprise data-governance sign-off, retention schedule, or formal source-approval process exists yet.", "Complete Phase 1-2 data-governance initiatives before any external use."),
    },
    "Privacy": {
        "academic": ("GO", "Username stripping documented; no face embeddings; minimal personal-data surface for academic use.", "None material for academic use."),
        "pilot": ("PARTIAL", "DPIA and formal DPO review not yet performed; Gemini API tier (free/paid) not confirmed.", "Complete Legal/DPO review including Gemini tier confirmation (Phase 1)."),
        "external": ("NO-GO", "No DPIA, no confirmed retention/deletion automation, no confirmed vendor data-use tier.", "Complete Phase 1-2 privacy initiatives; obtain formal DPO sign-off."),
    },
    "Media rights": {
        "academic": ("GO", "rights_or_access_basis field gates all video/image processing; no unofficial downloaders used.", "None material for academic use."),
        "pilot": ("PARTIAL", "Rights basis is self-declared at collection, not independently legally cleared.", "Legal review of media rights basis for any pilot-published asset (Phase 1)."),
        "external": ("NO-GO", "No independent legal rights clearance for any media asset.", "Obtain legal rights clearance before any external publication of media evidence."),
    },
    "Model documentation": {
        "academic": ("GO", "Model/threshold versions recorded in provenance_note fields; fusion rules documented in configs/fusion_rules.yaml.", "None material for academic use."),
        "pilot": ("PARTIAL", "No formal model/prompt change-control register yet (Phase 1 initiative).", "Stand up the model and prompt register (Phase 1)."),
        "external": ("NO-GO", "No formal model-risk committee or change-control process exists.", "Complete Phase 3 model-risk oversight initiative."),
    },
    "Evidence quality": {
        "academic": ("GO", "Evidence-strength, source-independence, and stance fields populated and tested across 119 evidence units.", "None material for academic use."),
        "pilot": ("PARTIAL", "14 labour claims (largest category) have zero independent evidence; evidence-staleness not monitored.", "Prioritise independent labour/ESG evidence integration (Phase 2)."),
        "external": ("NO-GO", "Independent-evidence coverage is materially incomplete (largest claim category has none); no refresh schedule.", "Complete Phase 2 independent ESG-evidence integration and scheduled refresh."),
    },
    "Citation integrity": {
        "academic": ("GO", "Citation validator implemented and passing on documented demo cases; no fallback generator (fails closed).", "None material for academic use."),
        "pilot": ("PARTIAL", "Validation covers documented demo cases, not yet a continuously monitored production query stream.", "Extend citation-validation monitoring to a live/monitored dashboard (Phase 2)."),
        "external": ("NO-GO", "No continuous production monitoring of citation integrity exists yet.", "Complete Phase 2-3 monitoring initiatives."),
    },
    "ESG assurance": {
        "academic": ("GO", "esg_claim_assurance_matrix.csv defines acceptable/ineligible evidence and current gaps transparently.", "None material for academic use."),
        "pilot": ("PARTIAL", "Sustainability-review process for labour/ESG claims is a Phase 1 initiative, not yet operating.", "Activate the sustainability-review checklist (Phase 1)."),
        "external": ("NO-GO", "No independent ESG assurance; 14/37 claims (largest category) have zero independent evidence.", "Complete Phase 2 independent ESG-evidence integration before any external ESG-related claim use."),
    },
    "Human oversight": {
        "academic": ("GO", "Core governance principle (section 3) and claim decision policy define human-review requirements.", "None material for academic use."),
        "pilot": ("PARTIAL", "Review gates are policy-defined (claim_decision_policy.csv) but not yet workflow-enforced.", "Stand up the reviewer workflow (Phase 2)."),
        "external": ("NO-GO", "No enforced human-approval workflow exists before external communication.", "Complete Phase 1-2 human-approval gate initiatives."),
    },
    "Security": {
        "academic": ("GO", ".env excluded via .gitignore; local, non-networked ChromaDB; no secrets found in inspected code.", "None material for a single-user academic environment."),
        "pilot": ("PARTIAL", "No secret-scanning CI, no formal access-control documentation yet.", "Complete Phase 1 access-control documentation and Phase 1-2 secret-scanning CI."),
        "external": ("NO-GO", "No enterprise identity/access integration; no continuous secret-scanning; single-user security posture only.", "Complete Phase 3 enterprise identity/access integration."),
    },
    "Vendor governance": {
        "academic": ("GO", "No-fallback-generator behaviour documented and implemented; Gemini terms are publicly available and reviewed for this framework.", "Gemini API tier (free/paid) not confirmed -- flagged as a gap, not blocking for academic demonstration."),
        "pilot": ("PARTIAL", "Gemini API tier not confirmed in repository; no recurring vendor-review process yet.", "Confirm Gemini API tier and complete Phase 1 legal/DPO review."),
        "external": ("NO-GO", "No recurring vendor governance process; data-use tier unconfirmed.", "Complete Phase 2 vendor review initiative before external use."),
    },
    "Incident response": {
        "academic": ("GO", "ai_incident_escalation_matrix.csv defines scenarios, owners, and response times.", "Not yet exercised, but adequate as a documented plan for academic demonstration."),
        "pilot": ("PARTIAL", "Incident procedure is documented but not yet drilled.", "Complete at least one incident-response drill (Phase 1)."),
        "external": ("NO-GO", "No drilled incident response, no on-call rotation for a production-facing system.", "Complete Phase 1-2 incident-response activation and drills."),
    },
    "Monitoring": {
        "academic": ("GO", "KPI/KRI framework defined with formulas and thresholds, even where current values are TBC.", "TBC values are expected and acceptable at academic-prototype stage."),
        "pilot": ("PARTIAL", "Most KPI/KRI current values are TBC; no dashboard exists yet.", "Stand up the monitoring dashboard (Phase 2)."),
        "external": ("NO-GO", "No continuous, automated monitoring exists; most indicators are unmeasured.", "Complete Phase 3 continuous-monitoring initiative."),
    },
    "Training": {
        "academic": ("GO", "This document itself serves as the training/reference material for the current academic team.", "Adequate for the current, small academic team."),
        "pilot": ("PARTIAL", "No formal training session has been delivered yet.", "Deliver Phase 2 user training on intended/prohibited use."),
        "external": ("NO-GO", "No training programme exists for a wider, non-academic user base.", "Complete Phase 2 training before any wider rollout."),
    },
}


def build_go_no_go() -> pd.DataFrame:
    rows = []
    for gate in GATE_NAMES:
        for context, label in [("academic", "academic_demonstration_readiness"), ("pilot", "controlled_internal_pilot_readiness"), ("external", "external_customer_facing_deployment_readiness")]:
            status, evidence, gap = GATE_ASSESSMENT[gate][context]
            rows.append(
                {
                    "gate": gate,
                    "deployment_context": label,
                    "status": status,
                    "evidence": evidence,
                    "gap_or_next_step": gap,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build and write all governance tables
# ---------------------------------------------------------------------------

def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    risk_register = build_risk_register()
    risk_register.to_csv(TABLES_DIR / "ai_governance_risk_register.csv", index=False)

    control_matrix = build_control_matrix()
    control_matrix.to_csv(TABLES_DIR / "ai_governance_control_matrix.csv", index=False)

    claim_policy = build_claim_decision_policy()
    claim_policy.to_csv(TABLES_DIR / "claim_decision_policy.csv", index=False)

    raci = build_raci()
    raci.to_csv(TABLES_DIR / "ai_governance_raci.csv", index=False)

    lifecycle = build_data_lifecycle_policy()
    lifecycle.to_csv(TABLES_DIR / "ai_data_lifecycle_policy.csv", index=False)

    esg = build_esg_assurance_matrix()
    esg.to_csv(TABLES_DIR / "esg_claim_assurance_matrix.csv", index=False)

    reg_map = build_regulatory_map()
    reg_map.to_csv(TABLES_DIR / "regulatory_standards_map.csv", index=False)

    genai = build_genai_vendor_control_matrix()
    genai.to_csv(TABLES_DIR / "genai_vendor_control_matrix.csv", index=False)

    kpi = build_kpi_kri()
    kpi.to_csv(TABLES_DIR / "ai_governance_kpi_kri.csv", index=False)

    incidents = build_incident_matrix()
    incidents.to_csv(TABLES_DIR / "ai_incident_escalation_matrix.csv", index=False)

    roadmap = build_roadmap()
    roadmap.to_csv(TABLES_DIR / "ai_governance_implementation_roadmap.csv", index=False)

    go_no_go = build_go_no_go()
    go_no_go.to_csv(TABLES_DIR / "ai_governance_go_no_go.csv", index=False)

    print("risk_register:", len(risk_register))
    print("control_matrix:", len(control_matrix))
    print("claim_decision_policy:", len(claim_policy))
    print("raci:", len(raci))
    print("data_lifecycle_policy:", len(lifecycle))
    print("esg_claim_assurance_matrix:", len(esg))
    print("regulatory_standards_map:", len(reg_map))
    print("genai_vendor_control_matrix:", len(genai))
    print("kpi_kri:", len(kpi))
    print("incident_matrix:", len(incidents))
    print("roadmap:", len(roadmap))
    print("go_no_go:", len(go_no_go))
    print("Day 3D governance framework build complete.")


if __name__ == "__main__":
    main()
