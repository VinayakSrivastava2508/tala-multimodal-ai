"""One-off generator for notebooks/07_ai_governance_ethics_esg.ipynb (Day 3D).
Run once to (re)build the notebook skeleton, then execute with nbconvert."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md("""# 07 — AI Governance, Ethics and ESG Framework (Day 3D)

## 1. Governance objective

Govern the multimodal AI system already built for the TALA claim-experience
divergence and creator-strategy analysis: public-data collection, text/
image/video processing, claim-evidence fusion, local ChromaDB retrieval,
Gemini-generated answers, citation validation, and creator-strategy
analysis. This notebook builds and inspects the risk register, control
matrix, claim decision policy, RACI, data lifecycle policy, ESG assurance
matrix, regulatory map, GenAI vendor controls, KPI/KRI, incident matrix,
implementation roadmap, and deployment gates.

No new data collection, no external API calls, no change to any existing
analytical result (fusion labels, creator classifications, RAG outputs).""")

code("""import sys
from pathlib import Path

PROJECT_ROOT = Path().resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import subprocess
import pandas as pd
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_columns", 30)

TABLES = PROJECT_ROOT / "outputs" / "tables"
FIGURES = PROJECT_ROOT / "outputs" / "figures"
from IPython.display import Image
print("Project root:", PROJECT_ROOT)""")

md("""## 2. Intended and prohibited use

This is an **internal decision-support prototype**. It must never be
described or used as an autonomous legal/regulatory decision-maker, proof
that a sustainability claim is true or false, a substitute for
sustainability assurance or product-quality investigation, a customer-facing
claims engine, a creator-risk scoring system, or a system that establishes
misconduct/greenwashing by itself. Four layers stay distinct: evidence
retrieval -> automated classification -> human judgement -> approved
external communication.""")

md("""## 3. Current safeguards

Already-implemented safeguards confirmed by code inspection: provenance
fields (`src/validation.py`), evidence-strength gates, citation-ID
validation (`src/rag/citation_validator.py`), modality-routing gates,
protected-characteristic exclusion (`src/image_features.py`), face-embedding
exclusion, `.gitignore` secret exclusion, and a 586-test automated suite.""")

code("""subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_governance_framework.py")], check=True)""")

md("""## 4. Risk methodology

5x5 (likelihood x impact) matrix; scores are structured management
judgements, not empirical probabilities. Levels: 1-4 Low, 5-9 Medium,
10-14 High, 15-25 Critical.""")

code("""risks = pd.read_csv(TABLES / "ai_governance_risk_register.csv")
print(f"{len(risks)} risks registered")
risks["inherent_risk_level"].value_counts()""")

md("""## 5. Risk register""")

code("""risks[["risk_id", "risk_domain", "likelihood_1_5", "impact_1_5", "inherent_risk_score",
       "inherent_risk_level", "control_status", "risk_owner"]].sort_values("inherent_risk_score", ascending=False).head(12)""")

code("""Image(filename=str(FIGURES / "ai_risk_heatmap.png"))""")

md("""## 6. Control framework

Every High/Critical risk has at least one preventive AND one detective
control (enforced by `scripts/validate_governance_outputs.py` and
`tests/test_governance_framework.py`).""")

code("""controls = pd.read_csv(TABLES / "ai_governance_control_matrix.csv")
print(f"{len(controls)} controls")
print(controls["control_status"].value_counts())
print(controls["control_type"].value_counts())
Image(filename=str(FIGURES / "prevent_detect_correct_control_map.png"))""")

md("""## 7. Claim decision policy

Aligned does not mean independently proven; mixed requires presenting both
supporting and challenging evidence; insufficient evidence must never be
described as false.""")

code("""policy = pd.read_csv(TABLES / "claim_decision_policy.csv")
policy[["decision_label", "meaning", "required_reviewer", "escalation_requirement"]]""")

code("""Image(filename=str(FIGURES / "claim_decision_escalation_flow.png"))""")

md("""## 8. Human accountability

Every governance activity has exactly one Accountable ('A') role.""")

code("""raci = pd.read_csv(TABLES / "ai_governance_raci.csv")
role_cols = [c for c in raci.columns if c not in ("activity_id", "activity")]
assert ((raci[role_cols] == "A").sum(axis=1) == 1).all()
raci[["activity_id", "activity"]]""")

md("""## 9. Privacy and data lifecycle""")

code("""lifecycle = pd.read_csv(TABLES / "ai_data_lifecycle_policy.csv")
lifecycle[["lifecycle_stage", "personal_data_risk", "existing_or_proposed_status"]]""")

md("""## 10. Creator/customer ethics

Creator evidence: 60/61 verified records YouTube, 1 TikTok, 0 Instagram --
a targeted recovery sample, never presented as representative.""")

code("""coverage = pd.read_csv(TABLES / "creator_strategy_evidence_coverage.csv")
coverage[["brand", "verified_creator_records", "platforms_represented_creator"]]""")

md("""## 11. ESG assurance

Labour is TALA's largest claim category (14/37) and has zero independent
evidence in the current corpus -- the corpus's largest ESG gap.""")

code("""esg = pd.read_csv(TABLES / "esg_claim_assurance_matrix.csv")
esg[["claim_category", "current_tala_evidence_status", "current_evidence_gap"]]""")

code("""Image(filename=str(FIGURES / "esg_claim_assurance_matrix.png"))""")

md("""## 12. Regulatory map

Verified 22 September 2026. Voluntary standards (NIST AI RMF, ISO/IEC 42001,
OECD AI Principles) are never labelled mandatory. No definitive EU AI Act
classification is made -- potential applicability only.""")

code("""reg = pd.read_csv(TABLES / "regulatory_standards_map.csv")
reg[["requirement_id", "authority", "instrument", "mandatory_status", "legal_review_required"]]""")

md("""## 13. GenAI vendor controls

There is no fallback model. On any Gemini failure the system stops
generation, shows a clear error, and prevents unsupported output.""")

code("""genai = pd.read_csv(TABLES / "genai_vendor_control_matrix.csv")
genai[["control_area", "status"]]""")

md("""## 14. Monitoring indicators

Current values are computed from repository outputs where feasible; `TBC`
is used everywhere a value cannot yet be calculated -- never fabricated.""")

code("""kpi = pd.read_csv(TABLES / "ai_governance_kpi_kri.csv")
kpi[["kpi_kri", "current_value", "target"]]""")

md("""## 15. Incident management""")

code("""incidents = pd.read_csv(TABLES / "ai_incident_escalation_matrix.csv")
incidents[["incident_scenario", "severity", "notification_owner", "target_response_time"]]""")

md("""## 16. Implementation roadmap""")

code("""roadmap = pd.read_csv(TABLES / "ai_governance_implementation_roadmap.csv")
print(roadmap["phase"].value_counts())
Image(filename=str(FIGURES / "ai_governance_implementation_roadmap.png"))""")

code("""subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_governance_figures.py")], check=True)""")

md("""## 17. Deployment gates

Academic demonstration: GO. Controlled internal pilot: PARTIAL. External/
customer-facing deployment: NO-GO. Evidence-based, not preassigned.""")

code("""gates = pd.read_csv(TABLES / "ai_governance_go_no_go.csv")
gates.pivot(index="gate", columns="deployment_context", values="status")""")

md("""## 18. Final recommendations

1. Present this system as academic-demonstration-ready, pilot-conditional,
   and not ready for external/customer-facing deployment.
2. Prioritise closing the labour-evidence gap (14/37 claims, 0 independent
   evidence) before any ESG-related external communication.
3. Confirm the Gemini API account tier (free vs. paid) before any wider use.
4. Stand up the human-approval and sustainability-review sign-off workflows
   before treating any finding as pilot-ready.
5. Present the single Critical risk (R09 -- defamation/reputational harm)
   and its controls explicitly to any faculty/business audience.

Full narrative: `docs/ai_governance_ethics_esg_framework.md`.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

out_path = "notebooks/07_ai_governance_ethics_esg.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote", out_path, "with", len(cells), "cells")
