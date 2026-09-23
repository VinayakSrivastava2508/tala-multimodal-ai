# AI Governance, Ethics and ESG Framework — TALA Multimodal AI Strategy

**Day 3D · SPJIMR ANA526-PPM.** Governs the system already built: public-data
collection, text/image/video processing, claim-evidence fusion, local
ChromaDB retrieval, Gemini-generated answers, citation validation,
creator-strategy analysis, and management use of aligned/mixed/
insufficient-evidence findings. Sources verified 22 September 2026 (see
§15 / `outputs/tables/regulatory_standards_map.csv`).

## 1. Executive governance position

This is an **internal decision-support prototype**, built for an MBA course
sprint, that surfaces where TALA's official claims are corroborated,
contested, or under-evidenced by the currently collected public corpus. Of
37 official claims, the verified corpus places 14 (37.8%) `aligned`, 5
(13.5%) `mixed` — consolidating into 2 underlying themes — and 18 (48.6%)
`insufficient_evidence`; 14 claims (the largest category) are labour-related
and none currently have independent evidence. The system is **academic-
demonstration-ready today**; it is **not** ready for external or
customer-facing use (see §18). Every material risk and control in this
document is tied to this system's actual data, code, or intended managerial
use — not generic AI-ethics boilerplate.

## 2. Intended use and prohibited use

**Intended use:** internal management decision support for investigating
divergence between TALA's official claims and observable customer/press
evidence, and for descriptively comparing TALA's creator strategy to three
named competitors, always with human review before any external use.

**This system must never be described or used as:**
- An autonomous legal or regulatory decision-maker
- Proof that a sustainability claim is true or false
- A substitute for sustainability assurance or product-quality investigation
- A customer-facing claims engine
- A creator-risk scoring system
- A system that establishes misconduct or greenwashing by itself

Governance preserves four distinct layers: **(1) evidence retrieval →
(2) automated analytical classification → (3) human management judgement →
(4) approved external communication.** No automated output skips directly
to (4).

## 3. Current system and evidence boundary

- **Text/image/video/reference pipelines**: separate per-modality processing
  (`src/text_features.py`, `src/image_features.py`, `src/video_features.py`,
  `src/reference_features.py`), each validated before fusion.
- **Claim-evidence fusion**: 119 evidence units across 37 claims
  (`data/processed/fusion/claim_evidence_units.csv`), evidence-gated and
  sensitivity-tested across 4 modality-weighting configurations.
- **Local ChromaDB**: claim, text-evidence, and visual-evidence collections,
  persisted locally, not networked (`src/rag/chroma_store.py`).
- **Gemini** is the **only** answer generator; there is **no fallback** — on
  any failure the system stops generation and shows a clear error
  (`src/rag/gemini_generator.py`).
- **Citation validation** checks every generated answer's citations resolve
  to real, relevant evidence (`src/rag/citation_validator.py`).
- **Creator evidence** is concentrated on YouTube (60 of 61 verified
  records; 1 TikTok; 0 Instagram) — a targeted recovery sample, not a
  statistically representative one.
- **No protected-characteristic inference, no face embeddings, no
  engagement prediction, no manual-labelling workflow** exist anywhere in
  the inspected codebase (confirmed by code inspection and the existing test
  suite, e.g. `tests/test_image_features.py`).

## 4. Governance operating model

Five decision layers (`outputs/figures/ai_governance_operating_model.png`;
full control detail in `outputs/tables/ai_governance_control_matrix.csv`):

1. **Data admission** — source legality, provenance, rights basis, minimisation,
   duplicate/search-snippet rejection, evidence-strength gates, retention.
2. **Modality processing** — text/image/video eligibility, protected-
   characteristic and face-embedding prohibition, missing-modality
   disclosure, model-version recording.
3. **Fusion and analytical classification** — claim linkage, visual-
   groundability gates, support/challenge/context separation, self-reported
   vs. independent tagging, sensitivity testing, human escalation flags.
4. **Retrieval and generation** — Chroma integrity, metadata filtering,
   retrieval trace, Gemini context minimisation, citation validation,
   label preservation, no-fallback failure handling.
5. **Managerial use and external communication** — intended-use restriction,
   human approval, ESG/legal sign-off, correction/retraction, audit
   retention.

## 5. Risk-assessment method

A 5×5 (likelihood × impact) matrix: likelihood 1 (rare) to 5 (near-certain
within a review cycle); impact 1 (negligible) to 5 (severe — reputational,
legal, or decision-quality harm). `inherent_risk_score = likelihood ×
impact` (max 25); levels: 1–4 Low, 5–9 Medium, 10–14 High, 15–25 Critical.
**These are structured management judgements calibrated against the
inspected system state, not empirical probabilities or actuarial
estimates.** Full register: `outputs/tables/ai_governance_risk_register.csv`
(26 risks; figure: `outputs/figures/ai_risk_heatmap.png`).

## 6. Principal risks

Of 26 registered risks: 1 Critical (**R09 — defamation and reputational
harm**, score 15), 10 High (evidence staleness, sampling bias, creator
misrepresentation, greenwashing false positive/negative, self-reported
evidence treated as independent, API data exposure, secret leakage,
unsupported external communication, overreliance by management), 15 Medium.
No risk in the current register is scored Low, reflecting the genuinely
sensitive nature of claim-experience and creator-strategy findings even at
academic-prototype scale.

## 7. Preventive, detective and corrective controls

52 controls (`outputs/tables/ai_governance_control_matrix.csv`; figure:
`outputs/figures/prevent_detect_correct_control_map.png`): 30 preventive, 19
detective, 3 corrective. **37 are already implemented** in the inspected
codebase (e.g. provenance fields, evidence-strength gates, citation
validation, protected-characteristic and face-embedding exclusion,
`.gitignore` secret exclusion, the 586-test automated suite); 3 are
partially implemented; **12 are proposed enterprise controls not yet
built** (e.g. human-approval sign-off workflow, sustainability-claim
approval gate, secret-scanning CI, scheduled evidence refresh). Every High
and Critical risk has at least one preventive and one detective control
(verified by `scripts/validate_governance_outputs.py` and
`tests/test_governance_framework.py`).

## 8. Claim decision policy

Full table: `outputs/tables/claim_decision_policy.csv` (figure:
`outputs/figures/claim_decision_escalation_flow.png`). Key rules:
**aligned does not mean independently proven**; **mixed requires explicit
presentation of both supporting and challenging evidence**, never a
binary verdict; **divergent/contradicted findings require human review**
before any use (0 claims currently hold this label, but the rule stands);
**insufficient evidence must never be described as false**; self-reported
evidence must be labelled as such; weight-sensitive findings carry an
additional caveat; labour/ESG claims lacking independent evidence require
Sustainability/ESG review before any external framing.

## 9. Human oversight and accountability

`outputs/tables/ai_governance_raci.csv` assigns exactly one Accountable role
per governance activity across 14 activities (source approval, rights-basis
approval, data minimisation, evidence refresh, model change, prompt change,
claim review, mixed/divergent escalation, sustainability-claim approval,
creator-related publication, incident response, correction/retraction,
production release, quarterly governance review). No activity has zero or
multiple Accountable roles.

## 10. Data privacy and lifecycle

`outputs/tables/ai_data_lifecycle_policy.csv` covers 13 stages from
discovery through deletion. Personal-data minimisation is applied at
collection (usernames stripped; no profile photos/personal account details
collected — `docs/governance_notes.md`); **public availability is not
treated as automatic permission for unlimited reuse.** Formal retention
schedule and automated deletion are **proposed**, not yet implemented — the
current academic-prototype posture relies on a documented "academic use
only" statement and manual discipline.

## 11. Creator and customer ethics

Creator partnership type and content intent are automated, evidence-gated
classifications, never human-adjudicated ground truth; records requiring
human validation are explicitly flagged
(`partnership_human_review_required`). Customer review data is analysed in
aggregate; no individual profiling. Creator evidence is overwhelmingly
YouTube-based and a targeted, non-random sample — never presented as
representative of a brand's complete creator strategy.

## 12. Copyright and media rights

Only `official_direct_public_asset`, `open_license`, `group_owned`, or
`user_authorised` bases permit local frame/temporal processing
(`configs/schema.yaml::video_assets`); no unofficial downloaders or
anti-bot bypass tools are used. Rights basis is self-declared at collection
time, **not independently legally cleared** — this is a documented gap for
any pilot/external use (see risk R02, control C05).

## 13. Generative-AI and vendor governance

`outputs/tables/genai_vendor_control_matrix.csv`. **There is no fallback
model.** On any Gemini API failure, the system stops generation, shows a
clear error, and prevents unsupported output — it never substitutes a
template or alternate generator (`src/rag/gemini_generator.py`). Context
sent to Gemini is minimised to retrieved evidence passages and IDs. **Open
gap:** whether the project's Gemini API key is on the free or paid tier is
not recorded in this repository, and this materially changes Google's
data-use/training terms (see §15 REG10).

## 14. ESG claim assurance

`outputs/tables/esg_claim_assurance_matrix.csv` covers 9 claim categories
with acceptable/ineligible evidence types and current gaps. The corpus's
largest and least-evidenced category is **labour** (14/37 claims, 0 with
independent evidence). The tier-3 supplier/material-innovation mixed theme
(4 claim variants) and the conscious-value-proposition mixed theme (1
claim) are the corpus's most concrete claim-experience tensions. This
framework **does not independently accuse TALA of greenwashing** — it
defines the evidence thresholds and review gates that would be needed
before any such characterisation could be responsibly made.

## 15. Regulatory and standards alignment

11 rows in `outputs/tables/regulatory_standards_map.csv`, each with an
official source URL, verified 22 September 2026: UK ICO AI/data-protection
guidance, UK CMA Green Claims Code, UK ASA/CAP environmental-claims
guidance, EU GDPR, EU AI Act (potential applicability only — **no
definitive legal classification is made in this document**; qualified
legal review is required), UK GDPR/DPA 2018, US FTC Endorsement Guides,
NIST AI RMF (voluntary), ISO/IEC 42001 (voluntary), OECD AI Principles
(voluntary), and Google's Gemini API terms. Voluntary standards are never
converted into legal obligations in this framework.

## 16. Monitoring and incident management

20 KPI/KRI indicators (`outputs/tables/ai_governance_kpi_kri.csv`) with
formula, target, warning, and breach thresholds; current values are
computed from repository outputs where feasible (e.g. independent-evidence
coverage, insufficient-evidence share, sensitivity share) and marked `TBC`
elsewhere — never fabricated. 11 incident scenarios
(`outputs/tables/ai_incident_escalation_matrix.csv`) span exposed
credentials through unsupported greenwashing allegations, each with a named
containment/investigation owner and target response time (as fast as <4
hours for an exposed API key).

## 17. Implementation roadmap

4 phases, 26 initiatives (`outputs/tables/ai_governance_implementation_roadmap.csv`;
figure: `outputs/figures/ai_governance_implementation_roadmap.png`):
**Phase 0** (current state, 3 already-implemented control groups),
**Phase 1** (0–30 days: named owners, access controls, approved source
register, intended-use statement, legal/DPO review, sustainability-review
process, incident procedure, model/prompt register),
**Phase 2** (31–90 days: scheduled evidence refresh, independent ESG-
evidence integration, monitoring dashboard, reviewer workflow, correction
process, vendor review, training, audit sampling),
**Phase 3** (3–6 months: enterprise identity/access, model-risk committee,
continuous monitoring, independent audit, business-system integration,
red-team testing, governance effectiveness review). Every initiative has a
named owner and completion criterion; cost bands are relative
(Low/Medium/High), never fabricated currency estimates.

## 18. Deployment gates

`outputs/tables/ai_governance_go_no_go.csv` assesses 13 gates across 3
deployment contexts:

| Context | Verdict |
|---|---|
| Academic demonstration | **GO** (all 13 gates) |
| Controlled internal pilot | **PARTIAL** (all 13 gates — core controls exist; named-owner sign-off, formal DPO/legal review, and reviewer workflows are Phase 1 gaps) |
| External/customer-facing deployment | **NO-GO** (all 13 gates — independent ESG assurance, enterprise access control, continuous monitoring, and drilled incident response do not yet exist) |

These are evidence-based outcomes, not preassigned — every gate documents
its supporting evidence and gap.

## 19. Limitations

- Automated fusion and creator-classification labels are not human-adjudicated.
- No legal or DPO review has yet been performed; the EU AI Act applicability
  row is explicitly non-definitive.
- Gemini API account tier (free vs. paid) is unconfirmed, affecting
  applicable data-use terms.
- Most KPI/KRI current values are `TBC` — no continuous monitoring exists yet.
- Retention/deletion automation is proposed, not implemented.
- This framework was built by inspecting code and existing tests, not by an
  independent third-party audit.

## 20. Deck-ready recommendations

1. Treat this as an academic-demonstration-ready, pilot-conditional,
   externally-not-ready system — say so explicitly in any presentation.
2. Prioritise closing the labour-evidence gap (14/37 claims, 0 independent
   evidence) before any ESG-related external communication.
3. Confirm the Gemini API account tier before any wider use.
4. Stand up the human-approval and sustainability-review sign-off workflows
   (Phase 1) before treating any finding as pilot-ready.
5. Present the risk register's single Critical risk (defamation/reputational
   harm) and its controls explicitly to any faculty/business audience.
