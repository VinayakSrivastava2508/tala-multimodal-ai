# TALA Multimodal AI Strategy
### SPJIMR ANA526-PPM | Group Project | 3-Day Sprint

## Project Objective

This project diagnoses the gap between TALA's official brand claims and actual customer/product experience using multimodal AI (text, image, video, and reference-document evidence), and benchmarks TALA's creator-led cross-platform strategy against three competitors (Adanola, Girlfriend Collective, Oner Active).

**Two core research questions:**
1. Where does TALA's official brand narrative (quality, sustainability, inclusivity) diverge from customer experience signals (reviews, complaints, returns), once official text, product/UGC images, video evidence, and reference documents are integrated?
2. How does TALA's creator-led D2C strategy differ structurally from competitor approaches — described, not predicted (see [Scope Boundaries](#scope-boundaries) below)?

> Full binding scope: `docs/project_charter.md`. Condensed enforcement summary: `CLAUDE.md` §9.

## Scope Boundaries

- **Engagement prediction is out of scope for the entire project.** No engagement
  regression/classification, no out-of-fold predictions, no actual-vs-predicted residuals,
  no predictive creator-performance scoring, no engagement-focused fusion. Real YouTube
  engagement metrics are retained for **descriptive, non-causal** comparison only (medians,
  IQRs, rates by brand/partnership-type/content-intent).
- **A video URL, title, thumbnail, description, or engagement statistic is not a processed
  video asset.** It is a "video lead" until genuine temporal processing (multiple sampled
  frames, motion/scene-change features, or a transcript) has run on a locally held,
  rights-cleared file. See `docs/assignment_alignment_audit.md` for current Image/Video
  package status.
- Do not call an observed brand difference a "moat" without outcome evidence — describe it
  as an observed distinctive pattern.

## Research Framing — Read Before Analysis

> **Evidence source discipline:**
> - **Quality & responsibility claims** → assessed from official brand copy, Trustpilot/review aggregators, Reddit threads, customer complaint posts, and press coverage.
> - **Creator/platform strategy** → assessed from Instagram, TikTok, YouTube, and creator posts. These sources are evidence for *representation, partnership intent, campaign tone and brand positioning only* — not for product quality or sustainability validation.
>
> Do not use Instagram engagement metrics as proxies for quality or responsibility.

## Repository Layout

```
tala-multimodal-ai/
├── data/
│   ├── raw/             # Source data — never modify in place
│   ├── interim/         # Cleaned, not yet featurised
│   ├── processed/       # Feature-ready datasets
│   ├── manual_labels/   # Human-annotated CSVs
│   └── corpora/         # Separated RAG corpora (official / creator / customer)
├── notebooks/           # Run in numbered order
├── src/                 # Reusable Python modules
├── configs/             # YAML config and schemas
├── docs/                # Data dictionary, governance, source log
└── outputs/
    ├── figures/
    ├── tables/
    └── deck_assets/
```

## 3-Day Workflow

### Day 1 — Data Compilation & EDA

**Step 1 — Copy templates**
```
data/raw/templates/ → data/raw/
```
Rename each copy: `official_claims_template.csv` → `official_claims_tala.csv`, etc.

**Step 2 — Collect sources manually**

| File | Sources | Evidence type |
|------|---------|---------------|
| `official_claims_tala.csv` | weartala.com/sustainability, /responsibility, /about, press interviews | `official` |
| `customer_reviews_tala.csv` | Trustpilot (public), Google Reviews, Reddit complaints | `customer_experience` |
| `press_reddit_sources_tala.csv` | Good on You, Guardian, r/femalefashionadvice, r/gymsnark | `press` / `community` |
| `creator_posts_tala.csv` | Instagram/TikTok creator tags, #ad posts, ambassador content | `creator_strategy` |
| `competitor_platforms.csv` | Adanola, GF Collective, Oner Active — IG/TikTok/YT profile snapshots | `competitor_benchmark` |

**Step 3 — Register every source**

Add a row to `docs/source_log_template.md` for every URL you collect from.

**Step 4 — Validate provenance**
```bash
python scripts/validate_data.py
```
Fix all FAIL errors. WARN and EMPTY are acceptable for Day 1.

**Step 5 — Run EDA notebook**
```bash
jupyter lab
# Open notebooks/01_data_compilation_and_eda.ipynb
# Kernel: Python (TALA Multimodal AI)
```

**Day 1 done when:** All loaded datasets pass validation; EDA plots for ratings, complaint types, creator tier, and competitor followers are visible.

---

### Day 2 — Modelling & Features
- Extract text, image, and platform features (`src/text_features.py`, `src/image_features.py`).
- Run `02_multimodal_features_and_fusion.ipynb` — text/visual feature construction, YouTube
  metadata enrichment, revised partnership/intent classification, descriptive engagement
  analysis, and modelling-feasibility reassessment (engagement-prediction tracks marked
  `OUT_OF_SCOPE` — see `outputs/tables/day2_modelling_feasibility.csv`).

### Day 2.5 — Modality Realignment
- Build genuine Image and Video packages (`scripts/collect_official_product_media.py`,
  `scripts/build_video_asset_manifest.py`, `scripts/process_video_assets.py`).
- Run `03_video_pipeline_and_multimodal_evidence.ipynb` — modality gap audit, image/video
  package construction, frame-level and temporal video features, claim-level multimodal
  evidence candidates.
- Full status: `docs/assignment_alignment_audit.md`.

### Day 3 — RAG, Evaluation & Deck
- Build claim-evidence fusion and RAG pipeline across text/image/video/reference corpora (`src/rag_pipeline.py`, `src/fusion_models.py`).
- Run `04_multimodal_rag_and_evaluation.ipynb`.
- Export `outputs/` figures and tables; assemble presentation deck with governance/ethics recommendations.

## Notebook Execution Order

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_data_compilation_and_eda.ipynb` | Load, validate, and explore all datasets |
| 02 | `02_multimodal_features_and_fusion.ipynb` | Text/visual feature extraction, YouTube enrichment, descriptive engagement |
| 03 | `03_video_pipeline_and_multimodal_evidence.ipynb` | Image/video package construction, temporal features, claim-evidence candidates |
| 04 | `04_multimodal_rag_and_evaluation.ipynb` | RAG pipeline construction and evaluation |

`03_creator_strategy_and_claim_divergence.ipynb` is scaffolded for the later claim-alignment
scoring task (not yet built — gated on `03_video_pipeline_and_multimodal_evidence.ipynb`'s
evidence candidates).

## Executive Decision Cockpit (Streamlit app)

A six-page executive Streamlit app sits on top of the validated Day 3 outputs and the
multimodal RAG pipeline: Executive Overview, Claim Diagnostic, Creator Strategy,
Multimodal RAG, Governance & Roadmap, Methodology & Limitations.

```bash
streamlit run app\streamlit_app.py
```

Full usage guide: `docs/executive_cockpit_user_guide.md`.

## Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate (Windows)
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env   # then edit as needed

# 5. Launch Jupyter
jupyter lab
```

## Academic Context

| Field | Value |
|-------|-------|
| Course | ANA526-PPM — Multimodal AI Strategy |
| Institution | SPJIMR |
| Focal brand | TALA (premium D2C activewear, UK, est. 2019) |
| Competitors | Adanola, Girlfriend Collective, Oner Active |
| Deadline | 3-day sprint |

## Key Constraints

- All data collection must respect platform Terms of Service.
- No paid scraping APIs without explicit team approval.
- No fabricated data — all datasets must be traceable to a real source or clearly labelled as a manual/synthetic template.
- Every claim in the final deck must cite a specific row in a dataset or a documented source.
