# TALA Multimodal AI Strategy
### SPJIMR ANA526-PPM | Group Project | 3-Day Sprint

## Project Objective

This project diagnoses the gap between TALA's official brand claims and actual customer/product experience using multimodal AI, and benchmarks TALA's creator-led cross-platform strategy against three competitors (Adanola, Girlfriend Collective, Oner Active).

**Two core research questions:**
1. Where does TALA's official brand narrative (quality, sustainability, inclusivity) diverge from customer experience signals (reviews, complaints, returns)?
2. How does TALA's creator-led D2C strategy differ structurally from competitor approaches, and what signals predict brand moat durability?

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
- Run `02_multimodal_features_and_fusion.ipynb` (early / late / hybrid fusion).
- Run `03_creator_strategy_and_claim_divergence.ipynb` (claim gap scoring, moat metrics).

### Day 3 — RAG, Evaluation & Deck
- Build RAG pipeline across three corpora (`src/rag_pipeline.py`).
- Run `04_multimodal_rag_and_evaluation.ipynb`.
- Export `outputs/` figures and tables; assemble presentation deck.

## Notebook Execution Order

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_data_compilation_and_eda.ipynb` | Load, validate, and explore all datasets |
| 02 | `02_multimodal_features_and_fusion.ipynb` | Multimodal feature extraction and fusion |
| 03 | `03_creator_strategy_and_claim_divergence.ipynb` | Creator benchmarking and claim-experience gap |
| 04 | `04_multimodal_rag_and_evaluation.ipynb` | RAG pipeline construction and evaluation |

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
