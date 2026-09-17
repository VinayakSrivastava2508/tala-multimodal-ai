# CLAUDE.md — AI Coding Assistant Rules for This Project

This file governs how Claude Code should behave when assisting with this project. Read this before taking any action.

## Project Identity

**Project:** TALA Multimodal AI Strategy
**Course:** SPJIMR ANA526-PPM
**Type:** 3-day MBA group project sprint
**Constraint:** Free/open-source tools only; public or manually compiled data only.

---

## 1. No-Fabrication Rule (Highest Priority)

- Never generate synthetic data and present it as real.
- Never invent review text, social post content, brand claims, or metrics.
- If a dataset is unavailable, create an empty CSV template with correct column headers and document the limitation in `docs/source_log_template.md`.
- Label any placeholder or illustrative data clearly with a `synthetic=True` column or a `# PLACEHOLDER` comment.
- All example outputs in notebooks must use real data or be clearly marked as illustrative.

---

## 2. Research Framing Rules

**Evidence source discipline — enforce this in all analysis code:**

| Research Question | Correct Evidence Sources |
|-------------------|--------------------------|
| Product quality, fit, sizing, durability | Trustpilot, Google Reviews, Reddit, press coverage, returns/complaints data |
| Sustainability / responsibility claims | Official brand website copy, annual/impact reports, press, NGO assessments |
| Creator strategy, brand positioning | Instagram, TikTok, YouTube, creator posts, partnership announcements |
| Claim–experience divergence | Official claims corpus **vs** customer/press corpus (not social engagement) |

Never use Instagram engagement (likes, follower counts, aesthetics) as a proxy for product quality or sustainability validity. Flag any analysis that conflates these.

---

## 3. Citation and Provenance Rules

- Every dataset loaded in a notebook must reference its row in `docs/source_log_template.md`.
- Every figure saved to `outputs/figures/` must include a title referencing its source dataset.
- Every claim in analysis outputs must have a traceable `source_id` linking to the source log.
- Do not copy-paste web content into source files without attributing URL, access date, and data type.

---

## 4. Coding Standards

- Python 3.10+. Type hints on all function signatures.
- All functions in `src/` must have a one-line docstring stating what it returns and what it expects. No multi-paragraph docstrings.
- No hardcoded file paths — use `pathlib.Path` and resolve from project root.
- Load configuration from `configs/` YAML files; do not hardcode brand names or schema field names in module code.
- Prefer `pandas` for tabular work; `scikit-learn` for ML; `sentence-transformers` for embeddings.
- Do not use OpenAI, Cohere, or any paid API without explicit team approval. Free Hugging Face models are fine.
- All random operations must use a fixed seed (`RANDOM_SEED = 42`).
- Notebooks are for analysis and narrative; business logic belongs in `src/` modules.

---

## 5. Data Ethics Rules

- Do not scrape personal user data (names, profile pictures, account details) from social platforms.
- Do not store raw scraped content that includes personally identifiable information.
- When collecting public review data, strip usernames before saving to `data/raw/`.
- Respect platform rate limits; include `time.sleep()` in any scraping loops.
- Do not attempt to bypass login walls, paywalls, or bot detection.
- If a data source is ambiguous, log it in `docs/governance_notes.md` and flag for team review.

---

## 6. File and Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `src/ingestion.py` | Load CSVs, call free public APIs, gentle HTTP requests |
| `src/validation.py` | Validate DataFrames against `configs/schema.yaml` |
| `src/text_features.py` | Sentence embeddings, TF-IDF, sentiment, topic modelling |
| `src/image_features.py` | Image feature extraction (CLIP/ViT via HuggingFace, color palettes) |
| `src/video_features.py` | Frame sampling, thumbnail analysis |
| `src/fusion_models.py` | Early, late, and hybrid multimodal fusion |
| `src/moat_metrics.py` | Brand moat and creator strategy metrics |
| `src/rag_pipeline.py` | ChromaDB/FAISS RAG construction and querying |
| `src/visualisations.py` | All plot generation; save to `outputs/figures/` |

---

## 7. What Claude Should NOT Do

- Do not add features beyond what the current task requires.
- Do not refactor working code unless asked.
- Do not create markdown documentation files beyond what is listed in the scaffold.
- Do not suggest paid tools or APIs (Apify, Bright Data, etc.) without being asked.
- Do not run web scraping without the user confirming it is acceptable for the target site.
- Do not commit or push without explicit user instruction.

---

## 8. Day-by-Day Priorities

**Day 1:** Data templates, schema validation, EDA only.
**Day 2:** Feature extraction, fusion baselines, claim divergence scoring.
**Day 3:** RAG pipeline, evaluation, output assembly.

Do not skip ahead. Working reproducible analysis beats incomplete ambitious pipelines.
