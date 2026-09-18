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
**Day 2:** Feature extraction, claim-evidence fusion groundwork, claim divergence scoring.
**Day 2.5:** Modality realignment — genuine Image and Video packages, claim-level multimodal
evidence candidates (see §9).
**Day 3:** RAG pipeline, evaluation, output assembly.

Do not skip ahead. Working reproducible analysis beats incomplete ambitious pipelines.

---

## 9. Assignment-Alignment Charter (Non-Negotiable)

Full detail in `docs/project_charter.md`. This section is the binding summary — if any
other document in this repo conflicts with it, this charter wins.

### The assignment requires

1. A **Text Package** (official claims, customer reviews, creator/social text, reference
   documents).
2. An **Image Package** (official catalog/product images, permitted UGC, creator
   thumbnails).
3. A **Video Package** — genuine temporal processing (sampled frames, motion/scene-change
   features, frame-sequence embeddings, transcripts where legitimately available). A video
   URL, title, thumbnail, description, or engagement statistic is **not** a video pipeline
   and must never be reported as one.
4. A **Multimodal Reference Package** (impact/responsibility reports, certifications,
   material specs, sizing/fabric-care guidance, return policies).
5. **Separate text, image, and video analytical pipelines** — each modality processed and
   validated in its own right before any fusion.
6. **Multimodal fusion** — mandatory, but its primary purpose is **claim-experience
   evidence integration** (official claim vs. customer/press/creator/visual evidence), not
   engagement prediction. See the fusion-scope note below.
7. A **working multimodal RAG prototype** over the three evidence corpora plus image/video/
   reference evidence.
8. **Governance, ethics, and implementation recommendations** as a first-class deliverable,
   not an afterthought.

### Primary objective

Diagnose divergence between TALA's official quality/responsibility claims and customer
experience by integrating official and customer text, product/catalog and permitted UGC
images, actual video frames/temporal features/transcripts where available, and enterprise
reference documents.

### Secondary objective

Compare TALA's creator-partnership and official-platform strategy with Adanola, Girlfriend
Collective, and Oner Active — descriptively, not predictively (see engagement scope below).

### Engagement scope — descriptive only

Engagement prediction is **out of scope**. Do not implement: engagement regression,
out-of-fold engagement predictions, actual-vs-predicted residuals, predictive
creator-performance scoring, or engagement-focused early/late/hybrid fusion. Existing
YouTube engagement metrics (view/like/comment counts, rates, within-channel z-scores) may
be **retained and used only for descriptive, non-causal comparison** (medians, IQRs,
coverage) — never as a modelling target.

### Fusion — what "fusion" means in this project

"Fusion" is not banned — it is **redirected**. Do not blindly strip every mention of early/
late/hybrid fusion from the codebase: `src/fusion_models.py`'s early/late/hybrid utilities
remain because they are the mechanism for **claim ↔ evidence** integration (Part I onward),
not for predicting engagement outcomes. When you see "fusion" in this repo, ask which
purpose it serves before touching it — see `docs/project_charter.md` §"Role of Multimodal
Fusion" for the classification rule used to audit historical mentions.

### Video-processing definition (binding)

A record counts as a **processed video asset** only if it has genuine temporal information:
multiple sampled frames, motion/scene-change features, frame-sequence embeddings, or a
transcript/audio evidence trail. A YouTube URL, oEmbed thumbnail, title, description, or
engagement statistic is a **video lead**, not a processed asset, and must be labelled
`platform_metadata_only` in `rights_or_access_basis` — never silently counted toward video
coverage.

### Access basis — what may be locally processed

Only `official_direct_public_asset`, `open_license`, `group_owned`, or `user_authorised`
permit local frame/temporal processing (see `configs/schema.yaml::video_assets`). Do not use
yt-dlp, unofficial downloaders, or stream extraction against YouTube/Instagram/TikTok — do
not bypass login, CAPTCHA, paywall, or anti-bot controls. If the genuinely processable video
count falls short of target, report the exact shortfall and mark the Video Package PARTIAL
or FAIL — never fabricate assets to hit a number.

### Required final deliverables

Text Package, Image Package, Video Package, Multimodal Reference Package, per-modality
pipelines, claim-evidence fusion, a working multimodal RAG prototype, and a governance/
ethics/implementation-recommendations section — all traceable to real, provenance-tracked
evidence.
