# TALA Multimodal AI Strategy -- Submission README

**SPJIMR ANA526-PPM | 3-day MBA group project sprint**

This README describes how to reproduce and navigate the submission. It is written for the
final, working-tree state of this repository as of the submission freeze.

---

## 1. Project objective

**Primary objective:** diagnose divergence between TALA's official quality/responsibility
claims and customer experience by integrating official and customer text, product/catalog
and permitted UGC images, actual video frames/temporal features where available, and
enterprise reference documents.

**Secondary objective:** compare TALA's creator-partnership and official-platform strategy
with Adanola, Girlfriend Collective, and Oner Active -- descriptively, not predictively.
Engagement prediction is explicitly out of scope (see `docs/project_charter.md`).

The full binding charter is `CLAUDE.md` Section 9 / `docs/project_charter.md`.

---

## 2. Repository structure

```
app/          Streamlit "TALA Multimodal AI Decision Cockpit" (6 pages)
configs/      YAML configuration (schema, fusion_rules) -- no hardcoded brand/field names
data/         raw -> interim -> corpora -> processed -> fusion -> vector_db
docs/         Audit trails, governance/methodology notes, project charter
notebooks/    7-notebook analytical sequence (see notebook_manifest.csv)
outputs/      tables/ (analysis CSVs) and figures/ (PNG charts)
scripts/      Build/validation scripts (one script per pipeline stage)
src/          Reusable modules (collectors, fusion, rag, image/video/text features)
submission/   This handoff package (deck facts, findings, figures, architecture docs)
tests/        pytest suite (636 tests)
```

## 3. Data packages

| Package | Location | Scale |
|---|---|---|
| Text | `data/corpora/{official_claims,customer_experience,creator_strategy}_corpus.csv` | 37 / 95 / 61 rows |
| Image | `data/interim/day2_5/image_assets.csv` + CLIP/palette features | 24 official product images |
| Video | `outputs/tables/video_asset_coverage.csv` | 8 genuinely processed (frames+temporal features) of 61 total leads; 53 leads are `platform_metadata_only` |
| Multimodal Reference | `data/interim/day2_6/multimodal_reference_assets.csv` | 162 rows (impact reports, certs, sizing/care, policies) |

## 4. Notebook sequence

Run in order (see `submission/repository_manifest/notebook_manifest.csv` for full detail,
including which cells need `GEMINI_API_KEY`):

1. `01_data_compilation_and_eda.ipynb` -- Day 1 EDA
2. `02_multimodal_features_and_fusion.ipynb` -- Day 2 features
3. `03_video_pipeline_and_multimodal_evidence.ipynb` -- Day 2.5 video package
4. `04_primary_multimodal_fusion.ipynb` -- Day 3A claim-evidence fusion
5. `05_multimodal_rag.ipynb` -- Day 3B RAG (one cell requires `GEMINI_API_KEY`)
6. `06_analytical_synthesis_and_creator_strategy.ipynb` -- Day 3C synthesis
7. `07_ai_governance_ethics_esg.ipynb` -- Day 3D governance

Two early scaffold notebooks (`03_creator_strategy_and_claim_divergence.ipynb`,
`04_multimodal_rag_and_evaluation.ipynb`) are unexecuted placeholders superseded by the
sequence above -- see `submission/repository_manifest/excluded_notebooks_note.md`.

## 5. Reproduction steps (Windows / Git Bash)

```bash
# From the repository root:
source .venv/Scripts/activate        # or: python -m venv .venv && pip install -r requirements.txt
cp .env.example .env                 # then fill in values -- never commit .env
python scripts/validate_data.py
python scripts/validate_synthesis_outputs.py
python scripts/validate_governance_outputs.py
python -m pytest -q
```

To regenerate figures (idempotent, no data changes):

```bash
python scripts/build_synthesis_figures.py
python scripts/build_governance_figures.py
```

To rebuild the RAG index (no Gemini call; embeddings only):

```bash
python scripts/build_chroma_index.py
```

## 6. Environment setup

- Python 3.10+ (developed/tested on 3.14).
- Create and activate a virtual environment, then `pip install -r requirements.txt`.
- Copy `.env.example` to `.env` and fill in values locally. `.env` is git-ignored and must
  never be committed.

## 7. Environment-variable names (never values)

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key, used only by `src/rag/gemini_generator.py` |
| `GEMINI_MODEL` | Gemini model id (e.g. `gemini-3.1-flash-lite`) |
| `APIFY_TOKEN` | Apify free-tier token (Trustpilot/search collection, Day 1) |
| `YOUTUBE_API_KEY` | YouTube Data API v3 (free quota), creator/platform collection |
| `HF_TOKEN` | Optional Hugging Face token (free tier) |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USER_AGENT` | Reddit public API |
| `CHROMA_PERSIST_DIR` | Local ChromaDB persistence path |
| `RANDOM_SEED` | Fixed at 42 for all random operations |

Note: `GEMINI_API_KEY` / `GEMINI_MODEL` are read by the code (`src/rag/gemini_generator.py`)
but are not yet listed in `.env.example` -- add them there before a fresh environment setup.

## 8. Commands for validators and tests

```bash
python scripts/validate_data.py                 # 33/33 PASS at submission freeze
python scripts/validate_synthesis_outputs.py     # 19/19 PASS
python scripts/validate_governance_outputs.py    # 12/12 PASS
python -m pytest -q                              # 636 passed at submission freeze
python -m pytest tests/test_audit_artifacts.py   # Acceptance-Patch audit integrity (8/8 PASS, 40/40 feature-test rows)
```

## 9. How to launch Streamlit

```bash
source .venv/Scripts/activate
streamlit run app/streamlit_app.py
```

Opens the "TALA Multimodal AI Decision Cockpit" at `http://localhost:8501` with 6 pages:
Executive Overview, Claim Diagnostic, Creator Strategy, Multimodal RAG, Governance &
Roadmap, Methodology & Limitations. Opening the RAG page never triggers a Gemini call --
generation only happens on an explicit "Investigate"/"Submit" click.

## 10. Known limitations

- 18 of 37 claims (largest group) are labelled `insufficient_evidence`; this means
  unvalidated, not false.
- Labour claims (14 of 37, the largest category) have zero independent evidence anywhere
  in the corpus.
- Only 8 of 61 total video leads are genuinely processed (frames + temporal features); the
  remaining 53 are platform-metadata-only leads.
- No Instagram creator records are verified for any of the 4 brands; the creator-strategy
  comparison is YouTube-dominated (60 of 61 records).
- Automated NLI/rule-based fusion labels are not human-adjudicated.
- Citation validation covers 3 fixed demo cases, not a continuously monitored production
  query stream.
- A dedicated smaller-laptop responsive-viewport pass and systematic keyboard-navigation
  accessibility pass have not been performed on the Streamlit app.
- `GEMINI_API_KEY` / `GEMINI_MODEL` are not yet documented in `.env.example` (see §7).

## 11. Data and model governance notes

- Governance verdict (13 gates each, `outputs/tables/ai_governance_go_no_go.csv`): **GO**
  for academic demonstration, **PARTIAL** for a controlled internal pilot, **NO-GO** for
  external customer-facing deployment. This is not a soft "in progress" -- treat the
  external-use NO-GO as binding until the documented Phase 1-2 gaps (data-governance
  sign-off, DPIA, media-rights legal clearance, independent labour/ESG evidence) are closed.
- 26 risks are catalogued (`ai_governance_risk_register.csv`); 37 of 52 controls are already
  implemented, 12 proposed, 3 partially implemented.
- No engagement-prediction model, no predicted-engagement field, exists anywhere in this
  repository (verified by `tests/test_governance_framework.py` and the RAG go/no-go gates).
- No face detection or protected-characteristic inference exists in the image pipeline
  (test-enforced).
- Usernames are stripped from review data before saving to `data/raw/`.
- `.env` is excluded from version control; no synthesis/governance build script reads it
  (test-enforced).
