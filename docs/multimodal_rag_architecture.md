# Multimodal RAG Architecture (Day 3B)

## Business question

Does multimodal evidence reveal divergence between TALA's official quality/responsibility
claims and externally observable customer experience? This RAG system exists to let a
person **investigate that question interactively**, grounded in real, provenance-tracked
evidence -- it is not a general-purpose TALA chatbot, and it does not predict engagement,
score creators, or infer anything about people from images.

## Why RAG is needed

Day 3A's automated fusion (`data/processed/fusion/claim_fusion_results.csv`) already produces
a label per claim (aligned / partially_aligned / mixed / divergent / insufficient_evidence).
What it does not give a human reader is an **interactive, evidence-grounded explanation** of
*why* a claim landed on that label, with the actual passages, images, and video frames that
drove it, filterable by brand/modality/source type, and answerable for open-ended analytical
questions that don't map to a single claim. RAG (retrieval-augmented generation) is the right
tool because it keeps the language model's explanation **tethered to retrieved, cited
evidence** rather than letting it reason freely from parametric memory about a brand it was
never specifically trained on.

## How the RAG supports claim-experience divergence analysis

Every retrieval and generation path is anchored to the Day 3A fusion output:
- **Claim investigation mode** starts from a real claim and its real fusion label, retrieves
  the evidence Day 3A actually linked to it plus semantically related evidence, and asks
  Gemini to *explain* -- never *re-decide* -- that label.
- **Open-question mode** classifies the question into deterministic analytical intents and
  routes retrieval only to genuinely relevant modalities, so a labour claim question can never
  be "supported" by an irrelevant product photo.
- Secondary uses (creator/platform context, competitor comparison) are available but always
  a lower retrieval priority than claim-experience evidence.

## Corpus sources

All three modalities are built exclusively from existing, already-verified Day 1-3A outputs
-- nothing here is freshly scraped or fabricated:

| Source | Rows used | Gate applied |
|---|---|---|
| `data/corpora/official_claims_corpus.csv` | 37 | `rag_usable=True`, evidence_strength in {strong, medium} |
| `data/corpora/customer_experience_corpus.csv` | 95 | same |
| `data/corpora/creator_strategy_corpus.csv` | 61 | same |
| `data/processed/reference_document_chunks.csv` | 461 (353 TALA + 108 competitor) | evidence_strength in {strong, medium} |
| `data/processed/video_level_features.csv` | 8 processed videos | `processing_status=='processed'` only |
| `data/interim/day2_5/image_assets.csv` | 24 | `processing_status=='downloaded'`, local file must exist |
| `data/processed/video_frame_features.csv` | 52 frames | parent video must be `processed`, local file must exist |
| `data/processed/fusion/claim_evidence_units.csv` | 119 | supplies claim-evidence links, stance, source independence |

**Excluded, and documented rather than silently dropped:** `data/processed/reference_images.csv`
(584 rows) has no downloaded local asset for any row (`local_path` is blank throughout) -- since
there is nothing to CLIP-embed, none of these rows are indexed as visual evidence. This is a
real corpus-composition gap inherited from Day 2.6A, not a Day 3B omission.

## Text, image and video processing

- **Text**: `all-MiniLM-L6-v2` (sentence-transformers), the same model used throughout Day 3A
  text fusion, cosine-normalised, 384-dim.
- **Image**: `openai/clip-vit-base-patch32` (Hugging Face Transformers), the same model used
  in Day 2.5 image feature extraction, 512-dim, disk-cached by content hash.
- **Video**: genuinely processed videos only (8 of 69 discovered; the other 61 are
  `platform_metadata_only` leads and are never treated as video evidence). Each processed
  video contributes: (a) individually CLIP-embedded, timestamped, position-labelled sampled
  frames (opening/interval/quarter/half/three_quarter/closing) in `tala_visual_evidence`, and
  (b) a deterministic textual summary of its genuine temporal/motion features (motion
  intensity, scene-change proxy, visual consistency, opening-to-closing change) in
  `tala_text_evidence`. **Sampled frames and a temporal-feature summary are not the same as
  having watched the complete video** -- the Gemini prompt says this explicitly, and the
  citation validator flags any response that claims otherwise.

## Chroma collection design

Local, persistent ChromaDB at `data/vector_db/chroma/` is the **only** vector store. Three
collections, each with `hnsw:space=cosine`:

- **`tala_claims`** (37 rows) -- one row per verified claim, embedded on its own claim text,
  carrying the Day 3A fusion label, confidence, and presentation restriction as metadata.
- **`tala_text_evidence`** (662 rows) -- official claims, customer reviews, press, creator
  content, official reference chunks, and video temporal-feature summaries, each carrying
  claim linkage (semicolon-joined, since one evidence item can support/challenge multiple
  claims), stance, source independence, and evidence strength.
- **`tala_visual_evidence`** (76 rows: 24 product images + 52 video frames) -- CLIP embeddings
  with local asset paths, video role/position/timestamp metadata for frames.

IDs are deterministic (`claim::<id>`, `text::<source_type>::<raw_id>`,
`visual::<modality>::<raw_id>`) so `scripts/build_chroma_index.py` is idempotent: reruns
upsert-by-ID (never duplicate) and remove any row whose ID has left the authoritative corpus.

## Modality routing

Two routers, both deterministic and inspectable (`src/rag/query_router.py`):
- **Claim investigation**: `route_modalities_for_claim(claim_category)` reads the *same*
  `configs/fusion_rules.yaml::visual_groundability.claim_category_groundability` mapping Day
  3A fusion uses. A labour/manufacturing/emissions/circularity/other claim is never routed to
  image or video retrieval, regardless of any similarity score.
- **Open questions**: `classify_intents(question)` is a plain keyword lookup (11 intents);
  `route_modalities_for_intents(intents)` adds image/video only for intents that can ever be
  visually meaningful (quality, fit_sizing, materials, durability_care, claim_validation).
  No LLM call decides what gets searched.

## Retrieval and rank fusion

Each eligible collection is queried independently (`src/rag/retriever.py`), then combined with
reciprocal-rank fusion (`src/rag/rank_fusion.py`, k=60) plus deterministic bonuses for direct
Day 3A claim-evidence links, evidence strength, and source independence. A source-diversity cap
prevents one source type from crowding out the rest. NLI is never used as the relevance/stance
authority here -- Day 3A already established NLI is auxiliary-only and its disagreement with
rule-based stance is a documented domain limitation, not something this pipeline resolves by
fiat. The full trace (original rank, distance, RRF contribution, final rank, exclusion reason)
is retained and surfaced in the Streamlit diagnostic panel.

## Gemini grounding

`google-genai` (official SDK) is the **only** generation path (`src/rag/gemini_generator.py`).
`GEMINI_API_KEY` and `GEMINI_MODEL` are loaded server-side from `.env` via `python-dotenv` and
never logged, printed, or embedded in any output file. The prompt contains only the question,
the selected claim and its existing fusion label (if applicable), the retrieved evidence
(bounded to `max_evidence_items_in_prompt`), and explicit grounding instructions -- never the
full corpus. Retrieved images and video frames are passed as actual image bytes when the
Gemini call supports it. **If the API key/model are missing, or the API call fails for any
reason, generation stops and raises -- there is no local-LLM fallback, no deterministic
template answer, and no offline mode.**

## Citation controls

`src/rag/citation_validator.py` runs after every Gemini response and checks: every cited
evidence ID exists in what was actually retrieved (no hallucinated IDs); every
supporting/challenging/contextual ID is valid; referenced local asset paths resolve and source
URLs are well-formed; the production fusion label was not silently overwritten; self-reported
evidence was never described as independent; and the response never claims to have inspected
evidence beyond what was supplied (e.g. "I watched the video"). A failed validation is shown as
a structured error, not silently patched.

## Evaluation approach

`scripts/run_multimodal_rag_evaluation.py` measures collection integrity, duplicate/missing-
asset rates, metadata completeness, direct-link retrieval recall@k and MRR, modality-routing
accuracy against the authoritative groundability config, retrieval latency, and a small set of
live Gemini demo cases (text+image+video claim, a mixed-label claim, an insufficient-evidence
claim) with citation validation. See `docs/multimodal_rag_evaluation.md` for results.

## Ethics

- No face detection, no protected-characteristic inference from any image.
- Named creators/influencers are treated as public communicators in their professional
  capacity, never as private individuals under personal analysis.
- No manual labelling, no coder adjudication, anywhere in this system.
- Engagement prediction remains explicitly out of scope.

## Limitations

- **ChromaDB runs locally** -- there is no cloud vector store and no backup index.
- **Gemini is the only generator** -- there is no fallback generator and no offline mode.
- Video retrieval uses **processed videos, sampled frames, and temporal features** -- sampled
  frames do not equal complete raw-video understanding, and the prompt/validator both enforce
  this distinction.
- Official evidence **may be self-reported**; the system labels this explicitly rather than
  presenting brand copy as independently verified.
- **Absence of independent evidence does not prove a claim false** -- `insufficient_evidence`
  claims are presented as evidence gaps, never as substantive negative conclusions.
- Creator/platform analysis is explicitly **secondary** to claim-experience divergence.
- **Automated evaluation does not substitute for human semantic judgement** of answer quality
  -- it measures structural correctness (recall, routing accuracy, citation validity), not
  whether an explanation reads well.
- **Human-label evaluation was deliberately excluded from scope** across this entire project,
  consistent with the Day 1-3A charter.

## User interaction workflow

1. Pick a mode in the sidebar (investigate a claim, or ask an open question), optionally
   filtering by brand/modality/source type/evidence strength.
2. **Claim investigation**: select one of the 37 claims. The page shows the claim, its
   existing Day 3A fusion label/confidence/presentation restriction, then a Gemini-generated
   explanation grounded in retrieved evidence split into supports/challenges/context.
3. **Open question**: type a free-text analytical question. The page shows the detected
   intent(s), the modalities actually searched, the generated answer, and evidence cards.
4. Toggling "Show retrieval trace" reveals the full diagnostic panel: collections searched,
   filters applied, candidates retrieved, exclusion reasons, rank-fusion contributions, the
   Gemini model used, and citation-validation status.
