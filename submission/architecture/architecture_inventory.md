# TALA Multimodal AI -- Architecture Inventory

This document describes the **actual implemented architecture** of this repository as of the
current working-tree state, not a proposed or aspirational design. Every component below
names the real module(s) that implement it. No secrets, local absolute paths, or
environment-variable *values* appear anywhere in this document -- only variable *names*
(e.g. `GEMINI_API_KEY`) where relevant to describe a control.

---

## 1. Data sources

- **Business purpose:** Ground every downstream claim, evidence unit, and image/video/reference asset in a real, provenance-tracked public source -- the No-Fabrication Rule at the top of `CLAUDE.md`.
- **Implementation:** `src/collectors/` (official.py, official_media.py, reviews.py, gdelt_press.py, reddit.py, youtube_creator.py, youtube_metrics.py, product_pages.py, reference_documents.py, creator_identity.py, search.py, api_clients.py, apify_client.py, hydration.py, media.py, utils.py).
- **Input:** Public TALA/competitor URLs, Google News RSS, DuckDuckGo/Google search, YouTube Data API, Trustpilot (via Apify free tier), Reddit/press.
- **Output:** `data/raw/day1/auto/*_candidates.csv` (28-127 rows per source type, per `docs/day1_evidence_audit.md`).
- **Model or method:** Gentle HTTP requests with `time.sleep()` rate-limiting; no login-wall/CAPTCHA/paywall bypass (Data Ethics Rules, `CLAUDE.md` §5).
- **Key limitation:** Apify free-tier limit and TALA Shopify JS rendering blocked some collection paths in this sprint (see `docs/automated_collection_notes.md`); collection failures are logged, not silently dropped (`collection_failures.csv`, 28 rows).

## 2. Collection and hydration

- **Business purpose:** Turn raw per-source CSVs into a single, schema-validated, de-duplicated dataset per entity type.
- **Implementation:** `src/collectors/hydration.py`; orchestrated by `scripts/hydrate_day2_sources.py`, `scripts/clean_day1_data.py`.
- **Input:** `data/raw/day1/auto/*_candidates.csv`.
- **Output:** `data/interim/day1/*_cleaned.csv` -> `data/interim/day1_5/*_patched.csv` -> `data/interim/day2/hydrated_*.csv`.
- **Model or method:** Deterministic pandas joins/dedup keyed on source URL and claim/product identifiers; strips usernames from review data before saving (Data Ethics Rules).
- **Key limitation:** Hydration enriches but does not invent fields; unresolved fields remain null rather than imputed.

## 3. Text processing

- **Business purpose:** Turn official-claim, customer-review, press, and creator text into features usable for relevance/stance scoring and RAG embedding.
- **Implementation:** `src/text_features.py`, `src/fusion/text_evidence.py`.
- **Input:** `data/corpora/official_claims_corpus.csv` (37 rows), `customer_experience_corpus.csv` (95 rows), `creator_strategy_corpus.csv` (61 rows).
- **Output:** `data/processed/customer_text_features.csv`, `official_claim_features.csv`; text evidence units in `data/processed/fusion/claim_evidence_units.csv` (93 of 119 rows are `modality=text`).
- **Model or method:** Sentence embeddings (`all-MiniLM-L6-v2`, 384-dim, via `sentence-transformers`), TF-IDF, sentiment, and rule/NLI-based stance classification (`src/fusion/nli_diagnostics.py`).
- **Key limitation:** Stance/relevance scoring is automated (NLI + rule-based), not human-adjudicated -- flagged throughout as a limitation in every downstream table's `caveat` field.

## 4. Image processing

- **Business purpose:** Extract genuine, interpretable features from official product/catalog images and permitted UGC without inferring protected characteristics.
- **Implementation:** `src/image_features.py`.
- **Input:** Official product images referenced in `data/interim/day2_5/image_assets.csv` (24 rows).
- **Output:** Dominant-colour palette (K-Means, `n_colors` configurable), brightness/contrast/saturation/edge-density, weak labels (`studio_vs_lifestyle`, `person_present`, `screenshot_or_meme_indicator`); CLIP ViT-B/32 embeddings (512-dim) for cross-modal retrieval.
- **Model or method:** `sklearn.cluster.KMeans` (fixed `random_state=42`) for palette; Hugging Face CLIP (`openai/clip-vit-base-patch32`) for embeddings.
- **Key limitation:** No face detection or protected-characteristic inference is implemented (explicit design constraint, verified by `tests/test_image_features.py::test_extract_interpretable_features_no_protected_characteristic_fields`). Only 3 of 37 claims currently have any linked image evidence.

## 5. Video/frame processing

- **Business purpose:** Provide genuine temporal video evidence (not just a URL/thumbnail) for visually-groundable claims, per the binding video-processing definition in `CLAUDE.md` §9.
- **Implementation:** `src/video_features.py` (frame sampling via `cv2.VideoCapture`, motion/scene-change proxy features, CLIP-based product-semantic-similarity per frame); `src/fusion/video_evidence.py` (claim linkage).
- **Input:** Locally downloadable TALA-owned videos with `rights_or_access_basis` in {`official_direct_public_asset`, `open_license`, `group_owned`, `user_authorised`} -- 8 of 61 total video leads met this bar (`outputs/tables/video_asset_coverage.csv`).
- **Output:** `data/processed/video_frame_features.csv`, `video_level_features.csv`; 52 sampled frames across the 8 processed videos (mean 6.5 frames/video), 8/8 with temporal features, 0/8 with a transcript.
- **Model or method:** OpenCV frame extraction at sampled timestamps + CLIP frame-level embeddings; scene-change is an explicitly documented *proxy*, not a validated shot-boundary detector.
- **Key limitation:** No transcript/audio evidence trail exists for any video yet (0% coverage); competitor brands have zero genuinely processed videos -- all 53 remaining leads are `platform_metadata_only` and must never be counted as processed video coverage.

## 6. Reference-document processing

- **Business purpose:** Extract structured evidence from TALA's impact/responsibility reports, certifications, sizing/fabric-care guidance, and return policies -- the Multimodal Reference Package required by the assignment.
- **Implementation:** `src/reference_features.py`, `src/collectors/reference_documents.py`, `src/fusion/reference_evidence.py`; built via `scripts/run_day2_6a_reference_package.py`.
- **Input:** `data/interim/day2_6/tala_product_reference_extracts.csv` (94 rows), `multimodal_reference_assets.csv` (162 rows).
- **Output:** `data/processed/reference_document_chunks.csv`, `reference_tables.csv`, `reference_images.csv`; 16 of 119 evidence units are `source_type=official_reference`.
- **Model or method:** Structured text-chunk extraction with `source_url`/`access_date`/`data_type` provenance columns (Citation and Provenance Rules, `CLAUDE.md` §3).
- **Key limitation:** Reference documents are brand-published; they corroborate claim wording, not independent third-party validation.

## 7. Claim-evidence matching

- **Business purpose:** Link each of the 37 official claims to every evidence unit (text/image/video/reference) that is actually relevant to it.
- **Implementation:** `src/fusion/evidence_fusion.py`, `src/fusion/image_evidence.py`, `src/fusion/text_evidence.py`, `src/fusion/video_evidence.py`, `src/fusion/reference_evidence.py`; `data/processed/claim_visual_evidence_matches.csv`, `claim_reference_matches.csv`, `claim_multimodal_evidence_candidates.csv`.
- **Input:** Claims corpus + all four evidence corpora.
- **Output:** `data/processed/fusion/claim_evidence_units.csv` -- 119 claim-evidence links with `relevance_score`, `relevance_method`, `stance`, `stance_score`, `reliability_score`, `visual_gate_passed`.
- **Model or method:** Relevance scoring (embedding similarity + rule gates), visual-groundability gating so image/video evidence is only ever attached to claim categories authoritatively defined as visually groundable (`configs/fusion_rules.yaml`).
- **Key limitation:** Matching is corpus-bound -- a claim with zero matched evidence units genuinely has zero collected evidence, not zero real-world evidence.

## 8. Fusion engine

- **Business purpose:** Combine each claim's matched evidence units into a single, auditable production label (`aligned` / `mixed` / `insufficient_evidence` / `divergent` / `partially_aligned`) -- the mechanism for **claim <-> evidence integration**, redirected away from engagement prediction per `CLAUDE.md` §9's fusion-scope note.
- **Implementation:** `src/fusion_models.py` (early/late/hybrid fusion utilities), `src/fusion/automated_evaluation.py`, `src/fusion/presentation_governance.py` (maps label + confidence to `presentation_restriction`).
- **Input:** `data/processed/fusion/claim_evidence_units.csv`.
- **Output:** `data/processed/fusion/claim_fusion_results.csv` -> `outputs/tables/analytical_synthesis_claim_master.csv` (37 rows: 14 aligned, 5 mixed, 18 insufficient_evidence, 0 divergent, 0 partially_aligned).
- **Model or method:** Deterministic rule/NLI-based aggregation of supporting vs. challenging evidence, with a fixed `RANDOM_SEED = 42` for any stochastic component.
- **Key limitation:** The correctness of the label itself was never independently (human) validated; only its *stability* under alternative configurations was tested (see §9).

## 9. Sensitivity and NLI diagnostics

- **Business purpose:** Test whether each claim's production label is robust to reasonable changes in fusion weights and evidence-modality inclusion, before that label is presented externally.
- **Implementation:** `src/fusion/nli_diagnostics.py`; `scripts/run_fusion_sensitivity.py`, `scripts/run_nli_diagnostics.py`.
- **Input:** `claim_fusion_results.csv` re-run under 4 configurations (text_only / text_image / text_image_video / text_image_video_reference).
- **Output:** `outputs/tables/fusion_sensitivity_stability.csv`, `fusion_unstable_claims.csv`, `modality_contribution_summary.csv`, `nli_*` tables. Label distribution is identical across all 4 configurations (0 claims changed label); mean confidence rises from 0.5378 (text-only) to 0.5676 (full bundle).
- **Model or method:** Re-running the deterministic fusion rule under each configuration and diffing outputs; 15 of 37 claims flagged `sensitive_to_weights=True`.
- **Key limitation:** Stability across *these* 4 configurations does not guarantee stability under configurations not tested.

## 10. RAG corpus builder

- **Business purpose:** Turn the claims, evidence, and reference tables into flat records ready for vector indexing, with explicit provenance (`source_origin`, `independence_status`) attached to every record.
- **Implementation:** `src/rag/corpus_builder.py` (`build_claims_records`, `build_text_evidence_records`, `build_visual_evidence_records`).
- **Input:** `claim_fusion_results.csv`, `claim_evidence_units.csv`, reference/video/image tables.
- **Output:** In-memory records upserted into ChromaDB by `scripts/build_chroma_index.py`.
- **Model or method:** Deterministic record construction; `classify_evidence_provenance()` (in `src/rag/schemas.py`) explicitly separates brand-official evidence (`Self-reported`) from customer/press/creator evidence (`Not independently verified`).
- **Key limitation:** A record's provenance classification is only as good as the `source_type`/`self_reported`/`independent_source` fields set upstream in corpus building.

## 11. Embedding models and dimensions

- **Business purpose:** Represent text and visual evidence in vector space for semantic retrieval.
- **Implementation:** `src/rag/embedders.py`.
- **Text:** `all-MiniLM-L6-v2` (sentence-transformers), 384-dim, cosine-normalised.
- **Visual:** CLIP ViT-B/32 (`openai/clip-vit-base-patch32`), 512-dim, disk-cached by content hash; a CLIP *text* encoder embeds natural-language queries into the same space for cross-modal retrieval.
- **Key limitation:** Both are general-purpose, free Hugging Face models (no paid API per `CLAUDE.md` §4), not fashion/apparel-domain-fine-tuned.

## 12. Chroma collections and current record counts

- **Business purpose:** Persistent, queryable vector store for the RAG prototype.
- **Implementation:** `src/rag/chroma_store.py`; persisted at `data/vector_db/chroma` (local, non-networked).
- **Collections (per `outputs/tables/rag_collection_summary.csv`):**
  - `tala_claims`: 37 records, 0 duplicates.
  - `tala_text_evidence`: 662 records, 0 duplicates.
  - `tala_visual_evidence`: 76 records, 0 duplicates, 0 missing-asset rejections.
- **Key limitation:** Index size reflects what was collected, not a completeness guarantee across every possible TALA claim or piece of evidence.

## 13. Query router

- **Business purpose:** Deterministically decide which modalities (text/image/video/reference) to search for a given claim or open question -- no LLM makes this routing decision.
- **Implementation:** `src/rag/query_router.py` (`classify_intents`, `route_modalities_for_intents`, `route_modalities_for_claim`).
- **Input:** A free-text question (keyword-matched against 11 intents) or a `claim_category` (matched against `configs/fusion_rules.yaml::visual_groundability`).
- **Output:** A list of collections to search.
- **Model or method:** Plain keyword/category lookup tables, unit-testable directly.
- **Key limitation:** Keyword-based intent classification can miss paraphrased questions that don't contain any listed keyword (falls back conservatively to `claim_validation`).

## 14. Retrieval

- **Business purpose:** Fetch the top-k candidate evidence items per collection for a query.
- **Implementation:** `src/rag/retriever.py` (`retrieve_text_evidence`, `retrieve_claims`, `retrieve_visual_evidence`, `direct_linked_evidence`).
- **Model or method:** Chroma similarity search with optional metadata filters (brand/modality/source_type/evidence_strength).
- **Evaluated performance:** Direct-link text-retrieval recall@10 = 0.7619 (21 claims with a known-linked evidence unit), MRR = 0.1986 (`outputs/tables/rag_retrieval_evaluation.csv`).
- **Key limitation:** Evaluated only on the subset of claims that have at least one linked text/reference evidence unit (21 of 37); not all claims are represented in this metric.

## 15. Rank fusion

- **Business purpose:** Combine multiple retrieval result lists (e.g. text + reference) into one final ranked, deduplicated candidate list with a transparent trace.
- **Implementation:** `src/rag/rank_fusion.py` (`fuse_candidates`).
- **Model or method:** Deterministic reciprocal-rank fusion (RRF), weighted by evidence-strength and source-independence scores; every ranked-out item's `exclusion_reason` is logged in the trace (visible in the app's "Show retrieval trace" toggle).
- **Key limitation:** RRF weighting choices (evidence-strength, independence) are management judgements, documented but not independently validated against a ground-truth ranking.

## 16. Gemini generation

- **Business purpose:** Produce a grounded, evidence-cited natural-language explanation for a claim or open question.
- **Implementation:** `src/rag/gemini_generator.py`.
- **Input:** Fused, ranked evidence candidates + the claim/question.
- **Output:** `concise_answer`, `claim_assessment`, `confidence_explanation`, `caveats`, `evidence_gaps` (JSON-structured).
- **Model or method:** Google Gemini (model id read from the `GEMINI_MODEL` environment variable, e.g. `gemini-3.1-flash-lite` in the last recorded run; API key read from `GEMINI_API_KEY`, never hardcoded). **No fallback or local-LLM backup generator exists anywhere in `src/rag/*`** -- if Gemini is unavailable, generation fails closed rather than silently substituting a different model.
- **Key limitation:** Live calls are billed/rate-limited (free-tier quota); this submission deliberately made zero new Gemini calls (see Part F/screenshot manifest and `rag_demo_cases.csv`, whose 3 cached demo generations are dated 2026-09-19 and are reused, not regenerated, for this handoff).

## 17. Citation validation

- **Business purpose:** Mechanically verify that every Gemini-generated answer only cites evidence that was actually retrieved, that cited assets resolve to real files, that the true production fusion label is preserved (never silently changed to a simplified true/false), and that self-reported evidence is never mislabelled as independent.
- **Implementation:** `src/rag/citation_validator.py` (`validate_response`).
- **Checks performed (all 5, per `outputs/tables/rag_citation_validation.csv`):** `citations_exist_in_context`, `referenced_assets_resolve`, `production_label_preserved`, `self_reported_not_called_independent`, `no_overclaimed_inspection`.
- **Result:** 3/3 documented demo cases passed all 5 checks.
- **Key limitation:** Validates 3 fixed, documented demo cases -- not a continuously monitored production query stream (flagged as a Phase 2 governance gap in `ai_governance_go_no_go.csv`).

## 18. Streamlit application

- **Business purpose:** The "TALA Multimodal AI Decision Cockpit" -- the executive-facing surface over all of the above.
- **Implementation:** `app/streamlit_app.py` (router) + `app/views/` (`executive_overview.py`, `claim_diagnostic.py`, `creator_strategy.py`, `rag_explorer.py`, `governance.py`, `methodology.py`) + `app/components/` (charts, evidence_cards, navigation, metric_cards, limitations) + `app/data_loader.py`, `app/state.py`, `app/styling.py`.
- **Verified state:** 40/40 PASS on the Acceptance-Patch feature test matrix (`outputs/app_audit/feature_test_matrix.csv`, mechanically enforced by `tests/test_audit_artifacts.py`); 6 pages, all navigable.
- **Key design property:** The RAG page's model/Chroma imports are lazy (inside `render()`), so simply opening the page never triggers Gemini or embedding-model initialisation -- generation only happens on an explicit "Investigate"/"Submit" click.
- **Key limitation:** A dedicated smaller-laptop responsive-viewport pass and a systematic keyboard-navigation accessibility pass have not been performed (open item IMP-106, P3).

## 19. Governance controls

- **Business purpose:** A first-class deliverable (not an afterthought) documenting risk, controls, and deployment readiness.
- **Implementation:** `scripts/build_governance_framework.py` (tables) + `scripts/build_governance_figures.py` (figures); `app/views/governance.py`.
- **Output tables:** `ai_governance_risk_register.csv` (26 risks), `ai_governance_control_matrix.csv` (52 controls: 37 implemented, 12 proposed, 3 partially implemented), `claim_decision_policy.csv`, `ai_governance_raci.csv`, `esg_claim_assurance_matrix.csv`, `ai_governance_go_no_go.csv` (13 gates x 3 deployment contexts).
- **Verdict:** 13/13 gates GO for academic demonstration; 13/13 PARTIAL for a controlled internal pilot; 13/13 NO-GO for external customer-facing deployment.
- **Key limitation:** This is a documented self-assessment produced within an academic sprint, not an independent or certified audit.

## 20. Privacy and security guardrails

- **Business purpose:** Keep personal data and secrets out of the pipeline and the repository.
- **Implementation (enforced, not just documented):**
  - Usernames stripped from review data before saving to `data/raw/` (Data Ethics Rule).
  - No face detection or protected-characteristic inference anywhere in `src/image_features.py` (test-enforced).
  - `.env` (holding `GEMINI_API_KEY`, `GEMINI_MODEL`) is excluded via `.gitignore` and never read by any synthesis/governance build script (test-enforced: `tests/test_analytical_synthesis.py::test_env_not_read_or_modified_by_synthesis_scripts`, `tests/test_governance_framework.py::test_no_env_access_in_governance_build_scripts`).
  - No external API call (`requests.*`, `genai.Client`, `openai.*`, `urlopen`, `httpx.*`) appears in any build/synthesis/governance script other than the single, explicit Gemini generator path (test-enforced).
  - Local, non-networked ChromaDB; no cloud vector-store dependency.
- **Key limitation:** No DPIA, no formal DPO review, and the Gemini API data-use tier (free vs. paid) is not yet formally confirmed -- all flagged as Phase 1 gaps in the governance go/no-go table, not silently assumed resolved.
