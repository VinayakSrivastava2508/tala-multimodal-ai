# Multimodal RAG Evaluation (Day 3B)

All evaluation in this document is automated, derived from existing structural relationships
already established in Day 1-3A (claim-evidence links, claim categories, modality eligibility,
stance, production fusion labels). **No new manual-label CSV was created. No coder_1/coder_2/
adjudication workflow exists anywhere in this project.** Gemini is never used to grade its own
answers.

## Collection integrity

`outputs/tables/rag_retrieval_evaluation.csv` (metric_group=`collection_integrity`): every
collection (`tala_claims`, `tala_text_evidence`, `tala_visual_evidence`) has
`duplicate_id_rate = 0.0` -- Chroma's upsert-by-deterministic-ID makes duplication structurally
impossible, and this is verified directly against the live collections, not assumed.

## Index idempotency

`scripts/build_chroma_index.py` was run twice in succession. Both runs produced identical final
counts (37 claims / 662 text-evidence / 76 visual-evidence) with `n_removed_stale=0` on the
second run -- confirming upserts replace rather than duplicate, and no record was spuriously
dropped.

## Missing-asset rate

`outputs/tables/rag_retrieval_evaluation.csv` (metric_group=`missing_asset_rate`): 0 rejected
out of the 76 attempted visual-evidence candidates (24 images + 52 video frames). Every local
asset path referenced by an indexed visual-evidence row exists on disk -- verified at build
time, not merely assumed from the source CSV.

## Metadata completeness

`outputs/tables/rag_retrieval_evaluation.csv` (metric_group=`metadata_completeness`): completeness
percentage per required field, per collection (e.g. `claim_id`, `fusion_label`, `evidence_id`,
`modality`, `evidence_strength`, `local_asset_path`).

## Direct-link retrieval recall@k and MRR

For every claim with at least one real Day 3A text/reference evidence link (21 of 37 claims --
the other 16 have zero linked text/reference evidence, mostly `labour`-category claims with no
eligible modality, and are correctly excluded from this specific metric rather than counted as
retrieval failures), the claim's own text was embedded and used to query `tala_text_evidence`
at k=10. **recall@10 = 0.762, MRR = 0.199** (`outputs/tables/rag_retrieval_evaluation.csv`).

The MRR is lower than the recall because several linked evidence items rank outside the top 3
even when they are eventually retrieved within the top 10 -- this is expected: Day 3A's link is
a category/keyword-gated match, not necessarily the single most semantically similar sentence
to the claim's exact wording, so a linked item legitimately competing with several other
relevant-but-unlinked documents does not always rank first.

## Modality-routing accuracy

`outputs/tables/rag_query_routing_evaluation.csv`: **37/37 real claims** routed correctly
against the authoritative `configs/fusion_rules.yaml::visual_groundability.claim_category_groundability`
mapping -- every `materials`/`packaging` claim is routed to image+video, and every
`labour`/`manufacturing`/`emissions`/`circularity`/`other` claim is not, with zero deviation.
This is close to tautological by construction (the router reads the same config the checker
reads), but it is still a genuine regression guard: it fails immediately if the router and the
config ever drift apart.

## Source-diversity coverage

`src/rag/rank_fusion.py`'s source-diversity cap (default: no more than a handful of results
from any one `source_type`) is exercised and unit-tested
(`tests/test_rag_rank_fusion.py::test_source_diversity_cap_excludes_overrepresented_source_type`);
qualitatively, a text-evidence retrieval for a materials/durability question typically surfaces
a mix of `official_claims`, `customer_experience`, `official_reference`, and `creator_strategy`
source types rather than one type dominating.

## Citation-ID validity / unsupported-citation rate / self-reported attribution

`outputs/tables/rag_citation_validation.csv` records, per live demo case, whether every cited
evidence ID exists in the retrieved context, whether self-reported evidence was ever described
as independent, and whether the response overclaimed evidence inspection. During development
(see "Live Gemini generation" below), three real, live-generated responses were validated end
to end and **all three passed every check** (`citations_exist_in_context`,
`referenced_assets_resolve`, `production_label_preserved`, `self_reported_not_called_independent`,
`no_overclaimed_inspection`) -- zero hallucinated citations, zero unsupported-citation instances,
zero self-reported/independent conflation, across all three.

## Production-label preservation

All three live-validated responses (aligned/materials, mixed, insufficient_evidence) preserved
the exact Day 3A `automated_label` -- Gemini explained each classification without asserting a
different one of the five canonical labels. `citation_validator.py`'s
`production_label_preserved` check is a hard automated gate on this, not a self-report.

## Evidence-gap honesty

The `insufficient_evidence` demo case's response (see below) did not convert the absence of
evidence into a negative finding -- it explicitly stated the retrieved evidence was insufficient
to support or challenge the claim, consistent with Day 3A's own `presentation_restriction=
present_as_evidence_gap_finding_only` for this claim.

## Latency

`outputs/tables/rag_latency_summary.csv` separates one-time local model-loading cost (first
call in a fresh Python process: sentence-transformers `all-MiniLM-L6-v2` / CLIP
`openai/clip-vit-base-patch32` weight loading) from steady-state per-query latency:
text retrieval is ~0.03s whether cold or warm (the sentence-transformer model loads fast),
visual retrieval's cold call is ~6-7s (CLIP model load) but the warm call drops to ~0.05-0.07s,
and end-to-end retrieval (no generation) is ~0.1s warm. Gemini generation latency was not
separately isolated in the automated script; interactively, each successful generation call on
`gemini-3.1-flash-lite` completed in a few seconds.

## Demonstration cases

`outputs/tables/rag_demo_cases.csv`:

1. **Text + image + video claim** -- `c_OC_0003_01` (materials, `aligned`), which genuinely has
   text, image, and video evidence units in `claim_evidence_units.csv`.
2. **Mixed-claim explanation** -- `c_OC_0010_00` (`mixed`).
3. **Insufficient-evidence explanation** -- `c_OC_0002_00` (`insufficient_evidence`).

### Live Gemini generation -- development-session verification

During interactive development of this pipeline (this same session, prior to the automated
evaluation script's own run), all three demo cases above were generated **live against the real
Gemini API** on `gemini-3.8-flash` and validated:

- `c_OC_0003_01`: citation validation **passed** (all 5 checks). The response cited
  text, reference-chunk, and image evidence together, explained the `aligned` label without
  overwriting it, and correctly distinguished self-reported reference content from customer/
  press evidence.
- `c_OC_0010_00` (mixed): citation validation **passed** (all 5 checks) after two real code
  fixes (below). The response explained both the supporting and challenging evidence rather
  than forcing a binary verdict, and preserved the `mixed` label.
- `c_OC_0002_00` (insufficient_evidence): citation validation **passed** (all 5 checks) on the
  first attempt.

**The automated evaluation script's own first two runs received `429 RESOURCE_EXHAUSTED` from
`gemini-3.8-flash` on all three attempts** (with retry/backoff on transient errors) -- the
free-tier quota for that specific model had been consumed by the extensive interactive testing
above, in the same development session. Rather than wait for that quota to reset,
`GEMINI_MODEL` in `.env` was switched to **`gemini-3.1-flash-lite`** (a lower-tier model with
separate, unexhausted quota -- confirmed empirically that the 429 was per-model, not
account-wide, before switching). All three demo cases were then re-run live and **succeeded**
(`rag_demo_cases.csv`, `outcome=success` for all three) -- this is what the current
`rag_go_no_go.csv` and `rag_citation_validation.csv` reflect.

**A third real defect found and fixed during this model switch:** on `gemini-3.1-flash-lite`,
several responses cited the raw underlying evidence ID (e.g. `ce_press_0016`, `tala_005`,
`tala_video_104_half_152`, `oc_0001`) instead of the fully-namespaced Chroma ID the prompt
asked for (`text::customer_experience::ce_press_0016`, `visual::image::tala_005`,
`visual::video_frame::tala_video_104_half_152`, `text::official_claims::oc_0001`) -- a lighter
model is less careful about the exact bracket format, but these citations named **genuinely
retrieved evidence**, not fabricated documents. The validator originally treated any ID absent
from the exact context-ID set as hallucinated, which incorrectly failed these responses.
Fixed by resolving a cited raw ID against each evidence item's own `evidence_id`/`asset_id`/
`frame_id`/`video_id`/`document_id` metadata fields before the hallucination check
(`src/rag/citation_validator.py::_build_raw_id_resolver`/`_normalize_cited_ids`) -- a raw-ID
citation that matches nothing actually retrieved (verified separately with a still-fabricated
ID such as `oc_9999`) correctly continues to fail. Covered by regression tests
(`test_raw_id_citation_without_namespace_prefix_is_not_hallucinated`,
`test_raw_id_that_matches_nothing_retrieved_is_still_hallucinated`,
`test_raw_id_self_reported_check_still_catches_violation`). After this fix, all three demo
cases pass citation validation on `gemini-3.1-flash-lite`
(`rag_citation_validation.csv`, `citations_resolve_to_retrieved_evidence = GO, 3/3 passed`).

### Three real defects found and fixed during this verification

1. **Sentence-boundary bleed across response fields**: the citation validator originally joined
   `concise_answer` + `claim_assessment` + `confidence_explanation` + `caveats` with a plain
   string join before sentence-splitting, so a citation at the end of one field could be
   matched against an unrelated "independent" mention that actually started a different field.
   Fixed by validating each structured response field as its own discrete unit
   (`src/rag/citation_validator.py::validate_response`).
2. **Multi-ID brackets silently escaping citation extraction**: Gemini sometimes wrote
   `[id1, id2]` in one bracket instead of `[id1][id2]` as instructed. The original regex only
   matched single-ID brackets, so such citations were invisible to both the hallucination check
   and the self-reported/independent check. Fixed by extending `CITATION_PATTERN` to tolerate
   comma/semicolon-separated IDs within one bracket and splitting them
   (`src/rag/citation_validator.py::extract_cited_ids`).
3. **Raw-ID citations (without the namespace prefix) wrongly flagged as hallucinated** -- see
   above; fixed by `_build_raw_id_resolver`/`_normalize_cited_ids`.

All three fixes are covered by regression tests
(`tests/test_rag_citation_validator.py::test_no_citation_bleed_across_response_fields`,
`::test_extract_cited_ids_handles_multiple_ids_in_one_bracket`,
`::test_self_reported_check_handles_multiple_ids_in_one_bracket`).

## Text/image/video retrieval success

- **Text retrieval**: works (`tests/test_rag_retriever.py::test_text_retrieval_returns_results`,
  and the recall@10=0.762 result above).
- **Image retrieval**: works (`test_visual_retrieval_returns_results`); a materials-claim query
  reliably surfaces `visual::image::*` results.
- **Processed-video retrieval**: works; the same visual query surfaces `visual::video_frame::*`
  results distinguishable from image results via `modality=video_frame`, each carrying its
  video ID, frame position, and timestamp.

## GO/NO-GO

See `outputs/tables/rag_go_no_go.csv` for the full, current gate-by-gate table. As of this
document (`gemini-3.1-flash-lite`): **all 12 gates GO**, 0 PARTIAL, 0 NO-GO.

## Limitations of this evaluation

- Automated evaluation measures **structural correctness** (recall, routing accuracy, citation
  validity, label preservation) -- it does not substitute for a human semantic-quality judgement
  of how well an explanation reads, and human-label evaluation was deliberately excluded from
  this project's scope throughout Day 1-3B.
- The demo-case GO/NO-GO gates depend on live third-party API availability/quota outside this
  codebase's control; a lighter model (`gemini-3.1-flash-lite`) was substituted after
  `gemini-3.8-flash`'s free-tier quota was exhausted by interactive testing in this session, and
  all gates now pass live end to end on that model.
