# Project Charter — TALA Multimodal AI Strategy

**Course:** SPJIMR ANA526-PPM | **Type:** 3-day MBA group project sprint, extended with a
Day 2.5 modality-realignment task. This charter is the binding reference for scope — if any
other document conflicts with it, this one wins (see `CLAUDE.md` §9 for the condensed,
enforced summary).

---

## 1. Primary Objective

Diagnose divergence between **TALA's official quality/responsibility claims** and **customer
experience**, by integrating:

- official and customer **text** (brand copy, press, Trustpilot/Reddit reviews);
- **product/catalog and permitted UGC images** (official product shots, creator thumbnails
  where publicly and legitimately available);
- **actual video frames, temporal features, and transcripts** where legitimately available
  (not URLs, titles, or engagement statistics standing in for video evidence);
- **enterprise reference documents** (impact/responsibility reports, certifications,
  material specs, sizing/fabric-care guidance, return policies).

The output is a claim-by-claim divergence picture, built from genuine multimodal evidence,
not a single-modality text analysis relabelled as multimodal.

## 2. Secondary Objective

Compare TALA's creator-partnership and official-platform strategy against **Adanola**,
**Girlfriend Collective**, and **Oner Active** — as a **descriptive** cross-brand comparison
of partnership mix, content intent, and platform presence. This is explicitly not a
predictive or causal exercise (see §5).

## 3. Required Modalities and Packages

| Package | Contents | Schema |
|---------|----------|--------|
| **Text Package** | Official claims, customer reviews, creator/social text, reference-document text | `official_claims_corpus`, `customer_experience_corpus`, `creator_strategy_corpus` schemas |
| **Image Package** | Official catalog/product images, permitted UGC/wear images, creator thumbnails, website/app screenshots where relevant | `configs/schema.yaml::image_assets` |
| **Video Package** | Genuinely processed video assets with temporal features, transcripts where available | `configs/schema.yaml::video_assets`, `video_frame_features`, `video_level_features` |
| **Multimodal Reference Package** | Impact/responsibility reports, certifications, material specs, sizing/fabric-care guidance, return policies, brand visual guidance | `configs/schema.yaml::multimodal_reference_assets` |

Each package is built and validated **as its own analytical pipeline** before any fusion is
attempted — see `docs/assignment_alignment_audit.md` for the current PASS/PARTIAL/FAIL
status of each.

## 4. Role of Multimodal Fusion

Fusion is **mandatory**, but its primary purpose in this project is **claim-experience
evidence integration**: for a given official claim, bring together the customer-review
text, product/UGC images, video evidence, and reference documents that bear on whether the
claim holds up — via `data/processed/claim_multimodal_evidence_candidates.csv` and, in a
later task, actual claim-alignment scoring.

Fusion is **not** used for predicting engagement outcomes in this project (see §5). When
auditing or extending fusion code, classify each occurrence:

1. **Remove or rewrite** — if it frames fusion purely as an engagement-prediction mechanism
   (e.g. "early-fuse text+image features to predict engagement_count").
2. **Preserve as historical audit information** — if it's a record of a past decision or
   table (e.g. `day2_modelling_feasibility.csv` rows already marked NO-GO/EXPLORATORY for
   engagement tracks) that documents why that direction was correctly not pursued.
3. **Retain because it refers to primary claim-evidence fusion** — generic fusion utilities
   (`src/fusion_models.py`'s early/late/hybrid concatenation and averaging functions) or any
   code that will serve claim↔evidence integration, regardless of modality.

## 5. Role of Multimodal RAG

A working RAG prototype over the three text corpora, extended with image/video/reference
evidence, is a required final deliverable — retrieval that can answer "what evidence exists
for or against claim X" across modalities. Not started in this task; gated on the Image and
Video packages reaching at least PARTIAL status with genuine (not proxy) coverage.

## 6. Descriptive-Only Engagement Scope

**Engagement prediction is out of scope for the entire project**, not just this task. This
means, permanently:

**Prohibited:**
- engagement regression or classification models;
- out-of-fold (OOF) engagement predictions;
- actual-vs-predicted residual analysis;
- predictive creator-performance scoring;
- engagement-focused early/late/hybrid fusion (i.e. fusing modalities specifically to
  predict `engagement_count`, `interaction_rate_by_views`, or similar).

**Permitted:** the real YouTube engagement metrics already collected (Day 2 Task 2B) may be
used for **descriptive, non-causal** comparison only — medians, interquartile ranges, rates
per brand/partnership-type/content-intent, with sample size and missingness always reported
alongside. See `outputs/tables/descriptive_creator_engagement.csv`.

## 7. Prohibited Overclaims

- Do not present a YouTube video URL, thumbnail, title, description, or engagement
  statistic as a "processed video asset" — it is a **video lead**
  (`rights_or_access_basis=platform_metadata_only`) until genuine temporal processing
  (multiple sampled frames, motion features, or transcript) has actually run on a locally
  held, rights-cleared file.
- Do not describe YouTube-only creator evidence as a balanced cross-platform sample (see
  `outputs/tables/day2_platform_evidence_boundary.csv`).
- Do not call an observed brand difference a "moat" without outcome/performance evidence —
  describe it as an observed distinctive pattern.
- Do not infer protected characteristics (age, gender, ethnicity, attractiveness, health
  status, or similar) from any image, video frame, or transcript, and do not retain face
  embeddings.
- Do not claim multimodal completeness for a claim that genuinely lacks image or video
  evidence — missing modalities are recorded explicitly in
  `claim_multimodal_evidence_candidates.csv::missing_modalities`, never backfilled with a
  fabricated or weakly-matched link merely to make every claim "multimodal."

## 8. Evidence and Platform Limitations

- Reddit JSON API returns 403 (OAuth required since 2023) — no Reddit evidence collected
  directly; press/community evidence comes via Google News RSS and DDG search leads only.
- TALA's official site is Shopify-rendered; some pages require DDG fallback for text
  extraction.
- Instagram and TikTok are bot-blocked for page-text hydration; Instagram identity/media
  resolution relies on already-captured structured metadata only, never a live scrape.
- YouTube Data API v3 (`videos.list`/`channels.list`) is used for real engagement
  statistics; `subscriber_count` from `channels.list` is a **retrieval-time snapshot**, not
  the historical count on each post's publish date.
- No yt-dlp, unofficial downloaders, or stream-extraction tools are used against YouTube,
  Instagram, or TikTok — video processing is restricted to official-page-embedded assets,
  openly licensed video, and locally supplied authorised assets under
  `data/raw/authorised_video_assets/`.
- Creator content in the current dataset is overwhelmingly YouTube-based (60/61 verified
  posts); the single TikTok record does not support platform-level inference.

## 9. Required Final Deliverables

1. Text Package — validated, schema-checked (`official_claims_corpus`,
   `customer_experience_corpus`, `creator_strategy_corpus`).
2. Image Package — validated `image_assets` manifest with real acquired images and
   verified `rights_or_access_basis`.
3. Video Package — validated `video_assets` manifest; genuinely processed subset with
   `video_frame_features`/`video_level_features`; explicit PASS/PARTIAL/FAIL status.
4. Multimodal Reference Package — validated `multimodal_reference_assets` manifest.
5. Per-modality analytical pipelines (text/image/video), each independently validated.
6. Claim-level multimodal evidence candidates
   (`claim_multimodal_evidence_candidates.csv`) — no fabricated links, missing modalities
   explicit.
7. Multimodal fusion for claim-evidence integration (future task, gated on the above).
8. A working multimodal RAG prototype (future task).
9. Governance, ethics, and implementation recommendations (future task, building on
   `docs/governance_notes.md`).

Every deliverable must retain provenance back to a real, dated source or be explicitly
labelled as a weak/automated label requiring human validation.
