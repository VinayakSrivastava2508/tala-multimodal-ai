# Data Dictionary

All schemas are defined in `configs/schema.yaml`. This document provides human-readable guidance for every field and every CSV template.

---

## Common Provenance Fields (present on ALL datasets)

These fields are **required** on every row in every dataset. They form the project's chain of custody.

| Field | Type | Description |
|-------|------|-------------|
| `source_url` | str | URL of the specific piece of content (the review page, post URL, claim page). Not the homepage. |
| `source_platform` | str | Domain or platform name (e.g. `trustpilot.com`, `instagram`, `weartala.com`, `reddit.com`). |
| `collection_date` | date | ISO 8601 (YYYY-MM-DD) — the date the researcher collected this row. |
| `collected_by` | str | Team member name or initials. Enables accountability. |
| `evidence_type` | str | One of: `official`, `customer_experience`, `creator_strategy`, `brand_positioning`, `competitor_benchmark`, `press`, `community`, `assessment`. |
| `notes` | str | Free-text researcher annotation — flag uncertainty, note context, record anomalies. |
| `citation_ready` | bool | `true` when URL, date, text, and source_id have all been verified and are ready for citation. Default `false`. |
| `source_id` | str | FK → `docs/source_log_template.md`. Every row's source must be registered there. |
| `synthetic` | bool | `true` if this is a placeholder/test row, NOT real collected data. Always `false` in final analysis. |

**Evidence type discipline:**
- `official` / `press` / `customer_experience` / `community` → valid for quality and responsibility claims.
- `creator_strategy` / `brand_positioning` → valid for creator strategy analysis ONLY.
- Do NOT mix evidence types when making a claim.

---

## 1. `social_posts_template.csv`

Schema: `social_posts` | Evidence type: `brand_positioning`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `post_id` | str | ✓ | Unique ID (e.g. `ig_tala_20240101_001`) |
| `brand` | str | ✓ | Brand slug from `configs/brands.yaml` |
| `platform` | str | ✓ | `instagram` / `tiktok` / `youtube` |
| `post_date` | date | ✓ | Date of the post (YYYY-MM-DD) |
| `caption` | str | | Post caption text |
| `media_type` | str | | `image` / `video` / `reel` / `story` / `carousel` |
| `hashtags` | str | | Pipe-separated hashtags (e.g. `#tala|#activewear`) |
| `likes` | int | | Like count at collection time |
| `comments` | int | | Comment count at collection time |
| `views` | int | | View count (video/reel only) |
| `image_path` | str | | Relative path to downloaded image if saved locally |
| + common provenance fields | | ✓ | See table above |

**Use for:** brand tone, campaign language, hashtag strategy, content format analysis.
**Do NOT use for:** quality, sustainability, or responsibility claims.

---

## 2. `creator_posts_template.csv`

Schema: `creator_posts` | Evidence type: `creator_strategy`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `post_id` | str | ✓ | Unique ID (e.g. `cr_tala_ig_001`) |
| `brand` | str | ✓ | Brand being mentioned/partnered with |
| `creator_handle` | str | ✓ | Public platform handle (no PII beyond public handle) |
| `platform` | str | ✓ | `instagram` / `tiktok` / `youtube` / `other` |
| `post_date` | date | ✓ | Date of the post |
| `partnership_type` | str | | `paid` / `gifted` / `organic` / `ambassador` / `unknown` — look for `#ad`, `gifted`, `*paid partnership*` signals |
| `caption` | str | | Caption text |
| `creator_followers` | int | | Follower count at collection time |
| `creator_tier` | str | | `nano` (<10K) / `micro` (10K–100K) / `mid` (100K–500K) / `macro` (500K–1M) / `mega` (>1M) |
| `likes` | int | | Like count |
| `comments` | int | | Comment count |
| `views` | int | | View/play count |
| `sentiment_label` | str | | `positive` / `neutral` / `negative` / `mixed` — researcher's read of the caption tone |
| + common provenance fields | | ✓ | See table above |

**Use for:** creator tier mix, platform strategy, partnership diversity, reach analysis.
**Do NOT use for:** quality, sustainability, or product experience claims.

---

## 3. `customer_reviews_template.csv`

Schema: `customer_reviews` | Evidence type: `customer_experience`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `review_id` | str | ✓ | Unique ID (e.g. `rev_tala_tp_001`) |
| `brand` | str | ✓ | Brand slug |
| `platform` | str | ✓ | Source platform (e.g. `trustpilot`, `google`, `sitejabber`, `reddit`) |
| `review_date` | date | ✓ | Date of the review |
| `star_rating` | float | | 1.0–5.0 (leave blank for Reddit/text-only sources) |
| `review_text` | str | ✓ | Full review text (strip username before saving) |
| `topics_mentioned` | str | | Pipe-separated topics (e.g. `quality|fit|delivery`) |
| `verified_purchase` | bool | | `true` if platform confirms verified purchase |
| `sentiment_label` | str | | Researcher or model label |
| `complaint_type` | str | | Primary complaint if negative: `quality` / `fit` / `sizing` / `delivery` / `returns` / `customer_service` / `sustainability` / `other` / `none` |
| + common provenance fields | | ✓ | See table above |

**Use for:** quality, fit, sizing, durability, returns, customer service claims.
**Primary evidence for claim–experience divergence workstream.**

---

## 4. `official_claims_template.csv`

Schema: `official_claims` | Evidence type: `official`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `claim_id` | str | ✓ | Unique ID (e.g. `cl_tala_sus_001`) |
| `brand` | str | ✓ | Brand slug |
| `claim_text` | str | ✓ | Verbatim quote from official source — do not paraphrase |
| `claim_category` | str | ✓ | `quality` / `sustainability` / `inclusivity` / `performance` / `ethics` / `community` / `pricing` / `other` |
| `source_type` | str | ✓ | `website` / `press_release` / `impact_report` / `packaging` / `social_bio` / `campaign_copy` / `interview` / `other` |
| `verifiable` | bool | | `true` if the claim can be checked against third-party evidence |
| `evidence_strength` | str | | Researcher's assessment: `strong` / `moderate` / `weak` / `unverifiable` |
| + common provenance fields | | ✓ | See table above |

**Priority sources for TALA:** `weartala.com/pages/sustainability`, `weartala.com/pages/responsibility`, About page, founder interviews.
**Primary corpus for RAG (official) and claim divergence scoring.**

---

## 5. `competitor_platforms_template.csv`

Schema: `competitor_platforms` | Evidence type: `competitor_benchmark`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `brand` | str | ✓ | Brand slug |
| `platform` | str | ✓ | `instagram` / `tiktok` / `youtube` / `website` |
| `followers` | int | | Follower count at `collection_date` |
| `following` | int | | Following count (useful context for ratio) |
| `avg_post_frequency` | float | | Estimated posts per week |
| `creator_tier_mix` | str | | Description: e.g. `mostly micro (10K–100K), some macro` |
| `content_format_mix` | str | | Description: e.g. `60% reels, 30% carousels, 10% static` |
| `verified_account` | bool | | Whether the account has a platform verification badge |
| + common provenance fields | | ✓ | See table above |

**One row per brand × platform.** Capture date matters — follower counts are point-in-time.
**Brands to cover:** TALA, Adanola, Girlfriend Collective, Oner Active.

---

## 6. `press_reddit_sources_template.csv`

Schema: `press_reddit_sources` | Evidence type: `press` or `community`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_id` | str | ✓ | Unique ID (e.g. `pr_tala_001`) — also used as FK in other datasets |
| `brand` | str | ✓ | Brand being discussed |
| `source_type` | str | ✓ | `press` / `reddit` / `blog` / `forum` / `news` / `podcast` / `other` |
| `publication` | str | ✓ | Outlet name (e.g. `Vogue UK`, `r/femalefashionadvice`, `Good on You`) |
| `title` | str | ✓ | Article or thread title |
| `author` | str | | Author name if available |
| `date_published` | date | | Publication date |
| `summary` | str | | 1–3 sentence researcher summary of the content |
| `key_quote` | str | | Most relevant verbatim quote |
| `topics_mentioned` | str | | Pipe-separated topics |
| `sentiment_label` | str | | Overall tone of the piece |
| + common provenance fields | | ✓ | See table above |

**Use for:** quality/responsibility claim validation, third-party sustainability assessments, customer voice aggregation.

---

## 7. `claim_assessments_template.csv`

Schema: `claim_assessments` | Evidence type: `assessment`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `assessment_id` | str | ✓ | Unique ID (e.g. `asmnt_001`) |
| `claim_id` | str | ✓ | FK → `official_claims_template.csv` |
| `brand` | str | ✓ | Brand slug |
| `claim_category` | str | ✓ | Inherited from the claim |
| `claim_text` | str | ✓ | Claim text (denormalised for readability) |
| `supporting_evidence` | str | | Passage(s) that support the claim |
| `contradicting_evidence` | str | | Passage(s) that contradict the claim |
| `supporting_source_ids` | str | | Pipe-separated source_ids of supporting evidence |
| `contradicting_source_ids` | str | | Pipe-separated source_ids of contradicting evidence |
| `divergence_score` | float | | 0.0 (fully supported) → 1.0 (fully contradicted) |
| `assessor` | str | ✓ | `human` / `model` / `hybrid` |
| `confidence` | str | | `high` / `medium` / `low` |
| + common provenance fields | | ✓ | See table above |

**Produced in Notebook 03.** Fill manually first; model-assisted scoring in Day 2.

---

## Day 2 Schemas (hydration + RAG corpora)

Produced by `scripts/hydrate_day2_sources.py` (and its recovery counterpart
`scripts/recover_day2_gates.py`). These are **distinct schemas** from the Day 1
candidate schemas above, even where filenames share a substring (e.g.
`official_claims_corpus.csv` vs. `official_claims_template.csv`) — see
[Schema Routing](#schema-routing-day-1-vs-day-2) below for why that distinction
matters and how the validator now enforces it. Defined in `configs/schema.yaml`
under the "Day 2 schemas" section.

### 8. `data/corpora/official_claims_corpus.csv`, `customer_experience_corpus.csv`, `creator_strategy_corpus.csv`

Schemas: `official_claims_corpus`, `customer_experience_corpus`, `creator_strategy_corpus`

One row = one RAG-ready document. All three share the same column layout:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `document_id` | str | ✓ | Unique within the corpus (e.g. `oc_0001`, `ce_rev_0001`, `cs_0001`) |
| `corpus` | str | ✓ | Must equal the corpus's own name (`official_claims` / `customer_experience` / `creator_strategy`) |
| `brand` | str | ✓ | Brand this document is evidence for |
| `source_platform` | str | ✓ | Domain/platform the text was hydrated from |
| `source_url` | str | ✓ | Must be `http(s)://` — the exact page/post hydrated |
| `document_title` | str | | Page title if the source had one; frequently blank for review snippets and DDG-sourced creator posts |
| `publication_date` | str | | If extractable from page metadata |
| `extracted_text` | str | | The hydrated/extracted text. Required to be non-empty **only when `rag_usable=True`** — a weak, non-rag-usable row may legitimately have empty text |
| `evidence_strength` | str | ✓ | One of `strong` / `medium` / `weak` / `unusable` |
| `rag_usable` | bool | ✓ | `true` only for `strong`/`medium` evidence — a `weak`/`unusable` row with `rag_usable=true` is a validation failure |
| `retrieved_at` | str | ✓ | ISO timestamp of hydration |
| `provenance_note` | str | ✓ | Free text tracing where the row's evidence_strength came from |

**`official_claims_corpus` specific:** `evidence_strength` here reflects the
**source page's** hydration quality (via `data/interim/day2/hydrated_claims.csv`,
joined on `source_id`), not the character length of the individual extracted claim
sentence — a short but genuine sentence lifted from a strongly-hydrated brand page is
strong evidence. Rows marked `strong` must carry `strength_source=source_page` in
`provenance_note`; this is enforced so a future change can't silently revert to
sentence-length scoring (where a strong page's five-word claim would wrongly score
`weak`/`unusable`) without anyone noticing.

**Fields the original Day 2 spec mentioned but that are NOT present as columns in
these files** (`claim_text`, `claim_category`, `numerical_claim`, `unit`,
`target_year`, `certification`, `material`, `geography`, `source_section`,
`source_quote`): these live one step upstream, in
`data/interim/day2/claims_expanded.csv`, not in the corpus export. They are
deliberately left out of the `*_corpus` schemas rather than declared as "optional"
columns that don't exist — a schema field that's absent as a column always fails
validation in this project's `validate()`, regardless of its `required` flag, so
declaring a genuinely-absent column would just reintroduce a false positive.

### 9. `data/interim/day2/creator_posts_enriched.csv`

Schema: `creator_posts_enriched`

Day 1 `creator_posts` plus YouTube-API/DDG-discovered candidates from
`enrich_creator_posts()`, repaired by Day 2 Task 1B
(`scripts/repair_day2_creator_evidence.py`, see
`docs/day2_evidence_hydration_audit.md` §7) before corpus-building filters down to
verified posts only. A pre-repair snapshot is kept at
`data/interim/day2/archive/creator_posts_enriched_pre_repair.csv` (excluded from
validator scanning — see [Schema Routing](#schema-routing-day-1-vs-day-2)).

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `brand` | str | ✓ | |
| `platform` | str | ✓ | |
| `source_url` | str | ✓ | Always populated |
| `post_url` | str | | Populated for newly-enriched rows only; pre-existing Day 1 rows carry the same URL in `source_url` instead |
| `direct_post_url` / `direct_post_id` | str | | Set only when `url_validation_status=valid`: the canonical post URL and its platform-native video ID / shortcode |
| `url_class` | str | | `youtube_video` / `youtube_short` / `tiktok_video` / `instagram_p` / `instagram_reel` / `instagram_tv`, or an invalid/lead class (`instagram_programme_or_campaign`, `youtube_search`, `youtube_channel`, `search_result`, ...) |
| `url_validation_status` | str | ✓ | `valid` / `invalid` / `lead_only` |
| `invalidation_reason` | str | | Set whenever `usable_as_creator_post=False`: e.g. `instagram_programme_or_campaign_page`, `creator_identity_unresolved`, `brand_link_unverified` |
| `creator_handle` | str | | **Semantic rule:** required non-empty whenever `usable_as_creator_post=True` |
| `creator_identity_source` | str | | `youtube_oembed` / `tiktok_url` / `existing_channel_metadata` / `existing_structured_metadata` / `page_metadata_og_title` / `none` |
| `creator_identity_confidence` | str | ✓ | `high` / `medium` / `low` / `unresolved`; a row may be `usable_as_creator_post=True` only when this is `high` or `medium` |
| `creator_identity_evidence` | str | | Free text: what was actually matched (e.g. oEmbed `author_name`) |
| `brand_link_verified` | bool | ✓ | **Semantic rule:** required `True` whenever `usable_as_creator_post=True`. Never derived from the inherited `brand` column alone — only from `video_title`/`caption_or_description`/hashtag/personalised-code evidence |
| `brand_link_evidence` / `brand_link_source` / `brand_link_confidence` | str | | What matched, which field it came from, and `high`/`medium`/`low`/`unresolved` |
| `partnership_type` | str | | `unknown` / `unclear` / `organic` / `gifted` / `paid_sponsorship` / `ambassador` / `affiliate` / `founder_or_employee` — `paid_sponsorship`/`gifted`/`ambassador`/`affiliate`/`founder_or_employee` all require an explicit disclosure marker, never inferred from a brand mention alone |
| `partnership_evidence` / `partnership_confidence` / `partnership_inferred` | | | What disclosure text matched, confidence, and whether the label was automatically inferred |
| `usable_as_creator_post` | bool | ✓ | |
| `evidence_strength` | str | ✓ | `strong` / `medium` / `weak` / `unusable` — see Part E rules in the audit doc; only `strong`/`medium` may be `usable_as_creator_post=True` |
| `source_record_ids` | str | | Pipe-separated `source_id`s of rows merged into this one during deduplication on `(platform, direct_post_id)` |

**Semantic rules (verified creator posts, i.e. `usable_as_creator_post=True`):**
- Must resolve, via `post_url` falling back to `source_url`, to a **direct** post/video
  URL — `instagram.com/p/...` or `/reel/...`, `tiktok.com/@handle/video/<id>`,
  `youtube.com/watch?v=...`, `/shorts/...`, or `youtu.be/...`. A profile page,
  campaign/ambassador-program page, or DDG/Google search-result URL never qualifies.
- `creator_handle` must be non-empty.
- `evidence_strength` must be present.
- `brand_link_verified` must be `True`.
- No `(platform, direct_post_id)` pair may repeat among usable rows.
- Missing follower/like/view/comment counts never fail validation — those columns
  aren't part of this schema's required set at all.

### 10. `data/interim/day2/platform_strategy_enriched.csv`

Schema: `platform_strategy_enriched`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `brand` | str | ✓ | |
| `platform` | str | ✓ | |
| `evidence_url` | str | ✓ | Hydrated platform URL |
| `evidence_strength` | str | ✓ | `strong` / `medium` / `weak` / `unusable` |

### 11. `data/interim/day2/hydrated_sources.csv`

Schema: `hydrated_sources`

Rolled-up hydration summary across press, claims, reviews, and platform sources.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source_type` | str | ✓ | `press` / `claims` / `reviews` / `platforms` |
| `source_id` | str | ✓ | |
| `brand` | str | ✓ | |
| `source_url` | str | ✓ | |
| `hydration_status` | str | ✓ | `success` / `cached` / `failed` / `skipped` / `blocked` |
| `evidence_strength` | str | ✓ | `strong` / `medium` / `weak` / `unusable` |
| `rag_usable` | bool | ✓ | |
| `text_length` | int | ✓ | |
| `retrieved_at` | str | ✓ | |

Note: this rolled-up file does **not** carry `final_url`, `hydrated_text`, or
`hydration_timestamp_utc` — those exist per-source (with an `h_` prefix) in the
individual `hydrated_press.csv` / `hydrated_claims.csv` / `reviews_labeled.csv` /
`platform_strategy_enriched.csv` files, not in this combined summary. The schema
only declares columns this file actually has.

## Schema Routing: Day 1 vs. Day 2

Several Day 2 filenames share a substring with a Day 1 schema name — e.g.
`official_claims_corpus.csv` starts with `official_claims`, and
`creator_posts_enriched.csv` starts with `creator_posts`. Until 2026-09-18, the
validator (`src/validation.py::guess_schema`) inferred schemas purely by filename
**prefix**, so both of those Day 2 files were silently routed to the Day 1 schemas
and failed validation against a column layout they were never meant to have
(`claim_id`, `collection_date`, `citation_ready`, etc.) — a routing false positive,
not a real data problem.

This is now fixed with a two-tier lookup in `guess_schema()`:
1. **Exact filename-stem match** (`EXACT_FILENAME_SCHEMA`) — checked first, covers
   all six Day 2 files above.
2. **Legacy prefix match** (`FILENAME_TO_SCHEMA`) — only consulted when no exact
   match is found; this is unchanged and still resolves all seven Day 1 schemas.

See `docs/day2_evidence_hydration_audit.md` for the before/after validation results.

## Naming Conventions

- **IDs:** `{type_prefix}_{brand_slug}_{platform_slug}_{sequence}` — e.g. `rev_tala_tp_001`
- **Dates:** ISO 8601 `YYYY-MM-DD` in all date fields.
- **List fields:** Pipe-separated in CSV — `quality|fit|delivery`
- **Filenames:** `{schema_name}_{brand_slug}.csv` or `{schema_name}.csv` for multi-brand files.
- **Templates:** Stored in `data/raw/templates/` — copy to `data/raw/` and rename before filling.
