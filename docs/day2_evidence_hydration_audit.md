# Day 2 Evidence Hydration & Validator Audit — TALA Multimodal AI Strategy
**Date:** 2026-09-18 | **Sprint:** SPJIMR ANA526-PPM | **Day:** 2 of 3

---

## 1. Hydration Pipeline Run

`scripts/hydrate_day2_sources.py` (Parts A–I) hydrated all five Day 1.5 patched files,
extracted official-claim sentences, weak-labeled reviews, enriched creator posts and
platform strategy signals, and built three RAG corpora. Outputs:

| Output | Rows |
|--------|-----:|
| `data/interim/day2/hydrated_press.csv` | 127 |
| `data/interim/day2/hydrated_claims.csv` | 24 |
| `data/interim/day2/claims_expanded.csv` | 159 |
| `data/interim/day2/reviews_labeled.csv` | 56 |
| `data/interim/day2/creator_posts_enriched.csv` | 67 |
| `data/interim/day2/platform_strategy_enriched.csv` | 37 |
| `data/interim/day2/hydrated_sources.csv` | 244 |
| `data/corpora/official_claims_corpus.csv` | 37 |
| `data/corpora/customer_experience_corpus.csv` | 95 |
| `data/corpora/creator_strategy_corpus.csv` | 67 |

### Hydration success by source type

| Source type | Attempted | Success | Failed | Strong | Medium | Weak | Unusable |
|-------------|----------:|--------:|-------:|-------:|-------:|-----:|---------:|
| press_reddit | 127 | 123 | 4 | 34 | 5 | 84 | 4 |
| official_claims | 24 | 21 | 3 | 21 | 0 | 0 | 3 |
| customer_reviews | 56 | 22 | 34 | 16 | 2 | 3 | 35 |
| competitor_platforms | 37 | 32 | 5 | 25 | 6 | 1 | 5 |

---

## 2. Quality-Gate Fix: `official_claims_strong`

The first full run finished with the `official_claims_strong` gate FAILing (0/15
strong records), driving an overall **NO-GO**. Root cause: `build_rag_corpora()`
scored each corpus row's `evidence_strength` from the length of the individual
extracted **claim sentence** (via `compute_evidence_strength`, 500-char threshold for
"strong"). A single extracted sentence almost never reaches 500 characters, so no
official-claim row could ever score `strong` regardless of how well-hydrated its
source page was — 21 of the 24 official-claims source pages hydrated as `strong`
(2,000+ chars from `wearetala.com`, `talaactivewear.com`, etc.), but that quality
never reached the corpus row.

**Fix:** `build_rag_corpora()` now joins each claim sentence back to its source page
(via `source_id`, added to `claims_expanded.csv`) and takes the corpus row's
`evidence_strength`/`rag_usable` from the **source page's** hydration quality
(`hydrated_claims.csv::h_evidence_strength`) rather than the sentence's own length.
`provenance_note` records `strength_source=source_page` on every affected row so this
can be verified structurally, and a validator semantic rule
(`validate_rag_corpus`, see §4) now fails any `strong`-rated `official_claims_corpus`
row that lacks that marker — guarding against an unnoticed regression back to
per-sentence scoring.

### Quality gates: before → after

| Gate | Threshold | Before | After | Status |
|------|----------:|-------:|------:|--------|
| official_claims_strong | 15 | 0 | 37 | PASS |
| customer_experience_strong_medium | 25 | 89 | 95 | PASS |
| creator_strategy_verified_strong_medium | 20 | 58 | 58 | PASS |
| total_rag_usable | 60 | 184 | 190 | PASS |
| brands_in_creator_evidence | 3 | 4 | 4 | PASS |
| no_search_url_as_creator_post | 0 | 0 | 0 | PASS |
| no_weak_record_as_rag_usable | 0 | 0 | 0 | PASS |

**Overall: NO-GO → GO.**

---

## 3. Validator Schema-Routing Fix

Running `scripts/validate_data.py` against the Day 2 outputs surfaced two apparent
failures:

- `data/corpora/official_claims_corpus.csv` — FAIL
- `data/interim/day2/creator_posts_enriched.csv` — FAIL

**Root cause (routing false positive, not a data problem):** `guess_schema()`
matched schema by filename **prefix**. `official_claims_corpus.csv` starts with
`official_claims`, so it was routed to the Day 1 `official_claims` candidate schema
(columns `claim_id`, `collection_date`, `citation_ready`, ...) instead of its own
RAG-corpus layout (`document_id`, `extracted_text`, `evidence_strength`, ...). Same
issue for `creator_posts_enriched.csv` vs. the Day 1 `creator_posts` schema.

**Fix:** added six explicit Day 2 schemas to `configs/schema.yaml`
(`official_claims_corpus`, `customer_experience_corpus`, `creator_strategy_corpus`,
`creator_posts_enriched`, `platform_strategy_enriched`, `hydrated_sources`), built from
the actual columns each file contains (see `docs/data_dictionary.md` §8–11), and
replaced prefix-only inference in `src/validation.py::guess_schema()` with an
exact-filename-stem lookup checked before the legacy prefix fallback, so a Day 2
filename can never fall through to a same-prefixed Day 1 schema. Day 1 routing is
unchanged — all seven original schemas still resolve via the same prefix fallback as
before.

New semantic (cross-field) validation was added alongside the schemas — see §4 —
covering rules the old column-presence-only `validate()` couldn't express: unique
`document_id`, URL-shaped `source_url`, `weak`/`unusable` evidence never marked
`rag_usable=True`, and (for `creator_posts_enriched`) a verified creator post must
resolve to a direct social-post URL with a non-blank handle.

Tests: `tests/test_validation_schema_routing.py` (35 tests, all passing) —
filename→schema routing for all Day 2 and Day 1 files, the specific "must not route to
the Day 1 schema" regression cases, and each semantic rule (weak+rag_usable=True,
duplicate document_id, non-social verified-post URL, etc.).

### Validation result: before → after

| File | Before | After |
|------|--------|-------|
| `data/corpora/official_claims_corpus.csv` | FAIL (wrong schema) | **PASS** |
| `data/interim/day2/creator_posts_enriched.csv` | FAIL (wrong schema) | FAIL — **real finding, see §5** |
| `data/corpora/customer_experience_corpus.csv` | PASS (schema=`unknown`, unchecked) | **PASS** (schema=`customer_experience_corpus`, actually checked) |
| `data/corpora/creator_strategy_corpus.csv` | PASS (schema=`unknown`, unchecked) | **PASS** (schema=`creator_strategy_corpus`, actually checked) |
| `data/interim/day2/platform_strategy_enriched.csv` | PASS (schema=`unknown`, unchecked) | **PASS** (schema=`platform_strategy_enriched`, actually checked) |
| `data/interim/day2/hydrated_sources.csv` | PASS (schema=`unknown`, unchecked) | **PASS** (schema=`hydrated_sources`, actually checked) |

All seven Day 1 schemas and all `data/raw/`, `data/interim/day1/`, `data/interim/day1_5/`
files continued to PASS throughout — unaffected by this fix.

---

## 4. Semantic Validation Rules Added

`src/validation.py::validate_rag_corpus()` (applied to the three `*_corpus` schemas):

- `document_id` non-blank and unique within the corpus.
- `corpus` column value must equal the schema's own corpus name (enforced via the
  existing generic `values` mechanism in `schema.yaml`).
- `source_url` non-blank and `http(s)://`-shaped.
- `extracted_text` non-blank whenever `rag_usable=True` (a `weak`, non-`rag_usable`
  row is allowed to have empty text — e.g. an empty creator caption).
- `evidence_strength` restricted to `strong`/`medium`/`weak`/`unusable` (generic
  `values` mechanism).
- `weak`/`unusable` rows can never be `rag_usable=True`.
- `official_claims_corpus` only: every row must carry text via `claim_text` or
  `extracted_text`; every `strong`-rated row must carry
  `strength_source=source_page` in `provenance_note` (see §2).

`src/validation.py::validate_creator_posts_enriched()` (applied to
`creator_posts_enriched`):

- `usable_as_creator_post=True` requires the row's URL (`post_url`, falling back to
  `source_url`) to be a **direct** Instagram/TikTok/YouTube post or video URL — not a
  profile page, campaign page, or search-result page.
- `usable_as_creator_post=True` requires non-blank `creator_handle`.
- `usable_as_creator_post=True` requires `evidence_strength` to be present.
- `brand`/`platform` non-blank on every row.
- `partnership_type` restricted to the vocabulary `_infer_partnership()` actually
  emits (generic `values` mechanism).
- Missing follower/like/view/comment counts never fail — those columns are not part
  of the required set.

---

## 5. Remaining Genuine Finding: `creator_posts_enriched.csv`

After the routing fix, `creator_posts_enriched.csv` still FAILs — this is a **real**
data-quality gap the semantic rules correctly surfaced, not a false positive:

| Issue | Rows affected | Detail |
|-------|---------------:|--------|
| `usable_as_creator_post=True` on a non-post URL | 2 / 67 | Two pre-existing Day 1 rows point at Instagram *program/ambassador* pages (`instagram.com/popular/oner-active-*`), not a direct post — they predate the Part E enrichment's `_is_verified_creator_url()` filter, which only runs on newly-discovered candidates, not on rows carried over from Day 1 |
| `creator_handle` blank for a verified post | 67 / 67 | No creator handle is ever captured for DDG-sourced candidates (`_creators_from_ddg()` never sets it) or for the carried-over Day 1 baseline rows; only a YouTube-Data-API-sourced candidate populates `creator_handle`, and the `YOUTUBE_API_KEY` was not configured for this run ([[project-tala-pipeline]] — known blocker) |
| `evidence_strength` blank for a verified post | 9 / 67 | The 9 carried-over Day 1 rows were never run through the Part E YouTube-page hydration step, so they never received a strength score |

This does **not** affect the Day 2 quality gates in §2 — `check_quality_gates()`
already filters on `evidence_strength` directly and the
`no_search_url_as_creator_post` gate already independently checks for
`google.`/`duckduckgo.` domains, so both gate sets and this validator are consistent:
the two bad Instagram rows and the 9 unscored rows were already excluded from
`creator_strategy_verified_strong_medium`'s count (58), and from the RAG corpus (which
only includes rows with `evidence_strength` in `{strong, medium}`).

**Fixed in §7 below** (Day 2 Task 1B) — resolved by (a) resolving creator identity
directly from the platform via the YouTube oEmbed endpoint and TikTok's own URL
structure (`YOUTUBE_API_KEY` was never configured this sprint and remains unused —
oEmbed needs no key), and (b) reclassifying the 2 legacy Instagram-profile rows as
`usable_as_creator_post=False` via strict URL classification, retained as discovery
leads rather than deleted.

---

## 7. Day 2 Task 1B: Creator-Evidence Repair (2026-09-18)

`scripts/repair_day2_creator_evidence.py` repairs `creator_posts_enriched.csv` so
every row marked `usable_as_creator_post=True` has a direct social post/video URL, an
identifiable creator, a verified brand link, an evidence-strength classification, and
valid provenance. A pre-repair snapshot is archived to
`data/interim/day2/archive/creator_posts_enriched_pre_repair.csv` (created once; the
script never overwrites it on a rerun). No handles, partnerships, disclosures,
metrics, or post content were fabricated — every new field is either parsed directly
from a URL (TikTok handle, all `direct_post_id`s), fetched live from a public platform
endpoint (YouTube oEmbed `author_name`/`author_url`), or matched against text the
pipeline already collected (captions, titles, hashtags).

### Repair pipeline (`src/collectors/creator_identity.py`)

1. **URL classification** (`classify_url`) — every URL is parsed and classified as a
   direct post (`youtube_video`, `youtube_short`, `tiktok_video`,
   `instagram_p`/`reel`/`tv`) or invalid (`instagram_programme_or_campaign`,
   `instagram_profile_or_other`, `youtube_search`, `search_result`, ...) / lead-only
   (`youtube_channel`, `youtube_playlist`). Invalid rows are never deleted — they're
   retained with `usable_as_creator_post=False` for use as discovery leads.
2. **Creator identity resolution**:
   - TikTok: `@handle` parsed directly from the URL path (confidence `high`).
   - YouTube: live call to `https://www.youtube.com/oembed?url=...&format=json`,
     using `author_name`/`author_url` (confidence `high`); falls back to any
     already-captured `channel_name` (confidence `medium`) if oEmbed fails; otherwise
     `unresolved`. Cached to `data/cache/youtube_oembed/` to avoid re-fetching.
   - Instagram: existing structured metadata only — no live fetch is attempted
     (`instagram.com` is bot-blocked; see `src/collectors/hydration.py
     BOT_BLOCKED_DOMAINS`), consistent with never inferring a handle from unrelated
     snippet text.
3. **Brand-link verification** (`verify_brand_link`) — requires the brand name (or an
   alias), a `#hashtag`/`@mention`, or a personalised code to appear in
   `video_title`/`caption_or_description` (never in the inherited `brand` column
   alone, since that's the search query that found the row, not evidence from the
   post). Brand name only in `channel_name` is explicitly flagged as unverified
   (likely the brand's own account, not third-party evidence).
4. **Partnership classification** (`classify_partnership`) — `paid_sponsorship` /
   `gifted` / `ambassador` / `affiliate` / `founder_or_employee` require an explicit
   disclosure marker (`#ad`, `#gifted`, "ambassador", a discount code, ...); a brand
   mention alone is never enough to infer `paid_sponsorship`.
5. **Evidence strength** (`compute_creator_evidence_strength`) — `strong` needs a
   valid URL + resolved identity + verified brand link + captured content text;
   `medium` drops the content-text requirement; `weak` covers an unresolved
   identity/brand link on an otherwise valid URL; `unusable` covers an invalid URL.
   Only `strong`/`medium` may be `usable_as_creator_post=True`.
6. **Deduplication** — keyed on `(platform, direct_post_id)`; two DDG-discovered
   duplicate video IDs (Girlfriend Collective, `krxlH6btY7U` and `NFrkpDYwFzI`, each
   found twice via different search queries with different URL query strings) were
   merged, keeping the stronger evidence and recording both contributing
   `source_id`s in `source_record_ids`.

### Results

| Metric | Before repair | After repair |
|--------|---------------:|---------------:|
| Total rows | 67 | 65 (2 duplicate video IDs merged) |
| `usable_as_creator_post=True` | 67 | 61 |
| Blank `creator_handle` among usable | 67 | 0 |
| Non-social/invalid URL among usable | 2 | 0 |
| Blank `evidence_strength` (any row) | 9 | 0 |
| Unverified `brand_link` among usable | — (not tracked) | 0 |

**Invalidated (retained as leads, not deleted):**

| Reason | Rows |
|--------|-----:|
| `instagram_programme_or_campaign_page` (Oner Active ambassador-programme pages) | 2 |
| `creator_identity_unresolved` (YouTube oEmbed 404 — video removed/private/geo-blocked, no fallback channel metadata) | 2 |

**Creator identity resolved by platform/method:**

| Method | Rows |
|--------|-----:|
| `youtube_oembed` (live oEmbed call, confidence `high`) | 60 |
| `tiktok_url` (handle parsed from URL path, confidence `high`) | 1 |

No row needed the `existing_channel_metadata` fallback (medium confidence) or
Instagram-metadata resolution — no valid direct Instagram post URL exists in the
current dataset.

**Verified strong/medium creator posts by brand x platform (all landed `strong`):**

| Brand | Platform | Count |
|-------|----------|------:|
| TALA | youtube | 24 |
| Adanola | youtube | 15 |
| Adanola | tiktok | 1 |
| Girlfriend Collective | youtube | 13 |
| Oner Active | youtube | 8 |
| **Total** | | **61** |

61 ≥ the 20-record gate threshold — **no replenishment was needed.**

### Corpus rebuild fix (parallel to §2)

Rebuilding `creator_strategy_corpus.csv` (`build_rag_corpora()` Corpus 3) surfaced
the same architectural flaw §2 fixed for official claims: the corpus was
re-deriving `evidence_strength` from the length of the (400-char-truncated)
`caption_or_description` text via `compute_evidence_strength`, which can never reach
the 500-char "strong" threshold regardless of how well-verified a post is. Fixed to
use the repaired `evidence_strength` field directly from `creators_enriched`, with
`extracted_text` falling back to `video_title` when `caption_or_description` is empty
(affected 7 Day 1 baseline rows that carry brand evidence in the title only).

### Validator additions

`src/validation.py::validate_creator_posts_enriched()` gained two checks:
`brand_link_verified` must be `True` for every `usable_as_creator_post=True` row, and
no `(platform, direct_post_id)` pair may repeat among usable rows. `configs/schema.yaml`'s
`creator_posts_enriched` schema now requires `creator_identity_confidence`,
`url_validation_status`, `brand_link_verified`, and non-blank `evidence_strength` on
every row, and its `partnership_type` vocabulary gained `founder_or_employee`.
`validate_all_files()` now excludes `archive/` directories from scanning (backup
snapshots reflect a file's old shape and shouldn't be validated against the current
schema).

Tests: `tests/test_creator_evidence_repair.py` (36 tests) — URL classification per
platform, TikTok/YouTube identity resolution (oEmbed mocked, no live calls in tests),
brand-link verification (including the inherited-`brand`-field rejection), partnership
classification, evidence-strength rules, row-level `usable_as_creator_post` gating, and
deduplication.

---

## 8. Test & Command Log

```
$ python scripts/repair_day2_creator_evidence.py
...
  Final: 67 -> 65 rows, 67 -> 61 verified usable_as_creator_post=True

$ python scripts/recover_day2_gates.py
...
  Overall: GO

$ python scripts/validate_data.py
...
  Passed        : 27 / 27
  Needs fixes   : 0

$ python -m pytest -q
71 passed in 7.42s
```
