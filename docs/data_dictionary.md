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

## Naming Conventions

- **IDs:** `{type_prefix}_{brand_slug}_{platform_slug}_{sequence}` — e.g. `rev_tala_tp_001`
- **Dates:** ISO 8601 `YYYY-MM-DD` in all date fields.
- **List fields:** Pipe-separated in CSV — `quality|fit|delivery`
- **Filenames:** `{schema_name}_{brand_slug}.csv` or `{schema_name}.csv` for multi-brand files.
- **Templates:** Stored in `data/raw/templates/` — copy to `data/raw/` and rename before filling.
