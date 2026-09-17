# Data Dictionary

All field definitions mirror `configs/schema.yaml`. This document provides human-readable descriptions and notes for each dataset.

---

## 1. `social_posts`

Saved to: `data/raw/social_posts_{brand}.csv`

| Field | Type | Description |
|-------|------|-------------|
| `post_id` | str | Unique identifier (platform_date_sequence) |
| `brand` | str | Brand slug from `configs/brands.yaml` |
| `platform` | str | instagram / tiktok / youtube |
| `post_date` | date | ISO 8601 (YYYY-MM-DD) |
| `caption` | str | Post caption text |
| `media_type` | str | image / video / reel / story / carousel |
| `hashtags` | list | Pipe-separated hashtags |
| `likes` | int | Like count at collection time |
| `comments` | int | Comment count at collection time |
| `views` | int | View count (video only) |
| `url` | str | Public URL of the post |
| `image_path` | str | Relative path to downloaded image (if applicable) |
| `source_id` | str | FK → `docs/source_log_template.md` |
| `synthetic` | bool | True if placeholder / not real data |

**Analysis use:** Brand positioning, campaign tone, content format strategy. NOT quality or sustainability evidence.

---

## 2. `creator_posts`

Saved to: `data/raw/creator_posts_{brand}.csv`

| Field | Type | Description |
|-------|------|-------------|
| `post_id` | str | Unique identifier |
| `creator_handle` | str | Platform handle (no PII beyond public handle) |
| `platform` | str | Platform |
| `brand_mentioned` | str | Brand slug |
| `post_date` | date | ISO 8601 |
| `partnership_type` | str | paid / gifted / organic / ambassador / unknown |
| `caption` | str | Post caption |
| `creator_followers` | int | Follower count at collection time |
| `creator_tier` | str | nano (<10K) / micro (10K–100K) / mid (100K–500K) / macro (500K–1M) / mega (>1M) |
| `likes` | int | Like count |
| `comments` | int | Comment count |
| `views` | int | View/play count |
| `sentiment_label` | str | Manual or model sentiment |
| `url` | str | Public URL |
| `source_id` | str | FK → source log |
| `synthetic` | bool | True if placeholder |

**Analysis use:** Creator tier distribution, platform strategy, partnership diversity, reach metrics.

---

## 3. `reviews`

Saved to: `data/raw/reviews_{brand}.csv`

| Field | Type | Description |
|-------|------|-------------|
| `review_id` | str | Unique identifier |
| `brand` | str | Brand slug |
| `platform` | str | Source platform (trustpilot, google, sitejabber, etc.) |
| `review_date` | date | ISO 8601 |
| `star_rating` | float | 1.0–5.0 |
| `review_text` | str | Full review text |
| `topics_mentioned` | list | Pipe-separated topics (quality, fit, delivery, etc.) |
| `verified_purchase` | bool | If confirmed by platform |
| `sentiment_label` | str | positive / neutral / negative / mixed |
| `complaint_type` | str | Primary complaint category if negative |
| `source_id` | str | FK → source log |
| `synthetic` | bool | True if placeholder |

**Analysis use:** Primary evidence for quality, fit, sizing, durability, returns, complaints.

---

## 4. `official_claims`

Saved to: `data/raw/official_claims_{brand}.csv`

| Field | Type | Description |
|-------|------|-------------|
| `claim_id` | str | Unique identifier (brand_category_seq) |
| `brand` | str | Brand slug |
| `claim_text` | str | Verbatim claim from official source |
| `claim_category` | str | Claim type |
| `source_type` | str | Where the claim was found |
| `source_url` | str | URL of source page |
| `date_captured` | date | When the claim was collected |
| `verifiable` | bool | Whether the claim can be checked against evidence |
| `evidence_strength` | str | Researcher's assessment: strong / moderate / weak / unverifiable |
| `source_id` | str | FK → source log |
| `synthetic` | bool | True if placeholder |

**Analysis use:** RAG corpus (official), claim validation baseline, divergence scoring.

---

## 5. `competitor_platforms`

Saved to: `data/raw/competitor_platforms.csv`

| Field | Type | Description |
|-------|------|-------------|
| `brand` | str | Brand slug |
| `platform` | str | Platform |
| `followers` | int | Follower count at collection date |
| `avg_post_frequency` | float | Posts per week (estimated) |
| `creator_tier_mix` | str | Description of tier composition |
| `content_format_mix` | str | Description of content format split |
| `date_captured` | date | Collection date |
| `notes` | str | Free-text notes |
| `source_id` | str | FK → source log |
| `synthetic` | bool | True if placeholder |

**Analysis use:** Cross-brand platform strategy benchmarking.

---

## 6. `claim_assessments`

Saved to: `data/processed/claim_assessments_{brand}.csv`

| Field | Type | Description |
|-------|------|-------------|
| `assessment_id` | str | Unique identifier |
| `claim_id` | str | FK → `official_claims` |
| `brand` | str | Brand slug |
| `claim_category` | str | Category inherited from claim |
| `claim_text` | str | Claim text (denormalised for readability) |
| `supporting_evidence` | str | Evidence text or row references that support the claim |
| `contradicting_evidence` | str | Evidence text or row references that contradict the claim |
| `divergence_score` | float | 0.0 (fully supported) → 1.0 (fully contradicted) |
| `assessor` | str | human / model / hybrid |
| `confidence` | str | Confidence in the assessment |
| `notes` | str | Analyst notes |
| `source_id` | str | FK → source log |
| `synthetic` | bool | True if placeholder |

---

## Naming Conventions

- All date fields: ISO 8601 `YYYY-MM-DD`.
- List fields: pipe-separated strings in CSV (`tag1|tag2|tag3`).
- All `source_id` values must match an entry in `docs/source_log_template.md`.
- File names: `{schema_name}_{brand_slug}.csv` or `{schema_name}.csv` for multi-brand files.
