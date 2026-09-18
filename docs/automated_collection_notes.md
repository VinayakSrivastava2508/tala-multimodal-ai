# Day 1 Automated Data Collection Notes

**Script:** `scripts/collect_day1_public_data.py`
**Output dir:** `data/raw/day1/auto/`
**Last run:** 2026-09-17

---

## Collection Pipeline Overview

The pipeline runs 9 steps in priority order, with no paid or authenticated APIs beyond the
Apify free-tier token.

| Step | Provider | Schema target | Status |
|------|----------|---------------|--------|
| 1 | Official brand pages (trafilatura) | official_claims | Partial — TALA Shopify JS-rendered |
| 1b | DuckDuckGo web search (TALA brand claims) | official_claims | Fallback for JS-rendered pages |
| 2 | GDELT Doc API | press_reddit_sources | Requires >=6s delay; may return 0 on rate-limit |
| 3 | Google News RSS (feedparser) | press_reddit_sources | Working — 20 items/query |
| 4 | Reddit public JSON API | press_reddit_sources | Currently 403 — Reddit now requires OAuth |
| 5 | Reddit review discussions | customer_reviews | Currently 403 — same OAuth issue |
| 5b | DuckDuckGo review search | customer_reviews | Partial — review snippets only |
| 6 | Apify `theagents~trustpilot-reviews` | customer_reviews | Working with free $5 credit |
| 7 | YouTube Data API v3 | creator_posts | Skipped — key not configured |
| 8 | DuckDuckGo supplementary | creator_posts, competitor_platforms | Partial — rate-limited |
| 9 | Apify Google Search Scraper | creator_posts, press | Actor not found on free tier |

---

## Known Blockers and Fixes Applied

### TALA Official Pages (Shopify JS-rendered)
- **Problem:** TALA's website (`weartala.com`) redirects to Shopify-rendered pages that return
  empty body to trafilatura because content is loaded via JavaScript.
- **Fix applied:** Added DuckDuckGo web search fallback (Step 1b) to extract brand claim
  snippets from press/cached sources mentioning TALA's sustainability and ethics copy.
- **Manual supplement needed:** Manually copy key claim sentences from the pages below and
  add to `data/raw/day1/manual/official_claims_manual.csv`:
  - `https://weartala.com/pages/sustainability`
  - `https://weartala.com/pages/responsibility`
  - `https://weartala.com/pages/about`

### Reddit 403 Forbidden
- **Problem:** Reddit's public JSON API (`reddit.com/search.json`) now returns 403 for all
  non-authenticated requests. Reddit requires OAuth since mid-2023.
- **Fix applied:** Code gracefully returns empty list on 403.
- **Manual supplement needed:** Manually search Reddit at:
  - `r/femalefashionadvice` — search "TALA activewear"
  - `r/gymsnark` — search "TALA"
  - `r/ukfashion` — search "TALA"
  Save posts to `data/raw/day1/manual/press_reddit_manual.csv`.

### Trustpilot Domain Correction
- **Problem:** Config had `weartala.com` but Trustpilot uses `wearetala.com`.
- **Fix applied:** Updated `configs/sources.yaml` to `wearetala.com` → `wearetala.com`.
  Adanola uses `www.adanola.com` (with www prefix).

### YouTube API Not Configured
- **Problem:** No `YOUTUBE_API_KEY` in `.env`.
- **Impact:** 0 creator posts from YouTube.
- **Resolution:** Get a free YouTube Data API v3 key from Google Cloud Console
  (free tier: 10,000 units/day, ~100 searches). Add to `.env` as `YOUTUBE_API_KEY=`.
- **DuckDuckGo fallback:** Enabled in Step 8 when YouTube key is absent. Returns
  search snippets only — manual verification required before analysis.

### Apify Actor IDs
- **Problem:** Original actor `bebity/trustpilot-reviews-scraper` no longer exists.
- **Fix applied:** Updated to `theagents~trustpilot-reviews` (Apify free $5/month credit).
- **Input format:** `{"startUrls": [{"url": "..."}], "maxReviews": 50, "reviewsLanguage": "en"}`.
- **Cost:** ~$0.0005/review. For 50 reviews × 4 brands ≈ $0.10 (well within $5 credit).
- **Note:** `apify/google-search-scraper` also returned 404 — this actor is not available
  on the free tier. Removed from pipeline, DDG text search used as fallback.

### GDELT Rate Limiting
- **Problem:** GDELT requires ≥5 seconds between requests. Original code used 1.5s → 429.
- **Fix applied:** Increased GDELT delay to 6.0 seconds in `api_clients.py`.

### DuckDuckGo Package Rename
- **Problem:** Package `duckduckgo_search` was renamed to `ddgs`.
- **Fix applied:** `api_clients.py` now tries `ddgs` first, falls back to `duckduckgo_search`.

---

## Output Files

| File | Schema | Min target |
|------|--------|-----------|
| `official_claims_candidates.csv` | official_claims | 20 |
| `creator_posts_candidates.csv` | creator_posts | 20 |
| `customer_reviews_candidates.csv` | customer_reviews | 40 |
| `competitor_platforms_candidates.csv` | competitor_platforms | 30 |
| `press_reddit_candidates.csv` | press_reddit_sources | 25 |
| `source_queue_for_manual_review.csv` | — | advisory |
| `collection_failures.csv` | — | audit log |

---

## Manual Supplement Priority (Day 1 EDA)

If automated counts fall below target, these manual sources fill the gap fastest:

1. **Customer reviews:** Visit Trustpilot pages directly in browser, copy 10–15 reviews
   per brand as text. Strip reviewer names before saving.
2. **Official claims:** Copy sustainability/ethics copy from TALA's website pages listed above.
3. **Reddit posts:** Search Reddit directly and copy post text.
4. **Creator posts:** YouTube search "TALA activewear haul" — note video URLs and captions.

All manual rows must include `source_url`, `collection_date`, `collected_by = "manual"`,
`citation_ready = False`, and `synthetic = False`.

---

## Evidence Type Discipline Reminder

| Schema | evidence_type | NOT valid |
|--------|---------------|-----------|
| official_claims | `official` | — |
| customer_reviews | `customer_experience` | `official` |
| press_reddit_sources (Reddit) | `community` | `press` |
| press_reddit_sources (news) | `press` | `community` |
| creator_posts | `creator_strategy` | `customer_experience` |
| competitor_platforms | `competitor_benchmark` | `official` |

Instagram engagement metrics are NOT valid evidence of product quality or sustainability.
Use only for creator strategy and brand positioning analysis.
