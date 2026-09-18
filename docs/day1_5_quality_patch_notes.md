# Day 1.5 Quality Patch Notes
**Date:** 2026-09-18 | **Sprint:** SPJIMR ANA526-PPM

---

## What this patch does

`scripts/day1_5_quality_patch.py` applies five targeted fixes to the Day 1 cleaned interim data before Day 2 hydration and modelling. No rows are deleted. No data is fabricated. Original source URLs are always preserved.

Input: `data/interim/day1/*_cleaned.csv`
Output: `data/interim/day1_5/*_patched.csv`

---

## 1. TALA focal platform baseline (competitor_platforms)

**Problem:** `competitor_platforms_cleaned.csv` contained only Adanola, Girlfriend Collective, and Oner Active. TALA had 0 rows. For the brand moat analysis, we need TALA's own platform presence as the focal baseline against which competitors are benchmarked.

**Fix:** The script attempts to discover TALA's official social links by:
1. Fetching `wearetala.com` static HTML and parsing social media links via regex
2. Falling back to known candidate URLs: Instagram (`instagram.com/wearetala`), TikTok (`tiktok.com/@wearetala`), YouTube (`youtube.com/@wearetala`), and `wearetala.com` website
3. Sending a HEAD/GET request to each candidate URL; only adding a row if HTTP status < 400

**What was added:** One row per verified platform, all with:
- `evidence_type = focal_platform_baseline` (distinct from `competitor_benchmark` — added to schema)
- `followers`, `following`, `avg_post_frequency` all `None` (TBC — must be filled manually)
- `needs_manual_review = True` with reason `focal_baseline_metrics_tbc`
- `notes` explicitly flagging this as focal brand data, not competitor evidence

**Why this approach:** Scraping follower counts from Instagram, TikTok, or YouTube profile pages violates their robots.txt and terms of service. We verify the URL exists (HTTP 200) but do not scrape profile metrics. A team member must manually record follower counts from each platform's public profile page.

**Remaining risk:** TikTok and Instagram often return 200 even for non-existent handles (they show a "user not found" page with HTTP 200). The URL verification confirms the domain is reachable but does not guarantee the handle `wearetala` is the correct official account. **Action required:** Team member to visually confirm each platform URL before Day 2 creator strategy analysis.

---

## 2. Creator post reclassification (creator_posts)

**Problem:** `creator_posts_cleaned.csv` had rows where `source_platform` was `wearetala.com`, `active.partners`, `gymfluencers.com`, `en.wikipedia.org`, and other non-creator domains. These are brand campaign pages, affiliate network pages, and press mentions — not actual creator posts. Using them as creator content in quantitative analysis would inflate creator counts.

**Fix:** Added three new columns:

| Column | Values | Description |
|--------|--------|-------------|
| `corrected_source_platform` | Clean domain from source_url | Replaces imprecise original source_platform |
| `creator_evidence_class` | `creator_post`, `brand_campaign_page`, `affiliate_platform_page`, `press_creator_mention`, `search_lead`, `unclear` | What kind of creator evidence this row actually is |
| `usable_as_creator_post` | `True/False` | `True` only if URL is on a known creator platform (YouTube, Instagram, TikTok) |

**Classification logic:**
- `creator_post`: URL domain in `{youtube.com, instagram.com, tiktok.com, twitch.tv}`
- `brand_campaign_page`: URL domain in known brand domains (`wearetala.com`, `adanola.com`, etc.)
- `affiliate_platform_page`: URL domain in affiliate networks (`active.partners`, `ltk.com`, etc.)
- `press_creator_mention`: URL domain in known press/media domains (`gymfluencers.com`, `en.wikipedia.org`, etc.)
- `search_lead` / `unclear`: Everything else

**`usable_for_creator_strategy`** remains `True` for all rows — brand campaign pages, affiliate pages, and press mentions all provide useful creator strategy signal (e.g., brand partnerships, creator tier diversity, affiliate programme existence). Only `usable_as_creator_post` distinguishes actual creator platform content from supporting evidence.

**Remaining risk:** Instagram rows are DDG search snippets pointing to Instagram URLs, not scraped post content. `usable_as_creator_post=True` means the URL points to a creator platform, not that we have the post caption or engagement data.

---

## 3. Google News redirect URL resolution (press_reddit_sources)

**Problem:** All 127 press rows had `source_url` pointing to `news.google.com/rss/articles/CBMi...` encoded URLs. These cannot be used for Day 2 article body fetching without first resolving them to publisher URLs.

**Fix:** Added four new columns to every press row:

| Column | Description |
|--------|-------------|
| `original_source_url` | Preserved copy of the original Google News URL |
| `resolved_source_url` | Publisher URL if resolution succeeded; Google News URL if not |
| `resolved_domain` | Clean domain of resolved_source_url |
| `url_resolution_status` | One of: `resolved`, `search_fallback_used`, `google_news_unresolved`, `timeout`, `blocked`, `invalid_url` |

**Resolution strategy (in order of attempt):**
1. **Base64 decode:** Try to extract publisher URL from the encoded article ID (fast, no network). Works occasionally for older RSS encoding formats.
2. **HTTP redirect following:** `requests.get(url, allow_redirects=True)` with browser User-Agent headers. If the final URL is not on a Google domain, resolution succeeds.
3. **HTML metadata parsing:** If we land on a Google page, parse `og:url`, `<link rel="canonical">`, and `<meta http-equiv="refresh">` tags from the response HTML.
4. **DuckDuckGo title search fallback:** Search `"article title" brand` in DDG and use the first non-Google, non-social result as the publisher URL. Status = `search_fallback_used` to distinguish from direct resolution.

**Rate limiting:** 1.0s delay between Google News requests; 1.5s delay between DDG calls.

**What `search_fallback_used` means:** The resolved URL was found by searching the article title in DDG, not by following the original URL. The publisher URL should be treated as a plausible match, not a confirmed match. Do not use these as canonical citations without manual verification.

**Day 2 action:** When fetching article body text (Day 2), always use `resolved_source_url` if `url_resolution_status in ['resolved', 'search_fallback_used']`. For `google_news_unresolved` rows, you must manually open the Google News URL in a browser (which handles the JavaScript redirect) to get the article URL.

---

## 4. TALA domain normalization (official_claims)

**Problem:** The `source_url` column in `official_claims_cleaned.csv` contained three TALA domain variants:
- `wearetala.com` — canonical, correct
- `weartala.com` — likely a typo/redirect domain
- `talaactivewear.com` — possibly an old domain or an alternate brand site

These inconsistencies would cause the RAG pipeline to treat the same brand's content as coming from different sources, reducing retrieval coherence.

**Fix:** Added two new columns without modifying `source_url`:

| Column | Description |
|--------|-------------|
| `original_source_url` | Copy of source_url for reference |
| `canonical_brand_domain` | Normalized TALA domain mapped to `wearetala.com`; third-party domains kept as-is |
| `domain_normalization_note` | Explains the mapping decision for each row |

**Mapping applied:**
- `weartala.com` → `wearetala.com` (note: "Redirect/typo domain")
- `talaactivewear.com` → `wearetala.com` (note: "Alternate domain - verify redirect before citing")
- Third-party domains (clickz.com, goodonyou.eco, etc.) → kept as-is (note: "Third-party domain - press/secondary source")

**Why `source_url` was not changed:** The source URL is the actual URL from which we retrieved the snippet. Changing it would misrepresent the provenance. The `canonical_brand_domain` field is used by the RAG pipeline to group documents by brand, not as a citation URL.

**Remaining risk:** `talaactivewear.com` may be a completely separate site unrelated to TALA, or it may redirect to `wearetala.com`. This needs manual verification before any claims from that domain are included in the official RAG corpus.

---

## 5. Customer reviews redirect check

The script checks all `customer_reviews` source URLs for Google redirect domains and applies the same resolution strategy as press sources. In the Day 1 collection, review source URLs were predominantly `trustpilot.com` direct links (not Google redirects), so the resolution rate for reviews should be near 100% with `status=resolved` for direct URLs.

---

## Remaining risks before Day 2 hydration

| Risk | Severity | Required action |
|------|----------|-----------------|
| TALA social handles unverified beyond HTTP 200 | Medium | Team member confirms @wearetala is official on each platform |
| `search_fallback_used` press URLs are plausible, not confirmed | Medium | Don't cite these directly; use for article body fetch and re-verify after fetch |
| `talaactivewear.com` domain identity uncertain | Low-Medium | Check redirect chain manually; exclude from RAG corpus if not TALA-owned |
| Instagram/TikTok creator rows have no post text | High | YouTube rows are more useful for Day 2; prioritise those for creator analysis |
| All official_claims are DDG secondary sources | High | Manual addition of verbatim TALA brand copy from wearetala.com remains highest priority before Day 2 |

---

*Generated by: `scripts/day1_5_quality_patch.py` | Patch date: 2026-09-18*
