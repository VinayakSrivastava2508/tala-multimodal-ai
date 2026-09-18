# Day 1 Evidence Audit — TALA Multimodal AI Strategy
**Date:** 2026-09-18 | **Sprint:** SPJIMR ANA526-PPM | **Day:** 1 of 3

---

## 1. Raw vs Cleaned Row Counts

| Schema | Raw rows | Cleaned rows | Removed | Removal reason |
|--------|----------|--------------|---------|----------------|
| official_claims | 24 | 24 | 0 | — |
| customer_reviews | 56 | 56 | 0 | — |
| press_reddit_sources | 127 | 127 | 0 | — |
| creator_posts | 36 | 36 | 0 | — |
| competitor_platforms | 33 | 33 | 0 | — |
| **Total** | **276** | **276** | **0** | — |

No rows were dropped. All 276 rows passed deduplication and provenance checks. The cleaning step added audit columns but did not discard data.

---

## 2. Evidence Readiness by Analysis Track

| Schema | usable_for_analysis | usable_for_rag | usable_for_creator_strategy | usable_for_quality_responsibility |
|--------|--------------------:|---------------:|----------------------------:|----------------------------------:|
| official_claims | 23 / 24 | 23 / 24 | 0 | 23 / 24 |
| customer_reviews | 55 / 56 | 55 / 56 | 0 | 55 / 56 |
| press_reddit_sources | 127 / 127 | 127 / 127 | 0 | 0 |
| creator_posts | 36 / 36 | 36 / 36 | 36 / 36 | 0 |
| competitor_platforms | 33 / 33 | 33 / 33 | 0 | 0 |
| **Total** | **274 / 276** | **274 / 276** | **36 / 276** | **78 / 276** |

**Interpretation:** `usable_for_analysis=True` means the row has sufficient text and provenance to be included in quantitative analysis. It does NOT mean the row is fully verified or citation-ready.

---

## 3. Manual Review Queue

| Schema | Needs review | Primary flag reason |
|--------|-------------:|---------------------|
| official_claims | 24 / 24 | `snippet_only_duckduckgo` — DDG search snippets, not verbatim brand copy |
| customer_reviews | 56 / 56 | `snippet_only_duckduckgo` — snippets pointing to review pages, not extracted review text |
| press_reddit_sources | 127 / 127 | `press_lead_no_article_body` — RSS summaries, article body not fetched |
| creator_posts | 36 / 36 | `social_unverified_creator_context` — DDG pointer rows, creator identity unconfirmed |
| competitor_platforms | 28 / 33 | `snippet_only_duckduckgo` — 5 rows from trafilatura extraction are verified |
| **Total** | **271 / 276** | — |

`needs_manual_review=True` is conservative by design. It flags rows where the text source is not directly verified. All flagged rows are still `usable_for_analysis` — the flag is a prompt for human review before citing or drawing final conclusions, not a blocker for Day 2 feature extraction.

---

## 4. Source Limitation Details

### official_claims (24 rows)
- **What we have:** DuckDuckGo search snippets referencing TALA brand claims in press and secondary sources.
- **What is missing:** Verbatim brand copy from weartala.com/pages/sustainability, /about, /responsibility.
- **Why:** TALA's website is Shopify JS-rendered; trafilatura returns empty body. Login-walled content not attempted.
- **Impact:** Claims corpus is secondary source quotes, not primary brand statements. Divergence scoring will be approximate until primary copy is added.
- **Action before Day 2:** Manually copy 5–10 verbatim claim statements from weartala.com and append to `data/interim/day1/official_claims_cleaned.csv`.

### customer_reviews (56 rows)
- **What we have:** DuckDuckGo search result snippets (~80–300 chars) pointing to Trustpilot, Google Reviews, and Sitejabber pages.
- **What is missing:** Full review text bodies. Star ratings and review dates are largely null.
- **Why:** Apify Trustpilot actor (`theagents~trustpilot-reviews`) hit the free monthly run limit after test runs. No paid API used.
- **Impact:** Sentiment and quality analysis requires full review text. Snippets are too short for meaningful topic modelling.
- **Action before Day 2:** Manually open 10–15 Trustpilot URLs from the manual review queue and paste review text. Alternatively, add Apify credits (~$3 for ~6,000 reviews).

### press_reddit_sources (127 rows)
- **What we have:** Google News RSS entries with article title + summary (~200–500 chars). 127 rows across 4 brands.
- **What is missing:** Full article body text. Reddit threads (403 from public JSON API).
- **Why:** RSS feeds provide title and summary only. Reddit requires OAuth since mid-2023.
- **Impact:** Topic modelling and TF-IDF will work on summaries, but key quotes and article sentiment will be approximate.
- **Action before Day 2:** Run `scripts/fetch_article_bodies.py` (to be built on Day 2) on top 20 TALA press rows to fetch full article text.

### creator_posts (36 rows)
- **What we have:** DuckDuckGo search result rows pointing to creator content URLs across Instagram, TikTok, and YouTube. Partnership type inferred from title/snippet keywords.
- **What is missing:** Actual post captions, view counts, follower counts, direct post dates.
- **Why:** Instagram and TikTok do not allow public API access without authentication. YouTube API key was not configured in this run.
- **Impact:** Creator strategy analysis can identify platform mix and partnership type distribution but not quantitative reach metrics.
- **Action before Day 2:** Configure YouTube Data API v3 key in `.env` to fetch video metadata for YouTube rows. Instagram/TikTok rows remain as candidate evidence.

### competitor_platforms (33 rows)
- **What we have:** 5 rows with trafilatura-extracted page text; 28 DDG pointer rows with URL and snippet only.
- **What is missing:** Follower counts, engagement rates, post frequency data.
- **Why:** Scraping follower counts from social profile pages is not allowed by robots.txt on these platforms.
- **Impact:** Competitor benchmark comparison will be qualitative (presence vs. absence) rather than quantitative until manual data entry.
- **Action before Day 2:** Manually record follower counts from public profile pages for each brand × platform combination.

---

## 5. How This Supports the Assignment Question

The assignment asks: *What is TALA's multimodal brand moat, and how does its creator strategy and claim–experience gap compare to competitors?*

Day 1 data supports the following workstreams:

| Workstream | Day 1 evidence available | Quality |
|-----------|--------------------------|---------|
| **Claim validation** (official claims vs customer experience) | 23 claim snippets + 55 review snippets | Moderate — snippets only; needs full text |
| **Creator strategy analysis** | 36 creator pointer rows across 4 brands | Low-moderate — no engagement data yet |
| **Brand positioning** | 24 official claim rows (DDG secondary sources) | Moderate — not verbatim brand copy |
| **Competitor benchmarking** | 33 platform rows (28 DDG pointers) | Low — no quantitative metrics |
| **Press/community sentiment** | 127 RSS summaries | Moderate — summaries are substantive for topic modelling |

The strongest corpus for Day 2 modelling is `press_reddit_sources` (127 rows with substantive summaries). The weakest is `customer_reviews` (all DDG snippets, short text).

---

## 6. What Should Move to Day 2 Modelling / RAG

### Ready for Day 2 — no blocking action required
- **press_reddit_sources:** 127 rows with title + summary → TF-IDF topic modelling, sentence embeddings for press corpus.
- **creator_posts:** 36 rows → platform mix analysis, partnership type frequency, brand × platform matrix.
- **official_claims:** 23 usable rows → seed the `official_claims` RAG corpus; annotate claim categories.
- **competitor_platforms:** 33 rows → qualitative presence analysis; foundation for Day 3 competitor benchmark table.

### Recommended human actions (1–2 hours) before Day 2 modelling
1. **Add 10 verbatim TALA brand claims** from weartala.com/pages/sustainability to `data/interim/day1/official_claims_cleaned.csv`.
2. **Add 15 full Trustpilot reviews** from the manual review queue into `data/interim/day1/customer_reviews_cleaned.csv`.
3. **Add follower counts** for each brand × platform in `data/interim/day1/competitor_platforms_cleaned.csv`.

### Day 2 pipeline tasks
- `scripts/fetch_article_bodies.py` — fetch full article text for top 20 TALA press rows.
- `src/text_features.py` — TF-IDF + sentence embeddings on `press_reddit_sources` and `official_claims`.
- `src/fusion_models.py` — baseline claim divergence scoring (official claims vs customer reviews + press).
- `src/moat_metrics.py` — creator tier distribution, platform diversity index.
- `src/rag_pipeline.py` (Day 3) — build three corpora: official_claims, creator_content, customer_experience.

---

## 7. Evidence Source Discipline Reminder

Per CLAUDE.md:

- `customer_reviews` and `press_reddit_sources` are the correct evidence for product quality, fit, sizing, and durability claims.
- `official_claims` is the correct corpus for what the brand says — not what customers experience.
- `creator_posts` is evidence for creator strategy only — not quality or sustainability.
- Instagram/TikTok engagement (likes, followers) must NOT be used as a proxy for product quality or sustainability validity.
- All rows in this dataset are candidate evidence. No row is a final finding. All rows labelled `needs_manual_review=True` must be verified by a team member before appearing in the final report as a citation.

---

*Generated by: `scripts/clean_day1_data.py` + manual audit | Collection date: 2026-09-17 | Audit date: 2026-09-18*
