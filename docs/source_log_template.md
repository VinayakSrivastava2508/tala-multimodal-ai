# Source Log

Every dataset row in this project must have a `source_id` that appears in this log.
This is the project's chain of custody document — fill it in as you collect data.

---

## How to Record a Source

When you add rows to any CSV template, create a corresponding entry here.

**Rules:**
1. One row per distinct source (e.g., one Trustpilot page = one entry, even if you collect 30 reviews from it).
2. The `source_id` in this log must exactly match the `source_id` column in your CSV rows.
3. Every source must have a working URL (or "manual/no-url" with a written description).
4. Record the access date — pages change; the date fixes the snapshot in time.
5. Set `citation_ready = true` in your CSV only after verifying URL, date, and content match.

---

## Evidence Type Quick Reference

| Evidence Type | Valid Sources | NOT valid for |
|--------------|---------------|---------------|
| `official` | weartala.com, press releases, sustainability page, impact reports | quality/experience judgements |
| `customer_experience` | Trustpilot, Google Reviews, Reddit complaint threads, press reviews | brand strategy |
| `creator_strategy` | Instagram/TikTok creator posts, YouTube videos, partnership announcements | quality claims |
| `brand_positioning` | Brand's own Instagram, TikTok, YouTube, campaign pages | quality/sustainability facts |
| `competitor_benchmark` | Competitor brand social profiles (snapshot) | any claim about TALA |
| `press` | Vogue, Refinery29, Good on You, Guardian, industry press | raw customer opinions |
| `community` | Reddit threads, forum posts, public complaint discussions | official brand positioning |

---

## Source Log Table

| ID | Dataset | Brand | Evidence Type | Source Name | URL | Access Method | Date Accessed | Row Count | Notes |
|----|---------|-------|---------------|-------------|-----|---------------|---------------|-----------|-------|
| SRC-001 | official_claims | TALA | official | TALA — About page | https://weartala.com/pages/about | manual | YYYY-MM-DD | — | Brand mission and values copy |
| SRC-002 | official_claims | TALA | official | TALA — Sustainability page | https://weartala.com/pages/sustainability | manual | YYYY-MM-DD | — | Environmental and ethical claims |
| SRC-003 | official_claims | TALA | official | TALA — Responsibility page | https://weartala.com/pages/responsibility | manual | YYYY-MM-DD | — | Social responsibility claims |
| SRC-004 | customer_reviews | TALA | customer_experience | Trustpilot — TALA | https://www.trustpilot.com/review/weartala.com | manual | YYYY-MM-DD | — | Public reviews; strip usernames |
| SRC-005 | press_reddit_sources | TALA | community | Reddit r/femalefashionadvice — TALA search | https://www.reddit.com/r/femalefashionadvice/search/?q=TALA | manual | YYYY-MM-DD | — | Strip usernames; aggregate themes |
| SRC-006 | press_reddit_sources | TALA | community | Reddit r/gymsnark — TALA search | https://www.reddit.com/r/gymsnark/search/?q=TALA | manual | YYYY-MM-DD | — | Strip usernames; aggregate themes |
| SRC-007 | press_reddit_sources | TALA | press | Good On You — TALA brand rating | https://goodonyou.eco/brand/tala/ | manual | YYYY-MM-DD | — | Third-party sustainability assessment |
| SRC-008 | creator_posts | TALA | creator_strategy | Instagram — @weartala creator mentions | https://www.instagram.com/weartala/ | manual | YYYY-MM-DD | — | Manual sample of creator tags |
| SRC-009 | competitor_platforms | Adanola | competitor_benchmark | Instagram — @adanola profile | https://www.instagram.com/adanola/ | manual | YYYY-MM-DD | — | Follower/frequency snapshot |
| SRC-010 | competitor_platforms | Girlfriend Collective | competitor_benchmark | Instagram — @girlfriendcollective | https://www.instagram.com/girlfriendcollective/ | manual | YYYY-MM-DD | — | Follower/frequency snapshot |
| SRC-011 | competitor_platforms | Oner Active | competitor_benchmark | Instagram — @oneractive | https://www.instagram.com/oneractive/ | manual | YYYY-MM-DD | — | Follower/frequency snapshot |
| SRC-012 | official_claims | Adanola | official | Adanola — About page | https://adanola.com/pages/about | manual | YYYY-MM-DD | — | Brand positioning language |
| SRC-013 | official_claims | Girlfriend Collective | official | Girlfriend Collective — Sustainability | https://girlfriend.com/pages/sustainability | manual | YYYY-MM-DD | — | Sustainability claims for comparison |
| SRC-014 | official_claims | Oner Active | official | Oner Active — About | https://oneractive.com/pages/about | manual | YYYY-MM-DD | — | Brand mission copy |

---

## Template Row

Copy and fill in when adding a new source:

```
| SRC-XXX | {dataset} | {brand} | {evidence_type} | {source name} | {url} | {manual/api/scrape} | YYYY-MM-DD | {row count} | {notes} |
```

---

## Access Method Codes

| Code | Description |
|------|-------------|
| `manual` | Researcher copied text/data from a public page by hand |
| `bs4` | BeautifulSoup light scrape of a public page (no login) |
| `praw` | Reddit API via PRAW (free tier, public posts) |
| `youtube_api` | YouTube Data API v3 (free quota) |
| `wayback` | Archived version via Wayback Machine public API |
| `download` | File downloaded directly from a public URL |

---

## What NOT to Record

- Any data requiring login or authentication.
- Personally identifiable usernames (strip before saving).
- Paywalled press articles (record title and outlet only; note "paywalled").
