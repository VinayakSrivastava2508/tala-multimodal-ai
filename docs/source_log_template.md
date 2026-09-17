# Source Log

Every dataset loaded into this project must have a row here.
The `source_id` field in all CSV files is a foreign key to the `ID` column below.

Copy the template row and fill it in when adding a new source.

---

## Log

| ID | Dataset | Brand | Source Type | Source Name / URL | Access Method | Date Accessed | Row Count | Notes |
|----|---------|-------|-------------|-------------------|---------------|---------------|-----------|-------|
| SRC-001 | official_claims | TALA | website | https://weartala.com/pages/about | manual | YYYY-MM-DD | — | Sustainability and brand mission page |
| SRC-002 | official_claims | TALA | website | https://weartala.com/pages/sustainability | manual | YYYY-MM-DD | — | Sustainability claims corpus |
| SRC-003 | reviews | TALA | trustpilot | https://www.trustpilot.com/review/weartala.com | manual / bs4 | YYYY-MM-DD | — | Public reviews, stripped usernames |
| SRC-004 | reviews | TALA | reddit | r/femalefashionadvice search: "TALA" | PRAW API | YYYY-MM-DD | — | Community comments, stripped usernames |
| SRC-005 | creator_posts | TALA | instagram | @weartala tagged posts | manual | YYYY-MM-DD | — | Sample of creator mentions |
| SRC-006 | competitor_platforms | Adanola | instagram | @adanola | manual | YYYY-MM-DD | — | Platform metrics snapshot |
| SRC-007 | competitor_platforms | Girlfriend Collective | instagram | @girlfriendcollective | manual | YYYY-MM-DD | — | Platform metrics snapshot |
| SRC-008 | competitor_platforms | Oner Active | instagram | @oneractive | manual | YYYY-MM-DD | — | Platform metrics snapshot |

---

## Template Row

```
| SRC-XXX | {dataset} | {brand} | {source_type} | {url_or_name} | {manual/api/scrape} | YYYY-MM-DD | {row_count} | {notes} |
```

---

## Access Method Codes

| Code | Meaning |
|------|---------|
| `manual` | Researcher copied data by hand from a public page |
| `bs4` | BeautifulSoup scrape of a public page (no login) |
| `api` | Official free-tier API (Reddit PRAW, YouTube Data API v3, etc.) |
| `wayback` | Archived version via Wayback Machine API |
| `download` | File downloaded directly from a public URL |
| `synthetic` | Placeholder data — not real; created for schema testing only |
