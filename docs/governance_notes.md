# Data Governance Notes

This document records data ethics decisions, access limitations, and flagged ambiguities for team review.

---

## Platform Terms of Service Summary

| Platform | Public data accessible? | Scraping status | Notes |
|----------|------------------------|-----------------|-------|
| Trustpilot | Yes (public reviews) | Gentle scraping acceptable; no login required for public pages | Respect rate limits; no bulk harvesting |
| Google Reviews | Limited | No official API for free tier; manual collection recommended | Limited to ~10 visible reviews without maps API |
| Reddit | Yes (public posts) | Reddit API (free tier) acceptable; PRAW library | Rate limits apply; strip usernames before saving |
| Instagram | Public profiles only | No scraping; manual collection only | TOS prohibits automated access; followers/posts observable manually |
| TikTok | Public profiles only | No scraping | Same as Instagram |
| YouTube | Yes (public videos) | YouTube Data API v3 (free quota: 10,000 units/day) | Comments accessible; no login required |
| Wayback Machine | Yes | Free public API | Useful for archived brand pages |

---

## Decisions Log

### 2025-Q4 Initial Setup

**Decision:** Use manual CSV templates for Instagram and TikTok data.
**Reason:** Automated scraping violates platform TOS; free-tier API access does not cover the required data fields.
**Alternative documented:** If team obtains institutional access to a social listening tool, data can be re-imported using existing schemas.

**Decision:** Strip usernames from Reddit and review data before saving.
**Reason:** Individual usernames are PII in context; analysis only requires text and metadata.

**Decision:** Do not collect creator profile pictures or personal photos.
**Reason:** Not required for analysis; reduces PII surface area.

---

## Flagged Ambiguities

Add rows here when a data source or collection method needs team sign-off before proceeding.

| Date | Flagged by | Issue | Status |
|------|------------|-------|--------|
| (add entries) | | | |

---

## Ethical Considerations

1. **No individual harm:** Analysis targets brand practices, not individual consumers or creators.
2. **Aggregation:** All customer-level data is analysed at aggregate level; no individual profiling.
3. **Attribution:** Creator posts are analysed as strategic evidence; creator names are only included where publicly available and relevant to the analysis (e.g., ambassador tier classification).
4. **Transparency:** All datasets must carry a `synthetic` flag. Fabricated data must never appear in final analysis without clear labelling.
5. **Academic use only:** Data collected for this project is for academic research and must not be redistributed.

---

## Known Limitations

- Instagram follower counts are point-in-time and cannot be time-series without paid tools.
- Review data on Trustpilot may be skewed toward extreme (1-star and 5-star) responses.
- Reddit data reflects self-selecting, text-heavy community; may not be representative of general customer base.
- Creator tier follower thresholds are approximate industry conventions, not platform-defined.
