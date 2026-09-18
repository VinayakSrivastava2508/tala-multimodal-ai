"""Customer review collection via Apify Trustpilot actor and Reddit fallback."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.collectors.apify_client import ApifyClient, run_trustpilot_scraper
from src.collectors.utils import log_failure, make_review_row, platform_from_url


def _parse_trustpilot_item(brand_name: str, item: Dict, domain: str) -> Optional[Dict]:
    """Convert a raw theagents~trustpilot-reviews Apify item to a customer_reviews row.

    Expected keys: body, title, rating (int), publishedDate, id, user (dict).
    """
    if item.get("error") or item.get("noResults"):
        return None

    text = item.get("body", "") or item.get("text", "") or item.get("reviewBody", "")
    if not text:
        return None

    title = item.get("title", "")
    rating_raw = item.get("rating")
    try:
        star = float(rating_raw) if rating_raw is not None else None
    except (TypeError, ValueError):
        star = None

    date_str = item.get("publishedDate", "") or item.get("date", "") or ""
    if date_str and "T" in date_str:
        date_str = date_str.split("T")[0]

    url = item.get("inputSource", "") or f"https://www.trustpilot.com/review/{domain}"
    row = make_review_row(
        brand=brand_name,
        review_text=text,
        source_url=url,
        platform="trustpilot",
        review_date=date_str,
        star_rating=star,
        title=title,
        provider_name="apify_trustpilot",
        source_method="actor",
    )
    # Strip user identity — store only review ID for dedup
    row["review_id"] = item.get("id", "")
    return row


def collect_trustpilot_reviews(
    brand_name: str,
    domain: str,
    apify_client: Optional[ApifyClient] = None,
    count: int = 50,
    failures: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """Scrape Trustpilot reviews via Apify actor.

    Falls back to empty list if Apify token unavailable.
    Returns customer_reviews rows (evidence_type='customer_experience').
    """
    failures = failures if failures is not None else []

    if not apify_client:
        print(f"  [reviews] Apify token not available, skipping Trustpilot for {domain}")
        log_failure("apify_trustpilot", domain, "No Apify token", failures)
        return []

    print(f"  [reviews] Fetching Trustpilot: {domain} ({count} reviews)")
    items = run_trustpilot_scraper(apify_client, domain, count=count)

    if not items:
        log_failure("apify_trustpilot", domain, "No items returned", failures)
        return []

    rows = []
    for item in items:
        row = _parse_trustpilot_item(brand_name, item, domain)
        if row:
            rows.append(row)

    print(f"    -> {len(rows)} Trustpilot reviews parsed")
    return rows


def collect_all_reviews(
    brand_name: str,
    trustpilot_domain: str,
    reddit_rows: Optional[List[Dict]] = None,
    apify_client: Optional[ApifyClient] = None,
    count: int = 50,
    failures: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """Collect customer reviews combining Trustpilot (Apify) and Reddit fallback.

    reddit_rows: pre-collected Reddit review rows to merge in.
    Returns unified customer_reviews list deduplicated by (source_url, text[:120]).
    """
    failures = failures if failures is not None else []
    all_rows: List[Dict] = []

    tp_rows = collect_trustpilot_reviews(
        brand_name, trustpilot_domain,
        apify_client=apify_client,
        count=count,
        failures=failures,
    )
    all_rows.extend(tp_rows)

    if reddit_rows:
        all_rows.extend(reddit_rows)
        print(f"  [reviews] Merged {len(reddit_rows)} Reddit review rows")

    return all_rows
