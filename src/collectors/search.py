"""DuckDuckGo web search collector — press, creator, and competitor evidence."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.collectors.api_clients import ddg_text_search
from src.collectors.utils import log_failure, make_press_row, make_creator_row, make_competitor_row


def collect_ddg_press(
    brand_name: str,
    queries: List[str],
    max_per_query: int = 8,
    failures: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """DuckDuckGo text search for press/editorial results about a brand.

    Returns press_reddit_sources rows (evidence_type='press').
    """
    failures = failures if failures is not None else []
    rows: List[Dict] = []

    for query in queries:
        print(f"    DDG press: '{query}'")
        results = ddg_text_search(query, max_results=max_per_query)
        if not results:
            log_failure("ddg_text", query, "No results returned", failures)
            continue

        for r in results:
            url = r.get("href", "")
            if not url:
                continue
            row = make_press_row(
                brand=brand_name,
                title=r.get("title", ""),
                source_url=url,
                publication=r.get("href", "").split("/")[2].replace("www.", ""),
                evidence_type="press",
                summary=r.get("body", ""),
                provider_name="duckduckgo",
                source_method="web_search",
                source_type="press",
            )
            rows.append(row)

        print(f"      -> {len(results)} results")

    return rows


def collect_ddg_creator(
    brand_name: str,
    queries: List[str],
    max_per_query: int = 8,
    failures: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """DuckDuckGo search for creator/ambassador content about a brand.

    Returns creator_posts_candidates rows (evidence_type='creator_strategy').
    Results are search snippets — not actual posts — pending manual review.
    """
    failures = failures if failures is not None else []
    rows: List[Dict] = []

    for query in queries:
        print(f"    DDG creator: '{query}'")
        results = ddg_text_search(query, max_results=max_per_query)
        if not results:
            log_failure("ddg_creator", query, "No results returned", failures)
            continue

        for r in results:
            url = r.get("href", "")
            if not url:
                continue
            platform = _guess_platform(url)
            row = make_creator_row(
                brand=brand_name,
                source_url=url,
                caption=r.get("body", ""),
                platform=platform,
                video_title=r.get("title", ""),
                provider_name="duckduckgo",
                source_method="web_search",
            )
            row["notes"] = "Search snippet — verify post content manually before using as evidence."
            rows.append(row)

        print(f"      -> {len(results)} results")

    return rows


def collect_ddg_competitor(
    brand_name: str,
    queries: List[str],
    max_per_query: int = 8,
    failures: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """DuckDuckGo search for competitor brand information.

    Returns competitor_platforms_candidates rows.
    """
    failures = failures if failures is not None else []
    rows: List[Dict] = []

    for query in queries:
        print(f"    DDG competitor: '{query}'")
        results = ddg_text_search(query, max_results=max_per_query)
        if not results:
            log_failure("ddg_competitor", query, "No results returned", failures)
            continue

        for r in results:
            url = r.get("href", "")
            if not url:
                continue
            platform = _guess_platform(url)
            row = make_competitor_row(
                brand=brand_name,
                platform=platform,
                source_url=url,
                context_text=r.get("body", ""),
                title=r.get("title", ""),
                provider_name="duckduckgo",
                source_method="web_search",
            )
            rows.append(row)

        print(f"      -> {len(results)} results")

    return rows


def collect_ddg_reviews(
    brand_name: str,
    queries: Optional[List[str]] = None,
    max_per_query: int = 8,
    failures: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """Use DuckDuckGo to find review pages and snippets for a brand.

    Returns customer_reviews_candidates rows (evidence_type='customer_experience').
    Snippets are search results from review sites — not actual parsed reviews;
    annotation: manual_verification_needed.
    """
    from src.collectors.utils import make_review_row
    failures = failures if failures is not None else []
    if queries is None:
        queries = [
            f"{brand_name} activewear review",
            f"{brand_name} activewear customer experience",
            f"site:trustpilot.com {brand_name}",
        ]
    rows: List[Dict] = []
    seen_urls: set = set()

    REVIEW_DOMAINS = ("trustpilot.com", "sitejabber.com", "reviews.io",
                      "google.com/maps", "reddit.com", "glassdoor.com",
                      "productreview.com.au", "resellerratings.com")

    for query in queries:
        print(f"    DDG reviews: '{query}'")
        results = ddg_text_search(query, max_results=max_per_query)
        if not results:
            log_failure("ddg_reviews", query, "No results", failures)
            continue

        added = 0
        for r in results:
            url = r.get("href", "")
            if not url or url in seen_urls:
                continue
            body = r.get("body", "")
            title = r.get("title", "")
            # Only keep results that look like review content
            if not (any(d in url.lower() for d in REVIEW_DOMAINS)
                    or any(kw in (body + title).lower()
                           for kw in ("review", "rating", "stars", "quality",
                                      "sizing", "fabric", "returns", "complaint"))):
                continue
            seen_urls.add(url)
            row = make_review_row(
                brand=brand_name,
                review_text=body,
                source_url=url,
                platform=_guess_platform(url),
                title=title,
                provider_name="duckduckgo",
                source_method="web_search",
            )
            row["notes"] = "DDG snippet — fetch full page to verify; manual review needed."
            rows.append(row)
            added += 1

        print(f"      -> {added} review snippets")

    return rows


def _guess_platform(url: str) -> str:
    """Infer platform type from URL domain."""
    url_lower = url.lower()
    if "instagram.com" in url_lower:
        return "instagram"
    if "tiktok.com" in url_lower:
        return "tiktok"
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    if "reddit.com" in url_lower:
        return "reddit"
    if "trustpilot.com" in url_lower:
        return "trustpilot"
    return "website"
