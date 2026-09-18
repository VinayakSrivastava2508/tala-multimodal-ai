"""Collect press evidence via GDELT Doc API and Google News RSS."""

from __future__ import annotations

from typing import Any, Dict, List

from src.collectors.api_clients import gdelt_articles, google_news_rss, ddg_news_search
from src.collectors.utils import log_failure, make_press_row


def collect_gdelt(
    brand_name: str,
    queries: List[str],
    max_per_query: int = 25,
    failures: Optional[List[Dict]] = None,  # type: ignore[name-defined]
) -> List[Dict[str, Any]]:
    """Collect press articles for brand from GDELT API.

    Returns list of press_reddit_sources rows (evidence_type='press').
    """
    from typing import Optional  # local import for Optional
    failures = failures if failures is not None else []
    rows: List[Dict] = []

    for query in queries:
        print(f"    GDELT: '{query}'")
        articles = gdelt_articles(query, max_records=max_per_query)
        if not articles:
            log_failure("gdelt", query, "No articles returned", failures)
            continue

        for article in articles:
            url = article.get("url", "")
            if not url:
                continue
            row = make_press_row(
                brand=brand_name,
                title=article.get("title", ""),
                source_url=url,
                publication=article.get("domain", ""),
                evidence_type="press",
                summary="",
                date_published=article.get("seendate", ""),
                provider_name="gdelt",
                source_method="api",
                source_type="press",
            )
            rows.append(row)

        print(f"      -> {len(articles)} articles")

    return rows


def collect_google_news(
    brand_name: str,
    queries: List[str],
    max_per_query: int = 20,
    failures: Optional[List[Dict]] = None,  # type: ignore[name-defined]
) -> List[Dict[str, Any]]:
    """Collect articles from Google News RSS for brand queries.

    Returns list of press_reddit_sources rows (evidence_type='press').
    """
    from typing import Optional
    failures = failures if failures is not None else []
    rows: List[Dict] = []

    for query in queries:
        print(f"    Google News RSS: '{query}'")
        entries = google_news_rss(query, max_items=max_per_query)
        if not entries:
            log_failure("google_news_rss", query, "No entries returned", failures)
            continue

        for entry in entries:
            url = entry.get("link", "")
            if not url:
                continue
            row = make_press_row(
                brand=brand_name,
                title=entry.get("title", ""),
                source_url=url,
                publication=entry.get("source", "Google News"),
                evidence_type="press",
                summary=entry.get("summary", ""),
                date_published=entry.get("published", ""),
                provider_name="google_news_rss",
                source_method="rss_feed",
                source_type="news",
            )
            rows.append(row)

        print(f"      -> {len(entries)} articles")

    return rows


def collect_ddg_news(
    brand_name: str,
    queries: List[str],
    max_per_query: int = 10,
    failures: Optional[List[Dict]] = None,  # type: ignore[name-defined]
) -> List[Dict[str, Any]]:
    """Collect news from DuckDuckGo news search as press fallback.

    Returns list of press_reddit_sources rows.
    """
    from typing import Optional
    failures = failures if failures is not None else []
    rows: List[Dict] = []

    for query in queries:
        print(f"    DDG news: '{query}'")
        results = ddg_news_search(query, max_results=max_per_query)
        if not results:
            log_failure("ddg_news", query, "No results", failures)
            continue

        for result in results:
            url = result.get("url", "")
            if not url:
                continue
            row = make_press_row(
                brand=brand_name,
                title=result.get("title", ""),
                source_url=url,
                publication=result.get("source", ""),
                evidence_type="press",
                summary=result.get("body", ""),
                date_published=result.get("date", ""),
                provider_name="duckduckgo",
                source_method="news_search",
                source_type="news",
            )
            rows.append(row)

        print(f"      -> {len(results)} results")

    return rows


# Fix Optional import issue at module level
from typing import Optional  # noqa: E402 (needed for function signatures)
