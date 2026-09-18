"""Collect community evidence from Reddit public JSON API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.collectors.api_clients import reddit_search, reddit_subreddit_posts
from src.collectors.utils import log_failure, make_press_row, make_review_row, clean_text


def _unix_to_date(ts: Any) -> str:
    """Convert Unix timestamp to ISO date string."""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _post_url(post: Dict) -> str:
    """Build permalink for a Reddit post."""
    permalink = post.get("permalink", "")
    if permalink:
        return f"https://www.reddit.com{permalink}"
    return post.get("url", "")


def collect_reddit_community(
    brand_name: str,
    queries: List[str],
    subreddits: Optional[List[str]] = None,
    limit: int = 25,
    failures: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """Collect Reddit posts as community evidence (press_reddit_sources schema).

    Searches globally and within specified subreddits.
    Returns rows with evidence_type='community'.
    """
    failures = failures if failures is not None else []
    rows: List[Dict] = []
    seen_urls: set = set()

    # Global search for each query
    for query in queries:
        print(f"    Reddit global: '{query}'")
        posts = reddit_search(query, limit=limit)
        if not posts:
            log_failure("reddit_json", query, "No posts returned", failures)
        for post in posts:
            url = _post_url(post)
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            text = clean_text(post.get("selftext", ""), 800)
            summary = text if text else post.get("title", "")
            row = make_press_row(
                brand=brand_name,
                title=post.get("title", ""),
                source_url=url,
                publication=f"r/{post.get('subreddit', 'reddit')}",
                evidence_type="community",
                summary=summary,
                date_published=_unix_to_date(post.get("created_utc", "")),
                author="[stripped]",
                provider_name="reddit_json_api",
                source_method="public_api",
                source_type="reddit",
            )
            rows.append(row)

        print(f"      -> {len(posts)} posts")

    # Subreddit-specific search
    for sub in (subreddits or []):
        for query in queries[:2]:  # top 2 queries per subreddit to avoid rate limits
            print(f"    Reddit r/{sub}: '{query}'")
            posts = reddit_subreddit_posts(sub, search_query=query, limit=15)
            for post in posts:
                url = _post_url(post)
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                text = clean_text(post.get("selftext", ""), 800)
                summary = text if text else post.get("title", "")
                row = make_press_row(
                    brand=brand_name,
                    title=post.get("title", ""),
                    source_url=url,
                    publication=f"r/{sub}",
                    evidence_type="community",
                    summary=summary,
                    date_published=_unix_to_date(post.get("created_utc", "")),
                    author="[stripped]",
                    provider_name="reddit_json_api",
                    source_method="public_api",
                    source_type="reddit",
                )
                rows.append(row)

            print(f"      -> {len(posts)} posts from r/{sub}")

    return rows


def collect_reddit_reviews(
    brand_name: str,
    review_queries: List[str],
    limit: int = 25,
    failures: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """Collect Reddit posts that look like quality/experience discussions.

    Returns rows for customer_reviews_candidates (evidence_type=customer_experience).
    """
    failures = failures if failures is not None else []
    rows: List[Dict] = []
    seen_urls: set = set()

    for query in review_queries:
        print(f"    Reddit reviews: '{query}'")
        posts = reddit_search(query, limit=limit)

        for post in posts:
            url = _post_url(post)
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            text = clean_text(post.get("selftext", ""), 1500)
            title = post.get("title", "")
            review_text = f"{title}\n{text}".strip() if text else title

            row = make_review_row(
                brand=brand_name,
                review_text=review_text,
                source_url=url,
                platform="reddit",
                review_date=_unix_to_date(post.get("created_utc", "")),
                title=title,
                provider_name="reddit_json_api",
                source_method="public_api",
            )
            row["source_platform"] = "reddit.com"
            rows.append(row)

        print(f"      -> {len(posts)} review posts")

    return rows
