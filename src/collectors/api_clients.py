"""Pure API wrappers: GDELT, Google News RSS, Reddit JSON, YouTube Data API."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

USER_AGENT = "tala-research-bot/0.1 (academic; non-commercial; SPJIMR ANA526-PPM)"
REDDIT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


# ── GDELT ─────────────────────────────────────────────────────────────────────

def gdelt_articles(
    query: str,
    max_records: int = 25,
    timespan: str = "3months",
) -> List[Dict[str, Any]]:
    """Search GDELT Doc API and return list of article dicts.

    Returns list with keys: url, title, seendate, domain, language.
    Free, no API key required.
    """
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": str(min(max_records, 250)),
        "timespan": timespan,
        "format": "json",
    }
    try:
        time.sleep(6.0)  # GDELT requires >= 5s between requests
        resp = requests.get(GDELT_URL, params=params, timeout=20,
                            headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        return data.get("articles", [])
    except Exception:
        return []


# ── Google News RSS ───────────────────────────────────────────────────────────

def google_news_rss(query: str, max_items: int = 20) -> List[Dict[str, Any]]:
    """Fetch articles from Google News RSS feed.

    Returns list with keys: title, link, summary, published.
    Free, no API key required.
    """
    import feedparser  # lazy import
    encoded = requests.utils.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-GB&gl=GB&ceid=GB:en"
    try:
        time.sleep(1.5)
        feed = feedparser.parse(url)
        entries = feed.entries[:max_items]
        return [
            {
                "title": getattr(e, "title", ""),
                "link": getattr(e, "link", ""),
                "summary": getattr(e, "summary", ""),
                "published": getattr(e, "published", ""),
                "source": getattr(getattr(e, "source", None), "title", "Google News"),
            }
            for e in entries
            if getattr(e, "link", "")
        ]
    except Exception:
        return []


# ── Reddit public JSON API ────────────────────────────────────────────────────

def reddit_search(
    query: str,
    subreddit: Optional[str] = None,
    limit: int = 25,
    sort: str = "relevance",
) -> List[Dict[str, Any]]:
    """Search Reddit posts via public JSON API (no auth needed).

    Returns list with keys: title, selftext, url, permalink, subreddit,
    created_utc, score, num_comments.
    """
    headers = {
        "User-Agent": REDDIT_UA,
        "Accept": "application/json",
        "Accept-Language": "en-GB,en;q=0.9",
    }
    params = {"q": query, "sort": sort, "limit": limit, "type": "link,self"}

    if subreddit:
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        params["restrict_sr"] = "1"
    else:
        url = REDDIT_SEARCH_URL

    try:
        time.sleep(3.0)
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code == 403:
            return []  # Reddit now requires OAuth for JSON API
        resp.raise_for_status()
        children = resp.json().get("data", {}).get("children", [])
        return [c.get("data", {}) for c in children]
    except Exception:
        return []


def reddit_subreddit_posts(
    subreddit: str,
    sort: str = "hot",
    limit: int = 25,
    search_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch posts from a subreddit (hot/new/top or search within subreddit)."""
    if search_query:
        return reddit_search(search_query, subreddit=subreddit, limit=limit)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    try:
        time.sleep(2.0)
        resp = requests.get(url, params={"limit": limit}, headers=headers, timeout=15)
        resp.raise_for_status()
        children = resp.json().get("data", {}).get("children", [])
        return [c.get("data", {}) for c in children]
    except Exception:
        return []


# ── YouTube Data API v3 ───────────────────────────────────────────────────────

def youtube_search(
    query: str,
    api_key: str,
    max_results: int = 25,
    order: str = "relevance",
) -> List[Dict[str, Any]]:
    """Search YouTube videos via Data API v3.

    Returns list with keys: video_id, title, description, channel_title,
    published_at, video_url.
    Requires a free API key from Google Cloud Console.
    """
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_results, 50),
        "order": order,
        "key": api_key,
    }
    try:
        time.sleep(1.0)
        resp = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        results = []
        for item in items:
            vid_id = item.get("id", {}).get("videoId", "")
            snippet = item.get("snippet", {})
            results.append({
                "video_id": vid_id,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "channel_id": snippet.get("channelId", ""),
                "published_at": snippet.get("publishedAt", ""),
                "video_url": f"https://www.youtube.com/watch?v={vid_id}" if vid_id else "",
            })
        return results
    except Exception:
        return []


# ── DuckDuckGo ────────────────────────────────────────────────────────────────

def _ddgs_instance():
    """Return a DDGS instance from whichever package is installed (ddgs or duckduckgo_search)."""
    try:
        from ddgs import DDGS
        return DDGS()
    except ImportError:
        from duckduckgo_search import DDGS  # legacy name
        return DDGS()


def ddg_text_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search DuckDuckGo and return text results.

    Returns list with keys: title, href, body.
    No API key needed; may have rate limits.
    """
    try:
        with _ddgs_instance() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        time.sleep(1.5)
        return results or []
    except Exception:
        return []


def ddg_news_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search DuckDuckGo news and return results.

    Returns list with keys: title, url, body, source, date.
    """
    try:
        with _ddgs_instance() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        time.sleep(1.5)
        return results or []
    except Exception:
        return []
