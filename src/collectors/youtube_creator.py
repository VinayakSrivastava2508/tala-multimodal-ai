"""Collect creator evidence via YouTube Data API v3."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.collectors.api_clients import youtube_search
from src.collectors.utils import log_failure, make_creator_row


def collect_youtube_creators(
    brand_name: str,
    queries: List[str],
    api_key: str,
    max_per_query: int = 25,
    failures: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    """Search YouTube for creator content related to a brand.

    Classifies evidence as creator_strategy (haul/review/try-on) or
    brand_positioning (brand-produced content).
    Returns list of creator_posts rows.
    """
    failures = failures if failures is not None else []
    rows: List[Dict] = []
    seen_ids: set = set()

    for query in queries:
        print(f"    YouTube: '{query}'")
        videos = youtube_search(query, api_key=api_key, max_results=max_per_query)

        if not videos:
            log_failure("youtube_api", query, "No videos returned", failures)
            continue

        for video in videos:
            video_id = video.get("video_id", "") or video.get("id", "")
            if not video_id or video_id in seen_ids:
                continue
            seen_ids.add(video_id)

            url = f"https://www.youtube.com/watch?v={video_id}"
            title = video.get("title", "")
            description = video.get("description", "")
            channel = video.get("channel_title", "") or video.get("channelTitle", "")
            published = video.get("published_at", "") or video.get("publishedAt", "")
            if published and "T" in published:
                published = published.split("T")[0]

            partnership = _infer_partnership_type(title, description)

            row = make_creator_row(
                brand=brand_name,
                source_url=url,
                caption=description,
                creator_handle=channel,
                platform="youtube",
                post_date=published,
                partnership_type=partnership,
                video_title=title,
                channel_name=channel,
                provider_name="youtube_api",
                source_method="api",
            )
            rows.append(row)

        print(f"      -> {len(videos)} videos")

    return rows


def _infer_partnership_type(title: str, description: str) -> str:
    """Infer whether a video is paid, gifted, organic, etc. from text cues."""
    combined = (title + " " + description).lower()
    if any(k in combined for k in ("paid partnership", "sponsored", "ad ", "#ad")):
        return "paid_partnership"
    if any(k in combined for k in ("gifted", "pr package", "pr product", "*gifted")):
        return "gifted"
    if any(k in combined for k in ("discount code", "affiliate", "use my code")):
        return "affiliate"
    return "organic"
