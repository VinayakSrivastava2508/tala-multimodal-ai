"""YouTube Data API v3 metadata/statistics enrichment for Day 2 Task 2B.

Pure, testable functions for video-ID extraction, batched videos.list/channels.list
calls (mockable via an injectable `get`), response parsing that preserves missing/
hidden statistics as None (never 0), and engagement-feature math that never divides
by zero and never imputes an outcome.

The API key is read once by the caller (scripts/enrich_youtube_metrics.py) from
.env via python-dotenv and passed in as a parameter -- never logged, printed, or
embedded in a cached response file here.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "youtube_api"

VIDEOS_ENDPOINT = "https://www.googleapis.com/youtube/v3/videos"
CHANNELS_ENDPOINT = "https://www.googleapis.com/youtube/v3/channels"
MAX_BATCH_SIZE = 50

_VIDEO_ID_RX = re.compile(
    r"(?:youtube\.com/watch\?(?:.*&)?v=|youtube\.com/shorts/|youtu\.be/)([\w-]{6,})", re.I
)
_ISO8601_DURATION_RX = re.compile(
    r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$"
)


def extract_video_id(url: str) -> Optional[str]:
    """Extract the YouTube video ID from a watch/shorts/youtu.be URL. Returns
    None if the URL doesn't match a recognised YouTube video pattern."""
    if not url:
        return None
    m = _VIDEO_ID_RX.search(str(url))
    return m.group(1) if m else None


def batch_ids(ids: List[str], batch_size: int = MAX_BATCH_SIZE) -> List[List[str]]:
    """Split a list of IDs into chunks of at most `batch_size` (YouTube API limit
    is 50 IDs per videos.list/channels.list call)."""
    ids = list(dict.fromkeys(ids))  # de-dupe, preserve order
    return [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]


def parse_iso8601_duration(duration: str) -> Optional[int]:
    """Parse an ISO 8601 duration (e.g. 'PT10M7S') into total seconds. Returns
    None if unparseable."""
    if not duration:
        return None
    m = _ISO8601_DURATION_RX.match(duration)
    if not m:
        return None
    days, hours, minutes, seconds = (int(g) if g else 0 for g in m.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_path(kind: str, id_: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / kind / f"{id_}.json"


def _load_cached(kind: str, id_: str, cache_dir: Path) -> Optional[dict]:
    p = _cache_path(kind, id_, cache_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cached(kind: str, id_: str, data: dict, cache_dir: Path) -> None:
    p = _cache_path(kind, id_, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ── Parsing ────────────────────────────────────────────────────────────────────

def parse_video_item(item: dict, now_iso: str) -> Dict:
    """Parse one videos.list API item into our flat record shape. Missing/hidden
    statistics fields stay None -- never coerced to 0."""
    snippet = item.get("snippet", {}) or {}
    stats = item.get("statistics", {}) or {}
    content = item.get("contentDetails", {}) or {}
    status = item.get("status", {}) or {}

    def _int_or_none(key: str) -> Optional[int]:
        v = stats.get(key)
        return int(v) if v is not None else None

    return {
        "video_id": item.get("id", ""),
        "channel_id": snippet.get("channelId", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "video_title": snippet.get("title", ""),
        "video_description": snippet.get("description", ""),
        "published_at": snippet.get("publishedAt", ""),
        "duration": content.get("duration", ""),
        "category_id": snippet.get("categoryId", ""),
        "view_count": _int_or_none("viewCount"),
        "like_count": _int_or_none("likeCount"),
        "comment_count": _int_or_none("commentCount"),
        "statistics_available": bool(stats),
        "statistics_retrieved_at": now_iso,
        "video_privacy_status": status.get("privacyStatus", ""),
        "api_retrieval_status": "success",
        "api_failure_reason": "",
    }


def parse_channel_item(item: dict, now_iso: str) -> Dict:
    """Parse one channels.list API item into our flat record shape."""
    snippet = item.get("snippet", {}) or {}
    stats = item.get("statistics", {}) or {}

    hidden = bool(stats.get("hiddenSubscriberCount", False))
    sub_count = stats.get("subscriberCount")

    return {
        "channel_id": item.get("id", ""),
        "channel_title": snippet.get("title", ""),
        "channel_custom_url": snippet.get("customUrl", ""),
        # NOTE: this is the subscriber count AT RETRIEVAL TIME, not on the
        # historical post date -- see channel_subscriber_count_current naming and
        # docs/day2_evidence_hydration_audit.md.
        "channel_subscriber_count_current": None if hidden or sub_count is None else int(sub_count),
        "hidden_subscriber_count": hidden,
        "channel_video_count": int(stats["videoCount"]) if stats.get("videoCount") is not None else None,
        "channel_view_count": int(stats["viewCount"]) if stats.get("viewCount") is not None else None,
        "channel_statistics_retrieved_at": now_iso,
    }


def _missing_video_record(video_id: str, reason: str, now_iso: str) -> Dict:
    return {
        "video_id": video_id, "channel_id": "", "channel_title": "", "video_title": "",
        "video_description": "", "published_at": "", "duration": "", "category_id": "",
        "view_count": None, "like_count": None, "comment_count": None,
        "statistics_available": False, "statistics_retrieved_at": now_iso,
        "video_privacy_status": "", "api_retrieval_status": "not_found", "api_failure_reason": reason,
    }


# ── Fetch (batched, cached, mockable) ─────────────────────────────────────────

def fetch_videos_metadata(
    video_ids: List[str],
    api_key: str,
    get: Callable[..., "requests.Response"] = requests.get,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> Dict[str, Dict]:
    """Fetch videos.list metadata for a list of video IDs, batched at
    MAX_BATCH_SIZE per call, cached per-video on disk. Returns {video_id: record}.
    Never raises on an individual video being missing/private -- records that
    reason in api_retrieval_status/api_failure_reason instead."""
    now_iso = datetime.now(timezone.utc).isoformat()
    unique_ids = list(dict.fromkeys(video_ids))
    results: Dict[str, Dict] = {}
    to_fetch: List[str] = []

    if use_cache:
        for vid in unique_ids:
            cached = _load_cached("videos", vid, cache_dir)
            if cached is not None:
                results[vid] = cached
            else:
                to_fetch.append(vid)
    else:
        to_fetch = unique_ids

    for batch in batch_ids(to_fetch, MAX_BATCH_SIZE):
        resp = get(
            VIDEOS_ENDPOINT,
            params={"part": "snippet,statistics,contentDetails,status", "id": ",".join(batch), "key": api_key},
            timeout=15,
        )
        found_ids = set()
        if resp.status_code == 200:
            data = resp.json()
            for item in data.get("items", []):
                record = parse_video_item(item, now_iso)
                vid = record["video_id"]
                found_ids.add(vid)
                results[vid] = record
                if use_cache:
                    _save_cached("videos", vid, record, cache_dir)
        else:
            reason = f"http_{resp.status_code}"
            for vid in batch:
                record = _missing_video_record(vid, reason, now_iso)
                results[vid] = record
            continue

        for vid in batch:
            if vid not in found_ids:
                record = _missing_video_record(vid, "not_returned_by_api_private_or_deleted", now_iso)
                results[vid] = record
                if use_cache:
                    _save_cached("videos", vid, record, cache_dir)

    return results


def fetch_channels_metadata(
    channel_ids: List[str],
    api_key: str,
    get: Callable[..., "requests.Response"] = requests.get,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> Dict[str, Dict]:
    """Fetch channels.list statistics for a list of channel IDs, batched and
    cached the same way as fetch_videos_metadata."""
    now_iso = datetime.now(timezone.utc).isoformat()
    unique_ids = [c for c in dict.fromkeys(channel_ids) if c]
    results: Dict[str, Dict] = {}
    to_fetch: List[str] = []

    if use_cache:
        for cid in unique_ids:
            cached = _load_cached("channels", cid, cache_dir)
            if cached is not None:
                results[cid] = cached
            else:
                to_fetch.append(cid)
    else:
        to_fetch = unique_ids

    for batch in batch_ids(to_fetch, MAX_BATCH_SIZE):
        resp = get(
            CHANNELS_ENDPOINT,
            params={"part": "snippet,statistics", "id": ",".join(batch), "key": api_key},
            timeout=15,
        )
        if resp.status_code != 200:
            continue
        data = resp.json()
        for item in data.get("items", []):
            record = parse_channel_item(item, now_iso)
            cid = record["channel_id"]
            results[cid] = record
            if use_cache:
                _save_cached("channels", cid, record, cache_dir)

    return results


# ── Engagement feature math (Part B) ──────────────────────────────────────────

def compute_engagement_features(
    view_count: Optional[int],
    like_count: Optional[int],
    comment_count: Optional[int],
    channel_subscriber_count_current: Optional[int],
    published_at: str = "",
    retrieved_at: str = "",
) -> Dict:
    """Derive engagement features from raw statistics. Every value is None
    (never 0 or fabricated) whenever an input needed for it is missing, hidden,
    or would require dividing by zero."""
    out = {
        "engagement_count": None, "interaction_rate_by_views": None,
        "like_rate_by_views": None, "comment_rate_by_views": None,
        "engagement_rate_by_current_subscribers": None,
        "log_view_count": None, "log_engagement_count": None,
        "video_age_days": None, "views_per_day": None,
    }

    if like_count is not None and comment_count is not None:
        out["engagement_count"] = like_count + comment_count
        out["log_engagement_count"] = round(math.log1p(out["engagement_count"]), 4)

    if view_count is not None and view_count > 0:
        out["log_view_count"] = round(math.log1p(view_count), 4)
        if like_count is not None:
            out["like_rate_by_views"] = round(like_count / view_count, 6)
        if comment_count is not None:
            out["comment_rate_by_views"] = round(comment_count / view_count, 6)
        if out["engagement_count"] is not None:
            out["interaction_rate_by_views"] = round(out["engagement_count"] / view_count, 6)

    if (out["engagement_count"] is not None and channel_subscriber_count_current
            and channel_subscriber_count_current > 0):
        out["engagement_rate_by_current_subscribers"] = round(
            out["engagement_count"] / channel_subscriber_count_current, 6)

    age_days = _video_age_days(published_at, retrieved_at)
    if age_days is not None:
        out["video_age_days"] = age_days
        if age_days > 0 and view_count is not None:
            out["views_per_day"] = round(view_count / age_days, 2)

    return out


def _video_age_days(published_at: str, retrieved_at: str) -> Optional[int]:
    if not published_at or not retrieved_at:
        return None
    try:
        pub = datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
        ret = datetime.fromisoformat(str(retrieved_at).replace("Z", "+00:00"))
        return max(0, (ret - pub).days)
    except Exception:
        return None


def within_channel_normalisation(
    records: List[Dict], min_posts: int = 3,
) -> List[Dict]:
    """Compute within-channel engagement z-score/percentile for channels with
    >= min_posts verified posts carrying a non-null engagement_count. Channels
    below the threshold get within_channel_normalisation_eligible=False and null
    zscore/percentile -- never silently substituted with a brand-level value."""
    from collections import defaultdict

    by_channel: Dict[str, List[int]] = defaultdict(list)
    for r in records:
        if r.get("channel_id") and r.get("engagement_count") is not None:
            by_channel[r["channel_id"]].append(r["engagement_count"])

    out = []
    for r in records:
        cid = r.get("channel_id", "")
        counts = by_channel.get(cid, [])
        n = len(counts)
        eligible = n >= min_posts
        row = dict(r)
        row["posts_from_channel_in_sample"] = n
        row["within_channel_normalisation_eligible"] = eligible
        row["within_channel_engagement_zscore"] = None
        row["within_channel_engagement_percentile"] = None
        if eligible and r.get("engagement_count") is not None:
            import statistics as _stats
            mean = _stats.mean(counts)
            std = _stats.pstdev(counts)
            row["within_channel_engagement_zscore"] = round((r["engagement_count"] - mean) / std, 4) if std > 0 else 0.0
            rank = sum(1 for c in counts if c <= r["engagement_count"])
            row["within_channel_engagement_percentile"] = round(rank / n, 4)
        out.append(row)
    return out
