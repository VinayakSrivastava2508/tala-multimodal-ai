"""Public, no-auth creator-post thumbnail acquisition for Day 2 Task 2 Part B.

Tries, in order: an already-captured metadata thumbnail (e.g. cached YouTube
oEmbed `thumbnail_url`), an Open Graph image on the post page, a platform's
predictable public thumbnail URL (YouTube's img.youtube.com), then gives up.
Never bypasses login/anti-bot controls and never downloads full video files --
only a single representative still image per post.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDIA_DIR = PROJECT_ROOT / "data" / "media" / "creator_posts"
OEMBED_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "youtube_oembed"

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": BROWSER_UA, "Accept-Language": "en-GB,en;q=0.5"}


@dataclass
class MediaResult:
    """One row of media acquisition output per creator post."""
    post_id: str = ""
    brand: str = ""
    platform: str = ""
    post_url: str = ""
    media_url: str = ""
    local_media_path: str = ""
    media_type: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    file_hash: str = ""
    retrieval_status: str = "not_attempted"  # success | unavailable | failed
    retrieval_method: str = ""
    retrieved_at: str = ""
    failure_reason: str = ""

    def to_dict(self) -> dict:
        """Return dataclass as plain dict."""
        return asdict(self)


def _oembed_cache_path(video_url: str) -> Path:
    key = hashlib.sha256(video_url.encode()).hexdigest()
    return OEMBED_CACHE_DIR / f"{key}.json"


def cached_youtube_thumbnail(video_url: str) -> Optional[str]:
    """Return the `thumbnail_url` from an already-cached YouTube oEmbed response,
    if one exists (from the Day 2 Task 1B creator-identity repair). No network call."""
    p = _oembed_cache_path(video_url)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        url = data.get("thumbnail_url", "")
        return url or None
    except Exception:
        return None


def youtube_direct_thumbnail_url(video_id: str) -> str:
    """Construct YouTube's predictable public thumbnail URL for a video ID (no
    auth, no scraping -- a stable, documented public endpoint)."""
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


def fetch_open_graph_image(
    page_url: str, get=requests.get, timeout: int = 12,
) -> Optional[str]:
    """Fetch a public post page and extract its og:image meta tag, if any.
    Returns None on any failure (blocked, no tag, non-200, etc.) -- never raises."""
    try:
        resp = get(page_url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return None
        from bs4 import BeautifulSoup  # lazy import
        soup = BeautifulSoup(resp.text, "lxml")
        tag = soup.find("meta", property="og:image")
        if tag and tag.get("content"):
            return str(tag["content"])
    except Exception:
        pass
    return None


def _download_bytes(url: str, get=requests.get, timeout: int = 15) -> Optional[bytes]:
    try:
        resp = get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        pass
    return None


def save_image(content: bytes, dest_path: Path) -> tuple[int, int, str]:
    """Save image bytes to disk. Returns (width, height, sha256_hash)."""
    from PIL import Image  # lazy import
    import io

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(content)
    file_hash = hashlib.sha256(content).hexdigest()
    try:
        with Image.open(io.BytesIO(content)) as img:
            width, height = img.size
    except Exception:
        width, height = None, None
    return width, height, file_hash


def acquire_media_for_post(
    post_id: str,
    brand: str,
    platform: str,
    post_url: str,
    direct_post_id: str = "",
    get=requests.get,
    media_dir: Path = MEDIA_DIR,
    delay: float = 1.0,
) -> MediaResult:
    """Acquire one representative public thumbnail for a creator post, trying
    each method in order and stopping at the first success. Never fails the
    calling record -- returns retrieval_status='unavailable' if nothing works."""
    now = datetime.now(timezone.utc).isoformat()
    result = MediaResult(
        post_id=post_id, brand=brand, platform=platform, post_url=post_url,
        retrieved_at=now,
    )
    platform_l = (platform or "").lower()

    candidates: list[tuple[str, str]] = []  # (method, url)

    if platform_l == "youtube":
        cached = cached_youtube_thumbnail(post_url)
        if cached:
            candidates.append(("cached_oembed_thumbnail", cached))
        elif direct_post_id:
            candidates.append(("youtube_public_thumbnail", youtube_direct_thumbnail_url(direct_post_id)))

    og_image = fetch_open_graph_image(post_url, get=get)
    if og_image:
        candidates.append(("open_graph_image", og_image))

    if platform_l == "youtube" and direct_post_id and not any(m == "youtube_public_thumbnail" for m, _ in candidates):
        candidates.append(("youtube_public_thumbnail", youtube_direct_thumbnail_url(direct_post_id)))

    if not candidates:
        result.retrieval_status = "unavailable"
        result.failure_reason = "no_public_thumbnail_method_available_for_platform"
        return result

    for method, media_url in candidates:
        time.sleep(delay)
        content = _download_bytes(media_url, get=get)
        if not content:
            continue
        ext = ".jpg"
        if media_url.lower().endswith(".png"):
            ext = ".png"
        dest = media_dir / f"{post_id}{ext}"
        width, height, file_hash = save_image(content, dest)
        result.media_url = media_url
        result.local_media_path = str(dest.relative_to(PROJECT_ROOT))
        result.media_type = "image"
        result.width = width
        result.height = height
        result.file_hash = file_hash
        result.retrieval_status = "success"
        result.retrieval_method = method
        return result

    result.retrieval_status = "unavailable"
    result.failure_reason = "all_candidate_urls_failed_to_download"
    return result
