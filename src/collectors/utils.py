"""Shared utilities for all collector modules."""

from __future__ import annotations

import csv
import hashlib
import re
import time
import urllib.robotparser
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_HEADERS = {
    "User-Agent": "tala-research-bot/0.1 (academic; non-commercial; SPJIMR ANA526-PPM)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.5",
}
TODAY = date.today().isoformat()


# ── HTTP ───────────────────────────────────────────────────────────────────────

def safe_get(
    url: str,
    delay: float = 2.0,
    timeout: int = 15,
    extra_headers: Optional[Dict] = None,
) -> Optional[requests.Response]:
    """Rate-limited GET; returns Response or None on any failure."""
    time.sleep(delay)
    headers = {**DEFAULT_HEADERS, **(extra_headers or {})}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return resp
    except Exception:
        return None


def can_fetch(url: str) -> bool:
    """Return True if robots.txt permits crawling this URL with our user-agent."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(DEFAULT_HEADERS["User-Agent"], url)
    except Exception:
        return True  # allow if robots.txt unreachable


def platform_from_url(url: str) -> str:
    """Extract clean domain name from a URL for use as source_platform."""
    try:
        netloc = urlparse(url).netloc
        return netloc.replace("www.", "").split(":")[0]
    except Exception:
        return "unknown"


# ── Text ──────────────────────────────────────────────────────────────────────

def clean_text(text: str, max_len: int = 2000) -> str:
    """Strip whitespace, control chars, and cap length."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = re.sub(r"\s+", " ", text.strip())
    return text[:max_len]


def extract_text_from_url(url: str, delay: float = 2.0) -> str:
    """Fetch a URL and extract main text via trafilatura (falls back to bs4)."""
    resp = safe_get(url, delay=delay)
    if not resp:
        return ""
    html = resp.text
    try:
        import trafilatura  # lazy import
        text = trafilatura.extract(
            html,
            include_links=False,
            include_images=False,
            include_tables=False,
            no_fallback=False,
        )
        return clean_text(text or "")
    except Exception:
        pass
    try:
        from bs4 import BeautifulSoup  # fallback
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return clean_text(soup.get_text(separator=" "))
    except Exception:
        return ""


def split_into_claims(text: str, min_len: int = 40, max_len: int = 500) -> List[str]:
    """Split text into sentence-sized claim chunks."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if min_len <= len(s.strip()) <= max_len]


# ── Row builders ──────────────────────────────────────────────────────────────

def _base(brand: str, evidence_type: str, source_url: str,
          provider_name: str, source_method: str) -> Dict[str, Any]:
    """Return the common provenance fields every row must have."""
    return {
        "brand": brand,
        "evidence_type": evidence_type,
        "source_url": source_url,
        "source_platform": platform_from_url(source_url),
        "collection_date": TODAY,
        "collected_by": "auto_pipeline",
        "provider_name": provider_name,
        "source_method": source_method,
        "citation_ready": False,
        "synthetic": False,
        "notes": "",
    }


_CLAIM_CATEGORY_MAP = {
    "brand_values": "other",
    "sustainability_claims": "sustainability",
    "quality_claims": "quality",
    "ethics_claims": "ethics",
    "community_claims": "community",
    "performance_claims": "performance",
}

_SOURCE_TYPE_MAP = {
    "press": "other",
    "news": "other",
    "blog": "other",
    "review": "other",
}

VALID_CLAIM_CATEGORIES = {"community", "ethics", "inclusivity", "other",
                           "performance", "pricing", "quality", "sustainability"}
VALID_SOURCE_TYPES_CLAIMS = {"campaign_copy", "impact_report", "interview", "other",
                              "packaging", "press_release", "social_bio", "website"}


def make_claim_row(
    brand: str, claim_text: str, source_url: str,
    claim_category: str = "other", source_type: str = "website",
    provider_name: str = "trafilatura", source_method: str = "html_extraction",
) -> Dict[str, Any]:
    """Build a row for official_claims_candidates."""
    # Normalise enums to schema-valid values
    claim_category = _CLAIM_CATEGORY_MAP.get(claim_category, claim_category)
    if claim_category not in VALID_CLAIM_CATEGORIES:
        claim_category = "other"
    source_type = _SOURCE_TYPE_MAP.get(source_type, source_type)
    if source_type not in VALID_SOURCE_TYPES_CLAIMS:
        source_type = "other"

    row = _base(brand, "official", source_url, provider_name, source_method)
    row.update({
        "claim_id": "",
        "claim_text": claim_text,
        "claim_category": claim_category,
        "source_type": source_type,
        "verifiable": None,
        "evidence_strength": None,
        "source_id": "",
    })
    return row


def make_press_row(
    brand: str, title: str, source_url: str, publication: str,
    evidence_type: str = "press", summary: str = "",
    date_published: str = "", author: str = "",
    provider_name: str = "gdelt", source_method: str = "api",
    source_type: str = "press",
) -> Dict[str, Any]:
    """Build a row for press_reddit_sources_candidates."""
    row = _base(brand, evidence_type, source_url, provider_name, source_method)
    row.update({
        "source_type": source_type,
        "publication": publication,
        "title": title,
        "author": author,
        "date_published": date_published,
        "summary": clean_text(summary, 500),
        "key_quote": "",
        "topics_mentioned": "",
        "sentiment_label": None,
        "source_id": "",
    })
    return row


def make_review_row(
    brand: str, review_text: str, source_url: str,
    platform: str = "", review_date: str = "",
    star_rating: Optional[float] = None, title: str = "",
    provider_name: str = "apify", source_method: str = "actor",
) -> Dict[str, Any]:
    """Build a row for customer_reviews_candidates."""
    row = _base(brand, "customer_experience", source_url, provider_name, source_method)
    row.update({
        "review_id": "",
        "platform": platform or platform_from_url(source_url),
        "review_date": review_date,
        "star_rating": star_rating,
        "review_text": clean_text(review_text, 1500),
        "title": title,
        "topics_mentioned": "",
        "verified_purchase": None,
        "sentiment_label": None,
        "complaint_type": None,
        "source_id": "",
    })
    return row


VALID_CREATOR_PLATFORMS = {"instagram", "tiktok", "youtube", "other"}


def make_creator_row(
    brand: str, source_url: str, caption: str = "",
    creator_handle: str = "", platform: str = "",
    post_date: str = "", partnership_type: str = "unknown",
    creator_followers: Optional[int] = None, video_title: str = "",
    channel_name: str = "", views: Optional[int] = None,
    provider_name: str = "youtube", source_method: str = "api",
) -> Dict[str, Any]:
    """Build a row for creator_posts_candidates."""
    if platform not in VALID_CREATOR_PLATFORMS:
        platform = "other"
    row = _base(brand, "creator_strategy", source_url, provider_name, source_method)
    row.update({
        "post_id": hashlib.md5(source_url.encode()).hexdigest()[:12],
        "creator_handle": creator_handle or "",
        "platform": platform,
        "post_date": post_date or "",
        "partnership_type": partnership_type,
        "caption": clean_text(caption, 800),
        "creator_followers": creator_followers,
        "creator_tier": None,
        "likes": None,
        "comments": None,
        "views": views,
        "sentiment_label": None,
        "video_title": video_title,
        "channel_name": channel_name,
        "source_id": "",
    })
    return row


def make_competitor_row(
    brand: str, platform: str, source_url: str,
    context_text: str = "", title: str = "",
    followers: Optional[int] = None,
    provider_name: str = "trafilatura", source_method: str = "html_extraction",
) -> Dict[str, Any]:
    """Build a row for competitor_platforms_candidates."""
    row = _base(brand, "competitor_benchmark", source_url, provider_name, source_method)
    row.update({
        "platform": platform,
        "followers": followers,
        "following": None,
        "avg_post_frequency": None,
        "creator_tier_mix": None,
        "content_format_mix": None,
        "verified_account": None,
        "title": title,
        "context_text": clean_text(context_text, 500),
        "source_id": "",
    })
    return row


def make_source_queue_row(
    brand: str, target_schema: str, source_url: str,
    reason: str, evidence_type: str = "",
) -> Dict[str, Any]:
    """Build a row for source_queue_for_manual_review."""
    return {
        "brand": brand,
        "target_schema": target_schema,
        "source_url": source_url,
        "reason": reason,
        "evidence_type": evidence_type,
        "collection_date": TODAY,
        "notes": "Requires manual collection or paid tool.",
    }


# ── Deduplication ──────────────────────────────────────────────────────────────

def _row_key(row: Dict, text_field: str) -> str:
    url = str(row.get("source_url", ""))
    text = str(row.get(text_field, ""))[:120]
    return hashlib.md5(f"{url}|{text}".encode()).hexdigest()


def deduplicate(rows: List[Dict], text_field: str = "title") -> List[Dict]:
    """Remove rows with identical (source_url, text[:120]) combinations."""
    seen: set = set()
    out: List[Dict] = []
    for row in rows:
        key = _row_key(row, text_field)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


# ── I/O ───────────────────────────────────────────────────────────────────────

def assign_ids(rows: List[Dict], prefix: str, id_field: str = "source_id") -> List[Dict]:
    """Add sequential IDs to the id_field of each row."""
    for i, row in enumerate(rows):
        if not row.get(id_field):
            row[id_field] = f"{prefix}_{i+1:04d}"
    return rows


def save_candidates(rows: List[Dict], output_path: Path) -> int:
    """Write list of dicts to CSV; creates parent dirs. Returns row count."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return 0
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def log_failure(provider: str, target: str, error: str, failures: List[Dict]) -> None:
    """Append a failure record."""
    failures.append({
        "provider": provider,
        "target": target,
        "error": str(error)[:300],
        "collection_date": TODAY,
    })
    print(f"  [FAIL] {provider} / {target}: {str(error)[:80]}")
