"""Web hydration: fetch public pages, extract text, score evidence strength."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.robotparser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
BOT_UA = "tala-research-bot/0.2 (academic; SPJIMR ANA526-PPM; non-commercial)"
BROWSER_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.5",
}
BOT_HEADERS = {
    "User-Agent": BOT_UA,
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-GB,en;q=0.5",
}

STRONG_MIN = 500
MEDIUM_MIN = 150

# JS-heavy domains where simple requests won't get page content
JS_HEAVY_DOMAINS = {"instagram.com", "tiktok.com", "news.google.com"}
# Domains known to block bots; attempt but expect failure
BOT_BLOCKED_DOMAINS = {"instagram.com", "tiktok.com", "facebook.com", "linkedin.com", "trustpilot.com"}


@dataclass
class HydrationResult:
    """One row of hydration output per source URL."""
    source_id: str = ""
    input_url: str = ""
    final_url: str = ""
    canonical_url: str = ""
    resolved_domain: str = ""
    page_title: str = ""
    publication_date: str = ""
    author: str = ""
    hydrated_text: str = ""
    text_length: int = 0
    extraction_method: str = ""
    http_status: Optional[int] = None
    hydration_status: str = "pending"   # success | failed | cached | skipped | blocked
    hydration_timestamp_utc: str = ""
    failure_reason: str = ""
    evidence_strength: str = "unusable" # strong | medium | weak | unusable
    rag_usable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return dataclass as plain dict."""
        return asdict(self)


# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_path(url: str, cache_dir: Path) -> Path:
    """Return shard/hash.json path for a URL."""
    key = hashlib.sha256(url.encode()).hexdigest()
    return cache_dir / key[:2] / f"{key}.json"


def load_cache(url: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Optional[HydrationResult]:
    """Return cached HydrationResult if present, else None."""
    p = _cache_path(url, cache_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return HydrationResult(**{k: data.get(k) for k in HydrationResult.__dataclass_fields__})
    except Exception:
        return None


def save_cache(result: HydrationResult, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Persist HydrationResult to JSON cache."""
    if not result.input_url:
        return
    p = _cache_path(result.input_url, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.to_dict(), ensure_ascii=False, default=str), encoding="utf-8")


# ── URL utilities ─────────────────────────────────────────────────────────────

def netloc_clean(url: str) -> str:
    """Return www-stripped domain from URL."""
    try:
        return urlparse(str(url)).netloc.lower().replace("www.", "").split(":")[0]
    except Exception:
        return ""


def select_best_url(row: Dict) -> str:
    """Return best URL from a row: resolved_source_url > source_url > original_source_url.

    Skips Google-domain URLs unless no other option exists.
    """
    for field in ("resolved_source_url", "source_url", "original_source_url"):
        val = str(row.get(field, "")).strip()
        if val and val.startswith("http") and "google." not in netloc_clean(val):
            return val
    # Fall back to source_url even if Google
    val = str(row.get("source_url", "")).strip()
    return val if val.startswith("http") else ""


def can_fetch_robots(url: str, ua: str = BOT_UA) -> bool:
    """Return True if robots.txt permits fetching this URL with our user-agent."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(ua, url)
    except Exception:
        return True


# ── HTTP fetch ────────────────────────────────────────────────────────────────

def _fetch_once(
    url: str, use_browser_ua: bool = True, timeout: int = 12,
) -> Tuple[Optional[requests.Response], str]:
    """Single GET; returns (response, failure_reason)."""
    headers = BROWSER_HEADERS if use_browser_ua else BOT_HEADERS
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        return resp, ""
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.TooManyRedirects:
        return None, "too_many_redirects"
    except requests.exceptions.ConnectionError:
        return None, "connection_error"
    except Exception as e:
        return None, str(e)[:80]


def fetch_with_retry(
    url: str,
    retries: int = 2,
    base_delay: float = 1.5,
    use_browser_ua: bool = True,
) -> Tuple[Optional[requests.Response], str]:
    """GET with exponential backoff; returns (response, failure_reason)."""
    last_reason = ""
    for attempt in range(retries + 1):
        if attempt > 0:
            time.sleep(base_delay * (2 ** attempt))
        resp, reason = _fetch_once(url, use_browser_ua=use_browser_ua)
        if resp is not None:
            return resp, ""
        last_reason = reason
    return None, last_reason


# ── Text extraction ───────────────────────────────────────────────────────────

def _try_trafilatura(html: str, url: str = "") -> Tuple[str, str]:
    """Extract with trafilatura; returns (text, 'trafilatura') or ('', '')."""
    try:
        import trafilatura
        text = trafilatura.extract(
            html, url=url or None,
            include_links=False, include_images=False,
            include_tables=True, no_fallback=False,
        )
        if text and len(text.strip()) >= MEDIUM_MIN:
            return text.strip(), "trafilatura"
    except Exception:
        pass
    return "", ""


def _try_bs4(html: str) -> Tuple[str, str]:
    """Extract with BeautifulSoup; returns (text, 'bs4') or ('', '')."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        raw = re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
        if len(raw) >= MEDIUM_MIN:
            return raw[:8000], "bs4"
    except Exception:
        pass
    return "", ""


def extract_main_text(html: str, url: str = "") -> Tuple[str, str]:
    """Extract article text: trafilatura first, then bs4. Returns (text, method)."""
    text, method = _try_trafilatura(html, url)
    if text:
        return text[:8000], method
    return _try_bs4(html)


def extract_page_metadata(html: str) -> Dict[str, str]:
    """Extract page_title, publication_date, author, canonical_url from HTML."""
    meta = {"page_title": "", "publication_date": "", "author": "", "canonical_url": ""}
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # Title (og:title preferred)
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            meta["page_title"] = og_title["content"].strip()[:200]
        elif soup.title:
            meta["page_title"] = soup.title.get_text().strip()[:200]

        # Canonical
        c = soup.find("link", rel="canonical")
        if c and str(c.get("href", "")).startswith("http"):
            meta["canonical_url"] = c["href"]

        # Publication date
        for key, attr in [
            ("property", "article:published_time"),
            ("name", "pubdate"),
            ("itemprop", "datePublished"),
            ("name", "date"),
        ]:
            tag = soup.find("meta", {key: attr})
            if tag and tag.get("content"):
                meta["publication_date"] = str(tag["content"])[:20]
                break
        if not meta["publication_date"]:
            t = soup.find("time")
            if t:
                meta["publication_date"] = (t.get("datetime") or t.get_text())[:20]

        # Author
        for key, attr in [
            ("property", "article:author"),
            ("name", "author"),
            ("itemprop", "author"),
        ]:
            tag = soup.find("meta", {key: attr})
            if tag and tag.get("content"):
                meta["author"] = str(tag["content"]).strip()[:100]
                break
    except Exception:
        pass
    return meta


# ── Evidence scoring ──────────────────────────────────────────────────────────

def compute_evidence_strength(
    text: str,
    http_status: Optional[int],
    has_title: bool = False,
    has_snippet: bool = False,
) -> Tuple[str, bool]:
    """Return (evidence_strength, rag_usable).

    strong:   >= 500 chars direct page text
    medium:   150-499 chars, or reliable title + snippet >= 50 chars
    weak:     any text < 150 chars or snippet-only
    unusable: no text, or HTTP >= 400
    """
    if http_status is not None and http_status >= 400:
        return "unusable", False
    length = len(text.strip()) if text else 0
    if length >= STRONG_MIN:
        return "strong", True
    if length >= MEDIUM_MIN:
        return "medium", True
    if has_title and has_snippet and length >= 50:
        return "medium", True
    if length > 0 or has_title:
        return "weak", False
    return "unusable", False


# ── Title similarity (no embeddings) ─────────────────────────────────────────

def token_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity for title matching."""
    STOPS = {"a", "an", "the", "in", "of", "for", "on", "at", "to", "by", "is", "are",
             "was", "were", "be", "been", "and", "or", "with", "as", "from"}
    ta = set(re.sub(r"[^\w]", " ", a.lower()).split()) - STOPS
    tb = set(re.sub(r"[^\w]", " ", b.lower()).split()) - STOPS
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── Main hydration ────────────────────────────────────────────────────────────

def hydrate_one(
    url: str,
    source_id: str = "",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    delay: float = 1.5,
) -> HydrationResult:
    """Fetch URL, extract text, score. Returns HydrationResult (cached on disk)."""
    ts = datetime.now(timezone.utc).isoformat()
    result = HydrationResult(source_id=source_id, input_url=url, hydration_timestamp_utc=ts)

    if not url or not url.startswith("http"):
        result.hydration_status = "skipped"
        result.failure_reason = "invalid_url"
        return result

    # Cache hit
    cached = load_cache(url, cache_dir)
    if cached:
        cached.hydration_status = "cached"
        return cached

    time.sleep(delay)

    resp, failure = fetch_with_retry(url, retries=2, base_delay=delay)
    if resp is None:
        result.hydration_status = "failed"
        result.failure_reason = failure
        save_cache(result, cache_dir)
        return result

    result.http_status = resp.status_code
    result.final_url = resp.url
    result.resolved_domain = netloc_clean(resp.url)

    if resp.status_code >= 400:
        result.hydration_status = "failed"
        result.failure_reason = f"http_{resp.status_code}"
        strength, usable = compute_evidence_strength("", resp.status_code)
        result.evidence_strength = strength
        result.rag_usable = usable
        save_cache(result, cache_dir)
        return result

    html = resp.text
    text, method = extract_main_text(html, url=resp.url)
    meta = extract_page_metadata(html)

    result.hydrated_text = text
    result.text_length = len(text)
    result.extraction_method = method or "requests_no_text"
    result.page_title = meta["page_title"]
    result.publication_date = meta["publication_date"]
    result.author = meta["author"]
    result.canonical_url = meta["canonical_url"]

    strength, usable = compute_evidence_strength(
        text, resp.status_code,
        has_title=bool(meta["page_title"]),
    )
    result.evidence_strength = strength
    result.rag_usable = usable
    result.hydration_status = "success"

    save_cache(result, cache_dir)
    return result


def hydrate_batch(
    rows: List[Dict],
    id_field: str = "source_id",
    url_field: Optional[str] = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    delay: float = 1.5,
    progress_every: int = 10,
) -> List[HydrationResult]:
    """Hydrate a list of row dicts. url_field=None uses select_best_url(). Returns results list."""
    results: List[HydrationResult] = []
    total = len(rows)
    for i, row in enumerate(rows):
        if i == 0 or (i + 1) % progress_every == 0:
            print(f"  [{i+1}/{total}]")
        url = str(row.get(url_field, "")) if url_field else select_best_url(row)
        sid = str(row.get(id_field, f"row_{i}"))
        results.append(hydrate_one(url, source_id=sid, cache_dir=cache_dir, delay=delay))
    return results
