"""Day 1.5 quality patch: domain normalization, creator reclassification, URL resolution, TALA focal baseline."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

INTERIM_IN  = PROJECT_ROOT / "data" / "interim" / "day1"
INTERIM_OUT = PROJECT_ROOT / "data" / "interim" / "day1_5"
TABLES      = PROJECT_ROOT / "outputs" / "tables"
TODAY       = "2026-09-18"
RANDOM_SEED = 42

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
}


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _netloc(url: str) -> str:
    """Return clean netloc (no www.) from a URL."""
    try:
        return urlparse(str(url)).netloc.lower().replace("www.", "").split(":")[0]
    except Exception:
        return ""


def _is_google(url: str) -> bool:
    """Return True if URL resolves to a Google-owned domain."""
    netloc = _netloc(str(url))
    return "google." in netloc or netloc in ("goo.gl", "news.google.com")


def _safe_get(url: str, delay: float = 1.0, timeout: int = 8) -> Optional[requests.Response]:
    """Rate-limited GET with browser headers; returns None on any failure."""
    time.sleep(delay)
    try:
        return requests.get(url, headers=BROWSER_HEADERS, timeout=timeout, allow_redirects=True)
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


def _safe_head(url: str, delay: float = 1.5, timeout: int = 8) -> Optional[int]:
    """Return HTTP status code for URL; tries HEAD then GET on failure."""
    time.sleep(delay)
    try:
        r = requests.head(url, headers=BROWSER_HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code == 405:
            raise ValueError("HEAD not allowed")
        return r.status_code
    except Exception:
        try:
            r = requests.get(url, headers=BROWSER_HEADERS, timeout=timeout, allow_redirects=True)
            return r.status_code
        except Exception:
            return None


def _ddgs_search(query: str, max_results: int = 3, delay: float = 1.5) -> List[Dict]:
    """Search DuckDuckGo; returns result dicts with 'href'. Returns empty list on failure."""
    time.sleep(delay)
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        return list(DDGS().text(query, max_results=max_results))
    except Exception:
        return []


def _extract_canonical_from_html(html: str) -> Optional[str]:
    """Parse og:url, canonical link, or meta-refresh from HTML; returns non-Google URL or None."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # og:url
        tag = soup.find("meta", property="og:url")
        if tag:
            c = tag.get("content", "")
            if c.startswith("http") and not _is_google(c):
                return c

        # canonical link
        tag = soup.find("link", rel="canonical")
        if tag:
            h = tag.get("href", "")
            if h.startswith("http") and not _is_google(h):
                return h

        # meta-refresh (noscript fallback used by Google News)
        for tag in soup.find_all("meta", attrs={"http-equiv": re.compile("refresh", re.I)}):
            content = tag.get("content", "")
            m = re.search(r"url=['\"]?([^'\"\s]+)", content, re.I)
            if m:
                u = m.group(1).strip("'\"")
                if u.startswith("http") and not _is_google(u):
                    return u
    except Exception:
        pass
    return None


def _decode_gnews_b64(gn_url: str) -> Optional[str]:
    """Attempt protobuf base64 decode of Google News article ID to extract publisher URL."""
    import base64
    m = re.search(r"/articles/([A-Za-z0-9_-]{10,})", gn_url)
    if not m:
        return None
    encoded = m.group(1) + "=" * (-len(m.group(1)) % 4)
    try:
        decoded = base64.urlsafe_b64decode(encoded)
        url_m = re.search(rb"https?://[^\x00-\x1f\x7f ]{15,}", decoded)
        if url_m:
            candidate = url_m.group(0).decode("utf-8", errors="replace")
            if not _is_google(candidate):
                return candidate
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 1. official_claims: TALA domain normalization
# ─────────────────────────────────────────────────────────────────────────────

# (canonical_domain, normalization_note)
TALA_DOMAIN_MAP: Dict[str, Tuple[str, str]] = {
    "weartala.com":       ("wearetala.com", "Redirect/typo domain - canonical is wearetala.com"),
    "wearetala.com":      ("wearetala.com", ""),
    "talaactivewear.com": ("wearetala.com", "Alternate domain - verify redirect before citing"),
    "tala.com":           ("wearetala.com", "Alternate domain - verify before citing"),
    "en-us.wearetala.com":("wearetala.com", "Locale subdomain - root is wearetala.com"),
    "uk.wearetala.com":   ("wearetala.com", "Regional subdomain - root is wearetala.com"),
}


def patch_official_claims(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Add canonical_brand_domain, domain_normalization_note, original_source_url."""
    df = df.copy()
    df["original_source_url"] = df["source_url"]

    canonical_domains: List[str] = []
    norm_notes: List[str] = []

    for _, row in df.iterrows():
        domain = _netloc(str(row.get("source_url", "")))
        if domain in TALA_DOMAIN_MAP:
            canon, note = TALA_DOMAIN_MAP[domain]
        else:
            canon = domain
            note = "Third-party domain - press/secondary source, not brand-owned"
        canonical_domains.append(canon)
        norm_notes.append(note)

    df["canonical_brand_domain"] = canonical_domains
    df["domain_normalization_note"] = norm_notes

    changed = sum(
        1 for cd, url in zip(canonical_domains, df["source_url"])
        if cd != _netloc(str(url)) and _netloc(str(url)) in TALA_DOMAIN_MAP
    )
    return df, changed


# ─────────────────────────────────────────────────────────────────────────────
# 2. creator_posts: evidence class reclassification
# ─────────────────────────────────────────────────────────────────────────────

CREATOR_PLATFORM_DOMAINS = {"youtube.com", "instagram.com", "tiktok.com", "twitch.tv", "pinterest.com"}
BRAND_DOMAINS_SET = {
    "wearetala.com", "weartala.com", "adanola.com", "adanolasport.com",
    "girlfriend.com", "girlfriend.co.uk", "oneractive.com", "us.oneractive.com",
}
AFFILIATE_DOMAINS_SET = {
    "active.partners", "ltk.com", "liketk.it", "shareasale.com", "awin1.com",
    "skimlinks.com", "partnerize.com", "nolalane.shop",
}
PRESS_DOMAINS_SET = {
    "gymfluencers.com", "en.wikipedia.org", "businesswire.com", "prnewswire.com",
    "retailgazette.co.uk", "fashionunited.com", "vogue.com", "elle.com",
    "cosmopolitan.com", "thecut.com", "whowhatwear.com", "grazia.co.uk",
    "harpersbazaar.com", "telegraph.co.uk", "theguardian.com",
}

# Domains that indicate a search snippet rather than actual creator content
SEARCH_INDICATOR_DOMAINS = {"duckduckgo.com", "bing.com"}


def _classify_creator_row(url: str) -> Tuple[str, str, bool]:
    """Return (corrected_source_platform, creator_evidence_class, usable_as_creator_post)."""
    domain = _netloc(url)
    if domain in CREATOR_PLATFORM_DOMAINS:
        return domain, "creator_post", True
    if domain in BRAND_DOMAINS_SET:
        return domain, "brand_campaign_page", False
    if domain in AFFILIATE_DOMAINS_SET:
        return domain, "affiliate_platform_page", False
    if domain in PRESS_DOMAINS_SET:
        return domain, "press_creator_mention", False
    if domain in SEARCH_INDICATOR_DOMAINS or not url.startswith("http"):
        return domain or "unknown", "search_lead", False
    return domain or "unknown", "unclear", False


def patch_creator_posts(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Add corrected_source_platform, creator_evidence_class, usable_as_creator_post."""
    df = df.copy()
    df["original_source_url"] = df["source_url"]

    corr_platforms: List[str] = []
    ev_classes: List[str] = []
    usable_as_post: List[bool] = []

    for _, row in df.iterrows():
        csp, cls, uap = _classify_creator_row(str(row.get("source_url", "")))
        corr_platforms.append(csp)
        ev_classes.append(cls)
        usable_as_post.append(uap)

    df["corrected_source_platform"] = corr_platforms
    df["creator_evidence_class"]    = ev_classes
    df["usable_as_creator_post"]    = usable_as_post

    # All rows remain usable for creator strategy (brand/affiliate/press rows give strategy signal too)
    df["usable_for_creator_strategy"] = True

    reclassified = sum(1 for cls in ev_classes if cls != "creator_post")
    return df, reclassified


# ─────────────────────────────────────────────────────────────────────────────
# 3. competitor_platforms: TALA focal platform baseline
# ─────────────────────────────────────────────────────────────────────────────

TALA_CANDIDATE_URLS: Dict[str, str] = {
    "instagram": "https://www.instagram.com/wearetala/",
    "tiktok":    "https://www.tiktok.com/@wearetala",
    "youtube":   "https://www.youtube.com/@wearetala",
    "website":   "https://www.wearetala.com",
}

# Schema-valid platform values for competitor_platforms
VALID_COMPETITOR_PLATFORMS = {"instagram", "tiktok", "youtube", "website"}


def _discover_tala_social_from_homepage() -> Dict[str, str]:
    """Fetch wearetala.com static HTML and extract social platform links."""
    found: Dict[str, str] = {}
    resp = _safe_get("https://www.wearetala.com", delay=2.0)
    if not resp:
        print("  [TALA discovery] Homepage unreachable, using candidate list only")
        return found
    html = resp.text
    patterns = {
        "instagram": r"https?://(?:www\.)?instagram\.com/([A-Za-z0-9_.]+)/?",
        "tiktok":    r"https?://(?:www\.)?tiktok\.com/@([A-Za-z0-9_.]+)/?",
        "youtube":   r"https?://(?:www\.)?youtube\.com/(?:@|c/|channel/)([A-Za-z0-9_.@-]+)/?",
    }
    for platform, pattern in patterns.items():
        m = re.search(pattern, html)
        if m:
            full_url = m.group(0).rstrip("/")
            found[platform] = full_url
            print(f"  [TALA discovery] Found {platform} link in HTML: {full_url}")
    return found


def _make_tala_focal_row(platform: str, url: str, col_template: List[str]) -> Dict:
    """Build a TALA focal baseline row matching the competitor_platforms column structure."""
    row: Dict = {col: None for col in col_template}
    row.update({
        "brand":              "TALA",
        "platform":           platform,
        "followers":          None,
        "following":          None,
        "avg_post_frequency": None,
        "creator_tier_mix":   None,
        "content_format_mix": None,
        "verified_account":   None,
        "source_url":         url,
        "source_platform":    _netloc(url),
        "collection_date":    TODAY,
        "collected_by":       "auto_pipeline_day1_5",
        "evidence_type":      "focal_platform_baseline",
        "citation_ready":     False,
        "synthetic":          False,
        "notes":              (
            "TALA focal brand presence baseline. Not competitor evidence. "
            "Follower counts and post frequency are TBC - verify manually from profile page."
        ),
        "provider_name":      "day1_5_patch",
        "source_method":      "url_verification",
        "source_id":          f"tala_focal_{platform}",
        "title":              f"TALA {platform.capitalize()} official presence",
        "context_text":       "",
        "original_source_url": url,
        # Audit columns
        "needs_manual_review":               True,
        "review_reason":                     "focal_baseline_metrics_tbc",
        "usable_for_analysis":               True,
        "usable_for_rag":                    False,
        "usable_for_creator_strategy":       False,
        "usable_for_quality_responsibility": False,
        "provisional_label":                 "focal_platform_baseline",
    })
    return row


def build_tala_focal_baseline(
    existing_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """Add TALA focal platform rows for each verified URL not already in the dataset."""
    print("  Discovering social links from wearetala.com homepage...")
    discovered = _discover_tala_social_from_homepage()

    # Merge: homepage discovery overrides candidates where found
    candidates = {**TALA_CANDIDATE_URLS, **discovered}
    # Keep only schema-valid platform values
    candidates = {k: v for k, v in candidates.items() if k in VALID_COMPETITOR_PLATFORMS}

    col_template = list(existing_df.columns) if not existing_df.empty else []
    new_rows: List[Dict] = []

    for platform, url in candidates.items():
        # Skip if TALA already has a row for this platform
        if not existing_df.empty:
            existing_tala = existing_df[
                (existing_df["brand"] == "TALA") & (existing_df["platform"] == platform)
            ]
            if not existing_tala.empty:
                print(f"  {platform}: TALA row already present, skipping")
                continue

        print(f"  Verifying {platform}: {url}")
        status = _safe_head(url, delay=2.0)
        if status is not None and status < 400:
            row = _make_tala_focal_row(platform, url, col_template)
            new_rows.append(row)
            print(f"  {platform}: HTTP {status} - row added")
        else:
            print(f"  {platform}: HTTP {status} - unverifiable, skipped")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Align columns to existing
        for col in existing_df.columns:
            if col not in new_df.columns:
                new_df[col] = None
        result = pd.concat([existing_df, new_df[existing_df.columns]], ignore_index=True)
    else:
        result = existing_df.copy()

    # Add original_source_url to existing rows that don't have it
    if "original_source_url" not in result.columns:
        result["original_source_url"] = result["source_url"]

    return result, len(new_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. press_reddit_sources: Google News URL resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_single_url(
    url: str,
    title: str = "",
    brand: str = "",
    use_ddg: bool = True,
) -> Tuple[str, str]:
    """Resolve one URL to a publisher URL. Returns (resolved_url, status)."""
    if not url or not url.startswith("http"):
        return url, "invalid_url"

    if not _is_google(url):
        return url, "resolved"

    # Strategy 1: base64 protobuf decode (fast, no network)
    decoded = _decode_gnews_b64(url)
    if decoded:
        return decoded, "resolved"

    # Strategy 2: follow redirects with browser headers
    try:
        resp = _safe_get(url, delay=1.0)
        if resp is not None:
            final_url = resp.url
            if not _is_google(final_url):
                return final_url, "resolved"
            # Strategy 3: parse canonical / og:url / meta-refresh from HTML
            canonical = _extract_canonical_from_html(resp.text)
            if canonical:
                return canonical, "resolved"
        else:
            # _safe_get returned None (timeout or connection error)
            return url, "timeout"
    except Exception:
        return url, "blocked"

    # Strategy 4: DDG title search fallback
    if use_ddg and title and len(title) >= 10:
        results = _ddgs_search(f'"{title[:60]}" {brand}', max_results=5, delay=1.5)
        for r in results:
            href = r.get("href", "")
            if not href or _is_google(href):
                continue
            domain = _netloc(href)
            if domain not in BRAND_DOMAINS_SET and domain not in CREATOR_PLATFORM_DOMAINS:
                return href, "search_fallback_used"

    return url, "google_news_unresolved"


def patch_press_sources(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int, int]:
    """Add resolved_source_url, resolved_domain, url_resolution_status, original_source_url."""
    df = df.copy()
    df["original_source_url"] = df["source_url"]

    resolved_urls: List[str] = []
    resolved_domains: List[str] = []
    statuses: List[str] = []

    total = len(df)
    n_resolved = 0
    n_unresolved = 0

    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{total}] resolving...")

        url   = str(row.get("source_url", ""))
        title = str(row.get("title", ""))
        brand = str(row.get("brand", ""))

        resolved_url, status = _resolve_single_url(url, title=title, brand=brand)

        if status in ("resolved", "search_fallback_used"):
            n_resolved += 1
        else:
            n_unresolved += 1

        resolved_urls.append(resolved_url)
        resolved_domains.append(_netloc(resolved_url))
        statuses.append(status)

    df["resolved_source_url"]  = resolved_urls
    df["resolved_domain"]      = resolved_domains
    df["url_resolution_status"] = statuses

    return df, n_resolved, n_unresolved


# ─────────────────────────────────────────────────────────────────────────────
# 5. customer_reviews: check for redirect URLs
# ─────────────────────────────────────────────────────────────────────────────

def patch_customer_reviews(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, int, int]:
    """Add URL resolution columns; resolves only non-direct review URLs."""
    df = df.copy()
    df["original_source_url"] = df["source_url"]

    resolved_urls: List[str] = []
    resolved_domains: List[str] = []
    statuses: List[str] = []

    n_resolved = 0
    n_unresolved = 0

    for _, row in df.iterrows():
        url = str(row.get("source_url", ""))
        if _is_google(url):
            snippet = str(row.get("review_text", ""))[:60]
            brand   = str(row.get("brand", ""))
            resolved_url, status = _resolve_single_url(url, title=snippet, brand=brand)
        else:
            resolved_url = url
            status = "resolved"

        if status in ("resolved", "search_fallback_used"):
            n_resolved += 1
        else:
            n_unresolved += 1

        resolved_urls.append(resolved_url)
        resolved_domains.append(_netloc(resolved_url))
        statuses.append(status)

    df["resolved_source_url"]   = resolved_urls
    df["resolved_domain"]       = resolved_domains
    df["url_resolution_status"] = statuses

    return df, n_resolved, n_unresolved


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all Day 1.5 quality patches and save to data/interim/day1_5/."""
    INTERIM_OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("[1/6] Loading cleaned interim files from data/interim/day1/...")
    claims_in      = pd.read_csv(INTERIM_IN / "official_claims_cleaned.csv")
    reviews_in     = pd.read_csv(INTERIM_IN / "customer_reviews_cleaned.csv")
    press_in       = pd.read_csv(INTERIM_IN / "press_reddit_sources_cleaned.csv")
    creators_in    = pd.read_csv(INTERIM_IN / "creator_posts_cleaned.csv")
    competitors_in = pd.read_csv(INTERIM_IN / "competitor_platforms_cleaned.csv")
    print(f"  claims={len(claims_in)}, reviews={len(reviews_in)}, press={len(press_in)}, "
          f"creators={len(creators_in)}, competitors={len(competitors_in)}")

    # ── Patch 1: official_claims domain normalization ─────────────────────────
    print("\n[2/6] Patching official_claims: TALA domain normalization...")
    claims_out, claims_changed = patch_official_claims(claims_in)
    print(f"  Domains mapped to canonical wearetala.com: {claims_changed}")
    norm_dist = claims_out.groupby("canonical_brand_domain").size()
    for domain, n in norm_dist.items():
        print(f"  canonical_brand_domain={domain}: {n} rows")

    # ── Patch 2: creator_posts reclassification ───────────────────────────────
    print("\n[3/6] Patching creator_posts: evidence class reclassification...")
    creators_out, creators_reclassified = patch_creator_posts(creators_in)
    print(f"  Rows reclassified (not creator_post): {creators_reclassified}")
    print("  creator_evidence_class distribution:")
    for cls, n in creators_out["creator_evidence_class"].value_counts().items():
        print(f"    {cls}: {n}")

    # ── Patch 3: TALA focal baseline ──────────────────────────────────────────
    print("\n[4/6] Building TALA focal platform baseline in competitor_platforms...")
    competitors_out, tala_rows_added = build_tala_focal_baseline(competitors_in)
    print(f"  TALA focal rows added: {tala_rows_added}")
    print("  Brands in competitor_platforms after patch:")
    for brand, n in competitors_out["brand"].value_counts().items():
        print(f"    {brand}: {n} rows")

    # ── Patch 4: press URL resolution ─────────────────────────────────────────
    print(f"\n[5/6] Resolving {len(press_in)} Google News URLs in press_reddit_sources...")
    print("  (rate-limited; approx 3-6 min)")
    press_out, press_resolved, press_unresolved = patch_press_sources(press_in)
    print(f"  Resolved: {press_resolved} / {len(press_in)}")
    print(f"  Unresolved: {press_unresolved} / {len(press_in)}")
    print("  url_resolution_status distribution:")
    for status, n in press_out["url_resolution_status"].value_counts().items():
        print(f"    {status}: {n}")

    # ── Patch 5: customer_reviews redirect check ──────────────────────────────
    print(f"\n[6/6] Patching customer_reviews: checking {len(reviews_in)} URLs for redirects...")
    reviews_out, rev_resolved, rev_unresolved = patch_customer_reviews(reviews_in)
    google_review_count = int(reviews_in["source_url"].apply(_is_google).sum())
    print(f"  Google redirect URLs in reviews: {google_review_count}")
    print(f"  Resolved: {rev_resolved} / {len(reviews_in)}")
    print(f"  Unresolved: {rev_unresolved} / {len(reviews_in)}")
    if google_review_count > 0:
        print("  url_resolution_status distribution:")
        for status, n in reviews_out["url_resolution_status"].value_counts().items():
            print(f"    {status}: {n}")

    # ── Save patched files ────────────────────────────────────────────────────
    print("\nSaving patched files to data/interim/day1_5/...")
    outputs = {
        "official_claims_patched.csv":      claims_out,
        "customer_reviews_patched.csv":     reviews_out,
        "press_reddit_sources_patched.csv": press_out,
        "creator_posts_patched.csv":        creators_out,
        "competitor_platforms_patched.csv": competitors_out,
    }
    for fname, df in outputs.items():
        path = INTERIM_OUT / fname
        df.to_csv(path, index=False)
        print(f"  {fname}: {len(df)} rows")

    # ── Summary table ─────────────────────────────────────────────────────────
    def _manual_count(df: pd.DataFrame) -> int:
        return int(df["needs_manual_review"].sum()) if "needs_manual_review" in df.columns else 0

    summary_rows = [
        {
            "file":                     "official_claims_patched.csv",
            "input_rows":               len(claims_in),
            "output_rows":              len(claims_out),
            "rows_changed":             claims_changed,
            "urls_resolved":            0,
            "urls_unresolved":          0,
            "rows_reclassified":        0,
            "tala_baseline_rows_added": 0,
            "manual_review_rows":       _manual_count(claims_out),
        },
        {
            "file":                     "customer_reviews_patched.csv",
            "input_rows":               len(reviews_in),
            "output_rows":              len(reviews_out),
            "rows_changed":             google_review_count,
            "urls_resolved":            rev_resolved,
            "urls_unresolved":          rev_unresolved,
            "rows_reclassified":        0,
            "tala_baseline_rows_added": 0,
            "manual_review_rows":       _manual_count(reviews_out),
        },
        {
            "file":                     "press_reddit_sources_patched.csv",
            "input_rows":               len(press_in),
            "output_rows":              len(press_out),
            "rows_changed":             len(press_in),
            "urls_resolved":            press_resolved,
            "urls_unresolved":          press_unresolved,
            "rows_reclassified":        0,
            "tala_baseline_rows_added": 0,
            "manual_review_rows":       _manual_count(press_out),
        },
        {
            "file":                     "creator_posts_patched.csv",
            "input_rows":               len(creators_in),
            "output_rows":              len(creators_out),
            "rows_changed":             creators_reclassified,
            "urls_resolved":            0,
            "urls_unresolved":          0,
            "rows_reclassified":        creators_reclassified,
            "tala_baseline_rows_added": 0,
            "manual_review_rows":       _manual_count(creators_out),
        },
        {
            "file":                     "competitor_platforms_patched.csv",
            "input_rows":               len(competitors_in),
            "output_rows":              len(competitors_out),
            "rows_changed":             tala_rows_added,
            "urls_resolved":            0,
            "urls_unresolved":          0,
            "rows_reclassified":        0,
            "tala_baseline_rows_added": tala_rows_added,
            "manual_review_rows":       _manual_count(competitors_out),
        },
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLES / "day1_5_quality_patch_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 66)
    print("DAY 1.5 QUALITY PATCH COMPLETE")
    print("=" * 66)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved: {summary_path}")
    print("\nNext: python scripts/validate_data.py")


if __name__ == "__main__":
    main()
