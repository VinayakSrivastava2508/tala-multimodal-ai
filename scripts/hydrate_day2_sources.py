"""Day 2 Task 1: Evidence hydration, corpus construction, and creator enrichment.

Parts A-I as specified in the Day 2 Task 1 brief.
Input:  data/interim/day1_5/*_patched.csv
Output: data/interim/day2/, data/corpora/, outputs/tables/, outputs/figures/
"""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.hydration import (
    HydrationResult,
    compute_evidence_strength,
    hydrate_batch,
    hydrate_one,
    netloc_clean,
    select_best_url,
    token_jaccard,
    DEFAULT_CACHE_DIR,
)

INTERIM_15  = PROJECT_ROOT / "data" / "interim" / "day1_5"
INTERIM_DAY2 = PROJECT_ROOT / "data" / "interim" / "day2"
CORPORA_DIR  = PROJECT_ROOT / "data" / "corpora"
TABLES       = PROJECT_ROOT / "outputs" / "tables"
CACHE_DIR    = DEFAULT_CACHE_DIR
TODAY        = datetime.now(timezone.utc).strftime("%Y-%m-%d")
NOW_ISO      = datetime.now(timezone.utc).isoformat()
RANDOM_SEED  = 42

BRANDS = ["TALA", "Adanola", "Girlfriend Collective", "Oner Active"]

# ── Approved TALA domain family ───────────────────────────────────────────────
TALA_DOMAINS = {"wearetala.com", "eu.tala.co.uk", "talaactivewear.com", "weartala.com"}

# ── Claim category keywords ───────────────────────────────────────────────────
CLAIM_CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "materials":     ["recycled", "organic", "regenerative", "natural fibre", "fabric",
                      "yarn", "plastic bottle", "repreve", "tencel", "nylon", "spandex"],
    "packaging":     ["packaging", "polybag", "compostable", "mailer", "paper bag"],
    "emissions":     ["carbon", "co2", "emission", "net zero", "climate",
                      "greenhouse", "scope", "offset"],
    "manufacturing": ["factory", "supplier", "production", "manufacture",
                      "made in", "facility", "tier 1", "tier 2"],
    "labour":        ["worker", "wage", "living wage", "fair wage", "labour",
                      "labor", "supply chain audit", "worker welfare"],
    "certification": ["certified", "gots", "grs", "b corp", "bluesign",
                      "oeko-tex", "accredited", "verified standard"],
    "circularity":   ["repair", "take-back", "resell", "second-hand", "circular",
                      "longevity", "return programme", "recycle old"],
    "donations":     ["donate", "donation", "charity", "fund", "1%", "tree", "clean ocean"],
    "governance":    ["transparency", "impact report", "governance",
                      "policy", "accountability", "disclose"],
    "other":         [],
}

NUMERICAL_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s*(%|percent|kg|tonne|ton|tree|bottle|piece|litre|liter|year|"
    r"worker|supplier|factory|certification|member)", re.I
)

# ── Customer experience classification ────────────────────────────────────────
EXPERIENCE_KEYWORDS: Dict[str, List[str]] = {
    "quality":           ["quality", "material", "fabric", "pilling", "cheap",
                          "luxurious", "sturdy", "durable", "flimsy", "see-through"],
    "sizing_fit":        ["size", "sizing", "fit", "tight", "loose", "runs small",
                          "runs large", "true to size", "tts", "petite"],
    "durability":        ["durable", "lasted", "worn", "washed", "fading",
                          "stretched", "held up", "fallen apart", "seam", "bobbling"],
    "delivery":          ["delivery", "shipping", "arrived", "dispatch",
                          "wait", "parcel", "tracking", "courier"],
    "returns_refunds":   ["return", "refund", "exchange", "money back", "credit", "sent back"],
    "customer_service":  ["customer service", "support", "help",
                          "contact", "response", "reply", "staff"],
    "price_value":       ["price", "expensive", "affordable", "worth", "value", "cost"],
    "responsibility":    ["sustainable", "recycled", "ethical", "eco",
                          "green", "carbon", "environment", "responsible"],
    "other":             [],
}

POSITIVE_WORDS = {"love", "great", "excellent", "perfect", "amazing", "wonderful",
                  "fantastic", "best", "recommend", "happy", "pleased", "comfortable",
                  "comfy", "soft", "beautiful", "gorgeous", "obsessed", "brilliant", "lovely"}
NEGATIVE_WORDS = {"terrible", "awful", "worst", "hate", "disappointed", "poor", "bad",
                  "broken", "issue", "problem", "fault", "avoid", "disappointing",
                  "nightmare", "waste", "useless", "never", "rubbish", "defect"}

# ── Creator search queries ────────────────────────────────────────────────────
CREATOR_DDG_QUERIES: Dict[str, List[str]] = {
    "TALA": [
        'site:youtube.com "TALA" activewear review',
        'site:youtube.com "TALA" activewear haul',
        'site:youtube.com "wearetala" try on',
        'site:youtube.com "TALA" activewear gifted OR ambassador',
        'site:youtube.com "TALA" gym wear 2024',
    ],
    "Adanola": [
        'site:youtube.com "Adanola" review',
        'site:youtube.com "Adanola" haul try on',
    ],
    "Girlfriend Collective": [
        'site:youtube.com "Girlfriend Collective" review',
        'site:youtube.com "Girlfriend Collective" haul',
    ],
    "Oner Active": [
        'site:youtube.com "Oner Active" review',
        'site:youtube.com "Oner Active" haul try on',
    ],
}

CREATOR_YT_QUERIES: Dict[str, List[str]] = {
    "TALA": [
        "TALA activewear review",
        "TALA activewear haul",
        "wearetala try on",
        "TALA gym wear honest review",
    ],
    "Adanola":              ["Adanola review", "Adanola haul"],
    "Girlfriend Collective":["Girlfriend Collective review", "Girlfriend Collective haul"],
    "Oner Active":          ["Oner Active review", "Oner Active haul"],
}

BRAND_CHANNEL_NAMES = {
    "tala", "wearetala", "adanola", "girlfriend collective",
    "oner active", "oneractive",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def _ddgs(query: str, max_results: int = 5, delay: float = 1.5) -> List[Dict]:
    """DuckDuckGo search; returns list of result dicts."""
    time.sleep(delay)
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        return list(DDGS().text(query, max_results=max_results))
    except Exception:
        return []


def _is_yt_video(url: str) -> bool:
    """Return True if URL is a YouTube video/short."""
    return bool(re.search(r"youtube\.com/watch\?v=|youtube\.com/shorts/|youtu\.be/", url, re.I))


def _is_ig_post(url: str) -> bool:
    """Return True if URL is a direct Instagram post/reel."""
    return bool(re.search(r"instagram\.com/(?:p|reel)/[A-Za-z0-9_-]+", url, re.I))


def _is_tt_post(url: str) -> bool:
    """Return True if URL is a TikTok video post."""
    return bool(re.search(r"tiktok\.com/@[^/]+/video/\d+", url, re.I))


def _is_verified_creator_url(url: str) -> bool:
    """Return True if URL is a direct social post URL (not a profile or search page)."""
    return _is_yt_video(url) or _is_ig_post(url) or _is_tt_post(url)


def _classify_claim_category(text: str) -> str:
    """Keyword-match claim text to a claim category."""
    text_l = text.lower()
    best, best_score = "other", 0
    for cat, kws in CLAIM_CATEGORY_PATTERNS.items():
        score = sum(1 for kw in kws if kw in text_l)
        if score > best_score:
            best, best_score = cat, score
    return best


def _extract_numerical(text: str) -> Tuple[str, str]:
    """Extract first number + unit from text; returns ('', '') if none."""
    m = NUMERICAL_RX.search(text)
    return (m.group(1), m.group(2).lower()) if m else ("", "")


def _classify_experience(text: str) -> str:
    """Keyword-classify review text into an experience category."""
    text_l = text.lower()
    best, best_score = "other", 0
    for cat, kws in EXPERIENCE_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in text_l)
        if score > best_score:
            best, best_score = cat, score
    return best


def _classify_sentiment(text: str) -> str:
    """Simple keyword-count sentiment classifier."""
    words = text.lower().split()
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    if pos > neg * 1.5:
        return "positive"
    if neg > pos * 1.5:
        return "negative"
    if pos > 0 and neg > 0:
        return "mixed"
    return "neutral"


def _category_sentiment(text: str, keywords: List[str]) -> Optional[str]:
    """Sentiment within sentences containing given keywords."""
    sentences = re.split(r"[.!?]", text.lower())
    relevant = [s for s in sentences if any(kw in s for kw in keywords)]
    if not relevant:
        return None
    words = " ".join(relevant).split()
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def _infer_partnership(title: str, desc: str) -> Tuple[str, str]:
    """Infer partnership_type and disclosure_type from title/description text."""
    text = (title + " " + desc).lower()
    if "#ad" in text or "paid partnership" in text or "paid promotion" in text:
        return "paid_sponsorship", "#ad"
    if "#gifted" in text or " gifted " in text or " c/o " in text:
        return "gifted", "#gifted"
    if "ambassador" in text:
        return "ambassador", "ambassador_program"
    if "affiliate" in text or "use my code" in text or "discount code" in text:
        return "affiliate", "affiliate_code"
    if any(kw in text for kw in ("haul", "review", "try on", "try-on", "unboxing", "honest")):
        return "organic", "none"
    return "unclear", "none"


def _yt_api_search(query: str, api_key: str, max_results: int = 10) -> List[Dict]:
    """YouTube Data API v3 video search; returns list of item dicts."""
    try:
        resp = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={"q": query, "key": api_key, "type": "video",
                    "part": "snippet", "maxResults": max_results, "order": "relevance"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("items", [])
        print(f"  [YT API] {resp.status_code} for query: {query[:40]}")
    except Exception as e:
        print(f"  [YT API] error: {e}")
    return []


def _results_to_df(results: List[HydrationResult], source_rows: List[Dict]) -> pd.DataFrame:
    """Merge hydration results back into source rows."""
    hydration_cols = [f.name for f in HydrationResult.__dataclass_fields__.values()]
    rows = []
    for i, res in enumerate(results):
        base = source_rows[i].copy() if i < len(source_rows) else {}
        for col in hydration_cols:
            base[f"h_{col}"] = getattr(res, col, None)
        rows.append(base)
    return pd.DataFrame(rows)


def _make_corpus_row(
    doc_id: str, corpus: str, brand: str,
    source_url: str, source_platform: str, title: str,
    pub_date: str, text: str, strength: str,
    rag_usable: bool, provenance: str,
) -> Dict:
    """Build a standard RAG corpus row."""
    return {
        "document_id":    doc_id,
        "corpus":         corpus,
        "brand":          brand,
        "source_platform": source_platform,
        "source_url":     source_url,
        "document_title": title[:200],
        "publication_date": pub_date,
        "extracted_text": text[:6000],
        "evidence_strength": strength,
        "rag_usable":     rag_usable,
        "retrieved_at":   NOW_ISO,
        "provenance_note": provenance[:300],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PART A + B: Press sources hydration + Google News re-resolution
# ─────────────────────────────────────────────────────────────────────────────

def hydrate_press_sources(press_df: pd.DataFrame) -> pd.DataFrame:
    """Hydrate resolved press URLs; attempt DDG fallback for still-unresolved rows."""
    print(f"  Total rows: {len(press_df)}")
    rows = press_df.to_dict("records")

    results = hydrate_batch(rows, id_field="source_id", cache_dir=CACHE_DIR,
                            delay=1.5, progress_every=20)

    # For google_news_unresolved rows that still failed, try DDG title similarity
    ddg_resolution: Dict[int, Tuple[str, float]] = {}
    for i, (row, res) in enumerate(zip(rows, results)):
        if res.hydration_status in ("failed", "skipped") and row.get("url_resolution_status") == "google_news_unresolved":
            title = str(row.get("title", ""))
            brand = str(row.get("brand", ""))
            if title and len(title) >= 10:
                ddg_results = _ddgs(f'"{title[:60]}" {brand}', max_results=5)
                for r in ddg_results:
                    href = r.get("href", "")
                    ddg_title = r.get("title", "")
                    if not href or "google." in netloc_clean(href):
                        continue
                    sim = token_jaccard(title, ddg_title)
                    if sim >= 0.7:
                        ddg_resolution[i] = (href, sim)
                        break

    # Re-hydrate rows that got DDG fallback URLs
    for i, (ddg_url, sim) in ddg_resolution.items():
        print(f"  [DDG re-resolve] row {i}: sim={sim:.2f} -> {ddg_url[:60]}")
        new_res = hydrate_one(ddg_url, source_id=rows[i].get("source_id", ""), cache_dir=CACHE_DIR, delay=1.0)
        if new_res.hydration_status in ("success", "cached"):
            results[i] = new_res
            results[i].hydration_status = "success"

    df = _results_to_df(results, rows)
    df["resolution_method"] = df.index.map(
        lambda i: "ddg_title_search" if i in ddg_resolution else "direct_url")
    df["title_similarity"] = df.index.map(
        lambda i: ddg_resolution[i][1] if i in ddg_resolution else None)
    df["resolution_confidence"] = df.apply(
        lambda r: "confirmed" if r["h_hydration_status"] in ("success", "cached")
        else "unresolved", axis=1)

    n_success = (df["h_hydration_status"].isin(["success", "cached"])).sum()
    print(f"  Hydration results: {n_success} success/cached, {len(df)-n_success} failed")
    print(f"  DDG fallback resolved: {len(ddg_resolution)}")
    strength_dist = df["h_evidence_strength"].value_counts()
    print(f"  Evidence strength: {strength_dist.to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART C: Official claims hydration + claim extraction
# ─────────────────────────────────────────────────────────────────────────────

def hydrate_claims(claims_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hydrate official claims pages; extract individual claim sentences."""
    print(f"  Total rows: {len(claims_df)}")
    rows = claims_df.to_dict("records")

    # Try TALA domain pages directly before DDG fallback
    enriched_rows = []
    for row in rows:
        domain = netloc_clean(str(row.get("source_url", "")))
        if domain in TALA_DOMAINS:
            enriched_rows.append(row)
        else:
            # Try to find TALA canonical page via DDG
            snippet = str(row.get("claim_text", ""))[:80]
            if snippet:
                ddg_r = _ddgs(f'site:wearetala.com {snippet[:50]}', max_results=3)
                for r in ddg_r:
                    href = r.get("href", "")
                    if href and "wearetala.com" in href:
                        row = {**row, "resolved_source_url": href}
                        break
            enriched_rows.append(row)

    results = hydrate_batch(enriched_rows, id_field="source_id", cache_dir=CACHE_DIR,
                            delay=2.0, progress_every=8)

    hydrated_df = _results_to_df(results, enriched_rows)

    # Extract individual claim records from hydrated text
    expanded_claims = []
    for i, (row, res) in enumerate(zip(enriched_rows, results)):
        text = res.hydrated_text or str(row.get("claim_text", ""))
        brand = str(row.get("brand", "TALA"))
        src_url = res.final_url or str(row.get("source_url", ""))
        src_platform = res.resolved_domain or netloc_clean(src_url)

        if len(text) < 30:
            continue

        # Split into sentences and filter for claim-like content
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) >= 40]
        claim_sentences = [s for s in sentences if any(
            kw in s.lower() for kws in CLAIM_CATEGORY_PATTERNS.values() for kw in kws
        )] or sentences[:5]  # fallback: first 5 sentences if no keyword matches

        for j, sentence in enumerate(claim_sentences[:20]):  # cap at 20 per page
            num_val, unit = _extract_numerical(sentence)
            cat = _classify_claim_category(sentence)
            strength, rag_usable = compute_evidence_strength(
                sentence, res.http_status, has_title=bool(res.page_title))
            expanded_claims.append({
                "claim_id":         f"c_{row.get('source_id','unk')}_{j:02d}",
                "source_id":        str(row.get("source_id", "unk")),
                "brand":            brand,
                "claim_text":       sentence,
                "claim_category":   cat,
                "numerical_claim":  num_val,
                "unit":             unit,
                "target_year":      "",
                "certification":    "",
                "material":         "",
                "geography":        "",
                "source_section":   "",
                "source_quote":     sentence,
                "source_url":       src_url,
                "source_platform":  src_platform,
                "retrieved_at":     NOW_ISO,
                "evidence_strength": strength,
                "rag_usable":       rag_usable,
                "extraction_method": res.extraction_method,
                "page_title":       res.page_title,
                "original_source_url": str(row.get("original_source_url", "")),
                "label_source":     "automated_extraction",
                "requires_human_validation": True,
            })

    expanded_df = pd.DataFrame(expanded_claims)
    n_strong = (expanded_df["evidence_strength"] == "strong").sum() if not expanded_df.empty else 0
    print(f"  Hydrated rows: {(hydrated_df['h_hydration_status'].isin(['success','cached'])).sum()}")
    print(f"  Expanded claim records: {len(expanded_df)} (strong: {n_strong})")
    return hydrated_df, expanded_df


# ─────────────────────────────────────────────────────────────────────────────
# PART D: Customer experience weak labeling (no new network requests)
# ─────────────────────────────────────────────────────────────────────────────

def label_customer_reviews(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based weak labels to customer reviews."""
    df = reviews_df.copy()
    texts = df["review_text"].fillna("").astype(str)

    df["experience_category"]  = texts.apply(_classify_experience)
    df["sentiment_label"]      = texts.apply(_classify_sentiment)
    df["fit_polarity"]         = texts.apply(
        lambda t: _category_sentiment(t, EXPERIENCE_KEYWORDS["sizing_fit"]))
    df["quality_polarity"]     = texts.apply(
        lambda t: _category_sentiment(t, EXPERIENCE_KEYWORDS["quality"]))
    df["durability_polarity"]  = texts.apply(
        lambda t: _category_sentiment(t, EXPERIENCE_KEYWORDS["durability"]))
    df["label_source"]         = "automated_weak_label"
    df["requires_human_validation"] = True

    # Hydrate accessible review pages (Trustpilot may block; try anyway for cache)
    print("  Attempting to hydrate review page URLs (Trustpilot may block)...")
    rows_to_hydrate = df.to_dict("records")
    hydration_results = hydrate_batch(
        rows_to_hydrate, id_field="source_id", cache_dir=CACHE_DIR,
        delay=2.0, progress_every=15)

    df["h_hydration_status"] = [r.hydration_status for r in hydration_results]
    df["h_hydrated_text"]    = [r.hydrated_text for r in hydration_results]
    df["h_text_length"]      = [r.text_length for r in hydration_results]
    df["h_evidence_strength"] = [r.evidence_strength for r in hydration_results]
    df["h_rag_usable"]       = [r.rag_usable for r in hydration_results]
    df["h_page_title"]       = [r.page_title for r in hydration_results]

    # Use hydrated text if it's longer than existing review snippet
    def _best_text(row: pd.Series) -> str:
        hydrated = str(row.get("h_hydrated_text", ""))
        original = str(row.get("review_text", ""))
        return hydrated if len(hydrated) > len(original) else original

    df["best_review_text"] = df.apply(_best_text, axis=1)
    df["best_text_length"] = df["best_review_text"].str.len()

    strength_ok = (df["h_evidence_strength"].isin(["strong", "medium"])).sum()
    print(f"  Weak labels applied: {len(df)} rows")
    print(f"  Trustpilot hydration success (strong/medium): {strength_ok}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART E: Creator post enrichment
# ─────────────────────────────────────────────────────────────────────────────

def _creators_from_ddg(brand: str, queries: List[str]) -> List[Dict]:
    """Run DDG searches and return verified creator post dicts."""
    candidates = []
    existing_urls: set = set()
    for query in queries:
        results = _ddgs(query, max_results=8)
        for r in results:
            href = r.get("href", "")
            title = r.get("title", "")
            body  = r.get("body", "")
            if not href or href in existing_urls:
                continue
            if not _is_verified_creator_url(href):
                continue
            # Require brand mention in title or snippet
            if brand.lower() not in (title + body).lower():
                continue
            existing_urls.add(href)
            platform = ("youtube" if "youtube" in href else
                        "instagram" if "instagram" in href else
                        "tiktok" if "tiktok" in href else "other")
            pt, dt = _infer_partnership(title, body)
            candidates.append({
                "post_url":        href,
                "platform":        platform,
                "brand":           brand,
                "caption_or_description": (title + " " + body)[:400],
                "post_date":       "",
                "creator_handle":  "",
                "creator_name":    "",
                "partnership_type": pt,
                "disclosure_type": dt,
                "personalised_code": "",
                "content_format":  "video" if "youtube" in href or "tiktok" in href else "image_post",
                "creator_tier":    "",
                "view_count":      None,
                "like_count":      None,
                "comment_count":   None,
                "follower_count":  None,
                "metrics_verified_at": "",
                "usable_as_creator_post": True,
                "evidence_strength": "weak",  # will be updated after hydration
                "source_method":   "ddg_search",
                "source_id":       "",
            })
    return candidates


def _creators_from_youtube_api(brand: str, queries: List[str], api_key: str) -> List[Dict]:
    """Search YouTube API and return verified creator post dicts."""
    candidates = []
    existing_urls: set = set()
    for query in queries:
        items = _yt_api_search(query, api_key, max_results=10)
        time.sleep(0.5)
        for item in items:
            snippet = item.get("snippet", {})
            video_id = item.get("id", {}).get("videoId", "")
            if not video_id:
                continue
            url = f"https://www.youtube.com/watch?v={video_id}"
            if url in existing_urls:
                continue
            existing_urls.add(url)

            channel_title = snippet.get("channelTitle", "").lower()
            title = snippet.get("title", "")
            description = snippet.get("description", "")

            # Skip brand-owned channels
            if any(b in channel_title for b in BRAND_CHANNEL_NAMES):
                continue
            # Require brand mention
            if brand.lower() not in (title + description).lower():
                continue

            pt, dt = _infer_partnership(title, description)
            pub_date = snippet.get("publishedAt", "")[:10]
            candidates.append({
                "post_url":          url,
                "platform":          "youtube",
                "brand":             brand,
                "caption_or_description": (title + " " + description)[:400],
                "post_date":         pub_date,
                "creator_handle":    f"youtube.com/channel/{snippet.get('channelId','')}",
                "creator_name":      snippet.get("channelTitle", ""),
                "partnership_type":  pt,
                "disclosure_type":   dt,
                "personalised_code": "",
                "content_format":    "video",
                "creator_tier":      "",
                "view_count":        None,
                "like_count":        None,
                "comment_count":     None,
                "follower_count":    None,
                "metrics_verified_at": "",
                "usable_as_creator_post": True,
                "evidence_strength": "weak",
                "source_method":     "youtube_api",
                "source_id":         "",
            })
    return candidates


def enrich_creator_posts(creators_df: pd.DataFrame) -> pd.DataFrame:
    """Extend creator posts to >=20 verified records via YouTube API and DDG."""
    api_key = os.environ.get("YOUTUBE_API_KEY", "")

    # Start with existing verified posts
    existing = creators_df[creators_df["usable_as_creator_post"] == True].copy()
    print(f"  Starting verified creator posts: {len(existing)}")

    new_candidates: List[Dict] = []
    existing_urls = set(existing["source_url"].dropna().tolist())

    for brand in BRANDS:
        print(f"  Searching: {brand}")
        # YouTube API (fast, authoritative)
        if api_key:
            yt_results = _creators_from_youtube_api(
                brand, CREATOR_YT_QUERIES.get(brand, []), api_key)
            print(f"    YT API: {len(yt_results)} candidates for {brand}")
            new_candidates.extend(yt_results)

        # DDG site-restricted search (supplement)
        ddg_results = _creators_from_ddg(brand, CREATOR_DDG_QUERIES.get(brand, []))
        print(f"    DDG: {len(ddg_results)} candidates for {brand}")
        new_candidates.extend(ddg_results)

    # Deduplicate against existing and between new candidates
    seen_urls = set(existing_urls)
    deduped: List[Dict] = []
    for c in new_candidates:
        url = c.get("post_url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(c)

    print(f"  New unique candidates after dedup: {len(deduped)}")

    # Hydrate YouTube pages to get better text (title + description from page)
    if deduped:
        print("  Hydrating YouTube pages for metadata...")
        for c in deduped:
            url = c.get("post_url", "")
            if "youtube.com" in url or "youtu.be" in url:
                res = hydrate_one(url, source_id=c.get("source_id",""), cache_dir=CACHE_DIR, delay=1.5)
                if res.hydration_status in ("success", "cached") and res.page_title:
                    c["caption_or_description"] = (res.page_title + " " + c.get("caption_or_description",""))[:500]
                    c["evidence_strength"] = res.evidence_strength
                    c["post_date"] = c.get("post_date") or res.publication_date
                else:
                    c["evidence_strength"] = "weak"

    # Assign source_ids
    for i, c in enumerate(deduped):
        c["source_id"] = f"creator_enriched_{i+1:04d}"
        c["source_url"] = c.get("post_url", "")
        c["source_platform"] = netloc_clean(c.get("post_url", ""))
        c["collection_date"] = TODAY
        c["collected_by"] = "auto_pipeline_day2"
        c["evidence_type"] = "creator_strategy"
        c["citation_ready"] = False
        c["synthetic"] = False
        c["needs_manual_review"] = True
        c["review_reason"] = "automated_enrichment_unverified"
        c["usable_for_analysis"] = True
        c["usable_for_rag"] = c.get("evidence_strength") in ("strong", "medium")
        c["usable_for_creator_strategy"] = True
        c["usable_for_quality_responsibility"] = False
        c["provisional_label"] = "creator_post_verified"
        c["partnership_evidence"] = f"inferred from title/description; source={c.get('source_method')}"

    new_df = pd.DataFrame(deduped) if deduped else pd.DataFrame()

    # Align columns before concat
    all_cols = list(set(list(existing.columns) + (list(new_df.columns) if not new_df.empty else [])))
    if not new_df.empty:
        for col in all_cols:
            if col not in new_df.columns:
                new_df[col] = None
            if col not in existing.columns:
                existing[col] = None
        enriched = pd.concat([existing, new_df[all_cols]], ignore_index=True)
    else:
        enriched = existing.copy()

    verified = enriched[enriched["usable_as_creator_post"] == True]
    print(f"  Total verified creator posts: {len(verified)}")
    print("  By brand:")
    for brand, n in verified["brand"].value_counts().items():
        print(f"    {brand}: {n}")
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# PART F: Platform strategy enrichment
# ─────────────────────────────────────────────────────────────────────────────

CHANNEL_ROLE_KEYWORDS = {
    "awareness":      ["launch", "new", "introducing", "meet", "discover"],
    "community":      ["community", "people", "together", "join", "share", "our team"],
    "education":      ["how to", "guide", "tip", "learn", "what is", "why"],
    "conversion":     ["shop", "buy", "discount", "code", "sale", "link in bio"],
    "product_launch": ["new collection", "new drop", "new arrival", "launching", "pre-order"],
    "social_proof":   ["review", "love this", "testimonial", "happy customer"],
    "responsibility": ["sustainable", "recycled", "ethical", "impact", "environment"],
    "customer_support": ["help", "faq", "contact", "question", "dm us"],
}


def _infer_channel_role(text: str) -> Tuple[str, str]:
    """Return (inferred_intent, inference_basis) from page text."""
    text_l = text.lower()
    scores = {role: sum(1 for kw in kws if kw in text_l)
              for role, kws in CHANNEL_ROLE_KEYWORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "unclear", "no_keywords_matched"
    matched = [kw for kw in CHANNEL_ROLE_KEYWORDS[best] if kw in text_l]
    return best, f"keywords: {', '.join(matched[:3])}"


def enrich_platform_strategy(platforms_df: pd.DataFrame) -> pd.DataFrame:
    """Hydrate platform URLs and infer strategic signals."""
    print(f"  Total rows: {len(platforms_df)}")
    rows = platforms_df.to_dict("records")
    results = hydrate_batch(rows, id_field="source_id", url_field="source_url",
                            cache_dir=CACHE_DIR, delay=2.0, progress_every=10)

    df = _results_to_df(results, rows)

    inferred_intents, inference_bases = [], []
    channel_roles, dominant_content_types = [], []
    evidence_urls, evidence_dates, evidence_strengths = [], [], []
    creator_usages = []

    for row, res in zip(rows, results):
        text = res.hydrated_text or ""
        platform = str(row.get("platform", "")).lower()

        intent, basis = _infer_channel_role(text)
        inferred_intents.append(intent)
        inference_bases.append(basis)

        # Channel role mapping by platform context
        if platform == "instagram":
            channel_roles.append("brand_social")
            dominant_content_types.append("image_and_reel")
        elif platform == "tiktok":
            channel_roles.append("brand_social")
            dominant_content_types.append("short_video")
        elif platform == "youtube":
            channel_roles.append("brand_video")
            dominant_content_types.append("long_form_video")
        elif platform == "website":
            channel_roles.append("commerce_and_brand")
            dominant_content_types.append("editorial_and_shop")
        else:
            channel_roles.append("unknown")
            dominant_content_types.append("unknown")

        creator_usage = "unclear"
        if "creator" in text.lower() or "ambassador" in text.lower() or "partner" in text.lower():
            creator_usage = "confirmed_creator_program"
        creator_usages.append(creator_usage)

        evidence_urls.append(res.final_url or str(row.get("source_url", "")))
        evidence_dates.append(res.publication_date or TODAY)
        evidence_strengths.append(res.evidence_strength)

    df["channel_role"]           = channel_roles
    df["dominant_content_type"]  = dominant_content_types
    df["inferred_intent"]        = inferred_intents
    df["inference_basis"]        = inference_bases
    df["creator_usage"]          = creator_usages
    df["evidence_url"]           = evidence_urls
    df["evidence_date"]          = evidence_dates
    df["evidence_strength"]      = evidence_strengths
    df["posting_intent"]         = inferred_intents  # alias
    df["commerce_integration"]   = df["platform"].apply(
        lambda p: "yes" if str(p).lower() in ("instagram", "website") else "unclear")
    df["responsibility_messaging"] = df.apply(
        lambda r: "yes" if "sustain" in str(r.get("h_hydrated_text","")).lower() else "unclear", axis=1)
    df["requires_human_validation"] = True

    strength_dist = df["evidence_strength"].value_counts()
    print(f"  Evidence strength: {strength_dist.to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART G: Build three RAG corpora
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_corpora(
    claims_expanded: pd.DataFrame,
    reviews_df: pd.DataFrame,
    press_hydrated: pd.DataFrame,
    creators_enriched: pd.DataFrame,
    platforms_enriched: pd.DataFrame,
    claims_hydrated: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build official_claims, customer_experience, and creator_strategy RAG corpora."""

    # ── Corpus 1: Official claims ─────────────────────────────────────────────
    # Evidence strength here reflects the SOURCE PAGE (hydrated in Part C), not
    # the length of the individual extracted sentence -- a short but genuine
    # claim sentence lifted from a strongly-hydrated page is strong evidence.
    page_strength: Dict[str, str] = {}
    if claims_hydrated is not None and not claims_hydrated.empty and "source_id" in claims_hydrated.columns:
        page_strength = dict(zip(
            claims_hydrated["source_id"].astype(str),
            claims_hydrated.get("h_evidence_strength", pd.Series(dtype=str)).fillna("unusable"),
        ))

    claims_rows = []
    if not claims_expanded.empty:
        usable = claims_expanded[claims_expanded["rag_usable"] == True].copy()
        if page_strength:
            usable["page_evidence_strength"] = usable["source_id"].astype(str).map(page_strength).fillna(
                usable["evidence_strength"])
        else:
            usable["page_evidence_strength"] = usable["evidence_strength"]
        for i, (_, row) in enumerate(usable.iterrows()):
            strength = str(row.get("page_evidence_strength", row.get("evidence_strength", "weak")))
            claims_rows.append(_make_corpus_row(
                doc_id=f"oc_{i+1:04d}",
                corpus="official_claims",
                brand=str(row.get("brand", "TALA")),
                source_url=str(row.get("source_url", "")),
                source_platform=str(row.get("source_platform", "")),
                title=str(row.get("page_title", row.get("claim_category", ""))),
                pub_date=str(row.get("retrieved_at", "")[:10]),
                text=str(row.get("claim_text", "")),
                strength=strength,
                rag_usable=strength in ("strong", "medium"),
                provenance=f"claim_category={row.get('claim_category')}; extraction={row.get('extraction_method')}; strength_source=source_page",
            ))

    # ── Corpus 2: Customer experience ─────────────────────────────────────────
    ce_rows = []
    # From reviews
    for i, (_, row) in enumerate(reviews_df.iterrows()):
        text = str(row.get("best_review_text", row.get("review_text", "")))
        strength, rag_usable = compute_evidence_strength(
            text, None, has_title=bool(row.get("h_page_title")),
            has_snippet=bool(text))
        if not rag_usable:
            continue
        ce_rows.append(_make_corpus_row(
            doc_id=f"ce_rev_{i+1:04d}",
            corpus="customer_experience",
            brand=str(row.get("brand", "")),
            source_url=str(row.get("source_url", "")),
            source_platform=str(row.get("source_platform", "")),
            title=str(row.get("h_page_title", "Customer Review")),
            pub_date=str(row.get("review_date", "")),
            text=text,
            strength=strength,
            rag_usable=rag_usable,
            provenance=f"category={row.get('experience_category')}; sentiment={row.get('sentiment_label')}",
        ))
    # From press (customer-experience relevant articles)
    if not press_hydrated.empty:
        ce_press = press_hydrated[
            press_hydrated["h_evidence_strength"].isin(["strong", "medium"])
        ]
        for i, (_, row) in enumerate(ce_press.iterrows()):
            text = str(row.get("h_hydrated_text", ""))
            if not text:
                continue
            ce_rows.append(_make_corpus_row(
                doc_id=f"ce_press_{i+1:04d}",
                corpus="customer_experience",
                brand=str(row.get("brand", "")),
                source_url=str(row.get("h_final_url", row.get("source_url", ""))),
                source_platform=str(row.get("h_resolved_domain", row.get("source_platform", ""))),
                title=str(row.get("h_page_title", row.get("title", ""))),
                pub_date=str(row.get("date_published", "")),
                text=text,
                strength=str(row.get("h_evidence_strength", "weak")),
                rag_usable=True,
                provenance="press_article; customer_experience_signal",
            ))

    # ── Corpus 3: Creator strategy ─────────────────────────────────────────────
    # evidence_strength here reflects the Day 2 Task 1B repair pipeline's own
    # classification (src/collectors/creator_identity.py::compute_creator_evidence_strength
    # -- direct URL + verified identity + verified brand link + content text), NOT a
    # recompute from the (400-char-truncated) caption text, which could never reach
    # the 500-char "strong" threshold regardless of how well-verified a post was.
    cs_rows = []
    if not creators_enriched.empty:
        verified = creators_enriched[creators_enriched["usable_as_creator_post"] == True]
        for i, (_, row) in enumerate(verified.iterrows()):
            text = str(row.get("caption_or_description", "") or "")
            if text.lower() == "nan":
                text = ""
            if not text.strip():
                # some verified posts (e.g. Day 1 baseline rows) carry brand evidence
                # in video_title with no separate caption -- fall back so a genuinely
                # content-backed row doesn't end up with an empty extracted_text.
                fallback_title = str(row.get("video_title", "") or "")
                text = "" if fallback_title.lower() == "nan" else fallback_title
            strength = str(row.get("evidence_strength", "weak")) if pd.notna(row.get("evidence_strength")) else "weak"
            rag_usable = strength in ("strong", "medium")
            cs_rows.append(_make_corpus_row(
                doc_id=f"cs_{i+1:04d}",
                corpus="creator_strategy",
                brand=str(row.get("brand", "")),
                source_url=str(row.get("direct_post_url") or row.get("source_url", "")),
                source_platform=str(row.get("platform", row.get("source_platform", ""))),
                title=str(row.get("creator_name", "Creator Post")),
                pub_date=str(row.get("post_date", "")),
                text=text,
                strength=strength,
                rag_usable=rag_usable,
                provenance=(
                    f"partnership_type={row.get('partnership_type')}; "
                    f"creator_identity_source={row.get('creator_identity_source')}; "
                    f"brand_link_confidence={row.get('brand_link_confidence')}"
                ),
            ))

    claims_corpus = pd.DataFrame(claims_rows)
    ce_corpus     = pd.DataFrame(ce_rows)
    cs_corpus     = pd.DataFrame(cs_rows)

    print(f"  official_claims_corpus:      {len(claims_corpus)} records")
    print(f"  customer_experience_corpus:  {len(ce_corpus)} records")
    print(f"  creator_strategy_corpus:     {len(cs_corpus)} records")
    return claims_corpus, ce_corpus, cs_corpus


# ─────────────────────────────────────────────────────────────────────────────
# PART H: Quality gates
# ─────────────────────────────────────────────────────────────────────────────

def check_quality_gates(
    claims_corpus: pd.DataFrame,
    ce_corpus: pd.DataFrame,
    cs_corpus: pd.DataFrame,
    creators_enriched: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate go/no-go quality thresholds. Returns gate report DataFrame."""
    def _strong(df: pd.DataFrame) -> int:
        return int((df["evidence_strength"] == "strong").sum()) if not df.empty else 0

    def _strong_medium(df: pd.DataFrame) -> int:
        return int(df["evidence_strength"].isin(["strong", "medium"]).sum()) if not df.empty else 0

    def _rag_usable(df: pd.DataFrame) -> int:
        return int(df["rag_usable"].sum()) if not df.empty else 0

    verified_creators = creators_enriched[
        creators_enriched["usable_as_creator_post"] == True
    ] if not creators_enriched.empty else pd.DataFrame()
    n_verified = len(verified_creators)
    verified_strong_med = _strong_medium(verified_creators) if not verified_creators.empty else 0
    brands_in_creator = verified_creators["brand"].nunique() if not verified_creators.empty else 0
    total_rag = _rag_usable(claims_corpus) + _rag_usable(ce_corpus) + _rag_usable(cs_corpus)

    # Check no search-result URLs as verified creator posts
    no_search_as_creator = 0
    if not verified_creators.empty and "source_method" in verified_creators.columns:
        search_urls = verified_creators[
            verified_creators.get("source_url", pd.Series()).str.contains(
                r"google\.|duckduckgo\.", na=False, regex=True)
        ]
        no_search_as_creator = len(search_urls)

    # Check no weak rag_usable
    weak_rag = 0
    for corpus in [claims_corpus, ce_corpus, cs_corpus]:
        if not corpus.empty and "rag_usable" in corpus.columns:
            weak_rag += int(
                ((corpus["evidence_strength"] == "weak") & (corpus["rag_usable"] == True)).sum()
            )

    gates = [
        {
            "gate":       "official_claims_strong",
            "threshold":  15,
            "actual":     _strong(claims_corpus),
            "pass":       _strong(claims_corpus) >= 15,
        },
        {
            "gate":       "customer_experience_strong_medium",
            "threshold":  25,
            "actual":     _strong_medium(ce_corpus),
            "pass":       _strong_medium(ce_corpus) >= 25,
        },
        {
            "gate":       "creator_strategy_verified_strong_medium",
            "threshold":  20,
            "actual":     verified_strong_med,
            "pass":       verified_strong_med >= 20,
        },
        {
            "gate":       "total_rag_usable",
            "threshold":  60,
            "actual":     total_rag,
            "pass":       total_rag >= 60,
        },
        {
            "gate":       "brands_in_creator_evidence",
            "threshold":  3,
            "actual":     brands_in_creator,
            "pass":       brands_in_creator >= 3,
        },
        {
            "gate":       "no_search_url_as_creator_post",
            "threshold":  0,
            "actual":     no_search_as_creator,
            "pass":       no_search_as_creator == 0,
        },
        {
            "gate":       "no_weak_record_as_rag_usable",
            "threshold":  0,
            "actual":     weak_rag,
            "pass":       weak_rag == 0,
        },
    ]
    df = pd.DataFrame(gates)
    df["status"] = df["pass"].map({True: "PASS", False: "FAIL"})
    df["overall"] = "GO" if df["pass"].all() else "NO-GO FOR MODELLING"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PART I: Audit outputs
# ─────────────────────────────────────────────────────────────────────────────

def generate_audit_outputs(
    press_hydrated: pd.DataFrame,
    claims_hydrated: pd.DataFrame,
    claims_expanded: pd.DataFrame,
    reviews_labeled: pd.DataFrame,
    creators_enriched: pd.DataFrame,
    platforms_enriched: pd.DataFrame,
    claims_corpus: pd.DataFrame,
    ce_corpus: pd.DataFrame,
    cs_corpus: pd.DataFrame,
    gates_df: pd.DataFrame,
) -> None:
    """Save all Day 2 audit tables to outputs/tables/."""

    def _strength_counts(df: pd.DataFrame, col: str = "h_evidence_strength") -> Dict:
        if df.empty or col not in df.columns:
            return {"strong": 0, "medium": 0, "weak": 0, "unusable": 0}
        vc = df[col].value_counts().to_dict()
        return {s: vc.get(s, 0) for s in ("strong", "medium", "weak", "unusable")}

    # day2_hydration_summary
    hydration_rows = []
    for name, df, url_col, status_col, strength_col in [
        ("press_reddit",    press_hydrated,    "source_url", "h_hydration_status", "h_evidence_strength"),
        ("official_claims", claims_hydrated,   "source_url", "h_hydration_status", "h_evidence_strength"),
        ("customer_reviews",reviews_labeled,   "source_url", "h_hydration_status", "h_evidence_strength"),
        ("competitor_platforms", platforms_enriched, "source_url", "h_hydration_status", "evidence_strength"),
    ]:
        if df.empty:
            continue
        attempted = len(df)
        success   = int(df[status_col].isin(["success", "cached"]).sum()) if status_col in df.columns else 0
        failed    = attempted - success
        sc = _strength_counts(df, strength_col)
        hydration_rows.append({
            "source_type":      name,
            "attempted":        attempted,
            "hydrated_success": success,
            "hydrated_failed":  failed,
            "strong":           sc["strong"],
            "medium":           sc["medium"],
            "weak":             sc["weak"],
            "unusable":         sc["unusable"],
        })

    pd.DataFrame(hydration_rows).to_csv(TABLES / "day2_hydration_summary.csv", index=False)

    # day2_corpus_summary
    corpus_rows = []
    for corpus_name, df in [
        ("official_claims", claims_corpus),
        ("customer_experience", ce_corpus),
        ("creator_strategy", cs_corpus),
    ]:
        if df.empty:
            corpus_rows.append({"corpus": corpus_name, "total": 0,
                                 "strong": 0, "medium": 0, "weak": 0, "rag_usable": 0})
            continue
        sc = {s: int((df["evidence_strength"] == s).sum()) for s in ("strong", "medium", "weak")}
        rag = int(df["rag_usable"].sum()) if "rag_usable" in df.columns else 0
        brands = df["brand"].value_counts().to_dict() if "brand" in df.columns else {}
        corpus_rows.append({
            "corpus":    corpus_name,
            "total":     len(df),
            "strong":    sc["strong"],
            "medium":    sc["medium"],
            "weak":      sc["weak"],
            "rag_usable": rag,
            "brands":    str(brands),
        })
    pd.DataFrame(corpus_rows).to_csv(TABLES / "day2_corpus_summary.csv", index=False)

    # day2_creator_coverage
    if not creators_enriched.empty:
        verified = creators_enriched[creators_enriched["usable_as_creator_post"] == True]
        creator_coverage = (
            verified.groupby(["brand", "platform"])
            .agg(
                count=("source_url", "count"),
                strong_medium=(
                    "evidence_strength",
                    lambda x: int(x.isin(["strong", "medium"]).sum()),
                ),
                yt_api=("source_method", lambda x: int((x == "youtube_api").sum())),
                ddg=("source_method", lambda x: int((x == "ddg_search").sum())),
            )
            .reset_index()
        )
        creator_coverage.to_csv(TABLES / "day2_creator_coverage.csv", index=False)

    # day2_platform_coverage
    if not platforms_enriched.empty:
        plat_cols = ["brand", "platform", "evidence_strength", "inferred_intent",
                     "creator_usage", "evidence_url"]
        plat_out = platforms_enriched[[c for c in plat_cols if c in platforms_enriched.columns]]
        plat_out.to_csv(TABLES / "day2_platform_coverage.csv", index=False)

    # day2_quality_gate_report
    gates_df.to_csv(TABLES / "day2_quality_gate_report.csv", index=False)

    print("  Audit tables saved to outputs/tables/")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    INTERIM_DAY2.mkdir(parents=True, exist_ok=True)
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("\n[LOAD] Reading day1_5 patched files...")
    claims_df      = pd.read_csv(INTERIM_15 / "official_claims_patched.csv")
    reviews_df     = pd.read_csv(INTERIM_15 / "customer_reviews_patched.csv")
    press_df       = pd.read_csv(INTERIM_15 / "press_reddit_sources_patched.csv")
    creators_df    = pd.read_csv(INTERIM_15 / "creator_posts_patched.csv")
    competitors_df = pd.read_csv(INTERIM_15 / "competitor_platforms_patched.csv")
    print(f"  claims={len(claims_df)}, reviews={len(reviews_df)}, press={len(press_df)}, "
          f"creators={len(creators_df)}, competitors={len(competitors_df)}")

    # ── Part A+B: Press ────────────────────────────────────────────────────────
    print("\n[PART A+B] Hydrating press sources + Google News re-resolution...")
    press_hydrated = hydrate_press_sources(press_df)
    press_hydrated.to_csv(INTERIM_DAY2 / "hydrated_press.csv", index=False)

    # ── Part C: Claims ─────────────────────────────────────────────────────────
    print("\n[PART C] Hydrating official claims + extracting claim records...")
    claims_hydrated, claims_expanded = hydrate_claims(claims_df)
    claims_hydrated.to_csv(INTERIM_DAY2 / "hydrated_claims.csv", index=False)
    if not claims_expanded.empty:
        claims_expanded.to_csv(INTERIM_DAY2 / "claims_expanded.csv", index=False)

    # ── Part D: Reviews ────────────────────────────────────────────────────────
    print("\n[PART D] Labeling customer reviews + attempting Trustpilot hydration...")
    reviews_labeled = label_customer_reviews(reviews_df)
    reviews_labeled.to_csv(INTERIM_DAY2 / "reviews_labeled.csv", index=False)

    # ── Part E: Creator enrichment ─────────────────────────────────────────────
    print("\n[PART E] Enriching creator posts...")
    creators_enriched = enrich_creator_posts(creators_df)
    creators_enriched.to_csv(INTERIM_DAY2 / "creator_posts_enriched.csv", index=False)

    # ── Part F: Platform strategy ──────────────────────────────────────────────
    print("\n[PART F] Enriching platform strategy data...")
    platforms_enriched = enrich_platform_strategy(competitors_df)
    platforms_enriched.to_csv(INTERIM_DAY2 / "platform_strategy_enriched.csv", index=False)

    # ── Part G: RAG corpora ────────────────────────────────────────────────────
    print("\n[PART G] Building RAG corpora...")
    claims_corpus, ce_corpus, cs_corpus = build_rag_corpora(
        claims_expanded, reviews_labeled, press_hydrated,
        creators_enriched, platforms_enriched,
        claims_hydrated=claims_hydrated,
    )
    claims_corpus.to_csv(CORPORA_DIR / "official_claims_corpus.csv", index=False)
    ce_corpus.to_csv(CORPORA_DIR / "customer_experience_corpus.csv", index=False)
    cs_corpus.to_csv(CORPORA_DIR / "creator_strategy_corpus.csv", index=False)

    # ── Part H: Quality gates ──────────────────────────────────────────────────
    print("\n[PART H] Checking quality gates...")
    gates_df = check_quality_gates(claims_corpus, ce_corpus, cs_corpus, creators_enriched)
    print(gates_df[["gate", "threshold", "actual", "status"]].to_string(index=False))
    overall = gates_df["overall"].iloc[0]
    print(f"\n  Overall: {overall}")

    # ── Part I: Audit outputs ──────────────────────────────────────────────────
    print("\n[PART I] Generating audit outputs...")
    generate_audit_outputs(
        press_hydrated, claims_hydrated, claims_expanded, reviews_labeled,
        creators_enriched, platforms_enriched, claims_corpus, ce_corpus,
        cs_corpus, gates_df,
    )

    # ── hydrated_sources.csv (combined summary) ───────────────────────────────
    hydrated_sources_frames = []
    for src_name, df, text_col, strength_col, status_col in [
        ("press",    press_hydrated,    "h_hydrated_text", "h_evidence_strength", "h_hydration_status"),
        ("claims",   claims_hydrated,   "h_hydrated_text", "h_evidence_strength", "h_hydration_status"),
        ("reviews",  reviews_labeled,   "h_hydrated_text", "h_evidence_strength", "h_hydration_status"),
        ("platforms",platforms_enriched,"h_hydrated_text", "evidence_strength",   "h_hydration_status"),
    ]:
        if df.empty:
            continue
        tmp = pd.DataFrame({
            "source_type":       src_name,
            "source_id":         df.get("source_id", pd.Series(range(len(df)))),
            "brand":             df.get("brand", ""),
            "source_url":        df.get("source_url", ""),
            "hydration_status":  df.get(status_col, "unknown"),
            "evidence_strength": df.get(strength_col, "unusable"),
            "rag_usable":        df.get("h_rag_usable", False),
            "text_length":       df.get("h_text_length", 0),
            "retrieved_at":      NOW_ISO,
        })
        hydrated_sources_frames.append(tmp)

    if hydrated_sources_frames:
        pd.concat(hydrated_sources_frames, ignore_index=True).to_csv(
            INTERIM_DAY2 / "hydrated_sources.csv", index=False)

    print("\n" + "=" * 66)
    print("DAY 2 HYDRATION COMPLETE")
    print("=" * 66)
    print(f"  Interim files: {INTERIM_DAY2}")
    print(f"  RAG corpora:   {CORPORA_DIR}")
    print(f"  Audit tables:  {TABLES}")
    print(f"  Overall status: {overall}")
    print("\nNext: python scripts/validate_data.py")

    return 0 if overall == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
