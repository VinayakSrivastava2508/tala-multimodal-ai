"""Creator-post URL classification, identity resolution, brand-link verification,
and partnership classification for Day 2 Task 1B creator-evidence repair.

Pure classification/verification logic lives here so it can be unit-tested without
any network access. The one function that makes a live call (fetch_youtube_oembed)
takes an injectable `get` callable so tests can supply a mock.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OEMBED_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "youtube_oembed"

CONFIDENCE_VALUES = ("high", "medium", "low", "unresolved")

# ── URL classification ────────────────────────────────────────────────────────

YOUTUBE_WATCH_RX = re.compile(r"^https?://(?:www\.)?youtube\.com/watch\b", re.I)
YOUTUBE_SHORTS_RX = re.compile(r"^https?://(?:www\.)?youtube\.com/shorts/([\w-]{6,})", re.I)
YOUTUBE_YOUTU_BE_RX = re.compile(r"^https?://youtu\.be/([\w-]{6,})", re.I)
YOUTUBE_SEARCH_RX = re.compile(r"^https?://(?:www\.)?youtube\.com/results\b", re.I)
YOUTUBE_PLAYLIST_RX = re.compile(r"^https?://(?:www\.)?youtube\.com/playlist\b", re.I)
YOUTUBE_CHANNEL_RX = re.compile(
    r"^https?://(?:www\.)?youtube\.com/(?:channel/|c/|@)", re.I
)

TIKTOK_VIDEO_RX = re.compile(r"^https?://(?:www\.)?tiktok\.com/@([\w.\-]+)/video/(\d+)", re.I)

INSTAGRAM_POST_RX = re.compile(
    r"^https?://(?:www\.)?instagram\.com/(p|reel|reels|tv)/([A-Za-z0-9_-]+)", re.I
)
INSTAGRAM_PROGRAMME_RX = re.compile(
    r"instagram\.com/(?:popular/)?[\w-]*(?:program|ambassador|affiliate|campaign)", re.I
)

SEARCH_ENGINE_RX = re.compile(r"google\.|duckduckgo\.|bing\.", re.I)


def _youtube_video_id(url: str) -> Optional[str]:
    """Extract the `v` query param from a youtube.com/watch URL."""
    try:
        qs = parse_qs(urlparse(url).query)
        vals = qs.get("v")
        return vals[0] if vals else None
    except Exception:
        return None


def classify_url(url: str) -> Dict[str, Optional[str]]:
    """Classify a creator-post URL.

    Returns a dict with url_class, direct_post_url, direct_post_id,
    url_validation_status ('valid' | 'invalid' | 'lead_only'), and invalidation_reason
    (empty string when valid).
    """
    result = {
        "url_class": "unknown",
        "direct_post_url": "",
        "direct_post_id": "",
        "url_validation_status": "invalid",
        "invalidation_reason": "unrecognised_url",
    }
    if not url or not str(url).startswith("http"):
        result["invalidation_reason"] = "missing_url"
        return result

    u = str(url).strip()

    if SEARCH_ENGINE_RX.search(u):
        result.update(url_class="search_result", invalidation_reason="search_engine_result_page")
        return result

    # ── YouTube ──────────────────────────────────────────────────────────────
    if "youtube.com" in u or "youtu.be" in u:
        if YOUTUBE_SEARCH_RX.match(u):
            result.update(url_class="youtube_search", invalidation_reason="youtube_search_page")
            return result
        if YOUTUBE_PLAYLIST_RX.match(u):
            result.update(
                url_class="youtube_playlist",
                url_validation_status="lead_only",
                invalidation_reason="youtube_playlist_page_retained_as_lead",
            )
            return result

        m = YOUTUBE_SHORTS_RX.match(u)
        if m:
            vid = m.group(1)
            return {
                "url_class": "youtube_short", "direct_post_url": u, "direct_post_id": vid,
                "url_validation_status": "valid", "invalidation_reason": "",
            }
        m = YOUTUBE_YOUTU_BE_RX.match(u)
        if m:
            vid = m.group(1)
            return {
                "url_class": "youtube_video", "direct_post_url": u, "direct_post_id": vid,
                "url_validation_status": "valid", "invalidation_reason": "",
            }
        if YOUTUBE_WATCH_RX.match(u):
            vid = _youtube_video_id(u)
            if vid:
                return {
                    "url_class": "youtube_video", "direct_post_url": u, "direct_post_id": vid,
                    "url_validation_status": "valid", "invalidation_reason": "",
                }
            result.update(url_class="youtube_watch_no_id", invalidation_reason="missing_video_id")
            return result
        if YOUTUBE_CHANNEL_RX.match(u):
            result.update(
                url_class="youtube_channel",
                url_validation_status="lead_only",
                invalidation_reason="youtube_channel_page_retained_as_lead",
            )
            return result
        result.update(url_class="youtube_other", invalidation_reason="unrecognised_youtube_url")
        return result

    # ── TikTok ───────────────────────────────────────────────────────────────
    if "tiktok.com" in u:
        m = TIKTOK_VIDEO_RX.match(u)
        if m:
            handle, vid = m.group(1), m.group(2)
            return {
                "url_class": "tiktok_video", "direct_post_url": u, "direct_post_id": vid,
                "url_validation_status": "valid", "invalidation_reason": "",
            }
        result.update(url_class="tiktok_profile_or_other", invalidation_reason="tiktok_non_video_page")
        return result

    # ── Instagram ────────────────────────────────────────────────────────────
    if "instagram.com" in u:
        m = INSTAGRAM_POST_RX.match(u)
        if m:
            kind, shortcode = m.group(1).lower(), m.group(2)
            kind_norm = "reel" if kind == "reels" else kind
            return {
                "url_class": f"instagram_{kind_norm}", "direct_post_url": u, "direct_post_id": shortcode,
                "url_validation_status": "valid", "invalidation_reason": "",
            }
        if INSTAGRAM_PROGRAMME_RX.search(u):
            result.update(
                url_class="instagram_programme_or_campaign",
                invalidation_reason="instagram_programme_or_campaign_page",
            )
            return result
        result.update(url_class="instagram_profile_or_other", invalidation_reason="instagram_non_post_page")
        return result

    result.update(url_class="other_domain", invalidation_reason="not_a_recognised_social_platform")
    return result


# ── TikTok handle extraction ──────────────────────────────────────────────────

def extract_tiktok_handle(url: str) -> Optional[str]:
    """Extract @handle from a tiktok.com/@handle/video/... URL. High confidence:
    the handle is part of the platform's own canonical URL structure."""
    m = TIKTOK_VIDEO_RX.match(str(url).strip())
    return m.group(1) if m else None


# ── YouTube identity resolution ───────────────────────────────────────────────

def _oembed_cache_path(video_url: str, cache_dir: Path) -> Path:
    key = hashlib.sha256(video_url.encode()).hexdigest()
    return Path(cache_dir) / f"{key}.json"


def fetch_youtube_oembed(
    video_url: str,
    get: Callable[..., "requests.Response"] = requests.get,
    cache_dir: Path = OEMBED_CACHE_DIR,
    delay: float = 1.0,
    timeout: int = 10,
) -> Optional[dict]:
    """Query the public YouTube oEmbed endpoint for author_name/author_url/title.

    Returns the parsed JSON dict, or None on failure. Caches successful responses
    to disk so repeated runs don't re-hit the endpoint. `get` is injectable so unit
    tests can pass a mock and avoid any live network call.
    """
    cache_path = _oembed_cache_path(video_url, cache_dir)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    oembed_url = f"https://www.youtube.com/oembed?url={video_url}&format=json"
    try:
        time.sleep(delay)
        resp = get(oembed_url, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
    except Exception:
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def resolve_youtube_identity(
    video_url: str,
    fallback_channel_name: str = "",
    get: Callable[..., "requests.Response"] = requests.get,
    cache_dir: Path = OEMBED_CACHE_DIR,
    delay: float = 1.0,
) -> Dict[str, str]:
    """Resolve creator identity for a YouTube video via oEmbed, falling back to
    already-captured channel_name metadata if oEmbed fails.

    Returns creator_handle, creator_name, creator_identity_source,
    creator_identity_confidence, creator_identity_evidence.
    """
    data = fetch_youtube_oembed(video_url, get=get, cache_dir=cache_dir, delay=delay)
    if data and data.get("author_name"):
        author_url = data.get("author_url", "")
        handle = author_url.rsplit("/", 1)[-1].lstrip("@") if author_url else data["author_name"]
        return {
            "creator_handle": handle,
            "creator_name": data["author_name"],
            "creator_identity_source": "youtube_oembed",
            "creator_identity_confidence": "high",
            "creator_identity_evidence": f"oEmbed author_name={data['author_name']!r} author_url={author_url!r}",
        }
    if fallback_channel_name and str(fallback_channel_name).strip():
        return {
            "creator_handle": str(fallback_channel_name).strip(),
            "creator_name": str(fallback_channel_name).strip(),
            "creator_identity_source": "existing_channel_metadata",
            "creator_identity_confidence": "medium",
            "creator_identity_evidence": f"channel_name field = {fallback_channel_name!r}",
        }
    return {
        "creator_handle": "",
        "creator_name": "",
        "creator_identity_source": "none",
        "creator_identity_confidence": "unresolved",
        "creator_identity_evidence": "",
    }


# ── Instagram identity resolution (no live fetch: instagram.com blocks bots) ──

def resolve_instagram_identity(
    existing_creator_name: str = "",
    existing_creator_handle: str = "",
    page_metadata: Optional[dict] = None,
) -> Dict[str, str]:
    """Resolve creator identity for an Instagram post from already-collected
    structured metadata or hydrated page metadata (og:title / JSON-LD). No live
    fetch is attempted here — instagram.com is bot-blocked (see
    src/collectors/hydration.py BOT_BLOCKED_DOMAINS) so any metadata must already
    have been captured upstream.
    """
    if existing_creator_handle and str(existing_creator_handle).strip():
        return {
            "creator_handle": str(existing_creator_handle).strip().lstrip("@"),
            "creator_name": existing_creator_name or existing_creator_handle,
            "creator_identity_source": "existing_structured_metadata",
            "creator_identity_confidence": "medium",
            "creator_identity_evidence": f"creator_handle field = {existing_creator_handle!r}",
        }
    meta = page_metadata or {}
    title = str(meta.get("page_title") or meta.get("og_title") or "")
    m = re.search(r"@([\w.]{2,30})", title)
    if m:
        return {
            "creator_handle": m.group(1),
            "creator_name": m.group(1),
            "creator_identity_source": "page_metadata_og_title",
            "creator_identity_confidence": "medium",
            "creator_identity_evidence": f"og:title contained handle: {title!r}",
        }
    return {
        "creator_handle": "",
        "creator_name": "",
        "creator_identity_source": "none",
        "creator_identity_confidence": "unresolved",
        "creator_identity_evidence": "",
    }


# ── Brand link verification ───────────────────────────────────────────────────

BRAND_ALIASES: Dict[str, list] = {
    "TALA": ["tala", "wearetala", "talaactivewear"],
    "Adanola": ["adanola"],
    "Girlfriend Collective": ["girlfriend collective", "girlfriendcollective", "gfc"],
    "Oner Active": ["oner active", "oneractive", "onerarmy"],
}


def _find_alias(text: str, aliases: list) -> Optional[str]:
    text_l = text.lower()
    for alias in aliases:
        if alias in text_l:
            return alias
    return None


def verify_brand_link(
    brand: str,
    video_title: str = "",
    caption_or_description: str = "",
    channel_name: str = "",
    creator_name: str = "",
    personalised_code: str = "",
    disclosure_type: str = "",
) -> Dict[str, object]:
    """Verify brand linkage from actual text/metadata evidence — never from the
    inherited `brand` search-query field alone.

    Returns brand_link_verified (bool), brand_link_evidence, brand_link_source,
    brand_link_confidence.
    """
    aliases = BRAND_ALIASES.get(brand, [brand.lower()]) if brand else []
    if not aliases:
        return {
            "brand_link_verified": False, "brand_link_evidence": "",
            "brand_link_source": "none", "brand_link_confidence": "unresolved",
        }

    title = str(video_title) if video_title and str(video_title).lower() != "nan" else ""
    caption = str(caption_or_description) if caption_or_description and str(caption_or_description).lower() != "nan" else ""

    # High confidence: explicit hashtag/mention, or brand name in the title itself.
    hashtag_rx = re.compile(
        "|".join(r"[#@]" + re.escape(a.replace(" ", "")) for a in aliases), re.I
    )
    for field_name, field_val in (("video_title", title), ("caption_or_description", caption)):
        m = hashtag_rx.search(field_val)
        if m:
            return {
                "brand_link_verified": True,
                "brand_link_evidence": f"matched {m.group(0)!r} in {field_name}",
                "brand_link_source": field_name,
                "brand_link_confidence": "high",
            }
    alias = _find_alias(title, aliases)
    if alias:
        return {
            "brand_link_verified": True,
            "brand_link_evidence": f"brand name {alias!r} present in video_title: {title[:120]!r}",
            "brand_link_source": "video_title",
            "brand_link_confidence": "high",
        }

    if personalised_code and str(personalised_code).strip():
        return {
            "brand_link_verified": True,
            "brand_link_evidence": f"personalised code present: {personalised_code!r}",
            "brand_link_source": "personalised_code",
            "brand_link_confidence": "high",
        }

    # Medium confidence: brand mentioned in body caption/description snippet.
    alias = _find_alias(caption, aliases)
    if alias:
        return {
            "brand_link_verified": True,
            "brand_link_evidence": f"brand name {alias!r} present in caption_or_description snippet",
            "brand_link_source": "caption_or_description",
            "brand_link_confidence": "medium",
        }

    # Low confidence: only the channel/creator name references the brand (usually
    # the brand's own channel, not a third-party creator — flagged, not verified).
    for field_name, field_val in (("channel_name", channel_name), ("creator_name", creator_name)):
        alias = _find_alias(str(field_val) if field_val else "", aliases)
        if alias:
            return {
                "brand_link_verified": False,
                "brand_link_evidence": f"brand name {alias!r} only found in {field_name} — likely the brand's own account, not corroborated elsewhere",
                "brand_link_source": field_name,
                "brand_link_confidence": "low",
            }

    return {
        "brand_link_verified": False,
        "brand_link_evidence": "",
        "brand_link_source": "none",
        "brand_link_confidence": "unresolved",
    }


# ── Partnership classification ────────────────────────────────────────────────

PARTNERSHIP_VALUES = (
    "paid_sponsorship", "gifted", "ambassador", "affiliate",
    "founder_or_employee", "organic", "unclear",
)

_PAID_RX = re.compile(r"#ad\b|paid partnership|paid promotion", re.I)
_GIFTED_RX = re.compile(r"#gifted\b|\bgifted\b|\bc/o\b", re.I)
_AMBASSADOR_RX = re.compile(r"\bambassador\b", re.I)
_AFFILIATE_RX = re.compile(r"\baffiliate\b|use my code|discount code|\bpromo code\b", re.I)
_FOUNDER_RX = re.compile(r"\bfounder\b|\bco-founder\b|\bco founder\b", re.I)
_ORGANIC_HINT_RX = re.compile(r"\bhaul\b|\breview\b|\btry.?on\b|\bunboxing\b|\bhonest\b", re.I)


def classify_partnership(
    caption_or_description: str = "",
    video_title: str = "",
    disclosure_type: str = "",
    personalised_code: str = "",
) -> Dict[str, object]:
    """Classify partnership type from explicit disclosure evidence only. Never
    infers paid_sponsorship merely because a brand is mentioned or worn."""
    text = " ".join(
        str(x) for x in (caption_or_description, video_title, disclosure_type)
        if x and str(x).lower() != "nan"
    )

    if _PAID_RX.search(text):
        m = _PAID_RX.search(text)
        return {"partnership_type": "paid_sponsorship", "partnership_evidence": f"matched {m.group(0)!r}",
                "partnership_confidence": "high", "partnership_inferred": True}
    if _GIFTED_RX.search(text):
        m = _GIFTED_RX.search(text)
        return {"partnership_type": "gifted", "partnership_evidence": f"matched {m.group(0)!r}",
                "partnership_confidence": "high", "partnership_inferred": True}
    if _AMBASSADOR_RX.search(text):
        m = _AMBASSADOR_RX.search(text)
        return {"partnership_type": "ambassador", "partnership_evidence": f"matched {m.group(0)!r}",
                "partnership_confidence": "high", "partnership_inferred": True}
    if (personalised_code and str(personalised_code).strip()) or _AFFILIATE_RX.search(text):
        m = _AFFILIATE_RX.search(text)
        ev = f"matched {m.group(0)!r}" if m else f"personalised_code={personalised_code!r}"
        return {"partnership_type": "affiliate", "partnership_evidence": ev,
                "partnership_confidence": "high", "partnership_inferred": True}
    if _FOUNDER_RX.search(text):
        m = _FOUNDER_RX.search(text)
        return {"partnership_type": "founder_or_employee", "partnership_evidence": f"matched {m.group(0)!r}",
                "partnership_confidence": "high", "partnership_inferred": True}
    if _ORGANIC_HINT_RX.search(text):
        m = _ORGANIC_HINT_RX.search(text)
        return {"partnership_type": "organic", "partnership_evidence": f"no disclosure found; matched {m.group(0)!r}",
                "partnership_confidence": "medium", "partnership_inferred": True}
    return {"partnership_type": "unclear", "partnership_evidence": "",
            "partnership_confidence": "low", "partnership_inferred": False}


# ── Partnership reassessment from full title+description (Day 2 Task 2B Part C) ─

def reassess_partnership(
    video_title: str = "",
    video_description: str = "",
    caption_or_description: str = "",
    personalised_code: str = "",
) -> Dict[str, object]:
    """Reassess partnership_type from the FULL video title + description (not a
    truncated caption snippet), against explicit disclosure phrases only
    (configs/feature_rules.yaml::partnership_disclosure_phrases). A bare brand-name
    mention is never sufficient -- this function doesn't even look at the brand
    name, only disclosure language.

    Returns revised_partnership_type, partnership_evidence_text (the matched
    phrase in context), partnership_rule_trigger (the matched phrase itself),
    partnership_confidence (high/medium/low).
    """
    from src.text_features import load_feature_rules  # lazy import, avoids cycle at module load

    phrases = load_feature_rules()["partnership_disclosure_phrases"]
    text = " ".join(
        t for t in (video_title, video_description, caption_or_description) if t and str(t).lower() != "nan"
    )
    text_l = text.lower()

    # precedence: paid > gifted > ambassador > affiliate > founder_or_employee
    for category in ("paid_sponsorship", "gifted", "ambassador", "affiliate", "founder_or_employee"):
        for phrase in phrases[category]:
            idx = text_l.find(phrase.lower())
            if idx != -1:
                start = max(0, idx - 40)
                end = min(len(text), idx + len(phrase) + 40)
                return {
                    "revised_partnership_type": category,
                    "partnership_evidence_text": text[start:end].strip(),
                    "partnership_rule_trigger": phrase,
                    "partnership_confidence": "high",
                }

    if personalised_code and str(personalised_code).strip():
        return {
            "revised_partnership_type": "affiliate",
            "partnership_evidence_text": f"personalised_code={personalised_code}",
            "partnership_rule_trigger": "personalised_code_present",
            "partnership_confidence": "high",
        }

    if _ORGANIC_HINT_RX.search(text):
        m = _ORGANIC_HINT_RX.search(text)
        return {
            "revised_partnership_type": "organic",
            "partnership_evidence_text": f"no disclosure phrase found; content framing: {m.group(0)!r}",
            "partnership_rule_trigger": "",
            "partnership_confidence": "medium",
        }

    return {
        "revised_partnership_type": "unclear",
        "partnership_evidence_text": "",
        "partnership_rule_trigger": "",
        "partnership_confidence": "low",
    }


# ── Evidence strength (Part E rules) ──────────────────────────────────────────

def compute_creator_evidence_strength(
    url_validation_status: str,
    creator_identity_confidence: str,
    brand_link_verified: bool,
    has_content_text: bool,
) -> str:
    """Apply Part E rules: strong/medium require a valid direct URL, resolved
    identity, and verified brand link; weak covers a lead with unresolved identity
    or brand link; unusable covers an invalid/irrelevant URL."""
    if url_validation_status != "valid":
        return "unusable"
    identity_ok = creator_identity_confidence in ("high", "medium")
    if identity_ok and brand_link_verified and has_content_text:
        return "strong"
    if identity_ok and brand_link_verified:
        return "medium"
    return "weak"
