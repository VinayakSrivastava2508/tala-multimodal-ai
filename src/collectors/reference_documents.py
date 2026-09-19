"""Day 2.6A: Multimodal Reference Package discovery, classification, and fetch.

Discovers and fetches enterprise reference documents (sustainability/impact
pages, material/product specs, size/care guides, policies, supplier
disclosures, certifications, brand style references) from official brand
domains and authoritative third-party registries -- never from search-result
snippets. Reuses src/collectors/hydration.py's robots-checked fetch, text
extraction, and evidence-strength scoring rather than re-implementing them.

Three discovery routes, in priority order:
  1. Reuse an already-hydrated strong/medium source (no duplicate network call).
  2. Probe a fixed list of common Shopify page/policy slugs on the brand's own
     domain (and parse sitemap.xml if present).
  3. DuckDuckGo site-restricted search for discovery ONLY -- the result is
     never used as content; the linked page is fetched and verified directly,
     and rejected if the fetched page's own domain doesn't match the intended
     official/authoritative source.
"""

from __future__ import annotations

import re
import time
import urllib.robotparser
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests

from src.collectors.hydration import (
    BOT_UA,
    BROWSER_HEADERS,
    can_fetch_robots,
    compute_evidence_strength,
    extract_main_text,
    extract_page_metadata,
    netloc_clean,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Controlled vocabularies ────────────────────────────────────────────────────

REFERENCE_TYPES = (
    "impact_report", "sustainability_page", "certification", "material_specification",
    "product_specification", "care_guide", "size_guide", "return_policy",
    "quality_or_warranty_policy", "supplier_disclosure", "packaging_policy",
    "emissions_disclosure", "labour_or_ethics_policy", "brand_style_reference", "other",
)

REFERENCE_STATUSES = (
    "collected", "already_available", "not_found", "blocked", "irrelevant",
    "duplicate", "extraction_failed",
)

# TALA's official-domain family (Priority 1). Order matters only for readability.
TALA_DOMAINS = ["wearetala.com", "eu.tala.co.uk", "talaactivewear.com", "weartala.com"]

COMPETITOR_DOMAINS = {
    "Adanola": ["adanola.com"],
    "Girlfriend Collective": ["girlfriend.com"],
    "Oner Active": ["oneractive.com"],
}

# Authoritative third-party ethical/sustainability registries this project treats
# as Priority-2 sources IF the page is directly about the brand in question (never
# a generic listing page).
AUTHORITATIVE_REGISTRY_DOMAINS = {
    "goodonyou.eco": "certification",
    "thegoodshoppingguide.com": "certification",
    "theeasyethical.com": "certification",
    "thecommons.earth": "certification",
}

# Common Shopify page/policy slugs to probe directly on a brand's own domain.
# Shopify auto-generates /policies/* for refund/shipping/terms/privacy; /pages/*
# is theme-dependent so several plausible slugs are tried per requirement.
# A sitemap can list image/asset URLs alongside real pages -- these are never
# reference documents and must never be run through HTML text extraction
# (decoding binary image bytes as text produces mojibake that can spuriously
# clear the STRONG_MIN length threshold -- a real bug this regex prevents).
_BINARY_EXTENSION_RX = re.compile(
    r"\.(jpg|jpeg|png|webp|gif|svg|bmp|ico|css|js|mjs|woff2?|ttf|eot|mp4|mov|zip)$", re.I)

COMMON_SLUGS = [
    "policies/refund-policy", "policies/shipping-policy", "policies/terms-of-service",
    "pages/size-guide", "pages/sizing", "pages/size-chart", "pages/fit-guide",
    "pages/care", "pages/fabric-care", "pages/care-guide", "pages/how-to-care",
    "pages/returns", "pages/returns-and-exchanges", "pages/faq",
    "pages/sustainability", "pages/purpose", "pages/responsibility", "pages/our-factories",
    "pages/materials", "pages/fabrics", "pages/natural-fibres",
    "pages/certifications", "pages/warranty", "pages/quality-promise", "pages/guarantee",
    "pages/packaging", "pages/supply-chain", "pages/our-suppliers",
    "pages/about-us", "pages/about", "pages/lookbook", "pages/brand-guidelines",
]

REFERENCE_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "sustainability_page": ["sustainab", "responsib", "purpose", "planet", "eco-friendly", "our impact"],
    "impact_report": ["impact report", "annual report", "esg report", "sustainability report"],
    "certification": ["certif", "bluesign", "oeko-tex", "grs certified", "gots", "fair wear",
                       "b corp", "bcorp", "good on you", "goodonyou", "ethical rating"],
    "material_specification": ["fabric", "material composition", "natural fibres", "natural fibers",
                                "recycled polyester", "made from"],
    "product_specification": ["technical spec", "product details", "product specification"],
    "care_guide": ["care guide", "wash care", "fabric care", "how to care", "washing instructions"],
    "size_guide": ["size guide", "size chart", "sizing", "measurements", "fit guide"],
    "return_policy": ["refund policy", "return policy", "returns and exchanges", "returns policy"],
    "quality_or_warranty_policy": ["warranty", "guarantee", "quality promise"],
    "supplier_disclosure": ["our factories", "factory", "supplier", "manufactur", "supply chain"],
    "packaging_policy": ["packaging"],
    "emissions_disclosure": ["carbon", "emissions", "co2", "climate impact", "net zero"],
    "labour_or_ethics_policy": ["labour", "labor rights", "code of conduct", "modern slavery",
                                 "ethics policy", "worker welfare"],
    "brand_style_reference": ["lookbook", "style guide", "brand guidelines", "brand identity"],
}


def _registry_domain_for(url: str) -> Optional[str]:
    """Return the matching AUTHORITATIVE_REGISTRY_DOMAINS key for url's domain,
    matching subdomains too (e.g. directory.goodonyou.eco -> goodonyou.eco)."""
    domain = netloc_clean(url)
    for registry_domain in AUTHORITATIVE_REGISTRY_DOMAINS:
        if domain == registry_domain or domain.endswith("." + registry_domain):
            return registry_domain
    return None


# A page's URL slug is a stronger signal than scraped-text keyword density --
# we chose that exact slug (from COMMON_SLUGS) because it names its own
# purpose, and a 200 response confirms the site published a page there. Checked
# as a substring of the URL PATH only (never the query string), in the order
# below; first match wins. This does not override the registry-domain check.
URL_SLUG_TYPE_OVERRIDES: List[Tuple[str, str]] = [
    ("size-guide", "size_guide"), ("size-chart", "size_guide"), ("sizing", "size_guide"),
    ("fit-guide", "size_guide"),
    ("fabric-care", "care_guide"), ("how-to-care", "care_guide"), ("care-guide", "care_guide"),
    ("refund-policy", "return_policy"), ("return-policy", "return_policy"),
    ("returns-and-exchanges", "return_policy"), ("returns", "return_policy"),
    ("quality-promise", "quality_or_warranty_policy"), ("warranty", "quality_or_warranty_policy"),
    ("guarantee", "quality_or_warranty_policy"),
    ("packaging", "packaging_policy"),
    ("our-factories", "supplier_disclosure"), ("supply-chain", "supplier_disclosure"),
    ("our-suppliers", "supplier_disclosure"),
    ("certifications", "certification"),
    ("brand-guidelines", "brand_style_reference"), ("lookbook", "brand_style_reference"),
    ("natural-fibres", "material_specification"), ("fabrics", "material_specification"),
    ("materials", "material_specification"),
    ("sustainability", "sustainability_page"), ("purpose", "sustainability_page"),
    ("responsibility", "sustainability_page"),
]


def classify_reference_type(url: str, title: str, text: str) -> str:
    """Keyword-based classification into REFERENCE_TYPES. Domain overrides (known
    authoritative registries) take precedence, then URL-slug overrides, then
    keyword-density scoring; falls back to 'other'."""
    registry_domain = _registry_domain_for(url)
    if registry_domain is not None:
        return AUTHORITATIVE_REGISTRY_DOMAINS[registry_domain]

    url_path = urlparse(url).path.lower()
    for slug, ref_type in URL_SLUG_TYPE_OVERRIDES:
        if slug in url_path:
            return ref_type

    # Normalise URL path separators to spaces so a keyword phrase like "size
    # guide" matches a hyphenated slug like "/pages/size-guide" -- without this,
    # the URL contributes almost nothing to classification because real slugs
    # are hyphenated, not space-separated.
    url_path_words = url_path.replace("-", " ").replace("_", " ").replace("/", " ")
    haystack = f"{url_path_words} {title} {text[:2000]}".lower()
    best_type, best_hits = "other", 0
    for ref_type, keywords in REFERENCE_TYPE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in haystack)
        if hits > best_hits:
            best_type, best_hits = ref_type, hits
    return best_type


def is_official_domain(url: str, brand: str) -> bool:
    """True if url's domain belongs to the brand's official-domain family."""
    domain = netloc_clean(url)
    if brand == "TALA":
        return domain in TALA_DOMAINS
    return domain in COMPETITOR_DOMAINS.get(brand, [])


def is_authoritative_registry(url: str) -> bool:
    return _registry_domain_for(url) is not None


# ── Candidate discovery ────────────────────────────────────────────────────────

@dataclass
class ReferenceCandidate:
    """One discovered URL before fetch/classification."""
    brand: str = ""
    source_url: str = ""
    discovery_method: str = ""  # existing_hydrated_source | common_slug_probe | sitemap | ddgs_discovery_then_direct_fetch
    official_source: bool = False


def candidates_from_hydrated_sources(hydrated_sources_df, brand: str) -> List[ReferenceCandidate]:
    """Priority-1 route: reuse already-hydrated strong/medium sources for this
    brand instead of re-fetching. Never duplicates a network call for a URL
    that has already been successfully hydrated."""
    if hydrated_sources_df.empty:
        return []
    rows = hydrated_sources_df[
        (hydrated_sources_df["brand"] == brand)
        & (hydrated_sources_df["evidence_strength"].isin(["strong", "medium"]))
        & (hydrated_sources_df["hydration_status"].isin(["success", "cached"]))
    ]
    out = []
    for _, r in rows.iterrows():
        url = str(r["source_url"])
        if not url.startswith("http"):
            continue
        official = is_official_domain(url, brand) or is_authoritative_registry(url)
        if not official:
            continue  # not an enterprise reference source (e.g. generic press)
        out.append(ReferenceCandidate(brand=brand, source_url=url, discovery_method="existing_hydrated_source", official_source=True))
    return out


def _head_or_get_ok(url: str, get=requests.get, timeout: int = 10) -> bool:
    try:
        resp = get(url, headers=BROWSER_HEADERS, timeout=timeout, allow_redirects=True)
        return resp.status_code == 200
    except Exception:
        return False


def discover_common_slugs(base_url: str, brand: str, get=requests.get, delay: float = 1.0) -> List[ReferenceCandidate]:
    """Priority-2 route: probe a fixed list of common Shopify page/policy slugs
    directly on the brand's own domain. Every probe is robots-checked."""
    out = []
    for slug in COMMON_SLUGS:
        url = base_url.rstrip("/") + "/" + slug
        if not can_fetch_robots(url, ua=BOT_UA):
            continue
        time.sleep(delay)
        if _head_or_get_ok(url, get=get):
            out.append(ReferenceCandidate(brand=brand, source_url=url, discovery_method="common_slug_probe", official_source=True))
    return out


def discover_sitemap(base_url: str, brand: str, get=requests.get, delay: float = 1.0) -> List[ReferenceCandidate]:
    """Priority-2 route: parse /sitemap.xml (or a pages/policies sub-sitemap) and
    keep only URLs whose path looks like a policy/reference page."""
    sitemap_url = base_url.rstrip("/") + "/sitemap.xml"
    if not can_fetch_robots(sitemap_url, ua=BOT_UA):
        return []
    try:
        resp = get(sitemap_url, headers=BROWSER_HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
    except Exception:
        return []

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "lxml-xml")
    locs = [loc.get_text().strip() for loc in soup.find_all("loc")]

    # Sub-sitemaps (Shopify splits sitemap.xml into sitemap_pages_1.xml etc.)
    sub_sitemaps = [l for l in locs if l.endswith(".xml")]
    page_urls = [l for l in locs if not l.endswith(".xml")]
    for sub in sub_sitemaps[:5]:
        time.sleep(delay)
        try:
            r2 = get(sub, headers=BROWSER_HEADERS, timeout=10)
            if r2.status_code == 200:
                soup2 = BeautifulSoup(r2.text, "lxml-xml")
                page_urls.extend(l.get_text().strip() for l in soup2.find_all("loc"))
        except Exception:
            continue

    keyword_filter = re.compile(
        r"(page|polic|size|care|sustain|material|return|warranty|factory|supplier|"
        r"certif|packag|emission|carbon|ethic|labour|labor|about)", re.I)
    out = []
    for url in page_urls:
        if _BINARY_EXTENSION_RX.search(url.split("?")[0]):
            continue  # image/css/js/font asset URLs in a sitemap are never a document
        if keyword_filter.search(url):
            out.append(ReferenceCandidate(brand=brand, source_url=url, discovery_method="sitemap", official_source=True))
    return out


def discover_via_ddgs_then_verify(
    brand: str, requirement_label: str, domains: List[str], ddg_search_fn, get=requests.get,
) -> Optional[ReferenceCandidate]:
    """Priority-3 fallback: DuckDuckGo is used ONLY to discover a candidate URL
    (site-restricted query). The search snippet is discarded -- the linked page
    is fetched and verified directly, and rejected unless its resolved domain
    matches one of `domains` (the brand's official family or an authoritative
    registry). Never returns a bare search-result snippet as a reference asset.
    """
    site_filter = " OR ".join(f"site:{d}" for d in domains)
    query = f"({site_filter}) {requirement_label}"
    results = ddg_search_fn(query, max_results=5)
    for r in results:
        url = r.get("href") or r.get("url") or ""
        if not url.startswith("http"):
            continue
        domain = netloc_clean(url)
        if domain not in domains:
            continue  # reject off-domain results -- no snippet-as-document
        if not can_fetch_robots(url, ua=BOT_UA):
            continue
        return ReferenceCandidate(brand=brand, source_url=url, discovery_method="ddgs_discovery_then_direct_fetch", official_source=True)
    return None


def dedupe_candidates(candidates: List[ReferenceCandidate]) -> List[ReferenceCandidate]:
    seen = set()
    out = []
    for c in candidates:
        key = c.source_url.split("?")[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# ── Fetch + extract (HTML and PDF) ─────────────────────────────────────────────

@dataclass
class ReferenceFetchResult:
    """Full multimodal_reference_assets.csv row (schema in Part D)."""
    reference_id: str = ""
    brand: str = ""
    reference_type: str = "other"
    document_title: str = ""
    source_url: str = ""
    canonical_url: str = ""
    source_domain: str = ""
    publication_date: str = ""
    effective_date: str = ""
    retrieved_at: str = ""
    mime_type: str = ""
    local_path: str = ""
    extracted_text: str = ""
    text_length: int = 0
    page_count: Optional[int] = None
    has_tables: bool = False
    has_images: bool = False
    extraction_method: str = ""
    evidence_strength: str = "unusable"
    rights_or_access_basis: str = "official_direct_public_asset"
    processing_status: str = "failed"
    file_hash: str = ""
    provenance_note: str = ""
    discovery_method: str = ""
    official_source: bool = False
    certification_authority: str = ""
    claim_relevance: str = ""
    reference_status: str = "not_found"
    failure_reason: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def _is_pdf_response(resp: requests.Response, url: str) -> bool:
    content_type = resp.headers.get("Content-Type", "").lower()
    return "application/pdf" in content_type or url.lower().split("?")[0].endswith(".pdf")


def extract_pdf_text(content: bytes) -> tuple[str, int, bool]:
    """Page-preserving PDF text extraction via pypdf. Returns
    (text_with_page_markers, page_count, has_tables_heuristic). Tables are
    detected heuristically (multiple consecutive lines with >=2 numeric/percent
    tokens each), never fabricated."""
    import io
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    page_texts = []
    has_tables = False
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        page_texts.append(f"[page {i + 1}]\n{text.strip()}")
        numeric_lines = sum(1 for line in text.splitlines() if len(re.findall(r"\d+%|\d+\s*(kg|cm|g|mm)", line)) >= 2)
        if numeric_lines >= 2:
            has_tables = True
    return "\n\n".join(page_texts).strip(), len(reader.pages), has_tables


def fetch_reference_document(
    candidate: ReferenceCandidate,
    reference_id: str,
    get=requests.get,
    media_dir: Optional[Path] = None,
    delay: float = 1.2,
) -> ReferenceFetchResult:
    """Fetch one candidate URL and return a full manifest row. Never raises --
    a failed fetch returns reference_status='blocked' or 'not_found', not an
    exception, and is never silently dropped from the manifest."""
    import hashlib

    now = datetime.now(timezone.utc).isoformat()
    result = ReferenceFetchResult(
        reference_id=reference_id, brand=candidate.brand, source_url=candidate.source_url,
        source_domain=netloc_clean(candidate.source_url), retrieved_at=now,
        discovery_method=candidate.discovery_method, official_source=candidate.official_source,
        rights_or_access_basis="official_direct_public_asset" if candidate.official_source else "unknown",
    )

    if not can_fetch_robots(candidate.source_url, ua=BOT_UA):
        result.reference_status = "blocked"
        result.failure_reason = "robots_disallowed"
        result.processing_status = "failed"
        return result

    time.sleep(delay)
    try:
        resp = get(candidate.source_url, headers=BROWSER_HEADERS, timeout=15, allow_redirects=True)
    except Exception as exc:
        result.reference_status = "blocked"
        result.failure_reason = str(exc)[:120]
        result.processing_status = "failed"
        return result

    if resp.status_code >= 400:
        result.reference_status = "blocked"
        result.failure_reason = f"http_{resp.status_code}"
        result.processing_status = "failed"
        return result

    content = resp.content
    file_hash = hashlib.sha256(content).hexdigest()
    result.file_hash = file_hash
    result.canonical_url = resp.url

    content_type = resp.headers.get("Content-Type", "").lower()
    is_pdf = _is_pdf_response(resp, candidate.source_url)
    if not is_pdf and content_type and not any(t in content_type for t in ("text/html", "application/xhtml", "text/plain")):
        # Binary/non-document content (image, CSS, JS, font, video, ...) must
        # never be run through HTML text extraction -- decoding binary bytes as
        # text can produce mojibake that spuriously clears the evidence-strength
        # length threshold. Reject explicitly rather than silently mis-extracting.
        result.mime_type = content_type.split(";")[0].strip()
        result.reference_status = "irrelevant"
        result.failure_reason = f"non_document_content_type:{result.mime_type}"
        result.processing_status = "failed"
        return result

    if is_pdf:
        result.mime_type = "application/pdf"
        try:
            text, page_count, has_tables = extract_pdf_text(content)
        except Exception as exc:
            result.reference_status = "extraction_failed"
            result.failure_reason = f"pdf_extraction_error: {str(exc)[:100]}"
            result.processing_status = "failed"
            return result
        result.extracted_text = text
        result.text_length = len(text)
        result.page_count = page_count
        result.has_tables = has_tables
        result.extraction_method = "pypdf"
        result.document_title = candidate.source_url.rsplit("/", 1)[-1]
        if media_dir is not None:
            dest = media_dir / candidate.brand.lower().replace(" ", "_") / f"{reference_id}.pdf"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            result.local_path = str(dest.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in dest.parents else str(dest)
    else:
        result.mime_type = "text/html"
        html = resp.text
        text, method = extract_main_text(html, url=resp.url)
        meta = extract_page_metadata(html)
        result.extracted_text = text
        result.text_length = len(text)
        result.extraction_method = method or "requests_no_text"
        result.document_title = meta["page_title"]
        result.publication_date = meta["publication_date"]
        result.canonical_url = meta["canonical_url"] or resp.url
        result.has_images = "<img" in html.lower()
        result.has_tables = bool(re.search(r"<table", html, re.I))

    if not result.extracted_text or len(result.extracted_text.strip()) == 0:
        result.reference_status = "extraction_failed"
        result.failure_reason = "empty_extraction"
        result.evidence_strength = "unusable"
        result.processing_status = "failed"
        return result

    result.reference_type = classify_reference_type(candidate.source_url, result.document_title, result.extracted_text)
    strength, _ = compute_evidence_strength(result.extracted_text, resp.status_code, has_title=bool(result.document_title))
    result.evidence_strength = strength
    result.reference_status = "already_available" if candidate.discovery_method == "existing_hydrated_source" else "collected"
    result.processing_status = "downloaded" if strength != "unusable" else "failed"
    result.provenance_note = f"discovered via {candidate.discovery_method}; official_source={candidate.official_source}"
    registry_domain = _registry_domain_for(candidate.source_url) if candidate.source_url else None
    if registry_domain is not None:
        result.certification_authority = registry_domain

    return result
