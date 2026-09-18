"""Collect official brand claims from public website pages."""

from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import urlparse

from src.collectors.utils import (
    can_fetch, extract_text_from_url, log_failure, make_claim_row,
    make_competitor_row, split_into_claims,
)


def collect_brand_official_pages(
    brand_slug: str,
    brand_name: str,
    pages: List[Dict],
    failures: List[Dict],
) -> List[Dict[str, Any]]:
    """Scrape a list of official brand pages and extract claim rows.

    pages: list of dicts with keys: url, source_type, claim_category.
    Returns list of official claim row dicts.
    """
    rows: List[Dict] = []
    for page in pages:
        url = page.get("url", "")
        if not url:
            continue

        if not can_fetch(url):
            log_failure("official_pages", url, "robots.txt disallows", failures)
            continue

        print(f"    Fetching: {url}")
        text = extract_text_from_url(url, delay=2.5)
        if not text:
            log_failure("official_pages", url, "No text extracted", failures)
            continue

        claims = split_into_claims(text, min_len=40, max_len=400)
        for claim in claims[:30]:  # cap per page
            row = make_claim_row(
                brand=brand_name,
                claim_text=claim,
                source_url=url,
                claim_category=page.get("claim_category", "other"),
                source_type=page.get("source_type", "website"),
                provider_name="trafilatura",
                source_method="html_extraction",
            )
            rows.append(row)

        print(f"      -> {len(claims[:30])} claim sentences extracted")

    return rows


def collect_competitor_pages(
    brand_slug: str,
    brand_name: str,
    pages: List[Dict],
    failures: List[Dict],
) -> List[Dict[str, Any]]:
    """Scrape competitor official pages and return competitor_platforms rows.

    Returns rows for competitor_platforms_candidates.
    """
    rows: List[Dict] = []
    for page in pages:
        url = page.get("url", "")
        if not url:
            continue

        if not can_fetch(url):
            log_failure("competitor_official", url, "robots.txt disallows", failures)
            continue

        print(f"    Fetching competitor page: {url}")
        text = extract_text_from_url(url, delay=2.5)
        if not text:
            log_failure("competitor_official", url, "No text extracted", failures)
            continue

        # Determine platform from URL
        netloc = urlparse(url).netloc.replace("www.", "")
        platform = "website"

        row = make_competitor_row(
            brand=brand_name,
            platform=platform,
            source_url=url,
            context_text=text[:500],
            title=f"{brand_name} — {netloc}",
            provider_name="trafilatura",
            source_method="html_extraction",
        )
        rows.append(row)
        print(f"      -> 1 competitor row added")

    return rows
