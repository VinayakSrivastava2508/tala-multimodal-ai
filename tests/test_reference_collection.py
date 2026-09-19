"""Tests for src/collectors/reference_documents.py (Day 2.6A Multimodal
Reference Package discovery/collection). No live network calls -- requests.get
and DDGS search are stubbed."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.hydration import extract_main_text
from src.collectors.reference_documents import (
    AUTHORITATIVE_REGISTRY_DOMAINS,
    REFERENCE_STATUSES,
    REFERENCE_TYPES,
    ReferenceCandidate,
    candidates_from_hydrated_sources,
    classify_reference_type,
    dedupe_candidates,
    discover_via_ddgs_then_verify,
    extract_pdf_text,
    fetch_reference_document,
    is_authoritative_registry,
    is_official_domain,
)


def _fake_response(text="", content=None, status_code=200, url="", headers=None):
    return SimpleNamespace(
        status_code=status_code, text=text, content=content if content is not None else text.encode(),
        url=url, headers=headers or {"Content-Type": "text/html"},
    )


# ── official-domain verification ──────────────────────────────────────────────

def test_is_official_domain_true_for_tala_family():
    assert is_official_domain("https://www.wearetala.com/pages/sustainability", "TALA")
    assert is_official_domain("https://eu.tala.co.uk/pages/sustainability", "TALA")


def test_is_official_domain_false_for_unrelated_domain():
    assert not is_official_domain("https://randomblog.com/tala-review", "TALA")


def test_is_official_domain_false_for_wrong_brand():
    assert not is_official_domain("https://adanola.com/pages/sustainability", "TALA")


# ── authoritative certification registry handling ─────────────────────────────

def test_is_authoritative_registry_recognises_known_domains():
    assert is_authoritative_registry("https://directory.goodonyou.eco/brand/tala")
    assert not is_authoritative_registry("https://randomblog.com/tala")


def test_classify_reference_type_domain_override_takes_precedence():
    """A Good On You page must be classified 'certification' regardless of its
    title/text content -- the domain override is authoritative."""
    ref_type = classify_reference_type("https://directory.goodonyou.eco/brand/tala", "Some unrelated title", "unrelated body text")
    assert ref_type == "certification"


# ── controlled reference types ─────────────────────────────────────────────────

def test_classify_reference_type_returns_only_controlled_vocabulary_values():
    samples = [
        ("https://www.wearetala.com/pages/size-guide", "Size Guide", "chest waist hip measurements"),
        ("https://www.wearetala.com/policies/refund-policy", "Refund Policy", "return within 30 days"),
        ("https://www.wearetala.com/pages/our-factories", "Our Factories", "our supplier factories manufacture"),
        ("https://www.wearetala.com/pages/random-page", "Random", "nothing matching any keyword at all"),
    ]
    for url, title, text in samples:
        assert classify_reference_type(url, title, text) in REFERENCE_TYPES


def test_reference_statuses_are_the_documented_controlled_vocabulary():
    assert REFERENCE_STATUSES == (
        "collected", "already_available", "not_found", "blocked", "irrelevant", "duplicate", "extraction_failed",
    )


# ── search-result snippet rejection (DDGS discovery-only) ─────────────────────

def test_ddgs_discovery_rejects_off_domain_results_never_uses_snippet_as_document():
    def fake_ddg_search(query, max_results=5):
        return [{"href": "https://unrelated-blog.example.com/tala-mentioned-here", "body": "some snippet text", "title": "Some blog"}]

    candidate = discover_via_ddgs_then_verify("TALA", "size guide", ["wearetala.com"], fake_ddg_search)
    assert candidate is None  # off-domain result must be rejected, not accepted as a document


def test_ddgs_discovery_accepts_on_domain_result_but_never_uses_body_as_content():
    def fake_ddg_search(query, max_results=5):
        return [{"href": "https://www.wearetala.com/pages/size-guide", "body": "irrelevant snippet", "title": "irrelevant"}]

    candidate = discover_via_ddgs_then_verify("TALA", "size guide", ["wearetala.com"], fake_ddg_search)
    assert candidate is not None
    assert candidate.source_url == "https://www.wearetala.com/pages/size-guide"
    assert candidate.discovery_method == "ddgs_discovery_then_direct_fetch"
    # The candidate carries only a URL -- no snippet/body text field exists on
    # ReferenceCandidate, so the snippet cannot leak into the manifest as content.
    assert not hasattr(candidate, "body")
    assert not hasattr(candidate, "snippet")


def test_ddgs_discovery_returns_none_when_no_results():
    def fake_ddg_search(query, max_results=5):
        return []
    assert discover_via_ddgs_then_verify("TALA", "size guide", ["wearetala.com"], fake_ddg_search) is None


# ── empty extraction rejection ─────────────────────────────────────────────────

def test_fetch_reference_document_marks_empty_extraction_as_extraction_failed(monkeypatch):
    monkeypatch.setattr("src.collectors.reference_documents.can_fetch_robots", lambda url, ua=None: True)
    candidate = ReferenceCandidate(brand="TALA", source_url="https://www.wearetala.com/pages/empty", discovery_method="common_slug_probe", official_source=True)

    def fake_get(url, headers=None, timeout=None, allow_redirects=True):
        return _fake_response(text="<html><body></body></html>", url=url)

    result = fetch_reference_document(candidate, "ref_tala_001", get=fake_get, delay=0)
    assert result.reference_status == "extraction_failed"
    assert result.failure_reason == "empty_extraction"
    assert result.evidence_strength == "unusable"


# ── blocked / robots-disallowed handling ───────────────────────────────────────

def test_fetch_reference_document_marks_robots_disallowed_as_blocked(monkeypatch):
    monkeypatch.setattr("src.collectors.reference_documents.can_fetch_robots", lambda url, ua=None: False)
    candidate = ReferenceCandidate(brand="TALA", source_url="https://www.wearetala.com/pages/disallowed", discovery_method="common_slug_probe")

    result = fetch_reference_document(candidate, "ref_tala_002", get=lambda *a, **k: None, delay=0)
    assert result.reference_status == "blocked"
    assert result.failure_reason == "robots_disallowed"


def test_fetch_reference_document_never_raises_on_network_exception(monkeypatch):
    monkeypatch.setattr("src.collectors.reference_documents.can_fetch_robots", lambda url, ua=None: True)
    candidate = ReferenceCandidate(brand="TALA", source_url="https://www.wearetala.com/pages/timeout", discovery_method="common_slug_probe")

    def fake_get(*a, **k):
        raise ConnectionError("simulated timeout")

    result = fetch_reference_document(candidate, "ref_tala_003", get=fake_get, delay=0)
    assert result.reference_status == "blocked"
    assert result.processing_status == "failed"


# ── missing publication date allowed ────────────────────────────────────────────

def test_fetch_reference_document_succeeds_without_a_publication_date(monkeypatch):
    monkeypatch.setattr("src.collectors.reference_documents.can_fetch_robots", lambda url, ua=None: True)
    candidate = ReferenceCandidate(brand="TALA", source_url="https://www.wearetala.com/pages/purpose", discovery_method="common_slug_probe", official_source=True)
    long_text = "Our sustainability purpose statement. " * 40  # clears STRONG_MIN with no date meta

    def fake_get(url, headers=None, timeout=None, allow_redirects=True):
        html = f"<html><head><title>Purpose</title></head><body><p>{long_text}</p></body></html>"
        return _fake_response(text=html, url=url)

    result = fetch_reference_document(candidate, "ref_tala_004", get=fake_get, delay=0)
    assert result.publication_date == ""  # no date -- must not block collection
    assert result.reference_status in ("collected", "already_available")
    assert result.evidence_strength in ("strong", "medium")


# ── duplicate URL / hash detection ─────────────────────────────────────────────

def test_dedupe_candidates_strips_query_string_and_trailing_slash():
    candidates = [
        ReferenceCandidate(source_url="https://www.wearetala.com/pages/sustainability?v=1"),
        ReferenceCandidate(source_url="https://www.wearetala.com/pages/sustainability/"),
        ReferenceCandidate(source_url="https://www.wearetala.com/pages/care"),
    ]
    out = dedupe_candidates(candidates)
    assert len(out) == 2


# ── API keys absent from outputs ────────────────────────────────────────────────

def test_fetch_result_never_contains_api_key_or_token_fields():
    candidate = ReferenceCandidate(brand="TALA", source_url="https://www.wearetala.com/pages/x")
    result = fetch_reference_document(candidate, "ref_tala_005", get=lambda *a, **k: _fake_response(status_code=404), delay=0)
    row = result.to_dict()
    forbidden_substrings = ("api_key", "apikey", "secret", "token", "authorization", "bearer")
    for key in row:
        assert not any(f in key.lower() for f in forbidden_substrings)


# ── binary/image content must never be extracted as text ──────────────────────

def test_fetch_reference_document_rejects_image_content_type(monkeypatch):
    """Regression: a sitemap or slug probe can resolve to an image/CSS/JS asset.
    Decoding its binary bytes as HTML text can produce mojibake that spuriously
    clears the evidence-strength length threshold -- this must be rejected by
    Content-Type before any text extraction is attempted."""
    monkeypatch.setattr("src.collectors.reference_documents.can_fetch_robots", lambda url, ua=None: True)
    candidate = ReferenceCandidate(brand="TALA", source_url="https://cdn.shopify.com/files/logo_size.png?v=1", discovery_method="sitemap", official_source=True)

    def fake_get(url, headers=None, timeout=None, allow_redirects=True):
        return _fake_response(content=b"\x89PNG\r\n\x1a\n" + b"\xff" * 2000, url=url, headers={"Content-Type": "image/png"})

    result = fetch_reference_document(candidate, "ref_tala_006", get=fake_get, delay=0)
    assert result.reference_status == "irrelevant"
    assert "non_document_content_type" in result.failure_reason
    assert result.extracted_text == ""


def test_fetch_reference_document_accepts_html_content_type(monkeypatch):
    monkeypatch.setattr("src.collectors.reference_documents.can_fetch_robots", lambda url, ua=None: True)
    candidate = ReferenceCandidate(brand="TALA", source_url="https://www.wearetala.com/pages/purpose", discovery_method="common_slug_probe", official_source=True)
    long_text = "Our purpose statement is about sustainability and responsibility. " * 20

    def fake_get(url, headers=None, timeout=None, allow_redirects=True):
        html = f"<html><head><title>Purpose</title></head><body><p>{long_text}</p></body></html>"
        return _fake_response(text=html, url=url, headers={"Content-Type": "text/html; charset=utf-8"})

    result = fetch_reference_document(candidate, "ref_tala_007", get=fake_get, delay=0)
    assert result.reference_status in ("collected", "already_available")


def test_sitemap_excludes_image_urls_by_extension():
    from src.collectors.reference_documents import _BINARY_EXTENSION_RX
    assert _BINARY_EXTENSION_RX.search("https://cdn.shopify.com/files/photo.webp")
    assert _BINARY_EXTENSION_RX.search("https://cdn.shopify.com/files/logo.png")
    assert not _BINARY_EXTENSION_RX.search("https://www.wearetala.com/pages/size-guide")


# ── PDF page provenance ──────────────────────────────────────────────────────

class _FakePdfPage:
    def __init__(self, text):
        self._text = text

    def extract_text(self):
        return self._text


class _FakePdfReader:
    def __init__(self, _stream):
        self.pages = [_FakePdfPage("Page one text."), _FakePdfPage("Page two text.")]


def test_extract_pdf_text_retains_page_provenance(monkeypatch):
    monkeypatch.setattr("pypdf.PdfReader", _FakePdfReader)
    text, page_count, has_tables = extract_pdf_text(b"%PDF-fake")
    assert page_count == 2
    assert "[page 1]" in text
    assert "[page 2]" in text
    assert "Page one text." in text
    assert "Page two text." in text
    # page order is preserved
    assert text.index("[page 1]") < text.index("[page 2]")


def test_extract_pdf_text_detects_table_like_numeric_content(monkeypatch):
    class _TablePage:
        def extract_text(self):
            return "Material: 65% cotton, 200 g\nMaterial: 35% polyester, 100 g\n"

    class _TableReader:
        def __init__(self, _stream):
            self.pages = [_TablePage()]

    monkeypatch.setattr("pypdf.PdfReader", _TableReader)
    _, _, has_tables = extract_pdf_text(b"%PDF-fake")
    assert has_tables is True


# ── HTML boilerplate removal (delegated to hydration.extract_main_text) ────────

def test_html_boilerplate_stripped_from_extracted_reference_text():
    html = """
    <html><body>
    <nav>Home | Shop | About | Cart</nav>
    <header>Site Header Banner</header>
    <main><p>""" + ("Our sustainability commitments are outlined here. " * 20) + """</p></main>
    <footer>Copyright 2026 | Privacy Policy | Terms</footer>
    </body></html>
    """
    text, method = extract_main_text(html, url="https://www.wearetala.com/pages/purpose")
    assert "sustainability commitments" in text
    assert "Copyright" not in text
    assert "Cart" not in text


# ── duplicate hash detection (primitive: hash determinism across candidates) ──

def test_identical_content_from_different_candidates_yields_identical_hash(monkeypatch):
    """The orchestrator (scripts/collect_reference_package.py) marks a second
    fetch as reference_status='duplicate' when its file_hash matches one
    already seen -- this only works if identical byte content deterministically
    hashes identically, which this test pins down at the fetch level."""
    monkeypatch.setattr("src.collectors.reference_documents.can_fetch_robots", lambda url, ua=None: True)
    shared_html = "<html><body><p>" + ("Identical policy text. " * 40) + "</p></body></html>"

    def fake_get(url, headers=None, timeout=None, allow_redirects=True):
        return _fake_response(text=shared_html, url=url)

    c1 = ReferenceCandidate(brand="TALA", source_url="https://www.wearetala.com/pages/a", discovery_method="common_slug_probe", official_source=True)
    c2 = ReferenceCandidate(brand="TALA", source_url="https://www.wearetala.com/pages/b-mirrors-a", discovery_method="common_slug_probe", official_source=True)
    r1 = fetch_reference_document(c1, "ref_tala_010", get=fake_get, delay=0)
    r2 = fetch_reference_document(c2, "ref_tala_011", get=fake_get, delay=0)
    assert r1.file_hash == r2.file_hash
    assert r1.file_hash != ""


# ── candidates_from_hydrated_sources: reuse, not duplicate collection ─────────

def test_candidates_from_hydrated_sources_only_reuses_strong_medium_official_rows():
    hydrated = pd.DataFrame([
        {"brand": "TALA", "source_url": "https://www.wearetala.com/pages/purpose", "evidence_strength": "strong", "hydration_status": "success"},
        {"brand": "TALA", "source_url": "https://randomblog.com/tala", "evidence_strength": "strong", "hydration_status": "success"},
        {"brand": "TALA", "source_url": "https://www.wearetala.com/pages/weak", "evidence_strength": "weak", "hydration_status": "success"},
        {"brand": "Adanola", "source_url": "https://adanola.com/pages/about", "evidence_strength": "strong", "hydration_status": "success"},
    ])
    candidates = candidates_from_hydrated_sources(hydrated, "TALA")
    urls = {c.source_url for c in candidates}
    assert "https://www.wearetala.com/pages/purpose" in urls
    assert "https://randomblog.com/tala" not in urls  # not an official/registry domain
    assert "https://www.wearetala.com/pages/weak" not in urls  # below strong/medium
    assert "https://adanola.com/pages/about" not in urls  # wrong brand
    assert all(c.discovery_method == "existing_hydrated_source" for c in candidates)
