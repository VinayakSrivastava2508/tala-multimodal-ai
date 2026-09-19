"""Day 2.6A Parts E/F/G: reference-document feature extraction.

Chunking, structured-entity extraction, and table/image capture for the
Multimodal Reference Package. Operates on rows of
data/interim/day2_6/multimodal_reference_assets.csv (already-fetched,
already-extracted text from src/collectors/reference_documents.py) -- this
module does not fetch anything itself.

No embeddings, no vector database here (Part F is explicit about that) --
this is chunk/table/image construction with provenance only.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# ── Section-aware chunking ─────────────────────────────────────────────────────

CHUNK_TARGET_WORDS_MIN = 250
CHUNK_TARGET_WORDS_MAX = 500

# A [page N] marker (inserted by src/collectors/reference_documents.py's PDF
# extraction) or a Markdown/HTML-derived heading-like line starts a new section.
_PAGE_MARKER_RX = re.compile(r"^\[page (\d+)\]$", re.I)
_HEADING_LIKE_RX = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Za-z0-9 /&\-]{2,60})$")


def split_into_sections(text: str) -> List[Dict]:
    """Split extracted_text into (section_title, page_number, body) blocks.
    A block with no detected heading gets section_title='' and page_number=None
    (or the page these lines belong to, for PDF text)."""
    lines = text.splitlines()
    sections: List[Dict] = []
    current_title = ""
    current_page: Optional[int] = None
    current_lines: List[str] = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"section_title": current_title, "page_number": current_page, "body": body})

    for line in lines:
        stripped = line.strip()
        page_match = _PAGE_MARKER_RX.match(stripped)
        if page_match:
            flush()
            current_lines = []
            current_page = int(page_match.group(1))
            continue
        if stripped and len(stripped) <= 80 and _HEADING_LIKE_RX.match(stripped) and stripped == stripped.strip():
            # Heuristic heading: short line, title-cased or markdown-hash-prefixed,
            # not itself a sentence (no trailing period).
            if not stripped.endswith((".", ",", ";")) and len(current_lines) > 0:
                flush()
                current_lines = []
            current_title = stripped.lstrip("#").strip()
            continue
        current_lines.append(line)

    flush()
    if not sections and text.strip():
        sections.append({"section_title": "", "page_number": None, "body": text.strip()})
    return sections


def chunk_section_body(body: str, target_min: int = CHUNK_TARGET_WORDS_MIN, target_max: int = CHUNK_TARGET_WORDS_MAX) -> List[str]:
    """Split a section's body into ~target_min-target_max word chunks, splitting
    on paragraph/sentence boundaries so a numerical claim ('30%') is never
    separated from its unit/qualifier mid-sentence. Never returns an empty chunk."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paragraphs:
        paragraphs = [body.strip()] if body.strip() else []

    chunks: List[str] = []
    current_words: List[str] = []
    for para in paragraphs:
        para_words = para.split()
        if len(current_words) + len(para_words) <= target_max:
            current_words.extend(para_words)
            if len(current_words) >= target_min:
                chunks.append(" ".join(current_words))
                current_words = []
        else:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
            if len(para_words) <= target_max:
                current_words = para_words
            else:
                # Oversized single paragraph (e.g. a dense policy clause): split by
                # sentence rather than mid-sentence, to avoid severing a claim
                # from its unit.
                sentences = re.split(r"(?<=[.!?])\s+", para)
                buf: List[str] = []
                for sent in sentences:
                    sent_words = sent.split()
                    if len(buf) + len(sent_words) > target_max and buf:
                        chunks.append(" ".join(buf))
                        buf = []
                    buf.extend(sent_words)
                if buf:
                    current_words = buf
    if current_words:
        chunks.append(" ".join(current_words))
    return [c for c in chunks if c.strip()]


# Chunk types that are exempt from the 250-500 word target because they carry
# dense, non-prose content (Part F: "use smaller chunks for tables, size
# charts and policy clauses").
SMALL_CHUNK_SECTION_KEYWORDS = ("size", "measurement", "table", "chart", "return window",
                                "return exclusion", "refund condition")


def build_chunks(
    reference_id: str, brand: str, reference_type: str, document_title: str,
    extracted_text: str, source_url: str, evidence_strength: str, retrieved_at: str,
) -> List[Dict]:
    """Build reference_document_chunks.csv rows for one reference document."""
    sections = split_into_sections(extracted_text)
    rows: List[Dict] = []
    seq = 0
    for section in sections:
        title_l = section["section_title"].lower()
        is_dense = any(kw in title_l for kw in SMALL_CHUNK_SECTION_KEYWORDS)
        chunk_texts = (
            [section["body"]] if is_dense and len(section["body"].split()) <= CHUNK_TARGET_WORDS_MAX
            else chunk_section_body(section["body"])
        )
        for chunk_text in chunk_texts:
            if not chunk_text.strip():
                continue
            entities = extract_entities(chunk_text)
            seq += 1
            rows.append({
                "chunk_id": f"{reference_id}_chunk_{seq:03d}",
                "reference_id": reference_id, "brand": brand, "reference_type": reference_type,
                "document_title": document_title, "section_title": section["section_title"],
                "page_number": section["page_number"], "chunk_sequence": seq,
                "chunk_text": chunk_text, "token_or_word_count": len(chunk_text.split()),
                "numerical_claims": ";".join(entities["percentages"] + entities["quantities"]),
                "materials": ";".join(entities["materials"]), "certifications": ";".join(entities["certifications"]),
                "geography": ";".join(entities["geography"]), "product_category": "",
                "source_url": source_url, "evidence_strength": evidence_strength,
                "retrieved_at": retrieved_at, "provenance_note": f"chunk {seq} of {reference_id}",
            })
    return rows


# ── Structured entity extraction (never inferred, only what's explicitly present) ─

MATERIAL_KEYWORDS = [
    "recycled polyester", "recycled nylon", "organic cotton", "cotton", "polyester", "nylon",
    "elastane", "spandex", "lycra", "merino wool", "wool", "bamboo", "tencel", "modal",
    "viscose", "polyamide", "econyl",
]
CERTIFICATION_KEYWORDS = [
    "oeko-tex", "bluesign", "grs", "gots", "fair wear", "b corp", "bcorp", "fsc",
    "recycled claim standard", "global recycled standard",
]
GEOGRAPHY_KEYWORDS = [
    "united kingdom", "uk", "china", "portugal", "turkey", "vietnam", "bangladesh",
    "india", "italy", "spain", "morocco", "cambodia", "sri lanka",
]
CARE_KEYWORDS = [
    "machine wash", "hand wash", "cold wash", "do not tumble dry", "do not bleach",
    "dry clean", "line dry", "iron on low", "wash inside out",
]

_PERCENT_RX = re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%")
_QUANTITY_UNIT_RX = re.compile(r"\b\d+(?:\.\d+)?\s?(kg|g|cm|mm|m|litres?|liters?|tonnes?|oz|lbs?)\b", re.I)
_TARGET_YEAR_RX = re.compile(r"\b(20[2-5]\d)\b")
_RETURN_WINDOW_RX = re.compile(r"\b(\d{1,3})\s*[- ]?day[s]?\b", re.I)


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract explicit structured entities from a chunk/document's text. Every
    value returned is a literal substring match -- nothing here is inferred or
    normalised beyond deduplication and lowercasing for keyword matches."""
    text_l = text.lower()
    return {
        "percentages": sorted(set(_PERCENT_RX.findall(text))),
        "quantities": sorted(set(m.group(0) for m in _QUANTITY_UNIT_RX.finditer(text))),
        "target_years": sorted(set(_TARGET_YEAR_RX.findall(text))),
        "materials": sorted({kw for kw in MATERIAL_KEYWORDS if kw in text_l}),
        "certifications": sorted({kw for kw in CERTIFICATION_KEYWORDS if kw in text_l}),
        "geography": sorted({kw for kw in GEOGRAPHY_KEYWORDS if kw in text_l}),
        "care_instructions": sorted({kw for kw in CARE_KEYWORDS if kw in text_l}),
        "return_windows": sorted(set(m.group(0) for m in _RETURN_WINDOW_RX.finditer(text))),
    }


# ── Table extraction (HTML) ────────────────────────────────────────────────────

def classify_table_type(headers: List[str], context_text: str = "") -> str:
    """Classify a table into the allowed table_type vocabulary from header text
    and surrounding context. Falls back to 'other', never fabricated."""
    haystack = " ".join(headers).lower() + " " + context_text.lower()[:500]
    if any(k in haystack for k in ("size", "uk 8", "us 4", "chest", "waist", "hip", "inseam")):
        return "size_chart"
    if any(k in haystack for k in ("material", "composition", "fabric", "%")):
        return "material_composition"
    if any(k in haystack for k in ("co2", "emission", "carbon")):
        return "emissions"
    if any(k in haystack for k in ("supplier", "factory", "manufactur")):
        return "supplier_list"
    if any(k in haystack for k in ("certif", "standard")):
        return "certification"
    if any(k in haystack for k in ("return", "refund", "exchange")):
        return "return_conditions"
    return "other"


def extract_html_tables(html: str, reference_id: str, source_url: str, evidence_strength: str) -> List[Dict]:
    """Extract <table> elements from raw HTML into reference_tables.csv rows.
    Uses pandas.read_html (which requires lxml/html5lib, both already project
    dependencies). A table that fails to parse is skipped, never fabricated."""
    try:
        import io
        import pandas as pd
        dfs = pd.read_html(io.StringIO(html))
    except Exception:
        return []

    rows = []
    for i, df in enumerate(dfs):
        if df.empty or df.shape[0] < 1:
            continue
        headers = [str(c) for c in df.columns]
        table_type = classify_table_type(headers)
        normalised_text = df.to_csv(index=False, sep="|")
        rows.append({
            "table_id": f"{reference_id}_table_{i + 1:02d}",
            "reference_id": reference_id, "page_number_or_section": "",
            "table_type": table_type, "headers": ";".join(headers),
            "rows_or_normalised_text": normalised_text[:4000],
            "units": "", "source_url": source_url, "extraction_method": "pandas_read_html",
            "evidence_strength": evidence_strength,
            "provenance_note": f"table {i + 1} extracted from {source_url}",
        })
    return rows


# ── Document images (HTML) ─────────────────────────────────────────────────────

DECORATIVE_IMAGE_HINTS = ("icon", "logo-small", "spacer", "arrow", "chevron", "sprite")


def is_decorative_image(img_url: str, alt_text: str) -> bool:
    """True if this <img> looks decorative (icon/logo/spacer) rather than an
    analytically meaningful reference image (a certification badge/mark IS kept
    -- see keep_despite_decorative_hint)."""
    haystack = f"{img_url} {alt_text}".lower()
    if any(k in haystack for k in ("certif", "badge", "bluesign", "oeko-tex", "b corp", "bcorp", "grs", "gots")):
        return False
    return any(hint in haystack for hint in DECORATIVE_IMAGE_HINTS)


def extract_html_images(html: str, reference_id: str, source_url: str, evidence_strength: str) -> List[Dict]:
    """Extract meaningful <img> references (not decorative icons/logos, unless
    they are certification/brand-identification evidence) into
    reference_images.csv candidate rows. Does NOT download bytes here --
    local_path/width/height/file_hash are populated by scripts/process_reference_package.py
    if/when the image is actually downloaded."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    rows = []
    seq = 0
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        alt = img.get("alt") or ""
        if not src:
            continue
        if is_decorative_image(src, alt):
            continue
        seq += 1
        role = "certification_badge" if any(
            k in f"{src} {alt}".lower() for k in ("certif", "badge", "bluesign", "oeko-tex", "b corp", "bcorp")
        ) else "document_figure"
        rows.append({
            "reference_image_id": f"{reference_id}_img_{seq:03d}",
            "reference_id": reference_id, "page_number_or_section": "",
            "image_role": role, "local_path": "", "caption": alt,
            "ocr_text": "", "width": None, "height": None, "file_hash": "",
            "source_url": src, "evidence_strength": evidence_strength,
            "provenance_note": f"discovered in HTML of {source_url}",
        })
    return rows
