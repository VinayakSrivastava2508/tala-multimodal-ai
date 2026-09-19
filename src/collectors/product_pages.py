"""Day 2.6B: targeted TALA product-page enumeration, section extraction, and
per-product video-source discovery.

wearetala.com (Shopify) renders three server-side tabs per product page under
CSS classes `product-information__description--{wear,care,aware}`:
  - "wear"  -> product description, materials bullet list, best-for, model info
  - "care"  -> explicit garment-care instructions (wash/dry/iron/bleach)
  - "aware" -> material composition %, certification, factory/supplier info
This content is present in the initial server response (verified by direct
`requests.get` inspection) -- no JavaScript execution/Playwright is required
to read it. Product-page size-chart data IS populated via a client-side
widget with no static table in the HTML; per-product fit/size evidence here
instead comes from the "wear" tab's model-info line and the product's
Shopify `options`/`variants` (available sizes), which are genuinely present
in the static page and in `/products.json`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests

from src.collectors.hydration import BROWSER_HEADERS, can_fetch_robots
from src.collectors.official_media import HEADERS
from src.collectors.reference_documents import BOT_UA

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TALA_PRODUCT_DOMAINS = ["https://www.wearetala.com", "https://eu.tala.co.uk"]

# Maps the assignment's 7 target categories to wearetala.com's actual
# Shopify `product_type` values (verified against a 300-product catalog scan).
CATEGORY_MAP: Dict[str, List[str]] = {
    "leggings": ["DayFlex Leggings", "Leggings", "Seamless Leggings", "Flares"],
    "shorts": ["Shorts", "Womens Shorts", "Skorts"],
    "sports_bras": ["Sports Bras"],
    "tops": ["Activewear Vest Tops", "Tops", "Shirts", "Tank Tops", "T-Shirts",
             "Activewear Tank Tops", "Activewear Tops", "Vest Tops"],
    "outerwear": ["Hoodies", "Fleece", "Activewear Jackets", "Cardigans"],
    "dresses_or_lifestyle": ["Active Dresses", "Bikini", "Swimsuit", "Bikini Tops",
                              "Bikini Bottoms", "Womens Pyjama Bottoms",
                              "Womens Pyjama Shorts", "Pyjama Tops"],
    "accessories": ["Cap"],
}
_TYPE_TO_CATEGORY = {t: cat for cat, types in CATEGORY_MAP.items() for t in types}


@dataclass
class ProductCandidate:
    product_name: str = ""
    product_handle: str = ""
    canonical_url: str = ""
    category: str = ""
    collection: str = ""
    availability: str = "unknown"
    discovery_method: str = "products_json"
    base_url: str = ""
    product_type_raw: str = ""


def fetch_catalog(base_url: str, get=requests.get, max_pages: int = 5, page_size: int = 100, delay: float = 1.0) -> List[dict]:
    """Fetch the full public /products.json catalog (paginated)."""
    products_url = base_url.rstrip("/") + "/products.json"
    if not can_fetch_robots(products_url, ua=BOT_UA):
        return []
    all_products = []
    for page in range(1, max_pages + 1):
        try:
            resp = get(products_url, headers=HEADERS, params={"limit": page_size, "page": page}, timeout=15)
        except Exception:
            break
        if resp.status_code != 200:
            break
        try:
            products = resp.json().get("products", [])
        except Exception:
            break
        if not products:
            break
        all_products.extend(products)
        if len(products) < page_size:
            break
        time.sleep(delay)
    return all_products


def enumerate_diverse_products(
    base_url: str, get=requests.get, per_category: int = 3, max_pages: int = 5,
) -> List[ProductCandidate]:
    """Enumerate products from the public catalog, selecting up to
    `per_category` in-stock products per target category for diversity."""
    catalog = fetch_catalog(base_url, get=get, max_pages=max_pages)
    by_category: Dict[str, List[dict]] = {}
    for p in catalog:
        category = _TYPE_TO_CATEGORY.get(p.get("product_type", ""))
        if category is None:
            continue
        by_category.setdefault(category, []).append(p)

    out: List[ProductCandidate] = []
    for category, products in by_category.items():
        in_stock = [p for p in products if any(v.get("available") for v in p.get("variants", []))]
        pool = in_stock if in_stock else products
        for p in pool[:per_category]:
            availability = "in_stock" if any(v.get("available") for v in p.get("variants", [])) else "out_of_stock"
            out.append(ProductCandidate(
                product_name=p.get("title", ""), product_handle=p.get("handle", ""),
                canonical_url=f"{base_url.rstrip('/')}/products/{p.get('handle', '')}",
                category=category, collection="", availability=availability,
                discovery_method="products_json", base_url=base_url,
                product_type_raw=p.get("product_type", ""),
            ))
    return out


def fetch_product_page_html(url: str, get=requests.get, delay: float = 1.2) -> Optional[tuple[int, str]]:
    """Fetch a product page's HTML. Returns (status_code, html) or None if
    robots.txt disallows or the request fails."""
    if not can_fetch_robots(url, ua=BOT_UA):
        return None
    time.sleep(delay)
    try:
        resp = get(url, headers=BROWSER_HEADERS, timeout=15)
    except Exception:
        return None
    return resp.status_code, resp.text


# ── Section extraction (Wear / Care / Aware tabs) ──────────────────────────────

_SECTION_CLASS_RX = {
    "wear": re.compile(r'product-information__description--wear"[^>]*><div class="metafield-rich_text_field">(.*?)</div></div>', re.S),
    "care": re.compile(r'product-information__description--care"[^>]*><div class="metafield-rich_text_field">(.*?)</div></div>', re.S),
    "aware": re.compile(r'product-information__description--aware"[^>]*><div class="metafield-rich_text_field">(.*?)</div></div>', re.S),
}


def _strip_html_tags(fragment: str) -> str:
    from bs4 import BeautifulSoup
    return BeautifulSoup(fragment, "lxml").get_text(separator="\n").strip()


def extract_product_sections(html: str) -> Dict[str, str]:
    """Extract the Wear/Care/Aware tab text (plain text, tags stripped) from a
    wearetala.com product page. Returns {} keys for tabs not present on this
    product (not every product has all three)."""
    sections = {}
    for name, rx in _SECTION_CLASS_RX.items():
        m = rx.search(html)
        if m:
            text = _strip_html_tags(m.group(1))
            if text.strip():
                sections[name] = text.strip()
    return sections


def extract_model_info(wear_text: str) -> str:
    """Return the 'Model Info: ...' line from the wear-tab text, if present."""
    m = re.search(r"Model Info:\s*(.+)", wear_text)
    return m.group(1).strip() if m else ""


# ── Care-instruction extraction (Part C: explicit instructions only) ──────────

_CARE_TEMPERATURE_RX = re.compile(r"(\d{2,3})\s*[°Â]?\s*[CcFf]\b")
_WASH_METHOD_KEYWORDS = {
    "machine wash": "machine_wash", "hand wash": "hand_wash", "delicate": "delicate_cycle",
    "cold wash": "cold_wash", "do not wash": "do_not_wash",
}
_DRY_METHOD_KEYWORDS = {
    "do not tumble dry": "do_not_tumble_dry", "tumble dry": "tumble_dry",
    "line dry": "line_dry", "dry flat": "dry_flat", "do not dry clean": "do_not_dry_clean",
    "dry clean": "dry_clean",
}
_IRON_KEYWORDS = {"do not iron": "do_not_iron", "iron on low": "iron_low", "iron": "iron"}
_BLEACH_KEYWORDS = {
    "do not bleach": "do_not_bleach", "non-chlorine bleach": "non_chlorine_bleach_only",
    "bleach": "bleach_permitted",
}
_COLOUR_TRANSFER_KEYWORDS = ["wash with similar colours", "wash separately", "wash with like colours"]

# Explicit-instruction gate: at least one of these concrete signals must be
# present, or the text is rejected as generic prose (Part C).
_EXPLICIT_CARE_SIGNAL_RX = re.compile(
    r"(machine wash|hand wash|do not tumble|tumble dry|do not iron|iron on|do not bleach|"
    r"non-chlorine bleach|dry clean|line dry|dry flat|\d{2,3}\s*[°Â]?\s*[CcFf]\b|"
    r"wash with similar colours|wash separately)", re.I)


def is_explicit_care_instruction(text: str) -> bool:
    """Part C gate: generic prose like 'care for your garments' is rejected --
    at least one concrete, actionable care signal must be present."""
    return bool(_EXPLICIT_CARE_SIGNAL_RX.search(text))


def extract_care_fields(care_text: str) -> Dict[str, str]:
    """Structured care fields from explicit instruction text only. Returns ''
    for any field with no explicit signal -- never inferred."""
    if not is_explicit_care_instruction(care_text):
        return {
            "care_temperature": "", "washing_method": "", "drying_method": "",
            "ironing_guidance": "", "bleaching_guidance": "", "special_handling": "",
        }
    text_l = care_text.lower()
    temp_match = _CARE_TEMPERATURE_RX.search(care_text)
    washing = next((v for k, v in _WASH_METHOD_KEYWORDS.items() if k in text_l), "")
    drying = next((v for k, v in _DRY_METHOD_KEYWORDS.items() if k in text_l), "")
    ironing = next((v for k, v in _IRON_KEYWORDS.items() if k in text_l), "")
    bleaching = next((v for k, v in _BLEACH_KEYWORDS.items() if k in text_l), "")
    colour_transfer = next((k for k in _COLOUR_TRANSFER_KEYWORDS if k in text_l), "")
    return {
        "care_temperature": f"{temp_match.group(1)}C" if temp_match else "",
        "washing_method": washing, "drying_method": drying,
        "ironing_guidance": ironing, "bleaching_guidance": bleaching,
        "special_handling": colour_transfer,
    }


# ── Material composition (Aware tab) ───────────────────────────────────────────

def extract_material_composition(aware_text: str, wear_text: str = "") -> str:
    """Return a short material-composition summary string from the Aware/Wear
    tab text (e.g. '78% recycled Polyamide'). Empty if no percentage+material
    pattern is present -- never inferred."""
    combined = f"{aware_text}\n{wear_text}"
    m = re.search(r"(\d{1,3}%\s*[A-Za-z][A-Za-z \-]{2,30})", combined)
    return m.group(1).strip() if m else ""


# ── Per-product video-source discovery (Part E) ────────────────────────────────

_VIDEO_SOURCE_RX = re.compile(r'<source[^>]*src="([^"]+\.mp4[^"]*)"[^>]*type="([^"]*)"', re.I)
_VIDEO_ID_RX = re.compile(r"/vp/([a-f0-9]{32})/")


@dataclass
class VideoSourceCandidate:
    video_id: str = ""
    asset_url: str = ""
    mime_type: str = "video/mp4"
    bitrate_label: str = ""


def discover_page_video_sources(html: str) -> List[VideoSourceCandidate]:
    """Extract every distinct embedded <source src=.mp4> on a product page,
    grouped by the CDN's own video ID (the 32-hex UUID in the /vp/<id>/ path)
    -- a page typically embeds several bitrate variants of the SAME video plus
    several genuinely DIFFERENT videos; grouping by ID keeps one representative
    per genuinely distinct video rather than treating every bitrate as new."""
    by_id: Dict[str, List[tuple[str, str]]] = {}
    for url, mime in _VIDEO_SOURCE_RX.findall(html):
        full_url = ("https:" + url) if url.startswith("//") else url
        m = _VIDEO_ID_RX.search(full_url)
        video_id = m.group(1) if m else full_url
        by_id.setdefault(video_id, []).append((full_url, mime))

    out = []
    for video_id, variants in by_id.items():
        # Prefer the lowest-bitrate variant (fastest to download; same content).
        def bitrate_of(u: str) -> float:
            bm = re.search(r"(\d+(?:\.\d+)?)Mbps", u)
            return float(bm.group(1)) if bm else 999.0
        variants.sort(key=lambda t: bitrate_of(t[0]))
        best_url, best_mime = variants[0]
        out.append(VideoSourceCandidate(video_id=video_id, asset_url=best_url, mime_type=best_mime))
    return out
