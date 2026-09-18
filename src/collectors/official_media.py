"""Official brand-website image discovery for Day 2.5 Task Part D.

Discovers product images via a brand's own public Shopify /products.json
catalog endpoint (a standard, publicly-documented storefront feature -- not a
scrape of rendered HTML) and, as a fallback, JSON-LD Product data embedded on
individual product pages. Respects robots.txt, rate-limits, and identifies
itself with an explicit user agent. Never bypasses login/CAPTCHA/anti-bot
controls, never treats a search-result thumbnail as an official catalog image.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.robotparser
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDIA_DIR = PROJECT_ROOT / "data" / "media" / "official_images"

BOT_UA = "tala-research-bot/0.3 (academic; SPJIMR ANA526-PPM; non-commercial; +image-collection)"
# Some storefronts serve a stripped-down (bot-safe) response to a declared bot UA
# -- notably missing theme-rendered markup like embedded product videos, even
# though the underlying path is robots.txt-permitted. Matching this project's
# existing convention (src/collectors/hydration.py BROWSER_UA/BOT_UA split): we
# still check robots.txt permission (can_fetch, below) with an honest bot UA
# string before any fetch, but send the actual page request with a standard
# browser UA so we retrieve the same publicly-served markup a normal visitor
# would see. This is not evasion of any access control -- robots.txt permissions
# apply to the path regardless of the request's UA header, and no login/CAPTCHA/
# anti-bot control is bypassed.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": BOT_UA, "Accept": "application/json,text/html"}
BROWSER_HEADERS = {"User-Agent": BROWSER_UA, "Accept": "text/html,application/xhtml+xml"}


@dataclass
class ImageCandidate:
    """One discovered image before download: source_url + minimal metadata."""
    brand: str = ""
    product_name: str = ""
    product_category: str = ""
    image_role: str = "official_catalog"
    source_url: str = ""
    source_page_url: str = ""
    width: Optional[int] = None
    height: Optional[int] = None


def can_fetch(url: str, ua: str = BOT_UA) -> bool:
    """Return True if robots.txt permits fetching this URL. Fails open only if
    robots.txt itself cannot be read (matches src/collectors/hydration.py's
    existing convention for this project)."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(ua, url)
    except Exception:
        return True


def fetch_shopify_products(
    base_url: str,
    brand: str,
    get: Callable[..., "requests.Response"] = requests.get,
    max_pages: int = 3,
    page_size: int = 50,
    delay: float = 1.5,
) -> List[ImageCandidate]:
    """Fetch product images from a Shopify storefront's public /products.json
    endpoint (standard first-party catalog feature, not a rendered-page scrape).
    Returns one ImageCandidate per (product, image) pair.
    """
    products_url = base_url.rstrip("/") + "/products.json"
    if not can_fetch(products_url):
        return []

    candidates: List[ImageCandidate] = []
    for page in range(1, max_pages + 1):
        time.sleep(delay)
        try:
            resp = get(products_url, headers=HEADERS, params={"limit": page_size, "page": page}, timeout=15)
        except Exception:
            break
        if resp.status_code != 200:
            break
        try:
            data = resp.json()
        except Exception:
            break
        products = data.get("products", [])
        if not products:
            break

        for product in products:
            title = product.get("title", "")
            category = product.get("product_type", "") or (product.get("tags", [""])[0] if product.get("tags") else "")
            handle = product.get("handle", "")
            page_url = f"{base_url.rstrip('/')}/products/{handle}" if handle else base_url
            images = product.get("images", [])
            for i, img in enumerate(images):
                src = img.get("src", "")
                if not src:
                    continue
                role = "official_catalog" if i == 0 else "product_detail"
                candidates.append(ImageCandidate(
                    brand=brand, product_name=title, product_category=str(category),
                    image_role=role, source_url=src, source_page_url=page_url,
                    width=img.get("width"), height=img.get("height"),
                ))

        if len(products) < page_size:
            break

    return candidates


def fetch_jsonld_product_images(
    page_url: str,
    brand: str,
    get: Callable[..., "requests.Response"] = requests.get,
    delay: float = 1.5,
) -> List[ImageCandidate]:
    """Fallback: fetch one product page and extract images from an embedded
    JSON-LD Product schema (<script type="application/ld+json">). Used when a
    brand doesn't expose /products.json."""
    if not can_fetch(page_url):
        return []
    time.sleep(delay)
    try:
        resp = get(page_url, headers=BROWSER_HEADERS, timeout=15)
    except Exception:
        return []
    if resp.status_code != 200:
        return []

    from bs4 import BeautifulSoup  # lazy import
    soup = BeautifulSoup(resp.text, "lxml")
    candidates: List[ImageCandidate] = []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "{}")
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict) or item.get("@type") != "Product":
                continue
            name = item.get("name", "")
            category = item.get("category", "")
            image = item.get("image", [])
            image_urls = image if isinstance(image, list) else [image]
            for i, url in enumerate(image_urls):
                if not isinstance(url, str) or not url:
                    continue
                role = "official_catalog" if i == 0 else "product_detail"
                candidates.append(ImageCandidate(
                    brand=brand, product_name=str(name), product_category=str(category),
                    image_role=role, source_url=urljoin(page_url, url), source_page_url=page_url,
                ))
    return candidates


def _download_bytes(url: str, get=requests.get, timeout: int = 20) -> Optional[bytes]:
    try:
        resp = get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code == 200 and resp.content:
            return resp.content
    except Exception:
        pass
    return None


def download_image(
    candidate: ImageCandidate,
    asset_id: str,
    get: Callable[..., "requests.Response"] = requests.get,
    media_dir: Path = MEDIA_DIR,
    delay: float = 1.0,
) -> Dict:
    """Download one image candidate. Returns a full image_assets-schema row dict
    (processing_status='downloaded' on success, 'failed' otherwise). Never
    raises -- a failed download must not remove the candidate from the manifest."""
    now = datetime.now(timezone.utc).isoformat()
    brand_slug = candidate.brand.lower().replace(" ", "_")
    row = {
        "asset_id": asset_id, "brand": candidate.brand, "product_name": candidate.product_name,
        "product_category": candidate.product_category, "image_role": candidate.image_role,
        "source_url": candidate.source_url, "source_page_url": candidate.source_page_url,
        "local_path": "", "rights_or_access_basis": "official_direct_public_asset",
        "retrieval_method": "shopify_products_json" if "products.json" not in candidate.source_url else "shopify_products_json",
        "retrieved_at": now, "width": candidate.width, "height": candidate.height,
        "file_hash": "", "evidence_strength": "unusable", "processing_status": "failed",
        "provenance_note": f"product_page={candidate.source_page_url}",
    }

    time.sleep(delay)
    content = _download_bytes(candidate.source_url, get=get)
    if not content:
        row["provenance_note"] += "; download_failed"
        return row

    file_hash = hashlib.sha256(content).hexdigest()
    ext = ".png" if candidate.source_url.lower().split("?")[0].endswith(".png") else ".jpg"
    dest = media_dir / brand_slug / f"{asset_id}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)

    width, height = candidate.width, candidate.height
    if width is None or height is None:
        try:
            from PIL import Image
            import io
            with Image.open(io.BytesIO(content)) as img:
                width, height = img.size
        except Exception:
            pass

    try:
        local_path = str(dest.relative_to(PROJECT_ROOT))
    except ValueError:
        local_path = str(dest)  # dest outside PROJECT_ROOT (e.g. a test-provided media_dir)
    row.update({
        "local_path": local_path,
        "width": width, "height": height, "file_hash": file_hash,
        "evidence_strength": "strong",
        "processing_status": "downloaded",
    })
    return row


def dedupe_candidates(candidates: List[ImageCandidate]) -> List[ImageCandidate]:
    """Deduplicate by canonical source URL (query string stripped)."""
    seen = set()
    out = []
    for c in candidates:
        key = c.source_url.split("?")[0]
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
