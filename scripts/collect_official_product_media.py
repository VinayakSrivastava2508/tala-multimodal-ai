"""Day 2.5 Part D: collect official product images from each brand's own public
Shopify storefront (products.json / JSON-LD fallback). No search-result
thumbnails, no login/anti-bot bypass, no yt-dlp or unofficial downloaders.

Usage: python scripts/collect_official_product_media.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.official_media import (  # noqa: E402
    MEDIA_DIR,
    dedupe_candidates,
    download_image,
    fetch_jsonld_product_images,
    fetch_shopify_products,
)

OUT_PATH = PROJECT_ROOT / "data" / "interim" / "day2_5" / "image_assets.csv"

BRAND_SITES = {
    "TALA": "https://www.wearetala.com",
    "Adanola": "https://adanola.com",
    "Girlfriend Collective": "https://girlfriend.com",
    "Oner Active": "https://oneractive.com",
}

PER_BRAND_TARGET = 6  # aim above the >=4/brand floor to allow for download failures


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for brand, base_url in BRAND_SITES.items():
        print(f"\n[{brand}] Discovering product images from {base_url} ...")
        candidates = fetch_shopify_products(base_url, brand, max_pages=2, page_size=20)
        if not candidates:
            print(f"  products.json yielded nothing -- trying JSON-LD fallback on homepage")
            candidates = fetch_jsonld_product_images(base_url, brand)
        candidates = dedupe_candidates(candidates)
        print(f"  {len(candidates)} unique image candidates discovered")

        # Prefer catalog (position-1) images first, then fill with detail shots,
        # spread across distinct products rather than many images of one product.
        catalog = [c for c in candidates if c.image_role == "official_catalog"]
        detail = [c for c in candidates if c.image_role != "official_catalog"]
        ordered = catalog + detail
        to_download = ordered[:PER_BRAND_TARGET]

        n_success = 0
        for i, cand in enumerate(to_download):
            asset_id = f"{brand.lower().replace(' ', '_')}_{i+1:03d}"
            row = download_image(cand, asset_id)
            all_rows.append(row)
            if row["processing_status"] == "downloaded":
                n_success += 1
        print(f"  Downloaded: {n_success}/{len(to_download)}")

    df = pd.DataFrame(all_rows)
    # deduplicate by file_hash (identical image reused across product variants)
    if not df.empty:
        has_hash = df["file_hash"] != ""
        deduped_hash = df[has_hash].drop_duplicates(subset="file_hash", keep="first")
        df = pd.concat([deduped_hash, df[~has_hash]], ignore_index=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH.relative_to(PROJECT_ROOT)} ({len(df)} rows)")
    if not df.empty:
        print(df.groupby(["brand", "image_role", "processing_status"]).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
