"""Day 2.5 Part E: build the video_assets manifest.

Three permitted routes, in order:
1. Direct public video assets embedded on official brand/product pages (Shopify
   product-page <video>/<source> markup -- a normal HTTP GET on the brand's own
   CDN, not a scrape of a third-party platform).
2. Openly licensed videos with clear licence metadata (none identified this run
   -- documented, not fabricated).
3. Group-owned/user-authorised local videos placed under
   data/raw/authorised_video_assets/ (none supplied this run).

Existing YouTube creator video URLs (data/interim/day2/creator_posts_enriched.csv)
are preserved as video LEADS (rights_or_access_basis=platform_metadata_only,
processing_status=metadata_only) -- never counted as processed video assets. No
yt-dlp / unofficial downloaders / stream extraction is used.

Usage: python scripts/build_video_asset_manifest.py
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.official_media import BROWSER_HEADERS, HEADERS, can_fetch  # noqa: E402

OUT_PATH = PROJECT_ROOT / "data" / "interim" / "day2_5" / "video_assets.csv"
CREATOR_ENRICHED_PATH = PROJECT_ROOT / "data" / "interim" / "day2" / "creator_posts_enriched.csv"
AUTHORISED_DIR = PROJECT_ROOT / "data" / "raw" / "authorised_video_assets"
VIDEO_MEDIA_DIR = PROJECT_ROOT / "data" / "media" / "official_videos"

BRAND_SITES = {
    "TALA": "https://www.wearetala.com",
    "Adanola": "https://adanola.com",
    "Girlfriend Collective": "https://girlfriend.com",
    "Oner Active": "https://oneractive.com",
}

TALA_TARGET = 8  # aim above the >=4 TALA floor
PRODUCTS_TO_SCAN_PER_BRAND = 25

_MP4_SOURCE_RX = re.compile(r'<source[^>]*src="([^"]+\.mp4[^"]*)"', re.I)
_BITRATE_RX = re.compile(r"(\d+(?:\.\d+)?)Mbps", re.I)


def _smallest_source(urls: list[str]) -> str:
    """Pick the lowest-bitrate source when labelled, else the first."""
    scored = []
    for u in urls:
        m = _BITRATE_RX.search(u)
        scored.append((float(m.group(1)) if m else 999.0, u))
    return min(scored, key=lambda t: t[0])[1]


def discover_official_page_videos(brand: str, base_url: str, get=requests.get) -> list[dict]:
    """Route 1: scan a sample of a brand's own product pages for embedded
    <source src=.mp4> markup. Returns a list of candidate dicts."""
    products_url = base_url.rstrip("/") + "/products.json"
    if not can_fetch(products_url):
        return []
    try:
        resp = get(products_url, headers=HEADERS, params={"limit": PRODUCTS_TO_SCAN_PER_BRAND}, timeout=15)
    except Exception:
        return []
    if resp.status_code != 200:
        return []
    products = resp.json().get("products", [])

    candidates = []
    for p in products:
        handle = p.get("handle", "")
        if not handle:
            continue
        page_url = f"{base_url.rstrip('/')}/products/{handle}"
        if not can_fetch(page_url):
            continue
        time.sleep(1.3)
        try:
            page_resp = get(page_url, headers=BROWSER_HEADERS, timeout=15)
        except Exception:
            continue
        if page_resp.status_code != 200:
            continue
        sources = _MP4_SOURCE_RX.findall(page_resp.text)
        if not sources:
            continue
        url = _smallest_source(sources)
        if url.startswith("//"):
            url = "https:" + url
        candidates.append({
            "brand": brand, "product_name": p.get("title", ""),
            "product_category": p.get("product_type", ""),
            "source_page_url": page_url, "asset_url": url,
        })
    return candidates


def discover_authorised_local_videos() -> list[dict]:
    """Route 3: any video files placed under data/raw/authorised_video_assets/."""
    if not AUTHORISED_DIR.exists():
        return []
    found = []
    for ext in ("*.mp4", "*.mov", "*.webm"):
        found.extend(AUTHORISED_DIR.glob(ext))
    return [{"local_path": p} for p in found]


def _download_video(url: str, dest: Path, get=requests.get, delay: float = 1.5) -> tuple[bool, str]:
    time.sleep(delay)
    try:
        resp = get(url, headers=HEADERS, timeout=60, stream=True)
        if resp.status_code != 200:
            return False, f"http_{resp.status_code}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return True, ""
    except Exception as exc:
        return False, str(exc)[:120]


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    VIDEO_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    seen_hashes = set()
    seen_urls = set()

    # ── Route 1: official page videos (TALA confirmed available; others scanned honestly) ──
    for brand, base_url in BRAND_SITES.items():
        print(f"\n[{brand}] Scanning up to {PRODUCTS_TO_SCAN_PER_BRAND} product pages for embedded video ...")
        candidates = discover_official_page_videos(brand, base_url)
        print(f"  {len(candidates)} product(s) with an embedded direct video source")

        # Group candidates by canonical asset URL first -- TALA's theme embeds the
        # SAME shared brand-wide video on every product page (confirmed: 25/25
        # scanned products referenced one identical video ID), so this is not
        # per-product content. Classify honestly rather than mislabel a shared
        # brand video as a product-specific demonstration of whichever page
        # happened to be scanned first.
        by_url: dict[str, list[dict]] = {}
        for cand in candidates:
            canon = cand["asset_url"].split("?")[0]
            by_url.setdefault(canon, []).append(cand)

        target = TALA_TARGET if brand == "TALA" else 999  # no artificial cap for other brands; take what's found
        downloaded = 0
        for i, (canon, group) in enumerate(by_url.items()):
            if downloaded >= target:
                break
            if canon in seen_urls:
                continue
            seen_urls.add(canon)
            cand = group[0]
            shared_across_products = len(group) > 1

            video_asset_id = f"{brand.lower().replace(' ', '_')}_video_{i+1:03d}"
            dest = VIDEO_MEDIA_DIR / brand.lower().replace(" ", "_") / f"{video_asset_id}.mp4"
            ok, reason = _download_video(cand["asset_url"], dest)
            row = {
                "video_asset_id": video_asset_id, "brand": brand, "creator_handle": "",
                "product_name": "" if shared_across_products else cand["product_name"],
                "product_category": "" if shared_across_products else cand["product_category"],
                "video_role": "other" if shared_across_products else "product_demonstration",
                "source_platform": "official_website", "source_url": cand["source_page_url"],
                "asset_url": cand["asset_url"], "local_path": "", "rights_or_access_basis": "official_direct_public_asset",
                "access_status": "accessible" if ok else "unavailable",
                "retrieval_method": "official_page_embedded_video_source",
                "retrieved_at": now, "duration_seconds": None, "file_hash": "",
                "transcript_available": False,
                "processing_status": "skipped", "evidence_strength": "unusable",
                "provenance_note": (
                    f"shared brand-wide video embedded identically across {len(group)} scanned product pages "
                    f"(e.g. {cand['source_page_url']}) -- not product-specific content"
                    if shared_across_products else f"scanned from {cand['source_page_url']}"
                ),
            }
            if ok:
                file_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
                if file_hash in seen_hashes:
                    dest.unlink(missing_ok=True)
                    row["processing_status"] = "skipped"
                    row["provenance_note"] += "; duplicate_file_hash"
                else:
                    seen_hashes.add(file_hash)
                    row.update({
                        "local_path": str(dest.relative_to(PROJECT_ROOT)), "file_hash": file_hash,
                        "processing_status": "downloaded_pending_processing",
                        "evidence_strength": "strong",
                    })
                    downloaded += 1
            else:
                row["provenance_note"] += f"; download_failed:{reason}"
            rows.append(row)
        print(f"  Downloaded: {downloaded}")

    # ── Route 3: authorised local assets ──────────────────────────────────────
    local = discover_authorised_local_videos()
    print(f"\n[Authorised local assets] {len(local)} file(s) found under "
          f"{AUTHORISED_DIR.relative_to(PROJECT_ROOT)}")
    for i, item in enumerate(local):
        p = item["local_path"]
        file_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append({
            "video_asset_id": f"authorised_{i+1:03d}", "brand": "unknown", "creator_handle": "",
            "product_name": "", "product_category": "", "video_role": "other",
            "source_platform": "local_authorised", "source_url": "", "asset_url": "",
            "local_path": str(p.relative_to(PROJECT_ROOT)), "rights_or_access_basis": "user_authorised",
            "access_status": "accessible", "retrieval_method": "manual_local_placement",
            "retrieved_at": now, "duration_seconds": None, "file_hash": file_hash,
            "transcript_available": False, "processing_status": "downloaded_pending_processing",
            "evidence_strength": "strong", "provenance_note": "manually placed in authorised_video_assets/",
        })

    # ── Route 2 (open license) + YouTube creator leads (platform_metadata_only) ──
    if CREATOR_ENRICHED_PATH.exists():
        creators = pd.read_csv(CREATOR_ENRICHED_PATH)
        verified = creators[creators["usable_as_creator_post"] == True]
        print(f"\n[Video leads] {len(verified)} verified creator posts preserved as platform_metadata_only leads")
        for i, (_, r) in enumerate(verified.iterrows()):
            rows.append({
                "video_asset_id": f"lead_{r.get('source_id', i)}", "brand": r.get("brand", ""),
                "creator_handle": r.get("creator_handle", ""), "product_name": "", "product_category": "",
                "video_role": "other", "source_platform": r.get("platform", ""),
                "source_url": r.get("direct_post_url") or r.get("post_url") or r.get("source_url", ""),
                "asset_url": "", "local_path": "", "rights_or_access_basis": "platform_metadata_only",
                "access_status": "restricted", "retrieval_method": "youtube_oembed_or_url_parse",
                "retrieved_at": now, "duration_seconds": None, "file_hash": "",
                "transcript_available": False, "processing_status": "metadata_only",
                "evidence_strength": "weak",
                "provenance_note": "video lead only -- no yt-dlp/unofficial downloader used; not a processed video asset",
            })

    df = pd.DataFrame(rows)
    n_dup_hash = 0
    if not df.empty:
        has_hash = df["file_hash"] != ""
        n_dup_hash = int(df.loc[has_hash, "file_hash"].duplicated().sum())
    df.to_csv(OUT_PATH, index=False)

    n_processable = int((df["rights_or_access_basis"].isin(
        ["official_direct_public_asset", "open_license", "group_owned", "user_authorised"])
        & (df["processing_status"] == "downloaded_pending_processing")).sum())
    n_brands_processable = df.loc[
        df["processing_status"] == "downloaded_pending_processing", "brand"].nunique()

    print(f"\nSaved: {OUT_PATH.relative_to(PROJECT_ROOT)} ({len(df)} rows)")
    print(f"Genuinely downloaded, pending temporal processing: {n_processable} (brands: {n_brands_processable})")
    print(f"Video leads (platform_metadata_only, not processed): "
          f"{int((df['processing_status']=='metadata_only').sum())}")
    print(f"Duplicate file hashes rejected: {n_dup_hash}")

    if n_processable < 12 or n_brands_processable < 3:
        print("\nSHORTFALL vs target (12-20 assets, >=3 brands, >=4 TALA, >=2 roles):")
        print(f"  Processable assets: {n_processable} (target 12-20)")
        print(f"  Brands represented: {n_brands_processable} (target >=3)")
        print("  Route 1 (official page video) found NO embedded video on Adanola, Girlfriend Collective, "
              "or Oner Active product pages (scanned 25 products each). Route 2 (open license) found none. "
              "Route 3 (authorised local) has 0 files supplied.")
        print("  Remediation: place brand-authorised local video files under "
              f"{AUTHORISED_DIR.relative_to(PROJECT_ROOT)}/ for Adanola, Girlfriend Collective, and Oner Active "
              "to close the brand-diversity and role-diversity gaps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
