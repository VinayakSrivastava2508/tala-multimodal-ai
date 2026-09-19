"""Day 2.6B Parts E/F: download and genuinely process the official TALA
product-page videos discovered by scripts/discover_tala_product_references.py.

Builds data/interim/day2_6/video_assets_expanded.csv (Day 2.5's video_assets.csv
plus these newly discovered official videos) and appends genuine frame-level /
video-level temporal features to the existing
data/processed/video_frame_features.csv / video_level_features.csv (preserving
prior rows, matching the idempotency contract already established in
scripts/process_video_assets.py).

Usage: python scripts/process_tala_official_videos.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.hydration import BROWSER_HEADERS, can_fetch_robots  # noqa: E402
from src.collectors.reference_documents import BOT_UA  # noqa: E402
from src.video_features import (  # noqa: E402
    FRAMES_DIR,
    PRODUCT_SIMILARITY_PROMPTS,
    VIDEO_EMBEDDINGS_DIR,
    compute_aggregate_features,
    compute_temporal_features,
    extract_frame_features,
    extract_frames,
    probe_video,
    sample_frame_positions,
    save_frame,
    validate_video,
)
from src.image_features import clip_is_available, get_clip_embedding, get_clip_text_embedding, cosine_similarity  # noqa: E402

DISCOVERY_PATH = PROJECT_ROOT / "data" / "interim" / "day2_6" / "tala_official_video_discovery.csv"
DAY2_5_VIDEO_ASSETS_PATH = PROJECT_ROOT / "data" / "interim" / "day2_5" / "video_assets.csv"
EXPANDED_VIDEO_ASSETS_PATH = PROJECT_ROOT / "data" / "interim" / "day2_6" / "video_assets_expanded.csv"
FRAME_FEATURES_OUT = PROJECT_ROOT / "data" / "processed" / "video_frame_features.csv"
LEVEL_FEATURES_OUT = PROJECT_ROOT / "data" / "processed" / "video_level_features.csv"
VIDEO_MEDIA_DIR = PROJECT_ROOT / "data" / "media" / "official_videos" / "tala"

NOW_ISO = datetime.now(timezone.utc).isoformat()
MAX_DOWNLOADS = 10  # cap -- these are a shared content gallery, not per-product distinct


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def download_video(url: str, dest: Path, get=requests.get, delay: float = 1.5) -> tuple[bool, str]:
    if not can_fetch_robots(url, ua=BOT_UA):
        return False, "robots_disallowed"
    time.sleep(delay)
    try:
        resp = get(url, headers=BROWSER_HEADERS, timeout=60, stream=True)
        if resp.status_code != 200:
            return False, f"http_{resp.status_code}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return True, ""
    except Exception as exc:
        return False, str(exc)[:120]


def process_one_video(video_asset_id: str, local_path: Path) -> tuple[list[dict], dict]:
    meta = probe_video(local_path)
    is_valid, invalid_reason = validate_video(local_path)

    level_row = {
        "video_asset_id": video_asset_id,
        "duration_seconds": meta["duration_seconds"], "width": meta["width"], "height": meta["height"],
        "fps": meta["fps"], "frame_count": meta["frame_count"], "has_audio_stream": meta["has_audio_stream"],
        "aspect_ratio": meta["aspect_ratio"], "n_sampled_frames": 0,
    }
    if not is_valid:
        level_row["invalid_reason"] = invalid_reason
        return [], level_row

    positions = sample_frame_positions(meta["duration_seconds"], meta["fps"], meta["frame_count"])
    frames = extract_frames(local_path, positions)

    text_embeddings = None
    if clip_is_available():
        text_embeddings = [get_clip_text_embedding(p) for p in PRODUCT_SIMILARITY_PROMPTS]
        text_embeddings = [e for e in text_embeddings if e is not None]

    frame_dir = FRAMES_DIR / video_asset_id
    frame_rows = []
    for f in frames:
        if f["frame_array"] is None:
            frame_rows.append({
                "video_asset_id": video_asset_id, "frame_id": f"{video_asset_id}_{f['sample_position']}_{f['frame_index']}",
                "frame_index": f["frame_index"], "timestamp_seconds": f["timestamp_seconds"],
                "sample_position": f["sample_position"], "local_frame_path": "",
                "clip_embedding_available": False, "clip_embedding_file_hash": "",
            })
            continue

        interp = extract_frame_features(f["frame_array"])
        f.update(interp)
        frame_id = f"{video_asset_id}_{f['sample_position']}_{f['frame_index']}"
        frame_path = frame_dir / f"{frame_id}.jpg"
        save_frame(f["frame_array"], frame_path)

        clip_emb, emb_hash = None, ""
        if clip_is_available():
            clip_emb = get_clip_embedding(frame_path)
            if clip_emb is not None:
                emb_hash = hashlib.sha256(clip_emb.tobytes()).hexdigest()
                VIDEO_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
                np_save_path = VIDEO_EMBEDDINGS_DIR / f"{emb_hash}.npy"
                if not np_save_path.exists():
                    import numpy as np
                    np.save(np_save_path, clip_emb)
                f["clip_embedding"] = clip_emb

        product_sim = None
        if clip_emb is not None and text_embeddings:
            product_sim = round(max(cosine_similarity(clip_emb, t) for t in text_embeddings), 4)
        f["product_semantic_similarity"] = product_sim

        frame_rows.append({
            "video_asset_id": video_asset_id, "frame_id": frame_id,
            "frame_index": f["frame_index"], "timestamp_seconds": f["timestamp_seconds"],
            "sample_position": f["sample_position"],
            "local_frame_path": str(frame_path.relative_to(PROJECT_ROOT)),
            "brightness": interp["brightness"], "contrast": interp["contrast"], "saturation": interp["saturation"],
            "dominant_color_group": interp["dominant_color_group"], "edge_density": interp["edge_density"],
            "perceptual_hash": interp["perceptual_hash"], "ocr_text": interp["ocr_text"],
            "clip_embedding_available": clip_emb is not None, "clip_embedding_file_hash": emb_hash,
            "product_semantic_similarity": product_sim,
        })

    temporal = compute_temporal_features(frames)
    agg = compute_aggregate_features(frames)
    mean_emb_hash = ""
    if agg.get("mean_frame_embedding") is not None:
        import numpy as np
        mean_emb = agg.pop("mean_frame_embedding")
        mean_emb_hash = hashlib.sha256(mean_emb.tobytes()).hexdigest()
        VIDEO_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(VIDEO_EMBEDDINGS_DIR / f"mean_{mean_emb_hash}.npy", mean_emb)
    else:
        agg.pop("mean_frame_embedding", None)

    level_row.update({
        "n_sampled_frames": sum(1 for f in frames if f["frame_array"] is not None),
        "mean_frame_embedding_file_hash": mean_emb_hash,
        **agg, **temporal,
    })
    return frame_rows, level_row


def main() -> int:
    if not DISCOVERY_PATH.exists():
        print(f"{DISCOVERY_PATH} not found -- run scripts/discover_tala_product_references.py first.")
        return 1

    discovery = pd.read_csv(DISCOVERY_PATH)
    # Idempotency baseline: the expanded manifest from a PRIOR Day 2.6B run if one
    # exists (it carries every video_asset_id/asset_url this script has ever added),
    # else Day 2.5's original file on a first run. Deduping against Day 2.5 alone
    # would make every rerun re-download and re-process the same videos under the
    # same IDs, corrupting video_frame/level_features.csv with duplicate rows.
    baseline_path = EXPANDED_VIDEO_ASSETS_PATH if EXPANDED_VIDEO_ASSETS_PATH.exists() else DAY2_5_VIDEO_ASSETS_PATH
    day2_5 = _safe_read_csv(DAY2_5_VIDEO_ASSETS_PATH)
    baseline = _safe_read_csv(baseline_path)
    already_have_urls = set(baseline.get("asset_url", pd.Series(dtype=str)).dropna())
    already_have_hashes = set(baseline.get("file_hash", pd.Series(dtype=str)).dropna()) - {""}

    print(f"{len(discovery)} discovered video(s); downloading up to {MAX_DOWNLOADS}, "
          f"deduping against {len(already_have_urls)} already-known asset URL(s) "
          f"(baseline: {baseline_path.relative_to(PROJECT_ROOT)})")

    new_rows = []
    seen_hashes = set(already_have_hashes)
    n_downloaded = 0
    for i, r in discovery.iterrows():
        if n_downloaded >= MAX_DOWNLOADS:
            break
        if r["direct_media_url"] in already_have_urls:
            continue
        video_asset_id = f"tala_video_{100 + i:03d}"
        dest = VIDEO_MEDIA_DIR / f"{video_asset_id}.mp4"
        print(f"\n[{video_asset_id}] downloading {r['direct_media_url'][:90]} ...")
        ok, reason = download_video(r["direct_media_url"], dest)
        row = {
            "video_asset_id": video_asset_id, "brand": "TALA", "creator_handle": "",
            "product_name": "", "product_category": r["product_category"],
            "video_role": "other",  # honestly labelled: shared brand content gallery, not product-specific
            "source_platform": "official_website", "source_url": r["source_product_page"].split(";")[0],
            "asset_url": r["direct_media_url"], "local_path": "",
            "rights_or_access_basis": r["rights_or_access_basis"], "access_status": "accessible" if ok else "unavailable",
            "retrieval_method": r["discovery_method"], "retrieved_at": NOW_ISO,
            "duration_seconds": None, "file_hash": "", "transcript_available": False,
            "processing_status": "skipped", "evidence_strength": "unusable",
            "provenance_note": (
                f"shared content gallery item, featured on {r.get('n_products_featured_on', '?')} product page(s) "
                f"across categories: {r['product_category']}"
            ),
        }
        if ok:
            content_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
            if content_hash in seen_hashes:
                dest.unlink(missing_ok=True)
                row["provenance_note"] += "; duplicate_file_hash"
            else:
                seen_hashes.add(content_hash)
                row.update({
                    "local_path": str(dest.relative_to(PROJECT_ROOT)), "file_hash": content_hash,
                    "processing_status": "downloaded_pending_processing", "evidence_strength": "strong",
                })
                n_downloaded += 1
        else:
            row["provenance_note"] += f"; download_failed:{reason}"
        new_rows.append(row)

    combined = pd.concat([baseline, pd.DataFrame(new_rows)], ignore_index=True) if not baseline.empty else pd.DataFrame(new_rows)
    combined.to_csv(EXPANDED_VIDEO_ASSETS_PATH, index=False)
    print(f"\nSaved: {EXPANDED_VIDEO_ASSETS_PATH.relative_to(PROJECT_ROOT)} ({len(combined)} rows, "
          f"{n_downloaded} newly downloaded)")

    # ── Process every genuinely downloaded, not-yet-processed video ────────────
    eligible = combined[
        (combined["processing_status"] == "downloaded_pending_processing")
        & combined["local_path"].notna() & (combined["local_path"] != "")
    ].copy()
    print(f"\n{len(eligible)} video(s) eligible for local temporal processing")

    existing_frame_df = _safe_read_csv(FRAME_FEATURES_OUT)
    existing_level_df = _safe_read_csv(LEVEL_FEATURES_OUT)

    all_frame_rows, all_level_rows, processed_ids = [], [], []
    for _, r in eligible.iterrows():
        vid = r["video_asset_id"]
        local_path = PROJECT_ROOT / r["local_path"]
        print(f"\n[{vid}] Processing {local_path.name} ...")
        if not local_path.exists():
            print("  File missing on disk -- skipping")
            continue
        frame_rows, level_row = process_one_video(vid, local_path)
        all_frame_rows.extend(frame_rows)
        all_level_rows.append(level_row)
        if level_row.get("n_sampled_frames", 0) >= 2:
            processed_ids.append(vid)
            print(f"  OK: {level_row['n_sampled_frames']} frames, duration={level_row['duration_seconds']}s")
        else:
            print(f"  FAILED validation: {level_row.get('invalid_reason', 'unknown')}")

    frame_df = pd.concat([existing_frame_df, pd.DataFrame(all_frame_rows)], ignore_index=True)
    level_df = pd.concat([existing_level_df, pd.DataFrame(all_level_rows)], ignore_index=True)
    frame_df.to_csv(FRAME_FEATURES_OUT, index=False)
    level_df.to_csv(LEVEL_FEATURES_OUT, index=False)

    combined.loc[combined["video_asset_id"].isin(processed_ids), "processing_status"] = "processed"
    combined.loc[
        combined["video_asset_id"].isin(eligible["video_asset_id"]) & ~combined["video_asset_id"].isin(processed_ids),
        "processing_status"
    ] = "failed"
    combined.to_csv(EXPANDED_VIDEO_ASSETS_PATH, index=False)

    n_tala_processed = int((combined["processing_status"] == "processed").sum())
    n_categories = combined.loc[combined["processing_status"] == "processed", "product_category"].nunique()
    print(f"\nSaved: {FRAME_FEATURES_OUT.relative_to(PROJECT_ROOT)} ({len(frame_df)} frame rows)")
    print(f"Saved: {LEVEL_FEATURES_OUT.relative_to(PROJECT_ROOT)} ({len(level_df)} video rows)")
    print(f"Saved: {EXPANDED_VIDEO_ASSETS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"\nGenuinely processed TALA videos (all-time, in expanded manifest): {n_tala_processed}")
    print(f"Distinct product_category values among processed: {n_categories}")
    if n_tala_processed < 4:
        print(f"SHORTFALL vs target (>=4 processed TALA videos): have {n_tala_processed}. "
              "Reporting honestly -- no fabricated assets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
