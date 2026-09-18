"""Day 2.5 Part F: genuine video processing pipeline.

Processes ONLY video_assets rows whose rights_or_access_basis permits local
processing (official_direct_public_asset, open_license, group_owned,
user_authorised) AND whose local_path points to a real, downloaded file.
platform_metadata_only rows are never processed -- they stay video leads.

Usage: python scripts/process_video_assets.py
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.video_features import (  # noqa: E402
    FRAMES_DIR,
    PRODUCT_SIMILARITY_PROMPTS,
    VIDEO_EMBEDDINGS_DIR,
    compute_aggregate_features,
    compute_temporal_features,
    extract_frame_features,
    extract_frames,
    ffmpeg_available,
    opencv_videocapture_available,
    probe_video,
    sample_frame_positions,
    save_frame,
    validate_video,
)
from src.image_features import clip_is_available, get_clip_embedding, get_clip_text_embedding, cosine_similarity  # noqa: E402

VIDEO_ASSETS_PATH = PROJECT_ROOT / "data" / "interim" / "day2_5" / "video_assets.csv"
FRAME_FEATURES_OUT = PROJECT_ROOT / "data" / "processed" / "video_frame_features.csv"
LEVEL_FEATURES_OUT = PROJECT_ROOT / "data" / "processed" / "video_level_features.csv"

PERMITTED_ACCESS_BASIS = ("official_direct_public_asset", "open_license", "group_owned", "user_authorised")
NOW_ISO = datetime.now(timezone.utc).isoformat()


def process_one_video(video_asset_id: str, local_path: Path) -> tuple[list[dict], dict]:
    """Process a single video file. Returns (frame_rows, level_row)."""
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
                "video_asset_id": video_asset_id, "frame_id": f"{video_asset_id}_{f['sample_position']}",
                "frame_index": f["frame_index"], "timestamp_seconds": f["timestamp_seconds"],
                "sample_position": f["sample_position"], "local_frame_path": "",
                "clip_embedding_available": False, "clip_embedding_file_hash": "",
            })
            continue

        interp = extract_frame_features(f["frame_array"])
        f.update(interp)

        frame_id = f"{video_asset_id}_{f['sample_position']}"
        frame_path = frame_dir / f"{frame_id}.jpg"
        save_frame(f["frame_array"], frame_path)

        clip_emb = None
        emb_hash = ""
        if clip_is_available():
            clip_emb = get_clip_embedding(frame_path)
            if clip_emb is not None:
                emb_hash = hashlib.sha256(clip_emb.tobytes()).hexdigest()
                VIDEO_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
                np.save(VIDEO_EMBEDDINGS_DIR / f"{emb_hash}.npy", clip_emb)
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
    print("Environment check:")
    print(f"  opencv VideoCapture available: {opencv_videocapture_available()}")
    print(f"  ffprobe (system) available:    {ffmpeg_available()} "
          f"{'(has_audio_stream can be determined)' if ffmpeg_available() else '(has_audio_stream will be None/unknown -- documented limitation)'}")
    print(f"  CLIP available:                {clip_is_available()}")

    if not VIDEO_ASSETS_PATH.exists():
        print(f"\n{VIDEO_ASSETS_PATH} not found -- run scripts/build_video_asset_manifest.py first.")
        return 1

    df = pd.read_csv(VIDEO_ASSETS_PATH)
    eligible = df[
        df["rights_or_access_basis"].isin(PERMITTED_ACCESS_BASIS)
        & (df["processing_status"] == "downloaded_pending_processing")
        & df["local_path"].notna() & (df["local_path"] != "")
    ].copy()
    print(f"\n{len(eligible)}/{len(df)} video_assets rows are eligible for local temporal processing "
          f"(permitted access basis + genuinely downloaded)")

    # Idempotency: rows already processing_status='processed' in a prior run are not
    # re-eligible above (their status is no longer 'downloaded_pending_processing'), so
    # this run must PRESERVE their existing feature rows rather than overwrite the output
    # files with only this run's (possibly empty) result.
    def _safe_read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    existing_frame_df = _safe_read_csv(FRAME_FEATURES_OUT)
    existing_level_df = _safe_read_csv(LEVEL_FEATURES_OUT)
    already_processed_ids = set(df.loc[df["processing_status"] == "processed", "video_asset_id"]) - set(eligible["video_asset_id"])
    if not existing_level_df.empty and already_processed_ids:
        existing_frame_df = existing_frame_df[existing_frame_df["video_asset_id"].isin(already_processed_ids)]
        existing_level_df = existing_level_df[existing_level_df["video_asset_id"].isin(already_processed_ids)]
        print(f"Preserving {len(already_processed_ids)} already-processed video(s) from a prior run: {sorted(already_processed_ids)}")
    else:
        existing_frame_df = pd.DataFrame()
        existing_level_df = pd.DataFrame()

    all_frame_rows = []
    all_level_rows = []
    processed_ids = []
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
            print(f"  OK: {level_row['n_sampled_frames']} frames, "
                  f"duration={level_row['duration_seconds']}s, fps={level_row['fps']}")
        else:
            print(f"  FAILED validation: {level_row.get('invalid_reason', 'unknown')}")

    frame_df = pd.concat([existing_frame_df, pd.DataFrame(all_frame_rows)], ignore_index=True)
    level_df = pd.concat([existing_level_df, pd.DataFrame(all_level_rows)], ignore_index=True)
    frame_df.to_csv(FRAME_FEATURES_OUT, index=False)
    level_df.to_csv(LEVEL_FEATURES_OUT, index=False)
    print(f"\nSaved: {FRAME_FEATURES_OUT.relative_to(PROJECT_ROOT)} ({len(frame_df)} frame rows)")
    print(f"Saved: {LEVEL_FEATURES_OUT.relative_to(PROJECT_ROOT)} ({len(level_df)} video rows)")

    # Update video_assets.csv processing_status for genuinely processed videos
    df.loc[df["video_asset_id"].isin(processed_ids), "processing_status"] = "processed"
    df.loc[
        df["video_asset_id"].isin(eligible["video_asset_id"]) & ~df["video_asset_id"].isin(processed_ids),
        "processing_status"
    ] = "failed"
    df.to_csv(VIDEO_ASSETS_PATH, index=False)
    print(f"\nGenuinely processed (>=2 valid frames): {len(processed_ids)}/{len(eligible)}")
    print(f"Updated: {VIDEO_ASSETS_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
