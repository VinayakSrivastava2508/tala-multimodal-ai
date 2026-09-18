"""Day 2 Task 2B Part A: enrich verified YouTube creator posts with real
video/channel statistics via YouTube Data API v3.

Reads YOUTUBE_API_KEY from .env (never printed/logged/cached). Batches videos.list
and channels.list calls at <=50 IDs. Missing/hidden statistics stay None, never 0.

Usage: python scripts/enrich_youtube_metrics.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.collectors.youtube_metrics import (  # noqa: E402
    extract_video_id,
    fetch_channels_metadata,
    fetch_videos_metadata,
)

ENRICHED_PATH = PROJECT_ROOT / "data" / "interim" / "day2" / "creator_posts_enriched.csv"
VIDEO_OUT = PROJECT_ROOT / "data" / "processed" / "youtube_video_metrics.csv"
CHANNEL_OUT = PROJECT_ROOT / "data" / "processed" / "youtube_channel_metrics.csv"


def main() -> int:
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        print("YOUTUBE_API_KEY not set in .env -- nothing to enrich. "
              "Downstream steps will fall back to null engagement fields.")
        pd.DataFrame(columns=["video_id"]).to_csv(VIDEO_OUT, index=False)
        pd.DataFrame(columns=["channel_id"]).to_csv(CHANNEL_OUT, index=False)
        return 0

    print("Loading verified YouTube creator posts ...")
    df = pd.read_csv(ENRICHED_PATH)
    verified = df[(df["usable_as_creator_post"] == True) & (df["platform"].str.lower() == "youtube")].copy()
    print(f"  {len(verified)} verified YouTube posts")

    video_ids = []
    for _, r in verified.iterrows():
        vid = r.get("direct_post_id") or extract_video_id(str(r.get("direct_post_url") or r.get("post_url") or ""))
        if vid:
            video_ids.append(str(vid))
    video_ids = list(dict.fromkeys(video_ids))
    print(f"  {len(video_ids)} unique video IDs extracted")

    print("\n[videos.list] Fetching video statistics (batched, cached) ...")
    video_records = fetch_videos_metadata(video_ids, api_key)
    video_df = pd.DataFrame(list(video_records.values()))
    video_df.to_csv(VIDEO_OUT, index=False)
    n_success = int((video_df["api_retrieval_status"] == "success").sum())
    print(f"  {n_success}/{len(video_df)} videos retrieved successfully")
    print(f"  Saved: {VIDEO_OUT.relative_to(PROJECT_ROOT)}")

    channel_ids = [c for c in video_df.loc[video_df["api_retrieval_status"] == "success", "channel_id"].unique() if c]
    print(f"\n[channels.list] Fetching channel statistics for {len(channel_ids)} unique channels ...")
    channel_records = fetch_channels_metadata(channel_ids, api_key)
    channel_df = pd.DataFrame(list(channel_records.values()))
    channel_df.to_csv(CHANNEL_OUT, index=False)
    print(f"  {len(channel_df)}/{len(channel_ids)} channels retrieved")
    n_hidden = int(channel_df["hidden_subscriber_count"].sum()) if not channel_df.empty else 0
    print(f"  {n_hidden} channel(s) have hidden subscriber counts")
    print(f"  Saved: {CHANNEL_OUT.relative_to(PROJECT_ROOT)}")
    print("  NOTE: channel_subscriber_count_current is a retrieval-time snapshot, "
          "NOT the subscriber count on the historical post date.")

    print("\nDone. Next: python scripts/build_multimodal_features.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
