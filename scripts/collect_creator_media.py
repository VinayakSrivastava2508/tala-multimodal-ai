"""Day 2 Task 2 Part B: acquire one public representative thumbnail per verified
creator post.

Reads the 61 usable_as_creator_post=True rows from creator_posts_enriched.csv and
attempts, per post: cached oEmbed thumbnail -> Open Graph image -> YouTube's public
thumbnail endpoint -> none. Never bypasses auth/anti-bot controls, never downloads a
full video. A failed acquisition never removes the underlying creator record.

Usage: python scripts/collect_creator_media.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.media import MEDIA_DIR, acquire_media_for_post  # noqa: E402

ENRICHED_PATH = PROJECT_ROOT / "data" / "interim" / "day2" / "creator_posts_enriched.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "creator_media_manifest.csv"


def main() -> int:
    print(f"Loading {ENRICHED_PATH.relative_to(PROJECT_ROOT)} ...")
    df = pd.read_csv(ENRICHED_PATH)
    verified = df[df["usable_as_creator_post"] == True].copy()
    print(f"  {len(verified)} verified creator posts")

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for i, (_, row) in enumerate(verified.iterrows()):
        post_id = str(row.get("source_id", f"row_{i}"))
        post_url = row.get("direct_post_url") or row.get("post_url") or row.get("source_url", "")
        if (i + 1) % 15 == 0 or i == 0:
            print(f"  [{i+1}/{len(verified)}] {post_id}")
        res = acquire_media_for_post(
            post_id=post_id,
            brand=str(row.get("brand", "")),
            platform=str(row.get("platform", "")),
            post_url=str(post_url),
            direct_post_id=str(row.get("direct_post_id", "") or ""),
        )
        results.append(res.to_dict())

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_PATH, index=False)

    n_success = int((out_df["retrieval_status"] == "success").sum())
    n_unavailable = int((out_df["retrieval_status"] == "unavailable").sum())
    n_failed = int((out_df["retrieval_status"] == "failed").sum())
    print(f"\nMedia acquisition: {n_success} success, {n_unavailable} unavailable, {n_failed} failed")
    print("By platform x status:")
    print(out_df.groupby(["platform", "retrieval_status"]).size().to_string())
    print(f"\nSaved: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Media files under: {MEDIA_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
