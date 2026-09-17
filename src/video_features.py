"""Video feature extraction: frame sampling and thumbnail analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_frames(
    video_path: str | Path,
    n_frames: int = 8,
) -> List[np.ndarray]:
    """Sample n_frames evenly spaced frames from a video file.

    Returns a list of RGB numpy arrays (H, W, 3).
    Requires opencv-python.
    """
    import cv2  # lazy import
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def get_thumbnail_features(video_path: str | Path) -> dict:
    """Extract brightness and contrast from the first frame (thumbnail proxy).

    Returns a dict with keys: video_path, brightness, contrast.
    """
    from src.image_features import get_image_brightness, get_image_contrast  # lazy import
    frames = extract_frames(video_path, n_frames=1)
    if not frames:
        return {"video_path": str(video_path), "brightness": None, "contrast": None}
    arr = frames[0]
    return {
        "video_path": str(video_path),
        "brightness": get_image_brightness(arr),
        "contrast": get_image_contrast(arr),
    }


def get_video_duration_seconds(video_path: str | Path) -> Optional[float]:
    """Return video duration in seconds, or None if unreadable."""
    import cv2  # lazy import
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return total / fps
    return None


def extract_video_features_batch(
    video_paths: List[str | Path],
    n_frames: int = 4,
) -> "pd.DataFrame":
    """Extract thumbnail features and duration for a list of video files.

    Returns a DataFrame with one row per video.
    """
    import pandas as pd  # lazy import
    from src.image_features import get_dominant_colors  # lazy import
    records = []
    for path in video_paths:
        try:
            duration = get_video_duration_seconds(path)
            frames = extract_frames(path, n_frames=n_frames)
            if frames:
                from src.image_features import get_image_brightness, get_image_contrast
                brightnesses = [get_image_brightness(f) for f in frames]
                contrasts = [get_image_contrast(f) for f in frames]
                row = {
                    "video_path": str(path),
                    "duration_seconds": duration,
                    "avg_brightness": float(np.mean(brightnesses)),
                    "avg_contrast": float(np.mean(contrasts)),
                }
            else:
                row = {"video_path": str(path), "duration_seconds": duration, "avg_brightness": None, "avg_contrast": None}
            records.append(row)
        except Exception as exc:
            print(f"[video_features] Failed on {path}: {exc}")
    return pd.DataFrame(records)
