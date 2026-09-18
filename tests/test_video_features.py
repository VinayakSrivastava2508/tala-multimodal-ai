"""Tests for src/video_features.py (Day 2.5 genuine video processing).
Builds tiny synthetic .mp4 files with OpenCV -- no network, no real brand
video needed. Focused on the binding rules in docs/image_video_codebook.md:
a thumbnail/single-frame file is never a processed video, temporal features
require >=2 distinct frames, and metadata-only rows are never processed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

cv2 = pytest.importorskip("cv2")

from src.video_features import (
    average_hash,
    compute_temporal_features,
    hamming_distance,
    probe_video,
    sample_frame_positions,
    validate_video,
)


def _write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> None:
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def _solid_frame(color: tuple[int, int, int], size: int = 32) -> np.ndarray:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = color
    return arr


def _checkerboard_frame(offset: int, size: int = 32, block: int = 4) -> np.ndarray:
    """A checkerboard pattern (not a solid color) so perceptual-hash tests
    exercise genuine above/below-mean pixel variation."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            if ((x // block) + (y // block) + offset) % 2 == 0:
                arr[y, x] = (230, 230, 230)
            else:
                arr[y, x] = (20, 20, 20)
    return arr


# ── validate_video: zero-duration / single-frame / thumbnail rejection ────────

def test_validate_video_rejects_unreadable_file(tmp_path):
    fake = tmp_path / "not_a_video.mp4"
    fake.write_bytes(b"this is not a real video file")
    is_valid, reason = validate_video(fake)
    assert not is_valid
    assert reason == "file_not_readable_by_opencv"


def test_validate_video_rejects_single_frame_thumbnail_style_file(tmp_path):
    """A file with exactly one frame (e.g. a still image saved as .mp4, akin to
    a thumbnail) must be rejected -- it is not a genuine video asset."""
    path = tmp_path / "single_frame.mp4"
    _write_video(path, [_solid_frame((10, 20, 30))])
    is_valid, reason = validate_video(path)
    assert not is_valid
    assert reason in ("fewer_than_two_frames", "no_distinct_frames_detected")


def test_validate_video_rejects_identical_frames_even_if_multiple(tmp_path):
    """Many copies of the SAME frame (a static image looped into a video
    container) must not count as genuine motion/temporal content."""
    path = tmp_path / "static.mp4"
    frame = _solid_frame((50, 60, 70))
    _write_video(path, [frame] * 5)
    is_valid, reason = validate_video(path)
    assert not is_valid
    assert reason == "no_distinct_frames_detected"


def test_validate_video_accepts_genuinely_distinct_multi_frame_video(tmp_path):
    path = tmp_path / "real.mp4"
    frames = [_solid_frame((i * 20, i * 10, i * 5)) for i in range(6)]
    _write_video(path, frames)
    is_valid, reason = validate_video(path)
    assert is_valid
    assert reason == ""


# ── probe_video: zero-duration handling ───────────────────────────────────────

def test_probe_video_missing_file_returns_unreadable():
    meta = probe_video(Path("does_not_exist_anywhere.mp4"))
    assert meta["readable"] is False
    assert meta["duration_seconds"] is None


def test_probe_video_reports_duration_for_real_file(tmp_path):
    path = tmp_path / "real.mp4"
    frames = [_solid_frame((i * 20, i * 10, i * 5)) for i in range(10)]
    _write_video(path, frames, fps=10.0)
    meta = probe_video(path)
    assert meta["readable"] is True
    assert meta["duration_seconds"] is not None
    assert meta["duration_seconds"] > 0


# ── sample_frame_positions ────────────────────────────────────────────────────

def test_sample_frame_positions_requires_multiple_positions_for_short_video():
    positions = sample_frame_positions(duration_seconds=5.0, fps=10.0, frame_count=50)
    assert len(positions) >= 2
    indices = [p[1] for p in positions]
    assert len(set(indices)) >= 2  # distinct frame indices, not the same frame repeated


def test_sample_frame_positions_adds_interval_frames_for_long_video():
    short = sample_frame_positions(duration_seconds=5.0, fps=10.0, frame_count=50, interval_seconds=10.0)
    long = sample_frame_positions(duration_seconds=60.0, fps=10.0, frame_count=600, interval_seconds=10.0)
    assert len(long) > len(short)


# ── perceptual hash / hamming distance ────────────────────────────────────────

def test_average_hash_identical_images_have_zero_hamming_distance():
    frame = _solid_frame((100, 150, 200))
    h1 = average_hash(frame)
    h2 = average_hash(frame)
    assert hamming_distance(h1, h2) == 0


def test_average_hash_different_images_have_nonzero_hamming_distance():
    h1 = average_hash(_checkerboard_frame(offset=0))
    h2 = average_hash(_checkerboard_frame(offset=1))
    assert hamming_distance(h1, h2) > 0


# ── compute_temporal_features: fewer-than-two-frames guard ────────────────────

def test_compute_temporal_features_returns_all_none_for_single_frame():
    """Temporal features genuinely require a sequence -- must never be
    fabricated from a single frame (e.g. a thumbnail)."""
    frame_records = [{"frame_array": _solid_frame((1, 2, 3)), "perceptual_hash": average_hash(_solid_frame((1, 2, 3)))}]
    out = compute_temporal_features(frame_records)
    assert all(v is None for v in out.values())


def test_compute_temporal_features_returns_all_none_for_zero_frames():
    out = compute_temporal_features([{"frame_array": None}, {"frame_array": None}])
    assert all(v is None for v in out.values())


def test_compute_temporal_features_populates_for_two_distinct_frames():
    f1 = _checkerboard_frame(offset=0)
    f2 = _checkerboard_frame(offset=1)
    frame_records = [
        {"frame_array": f1, "perceptual_hash": average_hash(f1)},
        {"frame_array": f2, "perceptual_hash": average_hash(f2)},
    ]
    out = compute_temporal_features(frame_records)
    assert out["mean_inter_frame_perceptual_distance"] is not None
    assert out["mean_inter_frame_perceptual_distance"] > 0
    assert out["scene_change_count_proxy"] is not None
    assert out["opening_to_closing_visual_change"] is not None


# ── manifest-level rule: metadata-only rows never reach processed status ──────

def test_video_assets_manifest_never_marks_metadata_only_row_as_processed():
    import pandas as pd
    path = PROJECT_ROOT / "data" / "interim" / "day2_5" / "video_assets.csv"
    if not path.exists():
        pytest.skip("video_assets.csv not present in this environment")
    df = pd.read_csv(path)
    leads = df[df["rights_or_access_basis"] == "platform_metadata_only"]
    assert (leads["processing_status"] != "processed").all()


def test_video_level_features_rows_have_at_least_two_sampled_frames():
    import pandas as pd
    path = PROJECT_ROOT / "data" / "processed" / "video_level_features.csv"
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("video_level_features.csv not present in this environment")
    df = pd.read_csv(path)
    if df.empty:
        pytest.skip("no processed video rows yet")
    assert (df["n_sampled_frames"] >= 2).all()
