"""Genuine video processing: technical validation, frame sampling, frame-level
and temporal feature extraction (Day 2.5 Part F).

Uses cv2.VideoCapture (opencv-python 5.0.0 bundles its own FFmpeg-based decoder,
confirmed working in this environment via a VideoWriter/VideoCapture round-trip
test -- no system ffmpeg/ffprobe binary is installed here). Audio-stream
detection genuinely requires ffprobe (OpenCV's VideoCapture is video-only), so
has_audio_stream stays None/unknown in this environment rather than a fabricated
guess -- documented as a limitation, not silently defaulted to False.

No face detection, no protected-characteristic inference, no retained face
embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = PROJECT_ROOT / "data" / "media" / "video_frames"
VIDEO_EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings" / "video"

# Documented CLIP prompts for product/brand semantic similarity (Part F.4/F.5).
# A frame's product_semantic_similarity is the max cosine similarity of its CLIP
# image embedding against these text-prompt embeddings.
PRODUCT_SIMILARITY_PROMPTS = [
    "a product photo of activewear clothing",
    "a close-up of fabric material or a clothing label",
    "a person wearing activewear leggings or a sports bra",
    "a person trying on or unboxing a clothing package",
]

# A frame is counted as "product-related" for pct_frames_product_similar when its
# max similarity to the prompts above exceeds this threshold. CLIP cosine
# similarities between unrelated image/text pairs typically sit well below this.
PRODUCT_SIMILARITY_THRESHOLD = 0.22

# Perceptual-hash Hamming distance above which two consecutive frames count as a
# scene change (proxy, not a validated shot-boundary detector).
SCENE_CHANGE_HAMMING_THRESHOLD = 16  # out of 64 bits


def ffmpeg_available() -> bool:
    """Return True if a system ffmpeg/ffprobe binary is on PATH. cv2 itself does
    not need this (it bundles its own decoder), but genuine has_audio_stream
    detection does."""
    import shutil
    return shutil.which("ffprobe") is not None


def opencv_videocapture_available() -> bool:
    """Return True if cv2 exposes a working VideoCapture/VideoWriter (this
    project's opencv-python 5.0.0 build has some APIs missing -- e.g.
    CascadeClassifier -- so we check explicitly rather than assuming)."""
    try:
        import cv2
        return hasattr(cv2, "VideoCapture") and hasattr(cv2, "VideoWriter")
    except ImportError:
        return False


def probe_video(path: str | Path) -> Dict:
    """Return technical metadata for a video file. duration/width/height/fps/
    frame_count are None if the file can't be opened. has_audio_stream is None
    (unknown) unless ffprobe is available."""
    import cv2

    result = {
        "duration_seconds": None, "width": None, "height": None, "fps": None,
        "frame_count": None, "has_audio_stream": None, "aspect_ratio": None,
        "readable": False,
    }
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return result

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()

    duration = (frame_count / fps) if (frame_count and fps and fps > 0) else None
    result.update({
        "duration_seconds": duration, "width": width, "height": height, "fps": fps,
        "frame_count": frame_count,
        "aspect_ratio": round(width / height, 4) if width and height else None,
        "readable": True,
    })

    if ffmpeg_available():
        result["has_audio_stream"] = _ffprobe_has_audio(path)

    return result


def _ffprobe_has_audio(path: str | Path) -> Optional[bool]:
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def validate_video(path: str | Path) -> Tuple[bool, str]:
    """Validate a video file is genuinely processable: readable, non-zero
    duration, and contains multiple genuinely distinct frames (not e.g. a
    still image saved with a .mp4 extension)."""
    import cv2

    meta = probe_video(path)
    if not meta["readable"]:
        return False, "file_not_readable_by_opencv"
    if not meta["duration_seconds"] or meta["duration_seconds"] <= 0:
        return False, "zero_or_unknown_duration"
    if not meta["frame_count"] or meta["frame_count"] < 2:
        return False, "fewer_than_two_frames"

    cap = cv2.VideoCapture(str(path))
    first = _read_frame_at_index(cap, 0)
    last_idx = max(0, meta["frame_count"] - 1)
    last = _read_frame_at_index(cap, last_idx)
    cap.release()
    if first is None or last is None:
        return False, "could_not_sample_frames"
    if first.shape != last.shape:
        return False, "inconsistent_frame_shape"
    if np.array_equal(first, last):
        return False, "no_distinct_frames_detected"

    return True, ""


def _read_frame_at_index(cap, frame_index: int) -> Optional[np.ndarray]:
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sample_frame_positions(
    duration_seconds: float, fps: float, frame_count: int, interval_seconds: float = 10.0,
) -> List[Tuple[str, int, float]]:
    """Return [(sample_position, frame_index, timestamp_seconds), ...] for
    opening/25%/50%/75%/closing, plus additional 'interval' frames every
    `interval_seconds` for videos longer than 2x that interval."""
    fractions = [("opening", 0.0), ("quarter", 0.25), ("half", 0.5),
                 ("three_quarter", 0.75), ("closing", 1.0)]
    positions = []
    for label, frac in fractions:
        idx = min(frame_count - 1, max(0, round(frac * (frame_count - 1))))
        ts = idx / fps if fps else frac * duration_seconds
        positions.append((label, idx, round(ts, 3)))

    if duration_seconds > 2 * interval_seconds:
        t = interval_seconds
        existing_idx = {p[1] for p in positions}
        while t < duration_seconds:
            idx = min(frame_count - 1, round(t * fps)) if fps else None
            if idx is not None and idx not in existing_idx:
                positions.append(("interval", idx, round(t, 3)))
                existing_idx.add(idx)
            t += interval_seconds

    positions.sort(key=lambda p: p[1])
    return positions


def extract_frames(path: str | Path, positions: List[Tuple[str, int, float]]) -> List[Dict]:
    """Extract the requested frame indices. Returns a list of dicts with keys
    sample_position, frame_index, timestamp_seconds, frame_array (or None if
    that specific frame couldn't be read)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    for label, idx, ts in positions:
        arr = _read_frame_at_index(cap, idx)
        out.append({"sample_position": label, "frame_index": idx, "timestamp_seconds": ts, "frame_array": arr})
    cap.release()
    return out


# ── Perceptual hash (manual aHash -- imagehash is not installed) ──────────────

def average_hash(image_array: np.ndarray, hash_size: int = 8) -> str:
    """Simple average-hash perceptual hash: resize to hash_size x hash_size
    grayscale, threshold against the mean, return a hex string. A lightweight
    manual implementation since the `imagehash` package is not installed."""
    from PIL import Image
    img = Image.fromarray(image_array).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = np.asarray(img, dtype=np.float32)
    mean = pixels.mean()
    bits = (pixels > mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return format(value, f"0{hash_size * hash_size // 4}x")


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Bit-level Hamming distance between two equal-length hex hash strings."""
    int_a, int_b = int(hash_a, 16), int(hash_b, 16)
    return bin(int_a ^ int_b).count("1")


def save_frame(frame_array: np.ndarray, dest_path: Path) -> None:
    """Save an RGB frame array as a JPEG."""
    from PIL import Image
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_array).save(dest_path, format="JPEG", quality=90)


# ── Frame-level feature extraction ────────────────────────────────────────────

def extract_frame_features(frame_array: np.ndarray) -> Dict:
    """Interpretable frame-level features (brightness/contrast/saturation/edge
    density/dominant colour/perceptual hash). No face detection, no OCR unless
    a valid OCR engine exists (documented as unavailable in this environment --
    see docs/day2_evidence_hydration_audit.md pattern)."""
    from src.image_features import (
        get_image_brightness, get_image_contrast, get_image_saturation,
        get_edge_density, get_dominant_colors, dominant_color_group,
    )
    colors = get_dominant_colors(frame_array, n_colors=3)
    return {
        "brightness": round(get_image_brightness(frame_array), 2),
        "contrast": round(get_image_contrast(frame_array), 2),
        "saturation": round(get_image_saturation(frame_array), 2),
        "dominant_color_group": dominant_color_group(colors),
        "edge_density": round(get_edge_density(frame_array), 4),
        "perceptual_hash": average_hash(frame_array),
        "ocr_text": None,  # pytesseract not installed in this environment
    }


# ── Temporal feature aggregation ──────────────────────────────────────────────

def compute_temporal_features(frame_records: List[Dict]) -> Dict:
    """Compute temporal features across >=2 valid sampled frames. Returns all
    None if fewer than 2 valid frames are available -- temporal features
    genuinely require a sequence, never fabricated from a single frame."""
    valid = [f for f in frame_records if f.get("frame_array") is not None]
    out = {
        "mean_inter_frame_perceptual_distance": None, "clip_embedding_change_mean": None,
        "scene_change_count_proxy": None, "motion_intensity_proxy": None,
        "visual_consistency_score": None, "pct_frames_product_similar": None,
        "opening_to_closing_visual_change": None,
    }
    if len(valid) < 2:
        return out

    hashes = [f["perceptual_hash"] for f in valid if f.get("perceptual_hash")]
    if len(hashes) >= 2:
        dists = [hamming_distance(hashes[i], hashes[i + 1]) / 64 for i in range(len(hashes) - 1)]
        out["mean_inter_frame_perceptual_distance"] = round(float(np.mean(dists)), 4)
        out["scene_change_count_proxy"] = int(sum(
            1 for i in range(len(hashes) - 1)
            if hamming_distance(hashes[i], hashes[i + 1]) >= SCENE_CHANGE_HAMMING_THRESHOLD
        ))
        out["visual_consistency_score"] = round(1 - out["mean_inter_frame_perceptual_distance"], 4)
        out["opening_to_closing_visual_change"] = round(hamming_distance(hashes[0], hashes[-1]) / 64, 4)

    arrays = [f["frame_array"] for f in valid]
    if len(arrays) >= 2:
        diffs = []
        for i in range(len(arrays) - 1):
            a, b = arrays[i], arrays[i + 1]
            if a.shape == b.shape:
                diffs.append(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()) / 255)
        if diffs:
            out["motion_intensity_proxy"] = round(float(np.mean(diffs)), 4)

    embeddings = [f["clip_embedding"] for f in valid if f.get("clip_embedding") is not None]
    if len(embeddings) >= 2:
        from src.image_features import cosine_similarity
        changes = [1 - cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]
        out["clip_embedding_change_mean"] = round(float(np.mean(changes)), 4)

    similarities = [f["product_semantic_similarity"] for f in valid if f.get("product_semantic_similarity") is not None]
    if similarities:
        out["pct_frames_product_similar"] = round(
            sum(1 for s in similarities if s > PRODUCT_SIMILARITY_THRESHOLD) / len(similarities), 4)

    return out


def compute_aggregate_features(frame_records: List[Dict]) -> Dict:
    """Mean/variance of interpretable frame-level features, plus embedding
    aggregates, across all valid sampled frames."""
    valid = [f for f in frame_records if f.get("frame_array") is not None]
    out = {
        "mean_brightness": None, "brightness_variance": None,
        "mean_contrast": None, "contrast_variance": None,
        "mean_saturation": None, "saturation_variance": None,
        "mean_edge_density": None, "edge_density_variance": None,
        "max_semantic_similarity": None, "embedding_std": None,
        "mean_frame_embedding": None,
    }
    if not valid:
        return out

    for stat in ("brightness", "contrast", "saturation", "edge_density"):
        values = [f[stat] for f in valid if f.get(stat) is not None]
        if values:
            out[f"mean_{stat}"] = round(float(np.mean(values)), 4)
            out[f"{stat}_variance"] = round(float(np.var(values)), 4)

    similarities = [f["product_semantic_similarity"] for f in valid if f.get("product_semantic_similarity") is not None]
    if similarities:
        out["max_semantic_similarity"] = round(float(max(similarities)), 4)

    embeddings = [f["clip_embedding"] for f in valid if f.get("clip_embedding") is not None]
    if embeddings:
        stacked = np.stack(embeddings)
        out["mean_frame_embedding"] = stacked.mean(axis=0)
        out["embedding_std"] = round(float(stacked.std()), 4)

    return out
