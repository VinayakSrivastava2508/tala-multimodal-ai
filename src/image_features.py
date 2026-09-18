"""Image feature extraction: interpretable stats (brightness/contrast/colour/edges)
and CLIP embeddings. No face detection, no protected-characteristic inference."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"

# Named colour groups for the "dominant colour group" weak label -- coarse hue
# buckets, not a precise colour-naming model.
_COLOR_GROUPS = {
    "red": (0, 15), "orange": (15, 45), "yellow": (45, 65), "green": (65, 170),
    "cyan": (170, 200), "blue": (200, 260), "purple": (260, 290), "pink": (290, 330),
    "red2": (330, 360),
}


def load_image_as_array(image_path: str | Path) -> np.ndarray:
    """Load an image from disk and return an RGB numpy array (H, W, 3)."""
    from PIL import Image  # lazy import
    img = Image.open(str(image_path)).convert("RGB")
    return np.array(img)


def get_dominant_colors(
    image_array: np.ndarray,
    n_colors: int = 5,
) -> List[Tuple[int, int, int]]:
    """Extract dominant RGB color palette from an image using K-Means.

    Returns a list of (R, G, B) tuples of length n_colors.
    """
    from sklearn.cluster import KMeans  # lazy import
    pixels = image_array.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init="auto")
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in centers]


def get_image_brightness(image_array: np.ndarray) -> float:
    """Return mean luminance (0–255) of an RGB image."""
    # Rec. 601 luminance weights
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return float(luminance.mean())


def get_image_contrast(image_array: np.ndarray) -> float:
    """Return standard deviation of luminance as a contrast proxy (0–127.5)."""
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return float(luminance.std())


def get_image_saturation(image_array: np.ndarray) -> float:
    """Return mean HSV saturation (0-255) of an RGB image."""
    import cv2  # lazy import
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean())


def get_edge_density(image_array: np.ndarray) -> float:
    """Return the fraction of pixels detected as edges (Canny), 0-1. A rough proxy
    for visual busyness -- higher for text-heavy or cluttered images, lower for
    plain studio backgrounds."""
    import cv2  # lazy import
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float((edges > 0).mean())


def _rgb_to_hue(r: int, g: int, b: int) -> float:
    import colorsys
    h, _, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return h * 360


def dominant_color_group(dominant_colors: List[Tuple[int, int, int]]) -> str:
    """Map the single most-dominant RGB colour to a coarse named hue group
    (weak label -- coarse hue buckets, not a precise colour-naming model)."""
    if not dominant_colors:
        return "unknown"
    r, g, b = dominant_colors[0]
    if max(r, g, b) - min(r, g, b) < 15:
        return "greyscale"
    hue = _rgb_to_hue(r, g, b)
    for group, (lo, hi) in _COLOR_GROUPS.items():
        if lo <= hue < hi:
            return "red" if group == "red2" else group
    return "unknown"


def text_area_indicator(image_array: np.ndarray, edge_density_threshold: float = 0.08) -> bool:
    """Heuristic indicator that the image likely contains overlaid text/graphics:
    high edge density concentrated in horizontal bands. This is NOT true OCR --
    pytesseract is not installed in this environment (documented limitation, see
    docs/day2_evidence_hydration_audit.md). It flags likely text-heavy images for
    human review, it does not read the text."""
    return get_edge_density(image_array) > edge_density_threshold


def screenshot_or_meme_indicator(image_array: np.ndarray) -> bool:
    """Heuristic: very low colour variety + very high edge density (flat blocks of
    colour plus sharp text/UI edges) is typical of a screenshot or meme graphic
    rather than a photographed scene. A weak label, not a classifier."""
    edge_density = get_edge_density(image_array)
    r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    flat_fraction = float((np.abs(np.diff(luminance, axis=1)) < 2).mean())
    return edge_density > 0.05 and flat_fraction > 0.85


def person_present_weak_label(image_array: np.ndarray) -> str:
    """Weak 'product_only' vs 'person_present' label from a boolean face-region
    COUNT only. No face embedding, crop, or identity/attribute inference is ever
    computed or stored -- only whether at least one face-shaped region was
    detected, discarded immediately after counting.

    Uses cv2.CascadeClassifier (Haar cascade) when the installed OpenCV build
    exposes it. This project's environment has opencv-python 5.0.0, whose Python
    bindings do not expose CascadeClassifier/cv2.data at all (confirmed via
    `dir(cv2)` -- not a missing-file issue, the API surface itself is absent in
    this build). Rather than substitute a skin-tone-based heuristic (which would
    introduce racial bias) or download an additional DNN face-detector model file
    at runtime, this returns 'unknown' and the limitation is documented in
    docs/day2_evidence_hydration_audit.md -- fix is to pin
    opencv-contrib-python<5 or supply the YuNet ONNX model for cv2.FaceDetectorYN."""
    import cv2  # lazy import
    if not hasattr(cv2, "CascadeClassifier"):
        return "unknown"
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return "person_present" if len(faces) > 0 else "product_only"


def studio_vs_lifestyle_weak_label(image_array: np.ndarray) -> str:
    """Weak label from background-edge-density in the image border region: a
    plain/uniform border suggests a studio backdrop; a busy border suggests a
    lifestyle/environmental setting. Not a validated classifier."""
    h, w = image_array.shape[:2]
    border = max(4, min(h, w) // 10)
    top = image_array[:border, :, :]
    bottom = image_array[-border:, :, :]
    border_stack = np.concatenate([top.reshape(-1, 3), bottom.reshape(-1, 3)], axis=0)
    border_std = float(border_stack.std())
    return "studio" if border_std < 25 else "lifestyle"


def extract_interpretable_features(image_array: np.ndarray) -> dict:
    """One dict of all deterministic/weak-label interpretable visual features for
    a single loaded image array (no embeddings)."""
    h, w = image_array.shape[:2]
    colors = get_dominant_colors(image_array, n_colors=3)
    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 4) if h else None,
        "mean_brightness": round(get_image_brightness(image_array), 2),
        "contrast": round(get_image_contrast(image_array), 2),
        "saturation": round(get_image_saturation(image_array), 2),
        "dominant_color_group": dominant_color_group(colors),
        "edge_density": round(get_edge_density(image_array), 4),
        "text_area_indicator": text_area_indicator(image_array),
        "logo_or_brand_text_ocr_indicator": None,  # pytesseract not installed -- see docs limitation note
        "person_or_product_label": person_present_weak_label(image_array),
        "studio_or_lifestyle_label": studio_vs_lifestyle_weak_label(image_array),
        "screenshot_or_meme_indicator": screenshot_or_meme_indicator(image_array),
    }


# ── CLIP embeddings (interpretable-feature-only fallback if unavailable) ───────

_CLIP_MODEL_CACHE: dict = {}


def clip_is_available() -> bool:
    """Return True if transformers + torch are importable in this environment."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def get_clip_embedding(
    image_path: str | Path,
    model_name: str = "openai/clip-vit-base-patch32",
) -> Optional[np.ndarray]:
    """Extract a CLIP image embedding via Hugging Face Transformers (free, no API
    key), running locally on CPU. Pretrained only -- never fine-tuned.

    Returns a 1D float32 ndarray of length 512, or None if transformers/torch are
    not installed (falls back gracefully; caller should proceed with only the
    interpretable features in that case).
    """
    if not clip_is_available():
        print("[image_features] transformers/torch not installed. Skipping CLIP embedding.")
        return None

    from PIL import Image  # lazy import
    import torch  # lazy import
    from transformers import CLIPModel, CLIPProcessor  # lazy import

    if model_name not in _CLIP_MODEL_CACHE:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        model.eval()
        _CLIP_MODEL_CACHE[model_name] = (processor, model)
    processor, model = _CLIP_MODEL_CACHE[model_name]

    image = Image.open(str(image_path)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = model.get_image_features(**inputs)
    # transformers >=5 wraps this in a ModelOutput (pooler_output) instead of
    # returning a bare tensor as older versions did -- handle both.
    features = output.pooler_output if hasattr(output, "pooler_output") else output
    return features.numpy().squeeze().astype(np.float32)


def get_clip_text_embedding(
    text: str,
    model_name: str = "openai/clip-vit-base-patch32",
) -> Optional[np.ndarray]:
    """Extract a CLIP text embedding for a prompt string, using the same
    pretrained/cached model as get_clip_embedding (so cosine similarity between
    the two is meaningful). Returns None if transformers/torch unavailable."""
    if not clip_is_available():
        return None

    import torch  # lazy import
    from transformers import CLIPModel, CLIPProcessor  # lazy import

    if model_name not in _CLIP_MODEL_CACHE:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        model.eval()
        _CLIP_MODEL_CACHE[model_name] = (processor, model)
    processor, model = _CLIP_MODEL_CACHE[model_name]

    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model.get_text_features(**inputs)
    features = output.pooler_output if hasattr(output, "pooler_output") else output
    return features.numpy().squeeze().astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_or_compute_clip_embedding(
    image_path: str | Path,
    file_hash: str,
    cache_dir: Path = EMBEDDINGS_DIR,
    model_name: str = "openai/clip-vit-base-patch32",
) -> Optional[np.ndarray]:
    """CLIP embedding with on-disk caching keyed by the image's own content hash
    (so identical images across posts are only embedded once). Stores a .npy file
    per unique image under cache_dir/clip/ -- never inline in a feature CSV."""
    clip_dir = Path(cache_dir) / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    npy_path = clip_dir / f"{file_hash}.npy"
    if npy_path.exists():
        return np.load(npy_path)
    emb = get_clip_embedding(image_path, model_name=model_name)
    if emb is not None:
        np.save(npy_path, emb)
    return emb


def extract_image_features_batch(
    image_paths: List[str | Path],
    include_clip: bool = False,
) -> "pd.DataFrame":
    """Extract brightness, contrast, and dominant colors for a list of images.

    Returns a DataFrame with one row per image and columns:
    image_path, brightness, contrast, color_0..color_{n-1}.
    """
    import pandas as pd  # lazy import
    records = []
    for path in image_paths:
        try:
            arr = load_image_as_array(path)
            brightness = get_image_brightness(arr)
            contrast = get_image_contrast(arr)
            colors = get_dominant_colors(arr, n_colors=3)
            row: dict = {
                "image_path": str(path),
                "brightness": brightness,
                "contrast": contrast,
            }
            for i, color in enumerate(colors):
                row[f"color_{i}_r"] = color[0]
                row[f"color_{i}_g"] = color[1]
                row[f"color_{i}_b"] = color[2]
            if include_clip:
                emb = get_clip_embedding(path)
                if emb is not None:
                    for j, v in enumerate(emb):
                        row[f"clip_{j}"] = v
            records.append(row)
        except Exception as exc:
            print(f"[image_features] Failed on {path}: {exc}")
    return pd.DataFrame(records)
