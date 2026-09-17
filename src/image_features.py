"""Image feature extraction: color palettes, CLIP embeddings, brightness/contrast."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def get_clip_embedding(
    image_path: str | Path,
    model_name: str = "openai/clip-vit-base-patch32",
) -> Optional[np.ndarray]:
    """Extract a CLIP image embedding via Hugging Face Transformers (free, no API key).

    Returns a 1D float32 ndarray of length 512, or None if transformers not installed.
    Falls back gracefully if transformers/torch are not available.
    """
    try:
        from PIL import Image  # lazy import
        from transformers import CLIPModel, CLIPProcessor  # lazy import
    except ImportError:
        print("[image_features] transformers/torch not installed. Skipping CLIP embedding.")
        return None

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    image = Image.open(str(image_path)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    import torch
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features.numpy().squeeze()


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
