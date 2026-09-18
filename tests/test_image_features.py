"""Tests for image-statistic extraction (src/image_features.py). No live network
calls -- all images are synthetic arrays generated in-test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.image_features import (
    dominant_color_group,
    extract_interpretable_features,
    get_dominant_colors,
    get_edge_density,
    get_image_brightness,
    get_image_contrast,
    get_image_saturation,
    studio_vs_lifestyle_weak_label,
)


def _solid_image(color: tuple, h: int = 40, w: int = 40) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = color
    return arr


def test_brightness_white_vs_black():
    white = _solid_image((255, 255, 255))
    black = _solid_image((0, 0, 0))
    assert get_image_brightness(white) > get_image_brightness(black)
    assert get_image_brightness(black) == pytest.approx(0.0, abs=1e-6)
    assert get_image_brightness(white) == pytest.approx(255.0, abs=1e-6)


def test_contrast_zero_for_solid_color():
    solid = _solid_image((120, 120, 120))
    assert get_image_contrast(solid) == pytest.approx(0.0, abs=1e-6)


def test_contrast_nonzero_for_checkerboard():
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[::2, ::2] = 255
    assert get_image_contrast(arr) > 0


def test_saturation_grey_is_low_colour_is_high():
    grey = _solid_image((128, 128, 128))
    red = _solid_image((255, 0, 0))
    assert get_image_saturation(grey) < get_image_saturation(red)


def test_edge_density_solid_color_is_zero():
    solid = _solid_image((100, 100, 100))
    assert get_edge_density(solid) == pytest.approx(0.0, abs=1e-6)


def test_edge_density_checkerboard_is_positive():
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[::2, ::2] = 255
    assert get_edge_density(arr) > 0


def test_dominant_colors_returns_requested_count():
    arr = _solid_image((200, 50, 50))
    colors = get_dominant_colors(arr, n_colors=2)
    assert len(colors) == 2
    assert all(len(c) == 3 for c in colors)


def test_dominant_color_group_greyscale():
    assert dominant_color_group([(128, 128, 128)]) == "greyscale"


def test_dominant_color_group_red():
    assert dominant_color_group([(220, 20, 20)]) == "red"


def test_dominant_color_group_empty_input():
    assert dominant_color_group([]) == "unknown"


def test_studio_vs_lifestyle_uniform_border_is_studio():
    arr = _solid_image((240, 240, 240), h=60, w=60)
    assert studio_vs_lifestyle_weak_label(arr) == "studio"


def test_studio_vs_lifestyle_noisy_border_is_lifestyle():
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, size=(60, 60, 3), dtype=np.uint8)
    assert studio_vs_lifestyle_weak_label(arr) == "lifestyle"


def test_extract_interpretable_features_no_protected_characteristic_fields():
    """No age, gender, ethnicity, or attractiveness inference field may exist
    anywhere in the interpretable feature output."""
    arr = _solid_image((150, 100, 50), h=50, w=80)
    feats = extract_interpretable_features(arr)
    forbidden_substrings = ("age", "gender", "ethnic", "race", "attractive", "emotion")
    for key in feats:
        key_l = key.lower()
        for bad in forbidden_substrings:
            assert bad not in key_l, f"forbidden field found: {key}"


def test_extract_interpretable_features_has_expected_keys():
    arr = _solid_image((150, 100, 50), h=50, w=80)
    feats = extract_interpretable_features(arr)
    for key in ("width", "height", "aspect_ratio", "mean_brightness", "contrast",
                "saturation", "dominant_color_group", "edge_density",
                "text_area_indicator", "person_or_product_label",
                "studio_or_lifestyle_label", "screenshot_or_meme_indicator"):
        assert key in feats
    assert feats["width"] == 80
    assert feats["height"] == 50
