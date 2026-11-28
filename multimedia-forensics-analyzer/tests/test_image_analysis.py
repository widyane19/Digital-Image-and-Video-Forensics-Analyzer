"""Tests for image comparison and copy-move analysis helpers."""
from pathlib import Path
import sys

SRC_FOLDER = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_FOLDER))

import cv2
import numpy as np
import pytest
from PIL import Image

from image_analysis import (
    compare_images,
    create_match_visualization,
    edge_texture_analysis,
    orb_copy_move_detection,
    region_iou,
)


@pytest.fixture
def copy_move_image(tmp_path):
    """Create a textured image with one exact copied patch."""
    rng = np.random.default_rng(123)
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    # Textured patch gives ORB distinctive corners and local features to match.
    patch = rng.integers(0, 256, size=(90, 90, 3), dtype=np.uint8)
    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    image[30:120, 30:120] = patch
    image[180:270, 180:270] = patch
    # Add a different background so the copied patch is not the whole image.
    background = rng.integers(0, 80, size=image.shape, dtype=np.uint8)
    mask = np.ones(image.shape[:2], dtype=bool)
    mask[30:120, 30:120] = False
    mask[180:270, 180:270] = False
    image[mask] = background[mask]
    path = tmp_path / "copy_move.png"
    Image.fromarray(image).save(path)
    return path, (30, 30, 90, 90), (180, 180, 90, 90)


def test_compare_images_identical_and_changed(copy_move_image, tmp_path):
    path, _, _ = copy_move_image
    identical = compare_images(path, path)

    changed = np.asarray(Image.open(path).convert("RGB")).copy()
    changed[0:20, 0:20] = 255
    changed_path = tmp_path / "changed.png"
    Image.fromarray(changed).save(changed_path)
    different = compare_images(path, changed_path)

    assert identical["mse"] == 0
    assert identical["mean_absolute_difference"] == 0
    assert identical["identical"] is True
    assert different["mse"] > 0
    assert different["mean_absolute_difference"] > 0
    assert different["identical"] is False


def test_compare_images_rejects_different_dimensions(copy_move_image, tmp_path):
    path, _, _ = copy_move_image
    smaller = tmp_path / "smaller.png"
    Image.new("RGB", (64, 64)).save(smaller)

    with pytest.raises(ValueError, match="same dimensions"):
        compare_images(path, smaller)


def test_region_iou_known_cases():
    assert region_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert region_iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert region_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_copy_move_detector_returns_valid_candidate_structure(copy_move_image):
    path, _, _ = copy_move_image
    result = orb_copy_move_detection(path, min_matches=4)

    assert set(result) == {"match_count", "candidate_regions", "matches"}
    assert result["match_count"] >= 0
    assert isinstance(result["candidate_regions"], list)
    assert isinstance(result["matches"], list)
    for candidate in result["candidate_regions"]:
        assert candidate["match_count"] >= 4
        assert len(candidate["source"]) == 4
        assert len(candidate["destination"]) == 4
        assert len(candidate["displacement"]) == 2
    # Region-level scoring of this result is tested in tests/test_evaluation.py.


def test_copy_move_detector_rejects_invalid_parameters(copy_move_image):
    path, _, _ = copy_move_image
    with pytest.raises(ValueError, match="min_matches"):
        orb_copy_move_detection(path, min_matches=1)
    with pytest.raises(ValueError, match="ratio"):
        orb_copy_move_detection(path, ratio=1.5)
    with pytest.raises(ValueError, match="min_displacement_fraction"):
        orb_copy_move_detection(path, min_displacement_fraction=-0.1)


def test_copy_move_detector_finds_the_copied_patch_with_defaults(copy_move_image):
    path, source, destination = copy_move_image
    candidates = orb_copy_move_detection(path)["candidate_regions"]

    assert candidates, "the exact copied patch should be found"
    best = candidates[0]
    assert region_iou(best["source"], source) > 0.3
    assert region_iou(best["destination"], destination) > 0.3


@pytest.fixture
def tiled_image(tmp_path):
    """An unedited 'tiled floor': one textured tile repeated in a grid.

    Real repeated texture is never pixel-identical (sensor noise, lighting), so each
    tile gets its own noise. A pasted copy, by contrast, is an exact duplicate.
    """
    rng = np.random.default_rng(7)
    tile = cv2.GaussianBlur(rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8), (3, 3), 0)
    floor = np.tile(tile, (8, 8, 1)).astype(float) + rng.normal(0, 20, size=(320, 320, 3))
    path = tmp_path / "tiles.png"
    Image.fromarray(np.clip(floor, 0, 255).astype(np.uint8)).save(path)
    return path


def test_repeated_texture_is_not_mistaken_for_copy_move(tiled_image):
    old_settings = dict(min_matches=6, max_distance=55, ratio=None, min_displacement_fraction=0)
    assert orb_copy_move_detection(tiled_image, **old_settings)["candidate_regions"]

    assert orb_copy_move_detection(tiled_image)["candidate_regions"] == []


def test_edge_texture_analysis_returns_valid_measurements(copy_move_image):
    path, _, _ = copy_move_image
    result = edge_texture_analysis(path)

    assert 0.0 <= result["edge_density"] <= 1.0
    assert result["laplacian_variance"] >= 0
    assert result["edge_map"].shape == (320, 320)


def test_match_visualization_returns_image_and_saves_file(copy_move_image, tmp_path):
    path, _, _ = copy_move_image
    result = {"candidate_regions": [{
        "source": (20, 20, 40, 40),
        "destination": (180, 180, 40, 40),
    }]}
    output = tmp_path / "matches.png"
    visual = create_match_visualization(path, result, output)

    assert visual.shape == (320, 320, 3)
    assert output.is_file()
    with Image.open(output) as saved:
        assert saved.size == (320, 320)
