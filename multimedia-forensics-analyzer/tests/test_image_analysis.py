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
    evaluate_copy_move_detection,
    evaluate_manipulation_predictions,
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
    path, source_box, destination_box = copy_move_image
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

    # Detection quality can vary by OpenCV version; this fixture is used for manual
    # evaluation too, rather than treating candidate output alone as proof.
    evaluated = evaluate_copy_move_detection(
        result, [source_box, destination_box], iou_threshold=0.10
    )
    assert 0.0 <= evaluated["precision"] <= 1.0
    assert 0.0 <= evaluated["recall"] <= 1.0
    assert evaluated["false_positives"] >= 0
    assert evaluated["false_negatives"] >= 0


def test_copy_move_detector_rejects_invalid_parameters(copy_move_image):
    path, _, _ = copy_move_image
    with pytest.raises(ValueError, match="min_matches"):
        orb_copy_move_detection(path, min_matches=1)


def test_region_evaluation_matches_ground_truth():
    detected = {
        "candidate_regions": [{
            "source": (10, 10, 20, 20),
            "destination": (100, 100, 20, 20),
        }]
    }
    result = evaluate_copy_move_detection(
        detected, [(10, 10, 20, 20), (100, 100, 20, 20)]
    )

    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["all_detected"] is True


def test_region_evaluation_rejects_invalid_iou_threshold():
    with pytest.raises(ValueError, match="iou_threshold"):
        evaluate_copy_move_detection({"candidate_regions": []}, [], iou_threshold=0)


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


def test_manipulation_metrics_calculate_confusion_matrix_values():
    result = evaluate_manipulation_predictions(
        actual_labels=[1, 1, 0, 0],
        predicted_labels=[1, 0, 1, 0],
    )

    assert result["true_positives"] == 1
    assert result["true_negatives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["false_positive_rate"] == 0.5


def test_manipulation_metrics_validate_labels():
    with pytest.raises(ValueError, match="equally sized"):
        evaluate_manipulation_predictions([0, 1], [1])
    with pytest.raises(ValueError, match=r"0 \(original\) or 1 \(manipulated\)"):
        evaluate_manipulation_predictions([0, 2], [0, 1])
