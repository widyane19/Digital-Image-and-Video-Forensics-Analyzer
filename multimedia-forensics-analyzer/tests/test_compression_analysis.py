"""Tests for image validation and compression-analysis helpers."""
from pathlib import Path
import sys

SRC_FOLDER = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_FOLDER))

import numpy as np
import pytest
from PIL import Image

from compression_analysis import (
    analyze_recompression,
    compare_compression,
    compression_diagnostics,
    create_compression_visualization,
    dct_histogram_analysis,
    jpeg_quantization_tables,
    load_image,
    validate_image,
)


@pytest.fixture
def sample_images(tmp_path):
    """Create deterministic original, JPEG, and recompressed test images."""
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    original = tmp_path / "original.png"
    jpeg = tmp_path / "compressed.jpg"
    recompressed = tmp_path / "recompressed.jpg"
    Image.fromarray(pixels).save(original)
    Image.fromarray(pixels).save(jpeg, quality=95)
    with Image.open(jpeg) as image:
        image.save(recompressed, quality=40)
    return original, jpeg, recompressed


def test_validate_and_load_image(sample_images):
    original, _, _ = sample_images
    info = validate_image(original)
    loaded = load_image(original)

    assert info["valid"] is True
    assert info["format"] == "PNG"
    assert (info["width"], info["height"]) == (128, 128)
    assert loaded.mode == "RGB"
    assert loaded.size == (128, 128)


def test_missing_and_corrupt_images_raise_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_image(tmp_path / "missing.png")

    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid or unreadable image"):
        validate_image(corrupt)


def test_unsupported_format_is_rejected(tmp_path):
    gif_path = tmp_path / "sample.gif"
    Image.new("RGB", (32, 32)).save(gif_path, format="GIF")

    with pytest.raises(ValueError, match="Unsupported image format"):
        validate_image(gif_path)


def test_compare_compression_identical_and_different_images(sample_images):
    original, jpeg, _ = sample_images
    identical = compare_compression(original, original)
    different = compare_compression(original, jpeg)

    assert identical["mse"] == 0
    assert identical["psnr"] == float("inf")
    assert identical["identical"] is True
    assert different["mse"] > 0
    assert different["psnr"] > 0
    assert different["identical"] is False
    assert (different["width"], different["height"]) == (128, 128)


def test_pixel_comparison_requires_matching_dimensions(sample_images, tmp_path):
    original, _, _ = sample_images
    smaller = tmp_path / "smaller.png"
    Image.new("RGB", (64, 64)).save(smaller)

    with pytest.raises(ValueError, match="same dimensions"):
        compare_compression(original, smaller)


def test_compression_diagnostics_returns_finite_measurements(sample_images):
    _, jpeg, _ = sample_images
    result = compression_diagnostics(jpeg)

    assert result["file_size_bytes"] > 0
    assert result["boundary_mean_difference"] >= 0
    assert result["interior_mean_difference"] >= 0
    assert np.isfinite(result["blockiness_ratio"])


def test_compression_visualization_dimensions_and_save(sample_images, tmp_path):
    original, _, _ = sample_images
    output = tmp_path / "details.png"
    visual = create_compression_visualization(original, output)

    assert visual.size == (256, 128)
    assert output.is_file()
    with Image.open(output) as saved:
        assert saved.size == visual.size


def test_jpeg_quantization_tables_are_only_returned_for_jpeg(sample_images):
    original, jpeg, _ = sample_images

    tables = jpeg_quantization_tables(jpeg)
    assert tables is not None
    assert tables
    assert all(len(values) == 64 for values in tables.values())
    assert jpeg_quantization_tables(original) is None


def test_dct_histogram_has_expected_shape_and_count(sample_images):
    _, jpeg, _ = sample_images
    result = dct_histogram_analysis(jpeg, coefficient=(2, 3))

    assert result["coefficient"] == (2, 3)
    assert result["block_count"] == 16 * 16
    assert len(result["histogram"]) == 4097
    assert sum(result["histogram"]) == result["block_count"]
    assert np.isfinite(result["periodicity_score"])
    assert result["periodicity_score"] >= 0


def test_dct_histogram_rejects_invalid_coefficient(sample_images):
    _, jpeg, _ = sample_images
    with pytest.raises(ValueError, match="between 0 and 7"):
        dct_histogram_analysis(jpeg, coefficient=(8, 0))


def test_analyze_recompression_returns_documented_fields(sample_images):
    original, _, recompressed = sample_images
    result = analyze_recompression(original, recompressed)

    assert result["pixel_comparison"]["mse"] >= 0
    assert result["original_blockiness"]["file_size_bytes"] > 0
    assert result["recompressed_blockiness"]["file_size_bytes"] > 0
    assert result["original_quantization_tables"] is None
    assert result["recompressed_quantization_tables"]
    assert result["quantization_tables_match"] is None
    assert result["dct_histogram"]["block_count"] > 0
    assert "clues only" in result["interpretation"]


def test_analyze_recompression_requires_matching_dimensions(sample_images, tmp_path):
    original, _, _ = sample_images
    smaller = tmp_path / "smaller.png"
    Image.new("RGB", (32, 32)).save(smaller)

    with pytest.raises(ValueError, match="same dimensions"):
        analyze_recompression(original, smaller)
