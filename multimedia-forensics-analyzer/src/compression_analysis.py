"""Image validation and compression-related forensic measurements.

These measurements provide clues, not proof, that an image was edited or
recompressed. Interpret them alongside the image's source and known ground truth.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageOps, UnidentifiedImageError

SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}


def validate_image(image_path):
    """Validate that a supported image exists and can be decoded."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        with Image.open(path) as image:
            image_format = image.format
            image.verify()
        if image_format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {image_format}")
        with Image.open(path) as image:
            width, height = image.size
        return {
            "path": str(path),
            "format": image_format,
            "width": width,
            "height": height,
            "valid": True,
        }
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError(f"Invalid or unreadable image: {path}") from error


def load_image(image_path):
    """Load an image as RGB, applying its EXIF orientation when present."""
    validate_image(image_path)
    with Image.open(image_path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def compare_compression(original_path, compressed_path):
    """Compare decoded pixels using MSE and PSNR (not a manipulation verdict)."""
    original = np.asarray(load_image(original_path), dtype=np.float64)
    compressed = np.asarray(load_image(compressed_path), dtype=np.float64)
    if original.shape != compressed.shape:
        raise ValueError("Images must have the same dimensions to compare pixels.")

    mse = float(np.mean((original - compressed) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10((255.0 ** 2) / mse))
    return {
        "mse": mse,
        "psnr": psnr,
        "identical": mse == 0,
        "width": int(original.shape[1]),
        "height": int(original.shape[0]),
    }


def compression_diagnostics(image_path):
    """Measure 8-pixel grid-edge strength relative to other edges.

    JPEG uses 8x8 blocks, so unusually strong grid boundaries can be a clue.
    Image content also creates edges, so this score is not a JPEG detector by itself.
    """
    image = np.asarray(load_image(image_path).convert("L"), dtype=np.float32)
    if min(image.shape) < 16:
        raise ValueError("Image must be at least 16x16 pixels for block analysis.")

    # Differences between adjacent pixels; indices 7, 15, ... cross 8-pixel blocks.
    horizontal = np.abs(np.diff(image, axis=1))
    vertical = np.abs(np.diff(image, axis=0))
    x_boundary = np.zeros(horizontal.shape[1], dtype=bool)
    y_boundary = np.zeros(vertical.shape[0], dtype=bool)
    x_boundary[7::8] = True
    y_boundary[7::8] = True

    boundary_values = np.concatenate((horizontal[:, x_boundary].ravel(), vertical[y_boundary, :].ravel()))
    interior_values = np.concatenate((horizontal[:, ~x_boundary].ravel(), vertical[~y_boundary, :].ravel()))
    boundary_mean = float(boundary_values.mean()) if boundary_values.size else 0.0
    interior_mean = float(interior_values.mean()) if interior_values.size else 0.0

    return {
        "file_size_bytes": Path(image_path).stat().st_size,
        "boundary_mean_difference": boundary_mean,
        "interior_mean_difference": interior_mean,
        "blockiness_ratio": boundary_mean / max(interior_mean, 1e-8),
    }


def create_compression_visualization(image_path, output_path=None):
    """Show the image beside an amplified high-frequency detail map.

    The detail map helps inspect artifacts but can also highlight normal texture/edges.
    """
    image = load_image(image_path)
    # Subtract a lightly blurred version to reveal fine details and block boundaries.
    from PIL import ImageFilter

    blurred = image.filter(ImageFilter.GaussianBlur(radius=1.0))
    detail = ImageChops.difference(image, blurred)
    detail = ImageEnhance.Contrast(detail).enhance(4.0)
    canvas = Image.new("RGB", (image.width * 2, image.height), "white")
    canvas.paste(image, (0, 0))
    canvas.paste(detail, (image.width, 0))
    if output_path is not None:
        canvas.save(output_path)
    return canvas


def jpeg_quantization_tables(image_path):
    """Return JPEG quantization tables, or None for a non-JPEG image."""
    validate_image(image_path)
    with Image.open(image_path) as image:
        if image.format != "JPEG":
            return None
        tables = image.quantization
        if not tables:
            return None
        return {int(table_id): [int(value) for value in table] for table_id, table in tables.items()}


def dct_histogram_analysis(image_path, coefficient=(2, 3)):
    """Summarize a decoded-image 8x8 DCT coefficient histogram (exploratory only).

    This computes DCTs from decoded pixels; it does NOT read JPEG's original
    quantized coefficients and must not be treated as a validated double-JPEG test.
    """
    import cv2

    gray = np.asarray(load_image(image_path).convert("L"), dtype=np.float32)
    height = (gray.shape[0] // 8) * 8
    width = (gray.shape[1] // 8) * 8
    if height == 0 or width == 0:
        raise ValueError("Image must be at least 8x8 pixels for DCT analysis.")
    row, col = coefficient
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError("DCT coefficient indices must be between 0 and 7.")

    values = []
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            block = gray[y:y + 8, x:x + 8] - 128.0
            values.append(float(cv2.dct(block)[row, col]))

    values = np.rint(values).astype(np.int32)
    # Use fixed limits so extreme values cannot cause an unbounded histogram.
    limits = 2048
    clipped = np.clip(values, -limits, limits)
    histogram = np.bincount(clipped + limits, minlength=2 * limits + 1)
    spectrum = np.abs(np.fft.rfft(histogram.astype(np.float64)))
    if spectrum.size > 1:
        spectrum[0] = 0  # Ignore the histogram's DC component.
    peak_index = int(np.argmax(spectrum)) if spectrum.size else 0
    return {
        "coefficient": (row, col),
        "block_count": int(values.size),
        "histogram": histogram.tolist(),
        "periodicity_score": float(spectrum[peak_index] / max(histogram.sum(), 1)),
        "period": int(round(histogram.size / peak_index)) if peak_index > 0 else None,
    }


def analyze_recompression(original_path, recompressed_path):
    """Compare two same-sized images and report compression-related clues.

    Matching quantization tables do not prove a single compression; different tables
    do not by themselves prove tampering. The known original is important context.
    """
    original_info = validate_image(original_path)
    recompressed_info = validate_image(recompressed_path)
    if (original_info["width"], original_info["height"]) != (
        recompressed_info["width"], recompressed_info["height"]
    ):
        raise ValueError("Images must have the same dimensions to compare recompression.")

    original_tables = jpeg_quantization_tables(original_path)
    recompressed_tables = jpeg_quantization_tables(recompressed_path)
    tables_match = None
    if original_tables is not None and recompressed_tables is not None:
        tables_match = original_tables == recompressed_tables

    return {
        "pixel_comparison": compare_compression(original_path, recompressed_path),
        "original_blockiness": compression_diagnostics(original_path),
        "recompressed_blockiness": compression_diagnostics(recompressed_path),
        "original_quantization_tables": original_tables,
        "recompressed_quantization_tables": recompressed_tables,
        "quantization_tables_match": tables_match,
        "dct_histogram": dct_histogram_analysis(recompressed_path) if recompressed_tables is not None else None,
        "interpretation": "Measurements are clues only; they do not establish manipulation on their own.",
    }
