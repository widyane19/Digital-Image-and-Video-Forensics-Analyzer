"""One entry point that runs every available detector on a file, for the web app.

Each detector is run separately: if one fails, its error is recorded as a warning
and the rest of the report is still returned.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image

from .compression_analysis import compression_diagnostics, create_compression_visualization
from .image_analysis import create_match_visualization, edge_texture_analysis, orb_copy_move_detection
from .metadata_analysis import analyze_file

# Visualizations are shrunk to this width before being sent to the browser.
MAX_PREVIEW_WIDTH = 1200


def _to_data_url(image: Image.Image | np.ndarray) -> str:
    """Encode an image as a PNG data URL the browser can show directly."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.width > MAX_PREVIEW_WIDTH:
        height = round(image.height * MAX_PREVIEW_WIDTH / image.width)
        image = image.resize((MAX_PREVIEW_WIDTH, height))
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _json_safe(value):
    """Turn dates, tuples and NumPy numbers into plain JSON values."""
    def default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        return str(obj)
    return json.loads(json.dumps(value, default=default))


def _image_detectors(path: str, warnings: list[str]) -> dict:
    """Run the pixel-based detectors. Each one that fails is skipped with a warning."""
    result: dict = {}

    try:
        detection = orb_copy_move_detection(path)
        result["copy_move"] = {
            "match_count": detection["match_count"],
            "candidate_regions": detection["candidate_regions"],
            "visualization": _to_data_url(create_match_visualization(path, detection)),
        }
    except Exception as exc:  # noqa: BLE001 - one detector must not break the report
        warnings.append(f"Copy-move detection failed: {exc}")

    try:
        result["compression"] = {
            **compression_diagnostics(path),
            "visualization": _to_data_url(create_compression_visualization(path)),
        }
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Compression analysis failed: {exc}")

    try:
        texture = edge_texture_analysis(path)
        result["edge_texture"] = {
            "edge_density": texture["edge_density"],
            "laplacian_variance": texture["laplacian_variance"],
            "edge_map": _to_data_url(texture["edge_map"]),
        }
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Edge and texture analysis failed: {exc}")

    return result


def analyze(path: str | os.PathLike, original_modified: datetime | None = None,
            display_name: str | None = None) -> dict:
    """Analyse an uploaded image or video and return one JSON-ready report.

    `original_modified` is the file date the browser reported, because saving the
    upload gave the file a new one. Raises UnsupportedMediaError for other file types.
    """
    path = str(path)
    metadata = analyze_file(path, uploaded=True, original_modified=original_modified).to_dict()
    if display_name:
        metadata["path"] = display_name

    report: dict = {
        "file_name": display_name or os.path.basename(path),
        "media_type": metadata["media_type"],
        "metadata": metadata,
        "warnings": [],
    }
    if metadata["media_type"] == "image":
        report["image"] = _image_detectors(path, report["warnings"])
    else:
        report["video_detectors"] = "not yet implemented"

    return _json_safe(report)
