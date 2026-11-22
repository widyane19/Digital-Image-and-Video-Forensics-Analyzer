"""One entry point that runs every available detector on a file, for the web app.

Results are grouped into four categories: metadata, image, compression and video
analysis. Every check, whatever detector it comes from, has the same shape:

    {"id", "name", "status", "severity", "result", "details", "visualization"}

where status is "flagged", "passed", "not_applicable" or "info" (a measurement
shown for context that does not decide anything by itself).

Each detector runs separately: if one fails, its error is recorded as a warning
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

from .compression_analysis import (
    compression_diagnostics, create_compression_visualization, dct_histogram_analysis,
    jpeg_quantization_tables,
)
from .image_analysis import create_match_visualization, edge_texture_analysis, orb_copy_move_detection
from .metadata_analysis import analyze_file

# Visualizations are shrunk to this width before being sent to the browser.
MAX_PREVIEW_WIDTH = 1200

# With this many metadata checks or fewer, there isn't enough to judge the file.
FEW_CHECKS = 3

# Metadata checks that belong to the other categories.
VIDEO_METADATA_CHECKS = ("creation_dates", "frame_count", "steady_frame_rate",
                         "audio_video_length", "frame_timing")
COMPRESSION_METADATA_CHECKS = ("compression_settings",)

CHECK_NAMES = {
    "editing_software": "Editing software",
    "modified_after_capture": "Modified after capture",
    "impossible_dates": "Impossible dates",
    "file_date_before_capture": "File date before capture",
    "size_mismatch": "Size mismatch",
    "metadata_removed": "Metadata removed",
    "preview_mismatch": "Embedded preview mismatch",
    "edit_history": "Edit history",
    "gps_time": "GPS time",
    "compression_settings": "JPEG compression settings",
    "creation_dates": "Creation dates",
    "frame_count": "Frame count",
    "steady_frame_rate": "Steady frame rate",
    "audio_video_length": "Audio and video length",
    "frame_timing": "Frame timing",
}

STATUS_FROM_METADATA = {"flagged": "flagged", "not_flagged": "passed", "not_applicable": "not_applicable"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _check(id: str, status: str, result: str, *, name: str | None = None, severity: str | None = None,
           details: dict | None = None, visualization: str | None = None) -> dict:
    return {
        "id": id,
        "name": name or CHECK_NAMES.get(id, id.replace("_", " ").capitalize()),
        "status": status,
        "severity": severity,
        "result": result,
        "details": details or {},
        "visualization": visualization,
    }


def _category(checks: list[dict], *, applicable: bool = True, note: str | None = None,
              failed: int = 0) -> dict:
    """Give a category its overall status and a one-line summary."""
    if not applicable:
        return {"status": "not_applicable", "summary": note or "Not applicable to this file.", "checks": checks}
    flagged = sum(c["status"] == "flagged" for c in checks)
    ran = sum(c["status"] in ("flagged", "passed", "info") for c in checks)
    if not checks and failed:
        status, summary = "error", "This analysis could not run on this file."
    elif flagged:
        status, summary = "flagged", f"{flagged} of {ran} check(s) found signs of editing."
    else:
        status, summary = "clear", f"No signs of editing found in {ran} check(s)."
    if note:
        summary = f"{summary} {note}"
    return {"status": status, "summary": summary, "checks": checks}


# ---------------------------------------------------------------------------
# The four categories
# ---------------------------------------------------------------------------

def _metadata_checks(metadata: dict, ids) -> list[dict]:
    """Turn the metadata analysis's results into checks, keeping only `ids`."""
    findings = {f["check"]: f for f in metadata["findings"]}
    checks = []
    for check_id, raw_status in metadata["results"].items():
        if check_id not in ids:
            continue
        status = STATUS_FROM_METADATA[raw_status]
        finding = findings.get(check_id)
        if finding:
            checks.append(_check(check_id, status, finding["message"], severity=finding["severity"],
                                 details=finding["evidence"]))
        elif status == "passed":
            checks.append(_check(check_id, status, "Nothing suspicious found."))
        else:
            checks.append(_check(check_id, status, "Not enough metadata to run this check."))
    # Flagged first, most serious first; then passed; then not applicable.
    order = {"flagged": 0, "passed": 1, "info": 2, "not_applicable": 3}
    severity = {"high": 0, "medium": 1, "low": 2, "info": 3, None: 4}
    checks.sort(key=lambda c: (order[c["status"]], severity[c["severity"]]))
    return checks


def _metadata_category(metadata: dict) -> dict:
    other = set(VIDEO_METADATA_CHECKS) | set(COMPRESSION_METADATA_CHECKS)
    ids = [name for name in metadata["results"] if name not in other]
    category = _category(_metadata_checks(metadata, ids))
    category["score"] = metadata["score"]
    category["checks_run"] = metadata["checks_run"]
    category["checks_total"] = metadata["checks_total"]
    category["conclusion"] = metadata["conclusion"]
    return category


def _image_category(path: str, is_image: bool, warnings: list[str]) -> tuple[dict, int]:
    """Copy-move detection and edge/texture measurements. Returns (category, copy-move candidates)."""
    if not is_image:
        return _category([], applicable=False, note="Pixel checks run on images only."), 0

    checks, failed, candidates = [], 0, 0
    try:
        detection = orb_copy_move_detection(path)
        candidates = len(detection["candidate_regions"])
        visual = _to_data_url(create_match_visualization(path, detection))
        if candidates:
            checks.append(_check(
                "copy_move", "flagged",
                f"{candidates} possible duplicated region pair(s) found and boxed in green. "
                "Repeating patterns can also match, so check them by eye.",
                name="Copy-move detection", severity="medium",
                details={"candidate_regions": candidates, "feature_matches": detection["match_count"]},
                visualization=visual))
        else:
            checks.append(_check(
                "copy_move", "passed", "No duplicated regions found.", name="Copy-move detection",
                details={"feature_matches": detection["match_count"]}, visualization=visual))
    except Exception as exc:  # noqa: BLE001 - one detector must not break the report
        failed += 1
        warnings.append(f"Copy-move detection failed: {exc}")

    try:
        texture = edge_texture_analysis(path)
        checks.append(_check(
            "edge_texture", "info",
            f"Edge density {texture['edge_density'] * 100:.1f}%, sharpness "
            f"{texture['laplacian_variance']:,.0f}. Look for areas that are unusually sharp or "
            "soft compared with their surroundings.",
            name="Edges and texture",
            details={"edge_density": round(texture["edge_density"], 4),
                     "laplacian_variance": round(texture["laplacian_variance"], 1)},
            visualization=_to_data_url(texture["edge_map"])))
    except Exception as exc:  # noqa: BLE001
        failed += 1
        warnings.append(f"Edge and texture analysis failed: {exc}")

    return _category(checks, failed=failed), candidates


def _compression_category(path: str, metadata: dict, is_image: bool, warnings: list[str]) -> dict:
    """Blockiness, quantization tables, a DCT histogram and the JPEG settings metadata check."""
    if not is_image:
        return _category([], applicable=False, note="Compression checks run on images only.")

    checks, failed = [], 0
    try:
        diag = compression_diagnostics(path)
        ratio = diag["blockiness_ratio"]
        meaning = ("strong 8×8 block edges, typical of JPEG compression" if ratio > 1.2
                   else "no strong 8×8 block pattern")
        checks.append(_check(
            "blockiness", "info", f"Blockiness ratio {ratio:.2f}: {meaning}.",
            name="Block pattern (8×8 grid)",
            details={"blockiness_ratio": round(ratio, 3), "file_size_bytes": diag["file_size_bytes"]},
            visualization=_to_data_url(create_compression_visualization(path))))
    except Exception as exc:  # noqa: BLE001
        failed += 1
        warnings.append(f"Block pattern analysis failed: {exc}")

    try:
        tables = jpeg_quantization_tables(path)
        if tables is None:
            checks.append(_check("quantization_tables", "not_applicable",
                                 "Not a JPEG, so there are no quantization tables.",
                                 name="Quantization tables"))
        else:
            checks.append(_check(
                "quantization_tables", "info",
                f"{len(tables)} quantization table(s) found. Their values show how strongly the "
                "image was compressed when it was last saved.",
                name="Quantization tables",
                details={"tables": len(tables), "first_values": tables[min(tables)][:8]}))
    except Exception as exc:  # noqa: BLE001
        failed += 1
        warnings.append(f"Quantization table analysis failed: {exc}")

    try:
        dct = dct_histogram_analysis(path)
        checks.append(_check(
            "dct_histogram", "info",
            f"Periodicity score {dct['periodicity_score']:.3f}. An exploratory measure of repeating "
            "patterns in the image's frequency data, which can hint at double compression. It is "
            "not calibrated, so read it alongside the other checks.",
            name="DCT histogram",
            details={"periodicity_score": round(dct["periodicity_score"], 4), "period": dct["period"],
                     "blocks": dct["block_count"]}))
    except Exception as exc:  # noqa: BLE001
        failed += 1
        warnings.append(f"DCT histogram analysis failed: {exc}")

    checks = _metadata_checks(metadata, COMPRESSION_METADATA_CHECKS) + checks
    return _category(checks, failed=failed)


def _video_category(metadata: dict, is_video: bool) -> dict:
    if not is_video:
        return _category([], applicable=False, note="Video checks run on videos only.")
    checks = _metadata_checks(metadata, VIDEO_METADATA_CHECKS)
    checks.append(_check(
        "frame_detectors", "not_applicable",
        "Duplicate-frame, temporal-change and per-frame compression detectors are not built yet.",
        name="Frame-level detectors"))
    return _category(checks, note="Frame-level detectors are not built yet.")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

LEVELS = ["no_strong_evidence", "some_signs", "likely_altered"]


def verdict(score: float, checks_run: int, pixel_evidence: int = 0) -> dict:
    """Combine the metadata score and the pixel/frame checks into one label.

    Only the metadata checks produce a calibrated score. `pixel_evidence` counts the
    copy-move candidates and flagged video detectors; any of them raises the label by
    one level, because they are visual evidence of editing. The label is "inconclusive"
    when almost nothing could be checked and nothing was found.
    """
    if score >= 0.6:
        level = 2
    elif score >= 0.3:
        level = 1
    else:
        level = 0
    if pixel_evidence:
        level = min(level + 1, 2)

    if level == 0 and checks_run <= FEW_CHECKS:
        return {"label": "inconclusive", "score": score}
    return {"label": LEVELS[level], "score": score}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze(path: str | os.PathLike, original_modified: datetime | None = None,
            display_name: str | None = None) -> dict:
    """Analyse an uploaded image or video and return one JSON-ready report.

    `original_modified` is the file date the browser reported, because saving the
    upload gave the file a new one. Raises UnsupportedMediaError for other file types.
    """
    path = str(path)
    metadata = analyze_file(path, uploaded=True, original_modified=original_modified).to_dict()
    media_type = metadata["media_type"]
    is_image, is_video = media_type == "image", media_type == "video"
    warnings = list(metadata["warnings"])

    image, candidates = _image_category(path, is_image, warnings)
    categories = {
        "metadata_analysis": _metadata_category(metadata),
        "image_analysis": image,
        "compression_analysis": _compression_category(path, metadata, is_image, warnings),
        "video_analysis": _video_category(metadata, is_video),
    }

    return _json_safe({
        "file_name": display_name or os.path.basename(path),
        "media_type": media_type,
        "verdict": verdict(metadata["score"], metadata["checks_run"], candidates),
        "categories": categories,
        "warnings": warnings,
    })
