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
from PIL import Image, ImageDraw

from .compression_analysis import (
    compression_diagnostics, create_compression_visualization, dct_histogram_analysis,
    jpeg_quantization_tables,
)
from .image_analysis import create_match_visualization, edge_texture_analysis, orb_copy_move_detection
from .metadata_analysis import analyze_file
from .video_analysis import analyze_video

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


def _times(findings: list[dict], limit: int = 5) -> str:
    """'at 0.67s, 0.70s and 3 more' for a check's message."""
    times = [f"{f['timestamp']:.2f}s" for f in findings]
    shown = ", ".join(times[:limit])
    return f"at {shown}" + (f" and {len(times) - limit} more" if len(times) > limit else "")


def _frame_difference_chart(analysis: dict, width: int = 900, height: int = 260) -> str:
    """Line chart of how much each frame differs from the one before, duplicates marked in red."""
    comparisons = analysis["comparisons"]
    times = [c["current_timestamp"] for c in comparisons]
    differences = [c["mean_pixel_difference"] for c in comparisons]
    left, right, top, bottom = 50, 15, 30, 35
    plot_w, plot_h = width - left - right, height - top - bottom
    t0, t1 = min(times), max(times)
    peak = max(max(differences), 1e-6)

    def point(t, value):
        x = left + (t - t0) / ((t1 - t0) or 1) * plot_w
        return x, top + plot_h - value / peak * plot_h

    chart = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(chart)
    draw.line([(left, top), (left, top + plot_h), (left + plot_w, top + plot_h)], fill="#888888")
    draw.line([point(t, v) for t, v in zip(times, differences)], fill="#4a7bd0", width=2)
    for duplicate in analysis["duplicate_findings"]:
        x, y = point(duplicate["timestamp"], duplicate["difference_score"])
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#d04a4a")

    draw.text((left, 8), "Change from the previous frame (blue); almost identical frames (red)", fill="#333333")
    draw.text((left, top + plot_h + 8), f"{t0:.1f}s", fill="#333333")
    draw.text((left + plot_w - 40, top + plot_h + 8), f"{t1:.1f}s", fill="#333333")
    draw.text((4, top - 4), f"{peak:.0f}", fill="#333333")
    draw.text((4, top + plot_h - 10), "0", fill="#333333")
    return _to_data_url(chart)


def _frame_checks(analysis: dict) -> list[dict]:
    """Turn video_analysis's frame-level findings into report checks."""
    duplicates = analysis["duplicate_findings"]
    repeated = analysis["repeated_findings"]
    inserted = analysis["inserted_clip_findings"]
    jumps = analysis["temporal_findings"]
    checks = []

    if duplicates:
        checks.append(_check(
            "duplicate_frames", "flagged",
            f"{len(duplicates)} frame(s) are almost identical to the frame before them while the video "
            f"around them is moving, {_times(duplicates)}. Repeated frames can be used to fill a gap left "
            "by an edit.", name="Duplicate frames", severity="medium",
            details={"duplicate_frames": len(duplicates)}, visualization=_frame_difference_chart(analysis)))
    else:
        checks.append(_check("duplicate_frames", "passed", "No frozen or repeated single frames found.",
                             name="Duplicate frames", visualization=_frame_difference_chart(analysis)))

    if repeated:
        checks.append(_check(
            "repeated_sequences", "flagged",
            f"{len(repeated)} stretch(es) of frames appear again later in the video. "
            f"{repeated[0]['evidence']}.", name="Repeated sequences", severity="medium",
            details={"repeated_sequences": len(repeated)}))
    else:
        checks.append(_check("repeated_sequences", "passed", "No stretch of frames appears twice.",
                             name="Repeated sequences"))

    if inserted:
        checks.append(_check(
            "inserted_clips", "flagged",
            f"{len(inserted)} short clip(s) may have been cut in. {inserted[0]['evidence']}.",
            name="Inserted clips", severity="medium", details={"inserted_clips": len(inserted)}))
    else:
        checks.append(_check("inserted_clips", "passed", "No short inserted clips found.",
                             name="Inserted clips"))

    checks.append(_check(
        "temporal_changes", "info",
        (f"{len(jumps)} unusually large jump(s) between frames, {_times(jumps)}. Normal scene cuts also "
         "cause these, so check them by eye." if jumps else "No unusually large jumps between frames."),
        name="Sudden changes", details={"sudden_changes": len(jumps)}))
    return checks


def _video_category(path: str, metadata: dict, is_video: bool, warnings: list[str]) -> tuple[dict, int]:
    """Metadata video checks plus frame-level checks. Returns (category, flagged frame checks)."""
    if not is_video:
        return _category([], applicable=False, note="Video checks run on videos only."), 0
    checks = _metadata_checks(metadata, VIDEO_METADATA_CHECKS)
    failed = 0
    try:
        frame_checks = _frame_checks(analyze_video(path))
    except Exception as exc:  # noqa: BLE001 - one detector must not break the report
        failed, frame_checks = 1, []
        warnings.append(f"Frame-level video analysis failed: {exc}")
    flagged = sum(c["status"] == "flagged" for c in frame_checks)
    # Flagged checks first, as in the other categories.
    checks = sorted(checks + frame_checks, key=lambda c: c["status"] != "flagged")
    return _category(checks, failed=failed), flagged


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

LEVELS = ["low", "some_signs", "likely_altered"]


def verdict(score: float, checks_run: int, copy_move_candidates: int, frame_flags: int = 0) -> dict:
    """Combine the metadata score and the pixel and frame checks into one label.

    Only the metadata checks produce a calibrated score: 0.3 and above is some_signs,
    0.6 and above likely_altered. Copy-move candidates and flagged frame checks
    raise the label by one level, because they are visual evidence of editing.

    Below 0.3, the label says how much the metadata showed: no_strong_evidence when
    the score is exactly 0, no_signs when there were only weak clues. With too few
    checks to judge, it is inconclusive instead.
    """
    if score >= 0.6:
        level = 2
    elif score >= 0.3:
        level = 1
    else:
        level = 0
    if copy_move_candidates or frame_flags:
        level = min(level + 1, 2)

    label = LEVELS[level]
    if level == 0:
        if checks_run <= FEW_CHECKS:
            label = "inconclusive"
        elif score == 0:
            label = "no_strong_evidence"
        else:
            label = "no_signs"
    return {"label": label, "score": score}


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
    video, frame_flags = _video_category(path, metadata, is_video, warnings)
    categories = {
        "metadata_analysis": _metadata_category(metadata),
        "image_analysis": image,
        "compression_analysis": _compression_category(path, metadata, is_image, warnings),
        "video_analysis": video,
    }

    return _json_safe({
        "file_name": display_name or os.path.basename(path),
        "media_type": media_type,
        "verdict": verdict(metadata["score"], metadata["checks_run"], candidates, frame_flags),
        "categories": categories,
        "warnings": warnings,
    })
