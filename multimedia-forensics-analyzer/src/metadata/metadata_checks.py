"""Metadata consistency checks: Step 4 of the metadata analysis pipeline.

    4. Run consistency checks

Decides what looks inconsistent in NormalizedMetadata and produces Findings.
Imports from metadata_extraction.py only; metadata_analysis.py (Steps 5-7) imports from here.

Each check is a small function registered with @check. It returns NOT_APPLICABLE when
the information it needs is missing, NOT_FLAGGED when nothing looks wrong, or flagged(...).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cached_property, lru_cache
from io import BytesIO
from statistics import median
from typing import Callable


from exiftool import ExifToolHelper

from .metadata_extraction import NormalizedMetadata, _find, _parse_date, open_exiftool

# Tunable settings. These are starting values, to adjust against real files.

# Names are matched in lower case, anywhere in a software tag.
EDITING_SOFTWARE = (
    "photoshop", "lightroom", "gimp", "snapseed", "canva", "affinity", "pixelmator", "paint.net",
    "premiere", "after effects", "media encoder", "capcut", "davinci", "imovie", "final cut",
    "filmora", "vegas pro",
)
RE_ENCODERS = ("ffmpeg", "handbrake", "lavf", "lavc")  # Lavf/Lavc are FFmpeg's own libraries

SEVERITY_WEIGHTS = {"info": 0.0, "low": 0.1, "medium": 0.3, "high": 0.6}

CLOCK_TOLERANCE_SECONDS = 120        # how far two clocks may disagree
UNKNOWN_ZONE_SECONDS = 14 * 3600     # extra allowance when only one date has a time zone
TIME_ZONE_STEP_SECONDS = 15 * 60     # every real time zone is a multiple of 15 minutes
AUDIO_VIDEO_TOLERANCE_SECONDS = 0.5
MIN_FRAMES_FOR_TIMING = 3
FRAME_COUNT_TOLERANCE = 2            # frames, or 1% of the expected count if that is larger
FRAME_RATE_TOLERANCE = 0.05          # 5%
LONG_GAP_FACTOR = 1.5                # a gap this many times the normal gap means missing frames
PREVIEW_ASPECT_TOLERANCE = 0.03      # 3%, because previews are small and sizes get rounded
PREVIEW_GRID = 8                     # the image is compared as an 8x8 grid of cells
PREVIEW_CELL_THRESHOLD = 25          # average brightness difference (0-255) that marks a cell
PREVIEW_BLACK_LEVEL = 16             # darker than this counts as a black bar around a preview

BOTH = ("image", "video")


@dataclass
class Finding:
    """One piece of evidence that a file may have been edited."""

    check: str
    severity: str  # "info", "low", "medium" or "high"
    message: str
    evidence: dict = field(default_factory=dict)  # the values that triggered it
    regions: list[tuple[int, int, int, int]] = field(default_factory=list)  # photos: x, y, width, height
    frames: list[int] = field(default_factory=list)                         # videos: frame numbers


@dataclass
class CheckResult:
    """The outcome of one check."""

    status: str  # "flagged", "not_flagged" or "not_applicable"
    finding: Finding | None = None


NOT_FLAGGED = CheckResult("not_flagged")
NOT_APPLICABLE = CheckResult("not_applicable")


def flagged(severity: str, message: str, **details) -> CheckResult:
    """Build a flagged result. The runner fills in the check's name."""
    return CheckResult("flagged", Finding("", severity, message, **details))


@dataclass
class CheckContext:
    """What a check can look at. Slow data is only loaded if a check asks for it."""

    meta: NormalizedMetadata
    file_date: datetime | None = None  # the file's last-saved date, when it can be trusted
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    et: ExifToolHelper | None = None

    @cached_property
    def frame_times(self) -> list[float]:
        return read_frame_times(self.meta.path)

    @cached_property
    def preview(self):
        return load_preview(self.meta, self.et)


# The check registry

@dataclass(frozen=True)
class Check:
    name: str
    applies_to: tuple[str, ...]
    run: Callable[[CheckContext], CheckResult]


CHECKS: dict[str, Check] = {}


def check(name: str, applies_to: tuple[str, ...] = BOTH):
    """Register a function as a check."""
    def register(func):
        CHECKS[name] = Check(name, applies_to, func)
        return func
    return register


def run_checks(ctx: CheckContext) -> tuple[dict[str, CheckResult], list[str]]:
    """Run every check. Returns (results by check name, warnings).

    A check that crashes counts as not applicable and adds a warning, so one bad
    check never loses the whole report.
    """
    results: dict[str, CheckResult] = {}
    warnings: list[str] = []
    for name, entry in CHECKS.items():
        if ctx.meta.media_type not in entry.applies_to:
            results[name] = NOT_APPLICABLE
            continue
        try:
            result = entry.run(ctx)
        except Exception as exc:
            warnings.append(f"{name} could not run: {exc}")
            result = NOT_APPLICABLE
        if result.finding:
            result.finding.check = name
        results[name] = result
    return results, warnings


# Date helpers

def is_later(a: datetime, b: datetime, margin: float = 0.0) -> bool:
    """True if `a` is definitely more than `margin` seconds after `b`.

    Python can't compare a date that has a time zone with one that doesn't. Those
    pairs are compared by their clock readings, with a wider margin to cover the
    unknown zone.
    """
    if (a.tzinfo is None) != (b.tzinfo is None):
        a, b = a.replace(tzinfo=None), b.replace(tzinfo=None)
        margin += UNKNOWN_ZONE_SECONDS
    return (a - b).total_seconds() > margin


def _iso(date: datetime) -> str:
    return date.isoformat()


# Checks for both photos and videos

@check("editing_software")
def editing_software(ctx: CheckContext) -> CheckResult:
    software = ctx.meta.software
    if not software:
        return NOT_APPLICABLE

    editors = [s for s in software if any(name in s.lower() for name in EDITING_SOFTWARE)]
    if editors:
        return flagged("high", f"The file was saved by editing software: {', '.join(editors)}.",
                       evidence={"software": editors})

    encoders = [s for s in software if any(name in s.lower() for name in RE_ENCODERS)]
    if encoders:
        return flagged("medium", f"The file was re-encoded by {', '.join(encoders)}. "
                                 "Platforms also do this to untouched videos.",
                       evidence={"software": encoders})
    return NOT_FLAGGED


@check("modified_after_capture")
def modified_after_capture(ctx: CheckContext) -> CheckResult:
    meta = ctx.meta
    if meta.created is None or meta.modified is None:
        return NOT_APPLICABLE

    allowed = CLOCK_TOLERANCE_SECONDS + (meta.duration or 0)
    if is_later(meta.modified, meta.created, allowed):
        return flagged("medium", "The file was modified noticeably later than it was captured.",
                       evidence={"created": _iso(meta.created), "modified": _iso(meta.modified)})
    return NOT_FLAGGED


@check("impossible_dates")
def impossible_dates(ctx: CheckContext) -> CheckResult:
    meta = ctx.meta
    if meta.created is None and meta.modified is None:
        return NOT_APPLICABLE

    for label, date in (("capture", meta.created), ("modified", meta.modified)):
        if date and is_later(date, ctx.now, CLOCK_TOLERANCE_SECONDS):
            return flagged("high", f"The {label} date is in the future.", evidence={label: _iso(date)})

    if meta.created and meta.modified and is_later(meta.created, meta.modified, CLOCK_TOLERANCE_SECONDS):
        return flagged("high", "The file says it was modified before it was captured.",
                       evidence={"created": _iso(meta.created), "modified": _iso(meta.modified)})
    return NOT_FLAGGED


@check("file_date_before_capture")
def file_date_before_capture(ctx: CheckContext) -> CheckResult:
    created = ctx.meta.created
    if created is None or ctx.file_date is None:
        return NOT_APPLICABLE

    if is_later(created, ctx.file_date, CLOCK_TOLERANCE_SECONDS):
        return flagged("medium", "The file was last saved before its embedded capture date, "
                                 "which suggests the metadata was changed.",
                       evidence={"created": _iso(created), "file_date": _iso(ctx.file_date)})
    return NOT_FLAGGED


@check("size_mismatch")
def size_mismatch(ctx: CheckContext) -> CheckResult:
    meta = ctx.meta
    declared = (meta.declared_width, meta.declared_height)
    actual = (meta.actual_width, meta.actual_height)
    if None in declared or None in actual:
        return NOT_APPLICABLE

    if declared in (actual, actual[::-1]):  # the reversed size is a rotated file
        return NOT_FLAGGED
    return flagged("medium", f"The metadata claims {declared[0]}x{declared[1]} but the file is "
                             f"{actual[0]}x{actual[1]}, which suggests cropping or resizing.",
                   evidence={"declared": list(declared), "actual": list(actual)})


@check("metadata_removed")
def metadata_removed(ctx: CheckContext) -> CheckResult:
    meta = ctx.meta
    if meta.make or meta.model or meta.created or meta.modified or meta.software:
        return NOT_FLAGGED
    return flagged("low", "The file has no device, date or software information. "
                          "Social media removes this from untouched files too.")


# Photo-only checks

def load_preview(meta: NormalizedMetadata, et: ExifToolHelper | None = None):
    """Return the preview image stored inside a photo, or None if it has none.

    The normal tag list only says that a preview exists, so its bytes need a separate read.
    """
    from PIL import Image

    name = next((n for n in ("ThumbnailImage", "PreviewImage") if _find(meta.tags, n)), None)
    if name is None:
        return None
    if et is None:
        with open_exiftool() as temp_et:
            return load_preview(meta, temp_et)

    values = et.get_tags(meta.path, [name], params=["-b"])[0]
    data = str(_find(values, name) or "")
    if not data.startswith("base64:"):
        return None
    return Image.open(BytesIO(base64.b64decode(data[len("base64:"):]))).convert("L")


def _aspect(size: tuple[int, int]) -> float:
    return size[0] / size[1]


def _trim_black_bars(preview, aspect: float):
    """Cameras often pad a preview with black bars. Remove them if that fits the photo better."""
    box = preview.point(lambda v: 255 if v > PREVIEW_BLACK_LEVEL else 0).getbbox()
    if box is None:
        return preview
    trimmed = preview.crop(box)
    if abs(_aspect(trimmed.size) - aspect) < abs(_aspect(preview.size) - aspect):
        return trimmed
    return preview


def _differing_regions(preview, photo) -> list[tuple[int, int, int, int]]:
    """Compare a preview with its photo, cell by cell. Returns the regions that differ most."""
    import numpy as np

    small = photo.convert("L").resize(preview.size)
    diff = np.abs(np.asarray(preview, dtype=float) - np.asarray(small, dtype=float))

    rows = np.linspace(0, diff.shape[0], PREVIEW_GRID + 1).astype(int)
    cols = np.linspace(0, diff.shape[1], PREVIEW_GRID + 1).astype(int)
    cells = np.array([[diff[rows[r]:rows[r + 1], cols[c]:cols[c + 1]].mean()
                       for c in range(PREVIEW_GRID)] for r in range(PREVIEW_GRID)])

    # An overall colour change lifts every cell, so a cell must also stand out from the rest.
    threshold = max(PREVIEW_CELL_THRESHOLD, 3 * float(np.median(cells)))
    scale_x = photo.width / preview.width
    scale_y = photo.height / preview.height
    return [(round(cols[c] * scale_x), round(rows[r] * scale_y),
             round((cols[c + 1] - cols[c]) * scale_x), round((rows[r + 1] - rows[r]) * scale_y))
            for r, c in zip(*np.nonzero(cells > threshold))]


@check("preview_mismatch", applies_to=("image",))
def preview_mismatch(ctx: CheckContext) -> CheckResult:
    from PIL import Image, UnidentifiedImageError

    preview = ctx.preview
    if preview is None:
        return NOT_APPLICABLE
    try:
        photo = Image.open(ctx.meta.path)
    except UnidentifiedImageError:  # Pillow can't open every format ExifTool reads, such as HEIC
        return NOT_APPLICABLE

    with photo:
        aspect = _aspect(photo.size)
        preview = _trim_black_bars(preview, aspect)
        preview_aspect = _aspect(preview.size)

        if abs(1 / preview_aspect - aspect) / aspect <= PREVIEW_ASPECT_TOLERANCE and photo.width != photo.height:
            return NOT_FLAGGED  # rotated after capture, so the two can't be compared
        if abs(preview_aspect - aspect) / aspect > PREVIEW_ASPECT_TOLERANCE:
            return flagged("high", "The stored preview has a different shape from the full image, "
                                   "so the photo was probably cropped.",
                           evidence={"preview_size": list(preview.size), "photo_size": list(photo.size)})

        regions = _differing_regions(preview, photo)
    if regions:
        return flagged("high", "Parts of the full image differ from the stored preview, "
                               "so those areas were probably edited.",
                       evidence={"cells": len(regions)}, regions=regions)
    return NOT_FLAGGED


@check("edit_history", applies_to=("image",))
def edit_history(ctx: CheckContext) -> CheckResult:
    found = sorted(
        key for key in ctx.meta.tags
        if key.startswith("XMP-crs:") or key.rpartition(":")[2].startswith(("History", "DocumentAncestors"))
    )
    if found:
        return flagged("high", "The file contains an editing history or Lightroom/Camera Raw settings.",
                       evidence={"tags": found[:10]})
    return NOT_FLAGGED


@check("gps_time", applies_to=("image",))
def gps_time(ctx: CheckContext) -> CheckResult:
    created, gps = ctx.meta.created, ctx.meta.gps_time
    if created is None or gps is None:
        return NOT_APPLICABLE

    if created.tzinfo:
        gap = abs((created - gps).total_seconds())
        agrees = gap <= CLOCK_TOLERANCE_SECONDS
    else:
        # The camera clock's zone is unknown, so the gap only needs to look like a time zone.
        gap = abs((created - gps.replace(tzinfo=None)).total_seconds())
        remainder = gap % TIME_ZONE_STEP_SECONDS
        agrees = (gap <= UNKNOWN_ZONE_SECONDS + CLOCK_TOLERANCE_SECONDS and
                  min(remainder, TIME_ZONE_STEP_SECONDS - remainder) <= CLOCK_TOLERANCE_SECONDS)

    if agrees:
        return NOT_FLAGGED
    return flagged("medium", "The GPS clock and the camera clock disagree, even allowing for time zones.",
                   evidence={"created": _iso(created), "gps_time": _iso(gps)})


@lru_cache(maxsize=1)
def _standard_jpeg_tables() -> list[set[tuple[int, ...]]]:
    """The generic compression tables common software uses, one set per quality level.

    Made by saving a tiny image at each quality, so they are in the same order as
    the tables read back from a file.
    """
    from PIL import Image

    tables = []
    for quality in range(1, 101):
        buffer = BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as saved:
            tables.append({tuple(t) for t in saved.quantization.values()})
    return tables


@check("compression_settings", applies_to=("image",))
def compression_settings(ctx: CheckContext) -> CheckResult:
    from PIL import Image

    meta = ctx.meta
    if meta.tags.get("File:MIMEType") != "image/jpeg" or not (meta.make or meta.model):
        return NOT_APPLICABLE

    with Image.open(meta.path) as photo:
        tables = {tuple(t) for t in photo.quantization.values()}
    if any(tables <= standard for standard in _standard_jpeg_tables()):
        return flagged("low", "The file says it came from a camera, but it uses the generic compression "
                              "tables found in common software, so it may have been re-saved.")
    return NOT_FLAGGED


# Video-only checks

def read_frame_times(path: str) -> list[float]:
    """Return every frame's timestamp in milliseconds. ExifTool can't see individual frames."""
    import cv2

    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise ValueError(f"OpenCV could not open {path}")
    times = []
    try:
        while video.grab():
            times.append(video.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        video.release()
    return times


def _frame_gaps(times: list[float]) -> list[float]:
    return [after - before for before, after in zip(times, times[1:])]


def _normal_gap(times: list[float]) -> float | None:
    """The usual time between frames, in milliseconds. A few odd gaps barely affect it."""
    if len(times) < MIN_FRAMES_FOR_TIMING:
        return None
    gaps = [gap for gap in _frame_gaps(times) if gap > 0]
    return median(gaps) if gaps else None


@check("creation_dates", applies_to=("video",))
def creation_dates(ctx: CheckContext) -> CheckResult:
    tags = ctx.meta.tags
    overall = _parse_date(tags.get("QuickTime:CreateDate"))
    track_dates = {
        key: date for key, value in tags.items()
        if key.startswith("Track") and key.endswith(("TrackCreateDate", "MediaCreateDate"))
        and (date := _parse_date(value))
    }
    if overall is None or not track_dates:
        return NOT_APPLICABLE

    different = {key: _iso(date) for key, date in track_dates.items()
                 if is_later(date, overall, CLOCK_TOLERANCE_SECONDS)
                 or is_later(overall, date, CLOCK_TOLERANCE_SECONDS)}
    if different:
        return flagged("medium", "The video's creation date and its tracks' creation dates disagree.",
                       evidence={"video": _iso(overall), "tracks": different})
    return NOT_FLAGGED


@check("frame_count", applies_to=("video",))
def frame_count(ctx: CheckContext) -> CheckResult:
    duration = ctx.meta.video_duration or ctx.meta.duration
    times = ctx.frame_times
    gap = _normal_gap(times)
    if not duration or gap is None:
        return NOT_APPLICABLE

    # ExifTool's frame rate is an average of the frames present, so it can't reveal
    # missing ones. The rate the frames actually run at can.
    expected = duration * 1000 / gap
    if abs(len(times) - expected) > max(FRAME_COUNT_TOLERANCE, expected * 0.01):
        return flagged("medium", f"The file has {len(times)} frames but its length and frame rate "
                                 f"suggest about {round(expected)}, so frames may have been cut or inserted.",
                       evidence={"frames": len(times), "expected": round(expected)})
    return NOT_FLAGGED


@check("steady_frame_rate", applies_to=("video",))
def steady_frame_rate(ctx: CheckContext) -> CheckResult:
    declared = ctx.meta.frame_rate
    gap = _normal_gap(ctx.frame_times)
    if not declared or gap is None:
        return NOT_APPLICABLE

    actual = 1000 / gap
    if abs(declared - actual) / actual > FRAME_RATE_TOLERANCE:
        return flagged("medium", f"The declared frame rate ({declared:.2f} fps) doesn't match the rate "
                                 f"the frames actually run at ({actual:.2f} fps).",
                       evidence={"declared": declared, "actual": round(actual, 2)})
    return NOT_FLAGGED


@check("audio_video_length", applies_to=("video",))
def audio_video_length(ctx: CheckContext) -> CheckResult:
    audio, video = ctx.meta.audio_duration, ctx.meta.video_duration
    if audio is None or video is None:
        return NOT_APPLICABLE

    if abs(audio - video) > AUDIO_VIDEO_TOLERANCE_SECONDS:
        return flagged("low", f"The audio ({audio:.2f}s) and video ({video:.2f}s) tracks differ in length.",
                       evidence={"audio": audio, "video": video})
    return NOT_FLAGGED


@check("frame_timing", applies_to=("video",))
def frame_timing(ctx: CheckContext) -> CheckResult:
    times = ctx.frame_times
    normal = _normal_gap(times)
    if normal is None:
        return NOT_APPLICABLE

    gaps = _frame_gaps(times)
    long_gaps = [i + 1 for i, gap in enumerate(gaps) if gap > normal * LONG_GAP_FACTOR]
    repeats = [i + 1 for i, gap in enumerate(gaps) if gap <= 0]
    if not long_gaps and not repeats:
        return NOT_FLAGGED

    return flagged("high", f"Frame timing is irregular: {len(long_gaps)} long gap(s), which suggest missing "
                           f"frames, and {len(repeats)} repeated timestamp(s), which suggest duplicated frames.",
                   evidence={"long_gaps": long_gaps, "repeated_timestamps": repeats,
                             "normal_gap_ms": round(normal, 3)},
                   frames=sorted(long_gaps + repeats))