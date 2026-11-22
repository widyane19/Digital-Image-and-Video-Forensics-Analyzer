"""The video detectors must show up in the web report and in the evaluation."""

import cv2
import numpy as np
import pytest

from src.evaluation import _video_detector, FLAGGED, NOT_FLAGGED
from src.pipeline import analyze


def _write(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    if not writer.isOpened():
        pytest.skip("MP4 video writer unavailable")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def videos(tmp_path):
    normal = []
    for i in range(40):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.circle(frame, (20 + i * 3, 60), 12, (255, 255, 255), -1)
        normal.append(frame)
    repeated = normal[:20] + [normal[19].copy() for _ in range(5)] + normal[20:]
    pasted = normal[:30] + normal[10:25] + normal[30:]     # a 15-frame stretch copied and pasted
    return (_write(tmp_path / "normal.mp4", normal), _write(tmp_path / "repeated.mp4", repeated),
            _write(tmp_path / "pasted.mp4", pasted))


def test_report_has_frame_checks(videos):
    report = analyze(str(videos[1]))
    checks = {c["id"]: c for c in report["categories"]["video_analysis"]["checks"]}
    assert checks["duplicate_frames"]["status"] == "flagged"
    assert checks["duplicate_frames"]["visualization"].startswith("data:image/png")
    assert "frame_detectors" not in checks
    assert report["verdict"]["label"] in ("some_signs", "likely_altered")


def test_clean_video_passes_frame_checks(videos):
    report = analyze(str(videos[0]))
    checks = {c["id"]: c for c in report["categories"]["video_analysis"]["checks"]}
    assert checks["duplicate_frames"]["status"] == "passed"


def test_pasted_stretch_is_flagged(videos):
    report = analyze(str(videos[2]))
    checks = {c["id"]: c for c in report["categories"]["video_analysis"]["checks"]}
    assert checks["repeated_sequences"]["status"] == "flagged"
    clean = {c["id"]: c for c in analyze(str(videos[0]))["categories"]["video_analysis"]["checks"]}
    assert clean["repeated_sequences"]["status"] == "passed"


def test_evaluation_detector_flags_repeated_frames(videos):
    assert _video_detector(videos[1], None)["video.duplicate_frames"] == FLAGGED
    assert _video_detector(videos[0], None)["video.duplicate_frames"] == NOT_FLAGGED
    assert _video_detector(videos[2], None)["video.repeated_sequences"] == FLAGGED
