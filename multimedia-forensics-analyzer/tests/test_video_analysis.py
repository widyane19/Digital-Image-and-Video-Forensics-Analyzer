"""
test_analysis.py
by Kaden Umnitz

Purpose: The purpose of test_analysis.py is to test the analysis functions, 
    expected outputs, and relevant edge cases.
"""
import cv2
import numpy as np
import pytest

from src.video_analysis import (
    validate_video,
    extract_frames,
    calculate_frame_difference,
    calculate_changed_pixel_ratio,
    calculate_histogram_distance,
    analyze_video,
    group_findings
)

@pytest.fixture
def sample_videos(tmp_path):
    height, width = 120, 160
    fps = 30

    def write_video(name, frames):
        path = str(tmp_path / name)
        writer = cv2.VideoWriter(
            path, 
            cv2.VideoWriter_fourcc(*"mp4v"), 
            fps, 
            (width, height)
        )

        if not writer.isOpened():
            pytest.skip("MP4 video writer unavailable")

        for frame in frames:
            writer.write(frame)
        
        writer.release()
        return path
    
    normal = []

    for i in range(30):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(frame, (30 + i * 3, 60), 12, (255, 255, 255), -1)
        normal.append(frame)

    duplicate = normal[:15]
    duplicate.insert(10, duplicate[9].copy())
    static_frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(static_frame, (30,30), (100,100), (255, 255, 255), -1)
    static = [static_frame.copy() for _ in range(30)]
    
    return {
        "normal": write_video("normal.mp4", normal),
        "duplicate": write_video("duplicate.mp4", duplicate),
        "static": write_video("static.mp4", static)
    }

def test_video_validation(sample_videos):
    result = validate_video(sample_videos["normal"])
    assert result["fps"] > 0
    assert result["frame_count"] > 0
    assert result["width"] == 160
    assert result["height"] == 120

def test_frame_extraction(sample_videos):
    frames = extract_frames(sample_videos["normal"])
    assert len(frames) > 0
    assert "timestamp" in frames[0]
    assert "image" in frames[0]
    
def test_identical_frames():
    frame = np.zeros((100, 100), dtype=np.uint8)
    assert calculate_frame_difference(frame, frame) == 0.0
    assert calculate_changed_pixel_ratio(frame, frame) == 0.0

def test_completley_different_frames():
    frame_a = np.zeros((100, 100), dtype=np.uint8)
    frame_b = np.full((100, 100), 255,  dtype=np.uint8)
    assert calculate_frame_difference(frame_a, frame_b) == 255.0

def test_histogram_distance():
    frame = np.zeros((100, 100), dtype=np.uint8)
    distance = calculate_histogram_distance(frame, frame)
    assert distance == 0.0

def test_duplicate_video(sample_videos):
    result = analyze_video(
        sample_videos["duplicate"], 
        duplicate_threshold=2.0
    )
    assert result["summary"]["duplicate_candidates"] > 0

def test_static_video(sample_videos):
    result = (analyze_video(sample_videos["static"]))
    assert result["summary"]["frames_processed"] > 0

def test_group_findings():
    findings = [
        {
            "timestamp": 1.0,
            "type": "temporal_inconsistency",
            "evidence": "Test finding"
        },
        {
            "timestamp": 1.5,
            "type": "temporal_inconsistency",
            "evidence": "Test finding"
        }
    ]
    
    intervals = group_findings(findings, gap_seconds=1.0)
    assert len(intervals) == 1
    assert intervals[0]["finding_count"] == 2
