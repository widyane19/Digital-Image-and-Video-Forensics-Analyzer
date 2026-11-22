"""
video_analysis.py 
by Kaden Umnitz

Features:
- Video validation and metadata extraction
- Frame extraction and timestamp tracking
- Consecutive frame pixel difference algorithm
- Near duplicate frame detection
- Scene change detection using color histograms
- Temporal anamoly detection
- Suspicious interval grouping
- JSON report export

The comparasion detection algorithms are implemented using NumPy.
OpenCV is used for video decoding, resizing, and frame processing.
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path


# VIDEO VALIDATION
def validate_video(video_path):
    video_path = str(video_path)
    
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <=0:
        raise ValueError("Video has an invalid or unsupported frame rate.")

    if width <=0 or height <=0:
        raise ValueError("Video has invalid dimensions.")

    duration = (frame_count / fps if fps > 0 else None)

    return {
        "video path": video_path,
        "fps": float(fps)
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration
    }


# FRAME EXTRACTION
def extract_frames(video_path, sample_every=1, resize_width=320):
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1.")

    if resize_width < 1:
        raise ValueError("resize_width must be positive.")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Unable to open video.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break
        
        if frame_index % sample_every == 0:
            height, width = frame.shape[:2]
            new_height = max(1, round(height * resize_width / width))

            resized = cv2.resize(frame,(resize_width, new_height),interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if timestamp < 0 or (timestamp == 0 and frame_index > 0):
                timestamp = (frame_index / fps if fps > 0 else None)
            
            frames.append({
                "frame_index": frame_index,
                "timestamp": float(timestamp),
                "image": gray
            })

            frame_index += 1

    cap.release()

    if len(frames) < 2:
        raise ValueError("At least 2 frames are required for analysis.")
        
    return frames


# PIXEL DIFFERENCE ALGORITHM
def calculate_frame_difference(frame_a, frame_b):
    """
    Difference score from 0 to 255.
        0   = identical frames
        255 = max possible grayscale difference
    """
    if frame_a.shape != frame_b.shape:
        raise ValueError("Frames must have matching dimensions.")
    
    difference = np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32))
    return float(np.mean(difference))

def calculate_changed_pixel_ratio(frame_a, frame_b, pixelthreshold=8):
    """
    The calculate_changed_pixel_ratio() function calculates the porportion of pixels that 
        changed by more than pixel_threshold intesity levels.
    """
    if frame_a.shape != frame_b.shape:
        raise ValueError("Frames must have matching dimensions.")

    difference = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16))
    changed_pixels = difference > pixel_threshold

    return float(np.mean(changed_pixels))


# HISTOGRAM COMPARISON
def calculate_histogram_distance(frame_a, frame_b, bins=32):
    """
    A distance value from 0 to 1.
        0 = Similar distrobutions
        1 = Very different distrobutions
    """
    if frame_a.shape != frame_b.shape:
        raise ValueError("Frames must have matching dimensions.")

    hist_a, _ = np.histogram(frame_a, bins=bins, range=(0,256))
    hist_b, _ = np.histogram(frame_b, bins=bins, range=(0,256))
    
    hist_a = hist_a.astype(np.float64)
    hist_b = hist_b.astype(np.float64)

    hist_a /= max(hist_a.sum(), 1)
    hist_b /= max(hist_b.sum(), 1)

    coefficient = np.clip(coefficient, 0.0, 1.0)
    distance = np.sqrt(1.0 - coefficient)

    return float(distance)


# CONSECTUTIVE-FRAME ANALYSIS
def analyze_frame_similarity(
    frames, 
    duplicate_threshold=2.0,
    pixel_threshold=8,
    scene_threshold=0.45
):
    comparisons = []
    duplicate_findings = []
    scene_changes = []

    for i in range(1, len(frames)):
        previous = frames[i - 1]
        current = frames[i]
        
        frame_a = previous["image"]
        frame_b = current["image"]
       
        mean_difference = calculate_frame_difference(frame_a, frame_b)
        changed_ratio = calculate_changed_pixel_ratio(frame_a, frame_b, pixel_threshold)
        histogram_distance = calculate_histogram_distance(frame_a, frame_b)
        is_duplicate_candidate = (mean_difference <= duplicate_threshold and changed_ratio <= 0.01)
        is_scene_change = (histogram_distance >= scene_threshold)

        comparison = {
            "previous_frame": previous["frame_index"],
            "current_frame": current["frame_index"],
            "previous_timestamp": previous["timestamp"]
            "current_timestamp": current["timestamp"],
            "mean_pixel_difference": mean_difference,
            "changed_pixel_ratio": changed_ratio,
            "histogram_distance": histogram_distance,
            "duplicate_candidate": bool(is_duplicate_candidate),
            "scene_change_candidate": bool(is_scene_change)
        }

        comparisons.append(comparison)

        if is_duplicate_candidate:
            duplicate_findings.append({
                "frame_index": current["frame_index"],
                "timestamp": current["timestamp"],
                "previous_timestamp": previous["timestamp"]
                "difference_score": mean_difference,
                "evidence": "Consecutive frames are unusually similar ",
                "type": "possible_frame_duplication"
            })

        if is_scene_change:
            scene_changes.append({
                "frame_index": current["frame_index"],
                "timestamp": current["timestamp"],
                "histogram_distance": histogram_distance,
                "evidence": "Large change in frame intensity distrobution.",
                "type": "possible_scene_transition"
            })

    return {
        "comparisons": comparisons,
        "duplicate_findings": duplicate_findings,
        "scene_changes": scene_changes
    }


# TEMPORAL ANOMALY ANALYSIS
def analyze_temporal_consistency(
    comparasions,
    scene_threshold=0.45,
    sensitivity=4.0,
    minimum_difference=35.0
):
    if not comparasions:
        return {
            "threshold": None,
            "findings": []
        }
    
    differences = np.array(
        [
            item["mean_pixel_difference"]
            for item in comparasions
            if item["histogram_distance"] < scene_threshold
        ],
        dtype=np.float64
    )

    if len(differences) == 0:
        return{
            "threshold": None,
            "findings": []
        }

    median = float(np.median(differences))
    # Median Absolute Deviation
    mad = float(np.median(np.abs(differences - median)))

    # mad is scaled to be comprable to std deviation for normally distrobuted data.
    spread = 1.4826 * mad

    threshold = max(
        minimum_difference,
        median + sensitivity * spread
    )

    findings = []

    for item in comparasions:
        if(item["histogram_distance"] >= scene_threshold):
            continue
        
        if item["mean_pixel_difference"] > threshold:
            findings.append({
                "frame_index": item["current_frame"],
                "timestamp": item["current_timestamp"],
                "previous_timestamp": item["previous_timestamp"],
                "difference_score": item["mean_pixel_difference"],
                "threshold": float(threshold),
                "evidence": "Unusually large consecutive frame change"
                "type": "temporal_inconsistency"
            })

    return {
        "baseline_median": median,
        "baseline_mad": mad,
        "threshold": float(threshold),
        "findings": finndings
    }


# GROUPED FINDINGS W/ INTERVALS
def group_findings(findings, gap_seconds=1.0):
    if not findings:
        return []

    grouped = []
    sorted_findings = sorted(
        findings,
        key=lamda item: (
            item["type"],
            item["timestamp"]
        )
    )

    current_group = [sorted_findings[0]]

    for finding in sorted_findings[1:]:
        previous = current_group[-1]
        same_type = (finding["type"] == previous["type"])
        close_in_time = (finding["timestamp"] - previous["timestamp"]) <= gap_seconds

        if same_type and close_in_time:
            current_group.append(finding)
        
        else:
            grouped.append(current_group)
            current_group = [finding]

    grouped.append(current_group)
    intervals = []

    for group in grouped:
        timestamps = [item["timestamp"] for item in group]
        intervals.append({
            "type": group[0]["type"],
            "start_time": float(min(timestamps)),
            "end_time": float(max(timestamps)),
            "finding_count": len(group),
            "evidence": group[0]["evidence"],
            "findings": group
        })
    
    return intervals

# COMPLETE VIDEO ANALYSIS
def analyze_video(
    video_path, 
    sample_every=1, 
    resize_width=320,
    duplicate_threshold=2.0,
    pixel_threshold=8,
    scene_threshold=0.45,
    temporal_sensitivity=4.0
):
    metadata = validate_video(video_path)

    frames = extract_frames(video_path, sample_every=sample_every, resize_width=resize_width)
    
    similarity_results = analyze_frame_similarity(
        frames, 
        duplicate_threshold=duplicate_threshold,
        pixel_threshold=pixel_threshold,
        scene_threshold=scene_threshold
    )

    temporal = analyze_temporal_consistency(
        similaraity["comparisons"],
        scene_threshold=scene_threshold,
        sensitivity=temporal_sensitivity
    )

    all_findings = similaraity["duplicate_findings"] + temporal["findings"]

    intervals = group_findings(all_findings)

    return {
        "video_path": str(video_path),
        "metadata": metadata,
        "analysis_settings": {
            "sample_every": sample_every,
            "resize_width": resize_width,
            "duplicate_threshold": duplicate_threshold,
            "pixel_threshold": pixel_threshold,
            "scene_threshold": scene_threshold,
            "temporal_sensitivity": temporal_sensitivity
        },
        "summary": {
            "frames_processed": len(frames),
            "comparasions_performed": len (similaraity["comparasions"]),
            "duplicate_candidates": len(similaraity["duplicate_findings"]),
            "scene_change_candidates": len(similaraity["scene_changes"]),
            "temporal_anomalies": len(temporal["findings"]),
            "suspicious_intervals": len(intervals)
        },
        "duplicate_findings": similaraity["duplicate_findings"],
        "scene_changes": similaraity["scene_changes"],
        "temporal_findings": temporal["findings"],
        "temporal_threshold": temporal["threshold"],
        "suspicious_intervals": intervals,
        "comparisons": similaraity["comparisons"],
        "limitations": [
            "Similar frames can occur naturally in static scenes.",
            "Scene changes are not automatically evidence of manipulation.",
            "Compression can affect pixel level comparasions.",
            "Thresholds require calibration on labeled videos.",
            "This module identifies potential anomalies, not definitive proof."
        ]
    }

# JSON REPORT EXPORT
def save_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open (output_path, "w", encodings="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    return str(output_path)
