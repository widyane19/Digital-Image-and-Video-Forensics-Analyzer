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
        "fps": float(fps),
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

            resized = cv2.resize(
                frame,
                (resize_width, new_height),
                interpolation=cv2.INTER_AREA
            )

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            if timestamp < 0 or (timestamp == 0 and frame_index > 0):
                timestamp = frame_index / fps if fps > 0 else None

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

def calculate_changed_pixel_ratio(frame_a, frame_b, pixel_threshold=8):
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

    coefficient = np.sum(np.sqrt(hist_a * hist_b))
    coefficient = np.clip(coefficient, 0.0, 1.0)
    distance = np.sqrt(1.0 - coefficient)

    return float(distance)


# CONSECTUTIVE-FRAME ANALYSIS
MIN_LOCAL_MOTION = 1.0 # typical frame-to-frame change needed before a frame means anything
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
            "previous_timestamp": previous["timestamp"],
            "current_timestamp": current["timestamp"],
            "mean_pixel_difference": mean_difference,
            "changed_pixel_ratio": changed_ratio,
            "histogram_distance": histogram_distance,
            "duplicate_candidate": bool(is_duplicate_candidate),
            "scene_change_candidate": bool(is_scene_change)
        }

        comparisons.append(comparison)

       # A frame only counts as a repeat when the video is moving around it. In a still
    # scene (or with a small object moving on a still background) every frame looks like
    # the last one, so low difference alone proves nothing.
    differences = np.array([c["mean_pixel_difference"] for c in comparisons], dtype=np.float64)
    for i, comparison in enumerate(comparisons):
        if not comparison["duplicate_candidate"]:
            continue
        window = np.concatenate([differences[max(0, i - 10):i], differences[i + 1:i + 11]])
        local_motion = float(np.median(window)) if len(window) else 0.0
        if local_motion < MIN_LOCAL_MOTION or comparison["mean_pixel_difference"] > local_motion / 3:
            comparison["duplicate_candidate"] = False
            continue
        duplicate_findings.append({
            "frame_index": comparison["current_frame"],
            "timestamp": comparison["current_timestamp"],
            "previous_timestamp": comparison["previous_timestamp"],
            "difference_score": comparison["mean_pixel_difference"],
            "evidence": "Frame is almost identical to the one before it while the video around it is moving",
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
                "evidence": "Unusually large consecutive frame change",
                "type": "temporal_inconsistency"
            })

    return {
        "baseline_median": median,
        "baseline_mad": mad,
        "threshold": float(threshold),
        "findings": findings
    }

def find_repeated_blocks(frames, block_size=10, max_difference=3.0, min_motion=1.0):
    """Find a run of frames that appears again later in the video.

    Frames are shrunk to 32x32 and every run of `block_size` frames is compared with
    every later run that does not overlap it. A match with real motion inside it is
    reported once per pasted stretch, at the moment the repeat starts.
    """
    count = len(frames)
    if count < 2 * block_size:
        return []

    small = np.stack([
        cv2.resize(item["image"], (32, 32), interpolation=cv2.INTER_AREA) for item in frames
    ]).astype(np.float32)

    distance = np.empty((count, count), dtype=np.float32)
    for i in range(count):
        distance[i] = np.abs(small - small[i]).mean(axis=(1, 2))

    runs = count - block_size + 1
    window = np.zeros((runs, runs), dtype=np.float32)
    for step in range(block_size):
        window += distance[step:step + runs, step:step + runs]
    window /= block_size

    step_motion = np.abs(np.diff(small, axis=0)).mean(axis=(1, 2))
    findings = []
    used = set()
    for i in range(runs):
        if step_motion[i:i + block_size - 1].mean() < min_motion:
            continue
        for j in range(i + block_size, runs):
            if window[i, j] <= max_difference and (j - i) not in used:
                used.add(j - i)
                findings.append({
                    "frame_index": frames[j]["frame_index"],
                    "timestamp": frames[j]["timestamp"],
                    "original_timestamp": frames[i]["timestamp"],
                    "difference_score": float(window[i, j]),
                    "evidence": (f"A stretch of {block_size} frames starting at "
                                 f"{frames[i]['timestamp']:.2f}s appears again at {frames[j]['timestamp']:.2f}s"),
                    "type": "possible_repeated_sequence"
                })
                break
    return findings

# INSERTED CLIPS (foreign footage cut into the video)
def find_inserted_clips(scene_changes, max_seconds=2.0):
    """Two scene changes close together mean a short clip was cut in and out again."""
    findings = []
    for first, second in zip(scene_changes, scene_changes[1:]):
        gap = second["timestamp"] - first["timestamp"]
        if 0 < gap <= max_seconds:
            findings.append({
                "frame_index": first["frame_index"],
                "timestamp": first["timestamp"],
                "end_timestamp": second["timestamp"],
                "evidence": (f"The picture changes completely at {first['timestamp']:.2f}s and again at "
                             f"{second['timestamp']:.2f}s, which can mean a clip was inserted"),
                "type": "possible_inserted_clip"
            })
    return findings

# GROUPED FINDINGS W/ INTERVALS
def group_findings(findings, gap_seconds=1.0):
    if not findings:
        return []

    grouped = []
    sorted_findings = sorted(
        findings,
        key=lambda item: (
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

    frames = extract_frames(
        video_path,
        sample_every=sample_every,
        resize_width=resize_width
    )

    similarity_results = analyze_frame_similarity(
        frames,
        duplicate_threshold=duplicate_threshold,
        pixel_threshold=pixel_threshold,
        scene_threshold=scene_threshold
    )

    temporal = analyze_temporal_consistency(
        similarity_results["comparisons"],
        scene_threshold=scene_threshold,
        sensitivity=temporal_sensitivity
    )

    repeated = find_repeated_blocks(frames)
    inserted = find_inserted_clips(similarity_results["scene_changes"])
    
    all_findings = (
        similarity_results["duplicate_findings"]
        + temporal["findings"] + repeated + inserted
    )

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
            "comparisons_performed": len(similarity_results["comparisons"]),
            "duplicate_candidates": len(similarity_results["duplicate_findings"]),
            "scene_change_candidates": len(similarity_results["scene_changes"]),
            "temporal_anomalies": len(temporal["findings"]),
            "repeated_sequences": len(repeated),
            "inserted_clips": len(inserted),
            "suspicious_intervals": len(intervals)
        },

        "duplicate_findings": similarity_results["duplicate_findings"],
        "scene_changes": similarity_results["scene_changes"],
        "temporal_findings": temporal["findings"],
        "repeated_findings": repeated,
        "inserted_clip_findings": inserted,
        "temporal_threshold": temporal["threshold"],
        "suspicious_intervals": intervals,
        "comparisons": similarity_results["comparisons"],

        "limitations": [
            "Similar frames can occur naturally in static scenes.",
            "Scene changes are not automatically evidence of manipulation.",
            "Compression can affect pixel level comparisons.",
            "Thresholds require calibration on labeled videos.",
            "This module identifies potential anomalies, not definitive proof."
        ]
    }

# JSON REPORT EXPORT
def save_report(report, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    return str(output_path)
