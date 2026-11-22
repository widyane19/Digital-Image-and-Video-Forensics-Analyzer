"""Image comparison, copy-move candidate detection, and visual evidence tools."""

import cv2
import numpy as np

try:  # imported as part of the src package (app.py, evaluation.py)
    from .compression_analysis import load_image
except ImportError:  # run directly, or with src/ on the path (tests/test_image_analysis.py)
    from compression_analysis import load_image

def compare_images(first_path, second_path):
    """Compare two same-sized images using pixel-difference measurements."""
    first = np.asarray(load_image(first_path), dtype=np.float64)
    second = np.asarray(load_image(second_path), dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("Images must have the same dimensions to compare pixels.")

    difference = np.abs(first - second)
    mse = float(np.mean((first - second) ** 2))
    return {
        "mse": mse,
        "mean_absolute_difference": float(difference.mean()),
        "identical": mse == 0,
        "width": int(first.shape[1]),
        "height": int(first.shape[0]),
    }


def _bounding_box(points, padding, width, height):
    """Create a clipped (x, y, width, height) box around keypoints."""
    points = np.asarray(points, dtype=np.float32)
    x1 = max(0, int(np.floor(points[:, 0].min())) - padding)
    y1 = max(0, int(np.floor(points[:, 1].min())) - padding)
    x2 = min(width, int(np.ceil(points[:, 0].max())) + padding + 1)
    y2 = min(height, int(np.ceil(points[:, 1].max())) + padding + 1)
    return (x1, y1, x2 - x1, y2 - y1)


def orb_copy_move_detection(image_path, min_matches=6, min_displacement=20, max_distance=55):
    """Find possible duplicated regions using ORB feature self-matching.

    Returned boxes are candidates for human review, not proof of copy-move editing.
    Repeated patterns can produce false positives; smooth regions may be missed.
    """
    if min_matches < 2 or min_displacement < 1 or max_distance <= 0:
        raise ValueError("Use min_matches >= 2, min_displacement >= 1, and max_distance > 0.")

    rgb = np.asarray(load_image(image_path))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    detector = cv2.ORB_create(nfeatures=4000)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    empty_result = {"match_count": 0, "candidate_regions": [], "matches": []}
    if descriptors is None or len(keypoints) < min_matches + 1:
        return empty_result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    neighbor_lists = matcher.knnMatch(descriptors, descriptors, k=min(12, len(descriptors)))
    # Group matches with approximately the same translation vector.
    groups = {}
    seen_pairs = set()
    for query_index, neighbors in enumerate(neighbor_lists):
        for match in neighbors:
            train_index = match.trainIdx
            if query_index == train_index or match.distance > max_distance:
                continue
            pair = tuple(sorted((query_index, train_index)))
            if pair in seen_pairs:
                continue

            first = np.array(keypoints[query_index].pt, dtype=float)
            second = np.array(keypoints[train_index].pt, dtype=float)
            dx, dy = second - first
            if np.hypot(dx, dy) < min_displacement:
                continue
            seen_pairs.add(pair)

            # Make the translation direction consistent so matching pairs group together.
            if dx < 0 or (dx == 0 and dy < 0):
                first, second = second, first
                dx, dy = -dx, -dy
            bucket = (int(round(dx / 8.0)), int(round(dy / 8.0)))
            groups.setdefault(bucket, []).append((first, second, float(match.distance)))
            break  # Keep the closest acceptable non-self match for each keypoint.

    candidates = []
    all_matches = []
    for displacement, pairs in groups.items():
        if len(pairs) < min_matches:
            continue
        source_points = np.array([pair[0] for pair in pairs])
        destination_points = np.array([pair[1] for pair in pairs])
        source_box = _bounding_box(source_points, 10, width, height)
        destination_box = _bounding_box(destination_points, 10, width, height)
        # Ignore groups whose two boxes substantially overlap; they are often local matches.
        if _box_iou(source_box, destination_box) > 0.25:
            continue
        candidates.append({
            "source": source_box,
            "destination": destination_box,
            "match_count": len(pairs),
            "displacement": (displacement[0] * 8, displacement[1] * 8),
        })
        all_matches.extend([
            {"source_point": tuple(map(float, first)), "destination_point": tuple(map(float, second)), "distance": distance}
            for first, second, distance in pairs
        ])

    candidates.sort(key=lambda item: item["match_count"], reverse=True)
    return {"match_count": len(all_matches), "candidate_regions": candidates, "matches": all_matches}


def _box_iou(first, second):
    """Calculate intersection-over-union for (x, y, width, height) boxes."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def region_iou(first_box, second_box):
    """Public helper: intersection-over-union for two (x, y, w, h) boxes."""
    return float(_box_iou(first_box, second_box))


def evaluate_copy_move_detection(detection_result, known_regions, iou_threshold=0.25):
    """Compare detected boxes with ground-truth boxes and report precision/recall.

    `known_regions` should contain every copied source and destination box. This is
    region-level evaluation; image-level accuracy and false-positive rate need a
    labeled dataset of both manipulated and untouched images.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1].")
    predicted = []
    for candidate in detection_result.get("candidate_regions", []):
        predicted.extend([candidate["source"], candidate["destination"]])

    unmatched_truth = list(known_regions)
    true_positives = 0
    for predicted_box in predicted:
        best_index = None
        best_score = iou_threshold
        for index, truth_box in enumerate(unmatched_truth):
            score = _box_iou(predicted_box, truth_box)
            if score >= best_score:
                best_score, best_index = score, index
        if best_index is not None:
            true_positives += 1
            unmatched_truth.pop(best_index)

    false_positives = len(predicted) - true_positives
    false_negatives = len(unmatched_truth)
    precision = true_positives / (true_positives + false_positives) if predicted else 0.0
    recall = true_positives / (true_positives + false_negatives) if known_regions else 0.0
    return {
        "detected_regions": true_positives,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "all_detected": false_negatives == 0 and bool(known_regions),
    }


def edge_texture_analysis(image_path):
    """Return edge density and texture-strength measurements for visual comparison.

    These are descriptive features, not standalone manipulation classifiers.
    """
    gray = cv2.cvtColor(np.asarray(load_image(image_path)), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "edge_density": float(np.count_nonzero(edges) / edges.size),
        "laplacian_variance": float(laplacian.var()),
        "edge_map": edges,
    }


def create_match_visualization(image_path, detection_result, output_path=None):
    """Draw detected copy-move candidate boxes over the image."""
    image = np.asarray(load_image(image_path)).copy()
    for candidate in detection_result.get("candidate_regions", []):
        for name in ("source", "destination"):
            x, y, width, height = candidate[name]
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    if output_path is not None:
        # OpenCV expects BGR when writing a color image.
        if not cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write visualization to {output_path}")
    return image


def evaluate_manipulation_predictions(actual_labels, predicted_labels):
    """Calculate image-level accuracy, precision, recall, and false-positive rate.

    Labels must be 1 for manipulated and 0 for original, in matching dataset order.
    Use this with predictions from a defined decision rule and labeled test images.
    """
    actual = np.asarray(actual_labels, dtype=int).ravel()
    predicted = np.asarray(predicted_labels, dtype=int).ravel()
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Provide equally sized, non-empty label lists.")
    if not np.isin(actual, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Labels must be 0 (original) or 1 (manipulated).")

    true_positive = int(np.sum((actual == 1) & (predicted == 1)))
    true_negative = int(np.sum((actual == 0) & (predicted == 0)))
    false_positive = int(np.sum((actual == 0) & (predicted == 1)))
    false_negative = int(np.sum((actual == 1) & (predicted == 0)))
    total = actual.size
    return {
        "accuracy": (true_positive + true_negative) / total,
        "precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0,
        "recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0,
        "false_positive_rate": false_positive / (false_positive + true_negative) if false_positive + true_negative else 0.0,
        "true_positives": true_positive,
        "true_negatives": true_negative,
        "false_positives": false_positive,
        "false_negatives": false_negative,
    }
