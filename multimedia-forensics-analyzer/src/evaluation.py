"""Evaluation: measure each forensic technique against the dataset's ground-truth labels.

Returns only what the project brief's Experimental Evaluation asks for:
    - precision, recall, detection accuracy and false-positive rate, for each
      technique and for the detector as a whole
    - the same metrics for each manipulation type
    - which manipulation types are easier or harder to detect

It reads each technique's own decision for each file (flagged, not flagged or not
applicable). It never uses the web app's report (pipeline.py), its suspicion score,
verdict, severity weights or visualizations.

Run from the multimedia-forensics-analyzer folder:
    python -m src.evaluation --labels ../data/labels.csv --data ../data
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

try:  # imported as part of the src package (app.py, tests)
    from .image_analysis import orb_copy_move_detection
    from .metadata.metadata_extraction import open_exiftool
    from .metadata_analysis import analyze_file
except ImportError:  # run directly: python src/evaluation.py
    from image_analysis import orb_copy_move_detection
    from metadata.metadata_extraction import open_exiftool
    from metadata_analysis import analyze_file

FLAGGED, NOT_FLAGGED, NOT_APPLICABLE = "flagged", "not_flagged", "not_applicable"
ALL_TECHNIQUES = "all"   # the detector as a whole: flagged when any technique flags the file

METRIC_COLUMNS = ["precision", "recall", "accuracy", "false_positive_rate"]
TECHNIQUE_COLUMNS = ["technique", *METRIC_COLUMNS]
MANIPULATION_COLUMNS = ["technique", "manipulation_type", *METRIC_COLUMNS]
DIFFICULTY_COLUMNS = ["rank", "manipulation_type", "detector_recall", "best_technique",
                      "best_technique_recall"]

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
# 1. Ground truth
# ---------------------------------------------------------------------------

@dataclass
class LabelledFile:
    """One row of labels.csv."""

    file: str                      # path relative to the data folder, as written in labels.csv
    path: Path
    manipulated: bool
    manipulation_type: str         # "none" for originals


def load_labels(labels_path: str | os.PathLike,
                data_dir: str | os.PathLike | None = None) -> tuple[list[LabelledFile], list[str]]:
    """Read labels.csv. Returns (files, warnings); missing files are skipped with a warning.

    Columns: file, label (original/manipulated) and manipulation_type. Any other
    columns are ignored.
    """
    labels_path = Path(labels_path)
    data_dir = Path(data_dir) if data_dir is not None else labels_path.parent

    with open(labels_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"file", "label", "manipulation_type"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{labels_path} is missing column(s): {', '.join(sorted(missing))}")
        rows = list(reader)

    files, warnings = [], []
    for line, row in enumerate(rows, start=2):  # line 1 is the header
        name = (row.get("file") or "").strip()
        if not name:
            continue
        label = (row.get("label") or "").strip().lower()
        kind = (row.get("manipulation_type") or "").strip().lower() or "none"
        if label not in ("original", "manipulated"):
            raise ValueError(f"Line {line}: label must be 'original' or 'manipulated', not {label!r}")
        if label == "original" and kind != "none":
            raise ValueError(f"Line {line}: an original file must have manipulation_type 'none'")
        if label == "manipulated" and kind == "none":
            raise ValueError(f"Line {line}: a manipulated file needs a manipulation_type")

        path = data_dir / name
        if not path.is_file():
            warnings.append(f"{name}: file not found, left out")
            continue
        files.append(LabelledFile(name, path, label == "manipulated", kind))
    return files, warnings


# ---------------------------------------------------------------------------
# 2. Technique registry
# ---------------------------------------------------------------------------

def _metadata_detector(path: Path, et) -> dict[str, str]:
    report = analyze_file(path, et)
    return {f"metadata.{name}": status for name, status in report.results.items()}


def _image_detector(path: Path, et) -> dict[str, str]:  # pylint: disable=unused-argument
    if not _is_image(path):
        return {"image.copy_move": NOT_APPLICABLE}
    found = orb_copy_move_detection(str(path))["candidate_regions"]
    return {"image.copy_move": FLAGGED if found else NOT_FLAGGED}


def _is_image(path: Path) -> bool:
    from PIL import Image, UnidentifiedImageError  # pylint: disable=import-outside-toplevel
    try:
        with Image.open(path):
            return True
    except (UnidentifiedImageError, OSError):
        return False


# Detector -> function(path, et) returning {technique: status}. `et` is a shared, open
# ExifTool process; detectors that don't need it ignore it. Add an entry to evaluate
# a new detector; techniques that only measure (no flagged/not-flagged decision)
# can't be evaluated until they make one.
DETECTORS: dict[str, Callable[[Path, object], dict[str, str]]] = {
    "metadata": _metadata_detector,
    "image": _image_detector,
}


# ---------------------------------------------------------------------------
# 3. Running the techniques
# ---------------------------------------------------------------------------

@dataclass
class FileResult:
    item: LabelledFile
    statuses: dict[str, str]       # every technique, plus ALL_TECHNIQUES


def run_techniques(files: Iterable[LabelledFile],
                   detectors: dict[str, Callable] | None = None) -> tuple[list[FileResult], list[str]]:
    """Run every detector on every file. A detector that fails on a file is skipped for
    that file and reported in the warnings; the other detectors still count."""
    detectors = DETECTORS if detectors is None else detectors
    results, warnings = [], []
    with open_exiftool() as et:
        for item in files:
            statuses = {}
            for name, detect in detectors.items():
                try:
                    statuses.update(detect(item.path, et))
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    # One detector failing on one file must not stop the evaluation.
                    warnings.append(f"{item.file}: {name} could not run ({exc})")
            if statuses:
                statuses[ALL_TECHNIQUES] = _combine(statuses.values())
                results.append(FileResult(item, statuses))
    return results, warnings


def _combine(statuses: Iterable[str]) -> str:
    """The detector flags a file when any of its techniques does."""
    statuses = list(statuses)
    if FLAGGED in statuses:
        return FLAGGED
    return NOT_FLAGGED if NOT_FLAGGED in statuses else NOT_APPLICABLE


# ---------------------------------------------------------------------------
# 4. Metrics
# ---------------------------------------------------------------------------

def evaluate_manipulation_predictions(actual_labels, predicted_labels) -> dict:
    """Calculate accuracy, precision, recall and false-positive rate.

    Labels must be 1 for manipulated and 0 for original, in matching order. A metric
    whose denominator is zero can't be calculated and is returned as None.
    """
    actual = np.asarray(actual_labels, dtype=int).ravel()
    predicted = np.asarray(predicted_labels, dtype=int).ravel()
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Provide equally sized, non-empty label lists.")
    if not np.isin(actual, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("Labels must be 0 (original) or 1 (manipulated).")

    tp = int(np.sum((actual == 1) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    return {
        "accuracy": (tp + tn) / actual.size,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "false_positive_rate": _ratio(fp, fp + tn),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _metrics(pairs: list[tuple[bool, bool]]) -> dict:
    """The brief's four metrics for (manipulated, flagged) pairs."""
    if not pairs:
        return dict.fromkeys(METRIC_COLUMNS)
    actual, predicted = zip(*pairs)
    result = evaluate_manipulation_predictions(actual, predicted)
    return {k: result[k] for k in METRIC_COLUMNS}


def _assessed(results: list[FileResult], technique: str) -> list[tuple[FileResult, bool]]:
    """(file, flagged) for every file the technique could assess."""
    return [(r, r.statuses[technique] == FLAGGED) for r in results
            if r.statuses.get(technique, NOT_APPLICABLE) != NOT_APPLICABLE]


def _technique_order(results: list[FileResult]) -> list[str]:
    """Every technique alphabetically, with the detector as a whole last."""
    names = {name for r in results for name in r.statuses} - {ALL_TECHNIQUES}
    return sorted(names) + [ALL_TECHNIQUES] if names else []


# ---------------------------------------------------------------------------
# 5. The result tables
# ---------------------------------------------------------------------------

def technique_table(results: list[FileResult]) -> list[dict]:
    """One row per technique, and one for the detector as a whole."""
    return [{"technique": name,
             **_metrics([(r.item.manipulated, flagged) for r, flagged in _assessed(results, name)])}
            for name in _technique_order(results)]


def manipulation_table(results: list[FileResult]) -> list[dict]:
    """One row per technique and manipulation type, using that type's files plus all originals.

    Rows are only produced when the technique assessed at least one file of that type.
    """
    kinds = sorted({r.item.manipulation_type for r in results if r.item.manipulated})
    rows = []
    for name in _technique_order(results):
        assessed = _assessed(results, name)
        for kind in kinds:
            subset = [(r.item.manipulated, flagged) for r, flagged in assessed
                      if not r.item.manipulated or r.item.manipulation_type == kind]
            if any(manipulated for manipulated, _ in subset):
                rows.append({"technique": name, "manipulation_type": kind, **_metrics(subset)})
    return rows


def difficulty_table(by_manipulation: list[dict]) -> list[dict]:
    """Manipulation types ranked from easiest to hardest to detect.

    Difficulty is measured by recall: how many files of that type the detector as a
    whole caught. The best single technique for each type is shown alongside it.
    """
    rows = []
    for kind in dict.fromkeys(row["manipulation_type"] for row in by_manipulation):
        of_kind = [row for row in by_manipulation if row["manipulation_type"] == kind]
        detector = next((row["recall"] for row in of_kind if row["technique"] == ALL_TECHNIQUES), None)
        techniques = [row for row in of_kind
                      if row["technique"] != ALL_TECHNIQUES and row["recall"] is not None]
        best = min(techniques, key=lambda row: (-row["recall"], row["technique"]), default=None)
        rows.append({"manipulation_type": kind, "detector_recall": detector,
                     "best_technique": best["technique"] if best else None,
                     "best_technique_recall": best["recall"] if best else None})

    rows.sort(key=lambda row: (row["detector_recall"] is None, -(row["detector_recall"] or 0),
                               -(row["best_technique_recall"] or 0), row["manipulation_type"]))
    return [{"rank": rank, **row} for rank, row in enumerate(rows, start=1)]


@dataclass
class EvaluationResults:
    techniques: list[dict]
    by_manipulation: list[dict]
    difficulty: list[dict]
    warnings: list[str]


def evaluate(labels_paths: str | os.PathLike | list, data_dir: str | os.PathLike | None = None,
             detectors: dict[str, Callable] | None = None) -> EvaluationResults:
    """Run every technique on the labelled dataset and return the result tables.

    `labels_paths` is one label file or a list of them (for example our generated set
    plus a public dataset); their rows are measured together.
    """
    if isinstance(labels_paths, (str, os.PathLike)):
        labels_paths = [labels_paths]
    files, warnings = [], []
    for labels_path in labels_paths:
        more_files, more_warnings = load_labels(labels_path, data_dir)
        files += more_files
        warnings += more_warnings
    results, run_warnings = run_techniques(files, detectors)
    by_manipulation = manipulation_table(results)
    return EvaluationResults(technique_table(results), by_manipulation,
                             difficulty_table(by_manipulation), warnings + run_warnings)


# ---------------------------------------------------------------------------
# 6. Output
# ---------------------------------------------------------------------------

def write_results(results: EvaluationResults, out_dir: str | os.PathLike) -> list[Path]:
    """Write metrics.csv, by_manipulation.csv and difficulty.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = [("metrics.csv", TECHNIQUE_COLUMNS, results.techniques),
              ("by_manipulation.csv", MANIPULATION_COLUMNS, results.by_manipulation),
              ("difficulty.csv", DIFFICULTY_COLUMNS, results.difficulty)]

    written = []
    for name, columns, rows in tables:
        path = out_dir / name
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows({k: _cell(row[k]) for k in columns} for row in rows)
        written.append(path)
    return written


def _cell(value):
    if value is None:
        return ""
    return round(value, 4) if isinstance(value, float) else value


def format_table(rows: list[dict], columns: list[str]) -> str:
    """A plain-text table for the terminal; blanks are shown as '-'."""
    cells = [[str(_cell(row[c])) or "-" for c in columns] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) if cells else len(c)
              for i, c in enumerate(columns)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(columns, widths))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in cells]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Measure each technique against the labelled dataset.")
    parser.add_argument("--labels", nargs="+", default=[DEFAULT_DATA_DIR / "labels.csv"],
                        help="one or more label files, measured together")
    parser.add_argument("--data", default=None, help="folder the label paths are relative to "
                                                     "(default: the folder holding each label file)")
    parser.add_argument("--out", default=DEFAULT_RESULTS_DIR, help="folder for the result CSVs")
    args = parser.parse_args()

    for labels_file in args.labels:
        if not Path(labels_file).is_file():
            sys.exit(f"Label file not found: {labels_file}")
    try:
        evaluation = evaluate(args.labels, args.data)
    except FileNotFoundError as exc:
        sys.exit(f"{exc}\nExifTool must be installed and on your PATH. See the README for setup steps.")
    except ValueError as exc:
        sys.exit(f"Problem in the label file: {exc}")

    print(format_table(evaluation.techniques, TECHNIQUE_COLUMNS))
    print("\nManipulation types, easiest to hardest to detect:")
    print(format_table(evaluation.difficulty, DIFFICULTY_COLUMNS))
    for warning in evaluation.warnings:
        print(f"warning: {warning}")
    for csv_path in write_results(evaluation, args.out):
        print(f"wrote {csv_path}")
