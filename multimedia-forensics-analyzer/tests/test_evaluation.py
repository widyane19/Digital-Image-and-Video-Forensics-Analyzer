"""Tests for src/evaluation.py.

Most tests use hand-made labels and decisions and need nothing installed. The
end-to-end test builds a small dataset with src/build_dataset.py and is skipped
without ExifTool.

Run from the multimedia-forensics-analyzer folder:  python -m pytest
"""

import csv
import shutil

import pytest

from src.evaluation import (
    ALL_TECHNIQUES,
    DIFFICULTY_COLUMNS,
    MANIPULATION_COLUMNS,
    TECHNIQUE_COLUMNS,
    EvaluationResults,
    FileResult,
    LabelledFile,
    _combine,
    difficulty_table,
    evaluate,
    evaluate_manipulation_predictions,
    load_labels,
    manipulation_table,
    technique_table,
    write_results,
)

requires_exiftool = pytest.mark.skipif(
    shutil.which("exiftool") is None, reason="ExifTool is not installed or not on PATH"
)

BRIEF_METRICS = ["precision", "recall", "accuracy", "false_positive_rate"]


def _result(name, manipulated, kind="none", **statuses):
    statuses[ALL_TECHNIQUES] = _combine(statuses.values())
    return FileResult(LabelledFile(name, None, manipulated, kind if manipulated else "none"), statuses)


def _row(rows, technique, kind=None):
    return next(r for r in rows if r["technique"] == technique
                and (kind is None or r.get("manipulation_type") == kind))


# ---------------------------------------------------------------------------
# Core metric function
# ---------------------------------------------------------------------------

class TestManipulationMetrics:
    def test_brief_metrics_from_known_counts(self):
        # 3 true positives, 1 false positive, 2 false negatives, 4 true negatives
        actual = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
        predicted = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        result = evaluate_manipulation_predictions(actual, predicted)
        assert result["precision"] == pytest.approx(0.75)
        assert result["recall"] == pytest.approx(0.6)
        assert result["accuracy"] == pytest.approx(0.7)
        assert result["false_positive_rate"] == pytest.approx(0.2)

    def test_confusion_matrix_values(self):
        result = evaluate_manipulation_predictions([1, 1, 0, 0], [1, 0, 1, 0])
        assert (result["true_positives"], result["true_negatives"],
                result["false_positives"], result["false_negatives"]) == (1, 1, 1, 1)
        assert result["accuracy"] == result["precision"] == result["recall"] == 0.5
        assert result["false_positive_rate"] == 0.5

    def test_metrics_that_cannot_be_calculated_are_none(self):
        only_originals_never_flagged = evaluate_manipulation_predictions([0, 0], [0, 0])
        assert only_originals_never_flagged["precision"] is None   # nothing flagged
        assert only_originals_never_flagged["recall"] is None      # nothing manipulated
        assert only_originals_never_flagged["false_positive_rate"] == 0
        assert only_originals_never_flagged["accuracy"] == 1

    def test_validates_labels(self):
        with pytest.raises(ValueError, match="equally sized"):
            evaluate_manipulation_predictions([0, 1], [1])
        with pytest.raises(ValueError, match=r"0 \(original\) or 1 \(manipulated\)"):
            evaluate_manipulation_predictions([0, 2], [0, 1])


# ---------------------------------------------------------------------------
# Label file
# ---------------------------------------------------------------------------

class TestLoadLabels:
    def _write(self, tmp_path, text):
        path = tmp_path / "labels.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_rows(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.mp4").write_bytes(b"x")
        labels = self._write(tmp_path, "file,label,manipulation_type\n"
                                       "a.jpg,original,none\n"
                                       "b.mp4,Manipulated,Frame_Removal\n")
        files, warnings = load_labels(labels)
        assert warnings == []
        assert [(f.file, f.manipulated, f.manipulation_type) for f in files] == [
            ("a.jpg", False, "none"), ("b.mp4", True, "frame_removal")]

    def test_extra_columns_are_ignored(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        labels = self._write(tmp_path, "file,label,manipulation_type,regions,notes\n"
                                       'a.jpg,original,none,"1,2,3,4",from my phone\n')
        files, _ = load_labels(labels)
        assert len(files) == 1

    def test_missing_file_is_left_out_with_a_warning(self, tmp_path):
        labels = self._write(tmp_path, "file,label,manipulation_type\ngone.jpg,original,none\n")
        files, warnings = load_labels(labels)
        assert files == [] and warnings == ["gone.jpg: file not found, left out"]

    @pytest.mark.parametrize("row, message", [
        ("a.jpg,fake,none", "label must be"),
        ("a.jpg,original,crop", "original file must have"),
        ("a.jpg,manipulated,none", "needs a manipulation_type"),
    ])
    def test_invalid_rows(self, tmp_path, row, message):
        labels = self._write(tmp_path, f"file,label,manipulation_type\n{row}\n")
        with pytest.raises(ValueError, match=message):
            load_labels(labels)

    def test_missing_columns(self, tmp_path):
        with pytest.raises(ValueError, match="manipulation_type"):
            load_labels(self._write(tmp_path, "file,label\na.jpg,original\n"))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

RESULTS = [
    # originals
    _result("o1.jpg", False, **{"metadata.software": "not_flagged", "metadata.video_only": "not_applicable"}),
    _result("o2.jpg", False, **{"metadata.software": "flagged", "metadata.video_only": "not_applicable"}),
    _result("o3.mp4", False, **{"metadata.software": "not_flagged", "metadata.video_only": "not_flagged"}),
    # manipulated
    _result("m1.jpg", True, "crop", **{"metadata.software": "flagged", "metadata.video_only": "not_applicable"}),
    _result("m2.jpg", True, "splice", **{"metadata.software": "not_flagged", "metadata.video_only": "not_applicable"}),
    _result("m3.mp4", True, "frame_removal", **{"metadata.software": "not_flagged", "metadata.video_only": "flagged"}),
]


class TestTechniqueTable:
    def test_only_the_brief_metrics(self):
        assert TECHNIQUE_COLUMNS == ["technique", *BRIEF_METRICS]
        assert all(list(row) == TECHNIQUE_COLUMNS for row in technique_table(RESULTS))

    def test_one_row_per_technique_then_the_whole_detector(self):
        assert [r["technique"] for r in technique_table(RESULTS)] == [
            "metadata.software", "metadata.video_only", ALL_TECHNIQUES]

    def test_metrics_for_one_technique(self):
        row = _row(technique_table(RESULTS), "metadata.software")
        # TP: m1. FP: o2. FN: m2, m3. TN: o1, o3.
        assert row["precision"] == 0.5
        assert row["recall"] == pytest.approx(1 / 3)
        assert row["accuracy"] == 0.5
        assert row["false_positive_rate"] == pytest.approx(1 / 3)

    def test_not_applicable_files_are_left_out(self):
        row = _row(technique_table(RESULTS), "metadata.video_only")
        # Only the two videos count: m3 caught, o3 correctly not flagged.
        assert row["recall"] == 1 and row["false_positive_rate"] == 0 and row["accuracy"] == 1

    def test_detector_flags_when_any_technique_flags(self):
        row = _row(technique_table(RESULTS), ALL_TECHNIQUES)
        # m1 (software) and m3 (video_only) caught; o2 is a false positive.
        assert row["recall"] == pytest.approx(2 / 3)
        assert row["false_positive_rate"] == pytest.approx(1 / 3)

    def test_empty_results(self):
        assert technique_table([]) == []


class TestManipulationTable:
    def test_one_row_per_technique_and_type(self):
        rows = manipulation_table(RESULTS)
        assert MANIPULATION_COLUMNS == ["technique", "manipulation_type", *BRIEF_METRICS]
        assert all(list(row) == MANIPULATION_COLUMNS for row in rows)
        software = {r["manipulation_type"]: r for r in rows if r["technique"] == "metadata.software"}
        assert set(software) == {"crop", "splice", "frame_removal"}
        assert software["crop"]["recall"] == 1
        assert software["splice"]["recall"] == 0

    def test_false_positive_rate_is_the_same_for_every_type(self):
        software = [r for r in manipulation_table(RESULTS) if r["technique"] == "metadata.software"]
        assert all(r["false_positive_rate"] == pytest.approx(1 / 3) for r in software)

    def test_no_row_when_the_technique_never_assessed_that_type(self):
        video_only = {r["manipulation_type"] for r in manipulation_table(RESULTS)
                      if r["technique"] == "metadata.video_only"}
        assert video_only == {"frame_removal"}


class TestDifficultyTable:
    rows = [
        {"technique": "a", "manipulation_type": "splice", "recall": 0.5},
        {"technique": "b", "manipulation_type": "splice", "recall": 0.25},
        {"technique": ALL_TECHNIQUES, "manipulation_type": "splice", "recall": 0.5},
        {"technique": "a", "manipulation_type": "frame_duplication", "recall": 0.0},
        {"technique": ALL_TECHNIQUES, "manipulation_type": "frame_duplication", "recall": 0.0},
        {"technique": "a", "manipulation_type": "crop", "recall": 0.75},
        {"technique": "b", "manipulation_type": "crop", "recall": 1.0},
        {"technique": ALL_TECHNIQUES, "manipulation_type": "crop", "recall": 1.0},
        {"technique": "a", "manipulation_type": "unassessed", "recall": None},
        {"technique": ALL_TECHNIQUES, "manipulation_type": "unassessed", "recall": None},
    ]

    def test_ranked_easiest_to_hardest(self):
        table = difficulty_table(self.rows)
        assert [r["manipulation_type"] for r in table] == ["crop", "splice", "frame_duplication", "unassessed"]
        assert [r["rank"] for r in table] == [1, 2, 3, 4]
        assert all(list(r) == DIFFICULTY_COLUMNS for r in table)

    def test_best_technique(self):
        table = {r["manipulation_type"]: r for r in difficulty_table(self.rows)}
        assert (table["crop"]["best_technique"], table["crop"]["best_technique_recall"]) == ("b", 1.0)
        assert (table["splice"]["best_technique"], table["splice"]["detector_recall"]) == ("a", 0.5)
        assert table["unassessed"]["best_technique"] is None

    def test_ties_broken_by_best_technique_then_name(self):
        rows = [
            {"technique": ALL_TECHNIQUES, "manipulation_type": "b_type", "recall": 0.5},
            {"technique": "x", "manipulation_type": "b_type", "recall": 0.5},
            {"technique": ALL_TECHNIQUES, "manipulation_type": "a_type", "recall": 0.5},
            {"technique": "x", "manipulation_type": "a_type", "recall": 0.5},
            {"technique": ALL_TECHNIQUES, "manipulation_type": "c_type", "recall": 0.5},
            {"technique": "x", "manipulation_type": "c_type", "recall": 0.25},
        ]
        assert [r["manipulation_type"] for r in difficulty_table(rows)] == ["a_type", "b_type", "c_type"]

    def test_from_real_tables(self):
        table = difficulty_table(manipulation_table(RESULTS))
        assert {r["manipulation_type"] for r in table} == {"crop", "splice", "frame_removal"}
        assert table[-1]["manipulation_type"] == "splice"      # nothing caught it


def test_write_results(tmp_path):
    by_manipulation = manipulation_table(RESULTS)
    results = EvaluationResults(technique_table(RESULTS), by_manipulation,
                                difficulty_table(by_manipulation), [])
    written = write_results(results, tmp_path)
    assert [p.name for p in written] == ["metrics.csv", "by_manipulation.csv", "difficulty.csv"]
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == TECHNIQUE_COLUMNS
    assert rows[0]["precision"] == "0.5" and rows[0]["recall"] == "0.3333"


# ---------------------------------------------------------------------------
# End to end: a dataset made by build_dataset.py
# ---------------------------------------------------------------------------

@requires_exiftool
def test_end_to_end(tmp_path):
    from tests.test_build_dataset import make_originals  # pylint: disable=import-outside-toplevel
    from src.build_dataset import build_dataset          # pylint: disable=import-outside-toplevel

    data = tmp_path / "data"
    make_originals(data, images=5, videos=2)
    rows, _ = build_dataset(data)

    results = evaluate(data / "labels.csv")
    written = write_results(results, tmp_path / "results")

    assert sorted(p.name for p in written) == ["by_manipulation.csv", "difficulty.csv", "metrics.csv"]
    for path, columns in zip(written, [TECHNIQUE_COLUMNS, MANIPULATION_COLUMNS, DIFFICULTY_COLUMNS]):
        with open(path, newline="", encoding="utf-8") as handle:
            assert next(csv.reader(handle)) == columns
    assert results.warnings == []

    generated = {row.manipulation_type for row in rows} - {"none"}
    assert {r["manipulation_type"] for r in results.difficulty} == generated

    software = _row(results.by_manipulation, "metadata.editing_software", "metadata_edit")
    assert software["recall"] == 1.0
    assert _row(results.by_manipulation, "metadata.size_mismatch", "crop")["recall"] == 1.0
    assert _row(results.by_manipulation, "metadata.frame_timing", "frame_removal")["recall"] == 1.0
