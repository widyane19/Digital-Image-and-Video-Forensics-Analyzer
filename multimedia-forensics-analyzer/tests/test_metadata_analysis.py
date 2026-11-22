"""Tests for Steps 5-7 of the metadata analysis (src/metadata_analysis.py).

Unit tests need nothing installed. Integration tests are the scenarios listed in
plan/metadata_analysis_plan.md. They create real files and run ExifTool, and are
skipped when ExifTool (or Pillow, NumPy or PyAV, where needed) is not available.

Run from the multimedia-forensics-analyzer folder:  python -m pytest
"""

import json
import shutil
from datetime import datetime, timezone
from fractions import Fraction

import pytest

from src.metadata.metadata_checks import Finding
from src.metadata_analysis import MetadataReport, analyze_batch, analyze_file, suspicion_score

requires_exiftool = pytest.mark.skipif(
    shutil.which("exiftool") is None, reason="ExifTool is not installed or not on PATH"
)


def _findings(*severities):
    return [Finding("test", severity, "message") for severity in severities]


def _report(findings=(), results=None):
    return MetadataReport("a.jpg", "image", suspicion_score(findings), list(findings),
                          results or {"one": "not_flagged"}, [])


# ---------------------------------------------------------------------------
# Step 6: the score
# ---------------------------------------------------------------------------

class TestSuspicionScore:
    def test_no_findings(self):
        assert suspicion_score([]) == 0

    def test_weak_clues_add_up(self):
        assert suspicion_score(_findings("medium", "medium")) == pytest.approx(0.51)

    def test_more_evidence_always_raises_the_score(self):
        assert suspicion_score(_findings("high", "low")) > suspicion_score(_findings("high"))

    def test_never_goes_above_one(self):
        assert suspicion_score(_findings(*["high"] * 50)) <= 1

    def test_info_adds_nothing(self):
        assert suspicion_score(_findings("info")) == 0


# ---------------------------------------------------------------------------
# Step 7: output
# ---------------------------------------------------------------------------

class TestReport:
    many_checks = {name: "not_flagged" for name in "abcdef"}

    def test_serious_findings_are_counted(self):
        assert "1 serious" in _report(_findings("high", "low"), self.many_checks).conclusion

    def test_few_checks_means_not_enough_metadata(self):
        assert "Not enough" in _report(_findings("low")).conclusion

    def test_clean_result_does_not_claim_the_file_is_unedited(self):
        assert "doesn't prove" in _report(results=self.many_checks).conclusion

    def test_not_applicable_checks_do_not_count_as_run(self):
        report = _report(results={"one": "flagged", "two": "not_applicable"})
        assert report.checks_run == 1

    def test_summary_row_has_one_column_per_check(self):
        report = _report(results={"one": "flagged", "two": "not_applicable"})
        assert report.summary_row() == {"file": "a.jpg", "score": 0, "one": "flagged", "two": "not_applicable"}

    def test_to_dict_can_be_written_as_json(self):
        finding = Finding("test", "high", "message", regions=[(1, 2, 3, 4)], frames=[5])
        data = json.loads(json.dumps(_report([finding]).to_dict()))
        assert data["findings"][0]["regions"] == [[1, 2, 3, 4]]


# ---------------------------------------------------------------------------
# Integration: the plan's scenarios, on real files
# ---------------------------------------------------------------------------

def _exiftool(path, *args):
    from exiftool import ExifToolHelper

    with ExifToolHelper() as et:
        et.execute("-overwrite_original", *args, str(path))


def _scene(width=640, height=480):
    """A smooth test picture, so a preview and its full image match closely."""
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")

    x = np.linspace(0, 255, width)[None, :].repeat(height, axis=0)
    y = np.linspace(0, 255, height)[:, None].repeat(width, axis=1)
    return Image.fromarray(np.dstack([x, y, 255 - x]).astype("uint8"))


def _embed_preview(path, picture, tmp_path):
    """Store a small copy of `picture` inside the photo, as a camera does."""
    thumbnail = tmp_path / "thumbnail.jpg"
    small = picture.copy()
    small.thumbnail((160, 160))
    small.save(thumbnail, quality=90)
    _exiftool(path, f"-ThumbnailImage<={thumbnail}")


@pytest.fixture
def camera_photo(tmp_path):
    """A JPEG with normal camera information and a stored preview."""
    path = tmp_path / "photo.jpg"
    picture = _scene()
    picture.save(path, quality=92)
    _exiftool(path, "-Make=Canon", "-Model=EOS 80D",
              "-DateTimeOriginal=2024:05:01 10:00:00", "-OffsetTimeOriginal=+01:00",
              "-ModifyDate=2024:05:01 10:00:00", "-OffsetTime=+01:00",
              "-ExifImageWidth=640", "-ExifImageHeight=480")
    _embed_preview(path, picture, tmp_path)
    return path


def _resave_keeping_metadata(path, picture):
    """Replace a photo's pixels but keep all its metadata, including the old preview."""
    edited = path.with_name("edited.jpg")
    picture.save(edited, quality=92)
    _exiftool(edited, "-TagsFromFile", str(path), "-all:all", "-ThumbnailImage")
    return edited


@requires_exiftool
class TestPhotoScenarios:
    def test_clean_photo_has_no_serious_findings(self, camera_photo):
        report = analyze_file(camera_photo)
        assert report.warnings == []
        assert report.results["preview_mismatch"] == "not_flagged"
        assert [f for f in report.findings if f.severity in ("medium", "high")] == []

    def test_photoshop_and_later_modified_date(self, camera_photo):
        _exiftool(camera_photo, "-Software=Adobe Photoshop 25.0", "-ModifyDate=2024:06:20 15:00:00")
        report = analyze_file(camera_photo)
        assert report.results["editing_software"] == "flagged"
        assert report.results["modified_after_capture"] == "flagged"
        assert report.findings[0].severity == "high"     # most serious first

    def test_cropped_photo(self, camera_photo):
        cropped = _scene().crop((0, 0, 400, 400))
        report = analyze_file(_resave_keeping_metadata(camera_photo, cropped))
        assert report.results["preview_mismatch"] == "flagged"
        assert "cropped" in next(f for f in report.findings if f.check == "preview_mismatch").message

    def test_pasted_region_is_found_in_the_right_place(self, camera_photo):
        picture = _scene()
        picture.paste((0, 0, 0), (400, 240, 560, 400))
        report = analyze_file(_resave_keeping_metadata(camera_photo, picture))

        finding = next(f for f in report.findings if f.check == "preview_mismatch")
        assert finding.regions
        for x, y, width, height in finding.regions:
            assert x + width > 400 and x < 560 and y + height > 240 and y < 400

    def test_stripped_photo_gets_only_a_weak_finding(self, tmp_path):
        path = tmp_path / "stripped.jpg"
        _scene().save(path)
        report = analyze_file(path)
        assert [f.check for f in report.findings] == ["metadata_removed"]
        assert report.score < 0.2
        assert "Not enough" in report.conclusion

    def test_png_runs_without_warnings(self, tmp_path):
        path = tmp_path / "picture.png"
        _scene().save(path)
        assert analyze_file(path).warnings == []


@pytest.fixture
def cut_video(tmp_path):
    """A 90-frame, 30 fps video with every tenth frame removed and the timing kept."""
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")

    path = tmp_path / "cut.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        stream.codec_context.time_base = Fraction(1, 30)
        for i in range(90):
            if i % 10 == 9:
                continue
            frame = av.VideoFrame.from_ndarray(np.full((120, 160, 3), i * 2, np.uint8), format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, 30)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return path


@requires_exiftool
class TestVideoScenarios:
    def test_cut_video_with_mismatched_dates(self, cut_video):
        _exiftool(cut_video, "-QuickTime:CreateDate=2024:05:01 10:00:00",
                  "-QuickTime:TrackCreateDate=2023:01:01 10:00:00")
        report = analyze_file(cut_video)

        assert report.warnings == []
        assert report.results["creation_dates"] == "flagged"
        assert report.results["frame_count"] == "flagged"
        assert report.results["steady_frame_rate"] == "flagged"
        timing = next(f for f in report.findings if f.check == "frame_timing")
        assert timing.frames == [9 * n for n in range(1, 9)]

    def test_steady_video_has_no_timing_findings(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")

        path = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
        for i in range(60):
            writer.write(np.full((120, 160, 3), i * 4, np.uint8))
        writer.release()

        report = analyze_file(path)
        for name in ("frame_count", "steady_frame_rate", "frame_timing"):
            assert report.results[name] == "not_flagged"


@requires_exiftool
class TestUploads:
    def test_upload_ignores_the_new_file_date(self, camera_photo):
        report = analyze_file(camera_photo, uploaded=True)
        assert report.results["file_date_before_capture"] == "not_applicable"

    def test_date_from_the_browser_is_used(self, camera_photo):
        before_capture = datetime(2020, 1, 1, tzinfo=timezone.utc)
        report = analyze_file(camera_photo, uploaded=True, original_modified=before_capture)
        assert report.results["file_date_before_capture"] == "flagged"


@requires_exiftool
def test_batch_keeps_going_after_failures(camera_photo, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not media")

    reports, failures = analyze_batch([camera_photo, notes])

    assert [r.path for r in reports] == [str(camera_photo)]
    assert set(failures) == {str(notes)}
