"""Tests for Step 4 of the metadata analysis (src/metadata/metadata_checks.py).

These use hand-built NormalizedMetadata and need nothing installed, apart from the
few that are marked as needing Pillow. Real files are tested in test_metadata_analysis.py.

Run from the multimedia-forensics-analyzer folder:  python -m pytest
"""

from dataclasses import fields as dataclass_fields
from datetime import datetime, timedelta, timezone

import pytest

from src.metadata.metadata_checks import (
    CHECKS,
    NOT_APPLICABLE,
    CheckContext,
    check,
    is_later,
    run_checks,
)
from src.metadata.metadata_extraction import NormalizedMetadata

UTC = timezone.utc
NOW = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
CAPTURED = datetime(2024, 5, 1, 10, 0, tzinfo=UTC)


def _meta(media_type="image", **fields) -> NormalizedMetadata:
    values = dict.fromkeys(f.name for f in dataclass_fields(NormalizedMetadata))
    values.update(path="test", media_type=media_type, software=[], tags={})
    values.update(fields)
    return NormalizedMetadata(**values)


def _run(name, meta, file_date=None, frame_times=None):
    ctx = CheckContext(meta, file_date=file_date, now=NOW)
    if frame_times is not None:
        ctx.frame_times = frame_times
    return CHECKS[name].run(ctx)


def _status(*args, **kwargs):
    return _run(*args, **kwargs).status


def _steady_times(frames, fps=30):
    return [i * 1000 / fps for i in range(frames)]


# ---------------------------------------------------------------------------
# Date helper
# ---------------------------------------------------------------------------

class TestIsLater:
    def test_same_kind_of_date(self):
        assert is_later(CAPTURED + timedelta(seconds=10), CAPTURED)
        assert not is_later(CAPTURED, CAPTURED + timedelta(seconds=10))

    def test_margin(self):
        assert not is_later(CAPTURED + timedelta(seconds=10), CAPTURED, margin=60)

    def test_time_zones_are_taken_into_account(self):
        later_clock_reading = datetime(2024, 5, 1, 11, 0, tzinfo=timezone(timedelta(hours=2)))
        assert not is_later(later_clock_reading, CAPTURED)  # 09:00 UTC, so actually earlier

    def test_mixed_dates_do_not_crash_and_allow_for_the_unknown_zone(self):
        naive = CAPTURED.replace(tzinfo=None)
        assert not is_later(naive + timedelta(hours=10), CAPTURED)
        assert is_later(naive + timedelta(days=2), CAPTURED)


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

class TestRunChecks:
    @pytest.fixture
    def broken_check(self):
        @check("broken")
        def broken(ctx):
            raise RuntimeError("boom")
        yield
        del CHECKS["broken"]

    def test_every_check_gets_a_result(self):
        results, _ = run_checks(CheckContext(_meta(), now=NOW))
        assert set(results) == set(CHECKS)
        assert len(results) == 15

    def test_checks_for_other_media_are_not_applicable(self):
        results, _ = run_checks(CheckContext(_meta("image"), now=NOW))
        assert results["frame_timing"] is NOT_APPLICABLE

    def test_finding_is_labelled_with_its_check(self):
        results, _ = run_checks(CheckContext(_meta(), now=NOW))
        assert results["metadata_removed"].finding.check == "metadata_removed"

    @pytest.mark.usefixtures("broken_check")
    def test_crashing_check_becomes_a_warning(self):
        results, warnings = run_checks(CheckContext(_meta(), now=NOW))
        assert results["broken"].status == "not_applicable"
        assert warnings == ["broken could not run: boom"]
        assert results["metadata_removed"].status == "flagged"  # the others still ran


# ---------------------------------------------------------------------------
# Checks for both photos and videos
# ---------------------------------------------------------------------------

class TestEditingSoftware:
    @pytest.mark.parametrize("name", ["Adobe Photoshop 25.0", "GIMP 2.10", "CapCut", "Adobe Premiere Pro 2021.0"])
    def test_editors_are_high(self, name):
        result = _run("editing_software", _meta(software=[name]))
        assert result.finding.severity == "high"

    @pytest.mark.parametrize("name", ["Lavf58.76.100", "HandBrake 1.6.1"])
    def test_re_encoders_are_medium(self, name):
        assert _run("editing_software", _meta(software=[name])).finding.severity == "medium"

    def test_editor_outranks_re_encoder(self):
        result = _run("editing_software", _meta(software=["Lavf58", "Adobe Lightroom"]))
        assert result.finding.severity == "high"

    def test_camera_firmware_is_not_flagged(self):
        assert _status("editing_software", _meta(software=["17.4.1"])) == "not_flagged"

    def test_no_software_is_not_applicable(self):
        assert _status("editing_software", _meta()) == "not_applicable"


class TestModifiedAfterCapture:
    def test_flagged(self):
        meta = _meta(created=CAPTURED, modified=CAPTURED + timedelta(days=3))
        assert _status("modified_after_capture", meta) == "flagged"

    def test_small_gap_is_allowed(self):
        meta = _meta(created=CAPTURED, modified=CAPTURED + timedelta(seconds=30))
        assert _status("modified_after_capture", meta) == "not_flagged"

    def test_allowed_gap_includes_video_length(self):
        modified = CAPTURED + timedelta(minutes=10)
        assert _status("modified_after_capture", _meta("video", created=CAPTURED, modified=modified,
                                                       duration=600.0)) == "not_flagged"
        assert _status("modified_after_capture", _meta("video", created=CAPTURED, modified=modified,
                                                       duration=5.0)) == "flagged"

    def test_missing_date_is_not_applicable(self):
        assert _status("modified_after_capture", _meta(created=CAPTURED)) == "not_applicable"


class TestImpossibleDates:
    def test_future_date(self):
        assert _status("impossible_dates", _meta(created=NOW + timedelta(days=30))) == "flagged"

    def test_modified_before_capture(self):
        meta = _meta(created=CAPTURED, modified=CAPTURED - timedelta(days=1))
        assert "before" in _run("impossible_dates", meta).finding.message

    def test_normal_dates(self):
        meta = _meta(created=CAPTURED, modified=CAPTURED + timedelta(days=1))
        assert _status("impossible_dates", meta) == "not_flagged"

    def test_naive_date_does_not_crash(self):
        assert _status("impossible_dates", _meta(created=CAPTURED.replace(tzinfo=None))) == "not_flagged"

    def test_no_dates_is_not_applicable(self):
        assert _status("impossible_dates", _meta()) == "not_applicable"


class TestFileDateBeforeCapture:
    def test_flagged(self):
        meta = _meta(created=CAPTURED)
        assert _status("file_date_before_capture", meta, file_date=CAPTURED - timedelta(days=5)) == "flagged"

    def test_saved_after_capture(self):
        meta = _meta(created=CAPTURED)
        assert _status("file_date_before_capture", meta, file_date=CAPTURED + timedelta(days=5)) == "not_flagged"

    def test_mixed_dates_within_a_time_zone_are_allowed(self):
        meta = _meta(created=CAPTURED.replace(tzinfo=None))
        assert _status("file_date_before_capture", meta, file_date=CAPTURED - timedelta(hours=5)) == "not_flagged"

    def test_unknown_file_date_is_not_applicable(self):
        assert _status("file_date_before_capture", _meta(created=CAPTURED)) == "not_applicable"


class TestSizeMismatch:
    def _meta(self, declared, actual):
        return _meta(declared_width=declared[0], declared_height=declared[1],
                     actual_width=actual[0], actual_height=actual[1])

    def test_flagged(self):
        assert _status("size_mismatch", self._meta((4000, 3000), (2000, 1500))) == "flagged"

    def test_same_size(self):
        assert _status("size_mismatch", self._meta((4000, 3000), (4000, 3000))) == "not_flagged"

    def test_rotated_file_is_allowed(self):
        assert _status("size_mismatch", self._meta((4000, 3000), (3000, 4000))) == "not_flagged"

    def test_missing_size_is_not_applicable(self):
        assert _status("size_mismatch", _meta(actual_width=100, actual_height=100)) == "not_applicable"


class TestMetadataRemoved:
    def test_flagged_as_weak(self):
        assert _run("metadata_removed", _meta()).finding.severity == "low"

    @pytest.mark.parametrize("field,value", [("make", "Canon"), ("created", CAPTURED), ("software", ["x"])])
    def test_any_information_is_enough(self, field, value):
        assert _status("metadata_removed", _meta(**{field: value})) == "not_flagged"


# ---------------------------------------------------------------------------
# Photo-only checks
# ---------------------------------------------------------------------------

class TestEditHistory:
    @pytest.mark.parametrize("tag", ["XMP-xmpMM:HistoryAction", "XMP-crs:RawFileName",
                                     "XMP-photoshop:DocumentAncestors"])
    def test_flagged(self, tag):
        assert _status("edit_history", _meta(tags={tag: "x"})) == "flagged"

    def test_other_xmp_is_fine(self):
        assert _status("edit_history", _meta(tags={"XMP-dc:Title": "x"})) == "not_flagged"


class TestGpsTime:
    def test_clocks_agree(self):
        assert _status("gps_time", _meta(created=CAPTURED, gps_time=CAPTURED)) == "not_flagged"

    def test_clocks_disagree(self):
        meta = _meta(created=CAPTURED, gps_time=CAPTURED + timedelta(hours=3))
        assert _status("gps_time", meta) == "flagged"

    def test_naive_clock_in_a_time_zone_is_allowed(self):
        meta = _meta(created=CAPTURED.replace(tzinfo=None) + timedelta(hours=5, minutes=30), gps_time=CAPTURED)
        assert _status("gps_time", meta) == "not_flagged"

    def test_naive_clock_off_by_an_odd_amount(self):
        meta = _meta(created=CAPTURED.replace(tzinfo=None) + timedelta(hours=1, minutes=7), gps_time=CAPTURED)
        assert _status("gps_time", meta) == "flagged"

    def test_naive_clock_far_outside_any_time_zone(self):
        meta = _meta(created=CAPTURED.replace(tzinfo=None) + timedelta(days=3), gps_time=CAPTURED)
        assert _status("gps_time", meta) == "flagged"

    def test_no_gps_is_not_applicable(self):
        assert _status("gps_time", _meta(created=CAPTURED)) == "not_applicable"


class TestCompressionSettings:
    def _jpeg(self, tmp_path, **save_options):
        Image = pytest.importorskip("PIL.Image")
        path = tmp_path / "photo.jpg"
        Image.new("RGB", (64, 64), (90, 120, 30)).save(path, **save_options)
        return str(path)

    def _meta(self, path, make="Canon"):
        return _meta(path=path, make=make, tags={"File:MIMEType": "image/jpeg"})

    def test_generic_tables_are_weak_evidence(self, tmp_path):
        result = _run("compression_settings", self._meta(self._jpeg(tmp_path, quality=85)))
        assert result.finding.severity == "low"

    def test_custom_tables_are_fine(self, tmp_path):
        custom = [[7 + i % 5 for i in range(64)], [11 + i % 3 for i in range(64)]]
        path = self._jpeg(tmp_path, qtables=custom)
        assert _status("compression_settings", self._meta(path)) == "not_flagged"

    def test_file_that_does_not_claim_a_camera_is_not_applicable(self):
        assert _status("compression_settings", _meta(tags={"File:MIMEType": "image/jpeg"})) == "not_applicable"

    def test_png_is_not_applicable(self):
        meta = _meta(make="Canon", tags={"File:MIMEType": "image/png"})
        assert _status("compression_settings", meta) == "not_applicable"


# ---------------------------------------------------------------------------
# Video-only checks
# ---------------------------------------------------------------------------

class TestCreationDates:
    def test_dates_agree(self):
        tags = {"QuickTime:CreateDate": "2024:05:01 10:00:00+00:00",
                "Track1:TrackCreateDate": "2024:05:01 10:00:00+00:00",
                "Track1:MediaCreateDate": "2024:05:01 10:00:30+00:00"}
        assert _status("creation_dates", _meta("video", tags=tags)) == "not_flagged"

    def test_dates_disagree(self):
        tags = {"QuickTime:CreateDate": "2024:05:01 10:00:00+00:00",
                "Track1:TrackCreateDate": "2023:01:01 10:00:00+00:00"}
        result = _run("creation_dates", _meta("video", tags=tags))
        assert list(result.finding.evidence["tracks"]) == ["Track1:TrackCreateDate"]

    def test_placeholder_track_dates_are_ignored(self):
        tags = {"QuickTime:CreateDate": "2024:05:01 10:00:00+00:00",
                "Track1:TrackCreateDate": "0000:00:00 00:00:00"}
        assert _status("creation_dates", _meta("video", tags=tags)) == "not_applicable"


class TestFrameCount:
    def test_matches(self):
        meta = _meta("video", duration=3.0)
        assert _status("frame_count", meta, frame_times=_steady_times(90)) == "not_flagged"

    def test_every_tenth_frame_removed(self):
        times = [t for i, t in enumerate(_steady_times(90)) if i % 10 != 9]
        result = _run("frame_count", _meta("video", duration=3.0), frame_times=times)
        assert result.finding.evidence == {"frames": 81, "expected": 90}

    def test_prefers_the_video_track_length(self):
        meta = _meta("video", duration=9.0, video_duration=3.0)
        assert _status("frame_count", meta, frame_times=_steady_times(90)) == "not_flagged"

    def test_too_few_frames_is_not_applicable(self):
        assert _status("frame_count", _meta("video", duration=1.0), frame_times=[0.0]) == "not_applicable"


class TestSteadyFrameRate:
    def test_matches(self):
        meta = _meta("video", frame_rate=30.0)
        assert _status("steady_frame_rate", meta, frame_times=_steady_times(90)) == "not_flagged"

    def test_average_rate_hides_the_real_rate(self):
        times = [t for i, t in enumerate(_steady_times(90)) if i % 10 != 9]
        meta = _meta("video", frame_rate=27.0)  # what ExifTool reports for this cut video
        assert _status("steady_frame_rate", meta, frame_times=times) == "flagged"

    def test_no_rate_is_not_applicable(self):
        assert _status("steady_frame_rate", _meta("video"), frame_times=_steady_times(90)) == "not_applicable"


class TestAudioVideoLength:
    def test_flagged(self):
        meta = _meta("video", audio_duration=10.0, video_duration=8.0)
        assert _run("audio_video_length", meta).finding.severity == "low"

    def test_small_difference_is_allowed(self):
        meta = _meta("video", audio_duration=10.0, video_duration=9.9)
        assert _status("audio_video_length", meta) == "not_flagged"

    def test_no_audio_is_not_applicable(self):
        assert _status("audio_video_length", _meta("video", video_duration=8.0)) == "not_applicable"


class TestFrameTiming:
    def test_steady_video(self):
        assert _status("frame_timing", _meta("video"), frame_times=_steady_times(90)) == "not_flagged"

    def test_reports_the_frames_after_each_gap(self):
        times = [t for i, t in enumerate(_steady_times(30)) if i % 10 != 9]
        result = _run("frame_timing", _meta("video"), frame_times=times)
        assert result.finding.frames == [9, 18]

    def test_repeated_timestamp(self):
        times = _steady_times(10)
        times[5] = times[4]
        result = _run("frame_timing", _meta("video"), frame_times=times)
        assert result.finding.evidence["repeated_timestamps"] == [5]

    def test_too_few_frames_is_not_applicable(self):
        assert _status("frame_timing", _meta("video"), frame_times=[0.0, 33.3]) == "not_applicable"
