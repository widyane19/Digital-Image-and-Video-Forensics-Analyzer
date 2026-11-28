"""Tests for src/build_dataset.py.

They create small originals, build a dataset from them and check each manipulation.
Skipped when ExifTool, PyAV, Pillow or NumPy is not available.

Run from the multimedia-forensics-analyzer folder:  python -m pytest
"""

import csv
import shutil
from collections import Counter

import pytest

from src.build_dataset import IMAGE_TYPES, VIDEO_TYPES, build_dataset

pytestmark = pytest.mark.skipif(shutil.which("exiftool") is None,
                                reason="ExifTool is not installed or not on PATH")


def make_originals(data, images=5, videos=1):
    """Camera-like photos (with metadata) and short videos in data/originals/."""
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from exiftool import ExifToolHelper  # pylint: disable=import-outside-toplevel

    (data / "originals" / "images").mkdir(parents=True)
    (data / "originals" / "videos").mkdir(parents=True)
    rng = np.random.default_rng(0)
    x = np.linspace(0, 255, 640)[None, :].repeat(480, axis=0)
    for i in range(images):
        texture = rng.integers(0, 60, size=(480, 640)) * 3
        picture = np.dstack([x, (x + i * 40) % 255, texture]).astype("uint8")
        Image.fromarray(picture).save(data / "originals" / "images" / f"photo{i}.jpg", quality=92)

    with ExifToolHelper() as et:
        for path in sorted((data / "originals" / "images").iterdir()):
            et.execute("-overwrite_original", "-Make=Canon", "-Model=EOS 80D",
                       "-Software=Firmware Version 1.0.2", "-DateTimeOriginal=2024:05:01 10:00:00",
                       "-ModifyDate=2024:05:01 10:00:00", "-ExifImageWidth=640", "-ExifImageHeight=480",
                       str(path))

    for v in range(videos):
        with av.open(str(data / "originals" / "videos" / f"clip{v}.mp4"), "w") as container:
            stream = container.add_stream("mpeg4", rate=30)
            stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
            for i in range(90):
                frame = av.VideoFrame.from_ndarray(np.full((240, 320, 3), (i * 3 + v * 80) % 255, np.uint8),
                                                   format="rgb24")
                container.mux(stream.encode(frame))
            container.mux(stream.encode(None))


def _labels(data):
    with open(data / "labels.csv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pixels(path):
    import numpy as np  # pylint: disable=import-outside-toplevel
    from PIL import Image  # pylint: disable=import-outside-toplevel
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=int)


def _frame_times(path):
    import av  # pylint: disable=import-outside-toplevel
    with av.open(str(path)) as container:
        return [float(frame.time) for frame in container.decode(container.streams.video[0])]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Built once and shared: the tests only read it (a rebuild with the same seed is identical)."""
    data = tmp_path_factory.mktemp("dataset") / "data"
    make_originals(data, images=5, videos=2)
    rows, warnings = build_dataset(data)
    return data, rows, warnings


def test_labels_list_every_original_and_copy(dataset):
    data, rows, warnings = dataset
    assert warnings == []
    labels = _labels(data)
    assert list(labels[0]) == ["file", "label", "manipulation_type"]
    assert [(r.file, r.label, r.manipulation_type) for r in rows] == [
        (r["file"], r["label"], r["manipulation_type"]) for r in labels]

    counts = Counter(r["manipulation_type"] for r in labels)
    assert counts["none"] == 7                                  # 5 photos + 2 videos
    assert all(counts[kind] == 1 for kind in IMAGE_TYPES)       # one copy per photo, types in turn
    assert all(counts[kind] == 2 for kind in VIDEO_TYPES)       # every video type for each video
    assert all((data / r["file"]).is_file() for r in labels)


def test_image_types_are_balanced(tmp_path):
    data = tmp_path / "data"
    make_originals(data, images=10, videos=0)
    rows, _ = build_dataset(data)
    counts = Counter(r.manipulation_type for r in rows if r.label == "manipulated")
    assert set(counts.values()) == {2}


def test_metadata_edit_changes_only_the_metadata(dataset):
    from exiftool import ExifToolHelper  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel

    data, _, _ = dataset
    original = data / "originals" / "images" / "photo4.jpg"
    edited = data / "manipulated" / "metadata_edit" / "photo4_metadata_edit.jpg"
    assert np.array_equal(_pixels(original), _pixels(edited))
    with ExifToolHelper() as et:
        tags = et.get_tags(str(edited), ["Software", "ModifyDate", "Make"])[0]
    assert tags["EXIF:Software"].startswith("Adobe Photoshop")
    assert tags["EXIF:ModifyDate"] == "2024:05:31 10:00:00"
    assert tags["EXIF:Make"] == "Canon"


def test_crop_keeps_the_old_metadata(dataset):
    from exiftool import ExifToolHelper  # pylint: disable=import-outside-toplevel

    data, _, _ = dataset
    cropped = data / "manipulated" / "crop" / "photo0_crop.jpg"
    with ExifToolHelper() as et:
        tags = et.get_tags(str(cropped), ["ExifImageWidth", "ImageWidth", "Make"])[0]
    assert tags["EXIF:Make"] == "Canon"
    assert tags["EXIF:ExifImageWidth"] == 640
    assert tags["File:ImageWidth"] < 640


def test_copy_move_changes_one_patch_sized_area(dataset):
    import numpy as np  # pylint: disable=import-outside-toplevel

    data, _, _ = dataset
    difference = np.abs(_pixels(data / "originals" / "images" / "photo1.jpg")
                        - _pixels(data / "manipulated" / "copy_move" / "photo1_copy_move.jpg")).max(axis=2)
    rows, cols = np.nonzero(difference > 60)        # ignore small JPEG re-save differences
    patch = min(640, 480) // 5
    assert rows.size
    assert rows.max() - rows.min() <= patch + 16 and cols.max() - cols.min() <= patch + 16


def test_small_copy_move_is_extra_and_half_the_size(dataset):
    import numpy as np  # pylint: disable=import-outside-toplevel

    data, rows, _ = dataset
    small = [r for r in rows if r.manipulation_type == "copy_move_small"]
    assert len(small) == 1                          # as many as each rotation type (5 photos / 5 types)
    edited = data / small[0].file
    original = data / "originals" / "images" / edited.name.replace("_copy_move_small", "")
    difference = np.abs(_pixels(original) - _pixels(edited)).max(axis=2)
    rows_changed, cols_changed = np.nonzero(difference > 60)
    patch = min(640, 480) // 10
    assert rows_changed.size
    assert rows_changed.max() - rows_changed.min() <= patch + 16
    assert cols_changed.max() - cols_changed.min() <= patch + 16


def test_frame_removal_leaves_a_gap(dataset):
    import numpy as np  # pylint: disable=import-outside-toplevel

    data, _, _ = dataset
    original = _frame_times(data / "originals" / "videos" / "clip0.mp4")
    for kind, expected in (("frame_removal", -15), ("frame_duplication", 15), ("frame_insertion", 15)):
        times = _frame_times(data / "manipulated" / kind / f"clip0_{kind}.mp4")
        assert len(times) - len(original) == expected
        largest_gap = max(np.diff(times))
        if kind == "frame_removal":
            assert largest_gap > 0.4                      # half a second missing
        else:
            assert largest_gap < 0.05                     # continuous timing


def test_same_seed_gives_the_same_dataset(tmp_path, dataset):
    data, _, _ = dataset
    copy = tmp_path / "copy"
    shutil.copytree(data / "originals", copy / "originals")
    build_dataset(copy)
    assert _labels(copy) == _labels(data)
    for row in _labels(data):
        if row["file"].endswith(".jpg"):
            assert (copy / row["file"]).read_bytes() == (data / row["file"]).read_bytes()


def test_refuses_to_overwrite_without_permission(dataset):
    data, _, _ = dataset
    with pytest.raises(FileExistsError, match="overwrite"):
        build_dataset(data)
    rows, _ = build_dataset(data, overwrite=True)
    assert rows


def test_no_originals(tmp_path):
    with pytest.raises(FileNotFoundError, match="originals"):
        build_dataset(tmp_path)


def test_one_image_cannot_be_spliced(tmp_path):
    data = tmp_path / "data"
    make_originals(data, images=1, videos=0)
    rows, warnings = build_dataset(data)
    assert [r.manipulation_type for r in rows if r.label == "manipulated"] == ["crop"]
    assert any("splice" in w for w in warnings)
