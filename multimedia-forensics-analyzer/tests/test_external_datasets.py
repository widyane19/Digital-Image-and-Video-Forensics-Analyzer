"""Tests for src/external_datasets (the VISION and CSVTED downloaders).

No internet is used: downloads go through a fake `fetch` that serves bytes from memory.

Run from the multimedia-forensics-analyzer folder:  python -m pytest
"""

import io
import random
import zipfile

import pytest

from src.external_datasets import vision
from src.external_datasets.attribution import record_source
from src.external_datasets.csvted import Tampered, choose, parse_name, scene_key
from src.external_datasets.download import RangeNotSupported, RemoteFile, download, open_remote, remote_size


def fake_server(files: dict[str, bytes], ranges=True):
    """A fetch(url, start, end) that serves `files` the way an HTTP server with Range support would."""
    calls = []

    def fetch(url, start, end):
        data = files[url]
        calls.append((start, end))
        if not ranges:
            raise RangeNotSupported(f"{url} does not allow partial downloads")
        end = len(data) - 1 if end is None else min(end, len(data) - 1)
        return 206, {"Content-Range": f"bytes {start}-{end}/{len(data)}"}, data[start:end + 1]

    fetch.calls = calls
    return fetch


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

class TestDownload:
    def test_remote_size(self):
        assert remote_size("u", fake_server({"u": b"x" * 1234})) == 1234

    def test_remote_file_reads_and_seeks(self):
        remote = RemoteFile("u", fake_server({"u": bytes(range(100))}))
        remote.seek(90)
        assert remote.read(5) == bytes([90, 91, 92, 93, 94])
        remote.seek(-2, io.SEEK_END)
        assert remote.read(10) == bytes([98, 99])

    def test_one_file_is_read_out_of_a_remote_zip_without_fetching_the_rest(self):
        big = random.Random(0).randbytes(3_000_000)          # does not compress
        server = fake_server({"z": _zip({"big.bin": big, "small.txt": b"hello"})})
        with zipfile.ZipFile(open_remote("z", server)) as archive:
            assert archive.read("small.txt") == b"hello"
        assert sum(end - start + 1 for start, end in server.calls) < 1_500_000

    def test_server_without_ranges_is_refused(self):
        with pytest.raises(RangeNotSupported):
            open_remote("u", fake_server({"u": b"data"}, ranges=False))

    def test_download_saves_once(self, tmp_path):
        server = fake_server({"u": b"content"})
        target = tmp_path / "sub" / "file.bin"
        assert download("u", target, server) is True
        assert target.read_bytes() == b"content"
        assert download("u", target, server) is False             # already there
        assert not list(tmp_path.rglob("*.part"))


def test_sources_file_keeps_one_section_per_dataset(tmp_path):
    record_source(tmp_path, "vision", "## VISION\nfirst")
    record_source(tmp_path, "csvted", "## CSVTED")
    record_source(tmp_path, "vision", "## VISION\nsecond")
    text = (tmp_path / "SOURCES.md").read_text(encoding="utf-8")
    assert text.count("## VISION") == 1 and "second" in text and "first" not in text
    assert "## CSVTED" in text


# ---------------------------------------------------------------------------
# VISION
# ---------------------------------------------------------------------------

BASE = "https://lesc.dinfo.unifi.it/VISION/dataset/"
DEVICES = ["D01_Samsung_GalaxyS3Mini", "D02_Apple_iPhone4s", "D03_Huawei_P9",
           "D04_LG_D290", "D05_Apple_iPhone5c"]


def _vision_list():
    lines = []
    for device in DEVICES:
        code = device[:3]
        for folder in ("nat", "flat", "natWA", "natFBH"):
            lines += [f"{BASE}{device}/images/{folder}/{code}_I_{folder}_{i:04}.jpg" for i in range(1, 4)]
        for folder in ("flat", "indoor", "outdoor", "flatWA", "indoorYT"):
            lines.append(f"{BASE}{device}/videos/{folder}/{code}_V_{folder}_move_0001.mp4")
    lines.append(f"{BASE}D03_Huawei_P9/videos/flat/D03_V_flat_still_0002.mp4")       # errata
    return "\n".join(lines)


class TestVision:
    files = vision.parse_file_list(_vision_list())

    def test_parses_every_line(self):
        assert len(self.files) == len(DEVICES) * (12 + 5) + 1
        first = self.files[0]
        assert (first.device, first.media, first.folder, first.brand) == (
            "D01_Samsung_GalaxyS3Mini", "images", "nat", "Samsung")

    def test_only_native_files_are_used(self):
        assert {f.folder for f in vision.native_photos(self.files)} == {"nat"}
        videos = vision.native_videos(self.files)
        assert {f.folder for f in videos} == {"flat", "indoor", "outdoor"}
        assert "D03_V_flat_still_0002.mp4" not in {f.name for f in videos}

    def test_photos_are_spread_across_devices_and_brands(self):
        picked = vision.pick_photos(self.files, 4, random.Random(1))
        assert len({f.device for f in picked}) == 4
        assert len({f.brand for f in picked}) == 4                   # 4 brands among 5 devices

    def test_same_seed_same_sample(self):
        first = vision.pick_photos(self.files, 7, random.Random(3))
        assert first == vision.pick_photos(self.files, 7, random.Random(3))
        assert len(first) == 7 and len(set(first)) == 7              # devices repeat, files don't

    def test_smallest_video_of_each_device(self):
        sizes = {f.url: len(f.name) + (0 if f.folder == "indoor" else 100) for f in self.files}
        picked = vision.pick_videos(self.files, 2, random.Random(0), sizes.get)
        assert [f.folder for f, _ in picked] == ["indoor", "indoor"]
        assert len({f.device for f, _ in picked}) == 2

    def test_plan_and_fetch(self, tmp_path):
        urls = {f.url: f.name.encode() for f in self.files}
        server = fake_server(urls)
        planned = vision.plan(tmp_path, 2, 1, seed=9, fetch=server, list_text=_vision_list())
        assert [p.target.parent.name for p in planned] == ["images", "images", "videos"]
        assert all(p.target.name.startswith("vision_") for p in planned)

        vision.fetch_vision(tmp_path, planned, server)
        assert all(p.target.is_file() for p in planned)
        assert "CC BY-SA 4.0" in (tmp_path / "SOURCES.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CSVTED
# ---------------------------------------------------------------------------

class TestCsvted:
    @pytest.mark.parametrize("member, expected", [
        ("Frame-Deletion/010_Del_021_030_Shop Banners.mp4", ("frame_removal", 21, 30, "shop banners")),
        ("Frame-Duplication/010_Dup_061_070_CCTV Customer Footage(E).mp4",
         ("frame_duplication", 61, 70, "cctv customer footage(e)")),
        ("Frame-Insertion/110_Ins_151_260_Laces_knot.avi", ("frame_insertion", 151, 260, "laces knot")),
    ])
    def test_name_gives_type_and_frames(self, member, expected):
        assert parse_name(member) == expected

    def test_other_names_are_ignored(self):
        assert parse_name("Frame-Deletion/readme.txt") is None

    def test_scene_names_match_across_archives(self):
        assert scene_key("Laces_knot") == scene_key("Laces knot ") == "laces knot"

    def test_choose_takes_whole_scenes_and_the_smallest_variant(self):
        originals = {scene: zipfile.ZipInfo(f"Original/{scene}.mp4") for scene in ("a", "b", "c")}
        for info in originals.values():
            info.file_size = 1_000_000
        tampered = [Tampered(f"{kind}/{scene}{size}", "x.zip", kind, 1, 10, scene, size)
                    for scene in ("a", "b") for kind in ("frame_removal", "frame_duplication")
                    for size in (5, 9)]
        tampered += [Tampered(f"ins/{scene}", "x.zip", "frame_insertion", 1, 10, scene, 7)
                     for scene in ("a", "c")]        # only "a" has all three types

        chosen = choose(originals, tampered, 5, random.Random(0))
        assert len(chosen) == 1
        original, group = chosen[0]
        assert original.filename == "Original/a.mp4"
        assert sorted(t.manipulation_type for t in group) == [
            "frame_duplication", "frame_insertion", "frame_removal"]
        assert all(t.size in (5, 7) for t in group)
