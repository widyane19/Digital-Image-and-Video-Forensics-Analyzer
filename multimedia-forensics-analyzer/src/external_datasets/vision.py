"""VISION: native photos and videos from 35 phones (University of Florence, CC BY-SA 4.0).

They are added to data/originals/ as extra unedited originals; build_dataset.py then makes
their manipulated copies, just as for our own photos and videos.
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .attribution import record_source
from .download import Fetch, PlannedFile, download, fetch_text, http_fetch, remote_size

LIST_URL = "https://lesc.dinfo.unifi.it/VISION/VISION_files.txt"
PREFIX = "vision_"
NATIVE_PHOTO_FOLDERS = {"nat"}                          # "flat" = plain walls and sky: nothing to copy-move
NATIVE_VIDEO_FOLDERS = {"flat", "indoor", "outdoor"}    # the WA / YT folders are re-encoded by WhatsApp / YouTube
ERRATA = {"D03_V_flat_still_0002.mp4", "D19_V_flat_move_0002.mov"}   # misplaced, so not native (README, 16/04/2019)

CITATION = """## VISION

- Source: <https://lesc.dinfo.unifi.it/en/datasets>
- Paper: D. Shullani, M. Fontani, M. Iuliani, O. Al Shaya and A. Piva, "VISION: a video and image dataset
  for source identification", *EURASIP Journal on Information Security*, 2017.
  doi:10.1186/s13635-017-0067-2
- Licence: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- Used as extra unedited originals (in `originals/`, names starting `vision_`); their manipulated copies
  are made by `build_dataset.py`."""


@dataclass(frozen=True)
class VisionFile:
    url: str
    device: str          # e.g. D01_Samsung_GalaxyS3Mini
    media: str           # "images" or "videos"
    folder: str          # e.g. nat, flat, indoor, natWA
    name: str

    @property
    def brand(self) -> str:
        return self.device.split("_")[1]


def parse_file_list(text: str) -> list[VisionFile]:
    """Turn VISION_files.txt (one URL per line) into VisionFile entries."""
    files = []
    for line in text.splitlines():
        parts = line.strip().split("/dataset/", 1)
        if len(parts) != 2:
            continue
        pieces = parts[1].split("/")
        if len(pieces) == 4:
            files.append(VisionFile(line.strip(), *pieces))
    return files


def native_photos(files: list[VisionFile]) -> list[VisionFile]:
    return [f for f in files if f.media == "images" and f.folder in NATIVE_PHOTO_FOLDERS]


def native_videos(files: list[VisionFile]) -> list[VisionFile]:
    return [f for f in files if f.media == "videos" and f.folder in NATIVE_VIDEO_FOLDERS
            and f.name not in ERRATA]


def devices_in_turn(files: list[VisionFile], rng: random.Random) -> list[str]:
    """Every device once, alternating brands, so a small sample covers as many brands as possible."""
    by_brand = defaultdict(set)
    for f in files:
        by_brand[f.brand].add(f.device)
    queues = {brand: rng.sample(sorted(devices), len(devices)) for brand, devices in by_brand.items()}
    brands = rng.sample(sorted(queues), len(queues))
    order = []
    while any(queues.values()):
        for brand in brands:
            if queues[brand]:
                order.append(queues[brand].pop())
    return order


def pick_photos(files: list[VisionFile], count: int, rng: random.Random) -> list[VisionFile]:
    """`count` native photos, one per device in turn (devices repeat only if count > devices)."""
    photos = native_photos(files)
    by_device = defaultdict(list)
    for f in photos:
        by_device[f.device].append(f)
    for group in by_device.values():
        rng.shuffle(group)
    devices = devices_in_turn(photos, rng)
    picked = []
    while len(picked) < count and any(by_device.values()):
        for device in devices:
            if by_device[device] and len(picked) < count:
                picked.append(by_device[device].pop())
    return picked


def pick_videos(files: list[VisionFile], count: int, rng: random.Random,
                size_of) -> list[tuple[VisionFile, int]]:
    """The smallest native video from each of `count` devices (small = quick to re-encode)."""
    videos = native_videos(files)
    picked = []
    for device in devices_in_turn(videos, rng)[:count]:
        sized = [(f, size_of(f.url)) for f in videos if f.device == device]
        picked.append(min(sized, key=lambda pair: (pair[1], pair[0].name)))
    return picked


def plan(data_dir: Path, photos: int, videos: int, seed: int, *,
         fetch: Fetch = http_fetch, list_text: str | None = None) -> list[PlannedFile]:
    """What `fetch_vision` would download: URL, target and size of each file."""
    files = parse_file_list(list_text if list_text is not None else fetch_text(LIST_URL))
    rng = random.Random(seed)
    originals = Path(data_dir) / "originals"

    planned = [PlannedFile(f.url, originals / "images" / f"{PREFIX}{f.name}", remote_size(f.url, fetch))
               for f in pick_photos(files, photos, rng)]
    planned += [PlannedFile(f.url, originals / "videos" / f"{PREFIX}{f.name}", size)
                for f, size in pick_videos(files, videos, rng, lambda url: remote_size(url, fetch))]
    return planned


def fetch_vision(data_dir: Path, planned: list[PlannedFile], fetch: Fetch = http_fetch) -> list[Path]:
    """Download the planned files into data/originals/ and record them in data/SOURCES.md."""
    for item in planned:
        download(item.source, item.target, fetch)
    taken = "\n".join(f"  - `{item.target.name}` from <{item.source}>" for item in planned)
    record_source(data_dir, "vision", f"{CITATION}\n- Files taken:\n{taken}")
    return [item.target for item in planned]
