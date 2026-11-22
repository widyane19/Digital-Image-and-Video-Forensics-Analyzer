"""CSVTED: videos with frames deleted, duplicated or inserted by another research team
(COMSATS / Cardiff, on Hugging Face).

A few scenes are taken, each with its original and one video of every tampering type. Only
those videos are read out of the multi-GB archives, using byte-range requests.
The result is its own labelled set: data/external/csvted/labels.csv, with paths relative
to that folder.
"""

import csv
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .attribution import record_source
from .download import Fetch, PlannedFile, http_fetch, open_remote, save

BASE_URL = "https://huggingface.co/datasets/CardiffVisualComputing/CSVTED/resolve/main/"
ORIGINALS_ARCHIVE = "Original.zip"
# Their tampering code -> (our manipulation type, archive). Same type names as build_dataset.py.
TYPES = {"Del": ("frame_removal", "Frame-Deletion.zip"),
         "Dup": ("frame_duplication", "Frame-Duplication.zip"),
         "Ins": ("frame_insertion", "Frame-Insertion.zip")}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
MIN_ORIGINAL_BYTES = 500_000          # a few entries are near-empty
LABEL_COLUMNS = ["file", "label", "manipulation_type", "frames"]
NAME = re.compile(r"^(\d+)_(Del|Dup|Ins)_(\d+)_(\d+)_(.+)$")

CITATION = """## CSVTED

- Source: <https://huggingface.co/datasets/CardiffVisualComputing/CSVTED>
- Paper: COMSATS Structured Video Tampering Evaluation Dataset (CSVTED), *Machine Vision and Applications*
  (post-print: <https://rosinpl.github.io/resources/papers/CSVTED-video-MVA-postprint.pdf>)
- Licence: not stated on the dataset page; check before publishing results.
- Used as a separate labelled set (`external/csvted/labels.csv`); the edits were made by the
  dataset's authors, not by us."""


@dataclass(frozen=True)
class Tampered:
    member: str          # path inside the archive
    archive: str
    manipulation_type: str
    first_frame: int
    last_frame: int
    scene: str
    size: int

    @property
    def frames(self) -> str:
        return f"{self.first_frame}-{self.last_frame}"


def parse_name(member: str) -> tuple[str, int, int, str] | None:
    """'Frame-Deletion/010_Del_021_030_Shop Banners.mp4' -> ('frame_removal', 21, 30, 'shop banners')."""
    match = NAME.match(PurePosixPath(member).stem)
    if not match:
        return None
    _, code, first, last, scene = match.groups()
    return TYPES[code][0], int(first), int(last), scene_key(scene)


def scene_key(name: str) -> str:
    """Scene names are written slightly differently across archives ('Laces_knot' / 'Laces knot')."""
    return re.sub(r"[\s_]+", " ", name).strip().lower()


def _videos(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return [i for i in archive.infolist()
            if not i.is_dir() and PurePosixPath(i.filename).suffix.lower() in VIDEO_EXTENSIONS]


def choose(originals: dict[str, zipfile.ZipInfo], tampered: list[Tampered], scenes: int,
           rng: random.Random) -> list[tuple[zipfile.ZipInfo, list[Tampered]]]:
    """Pick `scenes` scenes that have an original and every tampering type.

    For each scene the smallest video of each type is used; the scenes are drawn at random
    from the cheaper half to keep the download small.
    """
    kinds = {kind for kind, _ in TYPES.values()}
    smallest = {}
    for item in tampered:
        key = (item.scene, item.manipulation_type)
        if key not in smallest or (item.size, item.member) < (smallest[key].size, smallest[key].member):
            smallest[key] = item

    candidates = []
    for scene, original in originals.items():
        group = [smallest.get((scene, kind)) for kind in sorted(kinds)]
        if original.file_size >= MIN_ORIGINAL_BYTES and all(group):
            candidates.append((original.file_size + sum(g.size for g in group), scene, original, group))
    candidates.sort(key=lambda c: (c[0], c[1]))
    pool = candidates[:max(scenes, len(candidates) // 2)]
    return [(original, group) for _, _, original, group in rng.sample(pool, min(scenes, len(pool)))]


def _read_archives(fetch: Fetch):
    """Open the originals archive and every tampering archive.

    Only their tables of contents are read here; the caller closes the archives.
    """
    names = [ORIGINALS_ARCHIVE] + [archive for _, archive in TYPES.values()]
    archives = {name: zipfile.ZipFile(open_remote(BASE_URL + name, fetch))  # pylint: disable=consider-using-with
                for name in names}
    originals = {scene_key(PurePosixPath(i.filename).stem): i for i in _videos(archives[ORIGINALS_ARCHIVE])}
    tampered = []
    for _, archive_name in TYPES.values():
        for info in _videos(archives[archive_name]):
            parsed = parse_name(info.filename)
            if parsed:
                tampered.append(Tampered(info.filename, archive_name, *parsed, info.compress_size))
    return archives, originals, tampered


class CsvtedSource:
    """Plans and downloads the CSVTED sample into data/external/csvted/."""

    def __init__(self, data_dir: Path, scenes: int = 5, seed: int = 9, fetch: Fetch = http_fetch):
        self.folder = Path(data_dir) / "external" / "csvted"
        self.data_dir = Path(data_dir)
        self.archives, originals, tampered = _read_archives(fetch)
        self.chosen = choose(originals, tampered, scenes, random.Random(seed))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for archive in self.archives.values():
            archive.close()

    def _target(self, subfolder: str, member: str) -> Path:
        return self.folder / subfolder / PurePosixPath(member).name

    def plan(self) -> list[PlannedFile]:
        planned = []
        for original, group in self.chosen:
            planned.append(PlannedFile(original.filename, self._target("originals", original.filename),
                                       original.compress_size))
            planned += [PlannedFile(t.member, self._target(t.manipulation_type, t.member), t.size)
                        for t in group]
        return planned

    def fetch(self) -> Path:
        """Download the chosen videos, then write labels.csv and the SOURCES.md entry."""
        rows = []
        for original, group in self.chosen:
            target = self._target("originals", original.filename)
            self._extract(ORIGINALS_ARCHIVE, original.filename, target)
            rows.append([target.relative_to(self.folder).as_posix(), "original", "none", ""])
            for item in group:
                target = self._target(item.manipulation_type, item.member)
                self._extract(item.archive, item.member, target)
                rows.append([target.relative_to(self.folder).as_posix(), "manipulated",
                             item.manipulation_type, item.frames])

        labels = self.folder / "labels.csv"
        with open(labels, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(LABEL_COLUMNS)
            writer.writerows(rows)
        taken = "\n".join(f"  - `{row[0]}`" for row in rows)
        record_source(self.data_dir, "csvted", f"{CITATION}\n- Files taken:\n{taken}")
        return labels

    def _extract(self, archive: str, member: str, target: Path) -> None:
        if target.is_file() and target.stat().st_size > 0:
            return
        with self.archives[archive].open(member) as source:
            save(source, target, label=target.name)
