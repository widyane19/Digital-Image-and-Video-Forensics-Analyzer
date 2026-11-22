"""Build the controlled dataset: intentionally modified copies of the team's originals.

Put unedited originals (straight from a camera or phone, not downloaded from social
media) in:
    data/originals/images/
    data/originals/videos/

This creates modified copies in data/manipulated/<type>/ and writes data/labels.csv
from scratch, listing every original and every modified copy with its type.

    Images: one modified copy each, with the types taken in turn so each type gets
            about the same number of files.
    Videos: one modified copy per video type.

Every modified file gets the original's metadata copied back with ExifTool, as a
simple editor that keeps the camera information would. The same originals and seed
always give the same dataset.

Run from the multimedia-forensics-analyzer folder:
    python -m src.build_dataset --data ../data
"""

from __future__ import annotations

import csv
import random
import shutil
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_TYPES = ("crop", "copy_move", "splice", "recompress", "metadata_edit")
VIDEO_TYPES = ("frame_removal", "frame_duplication", "frame_insertion")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

DEFAULT_SEED = 9
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

JPEG_QUALITY = 92            # quality for re-saving edited JPEGs
RECOMPRESS_QUALITY = 60      # quality for the recompress manipulation
EDIT_SECONDS = 0.5           # length of the removed / duplicated / inserted block of frames
EDITOR_SOFTWARE = "Adobe Photoshop 25.0 (Windows)"


@dataclass
class LabelRow:
    file: str                # path relative to the data folder, with forward slashes
    label: str               # "original" or "manipulated"
    manipulation_type: str   # "none" for originals


# ---------------------------------------------------------------------------
# Building the dataset
# ---------------------------------------------------------------------------

def build_dataset(data_dir: str | Path, seed: int = DEFAULT_SEED,
                  overwrite: bool = False) -> tuple[list[LabelRow], list[str]]:
    """Create the modified copies and labels.csv. Returns (label rows, warnings)."""
    from exiftool import ExifToolHelper  # pylint: disable=import-outside-toplevel

    data_dir = Path(data_dir)
    images = _media(data_dir / "originals" / "images", IMAGE_EXTENSIONS)
    videos = _media(data_dir / "originals" / "videos", VIDEO_EXTENSIONS)
    if not images and not videos:
        raise FileNotFoundError(
            f"No originals found. Put unedited photos in {data_dir / 'originals' / 'images'} "
            f"and videos in {data_dir / 'originals' / 'videos'}.")

    out_dir = data_dir / "manipulated"
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_dir} already has files. Pass overwrite=True "
                                  "(--overwrite) to replace them.")
        shutil.rmtree(out_dir)

    rng = random.Random(seed)
    rows = [LabelRow(_relative(p, data_dir), "original", "none") for p in images + videos]
    warnings: list[str] = []

    with ExifToolHelper() as et:
        image_types = [t for t in IMAGE_TYPES if t != "splice" or len(images) > 1]
        for index, source in enumerate(images):
            kind = image_types[index % len(image_types)]
            donor = images[(index + 1) % len(images)]
            try:
                target = _make_image(source, kind, out_dir, rng, donor, et)
            except (OSError, ValueError) as exc:
                warnings.append(f"{source.name}: could not make {kind} ({exc})")
                continue
            rows.append(LabelRow(_relative(target, data_dir), "manipulated", kind))

        video_types = [t for t in VIDEO_TYPES if t != "frame_insertion" or len(videos) > 1]
        for index, source in enumerate(videos):
            donor = videos[(index + 1) % len(videos)]
            for kind in video_types:
                try:
                    target = _make_video(source, kind, out_dir, rng, donor, et)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    # PyAV raises its own error types; one bad video must not stop the build.
                    warnings.append(f"{source.name}: could not make {kind} ({exc})")
                    continue
                rows.append(LabelRow(_relative(target, data_dir), "manipulated", kind))

    if len(images) == 1:
        warnings.append("Only one original image, so no splice copies were made (splicing needs two).")
    if len(videos) == 1:
        warnings.append("Only one original video, so no frame_insertion copy was made (it needs two).")

    write_labels(rows, data_dir / "labels.csv")
    return rows, warnings


def write_labels(rows: list[LabelRow], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "label", "manipulation_type"])
        writer.writerows([row.file, row.label, row.manipulation_type] for row in rows)


def _media(folder: Path, extensions: set[str]) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions)


def _relative(path: Path, data_dir: Path) -> str:
    return path.relative_to(data_dir).as_posix()


def _copy_metadata(source: Path, target: Path, et) -> None:
    """Copy all of the original's metadata, including its embedded preview, onto the copy."""
    et.execute("-overwrite_original", "-TagsFromFile", str(source), "-all:all",
               "-ThumbnailImage", str(target))


# ---------------------------------------------------------------------------
# Image manipulations
# ---------------------------------------------------------------------------

def _make_image(source: Path, kind: str, out_dir: Path, rng: random.Random, donor: Path, et) -> Path:
    folder = out_dir / kind
    folder.mkdir(parents=True, exist_ok=True)
    is_jpeg = source.suffix.lower() in (".jpg", ".jpeg")

    if kind == "metadata_edit":
        # Pixels untouched: only the metadata says the file went through an editor.
        target = folder / f"{source.stem}_{kind}{source.suffix}"
        shutil.copy2(source, target)
        et.execute("-overwrite_original", f"-Software={EDITOR_SOFTWARE}",
                   "-ModifyDate+=0:0:30 0:0:0", str(target))
        return target

    with Image.open(source) as opened:
        picture = opened.convert("RGB")
    if kind == "crop":
        picture = _crop(picture, rng)
    elif kind == "copy_move":
        picture = _copy_move(picture, rng)
    elif kind == "splice":
        with Image.open(donor) as other:
            picture = _splice(picture, other.convert("RGB"), rng)

    if kind == "recompress" or is_jpeg:
        target = folder / f"{source.stem}_{kind}.jpg"
        quality = RECOMPRESS_QUALITY if kind == "recompress" else JPEG_QUALITY
        picture.save(target, quality=quality)
    else:
        target = folder / f"{source.stem}_{kind}{source.suffix}"
        picture.save(target)
    _copy_metadata(source, target, et)
    return target


def _crop(picture: Image.Image, rng: random.Random) -> Image.Image:
    """Cut 10-25% off the width and height, at a random position."""
    width, height = picture.size
    keep = 1 - rng.uniform(0.10, 0.25)
    new_w, new_h = int(width * keep), int(height * keep)
    left, top = rng.randint(0, width - new_w), rng.randint(0, height - new_h)
    return picture.crop((left, top, left + new_w, top + new_h))


def _patch_size(picture: Image.Image) -> int:
    return max(8, min(picture.size) // 5)


def _most_textured_patch(picture: Image.Image, size: int) -> tuple[int, int]:
    """Top-left corner of the size x size block with the most detail (highest variance)."""
    gray = np.asarray(picture.convert("L"), dtype=np.float32)
    best, corner = -1.0, (0, 0)
    for top in range(0, gray.shape[0] - size + 1, size // 2):
        for left in range(0, gray.shape[1] - size + 1, size // 2):
            variance = float(gray[top:top + size, left:left + size].var())
            if variance > best:
                best, corner = variance, (left, top)
    return corner


def _free_spot(picture: Image.Image, size: int, avoid: tuple[int, int], rng: random.Random) -> tuple[int, int]:
    """A random top-left corner whose block doesn't overlap the block at `avoid`."""
    width, height = picture.size
    for _ in range(100):
        left, top = rng.randint(0, width - size), rng.randint(0, height - size)
        if abs(left - avoid[0]) >= size or abs(top - avoid[1]) >= size:
            return left, top
    raise ValueError("image is too small to move a patch without overlapping")


def _copy_move(picture: Image.Image, rng: random.Random) -> Image.Image:
    """Copy the most textured block to another spot in the same image."""
    size = _patch_size(picture)
    left, top = _most_textured_patch(picture, size)
    patch = picture.crop((left, top, left + size, top + size))
    picture = picture.copy()
    picture.paste(patch, _free_spot(picture, size, (left, top), rng))
    return picture


def _splice(picture: Image.Image, donor: Image.Image, rng: random.Random) -> Image.Image:
    """Paste a block taken from a different original."""
    size = min(_patch_size(picture), *donor.size)
    left, top = _most_textured_patch(donor, size)
    patch = donor.crop((left, top, left + size, top + size))
    picture = picture.copy()
    width, height = picture.size
    picture.paste(patch, (rng.randint(0, width - size), rng.randint(0, height - size)))
    return picture


# ---------------------------------------------------------------------------
# Video manipulations
# ---------------------------------------------------------------------------

def _make_video(source: Path, kind: str, out_dir: Path, rng: random.Random, donor: Path, et) -> Path:
    import av  # pylint: disable=import-outside-toplevel

    folder = out_dir / kind
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{source.stem}_{kind}.mp4"

    with av.open(str(source)) as reader:
        stream = reader.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or Fraction(30)
        total = sum(1 for packet in reader.demux(stream) if packet.size)
    block = max(1, round(float(rate) * EDIT_SECONDS))
    if total < block * 3:
        raise ValueError(f"video is too short ({total} frames)")
    start = rng.randint(total // 4, total // 2)

    with av.open(str(source)) as reader, av.open(str(target), "w") as writer:
        stream = reader.streams.video[0]
        out = _video_stream(writer, stream, rate)
        frames = reader.decode(stream)

        if kind == "frame_removal":
            # The block is dropped but the timestamps are kept, as a cut leaves them: a gap.
            for index, frame in enumerate(frames):
                if not start <= index < start + block:
                    _write(writer, out, frame, index, rate)
        else:
            extra = (_donor_frames(donor, block, out.width, out.height) if kind == "frame_insertion"
                     else None)
            held, index = [], 0
            for position, frame in enumerate(frames):
                _write(writer, out, frame, index, rate)
                index += 1
                if start <= position < start + block:
                    held.append(frame)
                if position == start + block - 1:
                    # Duplicate the block just written, or insert the donor's frames,
                    # with continuous timing.
                    for added in (held if kind == "frame_duplication" else extra):
                        _write(writer, out, added, index, rate)
                        index += 1
        for packet in out.encode(None):
            writer.mux(packet)

    _copy_metadata(source, target, et)
    return target


def _video_stream(writer, source_stream, rate: Fraction):
    import av  # pylint: disable=import-outside-toplevel
    codec = "libx264"
    try:
        av.codec.Codec(codec, "w")
    except Exception:  # pylint: disable=broad-exception-caught  # H.264 not built into this PyAV
        codec = "mpeg4"
    out = writer.add_stream(codec, rate=rate)
    out.width = source_stream.codec_context.width // 2 * 2
    out.height = source_stream.codec_context.height // 2 * 2
    out.pix_fmt = "yuv420p"
    out.codec_context.time_base = 1 / rate
    if codec == "libx264":
        out.options = {"crf": "18"}
    else:
        out.bit_rate = 8_000_000
    return out


def _write(writer, out, frame, index: int, rate: Fraction) -> None:
    """Encode one frame at position `index`, converting it only if its size or format differs."""
    if (frame.width, frame.height, frame.format.name) != (out.width, out.height, "yuv420p"):
        frame = frame.reformat(width=out.width, height=out.height, format="yuv420p")
    frame.pts, frame.time_base = index, 1 / rate
    for packet in out.encode(frame):
        writer.mux(packet)


def _donor_frames(donor: Path, count: int, width: int, height: int) -> list:
    """The first `count` frames of another original, resized to match."""
    import av  # pylint: disable=import-outside-toplevel
    frames = []
    with av.open(str(donor)) as reader:
        for frame in reader.decode(reader.streams.video[0]):
            frames.append(frame.reformat(width=width, height=height, format="yuv420p"))
            if len(frames) == count:
                break
    if not frames:
        raise ValueError(f"no frames in {donor.name}")
    return frames


if __name__ == "__main__":
    import argparse
    import sys
    from collections import Counter

    parser = argparse.ArgumentParser(description="Make modified copies of the originals and write labels.csv.")
    parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="the data folder (holds originals/)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="same seed = same dataset")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing data/manipulated/")
    args = parser.parse_args()

    try:
        built, problems = build_dataset(args.data, args.seed, args.overwrite)
    except (FileNotFoundError, FileExistsError) as exc:
        sys.exit(str(exc))

    counts = Counter(row.manipulation_type for row in built)
    print(f"{counts.pop('none', 0)} originals")
    for made_type, made in sorted(counts.items()):
        print(f"{made:4} {made_type}")
    for problem in problems:
        print(f"warning: {problem}")
    print(f"wrote {Path(args.data) / 'labels.csv'}")
