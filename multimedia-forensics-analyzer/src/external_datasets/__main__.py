"""Command line: python -m src.external_datasets {vision,csvted} --data ../data [--dry-run]"""

import argparse
import sys
import urllib.error
from pathlib import Path

from . import vision
from .csvted import CsvtedSource
from .download import PlannedFile, RangeNotSupported, megabytes

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def show(planned: list[PlannedFile], data_dir: Path) -> None:
    for item in planned:
        print(f"  {megabytes(item.size):>10}  {item.target.relative_to(data_dir).as_posix()}")
    print(f"  {megabytes(sum(item.size for item in planned)):>10}  total, {len(planned)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download small samples of public datasets.")
    parser.add_argument("source", choices=["vision", "csvted"])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR, help="the data folder")
    parser.add_argument("--photos", type=int, default=12, help="vision: number of photos")
    parser.add_argument("--videos", type=int, default=2, help="vision: number of videos")
    parser.add_argument("--scenes", type=int, default=5,
                        help="csvted: scenes, each = 1 original + 1 video per tampering type")
    parser.add_argument("--seed", type=int, default=9, help="same seed = same sample")
    parser.add_argument("--dry-run", action="store_true", help="list what would be downloaded, then stop")
    args = parser.parse_args()

    data_dir = args.data.resolve()
    try:
        if args.source == "vision":
            planned = vision.plan(data_dir, args.photos, args.videos, args.seed)
            show(planned, data_dir)
            if not args.dry_run:
                vision.fetch_vision(data_dir, planned)
                print("Done. Rebuild the dataset: python -m src.build_dataset --data ../data --overwrite")
        else:
            with CsvtedSource(data_dir, args.scenes, args.seed) as source:
                show(source.plan(), data_dir)
                if not args.dry_run:
                    print(f"wrote {source.fetch()}")
    except RangeNotSupported as exc:
        sys.exit(f"{exc}. Stopped rather than download whole archives.")
    except urllib.error.URLError as exc:
        sys.exit(f"Download failed: {exc}")


if __name__ == "__main__":
    main()
