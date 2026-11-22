"""Keeps data/SOURCES.md up to date: one section per public dataset, with its citation,
licence and the files taken (the licences require this attribution)."""

import re
from pathlib import Path

HEADER = ("# Sources of the public media\n\n"
          "Written by `python -m src.external_datasets`. Our own photos and videos are not listed.\n")


def record_source(data_dir: Path, key: str, section: str) -> Path:
    """Add or replace the section for `key` in data/SOURCES.md."""
    path = Path(data_dir) / "SOURCES.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else HEADER
    start, end = f"<!-- {key} -->", f"<!-- /{key} -->"
    block = f"{start}\n{section.strip()}\n{end}"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    text = pattern.sub(lambda _: block, text) if pattern.search(text) else text.rstrip() + "\n\n" + block
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path
