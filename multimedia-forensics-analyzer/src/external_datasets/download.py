"""Downloading helpers shared by the public dataset sources.

`RemoteFile` lets Python's zipfile read an online zip by fetching only the byte ranges it
needs, so a few videos can be taken from a 9 GB archive without downloading all of it.
"""

import io
import re
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

USER_AGENT = "forensics-dataset-fetcher/1.0"
CHUNK = 1 << 20              # 1 MB

# fetch(url, start, end) -> (status, headers, body); end is inclusive, None = to the end.
Fetch = Callable[[str, int, int | None], tuple[int, dict, bytes]]


class RangeNotSupported(RuntimeError):
    """The server sent the whole file instead of the part asked for."""


def http_fetch(url: str, start: int = 0, end: int | None = None) -> tuple[int, dict, bytes]:
    """Fetch bytes start..end (inclusive) of `url` with an HTTP Range request."""
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Range": f"bytes={start}-{'' if end is None else end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise RangeNotSupported(f"{url} does not allow partial downloads")
        return response.status, dict(response.headers), response.read()


def fetch_text(url: str) -> str:
    """Download a small text file (such as a published file list) in one go."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


@dataclass
class PlannedFile:
    """One file a source will download: where from, where to, and how big."""

    source: str          # URL, or the member name inside a remote zip
    target: Path
    size: int


def remote_size(url: str, fetch: Fetch = http_fetch) -> int:
    """Size of a remote file in bytes, from the first byte's Content-Range header."""
    _, headers, _ = fetch(url, 0, 0)
    content_range = {k.lower(): v for k, v in headers.items()}.get("content-range", "")
    match = re.search(r"/(\d+)$", content_range)
    if not match:
        raise RangeNotSupported(f"{url} did not report its size")
    return int(match.group(1))


class RemoteFile(io.RawIOBase):
    """A read-only, seekable file whose bytes are fetched on demand with Range requests."""

    def __init__(self, url: str, fetch: Fetch = http_fetch, size: int | None = None):
        super().__init__()
        self.url, self._fetch = url, fetch
        self.size = size if size is not None else remote_size(url, fetch)
        self._position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self.size}[whence]
        self._position = max(0, base + offset)
        return self._position

    def readinto(self, buffer):
        if self._position >= self.size or len(buffer) == 0:
            return 0
        end = min(self._position + len(buffer), self.size) - 1
        _, _, body = self._fetch(self.url, self._position, end)
        buffer[:len(body)] = body
        self._position += len(body)
        return len(body)


def open_remote(url: str, fetch: Fetch = http_fetch) -> io.BufferedReader:
    """A buffered RemoteFile, ready to pass to zipfile.ZipFile."""
    return io.BufferedReader(RemoteFile(url, fetch), buffer_size=CHUNK)


def download(url: str, target: Path, fetch: Fetch = http_fetch) -> bool:
    """Download `url` to `target`, 1 MB at a time. Returns False if it was already there."""
    if target.is_file() and target.stat().st_size > 0:
        return False
    with open_remote(url, fetch) as source:
        return save(source, target, label=target.name)


def save(source, target: Path, label: str = "") -> bool:
    """Copy a readable stream to `target` through a .part file, so a cut-off copy is never kept."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with open(partial, "wb") as out:
        shutil.copyfileobj(source, out, CHUNK)
    partial.replace(target)
    if label:
        print(f"  saved {label} ({target.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    return True


def megabytes(size: int) -> str:
    return f"{size / 1e6:,.1f} MB"
