"""Tests for the web API in app.py and the combined pipeline in src/pipeline.py.

Skipped when FastAPI or ExifTool is not installed.
"""

import shutil
from io import BytesIO

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from app import app  # noqa: E402

requires_exiftool = pytest.mark.skipif(
    shutil.which("exiftool") is None, reason="ExifTool is not installed or not on PATH"
)

client = TestClient(app)


def _jpeg_bytes(size=(64, 48)) -> bytes:
    image = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            image.putpixel((x, y), ((x * 4) % 256, (y * 5) % 256, (x * y) % 256))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


@requires_exiftool
def test_analyze_image_returns_full_report():
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"last_modified": "1700000000000"},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["file_name"] == "photo.jpg"
    assert body["media_type"] == "image"
    metadata = body["metadata"]
    for key in ("score", "conclusion", "findings", "results", "warnings"):
        assert key in metadata
    assert 0 <= metadata["score"] <= 1
    assert metadata["path"] == "photo.jpg"  # the temporary path is not exposed

    image = body["image"]
    assert image["compression"]["visualization"].startswith("data:image/png;base64,")
    assert "candidate_regions" in image["copy_move"]
    assert "matches" not in image["copy_move"]


@requires_exiftool
def test_unsupported_file_is_rejected():
    response = client.post(
        "/api/analyze", files={"file": ("notes.txt", b"just some text", "text/plain")}
    )
    assert response.status_code == 415
    assert "tmp" not in response.json()["detail"].lower()


def test_empty_file_is_rejected():
    response = client.post("/api/analyze", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400
