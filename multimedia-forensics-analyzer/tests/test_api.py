"""Tests for the web API (app.py), the four-category pipeline (src/pipeline.py)
and the AI summary (src/ai_summary.py).

The Claude API is never called: the summary tests use a fake client.
Tests that analyse real files are skipped when ExifTool is not installed.
"""

import shutil
from io import BytesIO
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
anthropic = pytest.importorskip("anthropic")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import src.ai_summary as ai_summary  # noqa: E402
from app import app  # noqa: E402
from src.pipeline import verdict  # noqa: E402

requires_exiftool = pytest.mark.skipif(
    shutil.which("exiftool") is None, reason="ExifTool is not installed or not on PATH"
)

client = TestClient(app)

CATEGORIES = {"metadata_analysis", "image_analysis", "compression_analysis", "video_analysis"}
CHECK_FIELDS = {"id", "name", "status", "severity", "result", "details", "visualization"}


def _jpeg_bytes(size=(64, 48)) -> bytes:
    image = Image.new("RGB", size)
    for x in range(size[0]):
        for y in range(size[1]):
            image.putpixel((x, y), ((x * 4) % 256, (y * 5) % 256, (x * y) % 256))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _report(flagged=True):
    check = {"id": "editing_software", "name": "Editing software", "status": "flagged",
             "severity": "high", "result": "The file was saved by editing software: GIMP.",
             "details": {"software": ["GIMP"]}, "visualization": None}
    clean = {**check, "status": "passed", "severity": None, "result": "Nothing suspicious found."}
    empty = {"status": "not_applicable", "summary": "n/a", "checks": []}
    return {
        "file_name": "photo.jpg", "media_type": "image",
        "verdict": {"label": "likely_altered" if flagged else "no_strong_evidence", "score": 0.6},
        "categories": {
            "metadata_analysis": {"status": "flagged", "summary": "1 flag",
                                  "checks": [check if flagged else clean]},
            "image_analysis": {"status": "clear", "summary": "clear", "checks": [
                {**clean, "id": "copy_move", "name": "Copy-move detection",
                 "visualization": "data:image/png;base64,AAAA"}]},
            "compression_analysis": empty,
            "video_analysis": empty,
        },
        "warnings": [],
    }


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _fake_client(**kwargs):
    messages = FakeMessages(**kwargs)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def _text_response(text, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, model="claude-opus-5-5",
                           content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=text)])


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

class TestVerdict:
    def test_levels_follow_the_metadata_score(self):
        assert verdict(0.72, 5, 0)["label"] == "likely_altered"
        assert verdict(0.3, 5, 0)["label"] == "some_signs"
        assert verdict(0.0, 5, 0)["label"] == "no_strong_evidence"

    def test_copy_move_candidates_raise_the_level(self):
        assert verdict(0.0, 5, 2)["label"] == "some_signs"
        assert verdict(0.3, 5, 1)["label"] == "likely_altered"
        assert verdict(0.9, 5, 1)["label"] == "likely_altered"

    def test_few_checks_and_no_evidence_is_inconclusive(self):
        assert verdict(0.0, 2, 0)["label"] == "inconclusive"
        assert verdict(0.6, 2, 0)["label"] == "likely_altered"


# ---------------------------------------------------------------------------
# AI summary (fake client, never the real API)
# ---------------------------------------------------------------------------

class TestSummary:
    def test_generated_summary(self):
        fake, messages = _fake_client(response=_text_response("This photo shows signs of editing."))
        result = ai_summary.summarize(_report(), client=fake)

        assert result["generated"] is True
        assert result["text"] == "This photo shows signs of editing."
        assert [r["check"] for r in result["reasons"]] == ["Editing software"]

        sent = messages.calls[0]
        assert sent["model"] == "claude-opus-5-5"
        assert sent["fallbacks"] == "default"
        prompt = sent["messages"][0]["content"]
        assert "<analysis_results>" in prompt and "GIMP" in prompt
        assert "base64" not in prompt  # images are never sent

    def test_connection_error_falls_back_to_template(self):
        error = anthropic.APIConnectionError(request=SimpleNamespace(method="POST", url="x"))
        fake, _ = _fake_client(error=error)
        result = ai_summary.summarize(_report(), client=fake)

        assert result["generated"] is False
        assert "likely altered" in result["text"]
        assert "Editing software" in result["text"]
        assert result["note"]

    def test_refusal_falls_back_to_template(self):
        fake, _ = _fake_client(response=_text_response("", stop_reason="refusal"))
        assert ai_summary.summarize(_report(), client=fake)["generated"] is False

    def test_template_for_a_clean_file(self):
        text = ai_summary.template_summary(_report(flagged=False))
        assert "None of the checks found signs of editing" in text


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


@requires_exiftool
def test_analyze_image_returns_four_categories():
    response = client.post(
        "/api/analyze",
        files={"file": ("photo.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"last_modified": "1700000000000"},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["file_name"] == "photo.jpg"
    assert body["media_type"] == "image"
    assert set(body["categories"]) == CATEGORIES
    assert body["verdict"]["label"] in {"likely_altered", "some_signs", "no_signs", "no_strong_evidence",
                                      "inconclusive"}
    assert 0 <= body["verdict"]["score"] <= 1

    for category in body["categories"].values():
        assert category["status"] in {"flagged", "clear", "not_applicable", "error"}
        for check in category["checks"]:
            assert set(check) == CHECK_FIELDS
            assert check["status"] in {"flagged", "passed", "not_applicable", "info"}

    assert body["categories"]["video_analysis"]["status"] == "not_applicable"
    image_checks = {c["id"] for c in body["categories"]["image_analysis"]["checks"]}
    assert {"copy_move", "edge_texture"} <= image_checks
    compression_checks = {c["id"] for c in body["categories"]["compression_analysis"]["checks"]}
    assert {"blockiness", "quantization_tables", "dct_histogram"} <= compression_checks


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


def test_summary_endpoint_uses_fallback_without_api(monkeypatch):
    error = anthropic.APIConnectionError(request=SimpleNamespace(method="POST", url="x"))
    fake, _ = _fake_client(error=error)
    monkeypatch.setattr(ai_summary.anthropic, "Anthropic", lambda: fake)

    response = client.post("/api/summary", json=_report())
    assert response.status_code == 200
    body = response.json()
    assert body["generated"] is False
    assert body["reasons"][0]["severity"] == "high"


def test_summary_endpoint_rejects_other_json():
    assert client.post("/api/summary", json={"hello": "world"}).status_code == 422
