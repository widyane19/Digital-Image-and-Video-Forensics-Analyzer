"""Web API for the forensics analyzer.

Run from the multimedia-forensics-analyzer folder:
    python -m uvicorn app:app --reload

The React frontend in frontend/ calls POST /api/analyze. If frontend/dist exists
(after npm run build), it is served at / so one process runs the whole app.

Adds support for CLI video analysis
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile

from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.metadata.metadata_extraction import UnsupportedMediaError
from src.ai_summary import summarize
from src.pipeline import analyze

# CLI video analysis imports
from src.video_analysis import analyze_video, save_report

MAX_UPLOAD_BYTES = 200 * 1024 * 1024

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

app = FastAPI(title="Multimedia Forensics Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}

@app.post("/api/analyze")
def analyze_upload(
    file: UploadFile = File(...),
    last_modified: int | None = Form(None),
) -> dict:

    """Save uploaded media temporarily, analyze it, and delete it."""

    original_modified = None

    if last_modified:
        original_modified = datetime.fromtimestamp(
            last_modified / 1000,
            tz=timezone.utc,
        )

    suffix = Path(file.filename or "").suffix.lower()

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as tmp:

        temp_path = tmp.name

        try:
            shutil.copyfileobj(
                file.file,
                tmp,
                length=1024 * 1024,
            )

        except Exception:
            tmp.close()
            os.unlink(temp_path)
            raise

    try:

        file_size = os.path.getsize(temp_path)

        if file_size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            )

        if file_size == 0:
            raise HTTPException(
                400,
                "The uploaded file is empty.",
            )

        return analyze(
            temp_path,
            original_modified,
            display_name=file.filename,
        )

    except UnsupportedMediaError as exc:

        raise HTTPException(
            415,
            str(exc).replace(
                temp_path,
                file.filename or "file",
            ),
        ) from exc

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/api/summary")
def summary(report: dict = Body(...)) -> dict:

    """Generate a plain-English summary of an analysis report."""

    if (
        not isinstance(report.get("categories"), dict)
        or "verdict" not in report
    ):
        raise HTTPException(
            422,
            "Send the report returned by /api/analyze.",
        )

    return summarize(report)

# CLI VIDEO ANALYSIS

def main():

    """Run video forensics analysis from the command line."""

    parser = argparse.ArgumentParser(
        description="Multimedia Video Forensics Analyzer"
    )

    parser.add_argument(
        "video",
        help="Path to the video being analyzed",
    )

    parser.add_argument(
        "--output",
        default="results/reports/video_report.json",
        help="Path where the JSON report will be saved",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Duplicate detection threshold",
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="Analyze every Nth video frame",
    )

    args = parser.parse_args()

    video_path = Path(args.video)

    if not video_path.is_file():
        parser.error(f"Video file not found: {args.video}")

    if args.sample_every < 1:
        parser.error("--sample-every must be at least 1")

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nStarting video forensics analysis...")
    print("------------------------------------")

    report = analyze_video(
        args.video,
        sample_every=args.sample_every,
        duplicate_threshold=args.threshold,
    )

    save_report(
        report,
        str(output_path),
    )

    print("\nVIDEO ANALYSIS COMPLETE")
    print("-----------------------")

    print(
        json.dumps(
            report["summary"],
            indent=4,
        )
    )

    print(f"\nReport saved to: {output_path}")

if FRONTEND_DIST.is_dir():

    app.mount(
        "/",
        StaticFiles(
            directory=FRONTEND_DIST,
            html=True,
        ),
        name="frontend",
    )

if __name__ == "__main__":
    main()
