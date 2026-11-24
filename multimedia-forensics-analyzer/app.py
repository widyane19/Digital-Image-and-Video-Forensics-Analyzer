"""Web API for the forensics analyzer.

Run from the multimedia-forensics-analyzer folder:
    python -m uvicorn app:app --reload

The React frontend in frontend/ calls POST /api/analyze. If frontend/dist exists
(after `npm run build`), it is served at / so one process runs the whole app.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.metadata.metadata_extraction import UnsupportedMediaError
from src.ai_summary import summarize
from src.pipeline import analyze

# Read the API key from .env (kept out of Git). Variables already set in the
# terminal take precedence.
load_dotenv(Path(__file__).parent / ".env")

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

app = FastAPI(title="Multimedia Forensics Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze_upload(
    file: UploadFile = File(...),
    last_modified: int | None = Form(None),  # File.lastModified from the browser, in ms
) -> dict:
    """Save the upload to a temporary file, analyse it, then delete it.

    This is a plain `def`, so FastAPI runs it in a worker thread and slow
    analysis does not block other requests.
    """
    original_modified = None
    if last_modified:
        original_modified = datetime.fromtimestamp(last_modified / 1000, tz=timezone.utc)

    suffix = Path(file.filename or "").suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        temp_path = tmp.name
        try:
            shutil.copyfileobj(file.file, tmp, length=1024 * 1024)
        except Exception:
            tmp.close()
            os.unlink(temp_path)
            raise

    try:
        if os.path.getsize(temp_path) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        if os.path.getsize(temp_path) == 0:
            raise HTTPException(400, "The uploaded file is empty.")
        return analyze(temp_path, original_modified, display_name=file.filename)
    except UnsupportedMediaError as exc:
        # Show the user's file name, not the server's temporary path.
        raise HTTPException(415, str(exc).replace(temp_path, file.filename or "file")) from exc
    finally:
        os.unlink(temp_path)


@app.post("/api/summary")
def summary(report: dict = Body(...)) -> dict:
    """Write a plain-English summary of a report from /api/analyze.

    Kept separate from /api/analyze so the results can be shown straight away
    while the summary is generated. A plain `def`, so it runs in a worker thread.
    """
    if not isinstance(report.get("categories"), dict) or "verdict" not in report:
        raise HTTPException(422, "Send the report returned by /api/analyze.")
    return summarize(report)


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
