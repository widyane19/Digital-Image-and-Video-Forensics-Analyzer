"""Metadata analysis for images and videos: Steps 5-7 and the public entry point.

Pipeline (see plan/metadata_analysis_plan.md):
    1-3. Identify, extract and normalise   -> metadata_extraction.py
    4.   Run consistency checks             -> metadata_checks.py
    5.   Record findings                    -> this file
    6.   Calculate a suspicion score        -> this file
    7.   Provide output for the app and evaluation -> this file
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable

from exiftool import ExifToolHelper

try:  # imported as part of the src package (app.py, tests)
    from .metadata.metadata_checks import SEVERITY_WEIGHTS, CheckContext, Finding, run_checks
    from .metadata.metadata_extraction import (
        UnsupportedMediaError, extract_metadata, normalize, open_exiftool,
    )
except ImportError:  # run directly: python src/metadata_analysis.py <file>
    from metadata.metadata_checks import SEVERITY_WEIGHTS, CheckContext, Finding, run_checks
    from metadata.metadata_extraction import (
        UnsupportedMediaError, extract_metadata, normalize, open_exiftool,
    )

# With this many checks or fewer, a file without findings hasn't really been assessed.
FEW_CHECKS = 3


def suspicion_score(findings: Iterable[Finding]) -> float:
    """Combine findings into a score from 0 to 1.

    Each finding closes part of the remaining gap to 1, so weak clues add up,
    more evidence always raises the score, and it can never go above 1.
    """
    remaining = 1.0
    for finding in findings:
        remaining *= 1 - SEVERITY_WEIGHTS[finding.severity]
    return round(1 - remaining, 4)


@dataclass
class MetadataReport:
    """The result of analysing one file. It reports evidence of editing, not "real" or "fake"."""

    path: str
    media_type: str
    score: float
    findings: list[Finding]    # most serious first
    results: dict[str, str]    # check name -> "flagged", "not_flagged" or "not_applicable"
    warnings: list[str]

    @property
    def checks_run(self) -> int:
        return sum(status != "not_applicable" for status in self.results.values())

    @property
    def conclusion(self) -> str:
        """A plain-English summary. A low score alone doesn't mean a file is clean."""
        serious = [f for f in self.findings if SEVERITY_WEIGHTS[f.severity] >= SEVERITY_WEIGHTS["medium"]]
        if serious:
            return f"Found {len(serious)} serious sign(s) of editing in the metadata."
        if self.checks_run <= FEW_CHECKS:
            return "Not enough metadata to assess this file."
        return ("No serious signs of editing in the metadata. This doesn't prove the file is "
                "unedited, because a re-encoded file can carry no trace of an edit.")

    def to_dict(self) -> dict:
        """The detailed report, for the app and JSON output."""
        return {
            "path": self.path,
            "media_type": self.media_type,
            "score": self.score,
            "conclusion": self.conclusion,
            "checks_run": self.checks_run,
            "checks_total": len(self.results),
            "findings": [asdict(finding) for finding in self.findings],
            "results": self.results,
            "warnings": self.warnings,
        }

    def summary_row(self) -> dict:
        """One row for the evaluation: the file, the score and one column per check."""
        return {"file": self.path, "score": self.score, **self.results}


def analyze_file(
    path: str | os.PathLike,
    et: ExifToolHelper | None = None,
    original_modified: datetime | None = None,
    uploaded: bool = False,
) -> MetadataReport:
    """Analyse one image or video.

    Pass an already-open ExifTool process as `et` when analysing many files.

    Saving an upload gives it a new file date, so the app should set `uploaded=True`
    and pass the date the browser reports as `original_modified`, if it has one.
    """
    meta = normalize(extract_metadata(path, et))
    file_date = original_modified or (None if uploaded else meta.file_modified)

    results, warnings = run_checks(CheckContext(meta, file_date=file_date, et=et))
    findings = [result.finding for result in results.values() if result.finding]
    findings.sort(key=lambda finding: SEVERITY_WEIGHTS[finding.severity], reverse=True)

    return MetadataReport(
        path=meta.path,
        media_type=meta.media_type,
        score=suspicion_score(findings),
        findings=findings,
        results={name: result.status for name, result in results.items()},
        warnings=warnings,
    )


def analyze_batch(paths: Iterable[str | os.PathLike]) -> tuple[list[MetadataReport], dict[str, str]]:
    """Analyse many files using a single ExifTool process.

    Returns (reports, failures). One unreadable file does not stop the batch;
    its error message is recorded in `failures` under its path.
    """
    reports: list[MetadataReport] = []
    failures: dict[str, str] = {}
    with open_exiftool() as et:
        for path in paths:
            try:
                reports.append(analyze_file(path, et))
            except (FileNotFoundError, UnsupportedMediaError) as exc:
                failures[str(path)] = str(exc)
    return reports, failures


def _print_report(report: MetadataReport) -> None:
    print(f"\n{report.path}: {report.media_type}, score {report.score:.2f} "
          f"({report.checks_run} of {len(report.results)} checks ran)")
    print(f"  {report.conclusion}")
    for finding in report.findings:
        print(f"  [{finding.severity}] {finding.check}: {finding.message}")
    for warning in report.warnings:
        print(f"  warning: {warning}")


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Look for metadata evidence of editing.")
    parser.add_argument("files", nargs="+", help="images or videos to analyse")
    parser.add_argument("--json", action="store_true", help="print the full reports as JSON")
    args = parser.parse_args()

    try:
        found, failed = analyze_batch(args.files)
    except FileNotFoundError as exc:
        sys.exit(f"{exc}\nExifTool must be installed and on your PATH. See the README for setup steps.")

    if args.json:
        print(json.dumps({"reports": [r.to_dict() for r in found], "failures": failed}, indent=2))
    else:
        for report in found:
            _print_report(report)
        for path, error in failed.items():
            print(f"\n{path}: FAILED - {error}")
