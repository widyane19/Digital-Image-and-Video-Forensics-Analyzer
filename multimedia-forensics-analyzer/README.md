# Digital-Image-and-Video-Forensics-Analyzer

## Overview

A Python-based multimedia forensics system for analyzing digital images and videos to identify measurable technical indicators of editing or manipulation.

The project focuses on detecting potential signs of modification rather than determining whether media is inherently true or false. The system will analyze forensic features, highlight suspicious regions or video frames where possible, and evaluate detection performance using a controlled dataset.

## Objectives

The project aims to:

- Analyze media metadata for inconsistencies.
- Investigate JPEG compression artifacts and recompression indicators.
- Detect potential copied or duplicated image regions.
- Examine unusual edge and texture patterns.
- Identify duplicated or potentially removed video frames and suspicious temporal changes.
- Visually highlight suspicious regions or frames where feasible.
- Compare forensic techniques using quantitative evaluation metrics.
- Investigate which features are most effective for detecting different types of manipulation.

A combined confidence score may be explored as an optional extension.

## Planned Architecture

The repository is organized by forensic analysis category, with evaluation and application logic separated from the underlying detection methods.

```text
multimedia-forensics-analyzer/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├──metadata/
    └──metadata_checks.py
    └──metadata_extraction.py 
│   ├── metadata_analysis.py
│   ├── compression_analysis.py
│   ├── image_analysis.py
│   ├── video_analysis.py
│   └── evaluation.py
├── tests/
│   └── test_analysis.py
├── data/
│   ├── originals/
│   ├── manipulated/
│   └── labels.csv
├── results/
│   ├── metrics.csv
│   └── figures/
└── notebooks/
    └── initial_experiments.ipynb
```

### Module Responsibilities

- **`app.py`**: Application entry point for submitting media, coordinating analysis, and presenting results.
- **`src/metadata_analysis.py`**: Extracts image and video metadata and checks for potentially suspicious inconsistencies.
- **`src/compression_analysis.py`**: Investigates JPEG compression artifacts, recompression, and related compression-based indicators.
- **`src/image_analysis.py`**: Implements image-specific analysis, including copied-region detection, feature matching, and edge or texture analysis.
- **`src/video_analysis.py`**: Handles video frame extraction, duplicated-frame detection, and temporal anomaly analysis.
- **`src/evaluation.py`**: Calculates evaluation metrics, compares techniques, and generates quantitative results.
- **`tests/test_analysis.py`**: Tests analysis functions, expected outputs, and relevant edge cases.
- **`data/`**: Stores original and manipulated media, together with labels describing the ground truth and manipulation types.
- **`results/`**: Stores evaluation metrics, tables, and generated figures.
- **`notebooks/initial_experiments.ipynb`**: Supports preliminary experiments and exploratory analysis before techniques are integrated into the main codebase.

The modules represent the intended final organization. Individual algorithms and their implementation details will be selected and refined during development.

## Dataset and Experimental Design

The evaluation will use a controlled dataset containing original media and intentionally manipulated versions.

The initial target dataset is:

- **Images**: 50 original images and 50 corresponding manipulated versions.
- **Videos**: 5 to 10 short original videos and corresponding modified versions containing inserted, duplicated, or removed frames.

The [NIST Media Forensics Challenge (MFC)](https://mfc.nist.gov/) resources and the [NIST Public Data Repository](https://data.nist.gov/pdr/lps/ark:/88434/mds2-2410) will be investigated as potential supplementary sources. Additional media may be collected or created for controlled experiments.

Each sample should be associated with ground-truth labels describing whether it is original or manipulated and, where applicable, the manipulation type and affected region or frames.

The dataset will be organized to support reproducible experiments and comparisons between detection methods.

## Evaluation

The system will be evaluated against the dataset's ground-truth labels using:

- **Precision**: The proportion of detected manipulations that are actual manipulations.
- **Recall**: The proportion of actual manipulations successfully detected.
- **Accuracy**: The proportion of correctly classified samples.
- **False-positive rate**: The proportion of original samples incorrectly flagged as manipulated.

Results will be compared across forensic techniques and manipulation types to investigate which indicators are most effective and which are more difficult to detect.

Where suitable ground-truth annotations are available, region-level or frame-level detection performance may also be evaluated.

### Step by step: build the dataset and run the evaluation

Run every command from the `multimedia-forensics-analyzer` folder. The sections after this one explain each step in detail.

**0. Before you start**

- Install ExifTool and check that `exiftool -ver` prints a version number. See [Setup](#setup).
- Install the Python packages:

  ```bash
  python -m pip install -r requirements.txt
  ```

**1. Add your originals**

Put unedited photos straight from a phone or camera in `data/originals/images/`, and videos in `data/originals/videos/` (both folders are at the repository root). Then check that each file still has its camera information:

```bash
exiftool -T -FileName -Make -Model -FileSize ../data/originals/images ../data/originals/videos
```

- Move any file that shows `-` for both Make and Model to `data/excluded/`. Such a file was probably downloaded, screenshotted or exported, so it isn't a true original.
  - **Exception:** keep files starting `vision_`. Android phones don't write Make or Model into their videos, so the VISION videos show `-` even though they are genuine.
- Also move any file that is only a few bytes. That's an empty placeholder from a copy that didn't finish.
- Check that no videos ended up in the photos folder.

**2. Download the public samples** (optional; about 375 MB; slow from the VISION server)

```bash
python -m src.external_datasets vision --data ../data --dry-run    # see what it would take
python -m src.external_datasets vision --data ../data              # 12 photos + 2 videos into data/originals/
python -m src.external_datasets csvted --data ../data              # 20 videos into data/external/csvted/
```

**3. Build the manipulated copies** (about 15 minutes for 50 photos and 10 videos)

```bash
python -m src.build_dataset --data ../data --overwrite
```

This writes `data/manipulated/` and `data/labels.csv`. You only need `--overwrite` when `data/manipulated/` already exists.

**4. Run the evaluation**

```bash
# our generated dataset
python -m src.evaluation --labels ../data/labels.csv --out results/generated
# CSVTED's real edits alone
python -m src.evaluation --labels ../data/external/csvted/labels.csv --out results/csvted
# everything together
python -m src.evaluation --labels ../data/labels.csv ../data/external/csvted/labels.csv
```

Each run prints the results table and the easiest-to-hardest ranking. It also writes `metrics.csv`, `by_manipulation.csv` and `difficulty.csv` to its `--out` folder, or to `results/` if you leave out `--out`.

**After adding or removing originals**, repeat steps 1, 3 and 4. The public samples don't need downloading again: files already downloaded are skipped.

### Getting the public datasets

`src/external_datasets` downloads small samples of two public datasets. Run it from the `multimedia-forensics-analyzer` folder. Add `--dry-run` to see the file list and sizes without downloading anything.

```bash
python -m src.external_datasets vision --data ../data    # about 250 MB
python -m src.external_datasets csvted --data ../data    # about 125 MB
```

- **[VISION](https://lesc.dinfo.unifi.it/en/datasets)** (University of Florence, CC BY-SA 4.0):
  - Takes 12 native photos from 12 different phones (`--photos`) and 2 native videos (`--videos`), and saves them in `data/originals/` with names starting `vision_`.
  - These are **extra originals**: the next build makes their manipulated copies.
  - WhatsApp and YouTube copies, and the files in the dataset's errata, are skipped.
- **[CSVTED](https://huggingface.co/datasets/CardiffVisualComputing/CSVTED)** (COMSATS/Cardiff):
  - Videos where **the dataset's authors** deleted, duplicated or inserted frames.
  - Takes 5 scenes (`--scenes`), each with its original and one video per tampering type, and saves them in `data/external/csvted/`.
  - It writes that folder's own `labels.csv`, which has an extra `frames` column giving the tampered frame range.
  - Only those videos are read from the 9–11 GB archives. If the server stops allowing partial downloads, the command stops instead of downloading whole archives.

The same `--seed` always gives the same sample. Each command records the citation, licence and the files it took in `data/SOURCES.md`. CSVTED's page states no licence, so check before publishing results.

### Building the dataset

`src/build_dataset.py` creates the controlled dataset from the team's originals.

1. Put **unedited** originals, straight from a camera or phone, in `data/originals/images/` (photos) and `data/originals/videos/` (videos). Don't use files downloaded from social media: they're already stripped and re-encoded, so they aren't true originals.
2. From the `multimedia-forensics-analyzer` folder, run:

   ```bash
   python -m src.build_dataset --data ../data
   ```

This writes modified copies to `data/manipulated/<type>/` and rewrites `data/labels.csv` to list every original and every copy:

- **Photos:** one modified copy each, with the types taken in turn so each gets about the same number of files: `crop`, `copy_move`, `splice` (a patch from another original), `recompress` (JPEG quality 60) and `metadata_edit` (pixels unchanged; Software set to Adobe Photoshop and ModifyDate moved later).
- **Videos:** one copy per type for each video: `frame_removal` (half a second cut, leaving a gap in the timestamps), `frame_duplication` (half a second repeated) and `frame_insertion` (half a second from another original).

Every copy keeps the original's metadata, as a simple editor would. The same originals always give the same dataset (`--seed` changes it). Rerunning needs `--overwrite` to replace an existing `data/manipulated/`.

**Limitations to mention in the write-up:**

- The modified videos are re-encoded with FFmpeg, which tags them `Lavf`, so the editing-software check catches every one. That makes video edits look easier to detect than they would against a real video editor.
- Every edited photo is re-saved, so it carries new compression as well as the edit itself.

### Running the evaluation

`src/evaluation.py` measures each technique against `data/labels.csv` (columns `file`, `label` and `manipulation_type`; any other columns are ignored). It uses only each technique's own decision for each file: flagged, not flagged or not applicable. It does not use the web app's suspicion score or verdict.

From the `multimedia-forensics-analyzer` folder:

```bash
python -m src.evaluation --labels ../data/labels.csv --data ../data
```

`--labels` takes several label files and measures them together. Each file's paths are relative to its own folder, so leave out `--data`. Use `--out` to keep the runs apart:

```bash
# our edits and CSVTED's together
python -m src.evaluation --labels ../data/labels.csv ../data/external/csvted/labels.csv
# CSVTED alone, to compare real edits with our generated ones
python -m src.evaluation --labels ../data/external/csvted/labels.csv --out results/csvted
```

**Output** (in `results/`):

- `metrics.csv`: precision, recall, accuracy and false-positive rate for each technique. The final `all` row is the detector as a whole, which counts a file as flagged when any technique flags it.
- `by_manipulation.csv`: the same four metrics for each technique and manipulation type. Each row uses that type's manipulated files plus all the originals.
- `difficulty.csv`: manipulation types ranked from easiest to hardest to detect. For each type it gives the detector's recall and the single technique that catches it best.

A file is left out of a technique's metrics when the technique doesn't apply to it, for example video checks on photos. A metric that can't be calculated (no files to divide by) is left blank, not reported as 0. Only techniques that make a yes/no decision are evaluated:

- the metadata checks (`metadata.*`)
- copy-move detection (`image.copy_move`)
- the frame-level video checks (`video.duplicate_frames`, `video.repeated_sequences`, `video.inserted_clips` and `video.temporal_anomalies`). Each is flagged when `video_analysis.py` finds at least one case.

Measurements without a decision, such as blockiness, quantization tables, the DCT histogram and edge/texture, are not evaluated yet.

### Running the tests

From the `multimedia-forensics-analyzer` folder (not from inside `tests/`):

```bash
python -m pytest                                        # every test
python -m pytest tests/test_metadata_checks.py          # one file
python -m pytest tests/test_metadata_checks.py -k gps   # tests whose names contain "gps"
python -m pytest --ignore=tests/test_build_dataset.py   # skip the slower dataset tests
```

`pytest.ini` points pytest at `tests/` and makes `src` importable. Tests that need ExifTool are skipped if it isn't on your `PATH`.

## Technology and References

The project will use Python and relevant multimedia-processing and scientific-computing libraries. Candidate tools include:

- [ExifTool](https://exiftool.org/) and [PyExifTool](https://pypi.org/project/PyExifTool/): Image and video metadata extraction.
- [FFprobe](https://ffmpeg.org/ffprobe.html): Media metadata inspection.
- [OpenCV](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html): Image processing and video frame handling.
- [scikit-image](https://scikit-image.org/docs/stable/api/skimage.metrics.html): Image comparison and similarity metrics.
- [scikit-image ORB feature matching](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_orb.html): Feature-based image comparison.
- [NIST Media Forensics Challenge](https://mfc.nist.gov/): Potential forensic datasets.
- [NIST Public Data Repository](https://data.nist.gov/pdr/lps/ark:/88434/mds2-2410): Additional dataset resource.

The final dependency list will be maintained in `requirements.txt`.

## Setup

### 1. Install ExifTool (required for metadata analysis)

`src/metadata_analysis.py` uses [ExifTool](https://exiftool.org/) to read metadata from both images and videos. The Python package `PyExifTool` is only a wrapper, so the ExifTool program itself must also be installed and on your `PATH`.

**Windows**

1. Download the Windows 64-bit executable (`exiftool-<version>_64.zip`) from [exiftool.org](https://exiftool.org/).
2. Extract the zip to a permanent location outside the repository, for example `C:\exiftool-13.59_64`. The folder must contain both `exiftool.exe` and the `exiftool_files` folder.
3. Make sure the program is named exactly `exiftool.exe`. Windows can hide file extensions, so check that it hasn't become `exiftool.exe.exe` or `exiftool(-k).exe`.
4. Add the **folder** (for example `C:\exiftool-13.59_64`) to your `PATH`: search for *Edit the system environment variables* → **Environment Variables** → select **Path** → **Edit** → **New**. Add the folder path, not the path to `exiftool.exe`.
5. Close and reopen every terminal and VS Code window so they pick up the new `PATH`.

**macOS:** `brew install exiftool`

**Linux (Debian/Ubuntu):** `sudo apt install libimage-exiftool-perl`

**Check the installation** in a new terminal:

```bash
exiftool -ver
```

This should print a version number, such as `13.59`.

> Do not place ExifTool inside the repository folder; it would be committed to Git.

### 2. Install the Python dependencies

```bash
python -m pip install -U pyexiftool
```

### 3. Check that Python can reach ExifTool

```bash
python src/metadata_analysis.py path/to/photo.jpg path/to/video.mp4
```

Each file should be reported as `image` or `video` with a score and any findings. Add `--json` for the full report. An error saying `exiftool.exe is not found` means ExifTool isn't on your `PATH`; recheck step 1.

## Running the Web App

The app has two parts: a FastAPI backend (`app.py`) and a React + TypeScript frontend (`frontend/`, built with Vite). Complete the [Setup](#setup) steps above first.

### Prerequisites

- **Node.js** 20.19+ or 22.12+, which Vite requires. It comes with `npm`. Check with `node -v`.
- The Python dependencies installed from the `multimedia-forensics-analyzer` folder:

  ```bash
  python -m pip install -r requirements.txt
  ```

- *(Optional)* An Anthropic API key for the AI-written summaries. Without it, the app still runs and shows a basic fallback summary.

  ```bash
  # Windows (PowerShell)
  $env:ANTHROPIC_API_KEY = "your-key"
  # macOS / Linux
  export ANTHROPIC_API_KEY="your-key"
  ```

### Option A: Development mode (two terminals, live reload)

**Terminal 1: start the backend** from the `multimedia-forensics-analyzer` folder:

```bash
python -m uvicorn app:app --reload
```

The API runs at <http://localhost:8000>. Check it at <http://localhost:8000/api/health>.

**Terminal 2: start the frontend:**

```bash
cd frontend
npm install      # first time only, or after package.json changes
npm run dev
```

Open <http://localhost:5173>. The Vite dev server forwards `/api` requests to the backend on port 8000, so keep both terminals running. Changes to the frontend code reload in the browser automatically.

### Option B: Single process (built frontend)

Build the frontend once, then let FastAPI serve it:

```bash
cd frontend
npm install
npm run build    # outputs to frontend/dist
cd ..
python -m uvicorn app:app
```

Open <http://localhost:8000>. Rebuild with `npm run build` after changing any frontend code.

### Troubleshooting

- **Uploads fail or "Failed to fetch" in development:** the backend isn't running, or isn't on port 8000. Start it as shown in Terminal 1.
- **`npm` is not recognized:** install Node.js from [nodejs.org](https://nodejs.org/), then reopen the terminal.
- **Port 5173 already in use:** Vite picks the next free port and prints it. Use the URL shown in the terminal.

To run all tests:
```bash
cd multimedia-forensics-analyzer
python -m pytest