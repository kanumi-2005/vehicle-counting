# Vehicle Detection and Counting in Traffic Surveillance

> **Course Project** - CSC16004 - Thị giác máy tính
> University of Science, Vietnam National University Ho Chi Minh City
> Faculty of Information Technology

## Table of Contents

- [Overview](#overview)
- [Course Information](#course-information)
- [Project Structure](#project-structure)
- [What We Implemented](#what-we-implemented)
- [Setup](#setup)
- [Data and Configuration](#data-and-configuration)
- [Code Instructions](#code-instructions)
- [Report and Presentation](#report-and-presentation)
- [Team](#team)
- [License](#license)

## Overview

This project studies vehicle detection, multi-object tracking, and vehicle
counting in traffic surveillance videos. The proposed pipeline follows a
tracking-by-detection design: YOLOv8n detects vehicles in each frame, BYTE or
SORT associates detections across frames, and ROI-based counting aggregates
tracked IDs into traffic counts.

The project focuses on the UA-DETRAC dataset and provides scripts for
preprocessing, YOLO conversion, TrackEval MOT export, tracking evaluation, and
an interactive ROI-based tracking counter demo.

## Course Information

| | |
|---|---|
| **Course** | CSC16004 - Thị giác máy tính |
| **Class** | 23TN |
| **Semester** | 25-26/2 |
| **Topic** | Vehicle Detection and Counting in Traffic Surveillance |

## Project Structure

```text
vehicle-counting/
├── data/
│   ├── config.json                  # Default data and pipeline config
│   ├── preprocessed/                # Preprocessed images and split manifest
│   ├── raw/                         # Raw UA-DETRAC data
│   ├── trackeval/                   # TrackEval MOT-style data
│   └── yolo/                        # YOLO dataset and data.yaml
├── demo/
│   ├── demo1.mp4                    # Rendered tracking/counting demo
│   └── demo2.mp4                    # Rendered tracking/counting demo
├── presentation/
│   └── main.tex                     # Beamer presentation
├── report/
│   └── main.tex                     # LaTeX report
├── src/
│   ├── preprocessing.py             # Apply ignore boxes and split sequences
│   ├── convert_to_yolo.py           # Convert annotations to YOLO format
│   ├── convert_to_trackeval_mot.py  # Convert GT to TrackEval MOT format
│   ├── run_and_export_trackeval_mot.py
│   │                                 # Run detector/tracker and export results
│   ├── tracking_pipeline.py         # YOLO + BYTE/SORT tracking pipeline
│   └── run_tracking_counter.py      # Interactive ROI tracking/counting demo
├── third_party/
│   └── TrackEval/                   # Tracking evaluation toolkit
├── videos/
│   ├── demo1.mp4                    # Source demo video
│   └── demo2.mp4                    # Source demo video
├── weights/
│   └── best.pt                      # YOLO weights, if available locally
├── .apprc                           # App-level config for interactive demo
├── requirements.txt                 # Python dependencies
└── README.md                        # Project overview
```

## What We Implemented

1. A configurable preprocessing pipeline for UA-DETRAC images and annotations.
2. Deterministic train/validation splitting with a split manifest.
3. YOLO dataset export, including `data.yaml` generation.
4. TrackEval MOT-style ground-truth and tracker-result export.
5. A tracking pipeline supporting BYTE and SORT trackers.
6. Evaluation scripts for HOTA, CLEAR MOT, and Identity metrics.
7. An interactive tracking/counting application with editable polygon ROIs.
8. A LaTeX report and Beamer presentation summarizing the method and results.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TrackEval is included under `third_party/TrackEval`. When running evaluation
scripts, make sure it is importable:

```bash
export PYTHONPATH="$PWD/third_party/TrackEval:$PYTHONPATH"
```

## Data and Configuration

The main configuration file is:

```text
data/config.json
```

It stores paths, dataset metadata, detection defaults, tracking defaults, and
TrackEval metric settings. Most scripts support:

```bash
--config data/config.json
--set key.path=value
```

The interactive tracking counter also reads app-level defaults from:

```text
.apprc
```

## Code Instructions

Preprocess UA-DETRAC images and annotations:

```bash
python src/preprocessing.py --config data/config.json
```

If `data/preprocessed/splits.json` already exists and you want to regenerate
the split manifest, add `--overwrite-splits`.

Convert the preprocessed data to YOLO format:

```bash
python src/convert_to_yolo.py --config data/config.json
```

Run tracking and export TrackEval MOT results:

```bash
python src/run_and_export_trackeval_mot.py \
  --config data/config.json \
  --tracker byte
```

Run the interactive ROI tracking/counting demo:

```bash
python src/run_tracking_counter.py \
  --video videos/demo1.mp4 \
  --tracker byte \
  --save-output demo/result.mp4
```

Useful controls in the demo:

| Key | Action |
|---|---|
| `n` | Create a new ROI |
| `s` | Select an ROI |
| `x` | Delete an ROI by clicking inside it |
| `Enter` | Complete the current action |
| `Esc` | Cancel the current action |
| `h` | Toggle the current panel |
| `F1` | Toggle the control panel |
| `F2` | Toggle the tracking/counting panel |
| `q` | Confirm application exit |

## Report and Presentation

Build the report:

```bash
cd report
latexmk
```

Build the presentation:

```bash
cd presentation
latexmk
```

## Team

| Name | Student ID | Role |
|---|---|---|
| Hoàng Ngọc Phú | 23120010 | Team Lead |
| Hoàng Ngọc Quí | 23120077 | Member |
| Nguyễn Duy Bảo | 23120113 | Member |

## License

This project is developed for educational purposes as part of the Computer
Vision course at VNUHCM-US.
