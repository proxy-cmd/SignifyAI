# SignifyAI

SignifyAI is a real-time sign-to-speech project focused on practical usability.

The current build is designed to work fast on normal hardware, and it can learn new signs live without running full model training every time.

## What This Project Does

- Reads hand landmarks from webcam frames.
- Detects known signs in real time.
- Speaks detected text out loud.
- Lets you add new signs during runtime (`T` key or triple blink in supported modes).
- Saves learned signs so they are recognized later.

## Main Modes (UI options)

- `1` Realtime translation (`default`)
- `2` Demo mode (`demo`)
- `3` Emergency mode (`aid`)
- `4` Eye assist mode (`eye`)

Note:
- Extra internal/CLI options exist in backend, but frontend wiring is currently scoped to options 1-4.

## Why It Feels Fast

In `default` mode, the system uses a hybrid runtime method:

1. Hand landmarks are captured frame-by-frame.
2. A lightweight rule layer checks common sign patterns.
3. A prototype layer compares current sign geometry with stored sign vectors.
4. If matched, the sign is emitted and spoken.

This avoids full model inference in the critical loop for `default`, so latency stays low.

## "Single Clip" Learning (What is happening)

When you add a new sign:

1. You trigger teach flow (`T` or triple blink in supported modes).
2. Backend captures the current landmark signature.
3. It stores a prototype vector with metadata and a unique `sign_id`.
4. Next frames are matched against stored prototypes by distance + compatibility profile.

So yes, it can recognize a newly added sign immediately, without retraining a model.

## Data Strategy (Important)

The repository intentionally does **not** include heavy datasets/model artifacts by default.

Large folders can easily exceed hundreds of MB, which slows clone/pull for everyone.

Tracked data is kept minimal:
- folder placeholders
- lightweight prototype path files
- docs

Not tracked by default:
- large external datasets
- generated landmarks
- exports/logs
- most model binaries

See [data/README.md](data/README.md) for details.

## Quick Setup

### 1) Create and activate venv

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3) Run app

```bash
python -u src/main.py
```

## Kaggle Dataset Download (PowerShell + CMD)

### Before download: Kaggle auth

Either:
- set `KAGGLE_USERNAME` and `KAGGLE_KEY`, or
- place `kaggle.json` in `%USERPROFILE%\.kaggle\kaggle.json`

Also install Kaggle CLI once:

```bash
python -m pip install kaggle
```

### PowerShell command

```powershell
$env:ISL_DATASET_ID="<owner>/<dataset>"
powershell -ExecutionPolicy Bypass -File .\scripts\download_datasets.ps1
```

### CMD command

```cmd
set ISL_DATASET_ID=<owner>/<dataset>
scripts\download_datasets.cmd
```

What the wrappers download by default:
- `datamunge/sign-language-mnist`
- `ardamavi/sign-language-digits-dataset`
- plus `ISL_DATASET_ID` if you provide it

### Python-only alternative

```bash
python scripts/bootstrap_data.py --download-kaggle --dataset datamunge/sign-language-mnist --dataset ardamavi/sign-language-digits-dataset --dataset <owner>/<isl-dataset>
```

## About H/J/Y folders in ISL data

If your ISL dataset has:
- `ISL_Dataset/H`
- `ISL_Dataset/J`
- `ISL_Dataset/Y`

keep them under:
- `data/external/kaggle/indian-sign-language-dataset/ISL_Dataset/`

`bootstrap_data.py` checks and reports file counts for `H`, `J`, and `Y` when present.

## Useful Commands

```bash
python -u src/main.py run --mode default
python -u src/main.py run --mode demo
python -u src/main.py run --mode aid
python -u src/main.py run --mode eye
python -m pytest -q
```

## Runtime Notes

- In realtime default mode, emergency intents are isolated (not mixed from aid rules).
- Teach flow saves to:
  - `data/models/sign_prototypes.json`
  - `data/landmarks/raw/live_teach/clips.jsonl`
- If an old taught sign behaves unexpectedly after major logic updates, delete and re-teach it once.

## Troubleshooting

If `python -u src/main.py` fails from editor terminal:

- make sure you are using the same interpreter as `.venv`
- in VS Code, select `.venv\Scripts\python.exe`
- run again from repository root (`D:\SignifyAI`)

## Project Goal

The goal is practical assistive communication:
- low-latency translation
- easy live customization
- less dependency on heavy retraining for everyday usage

It does not replace full ML training pipelines, but it reduces friction for real-world demos and rapid personalization.
