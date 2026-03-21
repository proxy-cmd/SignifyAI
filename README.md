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
- `2` Emergency mode (`aid`)
- `3` Eye assist mode (`eye`)
- `4` Fast sign record mode (`teach`)
- `5` Manage taught signs

Note:
- Extra internal/CLI options exist in backend via advanced menu and commands.

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

## ZIP Handoff (Recommended)

If you are sharing this project with testers, use a zip that already includes your prepared `data/` content.

No Kaggle setup is required in this flow.

### What to include in your zip

Required folders:
- `src/`
- `data/models/`
- `data/landmarks/`
- `data/external/` (if you want dataset/build features to work)

Required runtime files (used directly by Python paths):
- `data/models/hand_landmarker.task`
- `data/models/face_landmarker.task` (needed for eye mode)
- `data/models/registry.json`
- `data/models/sign_prototypes.json`

Recommended if you want pretrained startup behavior:
- `data/models/global.joblib` and `data/models/global.json`
- `data/models/custom.joblib` and `data/models/custom.json` (if you trained custom)

Recommended if you want saved teach history:
- `data/landmarks/raw/live_teach/clips.jsonl`
- `data/landmarks/raw/live_teach/session.json`

### One local prep command (folder/file bootstrap only)

```bash
python scripts/bootstrap_data.py
```

This command only creates required local folders/placeholders and prints missing runtime files.

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

## Useful Commands

```bash
python -u src/main.py run --mode default
python -u src/main.py run --mode aid
python -u src/main.py run --mode eye
python -u src/main.py run --mode teach
python -u src/main.py promote --model-name custom --min-val-acc 0.70 --min-gain 0.01
python -u src/main.py rollback --notes "fallback to previous stable model"
python -m pytest -q
```

Runtime tuning flags are available on `run`, for example:
- `--uncertainty-min-conf 0.48`
- `--speech-repeat-cooldown-sec 1.8`
- `--speech-global-cooldown-sec 0.35`
- `--watchdog-reset-sec 5.0`

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
