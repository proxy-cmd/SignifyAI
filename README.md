# SignifyAI

Simple sign-to-speech prototype using webcam.

## Start (easiest way)

Run this:

```powershell
python -u .\app.py
```

This opens a simple menu (stage demo / normal run / collect / train).

Or double-click:
- `run_app.bat`

---

## Start (direct command)

Run this:

```powershell
python -u .\src\main.py
```

That is the main file.

You **do not need training** for demo.
If model files are missing, app runs in `rules` mode automatically.

## Stage Demo (recommended)

Use this one command:

```powershell
python -u .\src\stage_demo.py
```

This starts:
- clean stage UI
- hardcoded reliable gestures
- guided demo prompts

Or double-click:
- `run_stage.bat`

Developer view quick start:
- `run_dev.bat`

If you use VS Code, you can press **F5** on:
- `src/main.py`
- `src/stage_demo.py`

Notebook training option:
- `notebooks/train_signifyai.ipynb`

## Keyboard controls (inside camera window)
- `q` or `Esc` -> quit
- `c` -> clear sentence
- `space` -> add current word to sentence
- `enter` -> speak full sentence
- `v` -> voice on/off
- `a` -> auto-speak on/off
- `m` -> switch mode (rules/hybrid/ml)
- `h` -> show/hide help
- `s` -> show/hide sentence bar
- `p` -> save screenshot
- `k` -> start/stop demo recording (saved in `data/processed/videos/`)

Important: click once on the camera window, then press keys.

Sentence output includes a small grammar cleanup (example: `HELP` -> "Please help.").
Live quality hints are shown in the top-right (lighting/framing/steadiness).

## Modes
- `hybrid` (default): rules + ML together
- `rules`: hardcoded demo signs only
- `ml`: trained model only

## Run profiles (new)
- `balanced` (default)
- `speed` (best for older PCs)
- `accuracy` (more stable labels, slightly heavier)
- `stage` (rules + clean UI + guided prompts)

Run specific mode:

```powershell
python -u .\src\main.py run --mode rules
```

Run a profile:

```powershell
python -u .\src\main.py run --profile speed
python -u .\src\main.py run --profile stage
```

Manual speaking mode (no auto-speak):

```powershell
python -u .\src\main.py run --no-auto-speak
```

For old PCs, adaptive performance is ON by default and auto-tunes inference interval.
You can disable it with:

```powershell
python -u .\src\main.py run --no-adaptive-perf
```

## Hardcoded demo signs (rules mode)
- `HELLO` -> open palm
- `HELLO` -> wave open palm
- `YES` -> thumbs up
- `NO` -> thumbs down
- `STOP` -> fist
- `ONE` -> index up
- `TWO` -> index + middle close together
- `PEACE` -> index + middle wide V
- `OKAY` -> thumb + index circle
- `CALL ME` -> thumb + pinky
- `ROCK` -> index + pinky
- `I LOVE YOU` -> thumb + index + pinky
- `THANK YOU` -> two open palms
- `HELP` -> two fists

## If ML model is missing
If you see:
- `ML model unavailable`

it will automatically run in `rules` mode. This is normal.

## Train your own model (optional)
1. Collect data for each label:

```powershell
python -u .\src\main.py collect --label hello --samples 250
python -u .\src\main.py collect --label thanks --samples 250
```

2. Train:

```powershell
python -u .\src\main.py train
```

AutoML (recommended):

```powershell
python -u .\src\main.py train --automl
```

Files created:
- `models/gesture_model.joblib`
- `models/labels.json`
- `models/model_metadata.json`
- `data/processed/confusion_matrix.csv` (AutoML mode)

## Temporal sequence model (continuous-sign upgrade)

1) Collect sequence clips directly:

```powershell
python -u .\src\main.py collect-seq --label hello --clips 80 --seq-len 24
```

2) Or build sequence dataset from existing frame CSV:

```powershell
python -u .\src\main.py build-seq-dataset --frame-csv .\data\processed\dataset.csv --out-npz .\data\processed\sequence_dataset.npz --seq-len 24 --stride 4
```

3) Train temporal model:

```powershell
python -u .\src\main.py train-seq
```

4) Run temporal realtime mode:

```powershell
python -u .\src\main.py run --mode temporal
```

## Generate demo report (for judges)

After a run, create a clean markdown report:

```powershell
python -u .\src\main.py report
```

Output:
- `data/processed/session_report.md`

## Jupyter training workflow

Use:
- `notebooks/advanced_training.ipynb`

It walks through:
1. loading dataset,
2. AutoML training,
3. evaluation and saved artifacts.

Research roadmap:
- `docs/RESEARCH_ROADMAP.md`

## Full autonomous ML pipeline (one command)

If Kaggle credentials are configured on your machine:

```powershell
python -u .\src\main.py bootstrap-ml
```

This does:
1. import ASL images from Kaggle,
2. convert images to landmark CSV,
3. run AutoML training,
4. save model + labels + metadata.

Or run a full unattended pipeline script (doctor + benchmark + optional bootstrap + report):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\autonomous_pipeline.ps1
```

Quick double-click:
- `run_autonomous.bat`

## Test commands

```powershell
python -u .\src\camera.py
python -u .\src\speak_test.py
```

## If a Python process gets stuck

Close all Python windows:

```powershell
taskkill /F /IM python.exe
```

## Run health checks (before demo)

```powershell
python -u .\src\main.py doctor
```

Skip camera check:

```powershell
python -u .\src\main.py doctor --skip-camera
```

## Benchmark performance (for pitch numbers)

```powershell
python -u .\src\main.py benchmark
```

Shows:
- raw camera FPS
- hand-tracker FPS
