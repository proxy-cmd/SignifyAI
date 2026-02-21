# SignifyAI

Simple sign-to-speech prototype using webcam.

## Start (no setup commands here)

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

If you use VS Code, you can also press **F5** on:
- `src/hand_tracker.py`

Notebook training option:
- `notebooks/train_signifyai.ipynb`

## Keyboard controls (inside camera window)
- `q` or `Esc` -> quit
- `c` -> clear sentence
- `space` -> add current word to sentence
- `enter` -> speak full sentence
- `v` -> voice on/off
- `h` -> show/hide help
- `s` -> show/hide sentence bar
- `p` -> save screenshot

Important: click once on the camera window, then press keys.

## Modes
- `hybrid` (default): rules + ML together
- `rules`: hardcoded demo signs only
- `ml`: trained model only

Run specific mode:

```powershell
python -u .\src\main.py run --mode rules
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

Files created:
- `models/gesture_model.joblib`
- `models/labels.json`
- `models/model_metadata.json`

## Test commands

```powershell
python -u .\src\camera.py
python -u .\src\speak_test.py
```
