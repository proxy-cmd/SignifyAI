# SignifyAI

Real-time sign-to-speech communication system using webcam hand landmarks.

## What this project does
- Captures live video from webcam.
- Detects up to two hands using MediaPipe.
- Converts landmarks into a fixed-size feature vector.
- Trains a gesture classifier on your custom labels.
- Predicts gestures in real-time with smoothing and confidence threshold.
- Speaks predicted words in real-time using text-to-speech.
- Builds a sentence interactively during live mode.
- Saves live session logs (`timestamp,label,confidence,hand_count`) for analytics.
- Lets you take screenshots and toggle on-screen controls/help.

## Project layout
- `src/signifyai/hand_tracking.py`: webcam + MediaPipe wrapper.
- `src/signifyai/feature_extraction.py`: feature normalization.
- `src/signifyai/dataset.py`: CSV dataset read/write.
- `src/signifyai/modeling.py`: train/save/load model.
- `src/signifyai/collect.py`: label data collection loop.
- `src/signifyai/train.py`: training pipeline.
- `src/signifyai/realtime.py`: live prediction + speech + UI overlays.
- `src/signifyai/tts.py`: threaded speech engine.
- `src/main.py`: CLI command runner.
- `src/collect_data.py`, `src/train_model.py`, `src/run_realtime.py`: simple wrappers.
- `tests/`: non-camera unit tests.

## Requirements
- Python 3.10+
- Webcam
- Windows/macOS/Linux

Install deps:

```powershell
pip install -r requirements.txt
```

## End-to-end workflow

### 1) Collect samples for each gesture label
Use one command per label:

```powershell
python -u .\src\main.py collect --label hello --samples 250
python -u .\src\main.py collect --label thanks --samples 250
python -u .\src\main.py collect --label yes --samples 250
python -u .\src\main.py collect --label no --samples 250
```

Collection controls:
- `c`: capture current frame as a sample.
- `q`: quit.

Tips:
- Collect in different lighting and angles.
- Keep sample counts balanced across labels.

### 2) Train model

```powershell
python -u .\src\main.py train
```

Outputs:
- `models/gesture_model.joblib`
- `models/labels.json`

### 3) Run real-time sign-to-speech

```powershell
python -u .\src\main.py run
```

Live controls:
- `q`: quit
- `v`: voice on/off
- `h`: toggle help overlay
- `space`: append current predicted word to sentence
- `enter`: speak full sentence
- `c`: clear sentence
- `p`: save screenshot to `data/processed/screenshots/`

## Quick scripts
- `python -u .\src\camera.py`: camera smoke test.
- `python -u .\src\speak_test.py`: speech smoke test.
- `python -u .\src\hand_tracker.py`: compatibility wrapper for live mode.

## Testing
Run non-camera tests:

```powershell
python -m unittest discover -s tests -v
```

## Notes
- If speech is slow, reduce model complexity or camera resolution.
- If predictions are noisy, increase data per class and tune threshold/smoothing.
- Default feature size is 126 (`2 hands x 21 landmarks x 3 values`).
