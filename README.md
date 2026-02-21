# SignifyAI

Real-time sign-to-speech communication system using webcam hand landmarks.

## Quick Start (Simple)

If you just want to run the app:

```powershell
python -u .\src\main.py
```

That starts live mode directly with optimized defaults.

Alternative (same behavior):

```powershell
python -u .\src\hand_tracker.py
```

## Most-used commands

Run app:

```powershell
python -u .\src\main.py
```

Train model from existing dataset:

```powershell
python -u .\src\main.py train
```

Collect samples:

```powershell
python -u .\src\main.py collect --label hello --samples 250
```

## What this project does
- Captures live video from webcam.
- Detects up to two hands using MediaPipe.
- Converts landmarks into a fixed-size feature vector.
- Trains an ensemble classifier (RandomForest + ExtraTrees + LogisticRegression).
- Uses probability calibration for more reliable confidence thresholds.
- Predicts gestures in real-time with smoothing and confidence threshold.
- Speaks predicted words in real-time using text-to-speech.
- Builds a sentence interactively during live mode.
- Includes a prototype-ready rule mode with hardcoded signs (no training required).
- Saves live session logs (`timestamp,label,confidence,hand_count`) for analytics.
- Lets you take screenshots and toggle on-screen controls/help.

## Project layout
- `src/signifyai/hand_tracking.py`: webcam + MediaPipe wrapper.
- `src/signifyai/feature_extraction.py`: feature normalization.
- `src/signifyai/dataset.py`: CSV dataset read/write.
- `src/signifyai/modeling.py`: train/save/load model.
- `src/signifyai/external_data.py`: Kaggle/URL ZIP dataset import.
- `src/signifyai/image_dataset.py`: build landmark CSV from image folders.
- `src/signifyai/language.py`: sentence smoothing/grammar utilities.
- `src/signifyai/rules.py`: hardcoded gesture interpreter (presentation prototype mode).
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
- `models/model_metadata.json`

### 2B) Train from real sign-language image datasets (Kaggle/Web)

1. Import dataset from Kaggle:

```powershell
python -u .\src\main.py import-kaggle --slug grassknoted/asl-alphabet --out-dir .\data\raw\images
```

2. Or import from URL/ZIP:

```powershell
python -u .\src\main.py import-url --url https://example.com/sign_dataset.zip --out-dir .\data\raw\images
python -u .\src\main.py import-zip --zip-file .\downloads\signs.zip --out-dir .\data\raw\images
```

3. Build a landmark dataset from images:

```powershell
python -u .\src\main.py build-image-dataset --images-root .\data\raw\images --out-csv .\data\processed\dataset.csv --max-per-class 1500
```

4. Train:

```powershell
python -u .\src\main.py train
```

### 3) Run real-time sign-to-speech

```powershell
python -u .\src\main.py
```

For tomorrow's prototype demo (recommended):

```powershell
python -u .\src\main.py run --mode hybrid --threshold 0.65 --smooth 9 --rule-threshold 0.78
```

For older PCs (i5 7th gen class), use this optimized command:

```powershell
python -u .\src\main.py run --mode hybrid --width 960 --height 720 --infer-interval 1 --infer-scale 0.60 --threshold 0.62 --smooth 7
```

Live controls:
- `q`: quit
- `v`: voice on/off
- `h`: toggle help overlay
- `space`: append current predicted word to sentence
- `enter`: speak full sentence
- `c`: clear sentence
- `p`: save screenshot to `data/processed/screenshots/`

### Prototype hardcoded signs (rules mode)
- `HELLO`: open palm facing camera
- `HELLO`: waving open palm (left-right-left movement)
- `YES`: thumbs up
- `NO`: thumbs down
- `STOP`: fist
- `PEACE`: index + middle up
- `ONE`: index finger up
- `OKAY`: thumb-index circle with other fingers up
- `CALL ME`: thumb + pinky up
- `ROCK`: index + pinky up
- `I LOVE YOU`: thumb + index + pinky up
- `THANK YOU`: two open palms
- `HELP`: two fists

Note: these are heuristic prototype mappings and not full linguistic ASL grammar.

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
- For highest accuracy, mix webcam-collected data with Kaggle image-derived data.
- Overlapping-hand false positives are reduced with bbox area filtering + IoU duplicate suppression.
- CPU optimization: camera buffer reduction, low-complexity hand model, inference downscaling, and frame-skipped inference.
- Default feature size is 126 (`2 hands x 21 landmarks x 3 values`).
