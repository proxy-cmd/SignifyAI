# SignifyAI

A real-time **sign-to-speech** project in Python.
It reads hand gestures from webcam and speaks the detected word/sentence.

This repo has multiple run options:
- simple launcher
- stage demo mode
- lite demo mode
- full CLI (collect/train/run/report)

## 1) What we used

Main libraries:
- `opencv-python` -> webcam capture + window UI
- `mediapipe` -> hand landmark detection
- `numpy` -> math on hand points
- `scikit-learn` -> ML classifier training
- `pyttsx3` -> text-to-speech
- `pandas` -> dataset CSV handling
- `joblib` -> save/load trained models

Installed from `requirements.txt`.

## 2) Project idea in one line

Webcam -> hand landmarks -> rules/ML model -> label -> speech output.

## 3) Folder map (important files)

- `app.py` -> simple menu launcher
- `src/main.py` -> main CLI command runner
- `src/stage_demo.py` -> presentation-friendly demo
- `src/gui.py` -> GUI buttons for common tasks
- `src/signifyai/realtime.py` -> live camera inference loop
- `src/signifyai/rules.py` -> rule-based gesture logic
- `src/signifyai/hand_tracking.py` -> MediaPipe tracking wrapper
- `lite_demo/run_lite.py` -> separate simplified hackathon demo
- `for_hackathon/main.py` -> minimal hackathon-style standalone flow

## 4) Quick start (easiest)

From project root (`D:\SignifyAI`):

```powershell
python -u .\app.py
```

Then choose from menu:
- Stage demo
- Normal realtime
- Rules-only realtime
- Collect samples
- Train model
- Report, GUI, advanced tools

## 5) Other run options

### A) Stage demo (best for presentation)

```powershell
python -u .\src\stage_demo.py
```

### B) Lite demo (separate simple app)

```powershell
python -u .\lite_demo\run_lite.py
```

or double-click:
- `run_lite_demo.bat`

### C) Main app directly

```powershell
python -u .\src\main.py
```

`src/main.py` with no args auto-starts `run`.

### D) Production profile (recommended for stable deployment)

```powershell
python -u .\src\main.py run --profile production
```

This uses tuned hybrid settings with stricter smoothing/thresholds.
You can also double-click `run_production.bat`.
For safer startup (preflight first), double-click `run_production_safe.bat`.

### E) Smooth HD profile (for best visual smoothness)

```powershell
python -u .\src\main.py run --profile smoothhd
```

This profile requests:
- 1280x720 camera
- higher camera FPS request
- smoother landmarks
- hybrid recognition mode

You can also double-click `run_smoothhd.bat`.

### E2) Ultra speed profile (older CPUs)

```powershell
python -u .\src\main.py run --profile ultra-speed
```

This profile lowers processing load and targets higher FPS on weak hardware.

### E3) Ultra accuracy profile (strictest recognition)

```powershell
python -u .\src\main.py run --profile ultra-accuracy
```

This profile uses stronger smoothing, stricter thresholds, and strict consensus.

### F) Enterprise profile (strictest runtime policy)

```powershell
python -u .\src\main.py run --profile enterprise
```

This profile enables:
- stricter confidence thresholds
- quality gate (brightness/blur/hand-size checks)
- strict multi-source consensus in hybrid mode
- stronger debounce and stability checks before speaking

You can also double-click `run_enterprise.bat`.
For safer startup (preflight first), double-click `run_enterprise_safe.bat`.

## 6) Realtime keyboard controls

Inside camera window:
- `q` or `Esc` -> quit
- `v` -> voice on/off
- `a` -> auto-speak on/off
- `m` -> switch mode (`rules/hybrid/ml/temporal`)
- `space` -> add current label to sentence
- `enter` -> speak sentence
- `c` -> clear sentence
- `p` -> save screenshot
- `k` -> start/stop recording
- `h` -> help overlay
- `tab` -> stage/dev UI switch
- `f` -> fullscreen

## 7) Gesture usage guide (to avoid confusion)

### Greetings (`GOOD MORNING/AFTERNOON/EVENING/NIGHT`)
- Show **2 open palms** (all fingers up), facing camera.
- Keep both hands at similar height.
- Keep hands **apart** (not very close).
- Hold steady around 0.7 to 1.0 sec.
- Output depends on system time:
  - `05:00-11:59` -> `GOOD MORNING`
  - `12:00-16:59` -> `GOOD AFTERNOON`
  - `17:00-20:59` -> `GOOD EVENING`
  - `21:00-04:59` -> `GOOD NIGHT`

### Thank You
- Also 2 open palms, but keep hands **closer together**.
- If close and level, model prefers `THANK YOU`.

### OKAY
- Touch thumb tip + index tip (small ring).
- Keep middle, ring, pinky up.
- Hold for 0.5 to 1.0 sec.

### YES / NO
- Keep other fingers folded.
- Thumb up -> `YES`
- Thumb down -> `NO`
- Keep thumb mostly vertical (not sideways).

### ROCK / I LOVE YOU
- Both: index+pinky up, middle+ring down.
- `I LOVE YOU`: thumb clearly out.
- `ROCK`: thumb near palm/folded.

## 8) Camera quality tips

- Use front lighting (avoid backlight).
- Keep hand fully visible in frame.
- Do one sign at a time.
- Pause briefly between signs.
- If blur hint appears, slow down motion and clean lens.

## 9) Training flow (basic)

### Step 1: collect frame samples

```powershell
python -u .\src\main.py collect --label hello --samples 250
python -u .\src\main.py collect --label thanks --samples 250
```

Collection is now reliable by default:
- auto-capture mode on
- periodic flush to CSV while recording
- duplicate filtering to avoid repeated same-frame samples

Useful options:

```powershell
python -u .\src\main.py collect --label hello --samples 250 --no-auto
python -u .\src\main.py collect --label hello --samples 250 --capture-interval 0.30 --flush-every 10
```

### Step 2: train frame model

```powershell
python -u .\src\main.py train --automl
```

If some labels have very few samples, training now drops them automatically
(default `--min-samples-per-label 5`) to reduce noisy predictions.

### Step 3: run

```powershell
python -u .\src\main.py run --mode hybrid
```

## 10) Temporal (sequence) flow

Collect sequence clips:

```powershell
python -u .\src\main.py collect-seq --label hello --clips 80
python -u .\src\main.py collect-seq --label thanks --clips 80
```

Train sequence model:

```powershell
python -u .\src\main.py train-seq
```

Run temporal mode:

```powershell
python -u .\src\main.py run --mode temporal
```

## 11) Easy custom sentence gesture

Single command to map phrase + record sequence:

```powershell
python -u .\src\main.py record-combo --label watching_you --text "I am watching you." --clips 80
```

Then:

```powershell
python -u .\src\main.py train-seq
python -u .\src\main.py run --mode temporal
```

List custom phrase mappings:

```powershell
python -u .\src\main.py list-phrases
```

## 12) Adapt from images (auto learn)

Read landmarks from one image and save overlay:

```powershell
python -u .\src\main.py image-points --image .\path\to\sign.jpg --out .\data\processed\points_overlay.png
```

Learn one custom sign from image(s) or step-image folder:

```powershell
python -u .\src\main.py adapt-sign --label watching_you --images .\path\to\steps_folder --phrase "I am watching you."
```

Learn many signs from folder structure:

```powershell
python -u .\src\main.py adapt-signs-folder --images-root .\data\raw\images\my_signs --max-per-label 120
```

Realtime uses these saved prototypes automatically in hybrid/ml mode.

## 13) Useful commands

- Health check:
  - `python -u .\src\main.py doctor`
  - or double-click `run_doctor.bat`
- Production preflight:
  - `python -u .\src\main.py preflight --mode hybrid`
  - or double-click `run_preflight.bat`
- Performance benchmark:
  - `python -u .\src\main.py benchmark`
- Session report:
  - `python -u .\src\main.py report`
- Offline video inference:
  - `python -u .\src\main.py infer-video --input .\data\raw\demo.mp4`
- Full CLI help:
  - `python -u .\src\main.py -h`

## 14) One-command dataset bootstrap (web URL -> train)

If you have a direct ZIP URL for a sign dataset:

```powershell
python -u .\src\main.py bootstrap-url --url <zip-url>
```

Example used in this project:

```powershell
python -u .\src\main.py bootstrap-url --url https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/refs/heads/master.zip --out-dir .\data\raw\images --max-per-class 1200 --min-free-gb 2
```

This command:
- downloads ZIP to `D:\SignifyAI\data\raw`
- extracts into `data/raw/images`
- auto-detects nested class folder layout
- builds landmark CSV
- runs AutoML training

You can also do Kaggle one-command bootstrap:

```powershell
python -u .\src\main.py bootstrap-ml --slug grassknoted/asl-alphabet
```

## 15) Common errors and quick fixes

- Camera not opening:
  - close other apps using camera
  - run `python -u .\src\main.py doctor`
- Wrong interpreter/import issues:
  - check `python --version`
  - reinstall requirements with same interpreter:
    - `python -m pip install -r requirements.txt`
- Python process stuck:
  - `taskkill /F /IM python.exe`
- AutoML says at least 2 labels required:
  - collect data for 2+ labels, then retrain

## 16) Notes

- Rules mode is most stable for quick demos.
- Hybrid mode combines rules + ML + temporal fallback.
- Temporal gives better continuity but needs sequence training data.
