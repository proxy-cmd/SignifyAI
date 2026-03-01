# SignifyAI

A real-time **sign-to-speech** project in Python.
It reads hand gestures from webcam and speaks the detected word/sentence.

This repo has multiple run options:
- simple launcher
- stage demo mode
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

Optional deep-learning dependency:
- `tensorflow` (install from `requirements-deep.txt` for `train-deep` / deep stage of `train-all`)

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

### B) Main app directly

```powershell
python -u .\src\main.py
```

`src/main.py` with no args auto-starts `run`.

### C) Production profile (recommended for stable deployment)

```powershell
python -u .\src\main.py run --profile production
```

This uses tuned hybrid settings with stricter smoothing/thresholds.
You can also double-click `run_production.bat`.
For safer startup (preflight first), double-click `run_production_safe.bat`.

### D) Smooth HD profile (for best visual smoothness)

```powershell
python -u .\src\main.py run --profile smoothhd
```

This profile requests:
- 1280x720 camera
- higher camera FPS request
- smoother landmarks
- hybrid recognition mode

You can also double-click `run_smoothhd.bat`.

### D2) Ultra speed profile (older CPUs)

```powershell
python -u .\src\main.py run --profile ultra-speed
```

This profile lowers processing load and targets higher FPS on weak hardware.
It disables deep runtime inference by default for speed.

### D3) Ultra accuracy profile (strictest recognition)

```powershell
python -u .\src\main.py run --profile ultra-accuracy
```

This profile uses stronger smoothing, stricter thresholds, and strict consensus.
It also enables deep runtime fusion by default.

### E) Enterprise profile (strictest runtime policy)

```powershell
python -u .\src\main.py run --profile enterprise
```

This profile enables:
- stricter confidence thresholds
- quality gate (brightness/blur/hand-size checks)
- strict multi-source consensus in hybrid mode
- stronger debounce and stability checks before speaking
- deep model fusion with conservative disagreement blocking

You can also double-click `run_enterprise.bat`.
For safer startup (preflight first), double-click `run_enterprise_safe.bat`.

Deep runtime controls:
- disable for max FPS: `python -u .\src\main.py run --profile smoothhd --no-deep-runtime`
- custom deep files: `--deep-model`, `--deep-labels`, `--deep-preprocess`, `--deep-metadata`

## 6) Realtime keyboard controls

Inside camera window:
- `q` or `Esc` -> quit
- `v` -> voice on/off
- `a` -> auto-speak on/off
- `t` -> continuous sentence mode on/off
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

### Step 2: train models

Before training, validate your CSV quickly:

```powershell
python -u .\src\main.py check-dataset --dataset .\data\processed\dataset.csv
```

Frame model (AutoML):

```powershell
python -u .\src\main.py train --automl
```

If some labels have very few samples, training now drops them automatically
(default `--min-samples-per-label 5`) to reduce noisy predictions.

Deep model (TensorFlow):

```powershell
python -m pip install -r requirements-deep.txt
python -u .\src\main.py train-deep --dataset .\data\processed\dataset.csv
```

Full one-command pipeline (recommended):

```powershell
python -m pip install -r requirements-deep.txt
python -u .\src\main.py train-all --dataset .\data\processed\dataset.csv
```

`train-all` does all of this in one run:
- train frame AutoML model
- train deep TensorFlow model
- build sequence dataset
- train temporal model
- write summary JSON (`models/train_all_summary.json`)

### Step 3: run

```powershell
python -u .\src\main.py run --mode hybrid
```

Continuous sentence mode (auto-build + auto-speak on pause):

```powershell
python -u .\src\main.py run --mode hybrid --continuous-sentence --sentence-pause-sec 2.5
```

Calibration wizard (recommended before important demos):

```powershell
python -u .\src\main.py calibrate --seconds 20
```

This saves `data/processed/calibration_profile.json`, and `run` applies it automatically.

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

One-step teach flow for a new sign (collect + retrain):

```powershell
python -u .\src\main.py teach-sign --label hello --phrase "Hello" --samples 180
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

- Full QA validation benchmark (recommended before demo/release):
  - `python -u .\src\main.py validate-all`
- Final gate test (dataset + QA + final markdown/json report):
  - `python -u .\src\main.py final-test`
- Train full model stack (frame + deep + temporal):
  - `python -u .\src\main.py train-all`
- Train deep model only:
  - `python -u .\src\main.py train-deep`
- Teach one sign and retrain in one flow:
  - `python -u .\src\main.py teach-sign --label hello --phrase "Hello" --samples 180`
- Validate dataset before training:
  - `python -u .\src\main.py check-dataset --dataset .\data\processed\dataset.csv`
- Health check:
  - `python -u .\src\main.py doctor`
  - or double-click `run_doctor.bat`
- Calibration wizard:
  - `python -u .\src\main.py calibrate --seconds 20`
- Production preflight:
  - `python -u .\src\main.py preflight --mode hybrid`
  - or double-click `run_preflight.bat`
- Performance benchmark:
  - `python -u .\src\main.py benchmark`
- Session report:
  - `python -u .\src\main.py report`
- Model artifact report:
  - `python -u .\src\main.py model-report`
- Offline video inference:
  - `python -u .\src\main.py infer-video --input .\data\raw\demo.mp4`
- Full CLI help:
  - `python -u .\src\main.py -h`
- End-to-end training guide:
  - `how_to_train_your_model.txt`
- Notebook starter:
  - `notebooks/train_from_scratch.ipynb`
- One-click training scripts:
  - `run_check_dataset.bat`
  - `run_train_deep.bat`
  - `run_train_all.bat`
  - `run_validate_all.bat`
  - `run_train_and_validate.bat`
  - `run_final_test.bat`

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

Security protections are enabled for imports:
- blocks path-traversal ZIP members
- blocks local/private-host URLs
- enforces safe file and size limits while extracting/downloading

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

## 17) Security defaults

- Crash logs automatically redact likely secret CLI values.
- Session CSV logs sanitize formula-like text to prevent spreadsheet formula injection.
- Dataset ZIP import blocks unsafe paths and private/local URLs by default.
- Security review report: `security_best_practices_report.md`
