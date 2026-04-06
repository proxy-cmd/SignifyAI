# SignifyAI

SignifyAI is a patient-focused **emergency assist communication** project built for practical use in demos, hackathons, and assistive care scenarios.

It is designed to be:
- fast on normal laptops
- easy to teach new signs live
- stable in live sessions (less flicker, less false triggers)

## What It Does

- Reads hand and eye landmarks from webcam input.
- Detects signs and speaks the matched message.
- Supports emergency communication modes.
- Lets users teach custom signs without full retraining.
- Saves taught signs for future sessions.

## Main Modes

- `1` Emergency hand mode (`aid`)
- `2` Eye assist mode (`eye`)
- `3` Emergency sign teach mode (runtime teach in `aid`)
- `4` Manage taught signs

## What Makes It Unique

- **Live teach in runtime**: add a sign and use it immediately.
- **Adaptive sign memory**: custom signs are stored as reusable prototypes.
- **Confidence guard**: avoids speaking low-confidence wrong predictions.
- **Anti-confusion fixes**: reduces random jumps between similar signs.
- **Location-aware matching**: can separate similar handshapes by context (for example face vs head position).
- **Emergency-first reliability**: dedicated hand and eye emergency flows.

## How It Works (Simple Flow)

1. Camera frame is captured.
2. Landmarks are extracted.
3. Signs are matched using rules + adaptive prototype matching.
4. Low-confidence output is filtered.
5. Final label is shown and spoken.

## Recent Stability Improvements

- Reduced live label flicker.
- Reduced false landmark spikes (random neck/face glitches).
- Better behavior under lighting/background variation.
- Better eye-mode blink and hold stability.
- Better separation of similar custom signs.

## Quick Setup

### 1) Create virtual environment

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

### 3) Run

```bash
python -u src/main.py
```

## Useful Commands

```bash
python -u src/main.py run --mode aid
python -u src/main.py run --mode eye
python -m pytest -q
```

## Deploy Backend (Free Tier Friendly)

You can deploy the backend bridge (`src/web_bridge.py`) with the included configs:

- `Dockerfile`
- `render.yaml`
- `railway.json`

### Option A: Render

1. Push this repo to GitHub.
2. In Render, create a new **Web Service** from this repo.
3. Render will detect `render.yaml` and deploy.
4. After deploy, test:
   - `https://<your-service>.onrender.com/api/health`

### Option B: Railway

1. Push this repo to GitHub.
2. In Railway, create a project from the repo.
3. Railway uses `railway.json` + `Dockerfile`.
4. After deploy, test:
   - `https://<your-service>.up.railway.app/api/health`

### Connect Hosted Backend To UI

If your UI is hosted separately, set `ui/bridge-port.json` like:

```json
{
  "baseUrl": "https://<your-backend-domain>"
}
```

The UI script auto-reads this and talks to that backend.

### Important Camera Note

Bridge now supports two frame sources:

- `camera`: backend reads webcam on the machine where backend runs.
- `browser`: frontend captures browser webcam and uploads frames to backend (`/api/frame/upload`).

For hosted demos (Render/Railway), use `browser` frame source so each viewer's laptop camera is used.

## Data Notes

- Taught sign prototypes are saved in `data/models/sign_prototypes.json`.
- Teach trace files are optional and saved only when `--save-teach-data` is enabled.
- Heavy datasets/models are not fully tracked by default to keep the repo lightweight.

See [data/README.md](data/README.md) for data layout details.

## For Demo / Exhibition

Use this quick presentation flow:

1. Show emergency hand mode.
2. Teach one new emergency sign live and detect it immediately.
3. Show eye mode as no-hand fallback.

Core message:
- this is not just a static model demo
- it is a usable, adaptive, real-time communication system
