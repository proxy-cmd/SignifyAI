# SignifyAI

SignifyAI is a real-time **sign-to-speech** project built for practical use in demos, hackathons, and assistive communication scenarios.

It is designed to be:
- fast on normal laptops
- easy to teach new signs live
- stable in real-time (less flicker, less false triggers)

## What It Does

- Reads hand and eye landmarks from webcam input.
- Detects signs and speaks the matched message.
- Supports emergency communication modes.
- Lets users teach custom signs without full retraining.
- Saves taught signs for future sessions.

## Main Modes

- `1` Realtime translator (`default`)
- `2` Emergency hand mode (`aid`)
- `3` Eye assist mode (`eye`)
- `4` Fast teach mode (`teach`)
- `5` Manage taught signs

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

- Reduced realtime label flicker.
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
python -u src/main.py run --mode default
python -u src/main.py run --mode aid
python -u src/main.py run --mode eye
python -u src/main.py run --mode teach
python -m pytest -q
```

## Data Notes

- Taught sign prototypes are saved in `data/models/sign_prototypes.json`.
- Teach trace files are optional and saved only when `--save-teach-data` is enabled.
- Heavy datasets/models are not fully tracked by default to keep the repo lightweight.

See [data/README.md](data/README.md) for data layout details.

## For Demo / Exhibition

Use this quick presentation flow:

1. Show realtime translation.
2. Teach one new sign live and detect it immediately.
3. Show emergency hand mode.
4. Show eye mode as no-hand fallback.

Core message:
- this is not just a static model demo
- it is a usable, adaptive, real-time communication system
