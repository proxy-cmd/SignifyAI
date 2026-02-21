# SignifyAI

Webcam-based sign-to-speech prototype for hackathon demos.

## Quick start (recommended)

```powershell
python -u .\app.py
```

This opens a simple menu.

## One-command demo

```powershell
python -u .\src\stage_demo.py
```

Best for presentation day.

## Main run (normal)

```powershell
python -u .\src\main.py
```

If ML model is missing, it falls back to rules mode automatically.

## Keyboard controls (inside camera window)

- `q` or `Esc`: quit
- `v`: voice on/off
- `a`: auto-speak on/off
- `m`: switch mode
- `space`: add current word to sentence
- `enter`: speak sentence
- `c`: clear sentence
- `p`: screenshot
- `k`: start/stop recording

## Basic train flow (optional)

1. Collect:

```powershell
python -u .\src\main.py collect --label hello --samples 250
python -u .\src\main.py collect --label thanks --samples 250
```

2. Train:

```powershell
python -u .\src\main.py train --automl
```

## Temporal mode (stronger continuous behavior)

```powershell
python -u .\src\main.py train-production
python -u .\src\main.py run --mode temporal
```

## Useful extras

- Session report:
  - `python -u .\src\main.py report`
- Health check:
  - `python -u .\src\main.py doctor`
- Performance benchmark:
  - `python -u .\src\main.py benchmark`
- Offline video inference:
  - `python -u .\src\main.py infer-video --input .\data\raw\demo.mp4`

## If Python gets stuck

```powershell
taskkill /F /IM python.exe
```

## Advanced docs

- `docs/RESEARCH_ROADMAP.md`
- `docs/INDUSTRY_NOTES.md`
- `python -u .\src\main.py -h`

