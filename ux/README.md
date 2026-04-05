# SignifyAI Desktop UX (Tkinter)

This folder contains a simple desktop wizard UI for demos, connected to the backend bridge.

## Features

- 5 modes:
  - Realtime Translation
  - Emergency Hand
  - Eye Assist
  - Quick Record
  - Manage Signs
- Session controls (`Start`, `Stop`, `Toggle Speech`)
- Intent + confidence display
- Real taught-sign management (add / modify / delete) via `data/models/sign_prototypes.json`
- Activity log panel
- Live backend sync from `src/web_bridge.py` (`/api/state` polling)
- Buttons call real backend APIs (`start/stop`, `mode`, `voice`, `teach`, `aid`, `eye`)

## Run

From project root:

1) Start backend bridge:

```powershell
python -u src/web_bridge.py
```

2) Run desktop UX:

```powershell
python -u ux/app.py
```

No extra package is required for this UX (Tkinter is part of standard Python).

## Note

This keeps the code simple and beginner-friendly while still being connected to live backend actions.
