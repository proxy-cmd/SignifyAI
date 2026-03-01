# SignifyAI Lite Demo (Hackathon Mode)

This is a separate, simplified demo app.

## Run

```powershell
python -u .\lite_demo\run_lite.py
```

Or double-click:
- `run_lite_demo.bat`

## Why this exists

Main project became advanced.  
This folder gives a cleaner experience for hackathon demos:
- rules-first detection
- stable speaking logic
- simple UI
- 20 demo sign labels shown on screen

## Controls

- `q` / `Esc` : quit
- `v` : voice on/off
- `r` : reset speech memory (allows re-speaking same sign quickly)

## 20 demo labels shown

1. HELLO  
2. YES  
3. NO  
4. STOP  
5. ONE  
6. TWO  
7. THREE  
8. FOUR  
9. FIVE  
10. PEACE  
11. OKAY  
12. CALL ME  
13. ROCK  
14. I LOVE YOU  
15. THANK YOU  
16. HELP  
17. GOOD MORNING  
18. GOOD AFTERNOON  
19. GOOD EVENING  
20. GOOD NIGHT

> Note: time-based greeting labels are selected automatically by current system time when the two-hand open-palm pattern is detected.

## How to do confusing signs (quick guide)

- `GOOD MORNING / AFTERNOON / EVENING / NIGHT`:
  - Show **2 open palms**.
  - Keep both hands at similar height.
  - Keep them **apart** (not too close).
  - Hold for ~1 second.
  - Exact label depends on your PC time.

- `THANK YOU`:
  - Same 2 open palms, but keep hands **close together**.

- `OKAY`:
  - Touch **thumb tip + index tip** (make ring).
  - Keep other 3 fingers up.
  - Hold steady for 0.5-1 sec.

- `YES` / `NO`:
  - Use thumbs up/down with other fingers folded.
  - Thumb up = `YES`, thumb down = `NO`.

- `ROCK` / `I LOVE YOU`:
  - Index+pinky up, middle+ring down.
  - Thumb out clearly = `I LOVE YOU`.
  - Thumb tucked/near palm = `ROCK`.

## Better accuracy tips

- Use front lighting.
- Keep hand fully in frame.
- Do one sign at a time.
- Pause briefly between signs.
