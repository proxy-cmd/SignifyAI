@echo off
cd /d "%~dp0"

echo Running preflight checks...
python -u .\src\main.py preflight --mode hybrid
if errorlevel 1 (
  echo.
  echo Preflight failed. Fix issues above, then retry.
  pause
  exit /b 1
)

echo.
echo Starting SignifyAI in enterprise profile...
python -u .\src\main.py run --profile enterprise
pause
