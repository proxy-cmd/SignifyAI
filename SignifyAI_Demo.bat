@echo off
setlocal

set "BASE_DIR=%~dp0"
set "LAUNCHER=%BASE_DIR%launch_demo.py"

if not exist "%LAUNCHER%" (
  if exist "D:\SignifyAI\launch_demo.py" (
    set "BASE_DIR=D:\SignifyAI\"
    set "LAUNCHER=D:\SignifyAI\launch_demo.py"
  )
)

if not exist "%LAUNCHER%" (
  echo Could not find launch_demo.py
  echo Expected at:
  echo   %~dp0launch_demo.py
  echo or
  echo   D:\SignifyAI\launch_demo.py
  pause
  exit /b 2
)

cd /d "%BASE_DIR%"

set "PYEXE=%BASE_DIR%.venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

echo Starting SignifyAI demo...
"%PYEXE%" "%LAUNCHER%"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Demo launcher ended with error code %EXIT_CODE%.
  echo You may close this window now...
  pause
)

endlocal
