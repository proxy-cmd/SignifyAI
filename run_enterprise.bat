@echo off
cd /d "%~dp0"
python -u .\src\main.py run --profile enterprise
pause
