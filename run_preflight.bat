@echo off
cd /d "%~dp0"
python -u .\src\main.py preflight --mode hybrid
pause
