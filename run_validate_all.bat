@echo off
setlocal
cd /d "%~dp0"
python -u .\src\main.py validate-all --no-pytest --out .\data\processed\qa_validation_report.json
endlocal

