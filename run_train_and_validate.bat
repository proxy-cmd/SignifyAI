@echo off
setlocal
cd /d "%~dp0"
python -u .\src\main.py check-dataset --dataset .\data\processed\dataset.csv
if errorlevel 1 goto :end
python -u .\src\main.py train-all --dataset .\data\processed\dataset.csv
if errorlevel 1 goto :end
python -u .\src\main.py validate-all --no-pytest --out .\data\processed\qa_validation_report.json
:end
endlocal

