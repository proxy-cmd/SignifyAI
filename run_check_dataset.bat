@echo off
setlocal
cd /d "%~dp0"
python -u .\src\main.py check-dataset --dataset .\data\processed\dataset.csv
endlocal

