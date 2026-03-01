@echo off
setlocal
cd /d "%~dp0"
python -u .\src\main.py train-deep --dataset .\data\processed\dataset.csv
endlocal

