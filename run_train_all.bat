@echo off
setlocal
cd /d "%~dp0"
python -u .\src\main.py train-all --dataset .\data\processed\dataset.csv
endlocal

