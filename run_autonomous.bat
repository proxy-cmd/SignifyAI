@echo off
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File ".\scripts\autonomous_pipeline.ps1"
pause

