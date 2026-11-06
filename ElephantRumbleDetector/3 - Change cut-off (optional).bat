@echo off
setlocal
cd /d "%~dp0"
title Elephant rumble detector - change the cut-off
if exist ".venv\Scripts\python.exe" goto run
echo   Double-click "1 - Install.bat" first.
pause
exit /b 1
:run
".venv\Scripts\python.exe" rumble_detector.py cutoff
echo.
pause
