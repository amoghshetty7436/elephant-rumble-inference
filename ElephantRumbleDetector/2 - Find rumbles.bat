@echo off
setlocal
cd /d "%~dp0"
title Elephant rumble detector
if exist ".venv\Scripts\python.exe" goto run
echo.
echo   The detector is not installed on this computer yet.
echo   Double-click "1 - Install.bat" first.
echo.
pause
exit /b 1
:run
".venv\Scripts\python.exe" rumble_detector.py %*
echo.
pause
