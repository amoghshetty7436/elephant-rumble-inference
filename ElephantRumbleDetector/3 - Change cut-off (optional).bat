@echo off
setlocal
set "BASE=%~dp0"
cd /d "%BASE%"
if exist "%BASE%rumble_detector.py" goto unzipped
echo.
echo   THE FILES ARE STILL INSIDE THE ZIP.
echo   Close this window. Right-click ElephantRumbleDetector.zip, choose "Extract All...",
echo   open the extracted ElephantRumbleDetector folder, and double-click this file there.
echo.
pause
exit /b 1
:unzipped
title Elephant rumble detector - change the cut-off
if exist "%BASE%.venv\Scripts\python.exe" goto run
echo   Double-click "1 - Install.bat" first.
pause
exit /b 1
:run
"%BASE%.venv\Scripts\python.exe" "%BASE%rumble_detector.py" cutoff
echo.
pause
