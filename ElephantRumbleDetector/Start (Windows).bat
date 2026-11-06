@echo off
setlocal EnableExtensions
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
title Elephant rumble detector
set "VPY=%BASE%.venv\Scripts\python.exe"
if exist "%BASE%.venv\installed.txt" goto run
rem an install from an older version of this package (no installed.txt, but everything is there)
if exist "%VPY%" if exist "%BASE%model\aves\aves-base-bio.torchaudio.pt" (
  "%VPY%" -c "import torch, torchaudio, soundfile, scipy" >nul 2>&1 && (echo old install> "%BASE%.venv\installed.txt" & goto run)
)

rem ---------------------------------------------------------------- first time: install
set "PY="
call :findpython
if defined PY goto havepython
echo   Python 3.11 is not installed. Installing it now...
winget install --id Python.Python.3.11 -e --scope user --accept-source-agreements --accept-package-agreements
call :findpython
if defined PY goto havepython
echo.
echo   COULD NOT INSTALL PYTHON AUTOMATICALLY.
echo   Please install it by hand:
echo     1. Open https://www.python.org/downloads/release/python-3119/
echo     2. Download "Windows installer (64-bit)" and run it.
echo     3. Tick "Add python.exe to PATH", then click "Install Now".
echo     4. Double-click "Start (Windows).bat" again.
echo.
pause
exit /b 1
:havepython
%PY% "%BASE%install.py"
if errorlevel 1 (
  echo.
  pause
  exit /b 1
)

:run
rem The window. If it can't open (tkinter missing), fall back to the questions in this black window.
"%VPY%" -c "import tkinter" >nul 2>&1
if errorlevel 1 goto text
start "" "%BASE%.venv\Scripts\pythonw.exe" "%BASE%rumble_gui.py" %*
exit /b 0
:text
"%VPY%" "%BASE%rumble_detector.py" %*
echo.
pause
exit /b 0

:findpython
for %%V in (3.11 3.12 3.10) do (
  if not defined PY (
    py -%%V --version >nul 2>&1 && set "PY=py -%%V"
  )
)
if defined PY exit /b 0
for %%P in ("%LOCALAPPDATA%\Programs\Python\Python311\python.exe" "%ProgramFiles%\Python311\python.exe" "%LOCALAPPDATA%\Programs\Python\Python312\python.exe") do (
  if not defined PY if exist %%P set "PY="%%~P""
)
exit /b 0
