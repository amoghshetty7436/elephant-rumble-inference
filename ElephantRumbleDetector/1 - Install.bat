@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Elephant rumble detector - install
echo.
echo   ELEPHANT RUMBLE DETECTOR - INSTALL (only needed once per computer)
echo   =================================================================
echo   This takes 5 to 20 minutes and needs the internet. Leave this window open.
echo.

rem ---------------------------------------------------------------- 1. Python
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
echo     4. Double-click "1 - Install.bat" again.
echo.
pause
exit /b 1

:havepython
echo   [1/4] Python found: %PY%

rem ---------------------------------------------------------------- 2. private Python environment
if exist ".venv\Scripts\python.exe" goto havevenv
echo   [2/4] Making a private Python environment in this folder...
%PY% -m venv .venv
if errorlevel 1 goto failed
:havevenv
set "VPY=%~dp0.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
if errorlevel 1 goto failed

rem ---------------------------------------------------------------- 3. PyTorch for this computer
set "GPU="
for /f "delims=" %%G in ('nvidia-smi --query-gpu=name --format=csv^,noheader 2^>nul') do if not defined GPU set "GPU=%%G"
set "TORCH=torch==2.4.1 torchaudio==2.4.1"
set "INDEX=https://download.pytorch.org/whl/cu121"
if not defined GPU (
  echo   [3/4] No NVIDIA graphics card found: installing the processor-only version.
  set "INDEX=https://download.pytorch.org/whl/cpu"
  goto installtorch
)
echo %GPU% | findstr /i /c:"RTX 50" /c:"RTX PRO" /c:"Blackwell" >nul
if not errorlevel 1 (
  set "TORCH=torch==2.7.1 torchaudio==2.7.1"
  set "INDEX=https://download.pytorch.org/whl/cu128"
)
echo   [3/4] Graphics card found: %GPU%. Installing the graphics-card version (about 2.5 GB)...
:installtorch
"%VPY%" -m pip install %TORCH% --index-url %INDEX% --disable-pip-version-check
if errorlevel 1 goto failed
"%VPY%" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 goto failed

rem ---------------------------------------------------------------- 4. AVES and a test
echo   [4/4] Downloading AVES (378 MB, once) and testing the detector...
if exist "use_cpu.txt" del "use_cpu.txt"
"%VPY%" rumble_detector.py setup
if not errorlevel 1 goto done
if not defined GPU goto failed
echo.
echo   The graphics card did not pass the test. Trying the processor instead...
"%VPY%" rumble_detector.py setup --device cpu
if errorlevel 1 goto failed
echo The graphics card failed the installer's test, so the detector uses the processor. Delete this file to try the graphics card again.> "use_cpu.txt"
echo   The detector will use the processor on this computer (slower, but it works).

:done
echo.
echo   INSTALLED. To find rumbles, double-click "2 - Find rumbles.bat".
echo.
pause
exit /b 0

:failed
echo.
echo   THE INSTALL DID NOT FINISH (see the red or error lines above).
echo   Check the internet connection and double-click "1 - Install.bat" again: it carries on.
echo   If it fails again, take a screenshot of this window and send it to Amogh.
echo.
pause
exit /b 1

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
