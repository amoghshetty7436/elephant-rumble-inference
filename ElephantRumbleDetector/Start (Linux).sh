#!/bin/bash
# Elephant rumble detector: double-click this file (Mac), or run   bash "Start (Linux).sh"   (Linux).
# The first time it installs the detector; after that it opens the window straight away.
cd "$(dirname "$0")" || exit 1
BASE="$(pwd)"
pause() { echo; read -r -p "  Press Enter to close this window. " _; }

if [ ! -f "$BASE/rumble_detector.py" ]; then
  echo; echo "  THE FILES ARE STILL INSIDE THE ZIP. Unzip ElephantRumbleDetector.zip first,"
  echo "  then open the ElephantRumbleDetector folder and start this file there."; pause; exit 1
fi
VPY="$BASE/.venv/bin/python"

if [ ! -f "$BASE/.venv/installed.txt" ]; then
  PY=""
  for c in python3.11 python3.12 python3.10 /usr/local/bin/python3 /opt/homebrew/bin/python3 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys, tkinter; sys.exit(not (3, 9) <= sys.version_info[:2] <= (3, 12))' >/dev/null 2>&1; then
      PY="$(command -v "$c")"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo
    echo "  PYTHON 3.11 IS NEEDED FIRST (versions 3.10 to 3.12 work)."
    if [ "$(uname)" = "Darwin" ]; then
      echo "    1. In the web page that opens, download the \"macOS 64-bit universal2 installer\"."
      echo "    2. Open it and click through the installer."
      echo "    3. Double-click \"Start (Mac).command\" again."
      open "https://www.python.org/downloads/release/python-3119/" 2>/dev/null
    else
      echo "    On Ubuntu or Debian:  sudo apt install python3 python3-venv python3-tk"
      echo "    Then run this file again."
    fi
    pause; exit 1
  fi
  "$PY" "$BASE/install.py" || { pause; exit 1; }
fi

if "$VPY" -c "import tkinter" >/dev/null 2>&1; then
  echo "  The detector window is open. Leave this window open while it runs (you can minimise it)."
  "$VPY" "$BASE/rumble_gui.py" "$@"
else
  "$VPY" "$BASE/rumble_detector.py" "$@"
  pause
fi
