"""Installs the elephant rumble detector on this computer (Windows, Mac or Linux).

The Start file runs this the first time. It makes a private Python environment in the
.venv folder here, picks the PyTorch build for this computer (NVIDIA graphics card or
processor only), installs the other packages, downloads AVES and tests the detector.
Running it again carries on where it stopped.
"""
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
DONE = os.path.join(VENV, "installed.txt")  # the Start files check for this
USE_CPU = os.path.join(HERE, "use_cpu.txt")
SYSTEM = {"Windows": "Windows", "Darwin": "Mac"}.get(platform.system(), platform.system())
TEST = os.environ.get("RUMBLE_INSTALL_TEST")  # tests: reuse this Python's packages, no downloads


def say(text=""):
    print("  " + text if text else "", flush=True)


def venv_python():
    return os.path.join(VENV, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV, "bin", "python")


def run(cmd):
    say("> " + " ".join(cmd[2:] if cmd[1:2] == ["-m"] else cmd[1:]))
    return subprocess.call(cmd, cwd=HERE)


def nvidia_card():
    if SYSTEM == "Mac" or not shutil.which("nvidia-smi"):
        return ""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip().splitlines()[0].strip() if r.returncode == 0 and r.stdout.strip() else ""
    except Exception:  # noqa: BLE001
        return ""


def torch_choice():
    """(packages, index url or None for the normal package index, what to tell the person)."""
    if SYSTEM == "Mac":
        if platform.machine() == "arm64":
            return ["torch==2.4.1", "torchaudio==2.4.1"], None, "Mac (Apple processor): the processor version"
        # PyTorch stopped making new versions for Intel Macs after 2.2
        return ["torch==2.2.2", "torchaudio==2.2.2"], None, "Mac (Intel processor): the processor version"
    card = nvidia_card()
    if not card:
        return (["torch==2.4.1", "torchaudio==2.4.1"], "https://download.pytorch.org/whl/cpu",
                "no NVIDIA graphics card found: the processor-only version")
    if any(k in card.upper() for k in ("RTX 50", "RTX PRO", "BLACKWELL")):
        return (["torch==2.7.1", "torchaudio==2.7.1"], "https://download.pytorch.org/whl/cu128",
                f"graphics card found ({card}): the graphics-card version (about 2.5 GB)")
    return (["torch==2.4.1", "torchaudio==2.4.1"], "https://download.pytorch.org/whl/cu121",
            f"graphics card found ({card}): the graphics-card version (about 2.5 GB)")


def failed(why):
    say()
    say("THE INSTALL DID NOT FINISH: " + why)
    say("Check the internet connection and double-click the Start file again: it carries on.")
    say("If it fails again, take a screenshot of this window and send it to Amogh.")
    return 1


def main():
    say()
    say(f"ELEPHANT RUMBLE DETECTOR - INSTALL on {SYSTEM} (only needed once per computer)")
    say("=" * 70)
    say("This takes 5 to 20 minutes and needs the internet. Leave this window open.")
    say()
    say(f"[1/4] Python {platform.python_version()} found: {sys.executable}")
    if sys.version_info < (3, 9):
        return failed("this Python is too old; version 3.10, 3.11 or 3.12 is needed.")

    py = venv_python()
    if not os.path.exists(py):
        say("[2/4] Making a private Python environment in this folder...")
        cmd = [sys.executable, "-m", "venv", VENV] + (["--system-site-packages"] if TEST else [])
        if subprocess.call(cmd) != 0 or not os.path.exists(py):
            shutil.rmtree(VENV, ignore_errors=True)
            if SYSTEM == "Linux":
                say("On Ubuntu or Debian, first run:  sudo apt install python3-venv python3-tk")
            return failed("could not make the Python environment.")
    else:
        say("[2/4] The private Python environment is already there.")
    pip = [py, "-m", "pip", "--disable-pip-version-check"]
    if not TEST:
        if run(pip + ["install", "--upgrade", "pip", "--quiet"]) != 0:
            return failed("could not update pip.")

    packages, index, what = torch_choice()
    say(f"[3/4] PyTorch for this computer - {what}")
    if not TEST:
        ok = run(pip + ["install"] + packages + (["--index-url", index] if index else [])) == 0
        if not ok and index:  # the PyTorch site can be blocked by a firewall: try the normal index
            say("That did not work; trying the normal Python package index instead...")
            ok = run(pip + ["install"] + packages) == 0
        if not ok:
            return failed("could not install PyTorch.")
        if run(pip + ["install", "-r", os.path.join(HERE, "requirements.txt")]) != 0:
            return failed("could not install the other packages.")

    say("[4/4] Downloading AVES (378 MB, once) and testing the detector...")
    if os.path.exists(USE_CPU):
        os.remove(USE_CPU)
    det = [py, os.path.join(HERE, "rumble_detector.py"), "setup"]
    if subprocess.call(det, cwd=HERE) != 0:
        if "graphics card found" not in what:
            return failed("the detector test did not pass.")
        say()
        say("The graphics card did not pass the test. Trying the processor instead...")
        if subprocess.call(det + ["--device", "cpu"], cwd=HERE) != 0:
            return failed("the detector test did not pass.")
        with open(USE_CPU, "w") as f:
            f.write("The graphics card failed the installer's test, so the detector uses the processor. "
                    "Delete this file to try the graphics card again.\n")
        say("The detector will use the processor on this computer (slower, but it works).")

    with open(DONE, "w") as f:
        f.write(f"{SYSTEM} {platform.machine()} python {platform.python_version()} {' '.join(packages)}\n")
    say()
    say("INSTALLED. The detector window opens next; after this, the Start file opens it straight away.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
