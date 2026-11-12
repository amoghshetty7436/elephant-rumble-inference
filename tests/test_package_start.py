"""The ElephantRumbleDetector Start files and installer (the Linux one stands in for the Mac one)."""
import os
import shutil
import subprocess
import sys

import pytest

PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ElephantRumbleDetector")
SH, MAC, WIN = (os.path.join(PKG, n) for n in ("Start (Linux).sh", "Start (Mac).command", "Start (Windows).bat"))


def test_mac_and_linux_start_files_are_the_same_and_runnable():
    assert open(SH, "rb").read() == open(MAC, "rb").read()
    assert b"\r\n" not in open(SH, "rb").read()
    mode = subprocess.run(["git", "ls-files", "-s", SH, MAC], capture_output=True, text=True, cwd=PKG).stdout
    assert mode.count("100755") == 2, mode


def test_windows_start_file_has_windows_line_ends_in_the_zip():
    attrs = open(os.path.join(PKG, ".gitattributes")).read()
    assert "*.bat text eol=crlf" in attrs and "*.command text eol=lf" in attrs


def test_start_refuses_to_run_from_inside_a_zip(tmp_path):
    shutil.copy(SH, tmp_path / "Start (Linux).sh")
    r = subprocess.run(["bash", str(tmp_path / "Start (Linux).sh")], input="\n", capture_output=True, text=True)
    assert r.returncode == 1 and "STILL INSIDE THE ZIP" in r.stdout


@pytest.mark.skipif(not os.path.exists(os.path.join(PKG, "model", "aves", "aves-base-bio.torchaudio.pt")),
                    reason="needs the AVES files in model/aves")
def test_install_then_start_opens_the_window_without_installing_again(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("tkinter")
    d = tmp_path / "ERD"
    shutil.copytree(PKG, d, ignore=shutil.ignore_patterns(".venv", "__pycache__", "use_cpu.txt"))
    env = dict(os.environ, RUMBLE_INSTALL_TEST="1")
    r = subprocess.run([sys.executable, str(d / "install.py")], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALL GOOD" in r.stdout and "INSTALLED" in r.stdout
    assert (d / ".venv" / "installed.txt").exists()
    # second start: straight to the window (stopped after a few seconds)
    try:
        r = subprocess.run(["bash", str(d / "Start (Linux).sh")], capture_output=True, text=True, timeout=8,
                           env=dict(os.environ, DISPLAY=os.environ.get("DISPLAY", "")))
        out = r.stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    assert "INSTALL" not in out
