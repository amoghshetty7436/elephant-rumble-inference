"""Tests of ElephantRumbleDetector/rumble_detector.py, the packaged detector, on tiny made-up
recordings with the fake embedder (no AVES download). The real model's classifier is used."""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest

PKG = os.path.join(os.path.dirname(__file__), "..", "ElephantRumbleDetector")
sys.path.insert(0, PKG)
torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")
import rumble_detector as rd  # noqa: E402

FAKE = ["--fake-embedder", "--min-score", "0", "--stall-minutes", "0"]


def tone(path, sr, minutes, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * 60 * minutes)) / sr
    y = 0.05 * rng.standard_normal(len(t)) + 0.3 * np.sin(2 * np.pi * 20 * t) * (np.sin(t / 7) > 0.5)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y.astype(np.float32), sr, subtype="PCM_16")


@pytest.fixture
def deployment(tmp_path):
    d = tmp_path / "nn_202306_jun_dep17"
    tone(str(d / "nn03g_202306_jun" / "nn03g_20230825_000000.wav"), 8000, 3)
    tone(str(d / "nn03g_202306_jun" / "nn03g_20230826_000000.wav"), 8000, 2, 1)
    tone(str(d / "dzan01a" / "dzan01a_20251201_000000.wav"), 32000, 2, 2)
    return d


def read_table(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_whole_deployment(deployment, tmp_path):
    out = tmp_path / "results"
    assert rd.main(["run", str(deployment), "--out", str(out)] + FAKE) == 0
    site = out / "nn03g_202306_jun"  # one input: no extra level with the input's name
    assert (site / "nn03g_20230825_000000.rumbles.txt").exists()
    assert (site / rd.FOLDER_TABLE).exists() and (out / "dzan01a" / rd.FOLDER_TABLE).exists()
    probs = np.load(site / "nn03g_20230825_000000.probs.npy")
    assert abs(len(probs) - 3 * 60 / 0.64) <= 2
    probs32 = np.load(out / "dzan01a" / "dzan01a_20251201_000000.probs.npy")
    assert abs(len(probs32) - 2 * 60 / 0.64) <= 2  # 32 kHz works the same
    summary = (out / "summary.md").read_text()
    assert "**3 of 3 recordings done**" in summary
    assert (out / "ABOUT THESE RESULTS.txt").exists() and not (out / "failed.txt").exists()
    folder = read_table(site / rd.FOLDER_TABLE)
    assert len(folder) == sum(len(read_table(site / f"nn03g_2023082{k}_000000.rumbles.txt")) for k in (5, 6))
    info = json.load(open(out / "detector.json"))
    assert info["cut_off"] == 0.993 and info["rule"]["score"] == "max"
    # running again skips what is done
    rd.main(["run", str(deployment), "--out", str(out)] + FAKE)
    assert len(list(csv.DictReader(open(out / "timings.csv")))) == 3
    # the cut-off can be changed afterwards
    rd.main(["table", "--out", str(out), "--min-score", "1.01"])
    assert read_table(site / rd.FOLDER_TABLE) == []
    rd.main(["table", "--out", str(out), "--min-score", "0"])
    assert len(read_table(site / rd.FOLDER_TABLE)) == len(folder)


def test_one_site_and_one_file(deployment, tmp_path):
    out = tmp_path / "site_results"
    rd.main(["run", str(deployment / "nn03g_202306_jun"), "--out", str(out)] + FAKE)
    assert (out / "nn03g_20230825_000000.rumbles.txt").exists() and (out / rd.FOLDER_TABLE).exists()
    out = tmp_path / "file_results"
    wav = deployment / "dzan01a" / "dzan01a_20251201_000000.wav"
    rd.main(["run", str(wav), "--out", str(out)] + FAKE)
    assert (out / "dzan01a_20251201_000000.rumbles.txt").exists()
    assert "**1 of 1 recordings done**" in (out / "summary.md").read_text()


def test_several_inputs_keep_their_names(deployment, tmp_path):
    other = tmp_path / "dz_202512_dec_dep10"
    tone(str(other / "dzan02b" / "dzan02b_20251203_000000.wav"), 32000, 1)
    out = tmp_path / "both"
    rd.main(["run", str(deployment), str(other), "--out", str(out)] + FAKE)
    assert (out / "nn_202306_jun_dep17" / "dzan01a" / "dzan01a_20251201_000000.rumbles.txt").exists()
    assert (out / "dz_202512_dec_dep10" / "dzan02b" / "dzan02b_20251203_000000.rumbles.txt").exists()


def test_bad_files_are_reported_not_fatal(deployment, tmp_path):
    site = deployment / "nn03g_202306_jun"
    (site / "nn03g_20230827_000000.wav").write_bytes(b"")  # 0 bytes
    (site / "nn03g_20230828_000000.wav").write_bytes(b"RIFF\x00\x00garbage" * 3)  # damaged
    sf.write(site / "nn03g_20230829_000000.wav", np.zeros(0, dtype=np.int16), 8000, subtype="PCM_16")  # header only
    sf.write(site / "nn03g_20230830_000000.wav", np.zeros(4000, dtype=np.float32), 8000)  # half a second
    (site / "notes.txt").write_text("not audio")
    out = tmp_path / "results"
    assert rd.main(["run", str(deployment), "--out", str(out)] + FAKE) == 0
    failed = (out / "failed.txt").read_text()
    assert "nn03g_20230827_000000.wav" in failed and "empty (0 bytes)" in failed
    assert "nn03g_20230828_000000.wav" in failed and "not a readable sound file" in failed
    assert "20230829" not in failed and "20230830" not in failed
    for k in (29, 30):  # too short: an empty table, counted as done
        assert read_table(out / "nn03g_202306_jun" / f"nn03g_202308{k}_000000.rumbles.txt") == []
    summary = (out / "summary.md").read_text()
    assert "**5 of 7 recordings done**" in summary and "**2 recordings could not be read**" in summary
    assert "2 recordings were too short" in summary
    assert (out / "nn03g_202306_jun" / rd.FOLDER_TABLE).exists()


def test_file_that_vanishes_is_reported(deployment, tmp_path, monkeypatch):
    out = tmp_path / "results"
    gone = deployment / "dzan01a" / "dzan01a_20251201_000000.wav"
    real = rd.load_audio

    def load(path):
        if os.path.basename(path) == gone.name:
            os.remove(path)
        return real(path)

    monkeypatch.setattr(rd, "load_audio", load)
    rd.main(["run", str(deployment), "--out", str(out), "--prefetch", "1"] + FAKE)
    assert "no longer there" in (out / "failed.txt").read_text()
    assert "**2 of 3 recordings done**" in (out / "summary.md").read_text()


def test_drive_that_vanishes_stops_the_run(deployment, tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "DRIVE_WAIT_S", 0.2)
    wd = rd.Watchdog(0)
    assert rd.wait_for(str(deployment), wd, 0.2)
    assert not rd.wait_for(str(tmp_path / "unplugged"), wd, 0.2)


def test_missing_input_and_output_inside_input(deployment, tmp_path):
    with pytest.raises(SystemExit, match="Can't find"):
        rd.main(["run", str(tmp_path / "nope"), "--out", str(tmp_path / "r")] + FAKE)
    with pytest.raises(SystemExit, match="inside the recordings"):
        rd.main(["run", str(deployment), "--out", str(deployment / "results")] + FAKE)


def test_unreadable_subfolder_warns(deployment, tmp_path, capsys):
    if os.name == "nt" or os.geteuid() == 0:
        pytest.skip("permissions can't be taken away here")
    locked = deployment / "locked"
    locked.mkdir()
    os.chmod(locked, 0)
    try:
        rd.main(["run", str(deployment), "--out", str(tmp_path / "r")] + FAKE)
    finally:
        os.chmod(locked, 0o755)
    assert "could not open the folder" in capsys.readouterr().out


def test_cores_and_cpu(deployment, tmp_path):
    """--cores limits PyTorch's threads; --device cpu works without a graphics card."""
    out = tmp_path / "r"
    code = subprocess.call([sys.executable, os.path.join(PKG, "rumble_detector.py"), "run", str(deployment),
                            "--out", str(out), "--cores", "1", "--device", "cpu"] + FAKE)
    assert code == 0
    m = json.load(open(out / "detector.json"))["machine"]
    assert m["cores_used"] == 1


def test_interactive_questions(deployment, tmp_path):
    """What a double-click does: paste the folder, Enter for the default results folder... here a typed one."""
    out = tmp_path / "asked"
    answers = f'"{deployment}"\n{out}\n2\n'
    env = dict(os.environ, RUMBLE_TEST_ARGS=" ".join(FAKE))
    r = subprocess.run([sys.executable, os.path.join(PKG, "rumble_detector.py")], input=answers, text=True,
                       capture_output=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FINISHED" in r.stdout
    assert "**3 of 3 recordings done**" in (out / "summary.md").read_text()


def test_alarm_rule_matches_the_study():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    im = pytest.importorskip("improve")
    rng = np.random.default_rng(0)
    rule = {"on": 0.2, "min_frames": 3, "smooth": 5, "gap": 3, "score": "max"}
    for _ in range(50):
        p = np.clip(rng.random(3000) ** rng.uniform(0.5, 8), 0, 1).astype(np.float32)
        for x, y in zip(im.alarms(p, rule), rd.alarms(p, rule)):
            assert np.array_equal(x, y)


def test_window_runs_a_deployment_and_changes_the_cut_off(deployment, tmp_path, monkeypatch):
    """The window (rumble_gui.py): choose, Start, wait, then change the cut-off. Needs a display."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    monkeypatch.setenv("RUMBLE_TEST_ARGS", " ".join(FAKE))
    import rumble_gui as g
    from tkinter import messagebox
    said = []
    monkeypatch.setattr(messagebox, "askyesno", lambda *a, **k: said.append(a[1]) or False)
    monkeypatch.setattr(messagebox, "showwarning", lambda *a, **k: said.append(a[1]))
    monkeypatch.setattr(messagebox, "showerror", lambda *a, **k: said.append(a[1]))
    app = g.App(root)
    app.start()
    assert said and "Choose the recordings" in said[-1] and app.worker is None
    app.inp.set(str(deployment))
    assert app.out.get().endswith(os.path.join("Rumble Results", "nn_202306_jun_dep17"))
    out = tmp_path / "window_results"
    app.out.set(str(out))
    app.out_edited = True

    def wait():
        import time
        t0 = time.time()
        while app.worker and time.time() - t0 < 120:
            root.update()
            time.sleep(0.05)
        root.update()

    app.start()
    wait()
    assert app.status.cget("text").startswith("Finished"), app.log.get("1.0", "end")
    assert str(app.bar.cget("value")) == str(app.bar.cget("maximum"))
    assert "**3 of 3 recordings done**" in (out / "summary.md").read_text()
    d = g.CutoffDialog(app)
    d.value.set("1.01")
    d.apply()
    assert "between 0 and 1" in said[-1]
    d.value.set("0.999")
    d.apply()
    wait()
    assert json.load(open(out / "detector.json"))["min_score_in_tables"] == 0.999
    root.destroy()
