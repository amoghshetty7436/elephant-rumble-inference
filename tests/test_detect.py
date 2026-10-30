"""Tests of tools/detect.py: survey, run and table on tiny recordings at 32 kHz and 8 kHz,
with the fake embedder (no AVES download)."""
import csv
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")
import experiments as ex  # noqa: E402
import detect  # noqa: E402


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "deployment"
    for site, sr, minutes in (("dzan01a", 32000, 3), ("dzan02b", 32000, 2), ("nn06a", 8000, 4)):
        d = data / site
        d.mkdir(parents=True)
        for k in range(2):
            t = np.arange(int(sr * 60 * minutes)) / sr
            y = 0.05 * rng.standard_normal(len(t)) + 0.3 * np.sin(2 * np.pi * 20 * t) * (np.sin(t / 7) > 0.5)
            sf.write(d / f"{site}_2025120{k + 1}_000000.wav", y.astype(np.float32), sr, subtype="PCM_16")
    exp = tmp_path / "_study" / "E2_ctx2_test"
    exp.mkdir(parents=True)
    torch.manual_seed(0)
    torch.save(ex.make_model(2).state_dict(), exp / "model_seed1.pth")
    (exp / "spec.json").write_text(json.dumps({"ctx": 2}))
    fd = tmp_path / "_study" / "final_detections"
    fd.mkdir()
    rule = dict(on=0.2, min_frames=3, smooth=5, gap=3, score="max")
    (fd / "final_model.json").write_text(json.dumps({
        "experiment": "E2_ctx2_test", "run": 1, "model": str(tmp_path / "elsewhere" / "model_seed1.pth"),
        "rule": rule, "cut_off": 0.0}))
    return data, fd / "final_model.json", tmp_path / "results"


def read_table(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_survey_run_table(setup):
    data, model, out = setup
    base = ["--out", str(out), "--model", str(model), "--fake-embedder"]
    detect.main(["survey", str(data)] + base)
    md = (out / "survey.md").read_text()
    assert "**6 recordings**, 3 sites" in md and "32000 Hz x 4" in md and "8000 Hz x 2" in md
    assert "Estimated time" in md and not (out / "_write_test").exists()

    detect.main(["run", str(data)] + base + ["--min-score", "0"])
    tables = sorted(out.rglob("*" + detect.TABLE_SUFFIX))
    assert len(tables) == 6
    assert (out / "deployment" / "nn06a" / "nn06a_20251201_000000.rumbles.txt").exists()
    probs = np.load(out / "deployment" / "nn06a" / "nn06a_20251201_000000.probs.npy")
    assert abs(len(probs) - 4 * 60 / 0.64) <= 2  # 8 kHz resampled to 500 Hz: one frame per 0.64 s
    probs32 = np.load(out / "deployment" / "dzan01a" / "dzan01a_20251201_000000.probs.npy")
    assert abs(len(probs32) - 3 * 60 / 0.64) <= 2  # the same for 32 kHz
    n_all = sum(len(read_table(t)) for t in tables)
    folder = read_table(out / "deployment" / "dzan01a" / detect.FOLDER_TABLE)
    if folder:
        assert {r["Begin File"] for r in folder} <= {"dzan01a_20251201_000000.wav", "dzan01a_20251202_000000.wav"}
        assert all(float(r["Begin Time (s)"]) <= float(r["End Time (s)"]) for r in folder)
    timings = list(csv.DictReader(open(out / "timings.csv")))
    assert len(timings) == 6 and {t["sample_rate"] for t in timings} == {"32000", "8000"}
    assert {t["site"] for t in timings} == {"dzan01a", "dzan02b", "nn06a"}
    summary = (out / "summary.md").read_text()
    assert "**6 of 6 recordings done**" in summary and "PROBLEMS" not in summary
    assert len(list(csv.DictReader(open(out / "summary.csv")))) == 3
    assert "[6/6]" in (out / "run_log.txt").read_text()
    assert json.load(open(out / "detector.json"))["machine"]["torch"]

    # a rerun skips everything; table rewrites with a stricter cut-off
    detect.main(["run", str(data)] + base)
    assert len(list(csv.DictReader(open(out / "timings.csv")))) == 6
    detect.main(["table", "--out", str(out), "--min-score", "1.01"])
    assert sum(len(read_table(t)) for t in tables) == 0
    assert json.load(open(out / "detector.json"))["min_score_in_tables"] == 1.01
    detect.main(["table", "--out", str(out), "--min-score", "0"])
    assert sum(len(read_table(t)) for t in tables) == n_all


def test_out_inside_input_is_refused(setup):
    data, model, _ = setup
    with pytest.raises(SystemExit):
        detect.main(["run", str(data), "--out", str(data / "results"), "--model", str(model), "--fake-embedder"])
