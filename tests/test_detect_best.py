"""Tests of tools/detect_best.py on the tiny fake Datasets tree: export round 2's best model,
check it against round 2's own scores, run it on a fake deployment limited to a Hori-Harm
table (a sample first, then the rest), and read the results with compare_hori_harm.py."""
import csv
import json
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("scipy")
import candidates as cd  # noqa: E402
import dataset_configs as dc  # noqa: E402
import detect_best as db  # noqa: E402
import round2 as r2  # noqa: E402
import study as st  # noqa: E402
from tests.test_dataset_configs import source  # noqa: E402,F401


def test_spread_takes_sites_in_turn():
    files = [("r", f"/d/{s}_2023070{i}_000000.wav") for s in ("nn01a", "nn02b", "nn03c") for i in range(4)]
    got = db.spread(files, 5)
    sites = [os.path.basename(p)[:5] for _, p in got]
    assert len(got) == 5 and {sites.count(s) for s in set(sites)} == {1, 2} and len(set(sites)) == 3
    assert db.spread(files, 50) == sorted(files, key=lambda x: x[1])


def test_parse_name():
    assert db.parse_name("late_ens+efficientnet_b0_x5") == ("late_ens", "efficientnet_b0")
    assert db.parse_name("aves_ens") == ("aves_ens", None)
    with pytest.raises(SystemExit):
        db.parse_name("fusion_ens")


def test_detect_best_end_to_end(source, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dc, "HELDOUT_FOREST_DAYS", 2)
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder", "--configs", "A", "B", "C", "D"])
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder", "--configs", "E", "--best", "A"])
    for k, v in dict(CHECK_EVERY=150, PATIENCE=2, MAX_CHECKS=4, HELDOUT_ONE_IN=3).items():
        monkeypatch.setattr(st, k, v)
    small = dict(on=[0.5, 0.3], min_frames=[2, 4], smooth=[1], gap=[0], score=["mean", "max"])
    monkeypatch.setattr(st, "ROUND1_RULES", small)
    monkeypatch.setattr(st.im, "RULE_OPTIONS", small)
    st.main(["--source", str(source), "--seeds", "2", "--fake-embedder", "--finish-by", "none",
             "--round1", "E2", "--start-from", "E2", "--no-augmentation"])
    for k, v in dict(VER_MAX_EPOCHS=2, VER_PATIENCE=1, VER_NEG_PER_DAY=10, VER_POS_PER_DAY=10).items():
        monkeypatch.setattr(cd, k, v)
    monkeypatch.setattr(r2, "NEG_PER_STAGE", 10)
    cd.main(["--source", str(source), "--seeds", "2", "--fake-embedder", "--only", "m4", "m6"])
    r2.main(["--source", str(source), "--seeds", "2", "--fake-embedder", "--archs", "efficientnet_b0",
             "--skip", "fusion", "dasheng06b", "dasheng_ft"])
    lab = db.open_round2(str(source), True, 2)
    lab.evaluate_checked("late_ens", "efficientnet_b0", True)   # in case it was not among the 3 best

    # export, with the check against round 2's scores from the cached fingerprints
    model = tmp_path / "model"
    checks = []
    monkeypatch.setattr(db, "check_against_round2", _recorder(checks))
    db.main(["export", "--source", str(source), "--seeds", "2", "--fake-embedder", "--model", str(model)])
    info = json.load(open(model / "model.json"))
    assert info["stage"] == "late_ens" and len(info["heads"]) == 4 and len(info["checkers"]) == 2
    assert 0 < info["cut_off"] <= 1
    (dp, ds), = checks
    assert dp < 1e-3 and ds < 1e-3, (dp, ds)

    # a fake deployment: some of the dataset's recordings in site folders
    dep = tmp_path / "dep17"
    wavs = []
    for p in ("sampleA", "sampleB"):
        for k in lab.pools[p].files:
            src = lab.pools[p].sounds[k]
            dst = dep / k[:5] / os.path.basename(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            wavs.append(dst)
    assert len(wavs) >= 4
    hh = tmp_path / "hh.txt"
    head = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)",
            "Begin Path", "File Offset (s)", "Begin File", "Score", "Count"]
    with open(hh, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(head)
        for i, wv in enumerate(wavs[:3]):
            w.writerow([i + 1, "Spectrogram 1", 1, 10, 15, 5, 250, str(wv), 10, wv.name, 0.9, 1 if i else 0])
        w.writerow([9, "Spectrogram 1", 1, 10, 15, 5, 250, str(wavs[3]), 10, wavs[3].name, 0.9, ""])  # not reviewed
    out = tmp_path / "out"
    args = ["run", str(dep), "--hh", str(hh), "--out", str(out), "--model", str(model), "--fake-embedder"]
    db.main(args + ["--sample", "2"])
    tables = sorted(out.rglob("*" + db.TABLE_SUFFIX))
    assert len(tables) == 2
    md = (out / "summary.md").read_text()
    assert "Dasheng" in md and "The other 1 recordings in the Hori-Harm table" in md
    db.main(args)                          # the rest; the sample is not redone
    tables = sorted(out.rglob("*" + db.TABLE_SUFFIX))
    assert len(tables) == 3 and len(list(csv.DictReader(open(out / "timings.csv")))) == 3
    for t in tables:
        base = str(t)[:-len(db.TABLE_SUFFIX)]
        ev = np.load(base + db.EVENTS_SUFFIX)
        assert ev.shape[1] == 4 and len(np.load(base + db.PROBS_SUFFIX))
        rows = list(csv.DictReader(open(t), delimiter="\t"))
        assert len(rows) == int((ev[:, 3] >= info["cut_off"]).sum())
    assert json.load(open(out / "detector.json"))["kind"] == "checked"
    folders = sorted(out.rglob(db.detect.FOLDER_TABLE))
    assert folders and sum(len(open(f).readlines()) - 1 for f in folders) == sum(
        len(open(t).readlines()) - 1 for t in tables)
    # another model's results folder is refused, so nothing is overwritten
    d = json.load(open(out / "detector.json"))
    d["experiment"] = "something else"
    json.dump(d, open(out / "detector.json", "w"))
    with pytest.raises(SystemExit):
        db.main(args)
    d["experiment"] = info["name"]
    json.dump(d, open(out / "detector.json", "w"))

    # compare_hori_harm reads the checked scores and the cut-off from detector.json
    import compare_hori_harm as ch
    rep = tmp_path / "rep"
    ch.main(["--hh", str(hh), "--ours", str(out / "dep17"), "--out", str(rep)])
    text = (rep / "report.md").read_text()
    assert "give back exactly" in text and f"{info['cut_off']:.3f}"[:4] in text


def _recorder(store):
    orig = db.check_against_round2

    def rec(*a, **k):
        r = orig(*a, **k)
        store.append(r)
        return r
    return rec
