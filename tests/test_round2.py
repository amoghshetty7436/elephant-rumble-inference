"""Tests of tools/round2.py: the checkers' architectures, and the whole round end to end on
the tiny fake Datasets tree (after short study.py and candidates.py runs)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("scipy")
import candidates as cd  # noqa: E402
import dataset_configs as dc  # noqa: E402
import round2 as r2  # noqa: E402
import study as st  # noqa: E402
from tests.test_dataset_configs import source, snapshot  # noqa: E402,F401


@pytest.mark.parametrize("arch", r2.ARCHS)
def test_checkers_take_one_channel_spectrograms(arch):
    m = r2.make_checker(arch, pretrained=False).eval()
    cuts = np.random.default_rng(0).integers(0, 255, (3, 128, cd.cut_cols()), dtype=np.uint8)
    s = cd.verifier_scores(m, cuts, "cpu")
    assert s.shape == (3,) and ((s > 0) & (s < 1)).all()


def test_fit_pads_and_trims():
    assert list(r2.fit(np.array([1.0, 2.0]), 4)) == [1, 2, 2, 2]
    assert r2.fit(torch.ones(5, 3), 4).shape == (4, 3)
    assert r2.fit(torch.ones(2, 3), 4).shape == (4, 3)


def test_round2_end_to_end(source, monkeypatch):  # noqa: F811
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
    out = source / "Configurations"
    before = snapshot(source)
    r2.main(["--source", str(source), "--seeds", "2", "--fake-embedder"])
    root = out / "_round2"
    assert not (root / "failed.txt").exists(), (root / "failed.txt").read_text()
    for s in r2.STAGES:
        r = json.load(open(root / s / "results.json"))
        assert r["summary"]["runs"] == (1 if s.endswith("_ens") else 2)
    checked = sorted(os.listdir(root / "checked"))
    assert len(checked) == r2.CHECKED_STAGES * len(r2.ARCHS) * 2
    md = (root / "round2.md").read_text()
    assert "convnext_tiny" in md and "late_ens" in md
    fin = json.load(open(root / "final.json"))
    assert len(fin) == len(r2.STAGES) + len(checked) and "sampleB" in fin["aves"]["sets"]
    # the reference reproduces round 1's validation numbers for the final model
    a = json.load(open(root / "aves" / "results.json"))["summary"]["score"]
    # only round 2's folder was written
    after = snapshot(source)
    assert not [p for p, m in before.items() if after.get(p) != m and not p.startswith(str(root))]
    assert not [p for p in after if p not in before and not p.startswith(str(root))]
    # resuming retrains nothing and rescoring the tests reuses them
    t = os.path.getmtime(root / "fusion" / "model_seed0.pth")
    c = os.path.getmtime(root / "checkers" / "resnet18_seed0.pth")
    r2.main(["--source", str(source), "--seeds", "2", "--fake-embedder"])
    assert os.path.getmtime(root / "fusion" / "model_seed0.pth") == t
    assert os.path.getmtime(root / "checkers" / "resnet18_seed0.pth") == c
    assert a == json.load(open(root / "aves" / "results.json"))["summary"]["score"]
