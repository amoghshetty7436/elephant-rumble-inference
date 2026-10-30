"""Tests of tools/candidates.py: every candidate model end to end on the tiny fake Datasets
tree (after a short study.py run), plus the building blocks that must match study.py."""
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
import experiments as ex  # noqa: E402
import improve as im  # noqa: E402
import study as st  # noqa: E402
from tests.test_dataset_configs import source, snapshot  # noqa: E402,F401


def test_ctx_model_matches_the_final_models_classifier():
    a, b = ex.make_model(8), cd.make_ctx_model(768, 8)
    b.load_state_dict(a.state_dict())  # same layers, same names
    a.eval(), b.eval()
    x = torch.randn(300, 768).half()
    assert np.allclose(ex.day_probs(a, x, 8, "cpu"), ex.day_probs(b, x, 8, "cpu"))
    assert np.allclose(cd.window_probs(b, x, 8, "cpu"), ex.day_probs(b, x, 8, "cpu"), atol=1e-5)


def test_two_stream_and_sequence_shapes():
    m = cd.make_two_stream(8).eval()
    assert m(torch.randn(5, 17, 768 + cd.LF_SUB * cd.LF_BINS)).shape == (5, 2)
    s = cd.make_seq_model(776).eval()
    assert s(torch.randn(3, 128, 776)).shape == (3, 128)
    p = cd.seq_probs(s, torch.randn(1000, 776), "cpu")
    assert p.shape == (1000,) and (p > 0).all()  # every step scored once


def test_sweep_events_matches_improve_sweep():
    rng = np.random.default_rng(0)
    labels = [dc.fp_data.Label(t, 4.0, 10, 30, "d") for t in (30, 100, 200, 300)]
    pools = {"p": type("P", (), {"labels": {"d": labels}})()}
    truth = im.Truth(pools, [("p", "d", 0.2)])
    prob = rng.random(600) * 0.3
    for t in (30, 100, 300):
        i = dc.fp_data.time_to_index(t)
        prob[i:i + 6] = 0.9
    rule = dict(on=0.5, min_frames=2, smooth=1, gap=0, score="max")
    s, e, v = im.alarms(prob, rule)
    a = im.sweep(truth, {("p", "d"): prob}, rule)
    b = cd.sweep_events(truth, {("p", "d"): (s, e, v)})
    for k in ("recall", "precision", "fp_per_hour", "detections"):
        assert np.allclose(np.nan_to_num(a[k]), np.nan_to_num(b[k])), k


def test_features_have_the_right_grid():
    y = np.random.default_rng(1).normal(size=500 * 200).astype(np.float32) * 0.01
    t = np.arange(len(y)) / 500
    y[500 * 50:500 * 60] += 0.2 * np.sin(2 * np.pi * 15 * t[500 * 50:500 * 60])
    n = int(np.ceil(len(y) / 320))
    lf = cd.lf_features(y, n)
    assert lf.shape == (n, cd.LF_SUB * cd.LF_BINS) and lf.dtype == np.uint8
    pf = cd.pitch_features(y, n).astype(np.float32)
    assert pf.shape == (n, len(cd.PITCH_FEATURES))
    inside, outside = slice(80, 90), slice(150, 160)
    assert pf[inside, 0].mean() > pf[outside, 0].mean()        # more harmonic salience in the tone
    f0 = cd.PITCH_F0[0] + pf[inside, 2] * (cd.PITCH_F0[1] - cd.PITCH_F0[0])
    assert abs(np.median(f0) - 15) < 1.0
    assert np.allclose(cd.rolling_fast(pf[:, 2], 5, "std"), cd.rolling(pf[:, 2], 5, np.std), atol=1e-3)


def test_all_candidates_end_to_end(source, monkeypatch):  # noqa: F811
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
    out = source / "Configurations"
    assert "final" in json.load(open(out / "_study" / "choices.json"))
    for k, v in dict(SEQ_CHECK_EVERY=40, SEQ_PATIENCE=2, SEQ_MAX_CHECKS=3, SEQ_RUMBLE_PER_DAY=6,
                     SEQ_RANDOM_PER_DAY=4, VER_MAX_EPOCHS=2, VER_PATIENCE=1, VER_NEG_PER_DAY=10,
                     VER_POS_PER_DAY=10).items():
        monkeypatch.setattr(cd, k, v)
    before = snapshot(source)
    cd.main(["--source", str(source), "--seeds", "2", "--fake-embedder"])
    root = out / "_candidates"
    assert not (root / "features" / "failed.txt").exists()
    for m in cd.ALL_MODELS:
        fail = root / m / "failed.txt"
        assert not fail.exists(), fail.read_text()
        r = json.load(open(root / m / "results.json"))
        assert r["seeds"] == [0, 1] and r["validation_rumbles"] > 0
    assert json.load(open(root / "m5" / "training.json"))["dim"] == 768 + 256
    assert json.load(open(root / "m7" / "training.json"))["dim"] == 768 + 8
    md = (root / "candidates.md").read_text()
    assert "Final model" in md and "BirdAVES" in md and "Dasheng" in md
    fin = json.load(open(root / "final.json"))
    assert set(fin) == set(cd.ALL_MODELS) and "sampleB" in fin["m1"]
    assert "Final model of the rerun" in (root / "final.md").read_text()
    # only the candidates' own folders were written
    after = snapshot(source)
    mine = [str(out / n) for n in ("_candidates", "_features")] + [str(out / f"_embedding_cache_{e}") for e in cd.ENCODERS]
    assert not [p for p, m in before.items() if after.get(p) != m and not any(p.startswith(x) for x in mine)]
    assert not [p for p in after if p not in before and not any(p.startswith(x) for x in mine)]
    # resuming retrains nothing
    t = {m: os.path.getmtime(root / m / "model_seed0.pth") for m in cd.ALL_MODELS}
    cd.main(["--source", str(source), "--seeds", "2", "--fake-embedder"])
    assert all(os.path.getmtime(root / m / "model_seed0.pth") == t[m] for m in t)
