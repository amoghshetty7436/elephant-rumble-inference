"""End-to-end test of tools/experiments.py on the tiny fake Datasets tree of test_dataset_configs."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
torch = pytest.importorskip("torch")
import dataset_configs as dc  # noqa: E402
import experiments as ex  # noqa: E402
from tests.test_dataset_configs import source, snapshot  # noqa: E402,F401


def test_detection_rules_match_fruitpunch():
    from fp_replica import segments
    rng = np.random.default_rng(0)
    prob = (rng.random(3000) > 0.4).astype(np.float32) * 0.6 + rng.random(3000).astype(np.float32) * 0.39
    ours = ex.detections(prob, ex.FRUITPUNCH_RULE)
    theirs = [(a, b, float(prob[a:b + 1].mean())) for a, b in segments(torch.from_numpy(prob > 0.5))]
    assert [(a, b) for a, b, _ in ours] == [(a, b) for a, b, _ in theirs]
    assert np.allclose([s for *_, s in ours], [s for *_, s in theirs])


def test_rules_smooth_merge_score():
    p = np.array([0.1, 0.9, 0.9, 0.9, 0.9, 0.2, 0.9, 0.9, 0.9, 0.9, 0.1], dtype=np.float32)
    assert len(ex.detections(p, dict(smooth=1, gap=0, min_frames=4, score="mean"))) == 2
    merged = ex.detections(p, dict(smooth=1, gap=1, min_frames=4, score="max"))
    assert merged == [(1, 9, pytest.approx(0.9))]
    blip = np.array([0.1, 0.1, 0.55, 0.6, 0.55, 0.52, 0.1, 0.1], dtype=np.float32)
    assert ex.detections(blip, ex.FRUITPUNCH_RULE) and not ex.detections(blip, dict(smooth=5, gap=0, min_frames=4, score="mean"))


def test_context_model_day_probs_match_training_windows():
    torch.manual_seed(0)
    m = ex.make_model(3).eval()
    emb = torch.randn(50, 768).half()
    p = ex.day_probs(m, emb, 3, "cpu")
    idx = np.clip(np.arange(50)[:, None] + np.arange(-3, 4)[None, :], 0, 49)
    with torch.no_grad():
        q = torch.softmax(m(emb.float()[torch.from_numpy(idx)]), 1)[:, 1].numpy()
    assert np.allclose(p, q, atol=1e-5)


def test_all_and_final(source, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dc, "HELDOUT_FOREST_DAYS", 2)
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder"])
    before = snapshot(source)
    out = source / "Configurations"
    common = ["--source", str(source), "--minutes", "0.01", "--seeds", "2", "--fake-embedder"]
    monkeypatch.setattr(ex, "RULES", ex.RULES[:1] + ex.RULES[40:43])  # keep the test quick
    ex.main(common)
    exp = out / "_experiments"
    names = sorted(d for d in os.listdir(exp) if (exp / d / "results.json").exists())
    assert {"ctx0_neg1_hard0", "ctx4_neg1_hard0", "ctx8_neg1_hard0", "ctx16_neg1_hard0"} <= set(names)
    assert len([n for n in names if n.endswith(("hard1", "hard2"))]) >= 4
    for n in names:
        r = json.load(open(exp / n / "results.json"))
        assert 0.5 <= r["threshold"] <= 1 and len(r["val_per_seed"]) == 2
        assert (exp / n / "model_seed1.pth").exists()
        info = r["training"][0]
        assert info["negatives_per_rumble"] == pytest.approx(r["negatives_per_rumble"], rel=0.2)
        if r["hard_negative_rounds"]:
            assert (exp / n / "mined.json").exists()
    # the cache now has whole days for the training days, and the original frames files are untouched
    base = dc.find_config_dir(str(out), ex.base_from_choices(str(out)))
    cfg = json.load(open(os.path.join(base, "config.json")))
    for p, keys in cfg["train"].items():
        for k in keys:
            assert os.path.exists(dc.cache_path(str(out / "_embedding_cache"), p, k, "day"))
    table = (exp / "experiments.md").read_text()
    assert table.count("\n| ctx") == len(names) and "Choices made" in table
    chosen = (exp / "chosen.txt").read_text().strip()
    assert chosen in names

    ex.main(common + ["final"])
    fin = exp / chosen / "final"
    assert "Sample B" in (fin / "summary.md").read_text()
    assert (fin / "sampleB_detections.txt").exists()
    # rerunning resumes: nothing is retrained
    t = {n: os.path.getmtime(exp / n / "results.json") for n in names}
    ex.main(common)
    assert all(os.path.getmtime(exp / n / "results.json") == t[n] for n in names)
    after = snapshot(source)
    assert not [p for p, m in before.items() if after.get(p) != m and not p.startswith(str(out))]
    assert not [p for p in after if p not in before and not p.startswith(str(out))]


def test_hard_negative_mining(source, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dc, "HELDOUT_FOREST_DAYS", 2)
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder"])
    common = ["--source", str(source), "--minutes", "0.01", "--seeds", "1", "--fake-embedder", "--base", "B"]
    monkeypatch.setattr(ex, "RULES", ex.RULES[:1])
    monkeypatch.setattr(ex, "cmd_all", lambda lab: lab.ensure("ctx0_neg1_hard0"))
    ex.main(common)
    # a model that fires in short bursts everywhere: the bursts far from any rumble get mined
    monkeypatch.setattr(ex, "day_probs", lambda m, emb, ctx, dev: np.where(
        (np.arange(emb.shape[0]) // 6) % 4 == 0, 0.9, 0.1).astype(np.float32))
    lab = ex.Lab(type("A", (), dict(out=str(source / "Configurations"), base="B", seeds=1, minutes=0.01, seed=0,
                                    cache=str(source / "Configurations" / "_embedding_cache"), rerun=False,
                                    fake_embedder=True))(), dc.load_pools(str(source)),
                 dc.Writer([str(source / "Configurations")], []))
    mined, n_new = lab.mined_from("ctx0_neg1_hard0")
    assert n_new > 0
    for (p, k), frames in mined.items():
        for l in lab.pools[p].labels[k]:
            assert not [f for f in frames if l.start - ex.HARD_PAD_S < f * dc.FRAME_SECONDS < l.end + ex.HARD_PAD_S]
    bank, sets = ex.build_training_set(lab.cfg, lab.pools, lab.a.cache, 2, 1, mined, [0])
    info = sets[0][2]
    assert info["negatives_mined"] > 0 and info["negatives_per_rumble"] == pytest.approx(1, rel=0.1)
