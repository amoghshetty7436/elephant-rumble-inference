"""tools/aves_tuning.py: the tuned models give AVES's own fingerprints before training, a
training piece is exactly the detector's piece, and the whole run (after a small study.py
run) on the tiny fake Datasets tree, resuming and writing only where it may."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
import numpy as np  # noqa: E402
import dataset_configs as dc  # noqa: E402
import study as st  # noqa: E402
import finetune as ft  # noqa: E402
import aves_tuning as at  # noqa: E402
from tests.test_dataset_configs import source, snapshot  # noqa: E402,F401


def rep(**k):
    return dict({"backbone": "bio", "mix": "layer", "keep": 12}, **k)


def test_untrained_options_give_aves_fingerprints(monkeypatch):
    monkeypatch.setattr(at, "CHUNK_S", 60)  # 1-minute pieces keep the test fast
    monkeypatch.setattr(ft, "CHUNK_S", 60)
    aves = ft.load_aves(fake=True)
    torch.manual_seed(1)
    y = torch.randn(500 * 150) * 0.1
    want = ft.unit(ft.embed_layers(aves, y, [8, 12], "cpu")[12])
    for t in (at.tune("h", 1, rep(), "head"), at.tune("l", 2, rep(), "lora", n=4, r=4),
              at.tune("f", 2, rep(), "lora", n=12, r=8, ff=True), at.tune("a", 2, rep(), "adapter", n=12),
              at.tune("c", 2, rep(), "combo", n=12, r=8), at.tune("b", 2, rep(), "bitfit", n=12),
              at.tune("t", 3, rep(), "layers", n=2), at.tune("g", 3, rep(), "gradual", n=12)):
        m = at.make_tuned(t, 2, aves).eval()
        h = at.day_fingerprints(m, y, "cpu")
        if t["method"] == "combo":  # the layer mix starts as an equal mix, not layer 12
            assert h.shape == want.shape
            continue
        assert h.shape == want.shape and float((h * want).sum(1).min()) > 0.999, t["name"]
        assert len(h) == at.day_frames(len(y))
        if t["method"] != "head":
            assert 0 < m.trainable()
    short = at.make_tuned(at.tune("s", 3, rep(keep=8), "all", n=8), 2, aves).eval()
    l8 = ft.unit(ft.embed_layers(aves, y, [8, 12], "cpu")[8])
    assert float((at.day_fingerprints(short, y, "cpu") * l8).sum(1).min()) > 0.999
    lo = at.make_tuned(at.tune("l", 2, rep(), "lora", n=4, r=4), 2, aves)
    top = at.make_tuned(at.tune("t", 3, rep(), "layers", n=4), 2, aves)
    assert lo.trainable() < top.trainable()


def test_training_piece_is_the_detectors_piece(monkeypatch):
    monkeypatch.setattr(at, "CHUNK_S", 64)  # 100 frames: like the real hour (5,625), a whole number of frames
    monkeypatch.setattr(at, "PIECE_FRAMES", 64 * 500 // at.HOP)
    aves = ft.load_aves(fake=True)
    torch.manual_seed(2)
    y = (torch.randn(500 * 170) * 0.1).half().float()  # 2 whole pieces and a short last one
    m = at.make_tuned(at.tune("h", 1, rep(), "head"), 2, aves).eval()
    whole = at.day_fingerprints(m, y, "cpu")
    n = len(whole)

    class D(at.AudioData):
        def __init__(self):
            self.s = {"ctx": 2}
            self.days, self.noise = [y.half().float()], []
            self.pos = np.asarray([[0, i] for i in range(0, n, 7)])
            self.neg = np.asarray([[0, i] for i in range(3, n, 7)])
            self.label = [{**{i: 0 for i in range(3, n, 7)}, **{i: 1 for i in range(0, n, 7)}}]

    d = D()
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(30):
        wave, a, cnt, last, idx, yl = d.crop(rng)
        with torch.no_grad():
            e = m.fingerprints(wave)
        h = e[a:a + cnt]
        if last and h.shape[0] < cnt:
            h = torch.cat([h, e[-2:-1]])
        # find the piece by matching: every labelled frame's fingerprint equals the detector's
        for i, lab in zip(idx.tolist(), yl.tolist()):
            g = [k for k in range(0, n, at.PIECE_FRAMES) if k + i < n and float((h[i] * whole[k + i]).sum()) > 0.9999] \
                if i < h.shape[0] else []
            assert g, "a training fingerprint differs from the detector's"
            seen.add(g[0] + i)
            assert lab == d.label[0][g[0] + i]
    assert len(seen) > 20 and max(seen) >= 2 * at.PIECE_FRAMES  # the short last piece is used too


def test_whole_run_after_study(source, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dc, "HELDOUT_FOREST_DAYS", 2)
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder", "--configs", "A", "B", "C", "D"])
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder", "--configs", "E", "--best", "A"])
    for k, v in dict(CHECK_EVERY=150, PATIENCE=2, MAX_CHECKS=4, HELDOUT_ONE_IN=3, NOISE_REPEATS=[1], RANDOM_PER_DAY=[200],
                     GAP_PER_RUMBLE=[1], CONTEXTS=[1], HARD_ROUNDS=[1]).items():
        monkeypatch.setattr(st, k, v)
    small = dict(on=[0.5, 0.3], min_frames=[2, 4], smooth=[1], gap=[0], score=["mean"])
    monkeypatch.setattr(st, "ROUND1_RULES", small)
    monkeypatch.setattr(st.im, "RULE_OPTIONS", small)
    st.main(["--source", str(source), "--seeds", "1", "--fake-embedder", "--finish-by", "none",
             "--round1", "A", "E2", "--start-from", "E2", "--no-augmentation"])
    out = source / "Configurations"
    assert "final" in json.load(open(out / "_study" / "choices.json"))
    before = snapshot(source)
    for k, v in dict(CHECK_STEPS=2, PATIENCE=1, MAX_CHECKS=2, ACCUM=1, DISK_MARGIN_GB=0, FINALISTS=2).items():
        monkeypatch.setattr(at, k, v)
    names = st.name_of, st.write_report
    args = ["--source", str(source), "--fake-embedder", "--finish-by", "none"]
    at.main(args)
    assert (st.name_of, st.write_report) == names  # study.py is left as it was
    root = out / "_aves_tuning"
    assert not (root / "failed.txt").exists(), (root / "failed.txt").read_text()
    c = json.load(open(root / "choices.json"))
    assert {"stage1", "representation", "stage2", "stage3", "stage3_blends", "stage4", "finalists", "final"} <= set(c)
    exps = [d for d in os.listdir(root) if (root / d / "spec.json").exists()]
    for want in ("s1_bio_L12_head", "s1_bio_wsum_head", "s1_bioxnlarge_L24_head", "_lora4r8", "_adapters", "_combo",
                 "_bitfit", "_gradual", "_short_all", "_wise50", "_ema", "_aug", "_remine", "_focal"):
        assert any(want in d for d in exps), want
    for d in exps:
        assert any(f.startswith("results_") for f in os.listdir(root / d)), d
    fin = c["finalists"]["finalists"]
    assert len(fin) == 2
    for f in fin:  # finalists have 3 runs (a derived finalist's parent does)
        s = json.load(open(root / f / "spec.json"))
        d = s["tune"].get("derive")
        src = d["parent"] if d else f
        assert all((root / src / f"model_seed{i}.pth").exists() for i in range(3))
    assert (root / "aves_tuning.md").exists() and (root / "final.md").exists()
    assert "rerun_final" in (root / "aves_tuning.md").read_text()
    assert (out / "_embedding_cache" / "_audio").is_dir()
    # resuming retrains nothing
    t = {d: os.path.getmtime(root / d / "model_seed0.pth") for d in exps if (root / d / "model_seed0.pth").exists()}
    at.main(args)
    assert all(os.path.getmtime(root / d / "model_seed0.pth") == t[d] for d in t)
    after = snapshot(source)
    allowed = (str(root), str(out / "_embedding_cache" / "_audio"))
    assert not [p for p, m in before.items() if after.get(p) != m and not p.startswith(allowed)]
    assert not [p for p in after if p not in before and not p.startswith(allowed)]
    # the smoke test runs too, into its own folder
    at.main(args + ["--quick"])
    assert (out / "_aves_tuning_quick" / "aves_tuning.md").exists()
