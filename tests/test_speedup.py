"""Tests of tools/speedup.py: the 0.64 s grid, and the experiment end to end on the tiny fake
Datasets tree (after short study.py and candidates.py runs)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("scipy")
import candidates as cd  # noqa: E402
import dataset_configs as dc  # noqa: E402
import round2 as r2  # noqa: E402
import speedup as su  # noqa: E402
import study as st  # noqa: E402
from tests.test_dataset_configs import source, snapshot  # noqa: E402,F401


def test_speeds_map_to_the_usual_grid():
    assert su.frame_seconds("aves", 32) == pytest.approx(0.64)
    assert su.frame_seconds("dasheng", 16) == pytest.approx(0.64)
    assert su.frame_seconds("aves", 16) == pytest.approx(0.32)
    assert su.frame_seconds("aves", 48) == pytest.approx(0.96)
    assert su.frame_seconds("dasheng", 32) == pytest.approx(1.28)


def test_to_grid_keeps_same_length_and_follows_time():
    e = torch.nn.functional.normalize(torch.randn(10, 8), dim=1)
    assert torch.allclose(su.to_grid(e, 10), e, atol=1e-5)
    # a step change halfway through stays halfway through after halving or doubling the rate
    s = torch.cat([torch.tensor([[1.0, 0.0]]).repeat(20, 1), torch.tensor([[0.0, 1.0]]).repeat(20, 1)])
    for n in (10, 80):
        g = su.to_grid(s, n)
        assert g.shape == (n, 2) and torch.allclose(g.norm(dim=1), torch.ones(n))
        assert (g[: n // 2 - 1, 0] > 0.9).all() and (g[n // 2 + 1:, 1] > 0.9).all()
    assert su.to_grid(s, 0).shape == (0, 2)


def test_speedup_end_to_end(source, monkeypatch):  # noqa: F811
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
    cd.main(["--source", str(source), "--seeds", "2", "--fake-embedder", "--only", "m6"])
    out = source / "Configurations"
    before = snapshot(source)
    speeds = ["aves24", "dasheng8"]
    su.main(["--source", str(source), "--seeds", "2", "--fake-embedder", "--speeds"] + speeds)
    root = out / "_speedup"
    assert not (root / "failed.txt").exists(), (root / "failed.txt").read_text()
    names = su.REFERENCES + [s for n in speeds for s in (n, n + "_ens")]
    for s in names:
        r = json.load(open(root / s / "results.json"))
        assert r["summary"]["runs"] == (1 if s.endswith("_ens") else 2)
    fin = json.load(open(root / "final.json"))
    assert sorted(fin) == sorted(names) and "sampleB" in fin["aves24"]["sets"]
    md = (root / "speedup.md").read_text()
    assert "aves24_ens" in md and "AVES at 24x" in md
    assert "dasheng8" in (root / "speedup_final.md").read_text()
    # each speed's fingerprints are gone once its tests are scored
    for n in speeds:
        assert not (out / f"_embedding_cache_{n}").exists()
    # only the experiment's folders were written
    after = snapshot(source)
    mine = (str(root),) + tuple(str(out / f"_embedding_cache_{n}") for n in su.SPEEDS)
    assert not [p for p, m in before.items() if after.get(p) != m and not p.startswith(mine)]
    assert not [p for p in after if p not in before and not p.startswith(mine)]
    # resuming retrains and re-fingerprints nothing
    t = os.path.getmtime(root / "aves24" / "model_seed0.pth")
    su.main(["--source", str(source), "--seeds", "2", "--fake-embedder", "--speeds"] + speeds)
    assert os.path.getmtime(root / "aves24" / "model_seed0.pth") == t
    assert not (out / "_embedding_cache_aves24").exists()
