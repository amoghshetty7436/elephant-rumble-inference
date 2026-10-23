"""Tests of tools/improve_converged.py: the same staged run as improve.py, but every model is
trained until its held-out loss stops improving, into _improve_converged."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
torch = pytest.importorskip("torch")
import dataset_configs as dc  # noqa: E402
import improve_converged as ic  # noqa: E402
from tests.test_dataset_configs import source, snapshot  # noqa: E402,F401


def test_all_and_final_converged(source, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dc, "HELDOUT_FOREST_DAYS", 2)
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder", "--configs", "A", "B", "C", "D"])
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder", "--configs", "E", "--best", "A"])
    out = source / "Configurations"
    before = snapshot(source)
    monkeypatch.setattr(ic, "STAGES", [("2a", "noise", "noise", [1, 3]), ("2b", "rand", "random", [200, 0]),
                                       ("2c", "neg", "negatives", [1, 2]), ("3", "ctx", "window", [0, 1])])
    monkeypatch.setattr(ic, "HARD_ROUNDS", [1])
    monkeypatch.setattr(ic, "RULE_OPTIONS", dict(on=[0.5, 0.3], min_frames=[2, 4], smooth=[1, 3], gap=[0],
                                                 score=["mean"]))
    common = ["--source", str(source), "--seeds", "2", "--fake-embedder", "--target-recall", "0.5",
              "--patience", "2", "--max-epochs", "6"]
    ic.main(common)
    root = out / "_improve_converged"
    assert not (out / "_improve").exists()  # the first run's folder is never touched
    c = json.load(open(root / "choices.json"))
    assert {"rules", "2a", "2b", "2c", "3", "combos", "mistakes", "rules_final", "final"} <= set(c)
    final = c["final"]["winner"]
    tr = json.load(open(root / final / "training.json"))
    assert tr["training"].startswith("until converged")
    for run in tr["runs"]:
        assert 1 <= run["best_epoch"] <= run["epochs"] <= 6
    r = json.load(open(root / final / "results.json"))
    assert r["settings"]["training"] == tr["training"]

    # resuming retrains nothing and keeps the epoch counts
    names = [d for d in os.listdir(root) if (root / d / "model_seed0.pth").exists()]
    t = {n: os.path.getmtime(root / n / "model_seed0.pth") for n in names}
    ic.main(common)
    assert all(os.path.getmtime(root / n / "model_seed0.pth") == t[n] for n in names)
    assert "epochs" in json.load(open(root / final / "training.json"))["runs"][0]

    ic.main(common + ["final"])
    assert (root / "final.md").exists() and (root / "final_detections" / "final_model.json").exists()
    after = snapshot(source)
    assert not [p for p, m in before.items() if after.get(p) != m and not p.startswith(str(out))]


def test_stops_when_heldout_loss_stops_improving():
    g = torch.Generator().manual_seed(0)
    bank = torch.randn(400, 768, generator=g).half()
    y = (torch.arange(400) % 2).long()
    bank[y == 1, :8] += 3  # an easy signal, so the held-out loss soon stops improving
    win = torch.arange(400).unsqueeze(1)

    class F:
        def write(self, s): pass
    m = ic.train_until_converged(bank, win, y, 0, 0, F(), "cpu", patience=3, max_epochs=200)
    assert m.epochs - m.best_epoch <= 3 and m.epochs < 200
