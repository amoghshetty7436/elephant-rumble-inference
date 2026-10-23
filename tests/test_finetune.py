"""tools/finetune.py end to end on the tiny fake Datasets tree, after a small study.py run,
plus the check that the cached lower layers + the copied top layers give AVES's own output."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
torch = pytest.importorskip("torch")
pytest.importorskip("torchaudio")
import dataset_configs as dc  # noqa: E402
import study as st  # noqa: E402
import finetune as ft  # noqa: E402
from tests.test_dataset_configs import source, snapshot  # noqa: E402,F401


def test_top_layers_continue_aves():
    aves = ft.load_aves(fake=True)
    torch.manual_seed(1)
    y = torch.randn(500 * 600) * 0.1
    outs = ft.embed_layers(aves, y, [8, 12], "cpu")
    m = ft.make_top(aves, 4, 0).eval()
    h = ft.day_fingerprints(m, outs[8], "cpu")
    assert h.shape == outs[12].shape
    assert float((h * ft.unit(outs[12])).sum(1).min()) > 0.999  # one piece: identical
    lo = ft.make_top(aves, 4, 2, lora_r=4)
    assert 0 < lo.trainable() < ft.make_top(aves, 4, 2).trainable()


def test_finetune_after_study(source, monkeypatch):  # noqa: F811
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
    for k, v in dict(PROBE_LAYERS=[10], FINETUNE=[dict(mode="top", n=2), dict(mode="lora", n=2, r=2)], CROP=300,
                     CHECK_STEPS=3, PATIENCE=2, MAX_CHECKS=3, ACCUM=2, DISK_MARGIN_GB=0).items():
        monkeypatch.setattr(ft, k, v)
    name_of = st.name_of
    ft.main(["--source", str(source), "--seeds", "2", "--fake-embedder"])
    assert st.name_of is name_of  # study.py is left as it was
    root = out / "_finetune"
    assert not (root / "failed.txt").exists(), (root / "failed.txt").read_text()
    c = json.load(open(root / "choices.json"))
    assert {"probe", "rules_final", "final"} <= set(c)
    names = [d for d in os.listdir(root) if (root / d / "spec.json").exists()]
    assert any(n.endswith("_L10_probe") for n in names) and any(n.endswith("_L12_probe") for n in names)
    assert any(n.endswith("_top2") for n in names) and any(n.endswith("_lora2r2") for n in names)
    tr = json.load(open(root / [n for n in names if n.endswith("_lora2r2")][0] / "training.json"))
    assert all(r["trainable_parameters"] > 0 for r in tr["runs"])
    assert (root / "finetune.md").exists() and (root / "final.md").exists()
    assert (out / "_embedding_cache" / "_layers" / "L10").is_dir()
    # resuming retrains nothing
    t = {n: os.path.getmtime(root / n / "model_seed0.pth") for n in names}
    ft.main(["--source", str(source), "--seeds", "2", "--fake-embedder"])
    assert all(os.path.getmtime(root / n / "model_seed0.pth") == t[n] for n in names)
    after = snapshot(source)
    allowed = (str(root), str(out / "_embedding_cache" / "_layers"))
    assert not [p for p, m in before.items() if after.get(p) != m and not p.startswith(allowed)]
    assert not [p for p in after if p not in before and not p.startswith(allowed)]
