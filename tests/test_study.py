"""Tests of tools/study.py: the full rerun end to end on the tiny fake Datasets tree, plus its
building blocks (scoring and ties, early stopping, augmentation stretches)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
torch = pytest.importorskip("torch")
import dataset_configs as dc  # noqa: E402
import study as st  # noqa: E402
from tests.test_dataset_configs import source, snapshot  # noqa: E402,F401


def fake_summary(score, sd, max_recall, fp90=1.0, runs=5):
    s = (score, score - sd, score + sd, sd, runs)
    return {"score": s, "fp_at_80": s, "fp_at_85": s, "fp_at_90": (fp90, fp90, fp90, 0, runs),
            "max_recall": (max_recall, max_recall, max_recall, 0, runs), "runs": runs}


def test_choose_ties_go_to_recall():
    items = [("a", fake_summary(0.20, 0.02, 0.91), 0), ("b", fake_summary(0.21, 0.02, 0.97), 1),
             ("c", fake_summary(0.30, 0.01, 0.99), 0)]
    win, ranked, _ = st.choose(items)
    assert ranked[0] == "a" and win == "b"  # b is within a's spread and reaches more rumbles
    no = dict(fake_summary(0.1, 0.0, 0.84), score=None)
    win, ranked, _ = st.choose([("x", no, 0), ("a", fake_summary(0.2, 0.0, 0.9), 0)])
    assert win == "a" and ranked[-1] == "x"  # missing 85% recall ranks last


def test_summarize_needs_every_run():
    thr = np.array([0.1, 0.5, 0.9])
    sw = lambda rec, fp: {"threshold": thr, "recall": np.array(rec), "fp_per_hour": np.array(fp),  # noqa: E731
                          "precision": np.array([0.5, 0.6, 0.7]), "f1": np.array([0.5, 0.6, 0.6])}
    good = sw([0.95, 0.86, 0.5], [1.0, 0.4, 0.1])
    bad = sw([0.82, 0.81, 0.5], [1.0, 0.4, 0.1])
    s = st.summarize([good, good])
    assert s["score"][0] == pytest.approx(0.4) and s["runs_reaching"] == 2
    s = st.summarize([good, bad])
    assert s["score"] is None and s["runs_reaching"] == 1


def test_average_precision():
    y = torch.tensor([1, 0, 1, 0])
    assert st.average_precision(torch.tensor([0.9, 0.8, 0.7, 0.1]), y) == pytest.approx((1 + 2 / 3) / 2)
    assert st.average_precision(torch.tensor([0.9, 0.1, 0.8, 0.2]), y) == pytest.approx(1.0)


def test_merge_spans_and_variants():
    sp = st.merge_spans([10, 11, 40], 3, 100, 1000)
    assert sp == [(7, 15), (37, 44)]
    sp = st.merge_spans(list(range(0, 100)), 3, 100, 30)
    covered = set()
    for a, b in sp:
        assert b - a <= 30
        covered |= set(range(a + 3, b - 3))
    assert set(range(3, 97)) <= covered
    assert st.aug_variants(dict(type="both", level=0, copies=2)) == ["rumble_0_c0", "background_0_c0",
                                                                     "rumble_0_c1", "background_0_c1"]
    assert st.aug_text(dict(type="rumble", level=-10, copies=4)) == "rumblem10x4"


def test_stops_when_heldout_stops_improving():
    g = torch.Generator().manual_seed(0)
    bank = torch.randn(600, 768, generator=g).half()
    y = (torch.arange(600) % 2).long()
    bank[y == 1, :8] += 3
    win = torch.arange(600).unsqueeze(1)
    m = st.train_until_converged(bank[:], win[:500], y[:500], win[500:], y[500:], 0, 0, st.Quiet(), "cpu",
                                 check_every=200, patience=3, max_checks=200)
    ri = m.run_info
    assert ri["checks"] - ri["best_check"] <= 3 and ri["checks"] < 200 and ri["heldout_ap"] > 0.9


def test_full_study(source, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dc, "HELDOUT_FOREST_DAYS", 2)
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder"])
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder", "--configs", "E", "--best", "C"])
    out = source / "Configurations"
    setups = st.find_setups(str(out))
    assert {"A", "B", "C", "D", "E", "E2", "F", "G", "H"} <= set(setups)
    before = snapshot(source)
    for k, v in dict(CHECK_EVERY=150, PATIENCE=2, MAX_CHECKS=5, HELDOUT_ONE_IN=3, NOISE_REPEATS=[1, 3],
                     RANDOM_PER_DAY=[200, 0], GAP_PER_RUMBLE=[0, 1], CONTEXTS=[0, 2], HARD_ROUNDS=[1, 2],
                     AUG_TYPES=["clean", "both"], AUG_LEVELS=[0, "mix"], AUG_COPIES=[1, 2], AUG_DISK_MARGIN_GB=0).items():
        monkeypatch.setattr(st, k, v)
    small = dict(on=[0.5, 0.3], min_frames=[2, 4], smooth=[1, 3], gap=[0], score=["mean"])
    monkeypatch.setattr(st, "ROUND1_RULES", small)
    monkeypatch.setattr(st.im, "RULE_OPTIONS", dict(small, gap=[0, 1]))
    common = ["--source", str(source), "--seeds", "2", "--fake-embedder", "--finish-by", "none", "--start-from", "E2"]
    st.main(common)
    root = out / "_study"
    c = json.load(open(root / "choices.json"))
    assert {"round1", "rules", "3", "4", "5", "6", "7a", "8", "9", "rules_final", "final"} <= set(c)
    assert "round1_side" in c and "F" in c["round1_side"]["ranked"]  # F and H get a fair comparison
    assert set(c["round1"]["ranked"]) == {"A", "B", "C", "D", "E", "E2", "G"}  # F, H: other validation days
    assert c["round1"]["winner"] == "E2"
    final = c["final"]["winner"]
    w3 = c["3"]["winner"]  # later stages are scored with the stage-2 rule, not the first-round one
    assert (root / w3 / f"results_{st.rule_key(c['rules']['rule'])}.json").exists()
    assert c["rules"]["setup"] == "E2"
    tr = json.load(open(root / final / "training.json"))
    for run in tr["runs"]:
        assert 1 <= run["best_check"] <= run["checks"] <= 5
    assert tr["data"]["heldout_days"] >= 1
    assert (root / "study.md").exists() and (root / "study.csv").exists()
    assert (root / "rules_rules.csv").exists() and (root / "rules_rules_final.csv").exists()
    rules_rows = open(root / "rules_rules.csv").read().splitlines()
    assert len(rules_rows) == 1 + len(st.all_rules(st.im.RULE_OPTIONS))  # every rule saved
    assert (root / "final.md").exists() and (root / "final_detections" / "final_model.json").exists()
    fin = (root / "final.md").read_text()
    assert "10 final model" in fin and "Test 1" in fin
    # augmented fingerprints are kept on disk
    aug = out / "_embedding_cache" / "_augmented"
    assert any(p.endswith("done.json") for _, _, fs in os.walk(aug) for p in fs)
    names = [d for d in os.listdir(root) if (root / d / "spec.json").exists()]
    assert any("augboth" in n for n in names) and any("augclean" in n for n in names)
    assert any(n.endswith("_hard2_augnone") or "_hard2_" in n for n in names)
    # explicit recipes: no gap frames at all in the gap-0 experiment
    zero = [n for n in names if "_gap0_" in n]
    if zero:
        d = json.load(open(root / zero[0] / "training.json"))["data"]
        assert d["negatives_from_gaps"] == 0

    # resuming retrains nothing
    t = {n: os.path.getmtime(root / n / "model_seed0.pth") for n in names if (root / n / "model_seed0.pth").exists()}
    st.main(common)
    assert all(os.path.getmtime(root / n / "model_seed0.pth") == t[n] for n in t)
    after = snapshot(source)
    assert not [p for p, m in before.items() if after.get(p) != m and not p.startswith(str(out))]
    assert not [p for p in after if p not in before and not p.startswith(str(out))]


def test_deadline_stops_new_work(source, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dc, "HELDOUT_FOREST_DAYS", 2)
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder", "--configs", "A", "B", "C", "D"])
    soon = __import__("time").strftime("%Y-%m-%d %H:%M", __import__("time").localtime(__import__("time").time() + 60))
    with pytest.raises(SystemExit):  # not even one first-round setup fits before the finish time
        st.main(["--source", str(source), "--seeds", "1", "--fake-embedder", "--finish-by", soon])


def test_a_failing_stage_still_scores_the_tests(source, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dc, "HELDOUT_FOREST_DAYS", 2)
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder", "--configs", "A", "B", "C", "D"])
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder", "--configs", "E", "--best", "A"])
    for k, v in dict(CHECK_EVERY=150, PATIENCE=2, MAX_CHECKS=4, HELDOUT_ONE_IN=3).items():
        monkeypatch.setattr(st, k, v)
    small = dict(on=[0.5, 0.3], min_frames=[2, 4], smooth=[1], gap=[0], score=["mean"])
    monkeypatch.setattr(st, "ROUND1_RULES", small)
    monkeypatch.setattr(st.im, "RULE_OPTIONS", small)

    def broken(*a, **k):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(st, "aug_stage", broken)
    monkeypatch.setattr(st.Lab, "mine", broken)
    st.main(["--source", str(source), "--seeds", "1", "--fake-embedder", "--finish-by", "none",
             "--round1", "A", "E2", "--start-from", "E2", "--no-augmentation"])
    root = source / "Configurations" / "_study"
    assert "simulated failure" in (root / "failed.txt").read_text()
    c = json.load(open(root / "choices.json"))
    assert "final" in c and (root / "final.md").exists()
