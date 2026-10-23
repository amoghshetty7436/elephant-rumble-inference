"""Tests of tools/improve.py: the fast scorer matches elp_score, the alarm rules match
FruitPunch's, and the whole staged run works end to end on the tiny fake Datasets tree."""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
torch = pytest.importorskip("torch")
import dataset_configs as dc  # noqa: E402
import elp_score  # noqa: E402
import experiments as ex  # noqa: E402
import improve as im  # noqa: E402
from tests.test_dataset_configs import source, snapshot  # noqa: E402,F401


def noisy_probs(n, seed):
    rng = np.random.default_rng(seed)
    base = rng.random(n).astype(np.float32) * 0.45
    for s in rng.integers(0, n - 20, n // 60):
        base[s:s + rng.integers(1, 12)] = rng.uniform(0.3, 1.0)
    return base


def test_alarms_match_fruitpunch_and_experiments():
    for seed in range(5):
        p = noisy_probs(3000, seed)
        for rule in (dict(smooth=1, gap=0, min_frames=4, score="mean"), dict(smooth=5, gap=2, min_frames=3, score="max"),
                     dict(smooth=3, gap=1, min_frames=6, score="mean")):
            s, e, v = im.alarms(p, dict(rule, on=0.5))
            theirs = ex.detections(p, rule)
            assert [(int(a), int(b)) for a, b in zip(s, e)] == [(a, b) for a, b, _ in theirs]
            assert np.allclose(v, [x for *_, x in theirs])
    p = np.array([0.1, 0.35, 0.35, 0.35, 0.1], dtype=np.float32)
    assert len(im.alarms(p, dict(im.FRUITPUNCH_RULE, min_frames=2))[0]) == 0
    assert len(im.alarms(p, dict(im.FRUITPUNCH_RULE, min_frames=2, on=0.3))[0]) == 1


def test_fast_sweep_matches_elp_score():
    rng = np.random.default_rng(1)
    labels, probs, boxes, truth_boxes = {}, {}, [], []
    for d in range(4):
        key = f"nn01a_2018010{d}_000000"
        n = 4000
        ls = []
        for s in sorted(rng.uniform(0, n * 0.64 - 20, 15)):
            ls.append(SimpleNamespace(start=float(s), duration=float(rng.uniform(2, 8)),
                                      low_freq=float(rng.uniform(0, 30)), high_freq=float(rng.uniform(40, 300))))
        labels[key] = ls
        probs[("forest", key)] = noisy_probs(n, d + 10)
        truth_boxes += [elp_score.Box(key, l.start, l.start + l.duration, l.low_freq, l.high_freq) for l in ls]
    pools = {"forest": SimpleNamespace(labels=labels)}
    files = [("forest", k, 4000 * 0.64 / 3600) for k in labels]
    truth = im.Truth(pools, files)
    for rule in (im.FRUITPUNCH_RULE, dict(on=0.3, min_frames=2, smooth=3, gap=1, score="max")):
        for (_, k), p in probs.items():
            s, e, v = im.alarms(p, rule)
            boxes += [elp_score.Box(k, a * dc.FRAME_SECONDS, b * dc.FRAME_SECONDS, 5.0, 250.0, x) for a, b, x in zip(s, e, v)]
        fast = im.sweep(truth, probs, rule)
        slow = elp_score.score(truth_boxes, boxes, list(labels), hours_per_file=files[0][2], step=0.01)
        for r in slow:
            i = im.THRESHOLDS.index(round(r["threshold"], 3))
            assert fast["rumbles_found"][i] == r["rumbles_found"]
            assert fast["detections"][i] == r["detections"] and fast["fp"][i] == r["fp"]
        boxes = []


def test_take_repeats():
    rng = np.random.default_rng(0)
    assert len(im.take(range(10), 3, rng)) == 30
    assert len(im.take(range(10), 2.5, rng)) == 25
    assert len(im.take(range(10), 0.5, rng)) == 5
    assert len(im.take(range(10), 0, rng)) == 0


def test_all_and_final(source, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dc, "HELDOUT_FOREST_DAYS", 2)
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder", "--configs", "A", "B", "C", "D"])
    dc.main(["--source", str(source), "--minutes", "0.01", "--fake-embedder", "--configs", "E", "--best", "A"])
    out = source / "Configurations"
    assert (out / im.BASE_CONFIG / "config.json").exists()
    before = snapshot(source)
    for k, v in dict(NOISE_REPEATS=[1, 3], RANDOM_PER_DAY=[200, 0], NEG_RATIOS=[1, 2], CONTEXTS=[0, 2],
                     HARD_ROUNDS=[1]).items():
        monkeypatch.setattr(im, k, v)
    monkeypatch.setattr(im, "STAGES", [("2a", "noise", "noise", [1, 3]), ("2b", "rand", "random", [200, 0]),
                                       ("2c", "neg", "negatives", [1, 2]), ("3", "ctx", "window", [0, 2])])
    monkeypatch.setattr(im, "RULE_OPTIONS", dict(on=[0.5, 0.3], min_frames=[2, 4], smooth=[1, 3], gap=[0, 1],
                                                 score=["mean", "max"]))
    common = ["--source", str(source), "--minutes", "0.01", "--seeds", "2", "--fake-embedder", "--target-recall", "0.5"]
    im.main(common)
    root = out / "_improve"
    c = json.load(open(root / "choices.json"))
    assert {"rules", "2a", "2b", "2c", "3", "combos", "mistakes", "rules_final", "final"} <= set(c)
    final = c["final"]["winner"]
    r = json.load(open(root / final / "results.json"))
    assert r["rule"] == c["final"]["rule"] and len(r["summary"]["per_seed"]) == 2
    assert r["speed"]["classifier_seconds_per_24h_cpu"] >= 0 and r["speed"]["parameters"] > 0
    assert (root / final / "pr_curve.csv").exists() and (root / final / "pr_curve.svg").exists()
    names = sorted(d for d in os.listdir(root) if (root / d / "results.json").exists())
    assert "ctx2_noise1_rand200_neg1_hard0" in names or any(n.startswith("ctx2") for n in names)
    assert any(n.endswith("hard1") for n in names)
    table = (root / "improve.md").read_text()
    assert "Final model" in table and table.count("\n| ctx") == len(names)
    assert (root / "progress.txt").exists() and (root / "rules_rules.json").exists()
    info = json.load(open(root / "ctx0_noise3_rand200_neg1_hard0" / "training.json"))["runs"][0]
    base = json.load(open(root / "ctx0_noise1_rand200_neg1_hard0" / "training.json"))["runs"][0]
    assert info["negatives_from_noise"] > base["negatives_from_noise"]
    two = json.load(open(root / "ctx0_noise1_rand200_neg2_hard0" / "training.json"))["runs"][0] \
        if (root / "ctx0_noise1_rand200_neg2_hard0").exists() else None
    if two:
        assert two["negatives_per_rumble"] > base["negatives_per_rumble"]  # the fake days run out of gaps before 2

    # resuming retrains nothing
    t = {n: os.path.getmtime(root / n / "model_seed0.pth") for n in names}
    im.main(common)
    assert all(os.path.getmtime(root / n / "model_seed0.pth") == t[n] for n in names)

    im.main(common + ["final"])
    fin = (root / "final.md").read_text()
    assert "baseline" in fin and "6 final alarm rules" in fin and "Sample B" in fin
    assert (root / "final_detections" / "sampleB_detections.txt").exists()
    assert json.load(open(root / "final_detections" / "final_model.json"))["experiment"] == final
    assert c["3"]["winner"] in c["combos"]["ranked"]  # the stage winners are always among the combinations

    # an experiment that would not fit in memory stops early instead of swapping
    monkeypatch.setattr(im, "total_ram_bytes", lambda: 10_000)
    cfg = json.load(open(out / im.BASE_CONFIG / "config.json"))
    with pytest.raises(MemoryError):
        im.build_training_set(cfg, dc.load_pools(str(source)), str(out / "_embedding_cache"), dict(im.BASE), {}, [0, 1])
    after = snapshot(source)
    assert not [p for p, m in before.items() if after.get(p) != m and not p.startswith(str(out))]
    assert not [p for p in after if p not in before and not p.startswith(str(out))]
