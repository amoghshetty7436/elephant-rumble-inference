"""tools/detect.py with a fake embedder: Raven tables and timing, one or two processes."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")
import detect  # noqa: E402
import experiments as ex  # noqa: E402
import improve as im  # noqa: E402


def make_final(tmp_path, ctx=2):
    torch.manual_seed(0)
    m = ex.make_model(ctx)
    model = tmp_path / "model_seed0.pth"
    torch.save(m.state_dict(), model)
    rule = {"on": 0.2, "min_frames": 2, "smooth": 3, "gap": 2, "score": "max"}
    final = tmp_path / "final_model.json"
    final.write_text(json.dumps({"experiment": f"E2_ctx{ctx}_noise10_rand3000_gap0.5_hard0_augnone",
                                 "model": str(model), "rule": rule, "cut_off": 0.3}))
    return final, m.eval(), rule


def make_recordings(d, n=3, minutes=4, sr=8000):
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        y = 0.1 * rng.standard_normal(sr * 60 * minutes).astype(np.float32)
        t = np.arange(sr * 8) / sr
        y[sr * 30:sr * 38] += 0.5 * np.sin(2 * np.pi * 20 * t).astype(np.float32)  # a "rumble"
        sf.write(d / f"day{i}.wav", y, sr)
    return d


@pytest.mark.parametrize("workers", [1, 2])
def test_detect_tables_and_timing(tmp_path, workers):
    final, model, rule = make_final(tmp_path)
    rec = make_recordings(tmp_path / "rec")
    out = tmp_path / "out"
    s = detect.main([str(rec), "--final", str(final), "--out", str(out), "--device", "cpu",
                     "--workers", str(workers), "--threads", "1", "--fake-embedder", "--source", str(tmp_path)])
    assert s["files"] == 3 and abs(s["audio_hours"] - 0.2) < 0.01
    assert set(s["step_s_per_24h"]) >= set(detect.STEPS)
    assert (out / "timing.md").exists() and json.load(open(out / "timing.json"))["cut_off"] == 0.3
    # each table holds exactly the alarms the study's own functions give at the cut-off
    import fp_replica
    from dataset_configs import FakeEmbedder
    for i in range(3):
        y = fp_replica.load_audio(str(rec / f"day{i}.wav"))
        emb = FakeEmbedder().whole_file(y).half().float()
        prob = ex.day_probs(model, emb, 2, "cpu")
        want = sum(1 for sc in im.alarms(prob, rule)[2] if sc >= 0.3)
        rows = (out / f"day{i}.rumbles.txt").read_text().strip().splitlines()
        assert rows[0].startswith("Selection") and len(rows) - 1 == want
    assert not [p for p in os.listdir(rec) if not p.endswith(".wav")]  # nothing written next to the audio


def test_reading_ahead_gives_the_same_tables(tmp_path):
    final, _, _ = make_final(tmp_path)
    rec = make_recordings(tmp_path / "rec", n=4, minutes=2)
    for flag, name in ((["--readers", "1"], "a"), (["--readers", "0"], "b"), (["--readers", "3"], "c")):
        detect.main([str(rec), "--final", str(final), "--out", str(tmp_path / name), "--device", "cpu",
                     "--threads", "1", "--fake-embedder", "--source", str(tmp_path)] + flag)
    for i in range(4):
        a = (tmp_path / "a" / f"day{i}.rumbles.txt").read_text()
        assert a == (tmp_path / "b" / f"day{i}.rumbles.txt").read_text() == (tmp_path / "c" / f"day{i}.rumbles.txt").read_text()


def test_half_precision_is_compared_with_full(tmp_path):
    final, _, _ = make_final(tmp_path)
    rec = make_recordings(tmp_path / "rec", n=2, minutes=2)
    s = detect.main([str(rec), "--final", str(final), "--out", str(tmp_path / "o"), "--device", "cpu",
                     "--threads", "1", "--fake-embedder", "--half", "--source", str(tmp_path)])
    c = s["half_check"]
    assert c and c["cosine_mean"] > 0.99 and c["alarms_half"] >= 0
    assert "Half precision against full" in (tmp_path / "o" / "timing.md").read_text()
