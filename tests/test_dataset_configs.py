"""End-to-end test of tools/dataset_configs.py on a tiny fake Datasets tree."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")
import dataset_configs as dc  # noqa: E402

SR = 2000
MINUTES = 6
RAVEN = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)",
         "High Freq (Hz)", "Begin Path", "File Offset (s)", "Begin File"]


def wav(path, rumbles, seconds=MINUTES * 60, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.05, seconds * SR).astype(np.float32)
    t = np.arange(int(4 * SR)) / SR
    for s in rumbles:
        i = int(s * SR)
        y[i:i + len(t)] += 0.5 * np.sin(2 * np.pi * 30 * t).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, SR)


def rumble_times(seed):
    rng = np.random.default_rng(seed)
    return sorted(set(int(x) for x in rng.uniform(70, MINUTES * 60 - 20, 6)) | {30})


def raven_rows(name, times, sel=1):
    return [[sel + i, "Spectrogram 1", 1, 1000 + s, 1004 + s, 20, 45, "X:\\" + name, s, name]
            for i, s in enumerate(times)]


def table(path, rows, header=RAVEN):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "Dataset"
    j = lambda p: str(root / dc.rel(p))  # noqa: E731
    seed = [0]

    def add(folder, name, times, **k):
        seed[0] += 1
        wav(os.path.join(j(folder), name), times, seed=seed[0], **k)

    rows = []
    for n in ["ceb1_20111010_000000.wav", "ceb1_20111011_000000.wav", "ceb4_20111012_000000.wav"]:
        t = rumble_times(len(rows))
        add(r"Training\Sound_Files\rumble_clearing", n, t)
        rows += raven_rows(n, t, len(rows) + 1)
    table(j(dc.POOLS["clearing"]["tables"][0]), rows)

    rows = []
    for n in ["nn01a_20180101_000000.wav", "nn01a_20180102_000000.wav", "nn02b_20180103_000000.wav",
              "nn03f_20180104_000000.wav", "nn04a_20180608_000000.wav"]:
        t = rumble_times(len(rows) + 7)
        add(r"Training\Sound_Files\rumble_forest\TrainingSet_sounds", n, t)
        rows += raven_rows(n, t, len(rows) + 1)
    add(r"Training\Sound_Files\rumble_forest\TrainingSet_sounds", "nn05c_20180105_000000.wav", [])
    rows.append([99, "Spectrogram 1", 1, 0, 9.8, 0, 100, "x", 0.3, "nn05c_20180105_000000.wav"])
    table(j(dc.POOLS["forest"]["tables"][0]), rows, RAVEN)
    # mark the last row as a DUMMY_NoEles placeholder through an extra column
    p = j(dc.POOLS["forest"]["tables"][0])
    lines = open(p).read().splitlines()
    lines[0] += "\tTag 1"
    lines = [lines[0]] + [l + "\t" for l in lines[1:-1]] + [lines[-1] + "\tDUMMY_NoEles"]
    open(p, "w").write("\n".join(lines) + "\n")

    # Complex: an empty-named column, one elephant-free day
    hdr = RAVEN[:7] + [""] + RAVEN[7:] + ["Tag 1"]
    rows = []
    for n in ["nn04d_20180721_000000.wav", "nn06b_20180801_000000.wav"]:
        t = rumble_times(len(rows) + 20)
        add(r"Testing\Sounds\RumbleComplex-Forest", n, t)
        rows += [r[:7] + [""] + r[7:] + [""] for r in raven_rows(n, t, len(rows) + 1)]
    add(r"Testing\Sounds\RumbleComplex-Forest", "nn05c_20180416_000000.wav", [])
    rows.append([98, "Spectrogram 1", 1, 5, 14.8, 0, 100, "", "x", 0.28, "nn05c_20180416_000000.wav", "DUMMY_NoEles"])
    table(j(dc.POOLS["complex"]["tables"][0]), rows, hdr)

    # General v4 layout
    rows = []
    for n in ["nn01d_20180127_000000.wav", "nn07a_20180205_000000.wav"]:
        t = rumble_times(len(rows) + 30)
        add(r"Testing\Sounds\RumbleGeneral-Forest", n, t)
        rows += [[len(rows) + i + 1, 45, s, n, 4, ""] for i, s in enumerate(t)]
    table(j(dc.POOLS["general"]["tables"][0]), rows,
          ["Selection", "High Freq (Hz)", "File Offset (s)", "filename", "duration", "marginals"])

    # Sample A: the truth table calls nn04a_20200429_221358 "20200428"; one day is not on the schedule
    sa = r"Testing\Sounds\Nouabale-Ndoki_Grid-Forest\24-hr"
    t1, t2 = rumble_times(40), rumble_times(41)
    add(sa, "nn01c_20180111_000000.wav", t1)
    add(sa, "nn04a_20200429_221358.wav", t2)
    add(sa, "nn02d_20200112_000000.wav", rumble_times(42))
    table(j(dc.POOLS["sampleA"]["tables"][0]), raven_rows("nn01c_20180111_000000.wav", t1))
    table(j(dc.POOLS["sampleA"]["tables"][1]), raven_rows("nn04a_20200428_221358.wav", t2))

    # Sample B: day list from raw_reviewed_tables; nn05b_20230116 is a HOLD day (no table)
    sb = r"Testing\Sounds\Nouabale-Ndoki_Grid-Forest-Additional2024\sounds"
    rows = []
    for n in ["nn08c_20230104_000000.wav", "nn01b_20231007_000000.wav", "nn06e_20231025_000000.wav"]:
        t = rumble_times(len(rows) + 50)
        add(sb, n, t)
        rows += raven_rows(n, t, len(rows) + 1)
        raw = j(dc.POOLS["sampleB"]["extra_folders"][0])
        table(os.path.join(raw, n + "_SKT1.txt"), [])
    add(sb, "nn05b_20230116_000000.wav", [])
    table(j(dc.POOLS["sampleB"]["tables"][0]), rows)

    add(r"Testing\Sounds\Vehicle", "Kakum2018_8kHz_E02_20180715_000000.wav", [])
    add(r"Testing\Sounds\Vehicle", "nn05c_20190214_000000.wav", [])
    add(r"Training\Sound_Files\TrainingClips\thunder_8kHz", "thunder1.wav", [], seconds=30)
    add(r"Training\Sound_Files\TrainingClips\rain_8kHz", "rain1.wav", [], seconds=30)
    return root


def snapshot(folder):
    return {os.path.join(d, n): os.path.getmtime(os.path.join(d, n))
            for d, _, ns in os.walk(folder) for n in ns}


def test_pools(source):
    pools = dc.load_pools(str(source))
    assert pools["sampleA"].files == ["nn01c_20180111_000000", "nn04a_20200429_221358"]
    assert len(pools["sampleA"].labels["nn04a_20200429_221358"]) == 7
    assert sorted(pools["sampleB"].files) == ["nn01b_20231007_000000", "nn06e_20231025_000000",
                                              "nn08c_20230104_000000"]
    assert pools["vehicle"].files == ["kakum2018_8khz_e02_20180715_000000"]
    assert "nn05c_20180416_000000" in pools["complex"].dummy
    assert "nn05c_20180416_000000" not in pools["complex"].labels
    assert "nn05c_20180416_000000" in pools["complex"].files
    assert "nn05c_20180105_000000" not in pools["forest"].files  # DUMMY only: no rumbles to train on
    assert pools["general"].labels["nn01d_20180127_000000"][0].duration == 4


def test_refuses_original_data(source):
    with pytest.raises(SystemExit):
        dc.Writer([r"L:\ELP\Detectors\Elephant\Datasets\Configurations"], [dc.ORIGINAL_DATA])
    w = dc.Writer([str(source / "Configurations")], [str(source / "Training")])
    with pytest.raises(RuntimeError):
        w.check(str(source / "Training" / "x.txt"))


def test_all_steps(source, monkeypatch):
    monkeypatch.setattr(dc, "HELDOUT_FOREST_DAYS", 2)
    before = {k: v for k, v in snapshot(source).items()}
    out = source / "Configurations"
    common = ["--source", str(source), "--minutes", "0.02", "--fake-embedder"]
    dc.main(common)  # one command: A-D, pick the best on validation, E, pick again, F-H

    names = sorted(d for d in os.listdir(out) if not d.startswith("_") and os.path.isdir(out / d))
    choice = (out / "best_choice.txt").read_text().splitlines()
    best = choice[-1].split("picked ")[1][0]
    base = {"A": "clearing+forest", "B": "forest", "C": "clearing", "D": "forest+clearing3days",
            "E": choice[0].split("picked ")[1][0]}
    base["E"] = base[base["E"]] + "+noise"
    n_val = len(dc.heldout_forest_days(dc.Pool("forest", str(source))))  # whole sites, so 2 or more
    assert names == ["A_train-clearing+forest", "B_train-forest", "C_train-clearing",
                     "D_train-forest+clearing3days",  # 35 forest rumbles need all 3 clearing days (7 each)
                     "E_train-" + base["E"],
                     f"F_train-{base[best]}+complex+general_val-{n_val}forestdays",
                     f"G_train-{base[best]}+sampleA",
                     f"H_train-{base[best]}+complex+general+sampleA_val-{n_val}forestdays"]
    assert (out / "run_log.txt").read_text().count("results in") == 8

    a = out / "A_train-clearing+forest"
    assert (a / "Training/Selection_Tables/rumble_clearing/rumble_clearing_00-24hr_56days.txt").exists()
    lst = (a / "Training/Sound_Files/rumble_clearing/sound_files.txt").read_text().split()
    assert len(lst) == 3 and all(os.path.exists(p) for p in lst)
    assert (a / "Validation/Truth_Tables/nn_ele_00-24hr_GeneralTest_v4.txt").exists()
    assert (a / "Testing/Truth_Tables/Nouabale-Ndoki_Grid-Forest-Additional2024/raw_reviewed_tables").is_dir()
    assert (a / "Testing/Sounds/Vehicle/sound_files.txt").read_text().count(".wav") == 1

    f = out / names[5]
    assert (f / "Training/Sounds/RumbleComplex-Forest/sound_files.txt").exists()
    val = (f / "Validation/Sound_Files/rumble_forest/TrainingSet_sounds/sound_files.txt").read_text().split()
    tr = (f / "Training/Sound_Files/rumble_forest/TrainingSet_sounds/sound_files.txt").read_text().split()
    assert len(val) >= 2 and not set(val) & set(tr)

    for n in names:
        m = json.load(open(out / n / "results" / "metrics.json"))
        assert 0.5 <= m["threshold_from_validation"] <= 1
        assert "sampleB" in m["sets"] and "vehicle" in m["sets"]
        assert ("sampleA" in m["sets"]) == ("sampleA" not in m["components"])
        assert m["sets"]["sampleB"]["fruitpunch_method"]["support"] > 0
        assert (out / n / "results" / "summary.md").read_text().count("|") > 20
        assert (out / n / "results" / "detections" / "sampleB_detections.txt").exists()
    g = json.load(open(out / names[6] / "config.json"))
    assert "Test 2: Sample A" not in [t["name"] for t in g["tests"]]
    d = json.load(open(out / names[3] / "results" / "metrics.json"))["training"]
    by = d["rumble_frames_by_habitat"]
    assert by["clearing"] == by["forest"] and d["negatives_used_from_gaps"] == d["rumble_frames"]
    a_ = json.load(open(out / names[0] / "results" / "metrics.json"))["training"]
    assert a_["rumble_frames_by_habitat"]["clearing"] != a_["rumble_frames_by_habitat"]["forest"]
    e = json.load(open(out / names[4] / "results" / "metrics.json"))
    assert e["training"]["negatives_used_from_noise_and_random"] > 0
    days = (out / names[0] / "results" / "elp_method_sampleB_per_day.csv").read_text().splitlines()
    assert len(days) == 4 and "habitat" in days[0]
    prog = (out / "progress.txt").read_text()
    assert "Finished (expected -> actual)" in prog and "setup H_" in prog
    assert set(json.load(open(out / "_timing.json"))) >= {"embed_s_per_audio_hour", "score_s_per_day"}
    assert (out / "comparison.md").read_text().count("\n|") >= 9

    # repeat runs of E on two bases, compared at a target recall; first-run results left alone
    e_first = out / names[4] / "results" / "metrics.json"
    stamp = e_first.stat().st_mtime
    for base_letter in ("C", "A"):
        dc.main(common + ["--configs", "E", "--best", base_letter, "--seeds", "0", "1"])
    for folder in ("E_train-clearing+noise", "E_train-clearing+forest+noise"):
        for seed in (0, 1):
            m = json.load(open(out / folder / f"results_seed{seed}" / "metrics.json"))
            assert m["settings"]["seed"] == seed and m["target_recall"] == 0.9
            assert 0.5 <= m["threshold_for_target_recall"] <= 1
            assert "elp_at_target_recall_threshold" in m["sets"]["validation"]
    assert e_first.stat().st_mtime == stamp
    seeds = (out / "comparison_seeds.md").read_text()
    assert "E_train-clearing+noise | 1 |" in seeds and "E_train-clearing+forest+noise | min-max" in seeds
    assert "Fewest validation false alarms at 90% recall" in seeds

    # the dataset folders themselves are untouched
    after = snapshot(source)
    changed = [p for p, t in before.items() if after.get(p) != t]
    new = [p for p in after if p not in before and not p.startswith(str(out))]
    assert not changed and not new
