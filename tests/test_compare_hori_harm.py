"""tools/compare_hori_harm.py on a small made-up deployment where every answer is known."""
import csv
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "ElephantRumbleDetector"))
import compare_hori_harm as ch  # noqa: E402
import rumble_detector as rd  # noqa: E402

F = rd.FRAME_SECONDS
HH_HEADER = ["Selection", "Begin Time (s)", "End Time (s)", "File Offset (s)", "Begin Path", "Begin File",
             "Ambiguous", "Count", "Notes", "Score", "Site"]


def bump(prob, t0, t1, height):
    prob[int(t0 / F):int(t1 / F)] = height


def make(tmp, probs=True):
    """Two recordings at one site. Ours: probabilities with four events; Hori-Harm: six boxes."""
    rule = rd.json.load(open(os.path.join(rd.MODEL_DIR, "settings.json")))["rule"]
    out = tmp / "ours" / "aa01a_202306_jun"
    out.mkdir(parents=True)
    n = int(3600 / F)
    p1 = np.zeros(n, np.float32)
    bump(p1, 100, 110, 0.999)   # on HH rumble 1 (found at 0.993)
    bump(p1, 500, 508, 0.95)    # on HH rumble 2 (only found at cut-offs <= 0.95)
    bump(p1, 1000, 1010, 0.998)  # on a HH false alarm
    bump(p1, 2000, 2010, 0.997)  # on no HH box
    p2 = np.zeros(n, np.float32)
    files = {"aa01a_20230701_000000.wav": p1, "aa01a_20230702_000000.wav": p2}
    for name, p in files.items():
        base = str(out / name[:-4])
        rd.write_table(base + rd.TABLE_SUFFIX, rd.file_rows(p, rule, 0.993, "X:\\" + name))
        if probs:
            np.save(base + rd.PROBS_SUFFIX, p.astype(np.float16))
    boxes = [  # file, offset, length, count, score
        ("aa01a_20230701_000000.wav", 103.0, 2.0, 1, 0.7),     # rumble we find
        ("aa01a_20230701_000000.wav", 503.0, 2.0, 2, 0.5),     # rumble found only at 0.95
        ("aa01a_20230701_000000.wav", 1003.0, 2.0, 0, 0.45),   # false alarm we also fire on
        ("aa01a_20230701_000000.wav", 3000.0, 2.0, 1, 0.9),    # rumble we miss
        ("aa01a_20230702_000000.wav", 50.0, 2.0, 0, 0.41),     # false alarm, we are silent
        ("bb01a_20230702_000000.wav", 50.0, 2.0, 1, 0.6),      # recording not in our results
    ]
    hh = tmp / "hh.txt"
    with open(hh, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(HH_HEADER)
        for i, (name, off, ln, c, s) in enumerate(boxes, 1):
            w.writerow([i, off, off + ln, off, "X:\\" + name, name, "", c, "", s, name[:5]])
        w.writerow([99, 103.0, 105.0, 103.0, "X:\\a", "aa01a_20230701_000000.wav", "", "", "", 0.7, "aa01a"])  # no Count
        w.writerow([100, 103.0, 105.0, 103.0, "X:\\a", "aa01a_20230701_000000.wav", "", 1, "", 0.7, "aa01a"])  # duplicate
    return hh, tmp / "ours"


def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def test_known_answers(tmp_path):
    hh, ours = make(tmp_path)
    out = tmp_path / "cmp"
    assert ch.main(["--hh", str(hh), "--ours", str(ours), "--out", str(out), "--sample", "5"]) == 0
    rep = (out / "report.md").read_text()
    assert "6 reviewed detections on 3 recordings" in rep
    assert "1 rows with no Count" in rep and "1 exact duplicates" in rep
    assert "**2 of those recordings are in our results**" in rep and "sites bb01a" in rep
    assert "the saved probabilities give back exactly the saved table" in rep
    cuts = {r[0]: r for r in read(out / "our_cutoffs.csv")[1:]}
    m = cuts["0.993"]
    # our detections, per day, HH rumbles found, %, calls found, %, HH FAs, %, on rumble, on FA, on none
    assert m[1:4] == ["3", "1.5", "1"] and m[7] == "1" and m[9:12] == ["1", "1", "1"]
    assert m[5] == "1"  # calls: the box we find holds 1 of the 4 calls
    assert cuts["0.95"][3] == "2" and cuts["0.95"][5] == "3"  # the 0.95 event joins in
    missed = read(out / "hori_harm_rumbles_we_missed.txt")
    assert len(missed) == 3  # header + the 0.95 rumble + the one at 3000 s
    unmatched = [r for r in csv.reader(open(out / "ours_not_in_hori_harm.txt"), delimiter="\t")][1:]
    assert len(unmatched) == 1 and abs(float(unmatched[0][8]) - 2000) < 1
    site = read(out / "per_site.csv")[1]
    assert site[:6] == ["aa01a", "2", "5", "3", "1", "33.3%"]


def test_tables_only(tmp_path):
    hh, ours = make(tmp_path, probs=False)
    out = tmp_path / "cmp"
    assert ch.main(["--hh", str(hh), "--ours", str(ours), "--out", str(out)]) == 0
    rep = (out / "report.md").read_text()
    assert "one cut-off only" in rep
    assert [r[0] for r in read(out / "our_cutoffs.csv")[1:]] == ["0.993", "0.995", "0.997", "0.999"]


def test_self_check_catches_a_wrong_table(tmp_path):
    hh, ours = make(tmp_path)
    t = next(ours.rglob("aa01a_20230701_000000" + rd.TABLE_SUFFIX))
    lines = t.read_text().splitlines()
    t.write_text("\n".join(lines[:-1]) + "\n")  # drop one detection from the table
    out = tmp_path / "cmp"
    ch.main(["--hh", str(hh), "--ours", str(ours), "--out", str(out)])
    assert "do NOT give back their table" in (out / "report.md").read_text()
