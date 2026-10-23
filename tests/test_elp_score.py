import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import elp_score  # noqa: E402

HEADER = "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tFile Offset (s)\tBegin File\tScore\n"


def write_table(path, rows):
    with open(path, "w") as f:
        f.write(HEADER)
        for i, (file, offset, dur, lf, hf, score) in enumerate(rows, 1):
            # Begin Time deliberately offset as if from an earlier file in the table
            f.write(f"{i}\tSpectrogram 1\t1\t{86400 + offset}\t{86400 + offset + dur}\t{lf}\t{hf}\t{offset}\t{file}\t{score}\n")


def test_overlap_and_thresholds(tmp_path):
    truth = tmp_path / "truth.txt"
    dets = tmp_path / "dets.txt"
    write_table(truth, [
        ("nn01a_20200101_000000.wav", 100.0, 5.0, 15, 60, ""),
        ("nn01a_20200101_000000.wav", 500.0, 5.0, 15, 60, ""),
    ])
    write_table(dets, [
        ("nn01a_20200101_000000.wav", 102.0, 6.0, 5, 250, 0.9),   # hits rumble 1
        ("nn01a_20200101_000000.wav", 900.0, 4.0, 5, 250, 0.3),   # false alarm, low score
        ("nn01a_20200101_000000.wav", 500.0, 5.0, 300, 400, 0.8),  # right time, wrong frequency -> FP
        ("nn99z_20200101_000000.wav", 100.0, 5.0, 15, 60, 0.9),   # file not scheduled -> ignored
    ])
    files = ["nn01a_20200101_000000.wav", "nn02b_20200101_000000.wav"]  # 2nd day has no rumbles
    rows = elp_score.score(elp_score.read_raven_table(truth), elp_score.read_raven_table(dets), files)
    at = {r["threshold"]: r for r in rows}
    assert at[0.0]["detections"] == 3 and at[0.0]["tp_detections"] == 1 and at[0.0]["fp"] == 2
    assert at[0.0]["rumbles"] == 2 and at[0.0]["rumbles_found"] == 1
    assert abs(at[0.0]["fp_per_hour"] - 2 / 48) < 1e-9
    assert at[0.5]["detections"] == 2 and at[0.5]["precision"] == 0.5
    assert at[0.85]["detections"] == 1 and at[0.85]["precision"] == 1.0 and at[0.85]["recall"] == 0.5


def test_offset_only_rows_use_default_duration(tmp_path):
    t = tmp_path / "overlap.txt"
    t.write_text("Selection\tFile Offset (s)\tBegin File\tLow Freq (Hz)\tHigh Freq (Hz)\n1\t10\tdzan_20120201_000000.wav\t10\t50\n")
    (box,) = elp_score.read_raven_table(t)
    assert box.file == "dz_20120201_000000" and box.t0 == 10 and box.t1 == 14
