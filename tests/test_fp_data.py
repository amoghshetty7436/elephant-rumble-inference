import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import fp_data  # noqa: E402

HEADER = "Selection\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tFile Offset (s)\tBegin File\n"


def test_read_labels_pools_tables_and_drops_rows_without_end(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text(HEADER + "1\t90000\t90004\t10\t50\t300\tnn01a_20180101_000000.wav\n"
                          "2\t90010\t\t10\t50\t310\tnn01a_20180101_000000.wav\n")
    b.write_text(HEADER + "1\t5\t8\t10\t50\t100\tDZAN_20120201_000000.wav\n"
                          "2\t1\t3\t10\t50\t50\tnn01a_20180101_000000.wav\n")
    labels = fp_data.read_labels([a, b])
    assert [l.start for l in labels["nn01a_20180101_000000.wav"]] == [50, 300]
    assert labels["nn01a_20180101_000000.wav"][1].duration == 4
    assert "dz_20120201_000000.wav" in labels


def test_negative_windows_match_notebook_rules():
    L = fp_data.Label
    f = "x.wav"
    labels = [L(100, 5, 10, 50, f),   # first label -> window [0, 60)
              L(110, 5, 10, 50, f),   # gap 5 s: skipped
              L(200, 5, 20, 60, f),   # gap 85 s: one window between, 5 s margins
              L(400, 5, 10, 50, f),   # gap 195 s: two 60 s windows
              L(800, 5, 10, 50, f)]   # gap 395 s: two 60 s windows + 120 s in the middle
    neg = fp_data.negative_windows(labels)
    assert [(n.start, n.duration) for n in neg] == [
        (0, 60), (120, 75), (210, 60), (335, 60), (410, 60), (735, 60), (542.5, 120)]


def test_usable_files_needs_two_labels():
    L = fp_data.Label
    labels = {"a.wav": [L(100, 5, 0, 0, "a.wav")],
              "b.wav": [L(100, 5, 0, 0, "b.wav"), L(300, 5, 0, 0, "b.wav")],
              "c.wav": [L(100, 5, 0, 0, "c.wav"), L(300, 5, 0, 0, "c.wav")]}
    used, skipped = fp_data.usable_files(labels, {"a.wav": "/a.wav", "b.wav": "/b.wav"})
    assert used == ["b.wav"]
    assert skipped == {"a.wav": "fewer than 2 labels", "c.wav": "no sound file found"}


def test_time_to_index():
    assert fp_data.time_to_index(0.64) == 1 and fp_data.time_to_index(500) == 781
