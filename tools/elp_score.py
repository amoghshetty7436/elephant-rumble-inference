#!/usr/bin/env python
"""
Score a rumble detector the way ELP's "Detector Validation" tool does.

A detection is a true positive when its box overlaps a truth box by more
than OVERLAP (default 1%) in time AND in frequency, within the same sound
file. Precision / recall / false-positives-per-hour are reported for each
score threshold (default steps of 0.05), using the detections' Score column.

Times are compared per sound file using "File Offset (s)" + box duration,
because "Begin Time (s)" in a Raven table is measured from the first file of
the table, not from the start of each file.

Usage:
    python tools/elp_score.py --truth truth1.txt [truth2.txt ...] \
        --detections det1.txt [det2.txt ...] \
        --files schedule.txt          # one sound-file name per line
        [--hours-per-file 24] [--overlap 0.01] [--step 0.05] [--csv out.csv]

`--files` lists every sound file that was reviewed, including days with no
rumbles. Detections and truth rows in other files are ignored. Without it,
the files named in the truth tables are used, which leaves out elephant-free
days and makes precision look better than it is.
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Box:
    file: str
    t0: float
    t1: float
    f0: float
    f1: float
    score: float = 1.0


def normalize_file_name(name):
    name = os.path.basename(name.replace("\\", "/")).strip().strip('"').lower()
    name = re.sub(r"^dzan_", "dz_", name)
    return re.sub(r"\.(wav|flac|aiff?)$", "", name)


def _float(row, *keys):
    for k in keys:
        v = row.get(k)
        if v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                pass
    return None


def read_raven_table(path, default_duration=4.0, score_column="Score"):
    """Read a Raven selection table into Boxes with per-file times."""
    boxes = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}

        def col(name):
            return cols.get(name.lower())

        for raw in reader:
            row = {k.strip() if k else k: v for k, v in raw.items()}
            file_col = col("Begin File") or col("Begin Path")
            if not file_col or not row.get(file_col):
                continue
            begin = _float(row, col("Begin Time (s)") or "")
            end = _float(row, col("End Time (s)") or "")
            offset = _float(row, col("File Offset (s)") or "")
            duration = _float(row, col("Delta Time (s)") or "", col("duration (s)") or "")
            if duration is None and begin is not None and end is not None:
                duration = end - begin
            if duration is None:
                duration = default_duration  # e.g. RumbleOverlap tables: offset only
            t0 = offset if offset is not None else begin
            if t0 is None:
                continue
            f0 = _float(row, col("Low Freq (Hz)") or "")
            f1 = _float(row, col("High Freq (Hz)") or "")
            score = _float(row, col(score_column) or "")
            boxes.append(Box(
                file=normalize_file_name(row[file_col]),
                t0=t0, t1=t0 + duration,
                f0=f0 if f0 is not None else 0.0,
                f1=f1 if f1 is not None else 1e9,
                score=score if score is not None else 1.0,
            ))
    return boxes


def overlap_fraction(a0, a1, b0, b1):
    """Overlap as a fraction of the shorter interval (0 when disjoint)."""
    inter = min(a1, b1) - max(a0, b0)
    shortest = min(a1 - a0, b1 - b0)
    if inter <= 0 or shortest <= 0:
        return 0.0
    return inter / shortest


def matches(det, truth, overlap):
    return (overlap_fraction(det.t0, det.t1, truth.t0, truth.t1) > overlap
            and overlap_fraction(det.f0, det.f1, truth.f0, truth.f1) > overlap)


def score(truth, detections, files, hours_per_file=24.0, overlap=0.01, step=0.05):
    files = {normalize_file_name(f) for f in files}
    truth_by_file = defaultdict(list)
    for t in truth:
        if t.file in files:
            truth_by_file[t.file].append(t)
    dets = [d for d in detections if d.file in files]
    n_truth = sum(len(v) for v in truth_by_file.values())
    hours = hours_per_file * len(files)

    # For each detection: does it hit any truth box, and which ones?
    det_hits = []
    for d in dets:
        hit = [i for i, t in enumerate(truth_by_file[d.file]) if matches(d, t, overlap)]
        det_hits.append((d, hit))

    results = []
    n_steps = int(round(1 / step))
    for k in range(n_steps + 1):
        thr = round(k * step, 6)
        kept = [(d, hit) for d, hit in det_hits if d.score >= thr]
        tp_det = sum(1 for _, hit in kept if hit)
        fp = len(kept) - tp_det
        found = {(d.file, i) for d, hit in kept for i in hit}
        tp_truth = len(found)
        fn = n_truth - tp_truth
        precision = tp_det / len(kept) if kept else float("nan")
        recall = tp_truth / n_truth if n_truth else float("nan")
        results.append(dict(
            threshold=thr, detections=len(kept), tp_detections=tp_det,
            fp=fp, rumbles=n_truth, rumbles_found=tp_truth, fn=fn,
            precision=precision, recall=recall,
            fp_per_hour=fp / hours if hours else float("nan"),
        ))
    return results


def read_file_list(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truth", nargs="+", required=True)
    p.add_argument("--detections", nargs="+", required=True)
    p.add_argument("--files", help="text file listing every reviewed sound file (one per line)")
    p.add_argument("--hours-per-file", type=float, default=24.0)
    p.add_argument("--overlap", type=float, default=0.01)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--csv", help="also write the table to this CSV file")
    a = p.parse_args(argv)

    truth = [b for t in a.truth for b in read_raven_table(t)]
    dets = [b for t in a.detections for b in read_raven_table(t)]
    if a.files:
        files = read_file_list(a.files)
    else:
        files = sorted({b.file for b in truth})
        print("WARNING: no --files given; scoring only files that contain truth rumbles. "
              "Elephant-free days are left out, so precision will look too high.", file=sys.stderr)

    rows = score(truth, dets, files, a.hours_per_file, a.overlap, a.step)
    print(f"{len(files)} files, {rows[0]['rumbles']} truth rumbles, {len(dets)} detections loaded")
    print(f"{'thr':>5} {'dets':>6} {'TP':>6} {'FP':>6} {'found':>6} {'FN':>6} {'prec':>6} {'recall':>6} {'FP/h':>6}")
    for r in rows:
        print(f"{r['threshold']:>5.2f} {r['detections']:>6} {r['tp_detections']:>6} {r['fp']:>6} "
              f"{r['rumbles_found']:>6} {r['fn']:>6} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['fp_per_hour']:>6.2f}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


if __name__ == "__main__":
    main()
