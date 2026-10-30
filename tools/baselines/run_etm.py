#!/usr/bin/env python
"""
Run the EarthToolsMaker forest-elephant rumble detector (YOLOv8 on 164 s spectrogram
images, github.com/earthtoolsmaker/forest-elephants-rumble-detection) on every sound
file of a list written by `tools/baselines.py list`. Runs in its own environment
(ultralytics); see tools/baselines.py for the whole procedure.

It follows their predict_raven.py: same spectrogram images (their own
waveform_to_np_image), same 164 s windows with 10 s overlap, same box-to-time
conversion. Two changes, both needed for a fair score:

  - the confidence floor is 0.01 instead of ultralytics' 0.25, so the scorer can
    sweep the whole cut-off range (0.25 is one point on that curve);
  - the end of the file is covered (their chunking drops the last partial window),
    and a rumble seen twice in the 10 s overlap of two windows is kept once.

Output: <out>/<pool>/<file key>.csv with t_start, t_end, f_low, f_high, score
(seconds from the start of the file, Hz). Finished files are skipped on a rerun.

  python tools/baselines/run_etm.py --etm-repo C:/code/forest-elephants-rumble-detection ^
      --list <out>/files.tsv --out <out>/etm
"""
import argparse
import csv
import os
import sys
import time

WINDOW = 164.0  # seconds per image (their config.yaml)
OVERLAP = 10.0
CONFIG = dict(n_fft=4096, hop_length=1024, freq_max=250.0, width=640, height=256)
FREQ_MIN, FREQ_MAX = 0.0, 250.0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def window_offsets(total, window=WINDOW, overlap=OVERLAP):
    """Window starts every (window - overlap) s; the last window ends at the end of the file."""
    if total <= window:
        return [0.0]
    step = window - overlap
    offs, o = [], 0.0
    while o + window < total:
        offs.append(o)
        o += step
    offs.append(total - window)
    return offs


def time_iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def dedupe(boxes, iou=0.5):
    """boxes: (t0, t1, f0, f1, score, window). Drop a box that repeats a higher-scoring box
    of another window (both windows see the 10 s overlap)."""
    kept = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        if any(k[5] != b[5] and time_iou(k, b) > iou and min(k[3], b[3]) > max(k[2], b[2]) for k in kept):
            continue
        kept.append(b)
    return sorted(kept)


def detect_file(model, to_image, path, conf, batch, device):
    import numpy as np
    import soundfile as sf
    import torch
    from PIL import Image
    info = sf.info(path)
    sr, total = info.samplerate, info.frames / info.samplerate
    offs = window_offsets(total)
    boxes = []
    for b0 in range(0, len(offs), batch):
        part = offs[b0:b0 + batch]
        images = []
        for o in part:
            y, _ = sf.read(path, start=int(round(o * sr)), frames=int(round(WINDOW * sr)),
                           dtype="float32", always_2d=True)
            wav = torch.from_numpy(np.ascontiguousarray(y[:, 0]))[None]  # their code uses channel 0
            images.append(Image.fromarray(to_image(waveform=wav, sample_rate=sr, **CONFIG)))
        results = model.predict(images, conf=conf, verbose=False, device=device)
        for j, (o, r) in enumerate(zip(part, results)):
            dur = min(WINDOW, total - o)
            for xyxyn, c in zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = (float(v) for v in xyxyn)
                # time as in their from_yolov8_prediction; frequency un-flipped (their image
                # has 0 Hz at the bottom, their conversion forgets that; scoring uses time only)
                t0, t1 = o + min(x1, x2) * dur, o + max(x1, x2) * dur
                f0 = FREQ_MIN + (1 - max(y1, y2)) * (FREQ_MAX - FREQ_MIN)
                f1 = FREQ_MIN + (1 - min(y1, y2)) * (FREQ_MAX - FREQ_MIN)
                boxes.append((t0, t1, f0, f1, float(c), b0 + j))
    return dedupe(boxes), total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etm-repo", required=True, help="clone of forest-elephants-rumble-detection")
    ap.add_argument("--weights", help="default: <repo>/data/08_artifacts/model/rumbles/yolov8/weights/best.pt")
    ap.add_argument("--list", required=True, help="files.tsv from `baselines.py list`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: ultralytics picks)")
    a = ap.parse_args(argv)

    sys.path.insert(0, os.path.join(a.etm_repo, "src"))
    from forest_elephants_rumble_detection.data.spectrogram.torchaudio import waveform_to_np_image
    from ultralytics import YOLO
    weights = a.weights or os.path.join(a.etm_repo, "data", "08_artifacts", "model", "rumbles", "yolov8",
                                        "weights", "best.pt")
    model = YOLO(weights)

    with open(a.list, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    t_start = time.time()
    for i, row in enumerate(rows):
        out = os.path.join(a.out, row["pool"], row["key"] + ".csv")
        if os.path.exists(out):
            continue
        t0 = time.time()
        boxes, total = detect_file(model, waveform_to_np_image, row["path"], a.conf, a.batch, a.device)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".part", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_start", "t_end", "f_low", "f_high", "score"])
            w.writerows([f"{b[0]:.3f}", f"{b[1]:.3f}", f"{b[2]:.1f}", f"{b[3]:.1f}", f"{b[4]:.4f}"] for b in boxes)
        os.replace(out + ".part", out)
        log(f"{i + 1}/{len(rows)} {row['pool']} {row['key']}: {len(boxes)} boxes, "
            f"{total / 3600:.1f} h of audio in {time.time() - t0:.0f} s")
    log(f"done in {(time.time() - t_start) / 60:.0f} min")


if __name__ == "__main__":
    main()
