#!/usr/bin/env python
"""
Run the DeepForestSound elephant model (AST + LoRA, low-frequency "LF" model,
github.com/MNHN-OFVI/DeepForestSound) on every sound file of a list written by
`tools/baselines.py list`. Runs in its own environment (torch 2.1.0, timm 0.4.5,
peft); see tools/baselines.py for the whole procedure.

It follows their run_inference.py for the elephant model: 10 s chunks every 10 s
(the last one shifted back to end at the end of the file), resampled to 1 kHz,
their make_features_lf, sigmoid of the LF model = the "loxodonta" score. Only the
LF model is loaded (the MF model scores birds and primates). Audio is read in long
blocks instead of one file open per chunk, which gives the same chunks, faster.

Output: <out>/<pool>/<file key>.csv with t_start, t_end, score (one row per 10 s
chunk). Finished files are skipped on a rerun.

  python tools/baselines/run_dfs.py --dfs-repo C:/code/DeepForestSound ^
      --list <out>/files.tsv --out <out>/dfs
"""
import argparse
import csv
import math
import os
import sys
import time

SR = 1000
MEL_BINS, TARGET_LENGTH = 128, 256
CHUNK = HOP = 10.0
BLOCK_CHUNKS = 60  # chunks read from disk at once (10 min)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def chunk_offsets(total):
    """Their rule: enough chunks to cover the file, the last one ends at the end of the file."""
    n = max(1, math.ceil(round((total - CHUNK) / HOP, 3)) + 1)
    return [max(0.0, total - CHUNK) if i * HOP + CHUNK > total else i * HOP for i in range(n)]


def load_model(repo, weights, device, random_weights=False):
    import torch
    sys.path.insert(0, repo)
    from src import ast_models
    from src.LoRA_inject import inject_lora
    model = ast_models.ASTModel(label_dim=1, fstride=10, tstride=10, input_fdim=MEL_BINS,
                                input_tdim=TARGET_LENGTH, imagenet_pretrain=False,
                                audioset_pretrain=False, model_size="base384", verbose=False)
    model = inject_lora(model)
    if not random_weights:
        model.load_state_dict(torch.load(weights, map_location=device))
    return model.eval().to(device)


def score_file(model, utils, path, device, batch):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.transforms as T
    info = sf.info(path)
    sr = info.samplerate
    total = info.frames / sr
    offs = chunk_offsets(total)
    resample = T.Resample(orig_freq=sr, new_freq=SR) if sr != SR else None
    n_in, n_out = int(CHUNK * sr), int(SR * CHUNK)
    scores = []
    for b0 in range(0, len(offs), BLOCK_CHUNKS):
        part = offs[b0:b0 + BLOCK_CHUNKS]
        start = int(part[0] * sr)
        y, _ = sf.read(path, start=start, frames=int(part[-1] * sr) + n_in - start, dtype="float32",
                       always_2d=True)
        y = y.mean(axis=1)
        feats = []
        for o in part:
            i = int(o * sr) - start
            x = torch.from_numpy(np.ascontiguousarray(y[i:i + n_in]))[None]
            if resample is not None:
                x = resample(x)
            x = x.squeeze(0).numpy()
            x = np.pad(x, (0, max(0, n_out - len(x))))[:n_out]
            feats.append(utils.make_features_lf(x, SR, MEL_BINS, TARGET_LENGTH))
        with torch.no_grad():
            for j in range(0, len(feats), batch):
                X = torch.stack(feats[j:j + batch]).float().to(device)
                scores.extend(torch.sigmoid(model(X))[:, 0].float().cpu().tolist())
    return [(o, min(o + CHUNK, total), s) for o, s in zip(offs, scores)], total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dfs-repo", required=True, help="clone of DeepForestSound, with the LFS weights")
    ap.add_argument("--weights", help="default: <repo>/model_weights/DFS_LF.pth")
    ap.add_argument("--list", required=True, help="files.tsv from `baselines.py list`")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--random-weights", action="store_true", help=argparse.SUPPRESS)  # code test only
    a = ap.parse_args(argv)

    import torch
    device = torch.device("cuda" if a.device != "cpu" and torch.cuda.is_available() else "cpu")
    weights = a.weights or os.path.join(a.dfs_repo, "model_weights", "DFS_LF.pth")
    if not a.random_weights and os.path.getsize(weights) < 1_000_000:
        sys.exit(f"{weights} is a Git LFS pointer, not the weights: run `git lfs pull` in {a.dfs_repo}")
    model = load_model(a.dfs_repo, weights, device, a.random_weights)
    from src import utils
    log(f"DeepForestSound LF model on {device}")

    with open(a.list, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    t_start = time.time()
    for i, row in enumerate(rows):
        out = os.path.join(a.out, row["pool"], row["key"] + ".csv")
        if os.path.exists(out):
            continue
        t0 = time.time()
        chunks, total = score_file(model, utils, row["path"], device, a.batch)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".part", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_start", "t_end", "score"])
            w.writerows([f"{c[0]:.3f}", f"{c[1]:.3f}", f"{c[2]:.5f}"] for c in chunks)
        os.replace(out + ".part", out)
        log(f"{i + 1}/{len(rows)} {row['pool']} {row['key']}: {len(chunks)} chunks, "
            f"{total / 3600:.1f} h of audio in {time.time() - t0:.0f} s")
    log(f"done in {(time.time() - t_start) / 60:.0f} min")


if __name__ == "__main__":
    main()
