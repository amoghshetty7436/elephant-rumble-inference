#!/usr/bin/env python
"""
Re-train FruitPunch's elephant rumble classifier from ELP's own data.

This is notebooks/training_notebook.ipynb as one script, minus the plotting,
the debug cells tied to CEB1_20111010, and the download of the old baseline
model. Same AVES model, same 500 Hz trick, same positive/negative windows,
same 768 -> 192 -> 2 classifier, same Adam / 2-minute training budget.

Three steps:

  prepare  embed the labelled windows of every train/test day once (slow, GPU)
  train    balance, train, and report FruitPunch-style balanced test metrics
  detect   run a model over whole days, write Raven tables with a real Score,
           and (with --truth) score them the ELP way with tools/elp_score.py

Example (Windows, one line each):

  python tools/fp_replica.py prepare --cache D:\\fp_cache
      --tables  L:\\...\\rumble_clearing_00-24hr_56days.txt
                L:\\...\\nn_ele_hb_00-24hr_TrainingSet_Removed-nn03c20180611.txt
                L:\\...\\NN_Random_Sample_Hand_ele.txt
                L:\\...\\NN_Random_Sample_Hand_nn04aSupplemental_with_20200428.txt
      --train-sounds L:\\ELP\\Detectors\\Elephant\\Datasets\\Training
      --test-sounds  L:\\ELP\\Detectors\\Elephant\\Datasets\\Testing\\Sounds\\Nouabale-Ndoki_Grid-Forest
  python tools/fp_replica.py train --cache D:\\fp_cache --out D:\\fp_cache\\retrained.pth
      --compare 2024-07-03.pth
  python tools/fp_replica.py detect --model D:\\fp_cache\\retrained.pth --out D:\\fp_detections
      --sounds L:\\...\\Testing\\Sounds\\Nouabale-Ndoki_Grid-Forest\\24-hr
      --truth L:\\...\\NN_Random_Sample_Hand_ele.txt L:\\...\\NN_Random_Sample_Hand_nn04aSupplemental_with_20200428.txt

Like the notebook, labels for a day are pooled from every table given, so pass
all four tables to `prepare`; whether a day is train or test depends only on
which sound folder its .wav is in.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Import the two model files directly: the package __init__ pulls in
# torchaudio.io.StreamReader, which newer torchaudio releases removed.
sys.path.insert(0, os.path.join(HERE, "..", "elephant_rumble_inference"))
import fp_data  # noqa: E402
import elp_score  # noqa: E402
from elephant_rumble_classifier import ElephantRumbleClassifier  # noqa: E402

AVES_SR = 500  # pretend 500 Hz audio is 16 kHz: a 5-octave pitch shift
SAMPLES_PER_EMBEDDING = 320
PREROLL, POSTROLL = 500, 200  # seconds of context the notebook embeds around each window


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Progress:
    """Text progress bar with time left, one line per finished day (works with Tee-Object)."""

    def __init__(self, total_work, total_items):
        self.total_work, self.total_items = max(total_work, 1e-9), total_items
        self.work = self.items = 0
        self.t0 = time.time()

    def step(self, work=1.0):
        self.work += work
        self.items += 1
        frac = min(1.0, self.work / self.total_work)
        bar = "#" * int(frac * 30) + "." * (30 - int(frac * 30))
        elapsed = time.time() - self.t0
        left = elapsed / frac - elapsed if frac > 0 else 0
        return (f"[{bar}] {frac * 100:5.1f}%  day {self.items}/{self.total_items}  "
                f"elapsed {fmt_time(elapsed)}, about {fmt_time(left)} left")


def fmt_time(seconds):
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{int(seconds) % 60:02d}s"


################################################################################
# audio and embeddings
################################################################################

def load_audio(path, sr_out=AVES_SR, block_seconds=3600, pad_seconds=10):
    """Mono float32 tensor at sr_out, resampled an hour at a time to keep memory low."""
    with sf.SoundFile(path) as f:
        sr, total = f.samplerate, f.frames
        ratio = Fraction(sr_out, sr)
        up, down = ratio.numerator, ratio.denominator
        block, pad = block_seconds * sr, pad_seconds * sr
        out = []
        for start in range(0, total, block):
            a, b = max(0, start - pad), min(total, start + block + pad)
            f.seek(a)
            x = f.read(b - a, dtype="float32", always_2d=True)[:, 0]
            y = resample_poly(x, up, down)
            skip = (start - a) * up // down
            keep = (min(start + block, total) - start) * up // down
            out.append(torch.from_numpy(y[skip:skip + keep].astype(np.float32)))
    y = torch.cat(out)
    # the notebook cached its 500 Hz audio as float16; keep that rounding
    return y.to(torch.float16).to(torch.float32)


class Embedder:
    def __init__(self, device):
        from aves_torchaudio_wrapper import AvesTorchaudioWrapper  # needs torchaudio
        self.device = device
        self.aves = AvesTorchaudioWrapper().to(device)
        self.aves.eval()

    @torch.inference_mode()
    def __call__(self, y):
        """Unit-length AVES embeddings, one per 320 samples (0.64 s at 500 Hz)."""
        if y.shape[0] < SAMPLES_PER_EMBEDDING * 2:
            return torch.empty(0, 768)
        e = self.aves(y.view(1, -1).to(self.device))[0].float().cpu()
        return e / e.norm(p=2, dim=1, keepdim=True)

    def whole_file(self, y, chunk_seconds=3600, context=16):
        """Embed a whole day an hour at a time, with the same ~10 s of context
        on each side that AudioFileProcessor uses at inference time."""
        chunk, ctx = chunk_seconds * AVES_SR, context * SAMPLES_PER_EMBEDDING
        out = []
        for s in range(0, y.shape[0], chunk):
            pre = y[max(0, s - ctx):s]
            cur = y[s:s + chunk]
            post = y[s + chunk:s + chunk + ctx]
            e = self(torch.cat([pre, cur, post]))
            a = pre.shape[0] // SAMPLES_PER_EMBEDDING
            b = (pre.shape[0] + cur.shape[0]) // SAMPLES_PER_EMBEDDING
            out.append(e[a:b])
            if post.shape[0] == 0 and e.shape[0] > 1:
                out.append(e[-2:-1])  # AVES drops the last frame without postroll
        return torch.cat(out) if out else torch.empty(0, 768)


def window_embeddings_fp(embed, y, label):
    """The notebook's get_aves_embeddings_from_file_with_buffers, without its cache."""
    start, duration = label.start, label.duration
    preroll = min(PREROLL, start)
    a = start - preroll
    seg = y[int(a * AVES_SR):int((a + duration + preroll + POSTROLL) * AVES_SR + 1)]
    e = embed(seg)
    i0 = fp_data.time_to_index(preroll)
    i1 = fp_data.time_to_index(preroll + duration)
    return e[i0:i1]


def window_embeddings_from_day(day, label):
    i0 = max(0, fp_data.time_to_index(label.start))
    i1 = max(i0, fp_data.time_to_index(label.start + label.duration))
    return day[i0:i1]


################################################################################
# prepare
################################################################################

def read_day_list(path, split):
    """Day names from a list like fruitpunch_files_used.txt. If the list has
    '## TRAINING' / '## TESTING' sections, only that split's section counts, so a
    day that is a training day can't also be picked up from the test folder."""
    days, section, has_sections = {"train": set(), "test": set(), None: set()}, None, False
    for line in open(path).read().splitlines():
        head = line.strip().lower()
        if head.startswith("## train"):
            section, has_sections = "train", True
        elif head.startswith("## test"):
            section, has_sections = "test", True
        elif head and not head.startswith("#"):
            days[section].add(fp_data.normalize_file_name(head))
    return days[split] if has_sections else days[None]


def cmd_prepare(a):
    labels = fp_data.read_labels(a.tables)
    log(f"{sum(map(len, labels.values()))} labels for {len(labels)} sound files in {len(a.tables)} tables")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("using", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
    embed = Embedder(device)
    manifest = {"mode": a.mode, "tables": a.tables, "splits": {}}
    jobs = []
    for split, roots in (("train", a.train_sounds), ("test", a.test_sounds)):
        sounds = fp_data.find_sound_files(roots)
        used, skipped = fp_data.usable_files(labels, sounds)
        skipped = {k: v for k, v in skipped.items() if k in sounds}
        if a.only_files:
            keep = read_day_list(a.only_files, split)
            dropped = [n for n in used if n not in keep]
            used = [n for n in used if n in keep]
            if dropped:
                log(f"{split}: leaving out {len(dropped)} days not in {a.only_files}: {dropped}")
        if a.limit_files:
            used = used[:a.limit_files]
        log(f"{split}: {len(sounds)} sound files, {len(used)} usable, {len(skipped)} skipped "
            f"(fewer than 2 labels or negative windows)")
        out_dir = os.path.join(a.cache, a.mode, split)
        os.makedirs(out_dir, exist_ok=True)
        manifest["splits"][split] = {"roots": roots, "used": used, "skipped": skipped}
        for n, name in enumerate(used, 1):
            dst = os.path.join(out_dir, name + ".pt")
            if not os.path.exists(dst):
                jobs.append((split, n, len(used), name, sounds[name], dst))

    def work(name):  # fp mode: time grows with the number of windows; day mode: one unit per day
        if a.mode == "fp":
            return 1 + len(labels[name]) + len(fp_data.negative_windows(labels[name]))
        return 1
    log(f"{len(jobs)} days to process (days already in the cache are skipped)")
    progress = Progress(sum(work(j[3]) for j in jobs), len(jobs))
    for split, n, n_split, name, path, dst in jobs:
        t0 = time.time()
        pos, neg = labels[name], fp_data.negative_windows(labels[name])
        y = load_audio(path)
        if a.mode == "fp":
            pe = [window_embeddings_fp(embed, y, l) for l in pos]
            ne = [window_embeddings_fp(embed, y, l) for l in neg]
        else:
            day = embed.whole_file(y)
            pe = [window_embeddings_from_day(day, l) for l in pos]
            ne = [window_embeddings_from_day(day, l) for l in neg]
        pe, ne = torch.cat(pe), torch.cat(ne)
        torch.save({"pos": pe.half(), "neg": ne.half()}, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        log(f"{split} {n}/{n_split} {name}: {len(pos)} labels -> {pe.shape[0]} rumble frames, "
            f"{len(neg)} negative windows -> {ne.shape[0]} frames ({time.time() - t0:.0f} s)")
        log(progress.step(work(name)))
    with open(os.path.join(a.cache, a.mode, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    log("done")


################################################################################
# train
################################################################################

def load_split(cache, mode, split, seed, balance=True, exclude=()):
    d = os.path.join(cache, mode, split)
    files = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
    overlap = [f for f in files if f in exclude]
    if overlap:
        log(f"{split}: ignoring {len(overlap)} days that are also training days: {overlap}")
        files = [f for f in files if f not in exclude]
    pos, neg = [], []
    for f in files:
        x = torch.load(os.path.join(d, f))
        pos.append(x["pos"])
        neg.append(x["neg"])
    pos, neg = torch.cat(pos), torch.cat(neg)  # still float16: ~1.8M training negatives
    log(f"{split}: {len(files)} days, {pos.shape[0]} rumble frames, {neg.shape[0]} negative frames "
        f"(ratio {neg.shape[0] / max(1, pos.shape[0]):.1f})")
    if balance:  # the notebook keeps as many negatives as positives, picked at random
        g = torch.Generator().manual_seed(seed)
        neg = neg[torch.randperm(neg.shape[0], generator=g)[:pos.shape[0]]]
    x = torch.cat([pos, neg]).float()
    y = torch.cat([torch.ones(pos.shape[0], dtype=torch.long), torch.zeros(neg.shape[0], dtype=torch.long)])
    return torch.utils.data.TensorDataset(x, y)


def balanced_class_weights(y):
    """sklearn's compute_class_weight('balanced') for labels 0/1."""
    counts = torch.bincount(y, minlength=2).float()
    return y.shape[0] / (2 * counts)


@torch.no_grad()
def predict(model, ds, batch=100_000):
    model.eval()
    logits = torch.cat([model(ds.tensors[0][i:i + batch]) for i in range(0, len(ds), batch)])
    return logits, ds.tensors[1]


def report(name, logits, y):
    pred = logits.argmax(1)
    prob = torch.softmax(logits, 1)[:, 1]
    lines = [f"\n{name}", f"{'':>8} {'precision':>9} {'recall':>7} {'f1':>6} {'support':>8}"]
    for cls, label in ((0, "not"), (1, "rumble")):
        tp = ((pred == cls) & (y == cls)).sum().item()
        p = tp / max(1, (pred == cls).sum().item())
        r = tp / max(1, (y == cls).sum().item())
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        lines.append(f"{label:>8} {p:>9.3f} {r:>7.3f} {f1:>6.3f} {(y == cls).sum().item():>8}")
    acc = (pred == y).float().mean().item()
    lines.append(f"accuracy {acc:.3f}   ROC AUC {roc_auc(prob, y):.3f}")
    print("\n".join(lines), flush=True)
    return acc


def roc_auc(score, y):
    order = torch.argsort(score)
    ranks = torch.empty_like(score)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=score.dtype)
    n1 = (y == 1).sum().item()
    n0 = len(y) - n1
    if not n0 or not n1:
        return float("nan")
    return (ranks[y == 1].sum().item() - n1 * (n1 + 1) / 2) / (n0 * n1)


def cmd_train(a):
    torch.manual_seed(a.seed)
    train_all = load_split(a.cache, a.mode, "train", a.seed)
    train_days = set(os.listdir(os.path.join(a.cache, a.mode, "train")))
    test = load_split(a.cache, a.mode, "test", a.seed, exclude=train_days)
    train_ds, val_ds = torch.utils.data.random_split(
        train_all, [0.95, 0.05], generator=torch.Generator().manual_seed(42))
    if a.val_on_test:
        val_ds = test  # what the notebook actually did (cell 23): stops on the test set
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(a.seed))
    xv, yv = (val_ds.tensors if isinstance(val_ds, torch.utils.data.TensorDataset)
              else (train_all.tensors[0][val_ds.indices], train_all.tensors[1][val_ds.indices]))

    model = ElephantRumbleClassifier()
    criterion = torch.nn.CrossEntropyLoss(weight=balanced_class_weights(train_all.tensors[1]))
    optimizer = torch.optim.Adam(model.parameters())
    t0 = time.time()
    min_val = float("inf")
    for epoch in range(a.max_epochs):
        model.train()
        total, count = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.shape[0]
            count += x.shape[0]
        model.eval()
        with torch.no_grad():
            out = model(xv)
            val = criterion(out, yv).item()
            correct = (out.argmax(1) == yv).sum().item()
        log(f"epoch {epoch + 1}: train loss {total / count * 100:.2f} val loss {val * 100:.2f}, "
            f"{correct} correct, {len(yv) - correct} incorrect")
        if val > min_val * 1.15 * 2:
            log("val loss increasing, stopping")
            break
        min_val = min(min_val, val)
        if time.time() - t0 > a.minutes * 60:
            break
    torch.save(model.state_dict(), a.out)
    log("saved", a.out)

    report("This model, balanced test frames (how FruitPunch reported results)", *predict(model, test))
    if a.compare:
        old = ElephantRumbleClassifier()
        old.load_state_dict(torch.load(a.compare, map_location="cpu"))
        report(f"{os.path.basename(a.compare)}, same test frames", *predict(old, test))


################################################################################
# detect
################################################################################

RAVEN_COLUMNS = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)",
                 "Low Freq (Hz)", "High Freq (Hz)", "Begin Path", "File Offset (s)",
                 "Begin File", "Score"]


def segments(mask, min_len=3):
    """Runs of True; FruitPunch keeps runs where end - start >= 3 frames."""
    out, start = [], None
    for i, m in enumerate(mask.tolist() + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if (i - 1) - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def cmd_detect(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed = Embedder(device)
    model = ElephantRumbleClassifier()
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model.eval()
    sounds = fp_data.find_sound_files(a.sounds)
    os.makedirs(a.out, exist_ok=True)
    combined = os.path.join(a.out, "all_detections.txt")
    schedule = os.path.join(a.out, "files.txt")
    sel, elapsed = 0, 0.0
    progress = Progress(len(sounds), len(sounds))
    with open(combined, "w", newline="") as cf:
        w = csv.writer(cf, delimiter="\t")
        w.writerow(RAVEN_COLUMNS)
        for n, name in enumerate(sorted(sounds), 1):
            t0 = time.time()
            y = load_audio(sounds[name])
            with torch.no_grad():
                logits = model(embed.whole_file(y))
            prob = torch.softmax(logits, 1)[:, 1]
            rows = []
            for s0, s1 in segments(logits[:, 1] > logits[:, 0]):
                bt = s0 * SAMPLES_PER_EMBEDDING / AVES_SR
                et = s1 * SAMPLES_PER_EMBEDDING / AVES_SR
                rows.append((bt, et, prob[s0:s1 + 1].mean().item()))
            per_file = os.path.join(a.out, os.path.splitext(name)[0] + ".rumbles.txt")
            with open(per_file, "w", newline="") as pf:
                pw = csv.writer(pf, delimiter="\t")
                pw.writerow(RAVEN_COLUMNS)
                for i, (bt, et, score) in enumerate(rows, 1):
                    pw.writerow([i, "Spectrogram 1", 1, f"{bt:.3f}", f"{et:.3f}", 5, 250,
                                 sounds[name], f"{bt:.3f}", os.path.basename(sounds[name]), f"{score:.4f}"])
            for bt, et, score in rows:
                sel += 1  # Begin Time counts from the start of the first file, as Raven does
                w.writerow([sel, "Spectrogram 1", 1, f"{elapsed + bt:.3f}", f"{elapsed + et:.3f}", 5, 250,
                            sounds[name], f"{bt:.3f}", os.path.basename(sounds[name]), f"{score:.4f}"])
            elapsed += y.shape[0] / AVES_SR
            log(f"{n}/{len(sounds)} {name}: {len(rows)} detections ({time.time() - t0:.0f} s)")
            log(progress.step())
    with open(schedule, "w") as f:
        f.write("\n".join(os.path.basename(sounds[k]) for k in sorted(sounds)) + "\n")
    log("wrote", combined, "and", schedule)
    if a.truth:
        hours = elapsed / 3600 / max(1, len(sounds))
        elp_score.main(["--truth", *a.truth, "--detections", combined, "--files", schedule,
                        "--hours-per-file", f"{hours:.4f}", "--csv", os.path.join(a.out, "elp_score.csv")])


################################################################################

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="embed labelled windows of every train/test day")
    s.add_argument("--tables", nargs="+", required=True, help="Raven tables (all of them, train and test)")
    s.add_argument("--train-sounds", nargs="+", required=True, help="folders with training .wav files")
    s.add_argument("--test-sounds", nargs="+", required=True, help="folders with test .wav files")
    s.add_argument("--cache", required=True, help="where to keep the embeddings")
    s.add_argument("--mode", choices=["fp", "day"], default="fp",
                   help="fp: embed each window with 500 s / 200 s of context, exactly like the notebook "
                        "(~460 days of audio, slower). day: embed each day once, hour by hour, like "
                        "inference does (~190 days of audio, faster)")
    s.add_argument("--only-files", help="text file of sound file names to restrict to, e.g. "
                   "fruitpunch_files_used.txt for exactly the days FruitPunch trained and tested on")
    s.add_argument("--limit-files", type=int, help="only the first N days of each split (smoke test)")

    s = sub.add_parser("train", help="train the classifier and report balanced test metrics")
    s.add_argument("--cache", required=True)
    s.add_argument("--mode", choices=["fp", "day"], default="fp")
    s.add_argument("--out", required=True, help="where to save the .pth")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--minutes", type=float, default=2, help="training time budget (notebook: 2)")
    s.add_argument("--max-epochs", type=int, default=20000)
    s.add_argument("--val-on-test", action="store_true",
                   help="early-stop on the test set, as the notebook did (leaks test data)")
    s.add_argument("--compare", help="also score this .pth (e.g. 2024-07-03.pth) on the same test frames")

    s = sub.add_parser("detect", help="find rumbles in whole days and write Raven tables")
    s.add_argument("--model", required=True)
    s.add_argument("--sounds", nargs="+", required=True, help="folders of .wav files to process")
    s.add_argument("--out", required=True)
    s.add_argument("--truth", nargs="*", help="truth tables: also score the ELP way")

    a = p.parse_args(argv)
    {"prepare": cmd_prepare, "train": cmd_train, "detect": cmd_detect}[a.cmd](a)


if __name__ == "__main__":
    main()
