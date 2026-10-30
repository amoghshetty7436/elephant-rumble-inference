#!/usr/bin/env python
"""
The candidate rumble detectors (docs: candidate-models.html), each trained and scored
exactly like the final model of the full rerun (study.py), so the numbers compare:

  m1  long-memory reader   a BiGRU reads ~82 s of AVES fingerprints and scores every 0.64 s
  m2  day-normalised       each recording's median fingerprint is subtracted first
  m4  detect, then check   the final model finds candidates; a ResNet-18 checks a 32 s
                           low-pitch spectrogram around each one and re-scores it
  m5  two-stream           AVES + a small CNN on a 5-250 Hz PCEN spectrogram, side by side
  m6  other encoders       BirdAVES (birds, same design as AVES) and Dasheng (general sound)
                           instead of AVES, same classifier
  m7  pitch contour        AVES + 8 numbers per step from a pitch tracker (8-30 Hz)

Everything else is the final model's: setup E2 (clearing + forest + noise clips), its
training recipe (study.build_sets: noise x10, 3000 random forest frames per day, 0.5 gap
frames per rumble frame), 10.9 s windows, 5 runs, each trained until the average
precision on the held-out training days stops improving, then the alarm rule tuned over
all 1,960 rules on the validation days. Score = mean false alarms per hour at 80% and 85%
recall on the validation days (lower is better; the final model: see candidates.md).
The tests (Sample B, Sample A, vehicle days) are scored once, by `final`.

Nothing that study.py or dataset_configs.py wrote is changed. New files go to
<out>\\_candidates, features to <out>\\_features, other encoders' fingerprints to
<out>\\_embedding_cache_<encoder>.

On the lab PC, after study.py has finished (one line; rerunning it resumes):

  python tools\\candidates.py                 # everything, in the order below
  python tools\\candidates.py --only m4 m1    # some of them
  python tools\\candidates.py final           # the tests, for whatever has finished

Order of `all`: features (audio pass for m5 and m7), m4, m1, m2, m7, m5, then for each
encoder: fingerprints, m6. Progress: <out>\\_candidates\\progress.txt; results so far:
candidates.md.

Extra packages: torchvision (m4), dasheng (m6 with Dasheng: pip install dasheng).
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset_configs as dc  # noqa: E402
import experiments as ex  # noqa: E402
import improve as im  # noqa: E402
import study as st  # noqa: E402
from dataset_configs import log, fmt_dur, Day, FRAME_SECONDS  # noqa: E402

ROOT_NAME = "_candidates"
FEATURES_NAME = "_features"
SETUP = "E2"
ALL_MODELS = ["m4", "m1", "m2", "m7", "m5", "m6_birdaves", "m6_dasheng"]
TITLES = {"m1": "1 long-memory reader (BiGRU, 82 s)", "m2": "2 day-normalised fingerprints",
          "m4": "4 detect, then double-check (ResNet-18)", "m5": "5 two-stream: AVES + low-pitch CNN",
          "m6_birdaves": "6 encoder: BirdAVES", "m6_dasheng": "6 encoder: Dasheng",
          "m7": "7 AVES + pitch contour"}

# ---- model 1: sequences
SEQ_LEN = 128                  # frames a training stretch covers (82 s)
SEQ_RUMBLE_PER_DAY = 60        # stretches around labelled rumbles, per training day
SEQ_RANDOM_PER_DAY = 40        # stretches at random places, per training day
SEQ_BATCH = 32
SEQ_CHECK_EVERY = 4000         # stretches between checks on the held-out days
SEQ_PATIENCE = 10
SEQ_MAX_CHECKS = 80
SEQ_HIDDEN = 64                # per direction

# ---- model 2: day normalisation
NORM_MIN_FRAMES = 2800         # recordings shorter than this (30 min) use the training days' median
NORM_SAMPLE_EVERY = 5          # the median is taken over every 5th frame

# ---- models 5 and 7: features from the 500 Hz audio, on the 0.64 s grid
LF_NFFT, LF_HOP = 256, 80      # 1.95 Hz bins, 4 columns per 0.64 s step
LF_BINS = 64                   # 1.95-250 Hz pooled to 64 bins of 3.9 Hz
LF_SUB = 4
PCEN_T, PCEN_ALPHA, PCEN_DELTA, PCEN_R, PCEN_EPS = 10.0, 0.98, 2.0, 0.5, 1e-6
LF_SCALE = 64.0                # stored as uint8: value * LF_SCALE, clipped to 0..255 (PCEN 0..4)
PITCH_NFFT = 1024              # 0.49 Hz bins
PITCH_F0 = (8.0, 30.0, 0.5)    # f0 candidates: from, to, step (Hz)
PITCH_HARMONICS = 6
PITCH_FEATURES = ["salience", "salience_max", "f0", "f0_change", "f0_spread_15s", "salience_15s",
                  "f0_spread_60s", "band_level"]

# ---- model 4: the double-checker
VER_POS_PER_DAY = 100          # labelled rumbles per training day used as "rumble" cuts
VER_NEG_PER_DAY = 200          # first-stage false alarms per training day (highest scores first)
VER_SKIP_ABOVE = 0.99          # a "false alarm" this confident on a clearing day may be an unmarked rumble
VER_CUT_S = 40.0               # seconds of audio stored per cut (32 s used, +/- 4 s for shifting)
VER_USE_S = 32.0
VER_NFFT, VER_HOP = 256, 50    # 0.1 s columns, bins 1..128 = 1.95-250 Hz
VER_DB_RANGE = 40.0
VER_BATCH = 64
VER_MAX_EPOCHS = 30
VER_PATIENCE = 4
VER_COMBOS = ["verifier", "geomean"]

# ---- model 6: encoders
ENCODERS = {
    "birdaves": dict(kind="aves", sr=500, url="https://storage.googleapis.com/esp-public-files/birdaves/",
                     files=("birdaves-biox-base.torchaudio.model_config.json", "birdaves-biox-base.torchaudio.pt")),
    "dasheng": dict(kind="dasheng", sr=1000),   # 1 kHz audio played as 16 kHz: 16x, 40 ms = 0.64 s
}
DASHENG_BLOCK = 161280         # samples Dasheng attends over at once (1008 frames of 160)


################################################################################
# features: where the numbers for each 0.64 s step come from
################################################################################

class FeatDay:
    """The interface study's Day has: n, kind, frames(idx), whole()."""
    kind = "day"

    def __init__(self, x):
        self.x = x                 # [n, dim] float16 tensor
        self.n = x.shape[0]

    def frames(self, idx):
        import torch
        return self.x[torch.as_tensor(list(idx), dtype=torch.long)] if len(idx) else self.x[:0]

    def whole(self):
        return self.x


class Source:
    """AVES fingerprints from a cache folder (study's, or another encoder's)."""

    def __init__(self, cache, dim=768):
        self.cache, self.dim = cache, dim

    def day(self, pool, key):
        return Day(self.cache, pool, key)


class NormSource(Source):
    """Model 2: each recording minus its median fingerprint (short clips: the training days' median)."""

    def __init__(self, base, train_days):
        self.base, self.dim = base, base.dim
        self.ref = None
        self.train_days = train_days

    def median_of(self, x):
        import torch
        return torch.median(x[::NORM_SAMPLE_EVERY].float(), dim=0).values

    def reference(self):
        import torch
        if self.ref is None:
            meds = []
            for p, k in self.train_days[:40]:
                d = self.base.day(p, k)
                if d.kind == "day" and d.n >= NORM_MIN_FRAMES:
                    meds.append(self.median_of(d.whole()))
            self.ref = torch.stack(meds).median(0).values if meds else torch.zeros(self.dim)
        return self.ref

    def day(self, pool, key):
        d = self.base.day(pool, key)
        x = d.whole().float()
        med = self.median_of(x) if d.n >= NORM_MIN_FRAMES else self.reference()
        return FeatDay((x - med).half())


class ConcatSource(Source):
    """AVES plus per-step extra features from <features>/<name>/<pool>/<key>.npy, scaled."""

    def __init__(self, base, folder, name, dim_extra, scale=None):
        self.base, self.folder, self.name = base, folder, name
        self.dim = base.dim + dim_extra
        self.scale = scale  # (mean, std) arrays, or a single divisor, or None

    def extra(self, pool, key, n):
        import numpy as np
        import torch
        e = np.load(feature_path(self.folder, self.name, pool, key)).astype(np.float32)
        if len(e) < n:
            e = np.concatenate([e, np.repeat(e[-1:], n - len(e), 0)]) if len(e) else np.zeros((n, self.dim - self.base.dim), np.float32)
        e = e[:n]
        if isinstance(self.scale, tuple):
            e = (e - self.scale[0]) / self.scale[1]
        elif self.scale:
            e = e / self.scale
        return torch.from_numpy(e)

    def day(self, pool, key):
        import torch
        d = self.base.day(pool, key)
        x = d.whole()
        return FeatDay(torch.cat([x.float(), self.extra(pool, key, d.n)], 1).half())


def feature_path(folder, name, pool, key):
    return os.path.join(folder, name, pool, f"{key}.npy")


def pitch_scale(folder, days):
    """Mean and standard deviation of the pitch features over some training days."""
    import numpy as np
    xs = [np.load(feature_path(folder, "pitch", p, k)).astype(np.float32) for p, k in days[:40]
          if os.path.exists(feature_path(folder, "pitch", p, k))]
    if not xs:
        return None
    x = np.concatenate(xs)
    return x.mean(0), np.maximum(x.std(0), 1e-3)


################################################################################
# models
################################################################################

def make_ctx_model(in_dim, ctx):
    """The final model's classifier (experiments.make_model) for any number of inputs per step."""
    import torch.nn as nn

    class ContextClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.ctx = ctx
            self.drop = nn.Dropout(0.2)
            self.act = nn.LeakyReLU()
            self.proj = nn.Linear(in_dim, ex.PROJ_DIM)
            self.linear1 = nn.Linear((2 * ctx + 1) * ex.PROJ_DIM, 192)
            self.linear2 = nn.Linear(192, 2)

        def frame_features(self, x):
            return self.act(self.proj(self.drop(x)))

        def head(self, flat):
            return self.linear2(self.drop(self.act(self.linear1(self.drop(flat)))))

        def forward(self, x):
            h = self.frame_features(x)
            return self.head(h.reshape(h.shape[0], -1))

    return ContextClassifier()


def make_two_stream(ctx, aves_dim=768):
    """Model 5: the final model's AVES path, plus a small CNN over the window's
    low-pitch spectrogram (64 bins x 4 columns per step), joined before the last layers."""
    import torch
    import torch.nn as nn

    class TwoStream(nn.Module):
        def __init__(self):
            super().__init__()
            self.ctx, self.w = ctx, 2 * ctx + 1
            self.drop = nn.Dropout(0.2)
            self.act = nn.LeakyReLU()
            self.proj = nn.Linear(aves_dim, ex.PROJ_DIM)

            def block(i, o):
                return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
            self.cnn = nn.Sequential(*block(1, 16), nn.MaxPool2d(2), *block(16, 32), nn.MaxPool2d(2),
                                     *block(32, 64), nn.AdaptiveMaxPool2d((1, 4)))
            self.linear1 = nn.Linear(self.w * ex.PROJ_DIM + 64 * 4, 192)
            self.linear2 = nn.Linear(192, 2)

        def forward(self, x):  # [B, W, 768 + 256]
            b = x.shape[0]
            h = self.act(self.proj(self.drop(x[..., :aves_dim]))).reshape(b, -1)
            s = x[..., aves_dim:].reshape(b, self.w, LF_SUB, LF_BINS).permute(0, 3, 1, 2)
            s = s.reshape(b, 1, LF_BINS, self.w * LF_SUB)
            c = self.cnn(s).reshape(b, -1)
            return self.linear2(self.drop(self.act(self.linear1(self.drop(torch.cat([h, c], 1))))))

    return TwoStream()


def make_seq_model(in_dim):
    """Model 1: per-step projection, then a two-layer bidirectional GRU, one score per step."""
    import torch.nn as nn

    class SeqModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(0.2)
            self.proj = nn.Linear(in_dim, 128)
            self.act = nn.LeakyReLU()
            self.gru = nn.GRU(128, SEQ_HIDDEN, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
            self.out = nn.Linear(2 * SEQ_HIDDEN, 1)

        def forward(self, x):  # [B, L, D] -> [B, L]
            h = self.act(self.proj(self.drop(x)))
            h, _ = self.gru(h)
            return self.out(self.drop(h))[..., 0]

    return SeqModel()


def window_probs(model, x, ctx, device, chunk=4096):
    """Rumble probability per step for a windowed model without frame_features (model 5)."""
    import torch
    with torch.no_grad():
        x = x.to(device)
        n = x.shape[0]
        offs = torch.arange(-ctx, ctx + 1, device=device)
        out = []
        for i in range(0, n, chunk):
            c = torch.arange(i, min(n, i + chunk), device=device)
            w = (c[:, None] + offs[None]).clamp(0, n - 1)
            out.append(torch.softmax(model(x[w].float()), 1)[:, 1])
        return torch.cat(out).float().cpu().numpy()


def seq_probs(model, x, device, length=SEQ_LEN):
    """Rumble probability per step for the sequence model: windows of 2*length steps every
    `length` steps, keeping the middle half of each (the whole window at the day's ends)."""
    import numpy as np
    import torch
    n = x.shape[0]
    out = np.zeros(n, np.float32)
    win, hop, m = 2 * length, length, length // 2
    with torch.no_grad():
        starts = list(range(0, max(1, n - win + hop), hop))
        for b in range(0, len(starts), 16):
            group = starts[b:b + 16]
            xs = [x[s:s + win] for s in group]
            L = max(len(t) for t in xs)
            batch = torch.zeros(len(xs), L, x.shape[1], dtype=torch.float32)
            for j, t in enumerate(xs):
                batch[j, :len(t)] = t.float()
            p = torch.sigmoid(model(batch.to(device))).cpu().numpy()
            for j, s in enumerate(group):
                e = min(n, s + win)
                a = s if s == 0 else s + m
                z = e if e == n else s + win - m
                out[a:z] = p[j, a - s:z - s]
    return out


################################################################################
# training data and training (study.build_sets / train_until_converged, any features)
################################################################################

def build_sets(cl, s, src):
    """study.build_sets for any feature source: the same samples of the same days
    (no augmentation, no mined frames: the final model has neither)."""
    import numpy as np
    import torch
    lab = cl.lab
    cfg, pools = lab.cfg, lab.pools
    ctx = s["ctx"]
    rng = np.random.default_rng(lab.seeds[0])
    limit = int(st.MAX_RAM_SHARE * st.total_ram_bytes())
    held = lab.heldout
    pos_n, gap_n = st.label_counts(cfg, pools, held)
    n_pos = int(round(sum(pos_n.values())))
    noise_days = {key: src.day("noise", key) for key in cfg["train"].get("noise", [])} if s["noise"] else {}
    n_noise = sum(d.n for d in noise_days.values()) * s["noise"]
    forest_days = sum(1 for p, f in cfg["train"].items() if p != "noise" and dc.POOLS[p]["forest"]
                      for k in f if (p, k) not in held)
    n_rand = s["rand"] * forest_days
    gap_target = max(0, n_pos - n_noise - n_rand) if s["gap"] == "fill" else int(round(s["gap"] * n_pos))
    f_gap = {h: min(1.0, gap_target / max(1, sum(gap_n.values()))) for h in gap_n}
    banks, offset = [], 0
    parts = {"train": {"wins": [], "ys": []}, "held": {"wins": [], "ys": []}}
    counts = dict(rumble=0, gap=0, noise=0, random=0, held_rumble=0, held_not=0)
    offsets = np.arange(-ctx, ctx + 1)

    def add(day, groups):
        nonlocal offset
        picks = []
        for c, lab_, kind, f, part in groups:
            c = np.asarray(sorted(set(int(i) for i in c)), dtype=np.int64)
            c = c[(c >= 0) & (c < day.n)]
            picks.append((im.take(c, f, rng), lab_, kind, part))
        allc = [x for x, *_ in picks if len(x)]
        if not allc:
            return
        need = np.unique(np.clip(np.concatenate(allc)[:, None] + offsets[None, :], 0, day.n - 1))
        if limit and (offset + len(need)) * src.dim * 2 > limit:
            raise MemoryError(f"needs more than {limit / 2**30:.0f} GB of features in memory")
        rows = day.frames(need.tolist())
        banks.append(rows)
        for c, lab_, kind, part in picks:
            if len(c):
                w = np.searchsorted(need, np.clip(c[:, None] + offsets[None, :], 0, day.n - 1)) + offset
                parts[part]["wins"].append(torch.from_numpy(w))
                parts[part]["ys"].append(torch.full((len(c),), lab_, dtype=torch.long))
                counts[kind] += len(c)
        offset += len(need)

    for p, files in cfg["train"].items():
        for key in files:
            if p == "noise":
                if s["noise"]:
                    day = noise_days[key]
                    add(day, [(range(day.n), 0, "noise", s["noise"], "train")])
                continue
            h = "clearing" if p == "clearing" else "forest"
            day = src.day(p, key)
            labels = pools[p].labels.get(key, [])
            if (p, key) in held:
                pi, ni = dc.label_frames(labels)
                rnd = dc.random_negative_frames(labels, day.n, key + "-heldout", n=st.HELDOUT_RANDOM_PER_DAY)
                add(day, [(pi, 1, "held_rumble", 1.0, "held"), (sorted(set(ni) | set(rnd)), 0, "held_not", 1.0, "held")])
                continue
            groups = []
            if dc.is_usable(labels):
                pi, ni = dc.label_frames(labels)
                groups += [(pi, 1, "rumble", 1.0, "train"), (ni, 0, "gap", f_gap.get(h, 0.0), "train")]
            if s["rand"] and dc.POOLS[p]["forest"]:
                groups.append((dc.random_negative_frames(labels, day.n, key, n=s["rand"]), 0, "random", 1.0, "train"))
            add(day, groups)
    bank = torch.cat(banks)
    tr_w, tr_y = torch.cat(parts["train"]["wins"]), torch.cat(parts["train"]["ys"])
    if parts["held"]["wins"]:
        ho_w, ho_y = torch.cat(parts["held"]["wins"]), torch.cat(parts["held"]["ys"])
    else:
        ho_w, ho_y = tr_w[:0], tr_y[:0]
    info = {"examples": int(tr_y.shape[0]), "features_in_memory": int(bank.shape[0]), "dim": src.dim,
            **counts, "heldout_days": len(held)}
    log(f"training set: {info}")
    return bank, tr_w, tr_y, ho_w, ho_y, info


def train_windows(bank, win, y, ho_win, ho_y, make, seed, logf, device):
    """study.train_until_converged with any windowed model (same optimiser, batch, class
    weights, checks, patience and choice of the best check)."""
    import torch
    torch.manual_seed(seed)
    n = y.shape[0]
    bank_d = bank
    if device == "cuda":
        try:
            if bank.numel() * bank.element_size() < 0.5 * torch.cuda.mem_get_info()[0]:
                bank_d = bank.to(device)
        except RuntimeError:
            bank_d = bank
    win_d, ho_d = win.to(bank_d.device), ho_win.to(bank_d.device)
    y_d, ho_yd = y.to(device), ho_y.to(device)

    def batch_x(w, idx):
        return bank_d[w[idx.to(w.device)]].to(device).float()

    weights = (n / (2 * torch.bincount(y, minlength=2).float().clamp_min(1))).to(device)
    model = make().to(device)
    crit = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters())
    g = torch.Generator().manual_seed(seed)
    ho_n = ho_win.shape[0]

    def check():
        model.eval()
        with torch.no_grad():
            sc = [torch.softmax(model(batch_x(ho_d, torch.arange(i, min(ho_n, i + 8192)))), 1)[:, 1]
                  for i in range(0, ho_n, 8192)]
        model.train()
        return st.average_precision(torch.cat(sc), ho_yd) if sc else 0.0

    seen, checks, best, best_check, best_state, since = 0, 0, -1.0, 0, None, 0
    t0, done = time.time(), False
    model.train()
    while not done:
        order = torch.randperm(n, generator=g)
        for i in range(0, n, st.BATCH):
            b = order[i:i + st.BATCH]
            opt.zero_grad()
            loss = crit(model(batch_x(win_d, b)), y_d[b.to(device)])
            loss.backward()
            opt.step()
            seen += b.shape[0]
            if seen >= (checks + 1) * st.CHECK_EVERY:
                checks += 1
                ap = check()
                better = ap > best + 1e-5 or not ho_n
                logf.write(f"seed {seed} check {checks}: loss {loss.item():.4f}, held-out AP {ap:.4f}{' *' if better else ''}\n")
                if better:
                    best, best_check, since = ap, checks, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    since += 1
                if since >= st.PATIENCE or checks >= st.MAX_CHECKS:
                    done = True
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"checks": checks, "best_check": best_check, "epochs": round(seen / n, 2), "heldout_ap": round(best, 5),
            "seconds": round(time.time() - t0, 1)}
    logf.write(json.dumps(info) + "\n")
    return model.eval().cpu(), info


def build_sequences(cl, src):
    """Model 1: stretches of SEQ_LEN steps from the training days (around rumbles and at
    random places), the noise clips (x noise repeats), and the same from the held-out days.
    Labels per step: 1 rumble, 0 not, -1 ignored (steps next to a rumble's edge, padding)."""
    import numpy as np
    import torch
    lab = cl.lab
    cfg, pools, held, s = lab.cfg, lab.pools, lab.heldout, cl.base_spec
    rng = np.random.default_rng(lab.seeds[0])
    L = SEQ_LEN
    banks, offset = [], 0
    out = {"train": ([], []), "held": ([], [])}
    counts = {"train": 0, "held": 0, "noise": 0}

    def add(day, starts, labels_full, part, repeat=1):
        nonlocal offset
        starts = sorted(set(int(a) for a in starts))
        if not starts:
            return
        need = np.unique(np.concatenate([np.arange(a, min(day.n, a + L)) for a in starts]))
        banks.append(day.frames(need.tolist()))
        for a in starts:
            idx = np.arange(a, min(day.n, a + L))
            rows = np.full(L, -1, np.int64)
            ys = np.full(L, -1, np.int8)
            rows[:len(idx)] = np.searchsorted(need, idx) + offset
            ys[:len(idx)] = labels_full[idx]
            rows[len(idx):] = rows[len(idx) - 1]
            for _ in range(repeat):
                out[part][0].append(rows)
                out[part][1].append(ys)
                counts[part] += 1
        offset += len(need)

    for p, files in cfg["train"].items():
        for key in files:
            day = src.day(p, key)
            if p == "noise":
                if s["noise"] and day.n >= 2:
                    add(day, range(0, day.n, L), np.zeros(day.n, np.int8), "train", repeat=s["noise"])
                    counts["noise"] += 1
                continue
            labels = pools[p].labels.get(key, [])
            if not dc.is_usable(labels) and not dc.POOLS[p]["forest"]:
                continue
            y = np.zeros(day.n, np.int8)
            pi, _ = dc.label_frames(labels)
            pi = np.asarray([i for i in pi if i < day.n], dtype=np.int64)
            if len(pi):
                y[pi] = 1
                edge = np.unique(np.clip(np.concatenate([pi - 1, pi + 1]), 0, day.n - 1))
                y[edge[y[edge] == 0]] = -1
            r = np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest(), 16) % 2**32)
            starts = []
            if len(labels):
                centers = [dc.fp_data.time_to_index(l.start + l.duration / 2) for l in labels]
                pick = r.choice(len(centers), min(len(centers), SEQ_RUMBLE_PER_DAY), replace=False)
                starts += [min(max(0, centers[i] - int(r.integers(8, L - 8))), max(0, day.n - L)) for i in pick]
            if dc.POOLS[p]["forest"] or dc.is_usable(labels):
                starts += list(r.integers(0, max(1, day.n - L), SEQ_RANDOM_PER_DAY))
            add(day, starts, y, "held" if (p, key) in held else "train")
    bank = torch.cat(banks)
    tr = (torch.from_numpy(np.stack(out["train"][0])), torch.from_numpy(np.stack(out["train"][1])))
    ho = (torch.from_numpy(np.stack(out["held"][0])), torch.from_numpy(np.stack(out["held"][1]))) if out["held"][0] \
        else (tr[0][:0], tr[1][:0])
    info = {"stretches": counts["train"], "heldout_stretches": counts["held"], "noise_clips": counts["noise"],
            "steps_in_memory": int(bank.shape[0]), "rumble_steps": int((tr[1] == 1).sum()),
            "other_steps": int((tr[1] == 0).sum())}
    log(f"sequence training set: {info}")
    return bank, tr, ho, info


def train_sequences(bank, tr, ho, in_dim, seed, logf, device):
    """Model 1: Adam, batches of SEQ_BATCH stretches, loss weighted so rumble and other steps
    count equally, checked on the held-out stretches (average precision over their steps)."""
    import torch
    torch.manual_seed(seed)
    rows, ys = tr
    ho_rows, ho_ys = ho
    bank_d = bank
    if device == "cuda":
        try:
            if bank.numel() * bank.element_size() < 0.5 * torch.cuda.mem_get_info()[0]:
                bank_d = bank.to(device)
        except RuntimeError:
            bank_d = bank
    n_pos, n_neg = int((ys == 1).sum()), int((ys == 0).sum())
    pos_w = torch.tensor(n_neg / max(1, n_pos), device=device)
    model = make_seq_model(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)

    def batch(rw, idx):
        return bank_d[rw[idx].to(bank_d.device)].to(device).float()

    def check():
        model.eval()
        sc, yy = [], []
        with torch.no_grad():
            for i in range(0, ho_rows.shape[0], 64):
                idx = torch.arange(i, min(ho_rows.shape[0], i + 64))
                p = torch.sigmoid(model(batch(ho_rows, idx))).cpu()
                y = ho_ys[idx]
                keep = y >= 0
                sc.append(p[keep])
                yy.append(y[keep].long())
        model.train()
        return st.average_precision(torch.cat(sc), torch.cat(yy)) if sc else 0.0

    seen, checks, best, best_check, best_state, since = 0, 0, -1.0, 0, None, 0
    t0, done = time.time(), False
    n = rows.shape[0]
    model.train()
    while not done:
        order = torch.randperm(n, generator=g)
        for i in range(0, n, SEQ_BATCH):
            b = order[i:i + SEQ_BATCH]
            y = ys[b].to(device)
            mask = (y >= 0).float()
            logit = model(batch(rows, b))
            w = mask * torch.where(y == 1, pos_w, torch.ones_like(pos_w))
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, y.clamp_min(0).float(),
                                                                         reduction="none") * w).sum() / w.sum().clamp_min(1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            seen += b.shape[0]
            if seen >= (checks + 1) * SEQ_CHECK_EVERY:
                checks += 1
                ap = check()
                better = ap > best + 1e-5 or not ho_rows.shape[0]
                logf.write(f"seed {seed} check {checks}: loss {loss.item():.4f}, held-out AP {ap:.4f}{' *' if better else ''}\n")
                if better:
                    best, best_check, since = ap, checks, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    since += 1
                if since >= SEQ_PATIENCE or checks >= SEQ_MAX_CHECKS:
                    done = True
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"checks": checks, "best_check": best_check, "epochs": round(seen / n, 2), "heldout_ap": round(best, 5),
            "seconds": round(time.time() - t0, 1)}
    logf.write(json.dumps(info) + "\n")
    return model.eval().cpu(), info


################################################################################
# scoring events (model 4) the way improve.sweep scores alarms
################################################################################

def sweep_events(truth, events):
    """events: {(pool, key): (first frames, last frames, scores)} -> the same dict as im.sweep."""
    import numpy as np
    sc, hits, best = [], [], []
    for day, (s, e, v) in events.items():
        h, b = truth.match(day, np.asarray(s), np.asarray(e), np.asarray(v, dtype=np.float64))
        sc.append(np.asarray(v, dtype=np.float64))
        hits.append(h)
        best.append(b)
    sc, hits, best = np.concatenate(sc), np.concatenate(hits), np.concatenate(best)
    thr = np.array(im.THRESHOLDS)
    all_s, hit_s, best_s = np.sort(sc), np.sort(sc[hits]), np.sort(best)
    n_det = len(sc) - np.searchsorted(all_s, thr - 1e-12)
    tp = len(hit_s) - np.searchsorted(hit_s, thr - 1e-12)
    found = len(best_s) - np.searchsorted(best_s, thr - 1e-12)
    fp = n_det - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(n_det > 0, tp / np.maximum(n_det, 1), np.nan)
        recall = found / truth.n if truth.n else np.full(len(thr), np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return dict(threshold=thr, detections=n_det, true_detections=tp, fp=fp, rumbles=truth.n, rumbles_found=found,
                precision=precision, recall=recall, f1=np.nan_to_num(f1),
                fp_per_hour=fp / truth.hours if truth.hours else np.full(len(thr), np.nan))


def at_cut(sw, cut):
    i = int(abs(sw["threshold"] - cut).argmin())
    return {m: float(sw[m][i]) for m in ("recall", "precision", "fp_per_hour")}


################################################################################
# audio features (models 5 and 7) and other encoders (model 6)
################################################################################

def pcen(mag, hop_s):
    """Per-channel energy normalisation along time (Wang et al. 2017), librosa's formula."""
    import numpy as np
    from scipy.signal import lfilter
    b = 1 - math.exp(-hop_s / PCEN_T)
    zi = (1 - b) * mag[:, :1]
    m, _ = lfilter([b], [1, b - 1], mag, axis=1, zi=zi)
    return (mag / (PCEN_EPS + m) ** PCEN_ALPHA + PCEN_DELTA) ** PCEN_R - PCEN_DELTA ** PCEN_R


def stft_mag(y, nfft, hop):
    import torch
    return torch.stft(torch.as_tensor(y, dtype=torch.float32), nfft, hop, window=torch.hann_window(nfft),
                      center=True, return_complex=True).abs().numpy()


def lf_features(y, n):
    """[n, LF_SUB * LF_BINS] uint8: PCEN of the 1.95-250 Hz spectrogram, 4 columns per step."""
    import numpy as np
    mag = stft_mag(y * 32768.0, LF_NFFT, LF_HOP)[1:1 + 2 * LF_BINS]          # [128, cols]
    mag = mag.reshape(LF_BINS, 2, -1).mean(1)
    p = pcen(mag, LF_HOP / 500)
    cols = n * LF_SUB
    if p.shape[1] < cols:
        p = np.concatenate([p, np.repeat(p[:, -1:], cols - p.shape[1], 1)], 1)
    p = p[:, :cols].reshape(LF_BINS, n, LF_SUB).transpose(1, 2, 0)           # [n, sub, bins]
    return np.clip(np.round(p * LF_SCALE), 0, 255).astype(np.uint8).reshape(n, -1)


def pitch_matrix(nfft, sr=500):
    """Harmonic-sum weights: salience(f0) = W @ magnitude spectrum."""
    import numpy as np
    lo, hi, step = PITCH_F0
    f0s = np.arange(lo, hi + 1e-9, step)
    nb = nfft // 2 + 1
    W = np.zeros((len(f0s), nb), np.float32)
    for i, f0 in enumerate(f0s):
        for h in range(1, PITCH_HARMONICS + 1):
            b = h * f0 * nfft / sr
            j = int(b)
            if j + 1 >= nb:
                break
            w = 0.85 ** (h - 1)
            W[i, j] += w * (1 - (b - j))
            W[i, j + 1] += w * (b - j)
    return f0s, W


def rolling(x, half, fn):
    """Slow reference for rolling_fast (used by the tests)."""
    import numpy as np
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = fn(x[max(0, i - half):i + half + 1])
    return out


def pitch_features(y, n):
    """[n, 8] float16: harmonic salience and pitch (8-30 Hz) per step, and how steady they are."""
    import numpy as np
    f0s, W = pitch_matrix(PITCH_NFFT)
    sal_cols, f0_cols, lvl_cols = [], [], []
    hour = 500 * 3600
    for a in range(0, len(y), hour):
        seg = y[max(0, a - PITCH_NFFT):a + hour + PITCH_NFFT]
        mag = np.sqrt(stft_mag(seg, PITCH_NFFT, LF_HOP))
        skip = (a - max(0, a - PITCH_NFFT)) // LF_HOP
        keep = (min(len(y), a + hour) - a + LF_HOP - 1) // LF_HOP
        mag = mag[:, skip:skip + keep]
        band = mag[int(5 * PITCH_NFFT / 500):int(250 * PITCH_NFFT / 500)]
        s = W @ mag
        norm = band.mean(0) * W.sum(1).mean() + 1e-6
        sal_cols.append(s.max(0) / norm)
        f0_cols.append(f0s[s.argmax(0)])
        lvl_cols.append(np.log(band.mean(0) + 1e-6))
    sal, f0, lvl = (np.concatenate(c) for c in (sal_cols, f0_cols, lvl_cols))
    cols = n * LF_SUB
    pad = lambda v: np.concatenate([v, np.repeat(v[-1:], max(0, cols - len(v)))])[:cols].reshape(n, LF_SUB)  # noqa: E731
    sal, f0, lvl = pad(sal), pad(f0), pad(lvl)
    f0m = f0.mean(1)
    feats = np.stack([
        sal.mean(1), sal.max(1), (f0m - PITCH_F0[0]) / (PITCH_F0[1] - PITCH_F0[0]), f0[:, -1] - f0[:, 0],
        rolling_fast(f0m, 12, "std"), rolling_fast(sal.mean(1), 12, "mean"), rolling_fast(f0m, 47, "std"),
        lvl.mean(1) - np.median(lvl)], 1)
    return feats.astype(np.float16)


def rolling_fast(x, half, how):
    """Rolling mean or standard deviation with cumulative sums (same as rolling())."""
    import numpy as np
    n = len(x)
    c1 = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    c2 = np.concatenate([[0.0], np.cumsum(np.asarray(x, np.float64) ** 2)])
    i = np.arange(n)
    a, b = np.maximum(0, i - half), np.minimum(n, i + half + 1)
    cnt = b - a
    mean = (c1[b] - c1[a]) / cnt
    if how == "mean":
        return mean.astype(x.dtype)
    var = np.maximum(0.0, (c2[b] - c2[a]) / cnt - mean ** 2)
    return np.sqrt(var).astype(x.dtype)


def load_aves_like(spec, device):
    """BirdAVES (same torchaudio wav2vec2 design as AVES) as an fp_replica.Embedder."""
    import torch
    from torchaudio.models import wav2vec2_model
    import fp_replica
    folder = os.path.join(torch.hub.get_dir(), "birdaves")
    os.makedirs(folder, exist_ok=True)
    for f in spec["files"]:
        dst = os.path.join(folder, f)
        if not os.path.exists(dst):
            log(f"downloading {spec['url'] + f}")
            torch.hub.download_url_to_file(spec["url"] + f, dst)
    cfg = json.load(open(os.path.join(folder, spec["files"][0])))
    model = wav2vec2_model(**cfg, aux_num_out=None)
    model.load_state_dict(torch.load(os.path.join(folder, spec["files"][1]), map_location="cpu"))

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, sig):
            return self.model.extract_features(sig)[0][-1]

    e = fp_replica.Embedder.__new__(fp_replica.Embedder)
    e.device, e.aves = device, Wrap().to(device).eval()
    return e


class DashengEmbedder:
    """Dasheng-base on 1 kHz audio played as 16 kHz: one 768-number fingerprint per 640
    samples = 0.64 s, unit length like AVES's."""

    def __init__(self, device):
        from dasheng import dasheng_base
        self.device = device
        self.model = dasheng_base().to(device).eval()

    def whole_file(self, y):
        import torch
        n = y.shape[0] // 640
        if n == 0:
            return torch.empty(0, 768)
        out = []
        with torch.inference_mode():
            blocks = [y[i:i + DASHENG_BLOCK] for i in range(0, y.shape[0], DASHENG_BLOCK)]
            for b in range(0, len(blocks), 8):
                part = blocks[b:b + 8]
                x = torch.zeros(len(part), DASHENG_BLOCK)
                for j, t in enumerate(part):
                    x[j, :len(t)] = t
                e = self.model(x.to(self.device)).float().cpu()   # [B, 252, 768]
                out.append(e.reshape(-1, e.shape[-1]))
        e = torch.cat(out)[:n]
        return e / e.norm(dim=1, keepdim=True).clamp_min(1e-6)


class FakeEncoder:
    """Stand-in for the encoders in the tests (no downloads): fp_replica's fake, other features."""

    def __init__(self, sr):
        self.sr = sr

    def whole_file(self, y):
        import torch
        e = dc.FakeEmbedder().whole_file(y[::self.sr // 500] if self.sr > 500 else y)
        return torch.roll(e, 3, dims=1)


################################################################################
# the candidates lab
################################################################################

class CandLab:
    def __init__(self, a, pools, w):
        self.a, self.pools, self.w = a, pools, w
        sa = argparse.Namespace(out=a.out, cache=a.cache, seeds=a.seeds, seed=a.seed, deadline=None)
        self.lab = st.Lab(sa, pools, w)
        if SETUP not in self.lab.setups:
            sys.exit(f"setup {SETUP} not found in {a.out}: run dataset_configs.py and study.py first")
        self.lab.use_setup(SETUP)
        c = self.lab.choices()
        if "final" not in c:
            sys.exit("study.py has not chosen its final model yet (no 'final' in _study\\choices.json)")
        self.final_name, self.final_rule = c["final"]["winner"], c["final"]["rule"]
        self.base_spec = self.lab.spec(self.final_name)
        if self.base_spec.get("aug") or self.base_spec.get("hard"):
            log("note: the final model used augmentation or mined frames; the candidates are trained without them")
        self.root = os.path.join(a.out, ROOT_NAME)
        self.feat = os.path.join(a.out, FEATURES_NAME)
        self.device = self.lab.device
        self.seeds = self.lab.seeds
        self.t0 = time.time()

    # ---- bookkeeping ------------------------------------------------------------
    def dir(self, name):
        return os.path.join(self.root, name)

    def jload(self, path):
        return json.load(open(path)) if os.path.exists(path) else None

    def report(self, now):
        try:
            with self.w.open(os.path.join(self.root, "progress.txt")) as f:
                f.write(f"Progress, updated {time.strftime('%Y-%m-%d %H:%M')}\nNow: {now}\n"
                        f"Running for {fmt_dur(time.time() - self.t0)}. Results so far: candidates.md\n")
        except OSError as e:
            log(f"WARNING could not write progress.txt: {e}")
        log(f"PROGRESS: {now}")

    def days(self, which):
        """[(pool, key)] of 'train' (incl. noise clips and held-out days), 'validation', 'tests'."""
        cfg = self.lab.cfg
        if which == "train":
            return [(p, k) for p, ks in cfg["train"].items() for k in ks]
        if which == "validation":
            return [(p, k) for p, ks in cfg["validation"]["files"].items() for k in ks]
        return [(p, k) for s in cfg["tests"] for p, ks in s["files"].items() for k in ks]

    def train_days(self):
        return [(p, k) for p, k in self.days("train") if p != "noise" and (p, k) not in self.lab.heldout]

    # ---- feature sources ----------------------------------------------------------
    def source(self, model):
        aves = Source(self.a.cache)
        if model == "m2":
            return NormSource(aves, self.train_days())
        if model == "m5":
            return ConcatSource(aves, self.feat, "lf", LF_SUB * LF_BINS, scale=LF_SCALE)
        if model == "m7":
            return ConcatSource(aves, self.feat, "pitch", len(PITCH_FEATURES), scale=pitch_scale(self.feat, self.train_days()))
        if model.startswith("m6_"):
            return Source(self.encoder_cache(model[3:]))
        return aves

    def encoder_cache(self, enc):
        return os.path.join(self.a.out, f"_embedding_cache_{enc}")

    def make_fn(self, model, dim):
        ctx = self.base_spec["ctx"]
        if model == "m5":
            return lambda: make_two_stream(ctx)
        return lambda: make_ctx_model(dim, ctx)

    def day_probs(self, model_id, m, x):
        if model_id == "m1":
            return seq_probs(m.to(self.device), x, self.device)
        if hasattr(m, "frame_features"):
            return ex.day_probs(m, x, self.base_spec["ctx"], self.device)
        return window_probs(m, x, self.base_spec["ctx"], self.device)

    # ---- training -----------------------------------------------------------------
    def trained(self, name):
        return all(os.path.exists(os.path.join(self.dir(name), f"model_seed{s}.pth")) for s in self.seeds) and \
            os.path.exists(os.path.join(self.dir(name), "training.json"))

    def train(self, model):
        import torch
        if self.trained(model):
            return
        src = self.source(model)
        self.report(f"{model}: building the training set")
        t0 = time.time()
        if model == "m1":
            bank, tr, ho, info = build_sequences(self, src)
        else:
            bank, win, y, ho_w, ho_y, info = build_sets(self, self.base_spec, src)
        runs = []
        with self.w.open(os.path.join(self.dir(model), "training_log.txt"), "a") as logf:
            logf.write(json.dumps(info) + "\n")
            for seed in self.seeds:
                path = os.path.join(self.dir(model), f"model_seed{seed}.pth")
                if os.path.exists(path) and os.path.exists(path + ".json"):
                    runs.append(json.load(open(path + ".json")))
                    continue
                self.report(f"{model}: training run {seed}")
                if model == "m1":
                    m, ri = train_sequences(bank, tr, ho, src.dim, seed, logf, self.device)
                else:
                    m, ri = train_windows(bank, win, y, ho_w, ho_y, self.make_fn(model, src.dim), seed, logf, self.device)
                st.tsave(self.w, path, m.state_dict())
                st.wjson(self.w, path + ".json", ri)
                runs.append(ri)
                log(f"{model} run {seed}: {ri}")
        st.wjson(self.w, os.path.join(self.dir(model), "training.json"),
                 {"model": model, "base": self.final_name, "dim": src.dim, "data": info, "runs": runs,
                  "train_minutes": round((time.time() - t0) / 60, 1)})

    def models(self, model):
        import torch
        dim = self.jload(os.path.join(self.dir(model), "training.json"))["dim"]
        out = []
        for seed in self.seeds:
            m = make_seq_model(dim) if model == "m1" else self.make_fn(model, dim)()
            m.load_state_dict(torch.load(os.path.join(self.dir(model), f"model_seed{seed}.pth"), map_location="cpu"))
            out.append(m.eval().to(self.device))
        return out

    def probs(self, model, files):
        src = self.source(model)
        ms = self.models(model)
        out = [dict() for _ in ms]
        for p, k, _ in files:
            x = src.day(p, k).whole()
            for i, m in enumerate(ms):
                out[i][(p, k)] = self.day_probs(model, m, x)
        return out

    def files_of(self, model, keys):
        """[(pool, key, hours)] on the model's own step grid."""
        src = self.source(model) if model.startswith("m6_") else Source(self.a.cache)
        out = []
        for p, k in keys:
            d = src.day(p, k)
            out.append((p, k, d.n * FRAME_SECONDS / 3600))
        return out

    # ---- validation: tune the alarm rule, score -------------------------------------
    def evaluate(self, model):
        path = os.path.join(self.dir(model), "results.json")
        r = self.jload(path)
        if r and r.get("seeds") == self.seeds:
            return r
        if model == "m4":
            return self.evaluate_m4()
        self.report(f"{model}: alarm rules on the validation days")
        files = self.files_of(model, self.days("validation"))
        truth = im.Truth(self.pools, files)
        probs = self.probs(model, files)
        rules = st.all_rules(im.RULE_OPTIONS)
        rows, caches, t0 = [], [{} for _ in probs], time.time()
        for i, rule in enumerate(rules):
            summ = st.summarize([im.sweep(truth, pr, rule, cc) for pr, cc in zip(probs, caches)])
            simple = (rule["smooth"], rule["gap"], rule["score"] != "mean", -rule["min_frames"])
            rows.append((st.rule_key(rule), summ, simple, rule))
            if (i + 1) % 400 == 0:
                log(f"{model}: {i + 1}/{len(rules)} rules ({fmt_dur(time.time() - t0)})")
        win, ranked, why = st.choose([(k, sm, sp) for k, sm, sp, _ in rows])
        best = next(r for k, _, _, r in rows if k == win)
        summ = next(sm for k, sm, _, _ in rows if k == win)
        sweeps = [im.sweep(truth, pr, best) for pr in probs]
        im.write_curve(self.w, self.dir(model), model, sweeps)
        r = {"model": model, "title": TITLES[model], "rule": best, "why": why, "seeds": self.seeds,
             "summary": summ, "validation_days": len(files), "validation_rumbles": truth.n,
             "training": self.jload(os.path.join(self.dir(model), "training.json")),
             "finished": time.strftime("%Y-%m-%d %H:%M")}
        st.wjson(self.w, path, r)
        log(f"{model}: {st.fmt_score(summ)} with {im.rule_text(best)}")
        return r

    # ---- model 4 ----------------------------------------------------------------------
    def first_stage_events(self, models, ctx, keys, src=None):
        """{(pool, key): [(first, last, score) per model]} with the final model's alarm rule."""
        src = src or Source(self.a.cache)
        out = {}
        for p, k in keys:
            x = src.day(p, k).whole()
            ev = []
            for m in models:
                pr = ex.day_probs(m, x, ctx, self.device)
                ev.append(im.alarms(pr, self.final_rule))
            out[(p, k)] = ev
        return out

    def fold_models(self):
        """Two copies of the final model, each trained on half of the training days (one run),
        so every training day gets candidates from a model that never saw it."""
        import torch
        keys = self.train_days()
        fold = {pk: int(hashlib.md5(pk[1].encode()).hexdigest(), 16) % 2 for pk in keys}
        ms = []
        for f in (0, 1):
            path = os.path.join(self.dir("m4"), f"first_stage_fold{f}.pth")
            if not os.path.exists(path):
                self.report(f"m4: first stage trained without fold {f}")
                cfg0 = self.lab.cfg
                cfg = copy.deepcopy(cfg0)
                for p in cfg["train"]:
                    if p != "noise":
                        cfg["train"][p] = [k for k in cfg["train"][p] if fold.get((p, k), 1 - f) != f]
                self.lab.cfg = cfg
                try:
                    bank, win, y, ho_w, ho_y, info = build_sets(self, self.base_spec, Source(self.a.cache))
                finally:
                    self.lab.cfg = cfg0
                with self.w.open(os.path.join(self.dir("m4"), "training_log.txt"), "a") as logf:
                    m, ri = train_windows(bank, win, y, ho_w, ho_y, lambda: make_ctx_model(768, self.base_spec["ctx"]),
                                          self.seeds[0], logf, self.device)
                del bank, win, y
                st.tsave(self.w, path, m.state_dict())
            m = make_ctx_model(768, self.base_spec["ctx"])
            m.load_state_dict(torch.load(path, map_location="cpu"))
            ms.append(m.eval().to(self.device))
        return ms, fold

    def cut_path(self, part, p, k):
        return os.path.join(self.dir("m4"), "cuts", part, p, f"{k}.npz")

    def make_cuts(self, part, p, k, boxes):
        """Spectrogram cuts around each (first, last) box: [N, 128, cols] uint8."""
        import numpy as np
        path = self.cut_path(part, p, k)
        if os.path.exists(path):
            return np.load(path)["x"]
        y = dc.load_audio(self.pools[p].sounds[k]).numpy()
        cuts = np.stack([spectrogram_cut(y, (a + b + 1) / 2 * FRAME_SECONDS) for a, b in boxes]) if len(boxes) \
            else np.zeros((0, 128, cut_cols()), np.uint8)
        self.w.makedirs(os.path.dirname(path))
        with open(self.w.check(path + ".tmp"), "wb") as f:
            np.savez(f, x=cuts, boxes=np.asarray(boxes, np.int64).reshape(-1, 2))
        os.replace(self.w.check(path + ".tmp"), self.w.check(path))
        return cuts

    def verifier_training_set(self):
        """Cuts from the training days: labelled rumbles and the fold models' false alarms;
        the held-out days' cuts are kept apart for stopping."""
        import numpy as np
        ms, fold = self.fold_models()
        held = sorted(self.lab.heldout)
        xs = {"train": [], "held": []}
        ys = {"train": [], "held": []}
        keys = [(pk, "train") for pk in self.train_days()] + [(pk, "held") for pk in held]
        for i, ((p, k), part) in enumerate(keys):
            if i % 20 == 0:
                self.report(f"m4: cuts of training day {i + 1}/{len(keys)}")
            m = ms[1 - fold[(p, k)]] if part == "train" else ms[0]
            labels = self.pools[p].labels.get(k, [])
            s, e, v = self.first_stage_events([m], self.base_spec["ctx"], [(p, k)])[(p, k)][0]
            truth = im.Truth(self.pools, [(p, k, 1.0)])
            hit, _ = truth.match((p, k), s, e, v)
            neg = [(a, b, sc) for a, b, sc, h in zip(s, e, v, hit) if not h
                   and not (p == "clearing" and sc > VER_SKIP_ABOVE)]
            neg = sorted(neg, key=lambda t: -t[2])[:VER_NEG_PER_DAY]
            pos = [(dc.fp_data.time_to_index(l.start), dc.fp_data.time_to_index(l.start + l.duration)) for l in labels]
            r = np.random.default_rng(int(hashlib.md5(k.encode()).hexdigest(), 16) % 2**32)
            if len(pos) > VER_POS_PER_DAY:
                pos = [pos[j] for j in sorted(r.choice(len(pos), VER_POS_PER_DAY, replace=False))]
            hits = [(int(a), int(b)) for a, b, h in zip(s, e, hit) if h][:VER_POS_PER_DAY // 2]
            boxes = pos + hits + [(int(a), int(b)) for a, b, _ in neg]
            lab_ = [1] * (len(pos) + len(hits)) + [0] * len(neg)
            if not boxes:
                continue
            xs[part].append(self.make_cuts("train", p, k, boxes))
            ys[part].append(np.asarray(lab_, np.int64))
        cat = lambda v: np.concatenate(v) if v else np.zeros((0, 128, cut_cols()), np.uint8)  # noqa: E731
        return (cat(xs["train"]), np.concatenate(ys["train"])), \
            (cat(xs["held"]), np.concatenate(ys["held"]) if ys["held"] else np.zeros(0, np.int64))

    def train_m4(self):
        import torch
        if self.trained("m4"):
            return
        t0 = time.time()
        (x, y), (hx, hy) = self.verifier_training_set()
        info = {"cuts": int(len(y)), "rumble_cuts": int(y.sum()), "heldout_cuts": int(len(hy))}
        log(f"m4 verifier training set: {info}")
        runs = []
        with self.w.open(os.path.join(self.dir("m4"), "training_log.txt"), "a") as logf:
            for seed in self.seeds:
                path = os.path.join(self.dir("m4"), f"model_seed{seed}.pth")
                if os.path.exists(path) and os.path.exists(path + ".json"):
                    runs.append(json.load(open(path + ".json")))
                    continue
                self.report(f"m4: training checker run {seed}")
                m, ri = train_verifier(x, y, hx, hy, seed, logf, self.device, pretrained=not self.a.fake_embedder)
                st.tsave(self.w, path, m.state_dict())
                st.wjson(self.w, path + ".json", ri)
                runs.append(ri)
                log(f"m4 run {seed}: {ri}")
        st.wjson(self.w, os.path.join(self.dir("m4"), "training.json"),
                 {"model": "m4", "base": self.final_name, "dim": 0, "data": info, "runs": runs,
                  "train_minutes": round((time.time() - t0) / 60, 1)})

    def m4_events(self, keys, part):
        """Per run: {(pool, key): (first, last, first-stage score, checker score)}."""
        import numpy as np
        import torch
        finals = self.lab.models(self.final_name)
        checkers = []
        for seed in self.seeds:
            m = make_verifier(pretrained=False)
            m.load_state_dict(torch.load(os.path.join(self.dir("m4"), f"model_seed{seed}.pth"), map_location="cpu"))
            checkers.append(m.eval().to(self.device))
        ev = self.first_stage_events(finals, self.base_spec["ctx"], keys)
        out = [dict() for _ in self.seeds]
        for (p, k), per in ev.items():
            boxes = sorted({(int(a), int(b)) for s, e, _ in per for a, b in zip(s, e)})
            cuts = self.make_cuts(part, p, k, boxes)
            where = {b: i for i, b in enumerate(boxes)}
            for i, (s, e, v) in enumerate(per):
                idx = [where[(int(a), int(b))] for a, b in zip(s, e)]
                pv = verifier_scores(checkers[i], cuts[idx], self.device) if idx else np.zeros(0)
                out[i][(p, k)] = (np.asarray(s), np.asarray(e), np.asarray(v, np.float64), pv)
        return out

    @staticmethod
    def combine(ev, how):
        import numpy as np
        return {d: (s, e, pv if how == "verifier" else np.sqrt(np.clip(v, 0, 1) * pv))
                for d, (s, e, v, pv) in ev.items()}

    def evaluate_m4(self):
        self.report("m4: scoring the validation days")
        keys = self.days("validation")
        files = self.files_of("m4", keys)
        truth = im.Truth(self.pools, files)
        ev = self.m4_events(keys, "validation")
        options = []
        for how in VER_COMBOS:
            sweeps = [sweep_events(truth, self.combine(e, how)) for e in ev]
            options.append((how, st.summarize(sweeps), VER_COMBOS.index(how), sweeps))
        win, _, why = st.choose([(h, sm, i) for h, sm, i, _ in options])
        how, summ, _, sweeps = next(o for o in options if o[0] == win)
        im.write_curve(self.w, self.dir("m4"), "m4", sweeps)
        r = {"model": "m4", "title": TITLES["m4"], "rule": self.final_rule, "combine": how, "why": why,
             "seeds": self.seeds, "summary": summ, "validation_days": len(files), "validation_rumbles": truth.n,
             "options": {h: st.fmt_score(sm) for h, sm, _, _ in options},
             "training": self.jload(os.path.join(self.dir("m4"), "training.json")),
             "finished": time.strftime("%Y-%m-%d %H:%M")}
        st.wjson(self.w, os.path.join(self.dir("m4"), "results.json"), r)
        log(f"m4: {st.fmt_score(summ)} (scores: {how})")
        return r


################################################################################
# model 4: spectrogram cuts and the checker
################################################################################

def cut_cols():
    return int(VER_CUT_S * 500) // VER_HOP + 1


def spectrogram_cut(y, center_s, sr=500):
    """[128, cols] uint8: dB spectrogram 1.95-250 Hz around center_s, each frequency row
    minus its median (removes steady background), VER_DB_RANGE dB mapped to 0..255."""
    import numpy as np
    n = int(VER_CUT_S * sr)
    a = int(round((center_s - VER_CUT_S / 2) * sr))
    seg = np.zeros(n, np.float32)
    lo, hi = max(0, a), min(len(y), a + n)
    if hi > lo:
        seg[lo - a:hi - a] = y[lo:hi]
    mag = stft_mag(seg, VER_NFFT, VER_HOP)[1:129]
    db = 20 * np.log10(mag + 1e-7)
    db = db - np.median(db, axis=1, keepdims=True)
    return np.clip((db + VER_DB_RANGE / 4) / VER_DB_RANGE * 255, 0, 255).astype(np.uint8)


def make_verifier(pretrained=True):
    """ResNet-18 (ImageNet weights) on one-channel spectrograms, one output."""
    import torch
    import torchvision
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    w = m.conv1.weight.data.sum(1, keepdim=True)
    m.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    m.conv1.weight.data = w
    m.fc = torch.nn.Linear(m.fc.in_features, 1)
    return m


def crop(x, train, rng=None):
    """x: [B, 128, cols] uint8 -> float tensor [B, 1, 128, use_cols], random shift when training."""
    import numpy as np
    import torch
    use = int(VER_USE_S * 500) // VER_HOP
    extra = x.shape[2] - use
    if train:
        offs = rng.integers(0, extra + 1, len(x))
        x = np.stack([x[i, :, o:o + use] for i, o in enumerate(offs)])
    else:
        x = x[:, :, extra // 2:extra // 2 + use]
    t = torch.from_numpy(np.ascontiguousarray(x)).float().div_(255).sub_(0.5).div_(0.25).unsqueeze(1)
    if train:
        for i in range(len(t)):
            f0, fw = rng.integers(0, 128), rng.integers(0, 13)
            t[i, :, f0:f0 + fw] = 0
            t0, tw = rng.integers(0, use), rng.integers(0, 31)
            t[i, :, :, t0:t0 + tw] = 0
    return t


def verifier_scores(model, cuts, device):
    import numpy as np
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(cuts), 256):
            out.append(torch.sigmoid(model(crop(cuts[i:i + 256], False).to(device))[:, 0]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_verifier(x, y, hx, hy, seed, logf, device, pretrained=True):
    import numpy as np
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_verifier(pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    pos_w = torch.tensor(float((y == 0).sum()) / max(1, int((y == 1).sum())), device=device)
    best, best_ep, best_state, since, ep = -1.0, 0, None, 0, 0
    t0 = time.time()
    for ep in range(1, VER_MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(y))
        for i in range(0, len(order), VER_BATCH):
            b = order[i:i + VER_BATCH]
            xb = crop(x[b], True, rng).to(device)
            yb = torch.from_numpy(y[b]).float().to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb)[:, 0], yb, pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        if len(hy) and hy.min() != hy.max():
            ap = st.average_precision(torch.from_numpy(verifier_scores(model, hx, device)), torch.from_numpy(hy))
        else:
            ap = -loss.item()
        better = ap > best + 1e-5
        logf.write(f"seed {seed} epoch {ep}: loss {loss.item():.4f}, held-out AP {ap:.4f}{' *' if better else ''}\n")
        if better:
            best, best_ep, since = ap, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
            if since >= VER_PATIENCE:
                break
    model.load_state_dict(best_state)
    info = {"epochs": ep, "best_epoch": best_ep, "heldout_ap": round(best, 5), "seconds": round(time.time() - t0, 1)}
    logf.write(json.dumps(info) + "\n")
    return model.eval().cpu(), info


################################################################################
# commands
################################################################################

def cmd_features(cl):
    """One pass over the audio: low-pitch PCEN spectrogram (m5) and pitch features (m7)."""
    import numpy as np
    keys = cl.days("train") + cl.days("validation") + cl.days("tests")
    todo = [(p, k) for p, k in keys if not all(os.path.exists(feature_path(cl.feat, n, p, k)) for n in ("lf", "pitch"))]
    log(f"features: {len(todo)} of {len(keys)} recordings to do")
    t0 = time.time()
    for i, (p, k) in enumerate(todo, 1):
        y = dc.load_audio(cl.pools[p].sounds[k]).numpy()
        n = max(1, int(math.ceil(len(y) / 320)))
        for name, fn in (("lf", lf_features), ("pitch", pitch_features)):
            path = feature_path(cl.feat, name, p, k)
            if os.path.exists(path):
                continue
            cl.w.makedirs(os.path.dirname(path))
            with open(cl.w.check(path + ".tmp"), "wb") as f:
                np.save(f, fn(y, n))
            os.replace(cl.w.check(path + ".tmp"), cl.w.check(path))
        if i % 10 == 0 or i == len(todo):
            left = (time.time() - t0) / i * (len(todo) - i)
            cl.report(f"features: {i}/{len(todo)} recordings, about {fmt_dur(left)} left")


def cmd_embed(cl, enc):
    """Fingerprints of every recording the candidates use, from another encoder, in the
    same cache layout as AVES's (one per 0.64 s, unit length)."""
    import torch
    import fp_replica
    cache = cl.encoder_cache(enc)
    keys = cl.days("train") + cl.days("validation") + cl.days("tests")
    todo = [(p, k) for p, k in keys if not os.path.exists(dc.cache_path(cache, p, k, "day"))]
    log(f"{enc}: {len(todo)} of {len(keys)} recordings to fingerprint")
    if not todo:
        return
    spec = ENCODERS[enc]
    if cl.a.fake_embedder:
        embed = FakeEncoder(spec["sr"])
    elif spec["kind"] == "aves":
        embed = load_aves_like(spec, cl.device)
    else:
        embed = DashengEmbedder(cl.device)
    t0 = time.time()
    for i, (p, k) in enumerate(todo, 1):
        y = fp_replica.load_audio(cl.pools[p].sounds[k], sr_out=spec["sr"])
        e = embed.whole_file(y)
        dst = dc.cache_path(cache, p, k, "day")
        cl.w.makedirs(os.path.dirname(dst))
        torch.save({"kind": "day", "n": e.shape[0], "emb": e.half()}, cl.w.check(dst + ".tmp"))
        os.replace(cl.w.check(dst + ".tmp"), cl.w.check(dst))
        if i % 10 == 0 or i == len(todo):
            left = (time.time() - t0) / i * (len(todo) - i)
            cl.report(f"{enc} fingerprints: {i}/{len(todo)} recordings, about {fmt_dur(left)} left")


def run_model(cl, model):
    if model == "m4":
        cl.train_m4()
    else:
        cl.train(model)
    return cl.evaluate(model)


def baseline_summary(cl):
    r = cl.lab.result(cl.final_name, cl.final_rule)
    return r["summary"] if r else None


def write_report(cl):
    base = baseline_summary(cl)
    head = ["# Candidate models on the validation days", "",
            "Same validation days, scoring and number of runs (5) as the final model of the rerun. "
            "**Score = mean false alarms per hour at 80% and 85% recall** (lower is better), with the range "
            "over runs. A candidate beats the final model only if its score is lower by more than the "
            "run-to-run spread.", "",
            "| Model | Score | FA/h at 80% | at 85% | at 90% | Max recall | Alarm rule / notes |", "|---|---|---|---|---|---|---|"]
    rows = []

    def cells(s):
        f = lambda x: "-" if not x else f"{x[0]:.3f}"  # noqa: E731
        sc = "85% not reached by every run" if not s["score"] else f"**{s['score'][0]:.3f}** ({s['score'][1]:.3f}-{s['score'][2]:.3f})"
        return f"{sc} | {f(s['fp_at_80'])} | {f(s['fp_at_85'])} | {f(s['fp_at_90'])} | {f(s['max_recall'])}"
    if base:
        rows.append(f"| Final model ({cl.final_name}) | {cells(base)} | {im.rule_text(cl.final_rule)} |")
    for m in ALL_MODELS:
        r = cl.jload(os.path.join(cl.dir(m), "results.json"))
        if r:
            note = f"scores: {r['combine']}" if m == "m4" else im.rule_text(r["rule"])
            rows.append(f"| {TITLES[m]} | {cells(r['summary'])} | {note} |")
        elif os.path.exists(os.path.join(cl.dir(m), "failed.txt")):
            rows.append(f"| {TITLES[m]} | failed, see {m}\\failed.txt | | | | | |")
    with cl.w.open(os.path.join(cl.root, "candidates.md")) as f:
        f.write("\n".join(head + rows) + "\n")


def record_failure(cl, model):
    tb = traceback.format_exc()
    log(f"FAILED {model}:\n{tb}")
    try:
        with cl.w.open(os.path.join(cl.dir(model), "failed.txt"), "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M") + "\n" + tb + "\n")
    except OSError:
        pass


def cmd_all(cl, only):
    want = [m for m in ALL_MODELS if not only or m in only or m.split("_")[0] in only]
    if any(m in want for m in ("m5", "m7")):
        try:
            cl.report("features for models 5 and 7")
            cmd_features(cl)
        except Exception:
            record_failure(cl, "features")
    for m in want:
        try:
            if m.startswith("m6_"):
                cl.report(f"{m}: fingerprints")
                cmd_embed(cl, m[3:])
            run_model(cl, m)
        except Exception:
            record_failure(cl, m)
        write_report(cl)
    cl.report("finished the candidates; run `final` for the tests")


def cmd_final(cl, only):
    """The tests, once: each run at the cut-off that found 80% (85%) of the validation rumbles."""
    sets = [("sampleB", "Test 1: Sample B"), ("sampleA", "Test 2: Sample A"), ("vehicle", "Vehicle (false alarms only)")]
    tests = {s["key"]: [(p, k) for p, ks in s["files"].items() for k in ks] for s in cl.lab.cfg["tests"]}
    head = ["# Candidate models on the tests, scored once", "",
            "Each run uses the cut-off that found 80% (or 85%) of the rumbles on the validation days. Means over "
            f"{len(cl.seeds)} runs. Each cell: recall / precision / false alarms per hour.", "",
            "| Model | Validation score | " + " | ".join(f"{n} at 80% | at 85%" for _, n in sets) + " |",
            "|---|---|" + "---|---|" * len(sets)]
    rows, out = [], {}
    for m in ALL_MODELS:
        if only and m not in only and m.split("_")[0] not in only:
            continue
        r = cl.jload(os.path.join(cl.dir(m), "results.json"))
        if not r:
            continue
        try:
            cl.report(f"tests: {m}")
            cells, res = [], {}
            for key, _ in sets:
                keys = tests.get(key, [])
                if not keys:
                    cells += ["-", "-"]
                    continue
                files = cl.files_of(m, keys)
                truth = im.Truth(cl.pools, files)
                if m == "m4":
                    sweeps = [sweep_events(truth, cl.combine(e, r["combine"])) for e in cl.m4_events(keys, "tests")]
                else:
                    sweeps = [im.sweep(truth, pr, r["rule"]) for pr in cl.probs(m, files)]
                res[key] = {}
                for rec in (80, 85):
                    per = []
                    for sw, ps in zip(sweeps, r["summary"]["per_seed"]):
                        cut = (ps.get(f"at_{rec}") or {}).get("threshold")
                        if cut is not None:
                            per.append(at_cut(sw, cut))
                    mean = {k: statistics.mean([x[k] for x in per if x[k] == x[k]] or [float("nan")])
                            for k in ("recall", "precision", "fp_per_hour")}
                    res[key][rec] = {"per_run": per, "mean": mean}
                    cells.append(f"{im._f(mean['recall'])} / {im._f(mean['precision'])} / {im._f(mean['fp_per_hour'], 2)}")
            out[m] = res
            sc = r["summary"]["score"]
            rows.append(f"| {TITLES[m]} | {'-' if not sc else f'{sc[0]:.3f}'} | " + " | ".join(cells) + " |")
        except Exception:
            record_failure(cl, m)
    fin = cl.jload(os.path.join(cl.lab.root, "final.md").replace(".md", ".json"))
    if fin:
        for s in fin.get("steps", []):
            if s["step"].startswith("10"):
                cells = []
                for key, _ in sets:
                    for rec in ("80", "85"):
                        mm = ((s["sets"].get(key) or {}).get(rec) or {}).get("mean")
                        cells.append("-" if not mm else f"{im._f(mm['recall'])} / {im._f(mm['precision'])} / "
                                                        f"{im._f(mm['fp_per_hour'], 2)}")
                rows.insert(0, f"| Final model of the rerun | - | " + " | ".join(cells) + " |")
    st.wjson(cl.w, os.path.join(cl.root, "final.json"), out)
    with cl.w.open(os.path.join(cl.root, "final.md")) as f:
        f.write("\n".join(head + rows) + "\n")
    print("\n".join(head + rows))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", nargs="?", default="all", choices=["all", "features", "embed", "report", "final"])
    p.add_argument("--only", nargs="+", help="models to run, e.g. m4 m1 (m6 = both encoders)")
    p.add_argument("--encoder", choices=list(ENCODERS), help="for `embed`")
    p.add_argument("--source", default=dc.DEFAULT_SOURCE, help=f"the Datasets copy (default {dc.DEFAULT_SOURCE})")
    p.add_argument("--out", help="the Configurations folder (default <source>\\Configurations)")
    p.add_argument("--cache", help="the AVES cache (default <out>\\_embedding_cache)")
    p.add_argument("--seeds", type=int, default=st.SEEDS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.out = a.out or os.path.join(a.source, "Configurations")
    a.cache = a.cache or os.path.join(a.out, "_embedding_cache")
    im.keep_awake()
    root = dc.find_dataset_root(a.source)
    w = dc.Writer([os.path.join(a.out, ROOT_NAME), os.path.join(a.out, FEATURES_NAME)]
                  + [os.path.join(a.out, f"_embedding_cache_{e}") for e in ENCODERS],
                  [dc.ORIGINAL_DATA, os.path.join(root, "Training"), os.path.join(root, "Testing")])
    w.makedirs(os.path.join(a.out, ROOT_NAME))
    stdout = sys.stdout
    sys.stdout = dc.Tee(w.check(os.path.join(a.out, ROOT_NAME, "run_log.txt")))
    try:
        pools = dc.load_pools(root)
        cl = CandLab(a, pools, w)
        log(f"candidates: final model {cl.final_name}, rule {im.rule_text(cl.final_rule)}; writing to {cl.root}")
        if a.step == "features":
            cmd_features(cl)
        elif a.step == "embed":
            for e in ([a.encoder] if a.encoder else list(ENCODERS)):
                cmd_embed(cl, e)
        elif a.step == "report":
            write_report(cl)
        elif a.step == "final":
            cmd_final(cl, a.only)
        else:
            cmd_all(cl, a.only)
            cmd_final(cl, a.only)
        write_report(cl)
    finally:
        sys.stdout = stdout


if __name__ == "__main__":
    main()
