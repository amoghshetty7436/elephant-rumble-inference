#!/usr/bin/env python
"""
Improve the FruitPunch rumble detector one change at a time, starting from the
E2 data configuration (clearing + forest + noise) of dataset_configs.py, and
find the best combination of all the changes. Plan: walkthrough/16.

The yardstick, for every experiment: false alarms per hour on the validation days
(RumbleComplex + RumbleGeneral) at the cut-off that finds --target-recall (80%) of
their rumbles, averaged over --seeds training runs (5). A change is kept only if it
beats the current best by more than the spread between runs; within that spread the
simpler / faster option wins. The test sets are not looked at until `final`.

Stages (all run by `all`; rerunning resumes, nothing finished is redone):

  0  baseline     E2 with FruitPunch's settings
  1  rules        how frame probabilities become alarms (no retraining): "on" level,
                  minimum length, smoothing, joining nearby alarms, alarm score.
                  The winner is locked for stages 2-5.
  2a noise        how often each noise clip is used as a "not rumble" example
  2b random       random forest "not rumble" frames per forest training day
  2c negatives    "not rumble" examples per rumble example
  3  window       seconds of audio the classifier sees at once (0.64 s to 12.2 s)
  4  combos       every combination of the best 2 options of stages 2a, 2b, 2c and 3
  5  mistakes     1-3 rounds of adding the model's own false alarms on training days
  6  rules again  re-tune the alarm rules for the final model

Every model's inference time (classifier, per 24 h of audio, GPU and CPU) and size
are measured; AVES time comes from dataset_configs' _timing.json.

Steps:

  embed    whole-day embeddings of the training days (needed for stages 2-5; once)
  all      embed + every stage (default)
  compare  rewrite <out>\\_improve\\improve.md from the results so far
  final    score the baseline and every stage winner on the tests, once

On the lab PC, after dataset_configs.py has built E_train-clearing+forest+noise:

  python tools\\improve.py            # overnight; progress in _improve\\progress.txt
  python tools\\improve.py final      # when it has finished
"""
import argparse
import csv
import itertools
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset_configs as dc  # noqa: E402
import experiments as ex  # noqa: E402
from dataset_configs import log, fmt_dur, Day, FRAME_SECONDS  # noqa: E402

BASE_CONFIG = "E_train-clearing+forest+noise"
SEEDS = 5
MINUTES = 5          # training time per run, the same for every experiment
TARGET_RECALL = 0.80
ALSO_SHOWN = (0.85, 0.90)
EXTRA_CAP = 0.75     # noise + random frames may be at most this share of the negatives
HARD_SHARE = 1 / 3   # mined false alarms may be at most this share of the negatives
MAX_RAM_SHARE = 0.55  # an experiment whose training embeddings need more of the RAM is skipped

# the options of each stage; the first value is FruitPunch's / E2's
NOISE_REPEATS = [1, 2, 3, 5, 10, 20]
RANDOM_PER_DAY = [200, 0, 500, 1000, 2000]
NEG_RATIOS = [1, 2, 3, 5]
CONTEXTS = list(range(10))   # frames on each side: (2c+1) x 0.64 s = 0.64 ... 12.16 s
HARD_ROUNDS = [1, 2, 3]
BASE = dict(ctx=0, noise=1, rand=200, neg=1, hard=0)

FRUITPUNCH_RULE = dict(on=0.5, min_frames=4, smooth=1, gap=0, score="mean")
RULE_OPTIONS = dict(on=[0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2], min_frames=[2, 3, 4, 5, 6, 7, 8],
                    smooth=[1, 3, 5, 7, 9], gap=[0, 1, 2, 3], score=["mean", "max"])

# cut-offs: 0.01 steps, and 0.001 steps above 0.9 where the scores pile up
THRESHOLDS = sorted({round(x / 100, 3) for x in range(0, 90)} | {round(0.9 + x / 1000, 3) for x in range(0, 101)})

STAGES = [("2a", "noise", "noise clips used", NOISE_REPEATS),
          ("2b", "rand", "random forest frames per day", RANDOM_PER_DAY),
          ("2c", "neg", "negatives per rumble", NEG_RATIOS),
          ("3", "ctx", "listening window", CONTEXTS)]


def name_of(k):
    return f"ctx{k['ctx']}_noise{k['noise']}_rand{k['rand']}_neg{k['neg']}_hard{k['hard']}"


def knobs_of(name):
    out = {}
    for part in name.split("_"):
        key = part.rstrip("0123456789")
        out[key] = int(part[len(key):])
    return out


def window_seconds(ctx):
    return round((2 * ctx + 1) * FRAME_SECONDS, 2)


def describe(k):
    return (f"{window_seconds(k['ctx'])} s window, noise x{k['noise']}, {k['rand']} random frames/day, "
            f"1:{k['neg']} negatives, {k['hard']} hard-negative rounds")


def cost(k):
    """Simpler / faster first: used to break ties within the run-to-run spread."""
    return (k["hard"], k["ctx"], k["neg"], k["noise"], k["rand"])


def rule_key(r):
    return f"on{r['on']}_min{r['min_frames']}_smooth{r['smooth']}_gap{r['gap']}_{r['score']}"


def rule_text(r):
    return (f"on > {r['on']}, at least {r['min_frames']} frames, smoothing {r['smooth']}, "
            f"join gaps of {r['gap']}, score = {r['score']}")


################################################################################
# alarms and fast ELP scoring
################################################################################

def smoothed(prob, s):
    import numpy as np
    if s <= 1 or not len(prob):
        return prob
    k = np.ones(s)
    return np.convolve(prob, k, "same") / np.convolve(np.ones_like(prob), k, "same")


def runs_above(p, on):
    import numpy as np
    d = np.diff(np.concatenate([[0], (p > on).astype(np.int8), [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def merge_runs(starts, ends, gap):
    import numpy as np
    if len(starts) == 0 or gap <= 0:
        return starts, ends
    first = np.flatnonzero(np.concatenate([[True], starts[1:] - ends[:-1] - 1 > gap]))
    last = np.concatenate([first[1:] - 1, [len(starts) - 1]])
    return starts[first], ends[last]


def run_scores(prob, starts, ends, how):
    import numpy as np
    if len(starts) == 0:
        return np.zeros(0)
    if how == "mean":
        cs = np.concatenate([[0.0], np.cumsum(prob, dtype=np.float64)])
        return (cs[ends + 1] - cs[starts]) / (ends - starts + 1)
    ext = np.concatenate([prob, [0.0]]).astype(np.float64)
    return np.maximum.reduceat(ext, np.column_stack([starts, ends + 1]).ravel())[::2]


def alarms(prob, rule, runs=None):
    """(first frames, last frames, scores) of one day's alarms. With FRUITPUNCH_RULE this is
    FruitPunch's rule: runs of >= 4 frames over 0.5, scored by their mean probability.
    `runs` caches the runs above the "on" level per (smoothing, on level) when tuning."""
    import numpy as np
    key = (rule["smooth"], rule["on"])
    if runs is not None and key in runs:
        s, e = runs[key]
    else:
        s, e = runs_above(smoothed(prob, rule["smooth"]), rule["on"])
        if runs is not None:
            runs[key] = (s, e)
    s, e = merge_runs(s, e, rule["gap"])
    keep = e - s + 1 >= rule["min_frames"]
    s, e = s[keep], e[keep]
    return s, e, run_scores(prob, s, e, rule["score"]) if len(s) else np.zeros(0)


class Truth:
    """Truth boxes of a set of days, as arrays, for fast matching (ELP's rule: more than 1%
    overlap in time and in frequency, as a share of the shorter box; see elp_score)."""

    def __init__(self, pools, files, overlap=0.01):
        import numpy as np
        self.overlap, self.days, self.hours = overlap, {}, 0.0
        for p, k, h in files:
            ls = pools[p].labels.get(k, [])
            t0 = np.array([l.start for l in ls], dtype=np.float64)
            t1 = np.array([l.start + l.duration for l in ls], dtype=np.float64)
            f0 = np.array([l.low_freq for l in ls], dtype=np.float64)
            f1 = np.array([l.high_freq for l in ls], dtype=np.float64)
            fi = np.minimum(250.0, f1) - np.maximum(5.0, f0)
            fs = np.minimum(245.0, f1 - f0)
            freq_ok = (fi > 0) & (fs > 0) & (fi / np.where(fs > 0, fs, 1) > overlap)
            self.days[(p, k)] = (t0, t1, freq_ok)
            self.hours += h
        self.n = sum(len(v[0]) for v in self.days.values())

    def match(self, day, starts, ends, scores):
        """(detection hits a rumble?, best score of a detection hitting each rumble)."""
        import numpy as np
        t0, t1, freq_ok = self.days[day]
        if len(starts) == 0 or len(t0) == 0:
            return np.zeros(len(starts), bool), np.full(len(t0), -np.inf)
        d0, d1 = starts * FRAME_SECONDS, ends * FRAME_SECONDS
        inter = np.minimum(d1[:, None], t1[None]) - np.maximum(d0[:, None], t0[None])
        short = np.minimum((d1 - d0)[:, None], (t1 - t0)[None])
        hit = (inter > 0) & (short > 0) & (inter / np.where(short > 0, short, 1) > self.overlap) & freq_ok[None]
        best = np.where(hit, scores[:, None], -np.inf).max(0)
        return hit.any(1), best


def sweep(truth, probs, rule, cache=None):
    """ELP scoring of one model's validation probabilities at every cut-off in THRESHOLDS.
    Same numbers as elp_score.score, much faster. Returns dict of numpy arrays."""
    import numpy as np
    sc, hits, best = [], [], []
    for day, prob in probs.items():
        s, e, v = alarms(prob, rule, None if cache is None else cache.setdefault(day, {}))
        h, b = truth.match(day, s, e, v)
        sc.append(v)
        hits.append(h)
        best.append(b)
    sc, hits, best = np.concatenate(sc), np.concatenate(hits), np.concatenate(best)
    thr = np.array(THRESHOLDS)
    all_sorted, hit_sorted, best_sorted = np.sort(sc), np.sort(sc[hits]), np.sort(best)
    n_det = len(sc) - np.searchsorted(all_sorted, thr - 1e-12)
    tp = len(hit_sorted) - np.searchsorted(hit_sorted, thr - 1e-12)
    found = len(best_sorted) - np.searchsorted(best_sorted, thr - 1e-12)
    fp = n_det - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(n_det > 0, tp / np.maximum(n_det, 1), np.nan)
        recall = found / truth.n if truth.n else np.full(len(thr), np.nan)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return dict(threshold=thr, detections=n_det, true_detections=tp, fp=fp, rumbles=truth.n,
                rumbles_found=found, precision=precision, recall=recall, f1=np.nan_to_num(f1),
                fp_per_hour=fp / truth.hours if truth.hours else np.full(len(thr), np.nan))


def at_recall(sw, target):
    """The highest cut-off whose recall still reaches the target (None if it never does)."""
    import numpy as np
    ok = np.flatnonzero(sw["recall"] >= target - 1e-9)
    if not len(ok):
        return None
    i = ok.max()
    return {k: (v[i].item() if hasattr(v, "__len__") else v) for k, v in sw.items()}


def summarize(sweeps, target):
    """Per-seed and averaged numbers for one experiment."""
    import numpy as np
    out = {"per_seed": [], "target_recall": target}
    for sw in sweeps:
        row = {"max_recall": float(np.nanmax(sw["recall"])), "best_f1": float(sw["f1"].max()),
               "best_f1_threshold": float(sw["threshold"][int(sw["f1"].argmax())])}
        for t in (target,) + ALSO_SHOWN:
            row[f"at_{int(round(t * 100))}"] = at_recall(sw, t)
        out["per_seed"].append(row)

    def stat(f):
        v = [x for x in (f(r) for r in out["per_seed"]) if x is not None and x == x]
        return (statistics.mean(v), min(v), max(v), statistics.pstdev(v), len(v)) if v else None

    key = f"at_{int(round(target * 100))}"
    out["fp_per_hour"] = stat(lambda r: r[key] and r[key]["fp_per_hour"])
    out["precision"] = stat(lambda r: r[key] and r[key]["precision"])
    out["threshold"] = stat(lambda r: r[key] and r[key]["threshold"])
    for t in ALSO_SHOWN:
        k = f"at_{int(round(t * 100))}"
        out[f"fp_per_hour_{k}"] = stat(lambda r, k=k: r[k] and r[k]["fp_per_hour"])
    out["max_recall"] = stat(lambda r: r["max_recall"])
    out["best_f1"] = stat(lambda r: r["best_f1"])
    out["runs_reaching_target"] = sum(1 for r in out["per_seed"] if r[key])
    return out


def rank(summ):
    """Sort key for a summary: every run reaching the target recall first, then fewest
    false alarms per hour; if not every run reaches it, highest recall."""
    fp = summ["fp_per_hour"]
    if summ["runs_reaching_target"] == len(summ["per_seed"]) and fp:
        return (0, fp[0])
    return (1, -summ["max_recall"][0] if summ["max_recall"] else 0)


################################################################################
# training data
################################################################################

def take(idx, f, rng):
    """A fraction f of idx; f above 1 repeats every index int(f) times, plus a sample."""
    import numpy as np
    idx = np.asarray(sorted(set(idx)), dtype=np.int64)
    if len(idx) == 0 or f <= 0:
        return idx[:0]
    whole, rest = int(f), f - int(f)
    parts = [np.tile(idx, whole)] if whole else []
    if rest > 1e-9:
        parts.append(rng.choice(idx, int(round(rest * len(idx))), replace=False))
    return np.sort(np.concatenate(parts)) if parts else idx[:0]


def total_ram_bytes():
    """Physical memory of this PC (0 if unknown)."""
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.ullTotalPhys)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return 0


def build_training_set(cfg, pools, cache, k, mined, seeds):
    """Rumble frames + neg x as many negatives, as windows of 2*ctx+1 embeddings.

    Negatives: noise clips (each used `noise` times) and `rand` random frames per forest
    day, together at most EXTRA_CAP of them; mined false alarms, at most HARD_SHARE; gap
    frames between rumbles for the rest. The sample is drawn once (from the first seed) and
    shared by all runs, which then differ in initialisation and batch order: separate
    samples per run would multiply the embeddings held in memory (by up to 5x with wide
    windows). Stops early with MemoryError if the embeddings would not fit in RAM.
    Returns (bank [M, 768] fp16, [(windows [N, W], y [N], info)] one per seed)."""
    import numpy as np
    import torch
    ctx, ratio = k["ctx"], k["neg"]
    rngs = [np.random.default_rng(seeds[0])]
    limit = int(MAX_RAM_SHARE * total_ram_bytes())
    pos_n, gap_n = ex.label_counts(cfg, pools)
    n_pos = sum(pos_n.values())
    n_neg = ratio * n_pos
    noise_days = {key: Day(cache, "noise", key) for key in cfg["train"].get("noise", [])}
    n_noise = sum(d.n for d in noise_days.values())
    forest_days = sum(len(f) for p, f in cfg["train"].items() if p != "noise" and dc.POOLS[p]["forest"])
    extra_avail = n_noise * k["noise"] + k["rand"] * forest_days
    extra_target = min(extra_avail, int(n_neg * EXTRA_CAP))
    mined_avail = sum(len(v) for v in mined.values())
    mined_target = min(mined_avail, int(n_neg * HARD_SHARE))
    gap_target = max(0, n_neg - extra_target - mined_target)
    f_gap = min(1.0, gap_target / max(1, sum(gap_n.values())))
    f_extra = extra_target / max(1, extra_avail)
    f_mined = mined_target / max(1, mined_avail)

    banks, offset = [], 0
    per_seed = [{"wins": [], "ys": [], "counts": dict(rumble=0, gap=0, noise=0, random=0, mined=0)} for _ in rngs]
    offsets = np.arange(-ctx, ctx + 1)

    def add(day, groups):
        nonlocal offset
        picks = []
        for c, lab, kind, f in groups:
            c = np.asarray(sorted(set(c)), dtype=np.int64)
            c = c[(c >= 0) & (c < day.n)]
            for i, rng in enumerate(rngs):
                picks.append((i, take(c, f, rng), lab, kind))
        allc = [x for _, x, _, _ in picks if len(x)]
        if not allc:
            return
        need = np.unique(np.clip(np.concatenate(allc)[:, None] + offsets[None, :], 0, day.n - 1))
        if limit and (offset + len(need)) * 768 * 2 > limit:
            raise MemoryError(f"{describe(k)} needs more than {limit / 2**30:.0f} GB of embeddings in memory "
                              f"({MAX_RAM_SHARE:.0%} of this PC's RAM); skipped")
        rows = day.frames(need.tolist())
        if rows.shape[0] != len(need):
            raise RuntimeError("embeddings missing for this day: run `python tools\\improve.py embed`")
        banks.append(rows)
        for i, c, lab, kind in picks:
            if len(c):
                w = np.searchsorted(need, np.clip(c[:, None] + offsets[None, :], 0, day.n - 1)) + offset
                per_seed[i]["wins"].append(torch.from_numpy(w))
                per_seed[i]["ys"].append(torch.full((len(c),), lab, dtype=torch.long))
                per_seed[i]["counts"][kind] += len(c)
        offset += len(need)

    for p, files in cfg["train"].items():
        for key in files:
            if p == "noise":
                day = noise_days[key]
                add(day, [(range(day.n), 0, "noise", f_extra * k["noise"])])
                continue
            day = Day(cache, p, key)
            labels = pools[p].labels.get(key, [])
            groups = []
            if dc.is_usable(labels):
                pi, ni = dc.label_frames(labels)
                groups += [(pi, 1, "rumble", 1.0), (ni, 0, "gap", f_gap)]
            if k["rand"] and cfg["random_negatives"] and dc.POOLS[p]["forest"]:
                if day.kind != "day":
                    raise RuntimeError(f"{p} {key} has no whole-day embeddings: run `python tools\\improve.py embed`")
                groups.append((dc.random_negative_frames(labels, day.n, key, n=k["rand"]), 0, "random", f_extra))
            if (p, key) in mined:
                groups.append((mined[(p, key)], 0, "mined", f_mined))
            if ctx and day.kind != "day":
                raise RuntimeError(f"{p} {key} has no whole-day embeddings: run `python tools\\improve.py embed`")
            add(day, groups)
    # join the pieces without holding two full copies at once
    bank = torch.empty((offset, 768), dtype=banks[0].dtype)
    at = 0
    banks.reverse()
    while banks:
        piece = banks.pop()
        bank[at:at + piece.shape[0]] = piece
        at += piece.shape[0]
    log(f"training set: {bank.shape[0]} embeddings in memory ({bank.shape[0] * 1536 / 2**30:.1f} GB)")
    out = []
    for s in per_seed:
        y = torch.cat(s["ys"])
        c = s["counts"]
        neg = int(y.shape[0]) - c["rumble"]
        info = {"examples": int(y.shape[0]), "embeddings_in_memory": int(bank.shape[0]),
                "rumble": c["rumble"], "negatives_from_gaps": c["gap"], "negatives_from_noise": c["noise"],
                "negatives_random_forest": c["random"], "negatives_mined": c["mined"],
                "negatives_per_rumble": round(neg / max(1, c["rumble"]), 2),
                "noise_and_random_share_of_negatives": round((c["noise"] + c["random"]) / max(1, neg), 3)}
        out.append((torch.cat(s["wins"]), y, info))
    return bank, out * len(seeds)  # one shared sample, trained once per seed


################################################################################
# the lab: experiments, results, choices
################################################################################

class Lab:
    def __init__(self, a, pools, w):
        import torch
        self.a, self.pools, self.w = a, pools, w
        self.root = os.path.join(a.out, "_improve")
        self.cfg_dir = os.path.join(a.out, a.base_config)
        if not os.path.exists(os.path.join(self.cfg_dir, "config.json")):
            sys.exit(f"no {a.base_config} in {a.out}: run `python tools\\dataset_configs.py --configs E --best A` first")
        self.cfg = json.load(open(os.path.join(self.cfg_dir, "config.json")))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seeds = list(range(a.seed, a.seed + a.seeds))
        self._val = None
        self.t_start, self.trained, self.train_seconds = time.time(), 0, []
        self.stage = "starting"
        self.retrained = set()  # with --rerun: experiments already retrained by this process

    def dir(self, name):
        return os.path.join(self.root, name)

    def json(self, name, file):
        p = os.path.join(self.dir(name), file)
        return json.load(open(p)) if os.path.exists(p) else None

    # ---- validation data ---------------------------------------------------
    def validation(self):
        if self._val is None:
            files = []
            for p, keys in self.cfg["validation"]["files"].items():
                for k in keys:
                    files.append((p, k, Day(self.a.cache, p, k).hours))
            self._val = files, Truth(self.pools, files)
        return self._val

    def probs(self, models, ctx, files):
        """[{(pool, key): probabilities}] per model, reading each day once."""
        out = [dict() for _ in models]
        for p, k, _ in files:
            emb = Day(self.a.cache, p, k).whole()
            for i, m in enumerate(models):
                out[i][(p, k)] = ex.day_probs(m, emb, ctx, self.device)
        return out

    def models(self, name):
        import torch
        ctx = knobs_of(name)["ctx"]
        out = []
        for s in self.seeds:
            m = ex.make_model(ctx)
            m.load_state_dict(torch.load(os.path.join(self.dir(name), f"model_seed{s}.pth"), map_location="cpu"))
            out.append(m.eval().to(self.device))
        return out

    # ---- training -----------------------------------------------------------
    def trained_ok(self, name):
        return all(os.path.exists(os.path.join(self.dir(name), f"model_seed{s}.pth")) for s in self.seeds) \
            and os.path.exists(os.path.join(self.dir(name), "training.json"))

    def train(self, name):
        """Train every missing seed of one experiment (and the rounds it builds on)."""
        if self.trained_ok(name) and (not self.a.rerun or name in self.retrained):
            return
        self.retrained.add(name)
        import torch
        k = knobs_of(name)
        mined, n_new = {}, 0
        if k["hard"]:
            prev = name_of(dict(k, hard=k["hard"] - 1))
            self.train(prev)
            self.evaluate(prev, self.locked_rule())
            mined, n_new = self.mine(prev)
            self.w.json(os.path.join(self.dir(name), "mined.json"),
                        [{"pool": p, "key": key, "frames": v} for (p, key), v in sorted(mined.items())])
        self.report(f"training {name} ({describe(k)})")
        log(f"=== {name}: {describe(k)}; {len(self.seeds)} runs x {self.a.minutes} min")
        t0 = time.time()
        bank, sets = build_training_set(self.cfg, self.pools, self.a.cache, k, mined, self.seeds)
        log(f"training data read in {fmt_dur(time.time() - t0)}: {bank.shape[0]} embeddings in memory")
        infos = []
        with self.w.open(os.path.join(self.dir(name), "training_log.txt")) as logf:
            for s, (win, y, info) in zip(self.seeds, sets):
                logf.write(json.dumps(info) + "\n")
                path = os.path.join(self.dir(name), f"model_seed{s}.pth")
                if not os.path.exists(path) or self.a.rerun:
                    log(f"run {s}: {info}")
                    m = ex.train_model(bank, win, y, k["ctx"], self.a.minutes, s, logf, self.device)
                    torch.save(m.state_dict(), self.w.check(path + ".tmp"))
                    os.replace(path + ".tmp", path)  # a half-written model never looks finished
                infos.append(info)
        del bank, sets
        self.w.json(os.path.join(self.dir(name), "training.json"),
                    {"name": name, "knobs": k, "description": describe(k), "runs": infos,
                     "newly_mined_frames": n_new, "minutes_per_run": self.a.minutes,
                     "train_minutes": round((time.time() - t0) / 60, 1),
                     "base_configuration": self.a.base_config})
        for f in os.listdir(self.dir(name)):
            if f.startswith("results"):
                os.remove(self.w.check(os.path.join(self.dir(name), f)))
        self.trained += 1
        self.train_seconds.append(time.time() - t0)

    def mine(self, prev):
        """False alarms of `prev` (its first run, the locked rule, its cut-off for the target
        recall) on the training days, plus everything `prev` had mined itself."""
        r = self.result(prev, self.locked_rule())
        mined = {}
        for row in self.json(prev, "mined.json") or []:
            mined[(row["pool"], row["key"])] = list(row["frames"])
        rule = r["rule"]
        cut = (r["summary"]["per_seed"][0].get(f"at_{int(round(self.a.target_recall * 100))}") or {}).get("threshold", 0.5)
        # very confident false alarms may be unmarked rumbles, so they are not mined; but keep
        # the band above the cut-off open when the cut-off itself is very high
        skip_above = max(ex.HARD_SKIP_ABOVE, (cut + 1) / 2)
        ctx = knobs_of(prev)["ctx"]
        model = self.models(prev)[0]
        n_new = 0
        for p, keys in self.cfg["train"].items():
            if p == "noise":
                continue
            for key in keys:
                day = Day(self.a.cache, p, key)
                if day.kind != "day":
                    raise RuntimeError(f"{p} {key} has no whole-day embeddings: run `python tools\\improve.py embed`")
                prob = ex.day_probs(model, day.whole(), ctx, self.device)
                labels = self.pools[p].labels.get(key, [])
                got = set(mined.get((p, key), []))
                for a0, b0, sc in zip(*alarms(prob, rule)):
                    if sc < cut or sc >= skip_above:
                        continue
                    t0, t1 = a0 * FRAME_SECONDS - ex.HARD_PAD_S, b0 * FRAME_SECONDS + ex.HARD_PAD_S
                    if any(l.start < t1 and l.end > t0 for l in labels):
                        continue
                    new = set(range(int(a0), int(b0) + 1)) - got
                    n_new += len(new)
                    got |= new
                if got:
                    mined[(p, key)] = sorted(got)
        log(f"mined {n_new} new false-alarm frames from {prev} (scores {cut:.3f} to {skip_above:.4f})"
            + ("; WARNING nothing new to learn from, this round repeats the previous one" if not n_new else ""))
        return mined, n_new

    # ---- scoring ------------------------------------------------------------
    def evaluate(self, name, rule):
        """Validation results of a trained experiment under `rule` (cached per rule)."""
        r = self.result(name, rule)
        if r and not self.a.rerun_scoring:
            if self.json(name, "results.json") != r:
                self.w.json(os.path.join(self.dir(name), "results.json"), r)
            return r
        self.train(name)
        k = knobs_of(name)
        files, truth = self.validation()
        models = self.models(name)
        probs = self.probs(models, k["ctx"], files)
        sweeps = [sweep(truth, pr, rule) for pr in probs]
        summ = summarize(sweeps, self.a.target_recall)
        tr = self.json(name, "training.json")
        r = {"name": name, "knobs": k, "description": describe(k), "rule": rule,
             "summary": summ, "speed": self.speed(models[0], k["ctx"], files, rule),
             "training": tr["runs"][0] if tr else None, "train_minutes": tr and tr.get("train_minutes"),
             "validation": self.cfg["validation"]["name"], "validation_days": len(files),
             "validation_rumbles": truth.n, "finished": time.strftime("%Y-%m-%d %H:%M")}
        r["settings"] = {"target_recall": self.a.target_recall, "seeds": self.seeds, "minutes": self.a.minutes}
        self.w.json(os.path.join(self.dir(name), f"results_{rule_key(rule)}.json"), r)
        self.w.json(os.path.join(self.dir(name), "results.json"), r)
        try:
            write_curve(self.w, self.dir(name), name, sweeps)
        except OSError as e:  # e.g. the file is open in Excel: the results are saved anyway
            log(f"WARNING could not write the curve of {name}: {e}")
        fp = summ["fp_per_hour"]
        log(f"{name}: " + (f"{fp[0]:.3f} false alarms/h at {self.a.target_recall:.0%} recall "
                           f"(runs {fp[1]:.3f}-{fp[2]:.3f})" if fp else
                           f"never reaches {self.a.target_recall:.0%} recall (max {summ['max_recall'][0]:.3f})"))
        try:
            cmd_compare(self)
        except OSError as e:  # e.g. improve.csv is open in Excel; rewritten after the next experiment
            log(f"WARNING could not rewrite the results table: {e}")
        return r

    def result(self, name, rule):
        """Saved validation results of `name` under `rule` with this run's settings, or None."""
        r = self.json(name, f"results_{rule_key(rule)}.json")
        st = (r or {}).get("settings") or {}
        if r and r["rule"] == rule and st.get("target_recall") == self.a.target_recall \
                and st.get("seeds") == self.seeds and st.get("minutes") == self.a.minutes:
            return r
        return None

    def speed(self, model, ctx, files, rule):
        """Seconds per 24 h of audio for the classifier + alarm rule, on this GPU and CPU."""
        import copy
        import torch
        p, k, h = max(files, key=lambda f: f[2])
        emb = Day(self.a.cache, p, k).whole()
        scale = 24 / max(h, 1e-6)
        out = {"parameters": sum(x.numel() for x in model.parameters()), "window_seconds": window_seconds(ctx)}
        for dev in (["cuda"] if self.device == "cuda" else []) + ["cpu"]:
            m = copy.deepcopy(model).to(dev).eval()
            ex.day_probs(m, emb[:2000], ctx, dev)  # warm up
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                alarms(ex.day_probs(m, emb, ctx, dev), rule)
                if dev == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            out[f"classifier_seconds_per_24h_{dev}"] = round(statistics.median(times) * scale, 3)
        tp = os.path.join(self.a.out, "_timing.json")
        if os.path.exists(tp):
            m = json.load(open(tp)).get("embed_s_per_audio_hour")
            if m and m.get("units"):
                out["aves_seconds_per_24h"] = round(m["seconds"] / m["units"] * 24, 1)
        return out

    # ---- choices ------------------------------------------------------------
    def choices_path(self):
        return os.path.join(self.root, "choices.json")

    def choices(self):
        p = self.choices_path()
        return json.load(open(p)) if os.path.exists(p) else {}

    def save_choice(self, key, value):
        c = self.choices()
        c[key] = value
        self.w.json(self.choices_path(), c)

    def locked_rule(self):
        return self.choices().get("rules", {}).get("rule", FRUITPUNCH_RULE)

    def pick(self, stage, names, rule, why):
        """Fewest false alarms at the target recall; anything within one run-to-run spread
        (standard deviation over runs) of the best counts as a tie, and the tie goes to the
        simplest / fastest option."""
        rs = []
        for n in names:
            try:
                rs.append(self.evaluate(n, rule))
            except Exception:  # e.g. out of memory: skip this option, keep the run going
                import traceback
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                msg = f"{n}: FAILED, left out of {stage}:\n{traceback.format_exc()}"
                log(msg)
                with self.w.open(os.path.join(self.root, "failed.txt"), "a") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M ") + msg + "\n")
        if not rs:
            sys.exit(f"every experiment of {stage} failed; see failed.txt")
        ranked = sorted(rs, key=lambda r: rank(r["summary"]))
        top = ranked[0]
        win = top
        if rank(top["summary"])[0] == 0:
            sd = top["summary"]["fp_per_hour"][3]
            close = [r for r in ranked if rank(r["summary"])[0] == 0 and r["summary"]["fp_per_hour"][0] <= top["summary"]["fp_per_hour"][0] + sd]
            win = min(close, key=lambda r: cost(r["knobs"]))
        msg = f"{why}: picked {win['name']} ({win['description']})"
        if win is not top:
            msg += f"; {top['name']} scored lower but within the run-to-run spread, so the simpler one is kept"
        log(msg)
        self.save_choice(stage, {"winner": win["name"], "ranked": [r["name"] for r in ranked], "why": msg,
                                 "rule": rule, "time": time.strftime("%Y-%m-%d %H:%M")})
        return win["name"], [r["name"] for r in ranked]

    # ---- rule tuning --------------------------------------------------------
    def tune_rules(self, name, stage, why):
        """Every combination of RULE_OPTIONS on the saved validation probabilities of
        `name`'s runs; the one with the fewest false alarms at the target recall wins."""
        c = self.choices().get(stage)
        if c and c.get("experiment") == name and c.get("target_recall") == self.a.target_recall \
                and c.get("seeds") == self.seeds and not self.a.rerun_scoring:
            return c["rule"]
        self.train(name)
        self.report(f"{why}: trying {n_rules()} alarm rules on {name}")
        k = knobs_of(name)
        files, truth = self.validation()
        probs = self.probs(self.models(name), k["ctx"], files)
        rows, t0 = [], time.time()
        caches = [{} for _ in probs]
        for i, rule in enumerate(all_rules()):
            summ = summarize([sweep(truth, pr, rule, c) for pr, c in zip(probs, caches)], self.a.target_recall)
            rows.append((rank(summ), rule, summ))
            if (i + 1) % 200 == 0:
                log(f"{why}: {i + 1}/{n_rules()} rules ({fmt_dur(time.time() - t0)})")
        rows.sort(key=lambda r: r[0])
        best = rows[0][1]
        fp_rule = next(r for r in rows if r[1] == FRUITPUNCH_RULE)
        msg = (f"{why}: picked {rule_text(best)} on {name}; " + fmt_summary(rows[0][2], self.a.target_recall)
               + f" (FruitPunch's rule: {fmt_summary(fp_rule[2], self.a.target_recall)})")
        log(msg)
        top = [{"rule": r, "fp_per_hour": s["fp_per_hour"], "max_recall": s["max_recall"], "best_f1": s["best_f1"],
                "runs_reaching_target": s["runs_reaching_target"]} for _, r, s in rows[:30]]
        self.w.json(os.path.join(self.root, f"rules_{stage}.json"), {"experiment": name, "top": top,
                    "fruitpunch_rule": {"fp_per_hour": fp_rule[2]["fp_per_hour"], "max_recall": fp_rule[2]["max_recall"]}})
        self.save_choice(stage, {"experiment": name, "rule": best, "why": msg, "target_recall": self.a.target_recall,
                                 "seeds": self.seeds, "time": time.strftime("%Y-%m-%d %H:%M")})
        return best

    # ---- progress -------------------------------------------------------------
    def report(self, stage):
        self.stage = stage
        left = planned_left(self)
        per = statistics.mean(self.train_seconds) if self.train_seconds else (
            self.a.seeds * (self.a.minutes * 60 + 60) + 300)
        total = left * per
        now = time.time()
        t = [f"Progress, updated {time.strftime('%Y-%m-%d %H:%M')}", f"Now: {stage}",
             f"Running for {fmt_dur(now - self.t_start)}; {self.trained} experiments trained this run.",
             f"About {left} experiments left at about {fmt_dur(per)} each: {fmt_dur(total)}, "
             f"finishing around {time.strftime('%a %H:%M', time.localtime(now + total))}.", "",
             "The count left is an estimate: later stages depend on earlier winners. Results so far: improve.md"]
        with self.w.open(os.path.join(self.root, "progress.txt")) as f:
            f.write("\n".join(t) + "\n")
        log(f"PROGRESS: {stage}; about {fmt_dur(total)} left (finish ~{time.strftime('%a %H:%M', time.localtime(now + total))})")


def all_rules():
    keys = list(RULE_OPTIONS)
    rules = [dict(zip(keys, v)) for v in itertools.product(*(RULE_OPTIONS[k] for k in keys))]
    return [FRUITPUNCH_RULE] + [r for r in rules if r != FRUITPUNCH_RULE]


def n_rules():
    return len(all_rules())


def fmt_summary(s, target):
    fp = s["fp_per_hour"]
    if fp and s["runs_reaching_target"] == len(s["per_seed"]):
        return f"{fp[0]:.3f} false alarms/h at {target:.0%} recall"
    return f"{target:.0%} recall not reached by every run (max recall {s['max_recall'][0]:.3f})"


def planned_left(lab):
    """Rough count of experiments still to train, assuming the current winners hold."""
    state = current_state(lab)
    names = set()
    for _, knob, _, opts in STAGES:
        names |= {name_of(dict(state, **{knob: v})) for v in opts}
    names |= {name_of(dict(state, hard=h)) for h in HARD_ROUNDS}
    left = sum(1 for n in names if not lab.trained_ok(n))
    if "combos" not in lab.choices():
        left += 8  # about half of the 16 combinations are new
    return left


def current_state(lab):
    c = lab.choices()
    for key in ("mistakes", "combos", "3", "2c", "2b", "2a"):
        if key in c:
            return knobs_of(c[key]["winner"])
    return dict(BASE)


def write_curve(w, d, name, sweeps):
    """Precision-recall curve of every run: a CSV and a small SVG chart."""
    import numpy as np
    keys = ["threshold", "recall", "precision", "fp_per_hour", "f1", "detections", "rumbles_found"]
    with w.open(os.path.join(d, "pr_curve.csv")) as f:
        wr = csv.writer(f)
        wr.writerow(["run"] + keys)
        for i, sw in enumerate(sweeps):
            for j in range(len(sw["threshold"])):
                wr.writerow([i] + [round(float(sw[k][j]), 4) for k in keys])
    W, H, L, B = 520, 360, 50, 40
    colors = ["#2f6b4f", "#b5711b", "#3b6ea8", "#a8433a", "#7a5aa8", "#5a6760"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W / 2}" y="16" text-anchor="middle">{name}: precision vs recall (validation, each run)</text>']
    for t in range(0, 11, 2):
        x, y = L + t / 10 * (W - L - 20), H - B - t / 10 * (H - B - 30)
        parts += [f'<line x1="{x}" y1="30" x2="{x}" y2="{H - B}" stroke="#ddd"/>',
                  f'<text x="{x}" y="{H - B + 16}" text-anchor="middle">{t / 10:.1f}</text>',
                  f'<line x1="{L}" y1="{y}" x2="{W - 20}" y2="{y}" stroke="#ddd"/>',
                  f'<text x="{L - 6}" y="{y + 4}" text-anchor="end">{t / 10:.1f}</text>']
    parts += [f'<text x="{(W + L) / 2}" y="{H - 6}" text-anchor="middle">recall</text>',
              f'<text x="14" y="{H / 2}" transform="rotate(-90 14 {H / 2})" text-anchor="middle">precision</text>']
    for i, sw in enumerate(sweeps):
        ok = ~np.isnan(sw["precision"]) & ~np.isnan(sw["recall"])
        pts = " ".join(f"{L + r * (W - L - 20):.1f},{H - B - p * (H - B - 30):.1f}"
                       for r, p in zip(sw["recall"][ok], sw["precision"][ok]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i % len(colors)]}" stroke-width="1.5"/>')
    parts.append("</svg>")
    with w.open(os.path.join(d, "pr_curve.svg")) as f:
        f.write("\n".join(parts) + "\n")


################################################################################
# the stages
################################################################################

def cmd_all(lab):
    base = name_of(BASE)
    lab.report("stage 0: baseline")
    lab.train(base)
    lab.evaluate(base, FRUITPUNCH_RULE)
    rule = lab.tune_rules(base, "rules", "stage 1 (alarm rules)")
    state = dict(BASE)
    best_two = {}
    for stage, knob, label, opts in STAGES:
        lab.report(f"stage {stage} ({label})")
        names = [name_of(dict(state, **{knob: v})) for v in opts]
        win, ranked = lab.pick(stage, names, rule, f"stage {stage} ({label})")
        state = knobs_of(win)
        others = [knobs_of(n)[knob] for n in ranked if n != win]
        best_two[knob] = sorted({state[knob]} | set(others[:1]))  # the winner and the best other option
    lab.report("stage 4 (combinations of the best two of each stage)")
    combos = [dict(zip(best_two, v), hard=0) for v in itertools.product(*best_two.values())]
    win, _ = lab.pick("combos", [name_of(k) for k in combos], rule, "stage 4 (combinations)")
    state = knobs_of(win)
    lab.report("stage 5 (learning from its own false alarms)")
    names = [name_of(dict(state, hard=h)) for h in [0] + HARD_ROUNDS]
    win, _ = lab.pick("mistakes", names, rule, "stage 5 (hard negatives)")
    final_rule = lab.tune_rules(win, "rules_final", "stage 6 (alarm rules for the final model)")
    lab.evaluate(win, final_rule)
    lab.save_choice("final", {"winner": win, "rule": final_rule, "time": time.strftime("%Y-%m-%d %H:%M")})
    cmd_compare(lab)
    lab.report(f"finished: {win}")
    log(f"final model: {win} ({describe(knobs_of(win))}) with rule {rule_text(final_rule)}. "
        f"See {os.path.join(lab.root, 'improve.md')}; then run `python tools\\improve.py final`.")


def _f(v, d=3):
    return "-" if v is None or v != v else f"{v:.{d}f}"


def _stat(s, d=3):
    return "-" if not s else f"{s[0]:.{d}f} ({s[1]:.{d}f}-{s[2]:.{d}f})"


def cmd_compare(lab):
    rows = []
    if os.path.isdir(lab.root):
        for n in sorted(os.listdir(lab.root)):
            if not os.path.isdir(lab.dir(n)):
                continue
            fin = lab.choices().get("final") or {}
            r = (fin.get("winner") == n and lab.result(n, fin["rule"])) or lab.result(n, lab.locked_rule()) \
                or lab.json(n, "results.json")
            if r:
                rows.append(r)
    if not rows:
        return
    c = lab.choices()
    stage_of = {}
    for st, v in c.items():
        for n in v.get("ranked", []):
            stage_of.setdefault(n, st)
    target = int(round(lab.a.target_recall * 100))
    rows.sort(key=lambda r: rank(r["summary"]))
    t = ["# Improving the rumble detector", "",
         f"Base data: {lab.a.base_config}. Every number is on the validation days "
         f"({rows[0]['validation']}, {rows[0]['validation_days']} days, {rows[0]['validation_rumbles']} rumbles) "
         f"and averaged over {lab.a.seeds} training runs, with the range over runs in brackets. "
         f"The tests are untouched until `final`.", "",
         f"**The yardstick is false alarms per hour at the cut-off that finds {target}% of the rumbles** "
         "(lower is better). A difference smaller than the range in brackets can be luck.", ""]
    if "final" in c:
        t += [f"**Final model: {c['final']['winner']}** ({describe(knobs_of(c['final']['winner']))}), "
              f"alarm rule: {rule_text(c['final']['rule'])}.", ""]
    else:
        t += [f"Best so far: **{rows[0]['name']}** ({rows[0]['description']}).", ""]
    t += [f"| Experiment | Stage | Window (s) | Noise x | Random/day | Neg:1 | Hard | FA/h at {target}% | "
          "FA/h at 85% | FA/h at 90% | Max recall | Best F1 | Classifier s per 24 h (GPU / CPU) | Negatives per rumble (actual) | "
          "Noise+random share | Rule |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        k, s, sp = r["knobs"], r["summary"], r.get("speed") or {}
        tr = r.get("training") or {}
        t.append(f"| {r['name']} | {stage_of.get(r['name'], '')} | {window_seconds(k['ctx'])} | {k['noise']} | {k['rand']} | "
                 f"{k['neg']} | {k['hard']} | **{_stat(s['fp_per_hour'])}** | {_stat(s.get('fp_per_hour_at_85'))} | "
                 f"{_stat(s.get('fp_per_hour_at_90'))} | {_stat(s['max_recall'])} | {_stat(s['best_f1'])} | "
                 f"{_f(sp.get('classifier_seconds_per_24h_cuda'), 2)} / {_f(sp.get('classifier_seconds_per_24h_cpu'), 2)} | "
                 f"{_f(tr.get('negatives_per_rumble'), 2)} | {_f(tr.get('noise_and_random_share_of_negatives'), 2)} | "
                 f"{rule_key(r['rule'])} |")
    aves = next((r["speed"].get("aves_seconds_per_24h") for r in rows if r.get("speed", {}).get("aves_seconds_per_24h")), None)
    if aves:
        t += ["", f"AVES itself takes about {aves:.0f} s per 24 h of audio on this PC, for every model."]
    t += ["", "## Choices made", ""]
    for st in ("rules", "2a", "2b", "2c", "3", "combos", "mistakes", "rules_final"):
        if st in c:
            t.append(f"- {c[st]['time']} {c[st]['why']}")
    with lab.w.open(os.path.join(lab.root, "improve.md")) as f:
        f.write("\n".join(t) + "\n")
    keys = ["name", "stage", "window_s", "noise", "rand", "neg", "hard", "fa_h_mean", "fa_h_min", "fa_h_max",
            "fa_h_85", "fa_h_90", "max_recall", "best_f1", "gpu_s_24h", "cpu_s_24h", "parameters", "rule"]
    with lab.w.open(os.path.join(lab.root, "improve.csv")) as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for r in rows:
            k, s, sp = r["knobs"], r["summary"], r.get("speed") or {}
            fp = s["fp_per_hour"] or (None, None, None)
            wr.writerow([r["name"], stage_of.get(r["name"], ""), window_seconds(k["ctx"]), k["noise"], k["rand"],
                         k["neg"], k["hard"], fp[0], fp[1], fp[2],
                         (s.get("fp_per_hour_at_85") or [None])[0], (s.get("fp_per_hour_at_90") or [None])[0],
                         (s["max_recall"] or [None])[0], (s["best_f1"] or [None])[0],
                         sp.get("classifier_seconds_per_24h_cuda"), sp.get("classifier_seconds_per_24h_cpu"),
                         sp.get("parameters"), rule_key(r["rule"])])


def cmd_final(lab):
    """Each stage's winner on the test sets, once, at the cut-off each run picked on validation."""
    c = lab.choices()
    if "final" not in c:
        sys.exit("the stages have not finished yet: run `python tools\\improve.py` first")
    base = name_of(BASE)
    steps = [("baseline, FruitPunch's alarm rule", base, FRUITPUNCH_RULE),
             ("1 alarm rules", base, c["rules"]["rule"])]
    for st, label in (("2a", "2a noise clips"), ("2b", "2b random frames"), ("2c", "2c negatives"),
                      ("3", "3 window"), ("combos", "4 combinations"), ("mistakes", "5 own false alarms")):
        steps.append((label, c[st]["winner"], c["rules"]["rule"]))
    steps.append(("6 final alarm rules", c["final"]["winner"], c["final"]["rule"]))
    key = f"at_{int(round(lab.a.target_recall * 100))}"
    tests = [s for s in lab.cfg["tests"] if any(s["files"].values())]
    t = ["# What each step added (test sets, scored once)", "",
         f"Each run uses the cut-off that found {lab.a.target_recall:.0%} of the rumbles on the validation days. "
         f"Means over {lab.a.seeds} runs.", "",
         "| Step | Experiment | Validation FA/h | " + " | ".join(f"{s['name']} recall / precision / FA/h" for s in tests) + " |",
         "|---|---|---|" + "---|" * len(tests)]
    out = {"steps": []}
    cache = {}  # (experiment, set) -> (truth, probs per run); the baseline appears twice
    notes = []
    for label, name, rule in steps:
        r = lab.evaluate(name, rule)
        cuts = [(ps.get(key) or {}).get("threshold") for ps in r["summary"]["per_seed"]]
        if any(c is None for c in cuts):
            notes.append(f"{label}: {sum(c is None for c in cuts)} run(s) never reached "
                         f"{lab.a.target_recall:.0%} recall on validation and are left out.")
        row = {"step": label, "experiment": name, "rule": rule, "cut_offs": cuts, "sets": {}}
        cells = []
        for s in tests:
            if (name, s["key"]) not in cache:
                cache.clear()  # keep only one experiment's test probabilities in memory
                models = lab.models(name)
                keys = [(p, k) for p, ks in s["files"].items() for k in ks]
                probs = lab.probs(models, knobs_of(name)["ctx"], [(p, k, 0) for p, k in keys])
                files = [(p, k, len(probs[0][(p, k)]) * FRAME_SECONDS / 3600) for p, k in keys]
                cache[(name, s["key"])] = (Truth(lab.pools, files), probs, files)
            truth, probs, files = cache[(name, s["key"])]
            per = []
            for pr, cut in zip(probs, cuts):
                if cut is None:
                    continue
                sw = sweep(truth, pr, rule)
                i = int(abs(sw["threshold"] - cut).argmin())
                per.append({m: float(sw[m][i]) for m in ("recall", "precision", "fp_per_hour")})
            row["sets"][s["key"]] = per
            mean = {}
            for m in ("recall", "precision", "fp_per_hour"):
                v = [x[m] for x in per if x[m] == x[m]]
                mean[m] = statistics.mean(v) if v else float("nan")
            cells.append(f"{_f(mean['recall'])} / {_f(mean['precision'])} / {_f(mean['fp_per_hour'], 2)}")
            if label.startswith("6"):
                write_final_detections(lab, s, name, rule, r, probs, files, truth)
        out["steps"].append(row)
        t.append(f"| {label} | {name} | {_stat(r['summary']['fp_per_hour'])} | " + " | ".join(cells) + " |")
    if notes:
        t += [""] + [f"- {n}" for n in notes]
    t += ["", "The final model's detections (its best run on validation) are Raven tables in "
              "`final_detections\\`, with each test set's precision-recall curve."]
    lab.w.json(os.path.join(lab.root, "final.json"), out)
    with lab.w.open(os.path.join(lab.root, "final.md")) as f:
        f.write("\n".join(t) + "\n")
    print("\n".join(t))
    log(f"wrote {os.path.join(lab.root, 'final.md')}")


def write_final_detections(lab, s, name, rule, r, probs, files, truth):
    """Raven table of the final model's best run on one test set, at that run's cut-off,
    plus the set's precision-recall curve for every run."""
    key = f"at_{int(round(lab.a.target_recall * 100))}"
    per = r["summary"]["per_seed"]
    ok = [i for i, ps in enumerate(per) if ps.get(key)]
    if not ok:
        return
    best = min(ok, key=lambda i: per[i][key]["fp_per_hour"])
    cut = per[best][key]["threshold"]
    d = os.path.join(lab.root, "final_detections")
    with lab.w.open(os.path.join(d, f"{s['key']}_detections.txt")) as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)",
                     "High Freq (Hz)", "Begin Path", "File Offset (s)", "Begin File", "Score"])
        sel, elapsed = 0, 0.0
        for p, k, h in files:
            path = lab.pools[p].sounds[k]
            for a0, b0, sc in zip(*alarms(probs[best][(p, k)], rule)):
                if sc < cut:
                    continue
                sel += 1
                bt, et = a0 * FRAME_SECONDS, b0 * FRAME_SECONDS
                wr.writerow([sel, "Spectrogram 1", 1, f"{elapsed + bt:.3f}", f"{elapsed + et:.3f}", 5, 250,
                             path, f"{bt:.3f}", os.path.basename(path), f"{sc:.4f}"])
            elapsed += h * 3600
    write_curve(lab.w, os.path.join(d, s["key"]), f"{name} on {s['name']}", [sweep(truth, pr, rule) for pr in probs])
    lab.w.json(os.path.join(d, "final_model.json"), {"experiment": name, "run": lab.seeds[best],
               "model": os.path.join(lab.dir(name), f"model_seed{lab.seeds[best]}.pth"), "rule": rule, "cut_off": cut})


def keep_awake():
    """Ask Windows not to sleep while this script runs (what video players do; the screen
    may still turn off and lock). Ends by itself when the script exits."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000000 | 0x00000001))  # CONTINUOUS | SYSTEM_REQUIRED
        except Exception:
            pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", nargs="?", default="all", choices=["embed", "all", "compare", "final"])
    p.add_argument("--source", default=dc.DEFAULT_SOURCE, help=f"the Datasets copy (default {dc.DEFAULT_SOURCE})")
    p.add_argument("--out", help="the Configurations folder of dataset_configs.py (default <source>\\Configurations)")
    p.add_argument("--cache", help="its embedding cache (default <out>\\_embedding_cache)")
    p.add_argument("--base-config", default=BASE_CONFIG, help=f"data configuration to start from (default {BASE_CONFIG})")
    p.add_argument("--seeds", type=int, default=SEEDS, help="training runs per experiment")
    p.add_argument("--seed", type=int, default=0, help="first seed")
    p.add_argument("--minutes", type=float, default=MINUTES, help="training time per run")
    p.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    p.add_argument("--rerun", action="store_true", help="retrain experiments that already have models")
    p.add_argument("--rerun-scoring", action="store_true", help="rescore experiments (no retraining)")
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.out = a.out or os.path.join(a.source, "Configurations")
    a.cache = a.cache or os.path.join(a.out, "_embedding_cache")
    keep_awake()
    root = dc.find_dataset_root(a.source)
    imp_root = os.path.join(a.out, "_improve")
    w = dc.Writer([imp_root, a.cache], [dc.ORIGINAL_DATA, os.path.join(root, "Training"), os.path.join(root, "Testing")])
    w.makedirs(imp_root)
    stdout = sys.stdout
    sys.stdout = dc.Tee(w.check(os.path.join(imp_root, "run_log.txt")))
    try:
        log(f"improving {a.base_config}; writing to {imp_root}")
        pools = dc.load_pools(root)
        lab = Lab(a, pools, w)
        if a.step in ("embed", "all"):
            ex.cmd_embed(a, pools, w, lab.cfg)
        if a.step == "all":
            cmd_all(lab)
        elif a.step == "compare":
            cmd_compare(lab)
        elif a.step == "final":
            cmd_final(lab)
    finally:
        sys.stdout = stdout


if __name__ == "__main__":
    main()
