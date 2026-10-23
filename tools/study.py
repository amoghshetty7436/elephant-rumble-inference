#!/usr/bin/env python
"""
The full rerun (walkthrough/17): find the best rumble detector settings, with every
model trained the same fair way, in one unattended run that ends with the test scores.

This is a new script. dataset_configs.py, experiments.py, improve.py and
improve_converged.py are used (imported) but never changed, and nothing they wrote
is touched: everything goes to <out>\\_study (and augmented fingerprints to
<cache>\\_augmented).

How every model is trained (the same for all, see reports/fair-comparison-training-length.md)
  - until it stops improving: checked every CHECK_EVERY training examples on held-out
    *days* (about 1 in 12 training days, never trained on, the same days for every
    experiment of a setup); stops after PATIENCE checks without improvement, at most
    MAX_CHECKS; the best check is kept. Patience is counted in examples, so a big
    training set does not get more patience than a small one.
  - the best check is chosen by average precision on those held-out days.
  - constant Adam, batch 64, balanced class weights, SEEDS runs per experiment.

How every model is scored (validation = RumbleComplex + RumbleGeneral, 58 days)
  - false alarms per hour at the cut-off that finds 80% and 85% of the rumbles; the
    score is their mean (lower is better). 90% is shown too, for information.
  - every run must reach 85% recall, or the experiment ranks below all that do.
  - options within one run-to-run standard deviation of the best are a tie; ties go to
    the one that can reach the most rumbles (max recall), then fewer false alarms at
    90%, then the simpler option. Alarm rules are picked the same way.

Stages (`all` runs them in order, resumes after a crash, and stops starting new work
when --finish-by gets close, keeping time to finish the final steps)

  1   first round   setups A, B, C, D, E, E2, G (which recordings to train on), each with
                    its own tuned alarm rule (reduced grid)
  2   rules         all 1,960 alarm rules on the round-1 winner; every rule's score is saved
  3   noise         noise clips used 0, 1, 3, 10 times
  4   random        random forest frames per forest day: 0 ... 5,000
  5   gap           frames from gaps between rumbles, per rumble frame: 0 (no gap frames) ... 2
  6   window        listening window 0.64 s ... 16 s
  7   augmentation  noise mixed into the audio before AVES: which sounds (rumbles, forest
                    background, both, plus a no-noise control), how loud, how many copies
  8   combinations  best two options of stages 4-7 combined
  9   mistakes      0-5 rounds of learning from its own false alarms (cleaned mining)
  10  final rules   all 1,960 rules again on the final model
  11  tests         Test 1, Test 2 and the vehicle days, scored once for the baseline,
                    every stage winner and the final model; Raven tables of the final model

On the lab PC (one line; rerunning it resumes):

  python tools\\study.py

Results: <out>\\_study\\study.md (all experiments), final.md (tests), progress.txt.
"""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import shutil
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset_configs as dc  # noqa: E402
import experiments as ex  # noqa: E402
import improve as im  # noqa: E402
from dataset_configs import log, fmt_dur, Day, FRAME_SECONDS  # noqa: E402

ROOT_NAME = "_study"
SEEDS = 5
SCORE_RECALLS = (0.80, 0.85)   # the score: mean false alarms/h at these recalls
SHOWN_RECALLS = (0.80, 0.85, 0.90)

# training until converged
CHECK_EVERY = 300_000          # training examples between checks on the held-out days
PATIENCE = 15                  # checks without improvement before stopping
MAX_CHECKS = 100               # at most this many checks (30 M examples)
HELDOUT_ONE_IN = 12            # about 1 in 12 training days held out for stopping
HELDOUT_RANDOM_PER_DAY = 2000  # held-out "not rumble" frames per held-out day (besides gaps)
BATCH = 64

# the first round: setup ids (E = clearing + noise, E2 = clearing + forest + noise)
ROUND1 = ["A", "B", "C", "D", "E", "E2", "F", "G", "H"]

# the options of each stage; the first value is where the stage starts from
NOISE_REPEATS = [1, 0, 3, 10]
RANDOM_PER_DAY = [200, 0, 1000, 2000, 3000, 5000]
GAP_PER_RUMBLE = [0, 0.25, 0.5, 1, 2]     # plus the round-1 amount
CONTEXTS = list(range(10))               # (2c+1) x 0.64 s = 0.64, 1.92, 3.2 ... 12.16 s
HARD_ROUNDS = [1, 2, 3, 4, 5]
AUG_TYPES = ["clean", "rumble", "background", "both"]  # "clean": rumble copies, no noise (control)
AUG_LEVELS = [10, 0, -10, "mix"]          # rumble (or background) this many dB above the noise
AUG_COPIES = [1, 2, 4]
COMBO_TOP = 2                             # best this many options of stages 4-7 are combined

# rules: the reduced grid for the first round, the full grid of improve.py otherwise
FRUITPUNCH_RULE = im.FRUITPUNCH_RULE
ROUND1_RULES = dict(on=[0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2], min_frames=[2, 3, 4, 6],
                    smooth=[1, 3, 5], gap=[0, 2], score=["mean"])

# cleaned hard-negative mining
MINE_PAD_S = 60.0              # never mine within this many seconds of a labelled rumble
MINE_FRAMES_PER_ALARM = 3      # the highest-scoring frames of each false alarm
MINE_MAX_PER_DAY = 600         # at most this many mined frames per day (all rounds together)
MINE_SKIP_ABOVE = 0.99         # very confident "false alarms" may be unlabelled rumbles

# augmentation
AUG_CTX_MAX = 9                # augmented frames support windows up to 2*9+1 frames (12.16 s)
AUG_AUDIO_CONTEXT_S = 30.0     # extra audio AVES hears on each side of an augmented stretch
AUG_MAX_SPAN = 1800            # frames per augmented stretch (about 19 min)
AUG_CHUNK = 47                 # background chunk length in frames (30 s)
AUG_BAND = (10.0, 100.0)       # loudness is measured in the rumble band (Hz)
AUG_SKIP_NOISE = {"silence_8kHz"}
AUG_DISK_MARGIN_GB = 20

MAX_RAM_SHARE = 0.55
FINAL_RESERVE_S = 3 * 3600     # time kept back for the final rules and the tests
FIRST_GUESS_S = 50 * 60        # time per experiment until one has been measured


################################################################################
# experiments: a spec is a dict; its name is its folder
################################################################################

def wjson(w, path, obj, tries=6):
    """w.json with retries: on Windows an antivirus scan or an open viewer can briefly lock a file."""
    for i in range(tries):
        try:
            return w.json(path, obj)
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def safe(what, fn, *args, default=None, lab=None, **kw):
    """Run one step; on any error log it to failed.txt and carry on with `default`."""
    try:
        return fn(*args, **kw)
    except Exception:
        record_failure(lab, what)
        return default


def record_failure(lab, what):
    """Log the exception being handled, to the screen and to failed.txt."""
    msg = f"{what}: FAILED, carrying on:\n{traceback.format_exc()}"
    log(msg)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if lab is not None:
        try:
            with lab.w.open(os.path.join(lab.root, "failed.txt"), "a") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M ") + msg + "\n")
        except OSError:
            pass


def gap_text(g):
    return "fill" if g == "fill" else f"{g:g}"


def aug_text(a):
    if not a:
        return "none"
    lv = a["level"]
    lv = "mix" if lv == "mix" else (f"p{lv}" if lv > 0 else f"m{-lv}" if lv < 0 else "0")
    return f"{a['type']}{'' if a['type'] == 'clean' else lv}x{a['copies']}"


def name_of(s):
    return (f"{s['setup']}_ctx{s['ctx']}_noise{s['noise']}_rand{s['rand']}_gap{gap_text(s['gap'])}"
            f"_hard{s['hard']}_aug{aug_text(s['aug'])}")


def window_seconds(ctx):
    return round((2 * ctx + 1) * FRAME_SECONDS, 2)


def describe(s):
    a = s["aug"]
    aug = "no augmentation" if not a else (
        f"augmentation: {a['copies']} clean cop{'y' if a['copies'] == 1 else 'ies'} of the rumbles (control)"
        if a["type"] == "clean" else
        f"augmentation: {a['type']} + noise at {a['level']}{'' if a['level'] == 'mix' else ' dB'}, x{a['copies']}")
    gap = "gap frames to make 1:1" if s["gap"] == "fill" else f"{gap_text(s['gap'])} gap frames per rumble frame"
    return (f"setup {s['setup']}, {window_seconds(s['ctx'])} s window, noise x{s['noise']}, "
            f"{s['rand']} random frames/day, {gap}, {s['hard']} mistake rounds, {aug}")


def cost(s, info=None):
    """Simpler first, for ties: fewer mistake rounds, shorter window, less augmentation,
    fewer training examples."""
    a = s["aug"]
    return (s["hard"], s["ctx"], a["copies"] if a else 0, (info or {}).get("examples", 0))


def rule_key(r):
    return im.rule_key(r)


def all_rules(options):
    keys = list(options)
    rules = [dict(zip(keys, v)) for v in itertools.product(*(options[k] for k in keys))]
    return [FRUITPUNCH_RULE] + [r for r in rules if r != FRUITPUNCH_RULE]


################################################################################
# scoring
################################################################################

def per_seed_numbers(sw):
    import numpy as np
    row = {"max_recall": float(np.nanmax(sw["recall"])) if len(sw["recall"]) else 0.0,
           "best_f1": float(sw["f1"].max())}
    for t in SHOWN_RECALLS:
        r = im.at_recall(sw, t)
        row[f"at_{int(round(t * 100))}"] = None if r is None else {
            k: r[k] for k in ("threshold", "fp_per_hour", "precision", "recall")}
    need = [row[f"at_{int(round(t * 100))}"] for t in SCORE_RECALLS]
    row["score"] = statistics.mean(x["fp_per_hour"] for x in need) if all(need) else None
    return row


def stat(v):
    v = [x for x in v if x is not None and x == x]
    return (statistics.mean(v), min(v), max(v), statistics.pstdev(v), len(v)) if v else None


def summarize(sweeps):
    per = [per_seed_numbers(sw) for sw in sweeps]
    out = {"per_seed": per, "runs": len(per),
           "runs_reaching": sum(1 for r in per if r["score"] is not None)}
    out["score"] = stat([r["score"] for r in per]) if out["runs_reaching"] == len(per) else None
    for t in SHOWN_RECALLS:
        k = f"at_{int(round(t * 100))}"
        out[f"fp_{k}"] = stat([r[k] and r[k]["fp_per_hour"] for r in per])
        out[f"precision_{k}"] = stat([r[k] and r[k]["precision"] for r in per])
    out["max_recall"] = stat([r["max_recall"] for r in per])
    out["best_f1"] = stat([r["best_f1"] for r in per])
    return out


def rank_key(summ, simple):
    """Sort key: every run reaching 85% first, by score; then the rest by max recall."""
    if summ["score"]:
        return (0, summ["score"][0], simple)
    return (1, -(summ["max_recall"] or (0,))[0], simple)


def choose(items):
    """items: [(name, summary, simplicity)] -> (winner, ranked names, why). Lowest score wins;
    within one run-to-run standard deviation of the best it is a tie, broken by the highest
    max recall, then fewer false alarms at 90% recall, then simplicity."""
    ranked = sorted(items, key=lambda x: rank_key(x[1], x[2]))
    if not ranked:
        return None, [], "nothing to choose from"
    top = ranked[0]
    if not top[1]["score"]:
        return top[0], [x[0] for x in ranked], "no option reached 85% recall in every run; highest recall kept"
    band = top[1]["score"][0] + top[1]["score"][3]
    close = [x for x in ranked if x[1]["score"] and x[1]["score"][0] <= band + 1e-12]

    def tie(x):
        fp90 = x[1]["fp_at_90"]
        return (-round(x[1]["max_recall"][0], 3), fp90[0] if fp90 and fp90[4] == x[1]["runs"] else math.inf, x[2])
    win = min(close, key=tie)
    why = f"lowest score {top[1]['score'][0]:.3f} (+/- {top[1]['score'][3]:.3f}) by {top[0]}"
    if win is not top:
        why += (f"; {len(close)} options within the run-to-run spread, {win[0]} reaches the most rumbles "
                f"(max recall {win[1]['max_recall'][0]:.3f}) / fewer false alarms at 90%")
    return win[0], [x[0] for x in ranked], why


def fmt_score(s):
    if not s["score"]:
        return f"85% recall not reached by every run (max recall {s['max_recall'][0]:.3f})"
    return (f"score {s['score'][0]:.3f} (FA/h at 80% {s['fp_at_80'][0]:.3f}, 85% {s['fp_at_85'][0]:.3f}, "
            f"90% {s['fp_at_90'][0] if s['fp_at_90'] else float('nan'):.3f}; max recall {s['max_recall'][0]:.3f})")


def average_precision(score, y):
    """Frame-level average precision (rumble = positive)."""
    import torch
    order = torch.argsort(score, descending=True)
    ys = y[order].float()
    tp = torch.cumsum(ys, 0)
    prec = tp / torch.arange(1, len(ys) + 1, device=ys.device)
    n = ys.sum()
    return float((prec * ys).sum() / n) if n > 0 else 0.0


################################################################################
# days held out for early stopping
################################################################################

def heldout_days(cfg, pools):
    """About 1 in HELDOUT_ONE_IN usable training days of each recording set, picked by a hash
    of the day's name, so a day is held out in every setup that has it."""
    out = set()
    for p, files in cfg["train"].items():
        if p == "noise":
            continue
        usable = sorted(k for k in files if dc.is_usable(pools[p].labels.get(k, [])))
        h = {k: int(hashlib.md5(k.encode()).hexdigest(), 16) for k in usable}
        pick = [k for k in usable if h[k] % HELDOUT_ONE_IN == 0]
        if not pick and len(usable) >= 3:
            pick = [min(usable, key=h.get)]
        out |= {(p, k) for k in pick}
    return out


################################################################################
# training data
################################################################################

def total_ram_bytes():
    return im.total_ram_bytes()


def label_counts(cfg, pools, skip):
    pos, gap = {}, {}
    for p, files in cfg["train"].items():
        if p == "noise":
            continue
        h = "clearing" if p == "clearing" else "forest"
        for k in files:
            if (p, k) in skip:
                continue
            labels = pools[p].labels.get(k, [])
            if dc.is_usable(labels):
                pi, ni = dc.label_frames(labels)
                pos[h] = pos.get(h, 0) + len(pi)
                gap[h] = gap.get(h, 0) + len(ni)
    return pos, gap


class AugDay:
    """Augmented fingerprints of one day (only the stretches that were augmented)."""

    def __init__(self, path):
        import torch
        d = torch.load(path)
        self.kind, self.n, self.emb = "aug", int(d["n"]), d["emb"]
        self.idx = d["idx"].numpy()
        self.pos_centers, self.neg_centers = d["pos"].numpy(), d["neg"].numpy()
        self.where = {int(i): j for j, i in enumerate(self.idx.tolist())}

    def frames(self, idx):
        import torch
        rows = [self.where[i] for i in idx if i in self.where]
        return self.emb[torch.tensor(rows, dtype=torch.long)] if rows else self.emb[:0]


def build_sets(lab, s, mined):
    """(bank [M, 768] fp16, train windows [N, W], train labels, held-out windows, held-out labels,
    info). Negatives: noise clips x `noise`, `rand` random frames per forest day, gap frames
    (`gap` per rumble frame, or 'fill' to make 1:1), mined false alarms, and augmented copies.
    One sample, shared by the runs (they differ in initialisation and order)."""
    import numpy as np
    import torch
    cfg, pools, cache = lab.cfg, lab.pools, lab.a.cache
    ctx = s["ctx"]
    rng = np.random.default_rng(lab.seeds[0])
    limit = int(MAX_RAM_SHARE * total_ram_bytes())
    held = lab.heldout
    matched = "clearing-matched" in cfg["components"]
    pos_n, gap_n = label_counts(cfg, pools, held)
    if matched:
        n_each = min(pos_n.values())
        f_pos = {h: n_each / v for h, v in pos_n.items()}
    else:
        f_pos = {h: 1.0 for h in pos_n}
    n_pos = int(round(sum(pos_n[h] * f_pos[h] for h in pos_n)))
    noise_days = {key: Day(cache, "noise", key) for key in cfg["train"].get("noise", [])} if s["noise"] else {}
    n_noise = sum(d.n for d in noise_days.values()) * s["noise"]
    forest_days = sum(1 for p, f in cfg["train"].items() if p != "noise" and dc.POOLS[p]["forest"]
                      for k in f if (p, k) not in held)
    n_rand = s["rand"] * forest_days
    gap_target = max(0, n_pos - n_noise - n_rand) if s["gap"] == "fill" else int(round(s["gap"] * n_pos))
    if matched and gap_n:
        f_gap = {h: min(1.0, (gap_target / len(gap_n)) / max(1, v)) for h, v in gap_n.items()}
    else:
        f_gap = {h: min(1.0, gap_target / max(1, sum(gap_n.values()))) for h in gap_n}

    banks, offset = [], 0
    parts = {"train": {"wins": [], "ys": []}, "held": {"wins": [], "ys": []}}
    counts = dict(rumble=0, gap=0, noise=0, random=0, mined=0, aug_rumble=0, aug_background=0,
                  held_rumble=0, held_not=0)
    offsets = np.arange(-ctx, ctx + 1)

    def add(day, groups):
        nonlocal offset
        picks = []
        for c, lab_, kind, f, part in groups:
            c = np.asarray(sorted(set(int(i) for i in c)), dtype=np.int64)
            c = c[(c >= 0) & (c < day.n)]
            if isinstance(day, AugDay) and len(c):
                inside = np.isin(np.clip(c[:, None] + offsets[None, :], 0, day.n - 1), day.idx).all(1)
                c = c[inside]
            picks.append((im.take(c, f, rng), lab_, kind, part))
        allc = [x for x, *_ in picks if len(x)]
        if not allc:
            return
        need = np.unique(np.clip(np.concatenate(allc)[:, None] + offsets[None, :], 0, day.n - 1))
        if limit and (offset + len(need)) * 768 * 2 > limit:
            raise MemoryError(f"{name_of(s)} needs more than {limit / 2**30:.0f} GB of fingerprints in memory "
                              f"({MAX_RAM_SHARE:.0%} of this PC's RAM); skipped")
        rows = day.frames(need.tolist())
        if rows.shape[0] != len(need):
            raise RuntimeError("fingerprints missing for a day: the cache is incomplete")
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
            day = Day(cache, p, key)
            labels = pools[p].labels.get(key, [])
            if (p, key) in held:
                pi, ni = dc.label_frames(labels)
                rnd = dc.random_negative_frames(labels, day.n, key + "-heldout", n=HELDOUT_RANDOM_PER_DAY) \
                    if day.kind == "day" else []
                add(day, [(pi, 1, "held_rumble", 1.0, "held"), (sorted(set(ni) | set(rnd)), 0, "held_not", 1.0, "held")])
                continue
            groups = []
            if dc.is_usable(labels):
                pi, ni = dc.label_frames(labels)
                groups += [(pi, 1, "rumble", f_pos.get(h, 1.0), "train"), (ni, 0, "gap", f_gap.get(h, 0.0), "train")]
            if s["rand"] and dc.POOLS[p]["forest"]:
                if day.kind != "day":
                    raise RuntimeError(f"{p} {key} has no whole-day fingerprints: run `python tools\\improve.py embed`")
                groups.append((dc.random_negative_frames(labels, day.n, key, n=s["rand"]), 0, "random", 1.0, "train"))
            if (p, key) in mined:
                groups.append((mined[(p, key)], 0, "mined", 1.0, "train"))
            if ctx and day.kind != "day":
                raise RuntimeError(f"{p} {key} has no whole-day fingerprints: run `python tools\\improve.py embed`")
            add(day, groups)
            for variant in aug_variants(s["aug"]):
                path = lab.aug.path(variant, p, key)
                if not os.path.exists(path):
                    continue
                ad = AugDay(path)
                g = []
                if variant.startswith(("clean", "rumble")) and dc.is_usable(labels):
                    g.append((ad.pos_centers, 1, "aug_rumble", f_pos.get(h, 1.0), "train"))
                if variant.startswith("background"):
                    g.append((ad.neg_centers, 0, "aug_background", 1.0, "train"))
                add(ad, g)
    bank = torch.empty((offset, 768), dtype=banks[0].dtype)
    at = 0
    banks.reverse()
    while banks:
        piece = banks.pop()
        bank[at:at + piece.shape[0]] = piece
        at += piece.shape[0]
    tr_w, tr_y = torch.cat(parts["train"]["wins"]), torch.cat(parts["train"]["ys"])
    if parts["held"]["wins"]:
        ho_w, ho_y = torch.cat(parts["held"]["wins"]), torch.cat(parts["held"]["ys"])
    else:
        ho_w, ho_y = tr_w[:0], tr_y[:0]
    neg = int((tr_y == 0).sum())
    info = {"examples": int(tr_y.shape[0]), "fingerprints_in_memory": int(bank.shape[0]),
            "rumble": counts["rumble"], "aug_rumble": counts["aug_rumble"],
            "negatives_from_gaps": counts["gap"], "negatives_from_noise": counts["noise"],
            "negatives_random_forest": counts["random"], "negatives_mined": counts["mined"],
            "negatives_aug_background": counts["aug_background"],
            "negatives_per_rumble": round(neg / max(1, int((tr_y == 1).sum())), 2),
            "heldout_days": len(held), "heldout_rumble_frames": counts["held_rumble"],
            "heldout_other_frames": counts["held_not"]}
    log(f"training set: {info}")
    return bank, tr_w, tr_y, ho_w, ho_y, info


################################################################################
# training until converged
################################################################################

class Quiet:
    def write(self, s):
        pass


def train_until_converged(bank, win, y, ho_win, ho_y, ctx, seed, logf, device,
                          check_every=None, patience=None, max_checks=None):
    """FruitPunch's model and optimiser (Adam, batch 64, balanced class weights), trained
    until the average precision on the held-out days has not improved for `patience`
    checks (one check per `check_every` examples). The best check is kept. Without
    held-out examples the held-out loss is not available, so the last check is kept."""
    import torch
    check_every = check_every or CHECK_EVERY
    patience = patience or PATIENCE
    max_checks = max_checks or MAX_CHECKS
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
        x = bank_d[w[idx.to(w.device)]].to(device).float()
        return x[:, 0] if ctx == 0 else x

    weights = (n / (2 * torch.bincount(y, minlength=2).float().clamp_min(1))).to(device)
    model = ex.make_model(ctx).to(device)
    crit = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters())
    g = torch.Generator().manual_seed(seed)

    def check():
        model.eval()
        with torch.no_grad():
            sc, loss = [], 0.0
            for i in range(0, ho_w_n, 8192):
                idx = torch.arange(i, min(ho_w_n, i + 8192))
                lg = model(batch_x(ho_d, idx))
                loss += crit(lg, ho_yd[idx.to(device)]).item() * len(idx)
                sc.append(torch.softmax(lg, 1)[:, 1])
        model.train()
        if not sc:
            return 0.0, float("nan")
        return average_precision(torch.cat(sc), ho_yd), loss / ho_w_n

    ho_w_n = ho_win.shape[0]
    seen, checks, best, best_check, best_state, since = 0, 0, -1.0, 0, None, 0
    t0 = time.time()
    done = False
    model.train()
    while not done:
        order = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss = crit(model(batch_x(win_d, b)), y_d[b.to(device)])
            loss.backward()
            opt.step()
            seen += b.shape[0]
            if seen >= (checks + 1) * check_every:
                checks += 1
                ap, hl = check()
                better = ap > best + 1e-5 or not ho_w_n
                msg = (f"seed {seed} check {checks} ({seen / n:.1f} epochs): train loss {loss.item() * 100:.2f}, "
                       f"held-out days AP {ap:.4f}, loss {hl * 100:.2f}{' *' if better else ''}")
                logf.write(msg + "\n")
                if checks % 5 == 0 or better:
                    log(msg)
                if better:
                    best, best_check, since = ap, checks, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    since += 1
                if since >= patience or checks >= max_checks:
                    done = True
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    model = model.cpu()
    model.run_info = {"checks": checks, "best_check": best_check, "examples_seen": seen,
                      "epochs": round(seen / n, 2), "best_epoch": round(best_check * check_every / n, 2),
                      "heldout_ap": round(best, 5), "hit_max": checks >= max_checks and since < patience,
                      "seconds": round(time.time() - t0, 1)}
    logf.write(json.dumps(model.run_info) + "\n")
    return model


################################################################################
# augmentation: noise mixed into the audio, then AVES
################################################################################

def aug_variants(a):
    """Stored variants an augmentation spec needs, e.g. ['rumble_0_c0', 'background_0_c0']."""
    if not a:
        return []
    types = {"clean": ["clean"], "rumble": ["rumble"], "background": ["background"],
             "both": ["rumble", "background"]}[a["type"]]
    out = []
    for c in range(a["copies"]):
        for t in types:
            out.append(f"clean_c{c}" if t == "clean" else f"{t}_{a['level']}_c{c}")
    return out


def merge_spans(frames, pad, n, max_len):
    """Contiguous [start, end) stretches covering every frame +/- pad, at most max_len long."""
    import numpy as np
    if not len(frames):
        return []
    f = np.unique(np.asarray(frames))
    spans, a, b = [], max(0, f[0] - pad), min(n, f[0] + pad + 1)
    for i in f[1:]:
        s, e = max(0, i - pad), min(n, i + pad + 1)
        if s <= b:
            b = max(b, e)
        else:
            spans.append((a, b))
            a, b = s, e
    spans.append((a, b))
    out = []
    for a, b in spans:
        while b - a > max_len:
            out.append((a, a + max_len))
            a += max_len - 2 * pad  # overlap so every frame keeps its full window in one piece
        out.append((a, b))
    return out


def band_power(y, sr=500):
    import numpy as np
    if len(y) < 8:
        return 0.0
    Y = np.fft.rfft(np.asarray(y, dtype=np.float64))
    f = np.fft.rfftfreq(len(y), 1 / sr)
    m = (f >= AUG_BAND[0]) & (f <= AUG_BAND[1])
    return float((np.abs(Y[m]) ** 2).sum() / len(y) ** 2)


class Augmenter:
    """Makes and stores augmented fingerprints, one file per variant and training day:
    <cache>\\_augmented\\<setup>\\<variant>\\<pool>\\<day>.pt"""

    def __init__(self, lab):
        self.lab = lab
        self.root = os.path.join(lab.a.cache, "_augmented", lab.setup_id or "none")
        self._noise = None
        self._embed = None

    def path(self, variant, pool, key):
        return os.path.join(self.root, variant, pool, f"{key}.pt")

    def done_path(self, variant):
        return os.path.join(self.root, variant, "done.json")

    def ready(self, variant):
        return os.path.exists(self.done_path(variant))

    def noise_bank(self):
        import numpy as np
        if self._noise is None:
            pool = self.lab.pools["noise"]
            clips = []
            for k in sorted(pool.files):
                folder = os.path.basename(pool.sound_dir_of[k].rstrip("\\/"))
                if folder in AUG_SKIP_NOISE:
                    continue
                y = dc.load_audio(pool.sounds[k]).numpy().astype(np.float32)
                if len(y) > 50:
                    clips.append(y)
            if not clips:
                raise RuntimeError("no noise clips to mix in")
            self._noise = clips
            log(f"augmentation: {len(clips)} noise clips loaded ({sum(map(len, clips)) / 500 / 3600:.1f} h)")
        return self._noise

    def noise_track(self, n, rng):
        import numpy as np
        clips = self.noise_bank()
        out, fade = [], 25
        have = 0
        while have < n:
            c = clips[int(rng.integers(len(clips)))]
            start = int(rng.integers(0, max(1, len(c) // 2)))
            piece = c[start:].copy()
            if len(piece) > 2 * fade:
                ramp = np.linspace(0, 1, fade, dtype=np.float32)
                piece[:fade] *= ramp
                piece[-fade:] *= ramp[::-1]
            out.append(piece)
            have += len(piece)
        return np.concatenate(out)[:n]

    def embedder(self):
        if self._embed is None:
            self._embed = dc.get_embedder(self.lab.a.fake_embedder)
        return self._embed

    def release(self):
        self._embed = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def read(self, path, f0, f1):
        """500 Hz audio of frames [f0, f1) of a day, as load_audio would give it."""
        import numpy as np
        import soundfile as sf
        from fractions import Fraction
        from scipy.signal import resample_poly
        with sf.SoundFile(path) as f:
            sr, total = f.samplerate, f.frames
            ratio = Fraction(500, sr)
            i0 = int(round(f0 * FRAME_SECONDS * sr))
            i1 = min(total, int(round(f1 * FRAME_SECONDS * sr)))
            f.seek(i0)
            x = f.read(max(0, i1 - i0), dtype="float32", always_2d=True)[:, 0]
        y = resample_poly(x, ratio.numerator, ratio.denominator) if len(x) else x
        return np.asarray(y, dtype=np.float16).astype(np.float32)

    def embed_audio(self, y):
        import torch
        e = self.embedder()
        t = torch.from_numpy(y)
        return e.whole_file(t) if isinstance(e, dc.FakeEmbedder) else e(t)

    def make(self, variant):
        """Every training day (not held out) of the setup, for one variant."""
        import numpy as np
        import torch
        if self.ready(variant):
            return
        kind, *rest = variant.split("_")
        level = None if kind == "clean" else rest[0]
        lab = self.lab
        t0 = time.time()
        log(f"augmentation: making {variant}")
        n_pos_total, n_days = 0, 0
        jobs = []
        for p, files in lab.cfg["train"].items():
            if p == "noise":
                continue
            for key in files:
                if (p, key) in lab.heldout:
                    continue
                labels = lab.pools[p].labels.get(key, [])
                if kind in ("clean", "rumble") and not dc.is_usable(labels):
                    continue
                jobs.append((p, key, labels))
                n_pos_total += len(dc.label_frames(labels)[0]) if labels else 0
                n_days += 1
        chunks_per_day = max(1, int(math.ceil(n_pos_total / max(1, n_days) / AUG_CHUNK)))
        ctx_audio = int(round(AUG_AUDIO_CONTEXT_S / FRAME_SECONDS))
        hours = 0.0
        for j, (p, key, labels) in enumerate(jobs, 1):
            dst = self.path(variant, p, key)
            if os.path.exists(dst):
                continue
            path = lab.pools[p].sounds[key]
            n_day = Day(lab.a.cache, p, key).n
            rng = np.random.default_rng(int(hashlib.md5(f"{variant}/{p}/{key}".encode()).hexdigest()[:8], 16))
            if kind in ("clean", "rumble"):
                pi = [i for i in dc.label_frames(labels)[0] if i < n_day]
                centers = np.asarray(sorted(set(pi)), dtype=np.int64)
                spans = merge_spans(centers, AUG_CTX_MAX + 1, n_day, AUG_MAX_SPAN)
            else:
                bad = np.zeros(n_day, bool)
                pad = int(dc.RANDOM_NEG_MIN_GAP / FRAME_SECONDS) + AUG_CTX_MAX + 1
                for l in labels:
                    i0 = max(0, int(l.start / FRAME_SECONDS) - pad)
                    bad[i0:min(n_day, int(l.end / FRAME_SECONDS) + pad + 1)] = True
                ok = np.flatnonzero(~bad[:max(0, n_day - AUG_CHUNK)])
                starts = sorted(rng.choice(ok, min(len(ok), chunks_per_day), replace=False)) if len(ok) else []
                centers = np.unique(np.concatenate([np.arange(s, s + AUG_CHUNK) for s in starts])) \
                    if starts else np.zeros(0, np.int64)
                spans = merge_spans(centers, AUG_CTX_MAX + 1, n_day, AUG_MAX_SPAN)
            idx_parts, emb_parts = [], []
            for a, b in spans:
                f0, f1 = max(0, a - ctx_audio), min(n_day, b + ctx_audio)
                y = self.read(path, f0, f1)
                hours += len(y) / 500 / 3600
                if kind != "clean" and len(y):
                    lvl = int(rng.choice([10, 0, -10])) if level == "mix" else int(level)
                    if kind == "rumble":
                        sig = np.concatenate([y[max(0, (i - f0) * 320):(i - f0 + 1) * 320]
                                              for i in centers[(centers >= a) & (centers < b)]] or [y])
                    else:
                        sig = y
                    noise = self.noise_track(len(y), rng)
                    ps, pn = band_power(sig), band_power(noise)
                    if pn > 0 and ps > 0:
                        y = y + noise * math.sqrt(ps / (pn * 10 ** (lvl / 10)))
                e = self.embed_audio(y.astype(np.float32))
                lo, hi = a - f0, min(b - f0, e.shape[0])
                if hi <= lo:
                    continue
                idx_parts.append(np.arange(f0 + lo, f0 + hi))
                emb_parts.append(e[lo:hi].half())
            if not idx_parts:
                continue
            idx = np.concatenate(idx_parts)
            emb = torch.cat(emb_parts)
            idx, first = np.unique(idx, return_index=True)  # overlapping pieces: keep one copy
            emb = emb[torch.from_numpy(first)]
            keep = centers[np.isin(centers, idx)]
            item = {"n": n_day, "idx": torch.from_numpy(idx), "emb": emb,
                    "pos": torch.from_numpy(keep if kind != "background" else keep[:0]),
                    "neg": torch.from_numpy(keep if kind == "background" else keep[:0])}
            lab.w.makedirs(os.path.dirname(dst))
            torch.save(item, lab.w.check(dst + ".tmp"))
            os.replace(lab.w.check(dst + ".tmp"), lab.w.check(dst))
            if j % 20 == 0:
                log(f"augmentation {variant}: {j}/{len(jobs)} days ({fmt_dur(time.time() - t0)})")
        wjson(lab.w, self.done_path(variant), {"variant": variant, "days": len(jobs), "audio_hours": round(hours, 1),
                                             "minutes": round((time.time() - t0) / 60, 1),
                                             "finished": time.strftime("%Y-%m-%d %H:%M")})
        log(f"augmentation: {variant} done in {fmt_dur(time.time() - t0)} ({hours:.0f} h of audio)")

    def check_disk(self):
        free = shutil.disk_usage(self.lab.a.cache).free / 2**30
        return free > AUG_DISK_MARGIN_GB, free


################################################################################
# the lab
################################################################################

class Lab:
    def __init__(self, a, pools, w):
        import torch
        self.a, self.pools, self.w = a, pools, w
        self.root = os.path.join(a.out, ROOT_NAME)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seeds = list(range(a.seed, a.seed + a.seeds))
        self.setups = find_setups(a.out)
        self.t_start = time.time()
        self.deadline = a.deadline
        self.exp_seconds = []
        self.stage = "starting"
        self.setup_id = None
        self.cfg = None
        self.heldout = set()
        self._val = {}
        self.aug = None
        if os.path.isdir(self.root):  # a resumed run: start from the times measured so far
            for n in sorted(os.listdir(self.root)):
                tr = self.jload(os.path.join(self.dir(n), "training.json")) if os.path.isdir(self.dir(n)) else None
                if tr and tr.get("train_minutes"):
                    self.exp_seconds.append(tr["train_minutes"] * 60 + 300)

    def use_setup(self, sid):
        self.setup_id = sid
        self.cfg = json.load(open(os.path.join(self.setups[sid], "config.json")))
        self.heldout = heldout_days(self.cfg, self.pools)
        self.aug = Augmenter(self)

    def dir(self, name):
        return os.path.join(self.root, name)

    def jload(self, path):
        return json.load(open(path)) if os.path.exists(path) else None

    # ---- choices --------------------------------------------------------------
    def choices(self):
        return self.jload(os.path.join(self.root, "choices.json")) or {}

    def save_choice(self, key, value):
        c = self.choices()
        c[key] = dict(value, time=time.strftime("%Y-%m-%d %H:%M"))
        wjson(self.w, os.path.join(self.root, "choices.json"), c)
        log(f"CHOICE {key}: {value.get('why', '')} -> {value.get('winner', value.get('rule'))}")

    # ---- time -------------------------------------------------------------------
    def per_experiment(self):
        return statistics.mean(self.exp_seconds[-10:]) if self.exp_seconds else FIRST_GUESS_S

    def time_left(self):
        return self.deadline - time.time() if self.deadline else math.inf

    def can_start(self, n_after=0, what="experiment"):
        """Is there time for one more experiment, plus `n_after` that must follow, plus the
        final steps?"""
        need = (1 + n_after) * self.per_experiment() + FINAL_RESERVE_S
        ok = self.time_left() > need
        if not ok:
            log(f"TIME: not starting {what}: {fmt_dur(max(0, self.time_left()))} left before --finish-by, "
                f"about {fmt_dur(need)} needed")
        return ok

    def report(self, stage):
        self.stage = stage
        left = self.time_left()
        t = [f"Progress, updated {time.strftime('%Y-%m-%d %H:%M')}", f"Now: {stage}",
             f"Running for {fmt_dur(time.time() - self.t_start)}; {len(self.exp_seconds)} experiments trained in this run, "
             f"about {fmt_dur(self.per_experiment())} each.",
             (f"Must finish by {time.strftime('%a %d %b %H:%M', time.localtime(self.deadline))} "
              f"({fmt_dur(max(0, left))} left). Later stages are shortened if time runs out; the final "
              f"rules and the tests always run.") if self.deadline else "No finish time set.",
             "", "Results so far: study.md. Choices made: choices.json."]
        try:
            with self.w.open(os.path.join(self.root, "progress.txt")) as f:
                f.write("\n".join(t) + "\n")
        except OSError as e:
            log(f"WARNING could not write progress.txt: {e}")
        log(f"PROGRESS: {stage}" + (f"; {fmt_dur(max(0, left))} left before the finish time" if self.deadline else ""))

    # ---- validation -------------------------------------------------------------
    def validation(self):
        key = self.setup_id
        if key not in self._val:
            files = []
            for p, keys in self.cfg["validation"]["files"].items():
                for k in keys:
                    files.append((p, k, Day(self.a.cache, p, k).hours))
            self._val[key] = files, im.Truth(self.pools, files)
        return self._val[key]

    def probs(self, models, ctx, files):
        out = [dict() for _ in models]
        for p, k, _ in files:
            emb = Day(self.a.cache, p, k).whole()
            for i, m in enumerate(models):
                out[i][(p, k)] = ex.day_probs(m, emb, ctx, self.device)
        return out

    def models(self, name):
        import torch
        s = self.spec(name)
        out = []
        for seed in self.seeds:
            m = ex.make_model(s["ctx"])
            m.load_state_dict(torch.load(os.path.join(self.dir(name), f"model_seed{seed}.pth"), map_location="cpu"))
            out.append(m.eval().to(self.device))
        return out

    def spec(self, name):
        return self.jload(os.path.join(self.dir(name), "spec.json"))

    # ---- one experiment -----------------------------------------------------------
    def trained_ok(self, name):
        return os.path.exists(os.path.join(self.dir(name), "training.json")) and all(
            os.path.exists(os.path.join(self.dir(name), f"model_seed{s}.pth")) for s in self.seeds)

    def train(self, s):
        import torch
        name = name_of(s)
        if self.trained_ok(name):
            return name
        wjson(self.w, os.path.join(self.dir(name), "spec.json"), s)
        mined, n_new = {}, 0
        if s["hard"]:
            prev = name_of(dict(s, hard=s["hard"] - 1))
            if not self.trained_ok(prev):
                raise RuntimeError(f"{prev} is needed first")
            mined, n_new = self.mine(prev)
            wjson(self.w, os.path.join(self.dir(name), "mined.json"),
                        [{"pool": p, "key": k, "frames": v} for (p, k), v in sorted(mined.items())])
        for v in aug_variants(s["aug"]):
            if not self.aug.ready(v):
                raise RuntimeError(f"augmented fingerprints {v} are missing")
        log(f"=== {name}: {describe(s)}; {len(self.seeds)} runs, each until converged")
        t0 = time.time()
        bank, win, y, ho_w, ho_y, info = build_sets(self, s, mined)
        runs = []
        with self.w.open(os.path.join(self.dir(name), "training_log.txt")) as logf:
            logf.write(json.dumps(info) + "\n")
            for seed in self.seeds:
                path = os.path.join(self.dir(name), f"model_seed{seed}.pth")
                info_path = path + ".json"
                if os.path.exists(path) and os.path.exists(info_path):
                    runs.append(json.load(open(info_path)))
                    continue
                m = train_until_converged(bank, win, y, ho_w, ho_y, s["ctx"], seed, logf, self.device)
                torch.save(m.state_dict(), self.w.check(path + ".tmp"))
                os.replace(self.w.check(path + ".tmp"), self.w.check(path))
                wjson(self.w, info_path, m.run_info)
                runs.append(m.run_info)
                log(f"run {seed}: {m.run_info}")
        del bank, win, y, ho_w, ho_y
        wjson(self.w, os.path.join(self.dir(name), "training.json"),
                    {"name": name, "spec": s, "description": describe(s), "data": info, "runs": runs,
                     "newly_mined_frames": n_new, "train_minutes": round((time.time() - t0) / 60, 1),
                     "check_every": CHECK_EVERY, "patience": PATIENCE, "max_checks": MAX_CHECKS})
        return name

    def run(self, s, n_after=0):
        """Train (if needed) and score one experiment; None if it failed or there was no time."""
        name = name_of(s)
        r = self.result(name)
        if r:
            return r
        if not self.trained_ok(name) and not self.can_start(n_after, name):
            return None
        t0 = time.time()
        try:
            trained_now = not self.trained_ok(name)
            self.train(s)
            r = self.evaluate(name, self.rule_for(s))
            if trained_now:
                self.exp_seconds.append(time.time() - t0)
            return r
        except Exception:
            record_failure(self, name)
            return None

    def rule_for(self, s):
        c = self.choices()
        if "rules" in c and c["rules"].get("setup") == s["setup"]:
            return c["rules"]["rule"]
        r1 = c.get(f"round1_rule_{s['setup']}")
        return r1["rule"] if r1 else FRUITPUNCH_RULE

    def result(self, name, rule=None):
        rule = rule or (self.spec(name) and self.rule_for(self.spec(name)))
        if not rule:
            return None
        r = self.jload(os.path.join(self.dir(name), f"results_{rule_key(rule)}.json"))
        if r and r.get("seeds") == self.seeds and r.get("rule") == rule:
            return r
        return None

    def evaluate(self, name, rule, probs=None):
        r = self.result(name, rule)
        if r:
            return r
        s = self.spec(name)
        files, truth = self.validation()
        if probs is None:
            probs = self.probs(self.models(name), s["ctx"], files)
        sweeps = [im.sweep(truth, pr, rule) for pr in probs]
        summ = summarize(sweeps)
        tr = self.jload(os.path.join(self.dir(name), "training.json")) or {}
        r = {"name": name, "spec": s, "description": describe(s), "rule": rule, "seeds": self.seeds,
             "summary": summ, "speed": self.speed(name, s, files, rule), "data": tr.get("data"),
             "runs": tr.get("runs"), "train_minutes": tr.get("train_minutes"),
             "validation": self.cfg["validation"]["name"], "validation_days": len(files),
             "validation_rumbles": truth.n, "finished": time.strftime("%Y-%m-%d %H:%M")}
        wjson(self.w, os.path.join(self.dir(name), f"results_{rule_key(rule)}.json"), r)
        try:
            im.write_curve(self.w, self.dir(name), name, sweeps)
        except OSError as e:
            log(f"WARNING could not write the curve of {name}: {e}")
        log(f"{name}: {fmt_score(summ)}")
        safe("study.md", write_report, self, lab=self)
        return r

    def score_on(self, name, rule, vfiles, tag):
        """Summary of an experiment on another set of validation days (saved, not used for choices)."""
        path = os.path.join(self.dir(name), f"side_{tag}_{rule_key(rule)}.json")
        r = self.jload(path)
        if r and r.get("seeds") == self.seeds:
            return r["summary"]
        files = [(p, k, Day(self.a.cache, p, k).hours) for p, ks in vfiles.items() for k in ks]
        truth = im.Truth(self.pools, files)
        probs = self.probs(self.models(name), self.spec(name)["ctx"], files)
        summ = summarize([im.sweep(truth, pr, rule) for pr in probs])
        wjson(self.w, path, {"name": name, "rule": rule, "seeds": self.seeds, "days": len(files),
                           "rumbles": truth.n, "summary": summ})
        log(f"{name} on {tag} ({len(files)} days, {truth.n} rumbles): {fmt_score(summ)}")
        return summ

    def speed(self, name, s, files, rule):
        import torch
        p, k, h = max(files, key=lambda f: f[2])
        emb = Day(self.a.cache, p, k).whole()
        scale = 24 / max(h, 1e-6)
        m0 = ex.make_model(s["ctx"])
        m0.load_state_dict(torch.load(os.path.join(self.dir(name), f"model_seed{self.seeds[0]}.pth"), map_location="cpu"))
        out = {"parameters": sum(x.numel() for x in m0.parameters()), "window_seconds": window_seconds(s["ctx"])}
        for dev in (["cuda"] if self.device == "cuda" else []) + ["cpu"]:
            m = copy.deepcopy(m0).to(dev).eval()
            ex.day_probs(m, emb[:2000], s["ctx"], dev)
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                im.alarms(ex.day_probs(m, emb, s["ctx"], dev), rule)
                if dev == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            out[f"classifier_seconds_per_24h_{dev}"] = round(statistics.median(times) * scale, 3)
        tp = os.path.join(self.a.out, "_timing.json")
        if os.path.exists(tp):
            mm = json.load(open(tp)).get("embed_s_per_audio_hour")
            if mm and mm.get("units"):
                out["aves_seconds_per_24h"] = round(mm["seconds"] / mm["units"] * 24, 1)
        return out

    # ---- picking -----------------------------------------------------------------
    def pick(self, stage, specs, n_after=0):
        """Run every option of a stage (as time allows) and choose the winner."""
        c = self.choices().get(stage)
        names = [name_of(s) for s in specs]
        if c and c.get("options") == names and c.get("seeds") == self.seeds:
            return c["winner"]
        results = []
        for i, s in enumerate(specs):
            r = self.run(s, n_after=n_after)
            if r:
                results.append(r)
        if not results:
            return None
        items = [(r["name"], r["summary"], cost(r["spec"], r.get("data"))) for r in results]
        win, ranked, why = choose(items)
        complete = len(results) == len(specs)
        self.save_choice(stage, {"winner": win, "ranked": ranked, "why": why, "options": names if complete else
                                 [r["name"] for r in results], "seeds": self.seeds,
                                 "missing": [n for n in names if n not in {r["name"] for r in results}]})
        return win

    # ---- rule tuning ------------------------------------------------------------------
    def tune_rules(self, name, stage, options):
        c = self.choices().get(stage)
        if c and c.get("experiment") == name and c.get("seeds") == self.seeds:
            return c["rule"]
        s = self.spec(name)
        rules = all_rules(options)
        self.report(f"{stage}: {len(rules)} alarm rules on {name}")
        files, truth = self.validation()
        probs = self.probs(self.models(name), s["ctx"], files)
        rows, t0 = [], time.time()
        caches = [{} for _ in probs]
        for i, rule in enumerate(rules):
            summ = summarize([im.sweep(truth, pr, rule, cc) for pr, cc in zip(probs, caches)])
            simple = (rule["smooth"], rule["gap"], rule["score"] != "mean", -rule["min_frames"])
            rows.append((rule_key(rule), summ, simple, rule))
            if (i + 1) % 200 == 0:
                log(f"{stage}: {i + 1}/{len(rules)} rules ({fmt_dur(time.time() - t0)})")
        win, ranked, why = choose([(k, sm, simple) for k, sm, simple, _ in rows])
        best = next(r for k, _, _, r in rows if k == win)
        by_key = {k: (sm, r) for k, sm, _, r in rows}
        with self.w.open(os.path.join(self.root, f"rules_{stage}.csv")) as f:
            wr = csv.writer(f)
            wr.writerow(["rank", "rule", "on", "min_frames", "smooth", "gap", "score_by", "score", "score_sd",
                         "fa_h_80", "fa_h_85", "fa_h_90", "runs_reaching_90", "max_recall", "best_f1"])
            for i, k in enumerate(ranked, 1):
                sm, r = by_key[k]
                g = lambda x: None if not x else round(x[0], 4)  # noqa: E731
                wr.writerow([i, k, r["on"], r["min_frames"], r["smooth"], r["gap"], r["score"], g(sm["score"]),
                             None if not sm["score"] else round(sm["score"][3], 4), g(sm["fp_at_80"]),
                             g(sm["fp_at_85"]), g(sm["fp_at_90"]), (sm["fp_at_90"] or [0] * 5)[4],
                             g(sm["max_recall"]), g(sm["best_f1"])])
        fp = by_key[rule_key(FRUITPUNCH_RULE)][0]
        why = f"{why}; picked {im.rule_text(best)}: {fmt_score(by_key[win][0])} (FruitPunch's rule: {fmt_score(fp)})"
        self.save_choice(stage, {"experiment": name, "rule": best, "why": why, "seeds": self.seeds,
                                 "rules_tried": len(rules), "setup": self.setup_id})
        for k in range(len(probs)):
            probs[k] = None
        self.evaluate(name, best)
        return best

    # ---- mining ---------------------------------------------------------------------
    def mine(self, prev):
        """False alarms of `prev` (the mean of its runs) on the training days, cleaned: none
        within MINE_PAD_S of a labelled rumble, the top MINE_FRAMES_PER_ALARM frames of each
        alarm, at most MINE_MAX_PER_DAY per day over all rounds (highest scores kept)."""
        import numpy as np
        r = self.result(prev)
        s = self.spec(prev)
        rule = r["rule"]
        cuts = [(x.get("at_80") or {}).get("threshold") for x in r["summary"]["per_seed"]]
        cuts = [c for c in cuts if c is not None]
        cut = statistics.mean(cuts) if cuts else 0.5
        old = {}
        for row in self.jload(os.path.join(self.dir(prev), "mined.json")) or []:
            old[(row["pool"], row["key"])] = list(row["frames"])
        models = self.models(prev)
        mined, n_new = {}, 0
        for p, keys in self.cfg["train"].items():
            if p == "noise":
                continue
            for key in keys:
                if (p, key) in self.heldout:
                    continue
                day = Day(self.a.cache, p, key)
                if day.kind != "day":
                    raise RuntimeError(f"{p} {key} has no whole-day fingerprints")
                emb = day.whole()
                prob = np.mean([ex.day_probs(m, emb, s["ctx"], self.device) for m in models], axis=0)
                labels = self.pools[p].labels.get(key, [])
                got = {i: float(prob[i]) for i in old.get((p, key), []) if i < len(prob)}
                for a0, b0, sc in zip(*im.alarms(prob, rule)):
                    if sc < cut or sc >= MINE_SKIP_ABOVE:
                        continue
                    t0, t1 = a0 * FRAME_SECONDS - MINE_PAD_S, b0 * FRAME_SECONDS + MINE_PAD_S
                    if any(l.start < t1 and l.end > t0 for l in labels):
                        continue
                    seg = np.arange(int(a0), int(b0) + 1)
                    top = seg[np.argsort(-prob[seg])[:MINE_FRAMES_PER_ALARM]]
                    for i in top.tolist():
                        if i not in got:
                            n_new += 1
                        got[i] = float(prob[i])
                if got:
                    keep = sorted(got, key=lambda i: -got[i])[:MINE_MAX_PER_DAY]
                    mined[(p, key)] = sorted(keep)
        log(f"mined {n_new} new false-alarm frames from {prev} at cut-off {cut:.3f} "
            f"({sum(len(v) for v in mined.values())} in total after the per-day cap)")
        return mined, n_new


################################################################################
# setups
################################################################################

def find_setups(out):
    """{setup id: config folder} for the first-round setups dataset_configs.py made."""
    found = {}
    if not os.path.isdir(out):
        return found
    for d in sorted(os.listdir(out)):
        p = os.path.join(out, d, "config.json")
        if not os.path.exists(p):
            continue
        try:
            cfg = json.load(open(p))
        except Exception:
            continue
        letter, comps = cfg.get("letter"), cfg.get("components", [])
        if letter == "E":
            sid = {frozenset(["clearing", "forest", "noise"]): "E2", frozenset(["clearing", "noise"]): "E"}.get(frozenset(comps))
            if not sid:
                log(f"note: {d} is an E built on something else ({comps}); not used")
                continue
        else:
            sid = letter
        if sid in found:
            log(f"WARNING two folders for setup {sid}: using {os.path.basename(found[sid])}, not {d}")
        elif sid:
            found[sid] = os.path.join(out, d)
    return found


def round1_spec(cfg, sid):
    has_noise = "noise" in cfg["components"]
    return dict(setup=sid, ctx=0, noise=1 if has_noise else 0,
                rand=dc.RANDOM_NEG_PER_DAY if cfg.get("random_negatives") else 0,
                gap="fill", hard=0, aug=None)


################################################################################
# the stages
################################################################################

def cmd_all(lab):
    # ---- 1: first round ----------------------------------------------------------
    lab.report("stage 1: first round (which recordings to train on)")
    missing = [s for s in lab.a.round1 if s not in lab.setups]
    if missing:
        log(f"WARNING setups {missing} have no configuration folder in {lab.a.out}; left out of the first round")
    items, r1 = [], {}
    for sid in [s for s in lab.a.round1 if s in lab.setups]:
        lab.use_setup(sid)
        s = round1_spec(lab.cfg, sid)
        name = name_of(s)
        if not lab.trained_ok(name) and not lab.can_start(0, name):
            continue
        t0 = time.time()
        try:
            trained_now = not lab.trained_ok(name)
            lab.train(s)
            if trained_now:
                lab.exp_seconds.append(time.time() - t0)
        except Exception:
            record_failure(lab, name)
            continue

        def score_round1():
            lab.evaluate(name, FRUITPUNCH_RULE)
            rule = lab.tune_rules(name, f"round1_rule_{sid}", ROUND1_RULES)
            return lab.evaluate(name, rule), rule
        got = safe(f"{name} scoring", score_round1, lab=lab)
        if not got:
            continue
        r, rule = got
        r1[sid] = (r, rule)
        if lab.cfg["validation"]["name"] == "Complex + General":
            items.append((sid, r["summary"], 0))
    safe("comparison with F and H", side_comparison, lab, r1, lab=lab)
    if not items:
        sys.exit("no first-round setup could be trained; see failed.txt")
    start = lab.a.start_from
    if start and start not in r1:
        log(f"WARNING --start-from {start}: that setup has no first-round result; using the first-round winner")
        start = None
    if start:
        win = start
        _, ranked, why = choose(items)
        lab.save_choice("round1", {"winner": win, "why": f"--start-from {win} (the first round alone would pick: {why})",
                                   "ranked": ranked})
    else:
        win, ranked, why = choose(items)
        lab.save_choice("round1", {"winner": win, "ranked": ranked, "why": why})
    lab.use_setup(win)
    state = round1_spec(lab.cfg, win)
    base = name_of(state)

    # ---- 2: all rules on the winner ---------------------------------------------------
    # from here on every experiment of this setup is scored with this rule (lab.rule_for);
    # if tuning fails, the first-round rule stays
    rule = safe("stage 2 rules", lab.tune_rules, base, "rules", im.RULE_OPTIONS, lab=lab)
    if rule:
        safe(f"{base} scoring", lab.evaluate, base, rule, lab=lab)

    # the explicit recipe of the winner: its gap frames as a number per rumble frame
    info = (lab.jload(os.path.join(lab.dir(base), "training.json")) or {}).get("data") or {}
    if "negatives_from_gaps" not in info:
        raise RuntimeError(f"{base} has no training.json")
    g0 = round(info["negatives_from_gaps"] / max(1, info["rumble"]), 2)
    state = dict(state, gap=g0)
    has_forest = any(dc.POOLS[p]["forest"] for p in lab.cfg["train"] if p != "noise")
    has_noise = bool(lab.cfg["train"].get("noise"))
    stage_options = {}

    def stage(key, label, knob, values, n_after):
        nonlocal state
        lab.report(f"stage {key} ({label})")
        vals = list(dict.fromkeys([state[knob]] + list(values)))
        specs = [dict(state, **{knob: v}) for v in vals]
        win = lab.pick(key, specs, n_after=0)
        if win:
            state = lab.spec(win)
            ch = lab.choices()[key]
            stage_options[knob] = [lab.spec(n)[knob] for n in ch["ranked"]]
        return win

    # stages run in order of priority: when time runs short the later ones are cut
    later = lambda k: 0  # noqa: E731
    if has_noise:
        stage("3", "noise clips used", "noise", NOISE_REPEATS, later("noise"))
    if has_forest:
        stage("4", "random forest frames per day", "rand", RANDOM_PER_DAY, later("rand"))
    stage("5", "gap frames per rumble frame", "gap", GAP_PER_RUMBLE, later("gap"))
    stage("6", "listening window", "ctx", CONTEXTS, later("ctx"))

    # ---- 7: augmentation --------------------------------------------------------------
    if not lab.a.no_augmentation:
        ok, free = lab.aug.check_disk()
        if not ok:
            log(f"augmentation skipped: only {free:.0f} GB free on the cache drive")
        else:
            safe("stage 7 augmentation", aug_stage, lab, state, stage_options, lab=lab)
            lab.aug.release()
            ch = lab.choices()
            for k in ("7c", "7b", "7a"):
                if k in ch and ch[k].get("winner"):
                    state = lab.spec(ch[k]["winner"])
                    break

    # ---- 8: combinations ----------------------------------------------------------------
    lab.report("stage 8 (combinations)")
    keys = [k for k in ("rand", "gap", "ctx", "aug") if stage_options.get(k)]
    tops = {}
    for k in keys:
        opts = stage_options[k]
        tops[k] = [state[k]] + [o for o in opts if o != state[k]][:COMBO_TOP - 1]
    combos = []
    for v in itertools.product(*(tops[k] for k in keys)):
        combos.append(dict(state, **dict(zip(keys, v))))
    # the most promising first, so a time cut drops the least promising
    rank_of = {k: {json.dumps(o, sort_keys=True): i for i, o in enumerate(stage_options[k])} for k in keys}
    combos.sort(key=lambda sp: sum(rank_of[k].get(json.dumps(sp[k], sort_keys=True), 9) for k in keys))
    for sp in combos:
        for v in aug_variants(sp["aug"]):
            if not lab.aug.ready(v) and lab.can_start(len(HARD_ROUNDS), f"augmentation {v}") \
                    and lab.aug.check_disk()[0]:
                safe(f"augmentation {v}", lab.aug.make, v, lab=lab)
    lab.aug.release()
    combos = [sp for sp in combos if all(lab.aug.ready(v) for v in aug_variants(sp["aug"]))]
    win = lab.pick("8", combos, n_after=len(HARD_ROUNDS)) if combos else None
    if win:
        state = lab.spec(win)

    # ---- 9: mistakes ----------------------------------------------------------------------
    lab.report("stage 9 (learning from its own false alarms)")
    if not lab.result(name_of(state)):  # nothing after stage 2 could be trained: go on from the base
        log(f"WARNING {name_of(state)} has no result; stage 9 starts from {base}")
        state = lab.spec(base)
    names = [name_of(state)]
    for h in HARD_ROUNDS:
        s = dict(state, hard=h)
        r = lab.run(s, n_after=len(HARD_ROUNDS) - h)
        if not r:
            break
        names.append(r["name"])
    items = [(n, lab.result(n)["summary"], cost(lab.spec(n))) for n in names if lab.result(n)]
    w9, ranked, why = choose(items)
    lab.save_choice("9", {"winner": w9, "ranked": ranked, "why": why, "options": names, "seeds": lab.seeds})
    state = lab.spec(w9)

    # ---- 10: final rules ---------------------------------------------------------------------
    final = name_of(state)
    final_rule = safe("stage 10 rules", lab.tune_rules, final, "rules_final", im.RULE_OPTIONS, lab=lab) \
        or lab.rule_for(state)
    lab.save_choice("final", {"winner": final, "rule": final_rule, "setup": lab.setup_id,
                              "why": f"final model {final} with {im.rule_text(final_rule)}"})
    safe("study.md", write_report, lab, lab=lab)
    lab.report(f"stages finished: {final}; scoring the tests")


def side_comparison(lab, r1):
    """F and H train on the validation days (Complex + General), so they are scored on held-out
    forest days instead. Every first-round setup that never trained on those days is scored on
    them too, with its own tuned rule, so F and H get a fair comparison."""
    side = [sid for sid, (r, _) in r1.items() if r["validation"] != "Complex + General"]
    if not side:
        return
    lab.use_setup(side[0])
    vfiles = lab.cfg["validation"]["files"]
    days = {(p, k) for p, ks in vfiles.items() for k in ks}
    items = []
    for sid, (r, rule) in r1.items():
        lab.use_setup(sid)
        trained = {(p, k) for p, ks in lab.cfg["train"].items() for k in ks}
        if trained & days:
            continue
        summ = lab.score_on(r["name"], rule, vfiles, "heldout-forest")
        items.append((sid, summ, 0))
    if len(items) > 1:
        win, ranked, why = choose(items)
        lab.save_choice("round1_side", {"winner": win, "ranked": ranked, "why": why + " (on the held-out forest days "
                                        "F and H are validated on; for information, the chain continues from the "
                                        "first-round winner on Complex + General)",
                                        "scores": {sid: summ for sid, summ, _ in items}})


def aug_stage(lab, state, stage_options):
    """7a which sounds (1 copy, 0 dB), 7b how loud, 7c how many copies."""
    def make_all(specs):
        for sp in specs:
            for v in aug_variants(sp["aug"]):
                if not lab.aug.ready(v):
                    if not lab.can_start(0, f"augmentation {v}"):
                        return False
                    ok, free = lab.aug.check_disk()
                    if not ok:
                        log(f"augmentation stopped: only {free:.0f} GB free")
                        return False
                    try:
                        lab.aug.make(v)
                    except Exception:
                        log(f"augmentation {v} FAILED:\n{traceback.format_exc()}")
                        return False
        lab.aug.release()
        return True

    lab.report("stage 7a (augmentation: which sounds)")
    specs = [dict(state, aug=None)] + [dict(state, aug=dict(type=t, level=0, copies=1)) for t in AUG_TYPES]
    if not make_all(specs):
        specs = [sp for sp in specs if all(lab.aug.ready(v) for v in aug_variants(sp["aug"]))]
    w = lab.pick("7a", specs, n_after=0)
    if not w or not lab.spec(w)["aug"] or lab.spec(w)["aug"]["type"] == "clean":
        stage_options["aug"] = [lab.spec(n)["aug"] for n in lab.choices().get("7a", {}).get("ranked", [])]
        log("augmentation: no noise type beat no augmentation / the clean control; levels and sizes skipped")
        return
    best = lab.spec(w)["aug"]
    lab.report("stage 7b (augmentation: how loud)")
    specs = [dict(state, aug=dict(best, level=lv)) for lv in AUG_LEVELS]
    make_all(specs)
    specs = [sp for sp in specs if all(lab.aug.ready(v) for v in aug_variants(sp["aug"]))]
    w = lab.pick("7b", [dict(state, aug=None)] + specs, n_after=0) or w
    if lab.spec(w)["aug"]:
        best = lab.spec(w)["aug"]
    lab.report("stage 7c (augmentation: how many copies)")
    specs = [dict(state, aug=dict(best, copies=c)) for c in AUG_COPIES]
    make_all(specs)
    specs = [sp for sp in specs if all(lab.aug.ready(v) for v in aug_variants(sp["aug"]))]
    w = lab.pick("7c", [dict(state, aug=None)] + specs, n_after=0) or w
    ranked = []
    for k in ("7c", "7b", "7a"):
        for n in lab.choices().get(k, {}).get("ranked", []):
            a = lab.spec(n)["aug"]
            if a not in ranked:
                ranked.append(a)
        if ranked:
            break
    stage_options["aug"] = ranked


################################################################################
# the tests, once at the end
################################################################################

def cmd_final(lab):
    c = lab.choices()
    if "final" not in c:
        log("final: the stages have not finished; scoring what has been chosen so far")
    steps = []
    round1 = c.get("round1", {}).get("winner")
    if "A" in lab.setups:
        lab.use_setup("A")
        a = name_of(round1_spec(lab.cfg, "A"))
        if lab.trained_ok(a):
            steps.append(("FruitPunch's data (setup A) and alarm rule", "A", a, FRUITPUNCH_RULE))
    if "E2" in lab.setups:
        lab.use_setup("E2")
        e2 = name_of(round1_spec(lab.cfg, "E2"))
        if lab.trained_ok(e2) and "round1_rule_E2" in c:
            steps.append(("E2 (last run's start), its tuned rule", "E2", e2, c["round1_rule_E2"]["rule"]))
    if round1:
        lab.use_setup(round1)
        rule = lab.rule_for(round1_spec(lab.cfg, round1))
        for key, label in (("round1", "1 first-round winner"), ("3", "3 noise clips"), ("4", "4 random frames"),
                           ("5", "5 gap frames"), ("6", "6 window"), ("7a", "7a augmentation type"),
                           ("7b", "7b augmentation loudness"), ("7c", "7c augmentation copies"),
                           ("8", "8 combinations"), ("9", "9 own false alarms")):
            w = c.get(key, {}).get("winner")
            if not w:
                continue
            if key == "round1":
                w = name_of(round1_spec(lab.cfg, round1))
            if lab.trained_ok(w):
                steps.append((label, round1, w, rule))
        if "final" in c:
            steps.append(("10 final model and alarm rule", round1, c["final"]["winner"], c["final"]["rule"]))
    if not steps:
        log("final: nothing to score yet")
        return
    sets = [("sampleB", "Test 1: Sample B"), ("sampleA", "Test 2: Sample A"), ("vehicle", "Vehicle (false alarms only)")]
    head = ["# The tests, scored once at the end", "",
            "Each run uses the cut-off that found 80% (or 85%) of the rumbles on the validation days. "
            f"Means over {len(lab.seeds)} runs. Each cell: recall / precision / false alarms per hour. "
            "\"-\" = the model trained on that set, or it failed (see failed.txt).", "",
            "| Step | Experiment | Validation score | " + " | ".join(f"{n} at 80% | at 85%" for _, n in sets) + " |",
            "|---|---|---|" + "---|---|" * len(sets)]
    rows_md, out = [], {"steps": []}

    def one(label, sid, name, rule):
        lab.use_setup(sid)
        r = lab.evaluate(name, rule)
        tests = {s["key"]: s for s in lab.cfg["tests"] if any(s["files"].values())}
        models = lab.models(name)
        row = {"step": label, "experiment": name, "rule": rule, "sets": {}}
        cells = []
        for key, _ in sets:
            s = tests.get(key)
            if not s:
                cells += ["-", "-"]
                continue
            keys = [(p, k) for p, ks in s["files"].items() for k in ks]
            probs = lab.probs(models, lab.spec(name)["ctx"], [(p, k, 0) for p, k in keys])
            files = [(p, k, len(probs[0][(p, k)]) * FRAME_SECONDS / 3600) for p, k in keys]
            truth = im.Truth(lab.pools, files)
            res = {}
            for rec in (80, 85):
                cuts = [(ps.get(f"at_{rec}") or {}).get("threshold") for ps in r["summary"]["per_seed"]]
                per = []
                for pr, cut in zip(probs, cuts):
                    if cut is None:
                        continue
                    sw = im.sweep(truth, pr, rule)
                    i = int(abs(sw["threshold"] - cut).argmin())
                    per.append({m: float(sw[m][i]) for m in ("recall", "precision", "fp_per_hour")})
                mean = {m: statistics.mean([x[m] for x in per if x[m] == x[m]] or [float("nan")])
                        for m in ("recall", "precision", "fp_per_hour")}
                res[rec] = {"per_run": per, "mean": mean}
                cells.append(f"{im._f(mean['recall'])} / {im._f(mean['precision'])} / {im._f(mean['fp_per_hour'], 2)}")
            row["sets"][key] = res
            if label.startswith("10"):
                safe("final Raven tables", im.write_final_detections, lab_view(lab), s, name, rule, r_view(r),
                     probs, files, truth, lab=lab)
        sc = r["summary"]["score"]
        return row, f"| {label} | {name} | {'-' if not sc else f'{sc[0]:.3f}'} | " + " | ".join(cells) + " |"

    for step in steps:
        got = safe(f"tests for {step[0]}", one, *step, lab=lab)
        if not got:
            rows_md.append(f"| {step[0]} | {step[2]} | failed, see failed.txt |" + " |" * (2 * len(sets)))
            continue
        out["steps"].append(got[0])
        rows_md.append(got[1])
        try:  # written after every step, so a late failure keeps what was scored
            wjson(lab.w, os.path.join(lab.root, "final.json"), out)
            with lab.w.open(os.path.join(lab.root, "final.md")) as f:
                f.write("\n".join(head + rows_md + ["", "The final model's detections (its best run on validation, "
                        "at the 80% cut-off) are Raven tables in `final_detections\\`, with each test set's "
                        "precision-recall curve."]) + "\n")
        except OSError as e:
            log(f"WARNING could not write final.md: {e}")
    print("\n".join(head + rows_md))
    log(f"wrote {os.path.join(lab.root, 'final.md')}")


def lab_view(lab):
    """What improve.write_final_detections needs from a lab."""
    v = copy.copy(lab)
    v.a = copy.copy(lab.a)
    v.a.target_recall = 0.80
    return v


def r_view(r):
    """A result in the shape improve.write_final_detections reads (per-seed at_80)."""
    return {"summary": {"per_seed": [{"at_80": ps.get("at_80")} for ps in r["summary"]["per_seed"]]}}


################################################################################
# the results table
################################################################################

STAGE_LABELS = [("round1", "1 first round"), ("3", "3 noise"), ("4", "4 random"), ("5", "5 gap"), ("6", "6 window"),
                ("7a", "7a aug type"), ("7b", "7b aug loudness"), ("7c", "7c aug copies"), ("8", "8 combinations"),
                ("9", "9 mistakes")]


def write_report(lab):
    if not os.path.isdir(lab.root):
        return
    c = lab.choices()
    stage_of = {}
    for key, label in STAGE_LABELS:
        for n in c.get(key, {}).get("options", c.get(key, {}).get("ranked", [])):
            stage_of.setdefault(n, label)
    rows = []
    for n in sorted(os.listdir(lab.root)):
        d = lab.dir(n)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "spec.json")):
            continue
        best = None
        for f in os.listdir(d):
            if f.startswith("results_") and f.endswith(".json"):
                r = lab.jload(os.path.join(d, f))
                if r and r.get("seeds") == lab.seeds and r["rule"] != FRUITPUNCH_RULE:
                    if best is None or r["finished"] >= best["finished"]:
                        best = r
        if best:
            rows.append(best)
    if not rows:
        return
    rows.sort(key=lambda r: rank_key(r["summary"], 0))
    g = lambda s, d=3: "-" if not s else f"{s[0]:.{d}f} ({s[1]:.{d}f}-{s[2]:.{d}f})"  # noqa: E731
    t = ["# The full rerun: every experiment", "",
         "Every number is on the validation days (RumbleComplex + RumbleGeneral) and averaged over "
         f"{len(lab.seeds)} training runs, with the range over runs in brackets. **Score = mean false alarms per "
         "hour at 80% and 85% recall** (lower is better); 90% is shown for information. Every model was "
         "trained until it stopped improving on held-out training days. The tests are untouched until the end.", ""]
    if "final" in c:
        t += [f"**Final model: {c['final']['winner']}**, alarm rule {im.rule_text(c['final']['rule'])}.", ""]
    t += ["| Experiment | Stage | Score | FA/h at 80% | at 85% | at 90% | Max recall | Best F1 | Epochs (best) | "
          "Train min | Examples | Classifier s per 24 h (GPU / CPU) | Rule |", "|" + "---|" * 13]
    for r in rows:
        s, sp = r["summary"], r.get("speed") or {}
        runs = r.get("runs") or []
        ep = (f"{statistics.mean(x['epochs'] for x in runs):.1f} ({statistics.mean(x['best_epoch'] for x in runs):.1f})"
              if runs else "-")
        t.append(f"| {r['name']} | {stage_of.get(r['name'], '')} | **{g(s['score'])}** | {g(s['fp_at_80'])} | "
                 f"{g(s['fp_at_85'])} | {g(s['fp_at_90'])} | {g(s['max_recall'])} | {g(s['best_f1'])} | {ep} | "
                 f"{r.get('train_minutes') or '-'} | {(r.get('data') or {}).get('examples', '-')} | "
                 f"{im._f(sp.get('classifier_seconds_per_24h_cuda'), 2)} / {im._f(sp.get('classifier_seconds_per_24h_cpu'), 2)} | "
                 f"{rule_key(r['rule'])} |")
    t += ["", "## Choices made", ""]
    for key in ["round1", "rules"] + [k for k, _ in STAGE_LABELS[1:]] + ["rules_final"]:
        if key in c:
            t.append(f"- {c[key]['time']} **{key}**: {c[key].get('why', '')} -> {c[key].get('winner', '')}"
                     + (f" (not run, out of time or failed: {', '.join(c[key]['missing'])})" if c[key].get("missing") else ""))
    for k, v in c.items():
        if k.startswith("round1_rule_"):
            t.append(f"- {v['time']} **{k}**: {v.get('why', '')}")
    with lab.w.open(os.path.join(lab.root, "study.md")) as f:
        f.write("\n".join(t) + "\n")
    keys = ["name", "stage", "setup", "window_s", "noise", "rand", "gap", "hard", "aug", "score", "score_sd",
            "fa_h_80", "fa_h_85", "fa_h_90", "runs_reaching_90", "max_recall", "best_f1", "epochs", "best_epoch",
            "train_minutes", "examples", "rumble", "aug_rumble", "neg_gap", "neg_noise", "neg_random", "neg_mined",
            "neg_aug_background", "gpu_s_24h", "cpu_s_24h", "parameters", "rule"]
    with lab.w.open(os.path.join(lab.root, "study.csv")) as f:
        wr = csv.writer(f)
        wr.writerow(keys)
        for r in rows:
            s, sp, d, spec = r["summary"], r.get("speed") or {}, r.get("data") or {}, r["spec"]
            runs = r.get("runs") or []
            m = lambda x: None if not x else round(x[0], 4)  # noqa: E731
            wr.writerow([r["name"], stage_of.get(r["name"], ""), spec["setup"], window_seconds(spec["ctx"]), spec["noise"],
                         spec["rand"], gap_text(spec["gap"]), spec["hard"], aug_text(spec["aug"]), m(s["score"]),
                         None if not s["score"] else round(s["score"][3], 4), m(s["fp_at_80"]), m(s["fp_at_85"]),
                         m(s["fp_at_90"]), (s["fp_at_90"] or [0] * 5)[4], m(s["max_recall"]), m(s["best_f1"]),
                         statistics.mean(x["epochs"] for x in runs) if runs else None,
                         statistics.mean(x["best_epoch"] for x in runs) if runs else None,
                         r.get("train_minutes"), d.get("examples"), d.get("rumble"), d.get("aug_rumble"),
                         d.get("negatives_from_gaps"), d.get("negatives_from_noise"), d.get("negatives_random_forest"),
                         d.get("negatives_mined"), d.get("negatives_aug_background"),
                         sp.get("classifier_seconds_per_24h_cuda"), sp.get("classifier_seconds_per_24h_cpu"),
                         sp.get("parameters"), rule_key(r["rule"])])


################################################################################

def default_deadline():
    """Next Tuesday 07:00 on this PC's clock (today if it is Tuesday before 07:00)."""
    now = time.localtime()
    days = (1 - now.tm_wday) % 7
    t = time.mktime((now.tm_year, now.tm_mon, now.tm_mday + days, 7, 0, 0, 0, 0, -1))
    if t <= time.time():
        t = time.mktime((now.tm_year, now.tm_mon, now.tm_mday + days + 7, 7, 0, 0, 0, 0, -1))
    return t


def parse_deadline(v):
    if not v:
        return default_deadline()
    if v.lower() == "none":
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(v, fmt))
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"--finish-by: use YYYY-MM-DD HH:MM, not {v!r}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", nargs="?", default="all", choices=["all", "report", "final"],
                   help="all: every stage, then the tests (default); report: rewrite study.md; final: only the tests")
    p.add_argument("--source", default=dc.DEFAULT_SOURCE, help=f"the Datasets copy (default {dc.DEFAULT_SOURCE})")
    p.add_argument("--out", help="the Configurations folder of dataset_configs.py (default <source>\\Configurations)")
    p.add_argument("--cache", help="its embedding cache (default <out>\\_embedding_cache)")
    p.add_argument("--seeds", type=int, default=SEEDS, help="training runs per experiment")
    p.add_argument("--seed", type=int, default=0, help="first seed")
    p.add_argument("--finish-by", help="stop starting new work so everything is done by then, e.g. "
                                       "'2026-10-27 07:00' (default: next Tuesday 07:00; 'none' for no limit)")
    p.add_argument("--round1", nargs="+", default=ROUND1, help=f"first-round setups (default {' '.join(ROUND1)})")
    p.add_argument("--start-from", help="skip the first-round choice and build on this setup")
    p.add_argument("--no-augmentation", action="store_true", help="leave out stage 7")
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.out = a.out or os.path.join(a.source, "Configurations")
    a.cache = a.cache or os.path.join(a.out, "_embedding_cache")
    a.deadline = parse_deadline(a.finish_by)
    im.keep_awake()
    root = dc.find_dataset_root(a.source)
    st_root = os.path.join(a.out, ROOT_NAME)
    w = dc.Writer([st_root, a.cache], [dc.ORIGINAL_DATA, os.path.join(root, "Training"), os.path.join(root, "Testing")])
    w.makedirs(st_root)
    stdout = sys.stdout
    sys.stdout = dc.Tee(w.check(os.path.join(st_root, "run_log.txt")))
    try:
        log(f"the full rerun; writing to {st_root}"
            + (f"; finishing by {time.strftime('%a %d %b %H:%M', time.localtime(a.deadline))}" if a.deadline else ""))
        pools = dc.load_pools(root)
        lab = Lab(a, pools, w)
        log(f"setups found: {lab.setups}")
        if a.step == "report":
            write_report(lab)
            return
        if a.step == "all":
            cfgs = []
            for sid in a.round1:
                if sid in lab.setups:
                    cfgs.append(json.load(open(os.path.join(lab.setups[sid], "config.json"))))
            for cfg in cfgs:  # whole-day fingerprints of every training day (most exist already)
                ex.cmd_embed(a, pools, w, cfg)
            try:
                cmd_all(lab)
            except SystemExit:
                raise
            except Exception:  # whatever happened, still score the tests on what was chosen so far
                record_failure(lab, "the stages")
        cmd_final(lab)
        lab.report("finished: see study.md and final.md")
    finally:
        sys.stdout = stdout


if __name__ == "__main__":
    main()
