#!/usr/bin/env python
"""
Find the best model settings on top of the winning data configuration from
dataset_configs.py, one stage at a time, judged only on validation days.

Every experiment trains the small classifier with 3 seeds, scores the validation
days, and tunes the decision rules (smoothing, minimum length, gap merging, the
detection Score and the threshold) for it. Sample A and Sample B are never looked
at until `final`.

Stages (run by `all`, which keeps the winner of each stage):

  1 context    the classifier sees 1 fingerprint (FruitPunch) or ~6, ~11 or ~21 s
  2 negatives  1 or 5 "not rumble" frames per rumble frame, and 0, 1 or 2 rounds
               of hard negatives (the model's own false alarms on training days)
  3 recheck    the runner-up of stage 1 with the winning stage-2 settings

Steps:

  embed    whole-day fingerprints for the base configuration's training days
           (context windows and hard negatives need them; ~21 s and ~200 MB per day)
  all      embed, the 3 stages, compare (default; rerunning it resumes)
  compare  rewrite the results table: <out>\\_experiments\\experiments.md
  final    score the chosen experiment on the test sets, once, with its locked
           rules and threshold

Typical use on the lab PC, after `python tools\\dataset_configs.py` has finished:

  python tools\\experiments.py            # picks the base from best_choice.txt
  python tools\\experiments.py final      # when you are happy with the choice
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset_configs as dc  # noqa: E402
from dataset_configs import log, fmt_dur, Day, FRAME_SECONDS  # noqa: E402

SEEDS = 3
MINUTES = 5            # training time per seed (FruitPunch used 2; their val loss was still falling)
BATCH = 64
PROJ_DIM = 64          # context models: each fingerprint is first squeezed from 768 to 64 numbers
EXTRA_SHARE = 1 / 3    # share of negatives from noise clips + random forest frames (as dataset_configs E)
HARD_SHARE = 1 / 3     # at most this share of negatives from mined false alarms
HARD_PAD_S = 10.0      # a false alarm this close to a labelled rumble is not mined (may be the same call)
HARD_SKIP_ABOVE = 0.99  # very confident false alarms are not mined: they may be unmarked rumbles

STAGE1_CONTEXT = [0, 4, 8, 16]  # fingerprints on each side: 1, 9, 17, 33 fingerprints in total

# Decision rules tried for every experiment. Simplest first; the first one is FruitPunch's.
RULES = [dict(smooth=s, gap=g, min_frames=m, score=sc)
         for s in (1, 3, 5, 7) for g in (0, 1, 2) for m in (4, 5, 6) for sc in ("mean", "top3", "max")]
FRUITPUNCH_RULE = RULES[0]
assert FRUITPUNCH_RULE == dict(smooth=1, gap=0, min_frames=4, score="mean")


def exp_name(ctx, ratio, hard):
    return f"ctx{ctx}_neg{ratio}_hard{hard}"


def parse_name(name):
    c, r, h = name.split("_")
    return int(c[3:]), int(r[3:]), int(h[4:])


def context_seconds(ctx):
    return (2 * ctx + 1) * FRAME_SECONDS


def complexity(name):
    c, r, h = parse_name(name)
    return (c > 0, c, h, r)


################################################################################
# models
################################################################################

def make_model(ctx):
    import torch.nn as nn
    if ctx == 0:
        from fp_replica import ElephantRumbleClassifier
        return ElephantRumbleClassifier()  # FruitPunch's 768 -> 192 -> 2, same .pth format

    class ContextClassifier(nn.Module):
        """Squeeze each fingerprint to PROJ_DIM numbers, then decide from 2*ctx+1 of them side by side."""

        def __init__(self):
            super().__init__()
            self.ctx = ctx
            self.drop = nn.Dropout(0.2)
            self.act = nn.LeakyReLU()
            self.proj = nn.Linear(768, PROJ_DIM)
            self.linear1 = nn.Linear((2 * ctx + 1) * PROJ_DIM, 192)
            self.linear2 = nn.Linear(192, 2)

        def frame_features(self, x):  # [..., 768] -> [..., PROJ_DIM]
            return self.act(self.proj(self.drop(x)))

        def head(self, flat):  # [B, W * PROJ_DIM] -> [B, 2]
            return self.linear2(self.drop(self.act(self.linear1(self.drop(flat)))))

        def forward(self, x):  # [B, W, 768]
            h = self.frame_features(x)
            return self.head(h.reshape(h.shape[0], -1))

    return ContextClassifier()


def day_probs(model, emb, ctx, device, chunk=20_000):
    """Rumble probability for every frame of one day (numpy float32)."""
    import torch
    with torch.no_grad():
        x = emb.to(device).float()
        if ctx == 0:
            lg = torch.cat([model(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)])
        else:
            h = model.frame_features(x)
            hp = torch.cat([h[:1].expand(ctx, -1), h, h[-1:].expand(ctx, -1)])  # repeat the edge frames
            win = hp.unfold(0, 2 * ctx + 1, 1).transpose(1, 2)  # [n, W, PROJ_DIM], a view
            lg = torch.cat([model.head(win[i:i + chunk].reshape(-1, (2 * ctx + 1) * PROJ_DIM))
                            for i in range(0, win.shape[0], chunk)])
        return torch.softmax(lg, 1)[:, 1].float().cpu().numpy()


################################################################################
# decision rules: per-frame probabilities -> detections
################################################################################

def detections(prob, rule):
    """[(first frame, last frame, Score)] for one day. With FRUITPUNCH_RULE this is exactly
    fp_replica.cmd_detect: runs of >= 4 frames over 0.5, scored by their mean probability."""
    import numpy as np
    p = prob
    s = rule["smooth"]
    if s > 1 and len(p):
        k = np.ones(s)
        p = np.convolve(prob, k, "same") / np.convolve(np.ones_like(prob), k, "same")
    mask = np.concatenate([[False], p > 0.5, [False]])
    d = np.diff(mask.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    runs = []
    for a, b in zip(starts.tolist(), ends.tolist()):
        if runs and a - runs[-1][1] - 1 <= rule["gap"]:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    out = []
    for a, b in runs:
        if b - a + 1 < rule["min_frames"]:
            continue
        seg = prob[a:b + 1]
        if rule["score"] == "mean":
            sc = float(seg.mean())
        elif rule["score"] == "max":
            sc = float(seg.max())
        else:
            sc = float(np.sort(seg)[-3:].mean())
        out.append((a, b, sc))
    return out


def boxes_of(probs, rule):
    """probs: {(pool, key): prob array} -> elp_score Boxes."""
    out = []
    for (_, k), prob in probs.items():
        for a, b, sc in detections(prob, rule):
            out.append(dc.elp_score.Box(k, a * FRAME_SECONDS, b * FRAME_SECONDS, 5.0, 250.0, sc))
    return out


def tune_rules(probs_by_seed, truth, files):
    """Pick the rule + threshold with the best mean F1 over seeds (thresholds >= 0.5)."""
    best, fp_best = None, None
    for i, rule in enumerate(RULES):
        sweeps = [dc.elp_sweep(truth, boxes_of(pr, rule), files) for pr in probs_by_seed]
        for j, r0 in enumerate(sweeps[0]):
            if r0["threshold"] < 0.5:
                continue
            rows = [s[j] for s in sweeps]
            f1s = [r["f1"] for r in rows]
            cand = {"rule": rule, "rule_index": i, "threshold": r0["threshold"],
                    "f1_mean": statistics.mean(f1s), "f1_sd": statistics.pstdev(f1s), "per_seed": rows}
            if best is None or cand["f1_mean"] > best["f1_mean"] + 1e-12:
                best = cand
            if i == 0 and (fp_best is None or cand["f1_mean"] > fp_best["f1_mean"] + 1e-12):
                fp_best = cand
    return best, fp_best


def mean_of(rows, field):
    v = [r[field] for r in rows if r[field] == r[field]]
    return statistics.mean(v) if v else float("nan")


################################################################################
# training data: a bank of fingerprints + windows into it
################################################################################

def label_counts(cfg, pools):
    """Rumble and gap-negative frame counts per habitat, from the tables alone (no audio)."""
    pos, gap = {}, {}
    for p, files in cfg["train"].items():
        if p == "noise":
            continue
        h = "clearing" if p == "clearing" else "forest"
        for k in files:
            labels = pools[p].labels.get(k, [])
            if dc.is_usable(labels):
                pi, ni = dc.label_frames(labels)
                pos[h] = pos.get(h, 0) + len(pi)
                gap[h] = gap.get(h, 0) + len(ni)
    return pos, gap


def build_training_set(cfg, pools, cache, ctx, ratio, mined, seeds):
    """Rumble frames, ratio x as many negatives, as windows of 2*ctx+1 fingerprints.

    Negatives follow dataset_configs: gap negatives, plus (in E-based setups) noise clips
    and random forest frames as EXTRA_SHARE of them, plus up to HARD_SHARE mined false
    alarms. Each seed draws its own sample, but every day is read from disk only once.
    Returns (bank [M,768] fp16, [(windows [N,W] long, y [N], info)] one per seed)."""
    import numpy as np
    import torch
    rngs = [np.random.default_rng(s) for s in seeds]
    matched = "clearing-matched" in cfg["components"]
    pos_n, gap_n = label_counts(cfg, pools)
    if matched:  # setup D: the same number of rumble frames from each habitat
        n_each = min(pos_n.values())
        f_pos = {h: n_each / v for h, v in pos_n.items()}
        n_pos = n_each * len(pos_n)
    else:
        f_pos = {h: 1.0 for h in pos_n}
        n_pos = sum(pos_n.values())
    n_neg = ratio * n_pos

    # extra (noise + random) and mined availability, estimated before reading the big days
    n_noise = 0
    noise_days = {}
    for k in cfg["train"].get("noise", []):
        noise_days[k] = Day(cache, "noise", k)
        n_noise += noise_days[k].n
    forest_days = sum(len(f) for p, f in cfg["train"].items() if p != "noise" and dc.POOLS[p]["forest"])
    extra_avail = n_noise + (dc.RANDOM_NEG_PER_DAY * forest_days if cfg["random_negatives"] else 0)
    extra_target = min(extra_avail, int(n_neg * EXTRA_SHARE)) if cfg["random_negatives"] else 0
    mined_avail = sum(len(v) for v in mined.values())
    mined_target = min(mined_avail, int(n_neg * HARD_SHARE))
    gap_target = n_neg - extra_target - mined_target
    if matched:
        f_gap = {h: min(1.0, (gap_target / len(gap_n)) / v) for h, v in gap_n.items()}
    else:
        tot = sum(gap_n.values())
        f_gap = {h: min(1.0, gap_target / max(1, tot)) for h in gap_n}
    f_extra = extra_target / max(1, extra_avail)
    f_mined = mined_target / max(1, mined_avail)

    def take(idx, f, rng):
        idx = np.asarray(sorted(set(idx)), dtype=np.int64)
        if f >= 1 or len(idx) == 0:
            return idx
        return np.sort(rng.choice(idx, int(round(f * len(idx))), replace=False))

    banks = []
    per_seed = [{"wins": [], "ys": [], "counts": {"rumble": 0, "gap": 0, "extra": 0, "mined": 0}} for _ in seeds]
    offset = 0
    offsets = np.arange(-ctx, ctx + 1)

    def add(day, groups):
        """groups: [(candidate frames, label, kind, fraction)]; each seed samples its own subset."""
        nonlocal offset
        picks = []  # (seed no, centers, label, kind)
        for c, lab, kind, f in groups:
            c = np.asarray(sorted(set(c)), dtype=np.int64)
            c = c[(c >= 0) & (c < day.n)]
            for i, rng in enumerate(rngs):
                picks.append((i, take(c, f, rng), lab, kind))
        allc = [x for _, x, _, _ in picks if len(x)]
        if not allc:
            return
        need = np.unique(np.clip(np.concatenate(allc)[:, None] + offsets[None, :], 0, day.n - 1))
        rows = day.frames(need.tolist())
        if rows.shape[0] != len(need):
            raise RuntimeError("fingerprints missing for this day: run `python tools\\experiments.py embed`")
        banks.append(rows)
        for i, c, lab, kind in picks:
            if len(c) == 0:
                continue
            w = np.searchsorted(need, np.clip(c[:, None] + offsets[None, :], 0, day.n - 1)) + offset
            per_seed[i]["wins"].append(torch.from_numpy(w))
            per_seed[i]["ys"].append(torch.full((len(c),), lab, dtype=torch.long))
            per_seed[i]["counts"][kind] += len(c)
        offset += len(need)

    for p, files in cfg["train"].items():
        h = "clearing" if p == "clearing" else "forest"
        for k in files:
            if p == "noise":
                day = noise_days[k]
                add(day, [(range(day.n), 0, "extra", f_extra)])
                continue
            day = Day(cache, p, k)
            if (ctx or mined) and day.kind != "day":
                raise RuntimeError(f"{p} {k} has no whole-day fingerprints: run `python tools\\experiments.py embed`")
            labels = pools[p].labels.get(k, [])
            groups = []
            if dc.is_usable(labels):
                pi, ni = dc.label_frames(labels)
                groups += [(pi, 1, "rumble", f_pos[h]), (ni, 0, "gap", f_gap[h])]
            if cfg["random_negatives"] and dc.POOLS[p]["forest"]:
                groups.append((dc.random_negative_frames(labels, day.n, k), 0, "extra", f_extra))
            if (p, k) in mined:
                groups.append((mined[(p, k)], 0, "mined", f_mined))
            add(day, groups)
    bank = torch.cat(banks)
    out = []
    for s in per_seed:
        y = torch.cat(s["ys"])
        c = s["counts"]
        info = {"windows": int(y.shape[0]), "fingerprints_in_memory": int(bank.shape[0]),
                "rumble_frames": c["rumble"], "negatives_from_gaps": c["gap"],
                "negatives_from_noise_and_random": c["extra"], "negatives_mined": c["mined"],
                "negatives_per_rumble": round((y.shape[0] - c["rumble"]) / max(1, c["rumble"]), 2)}
        out.append((torch.cat(s["wins"]), y, info))
    return bank, out


def train_model(bank, win, y, ctx, minutes, seed, logf, device):
    """FruitPunch's loop (dataset_configs.train): 95/5 split, Adam, batch 64, balanced
    class weights, a time budget, and the same early-stopping rule."""
    import torch
    torch.manual_seed(seed)
    n = y.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    n_val = max(1, int(round(n * 0.05)))
    va, tr = perm[:n_val], perm[n_val:]
    try:
        bank_d = bank.to(device)
    except RuntimeError:  # GPU too small for the bank: gather on the CPU instead
        bank_d = bank
    win_d = win.to(bank_d.device)
    y_d = y.to(device)

    def batch_x(idx):
        x = bank_d[win_d[idx.to(win_d.device)]].to(device).float()
        return x[:, 0] if ctx == 0 else x

    weights = (n / (2 * torch.bincount(y, minlength=2).float())).to(device)
    model = make_model(ctx).to(device)
    crit = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters())
    g = torch.Generator().manual_seed(seed)
    t0, min_val = time.time(), float("inf")
    for epoch in range(20000):
        model.train()
        order = tr[torch.randperm(tr.shape[0], generator=g)]
        tot = cnt = 0
        for i in range(0, order.shape[0], BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss = crit(model(batch_x(b)), y_d[b.to(device)])
            loss.backward()
            opt.step()
            tot += loss.item() * b.shape[0]
            cnt += b.shape[0]
        model.eval()
        with torch.no_grad():
            val = sum(crit(model(batch_x(va[i:i + 4096])), y_d[va[i:i + 4096].to(device)]).item()
                      * min(4096, va.shape[0] - i) for i in range(0, va.shape[0], 4096)) / va.shape[0]
        msg = f"seed {seed} epoch {epoch + 1}: train loss {tot / max(1, cnt) * 100:.2f}, held-out 5% loss {val * 100:.2f}"
        log(msg)
        logf.write(msg + "\n")
        if val > min_val * 1.15 * 2:
            logf.write("held-out loss rising, stopping\n")
            break
        min_val = min(min_val, val)
        if time.time() - t0 > minutes * 60:
            break
    model.eval()
    return model.cpu()


################################################################################
# one experiment
################################################################################

class Lab:
    def __init__(self, a, pools, w):
        self.a, self.pools, self.w = a, pools, w
        self.root = os.path.join(a.out, "_experiments")
        self.cfg_dir = dc.find_config_dir(a.out, a.base)
        if not self.cfg_dir:
            sys.exit(f"no configuration {a.base}_* in {a.out}: run tools\\dataset_configs.py first")
        self.cfg = json.load(open(os.path.join(self.cfg_dir, "config.json")))
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._val = None

    def dir(self, name):
        return os.path.join(self.root, name)

    def result(self, name):
        p = os.path.join(self.dir(name), "results.json")
        return json.load(open(p)) if os.path.exists(p) else None

    def validation(self):
        """(files [(pool, key, hours)], truth Boxes) for the validation days."""
        if self._val is None:
            files = []
            for p, keys in self.cfg["validation"]["files"].items():
                for k in keys:
                    files.append((p, k, Day(self.a.cache, p, k).hours))
            self._val = files, dc.truth_boxes(self.pools, files)
        return self._val

    def load_models(self, name):
        import torch
        ctx = parse_name(name)[0]
        out = []
        for s in range(self.a.seeds):
            m = make_model(ctx)
            m.load_state_dict(torch.load(os.path.join(self.dir(name), f"model_seed{s}.pth"), map_location="cpu"))
            m.eval()
            out.append(m.to(self.device))
        return out

    def probs_for(self, models, ctx, files):
        """[{(pool, key): prob}] per model, loading each day once."""
        out = [dict() for _ in models]
        for p, k, _ in files:
            emb = Day(self.a.cache, p, k).whole()
            for i, m in enumerate(models):
                out[i][(p, k)] = day_probs(m, emb, ctx, self.device)
        return out

    def ensure(self, name):
        """Run an experiment (and the ones it builds on) unless it already has results."""
        r = self.result(name)
        if r and not self.a.rerun:
            return r
        ctx, ratio, hard = parse_name(name)
        if hard:
            self.ensure(exp_name(ctx, ratio, hard - 1))
        return self.run(name)

    def mined_from(self, prev):
        """False alarms of `prev` (its first seed, its locked rule and threshold) on the
        training days, plus everything `prev` itself had mined."""
        import torch
        r = self.result(prev)
        mined = {}
        mp = os.path.join(self.dir(prev), "mined.json")
        if os.path.exists(mp):
            for row in json.load(open(mp)):
                mined[(row["pool"], row["key"])] = list(row["frames"])
        ctx = parse_name(prev)[0]
        m = make_model(ctx)
        m.load_state_dict(torch.load(os.path.join(self.dir(prev), "model_seed0.pth"), map_location="cpu"))
        m.eval()
        m = m.to(self.device)
        rule, thr = r["rule"], r["threshold"]
        n_new = 0
        for p, keys in self.cfg["train"].items():
            if p == "noise":
                continue
            for k in keys:
                day = Day(self.a.cache, p, k)
                if day.kind != "day":
                    raise RuntimeError(f"{p} {k} has no whole-day fingerprints: run `python tools\\experiments.py embed`")
                prob = day_probs(m, day.whole(), ctx, self.device)
                labels = self.pools[p].labels.get(k, [])
                got = set(mined.get((p, k), []))
                for a, b, sc in detections(prob, rule):
                    if sc < thr or sc >= HARD_SKIP_ABOVE:
                        continue
                    t0, t1 = a * FRAME_SECONDS - HARD_PAD_S, b * FRAME_SECONDS + HARD_PAD_S
                    if any(l.start < t1 and l.end > t0 for l in labels):
                        continue
                    new = set(range(a, b + 1)) - got
                    n_new += len(new)
                    got |= new
                if got:
                    mined[(p, k)] = sorted(got)
        return mined, n_new

    def run(self, name):
        a, w = self.a, self.w
        ctx, ratio, hard = parse_name(name)
        d = self.dir(name)
        t_start = time.time()
        log(f"=== experiment {name}: {context_seconds(ctx):.1f} s of context, {ratio} negatives per rumble, "
            f"{hard} rounds of hard negatives, {a.seeds} seeds x {a.minutes} min")
        mined, n_new = {}, 0
        if hard:
            mined, n_new = self.mined_from(exp_name(ctx, ratio, hard - 1))
            log(f"mined {n_new} new false-alarm frames ({sum(len(v) for v in mined.values())} in total)")
            w.json(os.path.join(d, "mined.json"),
                   [{"pool": p, "key": k, "frames": v} for (p, k), v in sorted(mined.items())])
        import torch
        models, infos = [], []
        t0 = time.time()
        bank, sets = build_training_set(self.cfg, self.pools, a.cache, ctx, ratio, mined,
                                        [a.seed + s for s in range(a.seeds)])
        log(f"training data read in {fmt_dur(time.time() - t0)}: {bank.shape[0]} fingerprints in memory")
        with w.open(os.path.join(d, "training_log.txt")) as logf:
            for s, (win, y, info) in enumerate(sets):
                log(f"seed {s}: {info}")
                logf.write(json.dumps(info) + "\n")
                m = train_model(bank, win, y, ctx, a.minutes, a.seed + s, logf, self.device)
                torch.save(m.state_dict(), w.check(os.path.join(d, f"model_seed{s}.pth")))
                models.append(m.to(self.device).eval())
                infos.append(info)
        del bank, sets
        log(f"scoring {self.cfg['validation']['name']} and tuning the decision rules")
        files, truth = self.validation()
        probs = self.probs_for(models, ctx, files)
        best, fp_best = tune_rules(probs, truth, files)
        res = {"name": name, "context_frames_each_side": ctx, "context_seconds": round(context_seconds(ctx), 2),
               "negatives_per_rumble": ratio, "hard_negative_rounds": hard, "seeds": a.seeds,
               "minutes_per_seed": a.minutes, "base_configuration": os.path.basename(self.cfg_dir),
               "validation": self.cfg["validation"]["name"], "validation_days": len(files),
               "rule": best["rule"], "threshold": best["threshold"],
               "val_f1_mean": best["f1_mean"], "val_f1_sd": best["f1_sd"],
               "val_precision_mean": mean_of(best["per_seed"], "precision"),
               "val_recall_mean": mean_of(best["per_seed"], "recall"),
               "val_fp_per_hour_mean": mean_of(best["per_seed"], "fp_per_hour"),
               "val_per_seed": best["per_seed"],
               "fruitpunch_rule": {"threshold": fp_best["threshold"], "f1_mean": fp_best["f1_mean"],
                                   "f1_sd": fp_best["f1_sd"]},
               "training": infos, "newly_mined_frames": n_new,
               "run_minutes": round((time.time() - t_start) / 60, 1),
               "finished": time.strftime("%Y-%m-%d %H:%M")}
        w.json(os.path.join(d, "results.json"), res)
        log(f"{name}: validation F1 {res['val_f1_mean']:.3f} +/- {res['val_f1_sd']:.3f} "
            f"(P {res['val_precision_mean']:.3f}, R {res['val_recall_mean']:.3f}) with rule {best['rule']} "
            f"at threshold {best['threshold']:.2f}; FruitPunch's rule gave {fp_best['f1_mean']:.3f} "
            f"({fmt_dur(time.time() - t_start)})")
        cmd_compare(self)
        return res

    def pick(self, names, why):
        """Best mean validation F1; a simpler experiment within one seed-spread of it wins."""
        rs = [self.result(n) for n in names]
        rs = [r for r in rs if r]
        top = max(rs, key=lambda r: r["val_f1_mean"])
        close = [r for r in rs if r["val_f1_mean"] >= top["val_f1_mean"] - top["val_f1_sd"]]
        win = min(close, key=lambda r: complexity(r["name"]))
        ranked = sorted(rs, key=lambda r: -r["val_f1_mean"])
        msg = (f"{why}: picked {win['name']} (validation F1 " +
               ", ".join(f"{r['name']} {r['val_f1_mean']:.3f}+/-{r['val_f1_sd']:.3f}" for r in ranked) + ")")
        if win is not top:
            msg += f"; {top['name']} scored higher but within one seed-spread, so the simpler one is kept"
        log(msg)
        with self.w.open(os.path.join(self.root, "choices.txt"), "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M ") + msg + "\n")
        return win["name"], [r["name"] for r in ranked]


################################################################################
# steps
################################################################################

def cmd_embed(a, pools, w, cfg):
    import torch
    jobs = [(p, k) for p, keys in cfg["train"].items() if p != "noise" for k in keys
            if not os.path.exists(dc.cache_path(a.cache, p, k, "day"))]
    if not jobs:
        log("embed: every training day already has whole-day fingerprints")
        return
    log(f"embed: {len(jobs)} training days need whole-day fingerprints "
        f"(about {fmt_dur(len(jobs) * 22)} on the RTX 4060 and {len(jobs) * 0.21:.0f} GB of disk)")
    embed = dc.get_embedder(a.fake_embedder)
    t_start = time.time()
    for n, (p, k) in enumerate(jobs, 1):
        t0 = time.time()
        day = embed.whole_file(dc.load_audio(pools[p].sounds[k]))
        dst = dc.cache_path(a.cache, p, k, "day")
        torch.save({"kind": "day", "n": day.shape[0], "emb": day.half()}, w.check(dst + ".tmp"))
        os.replace(dst + ".tmp", dst)
        left = (time.time() - t_start) / n * (len(jobs) - n)
        log(f"embed {n}/{len(jobs)} {p} {k}: {day.shape[0]} frames ({time.time() - t0:.0f} s), about {fmt_dur(left)} left")


def _f(v, d=3):
    return "-" if v is None or v != v else f"{v:.{d}f}"


def rule_text(r):
    return (f"smooth {r['smooth']}, gap {r['gap']}, min {r['min_frames']} frames, {r['score']}")


def cmd_compare(lab):
    rows = []
    if os.path.isdir(lab.root):
        for n in sorted(os.listdir(lab.root)):
            r = lab.result(n) if os.path.isdir(lab.dir(n)) else None
            if r:
                rows.append(r)
    if not rows:
        log("no experiments yet")
        return
    rows.sort(key=lambda r: -r["val_f1_mean"])
    t = ["# Experiments", "",
         f"Base data configuration: {rows[0]['base_configuration']}. Scored on validation only "
         f"({rows[0]['validation']}); the tests are untouched until `final`.", "",
         "F1, precision, recall and false alarms per hour are means over the seeds, at the decision rule "
         "and threshold tuned for that experiment. **± is the spread between seeds**: differences smaller "
         "than that are luck. The last column is the same models with FruitPunch's fixed rule.", "",
         "| Experiment | Context (s) | Negatives per rumble | Hard-negative rounds | Val F1 | ± | Precision | "
         "Recall | FP/h | Rule | Threshold | F1 with FruitPunch's rule |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t.append(f"| {r['name']} | {r['context_seconds']:.1f} | {r['negatives_per_rumble']} | "
                 f"{r['hard_negative_rounds']} | **{_f(r['val_f1_mean'])}** | {_f(r['val_f1_sd'])} | "
                 f"{_f(r['val_precision_mean'])} | {_f(r['val_recall_mean'])} | {_f(r['val_fp_per_hour_mean'], 2)} | "
                 f"{rule_text(r['rule'])} | {r['threshold']:.2f} | {_f(r['fruitpunch_rule']['f1_mean'])} |")
    cp = os.path.join(lab.root, "choices.txt")
    if os.path.exists(cp):
        t += ["", "## Choices made", ""] + ["- " + l for l in open(cp).read().splitlines()]
    with lab.w.open(os.path.join(lab.root, "experiments.md")) as f:
        f.write("\n".join(t) + "\n")
    keys = ["name", "context_seconds", "negatives_per_rumble", "hard_negative_rounds", "val_f1_mean",
            "val_f1_sd", "val_precision_mean", "val_recall_mean", "val_fp_per_hour_mean", "threshold",
            "run_minutes", "finished"]
    with lab.w.open(os.path.join(lab.root, "experiments.csv")) as f:
        wr = csv.writer(f)
        wr.writerow(keys + ["rule", "fruitpunch_rule_f1"])
        for r in rows:
            wr.writerow([r[k] for k in keys] + [rule_text(r["rule"]), r["fruitpunch_rule"]["f1_mean"]])


def cmd_all(lab):
    cp = os.path.join(lab.root, "choices.txt")
    if os.path.exists(cp):
        os.remove(lab.w.check(cp))  # rewritten from the results on every run
    stage1 = [exp_name(c, 1, 0) for c in STAGE1_CONTEXT]
    for n in stage1:
        lab.ensure(n)
    win1, ranked1 = lab.pick(stage1, "stage 1 (context)")
    c = parse_name(win1)[0]
    stage2 = [exp_name(c, r, h) for r in (1, 5) for h in (0, 1, 2)]
    for n in stage2:
        lab.ensure(n)
    win2, _ = lab.pick(stage2, "stage 2 (negatives)")
    finalists = [win2]
    runner_up = next((n for n in ranked1 if n != win1), None)
    if runner_up:
        _, r, h = parse_name(win2)
        re_n = exp_name(parse_name(runner_up)[0], r, h)
        lab.ensure(re_n)
        finalists.append(re_n)
    win, _ = lab.pick(finalists, "stage 3 (recheck the runner-up of stage 1)")
    with lab.w.open(os.path.join(lab.root, "chosen.txt")) as f:
        f.write(win + "\n")
    cmd_compare(lab)
    log(f"chosen: {win}. Look at {os.path.join(lab.root, 'experiments.md')}; "
        f"when you are happy, run `python tools\\experiments.py final` (scores the tests once).")


def cmd_final(lab, name):
    a = lab.a
    if not name:
        cp = os.path.join(lab.root, "chosen.txt")
        if not os.path.exists(cp):
            sys.exit("no chosen experiment yet: run the stages first, or pass --experiment NAME")
        name = open(cp).read().strip()
    r = lab.result(name)
    if not r:
        sys.exit(f"{name} has no results")
    ctx = parse_name(name)[0]
    rule, thr = r["rule"], r["threshold"]
    best_seed = max(range(len(r["val_per_seed"])), key=lambda i: r["val_per_seed"][i]["f1"])
    models = lab.load_models(name)
    out = os.path.join(lab.dir(name), "final")
    res = {"experiment": name, "rule": rule, "threshold": thr, "delivered_seed": best_seed, "sets": {}}
    t = [f"# Final test: {name}", "",
         f"Locked on validation: rule {rule_text(rule)}, threshold {thr:.2f}. "
         f"Mean over {len(models)} seeds; the delivered model is `model_seed{best_seed}.pth` "
         "(best seed on validation) and its detections are in this folder.", "",
         "| Set | Days | Rumbles | Precision | Recall | F1 | FP/h | F1 range over seeds |",
         "|---|---|---|---|---|---|---|---|"]
    for s in lab.cfg["tests"]:
        files = [(p, k, Day(a.cache, p, k).hours) for p, keys in s["files"].items() for k in keys]
        if not files:
            continue
        truth = dc.truth_boxes(lab.pools, files)
        probs = lab.probs_for(models, ctx, files)
        rows = [dc.at(dc.elp_sweep(truth, boxes_of(pr, rule), files), thr) for pr in probs]
        res["sets"][s["key"]] = {"name": s["name"], "per_seed": rows}
        f1s = [x["f1"] for x in rows]
        t.append(f"| {s['name']} | {len(files)} | {rows[0]['rumbles']} | {_f(mean_of(rows, 'precision'))} | "
                 f"{_f(mean_of(rows, 'recall'))} | {_f(mean_of(rows, 'f1'))} | {_f(mean_of(rows, 'fp_per_hour'), 2)} | "
                 f"{_f(min(f1s))}-{_f(max(f1s))} |")
        with lab.w.open(os.path.join(out, f"{s['key']}_detections.txt")) as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerow(["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)",
                         "High Freq (Hz)", "Begin Path", "File Offset (s)", "Begin File", "Score"])
            sel, elapsed = 0, 0.0
            for p, k, h in files:
                path = lab.pools[p].sounds[k]
                for a0, b0, sc in detections(probs[best_seed][(p, k)], rule):
                    if sc < thr:
                        continue
                    sel += 1
                    bt, et = a0 * FRAME_SECONDS, b0 * FRAME_SECONDS
                    wr.writerow([sel, "Spectrogram 1", 1, f"{elapsed + bt:.3f}", f"{elapsed + et:.3f}", 5, 250,
                                 path, f"{bt:.3f}", os.path.basename(path), f"{sc:.4f}"])
                elapsed += h * 3600
    lab.w.json(os.path.join(out, "final.json"), res)
    with lab.w.open(os.path.join(out, "summary.md")) as f:
        f.write("\n".join(t) + "\n")
    print("\n".join(t))
    log(f"wrote {out}")


def base_from_choices(out):
    p = os.path.join(out, "best_choice.txt")
    if not os.path.exists(p):
        return None
    lines = [l for l in open(p).read().splitlines() if "picked " in l]
    return lines[-1].split("picked ")[1][0] if lines else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", nargs="?", default="all", choices=["embed", "all", "compare", "final"])
    p.add_argument("--source", default=dc.DEFAULT_SOURCE, help=f"the Datasets copy (default {dc.DEFAULT_SOURCE})")
    p.add_argument("--out", help="the Configurations folder of dataset_configs.py (default <source>\\Configurations)")
    p.add_argument("--cache", help="its embedding cache (default <out>\\_embedding_cache)")
    p.add_argument("--base", type=str.upper,
                   help="data configuration to build on (default: the last pick in best_choice.txt)")
    p.add_argument("--experiment", help="for final: which experiment (default: the one in chosen.txt)")
    p.add_argument("--seeds", type=int, default=SEEDS)
    p.add_argument("--minutes", type=float, default=MINUTES, help="training time per seed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rerun", action="store_true", help="redo experiments that already have results")
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.out = a.out or os.path.join(a.source, "Configurations")
    a.cache = a.cache or os.path.join(a.out, "_embedding_cache")
    a.base = a.base or base_from_choices(a.out)
    if not a.base:
        sys.exit(f"no best_choice.txt in {a.out} yet: let dataset_configs.py finish, or pass --base LETTER")
    root = dc.find_dataset_root(a.source)
    exp_root = os.path.join(a.out, "_experiments")
    w = dc.Writer([exp_root, a.cache], [dc.ORIGINAL_DATA, os.path.join(root, "Training"), os.path.join(root, "Testing")])
    w.makedirs(exp_root)
    stdout = sys.stdout
    sys.stdout = dc.Tee(w.check(os.path.join(exp_root, "run_log.txt")))
    try:
        log(f"experiments on configuration {a.base}; writing to {exp_root}")
        pools = dc.load_pools(root)
        lab = Lab(a, pools, w)
        if a.step in ("embed", "all"):
            cmd_embed(a, pools, w, lab.cfg)
        if a.step == "all":
            cmd_all(lab)
        elif a.step == "compare":
            cmd_compare(lab)
        elif a.step == "final":
            cmd_final(lab, a.experiment)
    finally:
        sys.stdout = stdout


if __name__ == "__main__":
    main()
