#!/usr/bin/env python
"""
Round 2 of the candidate detectors: build on the two winners of round 1 (candidates.py),
Dasheng fingerprints (m6_dasheng) and the double-checker (m4), without making inference
much slower. Every experiment is scored exactly like the final model of the rerun: same
validation days, same score (mean false alarms per hour at 80% and 85% recall), alarm rule
tuned on the validation days only, the tests scored once at the end.

First stages (a score every 0.64 s, then an alarm rule):
  aves          the final model of the rerun (5 runs; the reference)
  dasheng       round 1's Dasheng model (5 runs)
  fusion        one classifier reading AVES and Dasheng fingerprints side by side (5 new runs)
  late          run i = mean of AVES run i and Dasheng run i
  dasheng06b    Dasheng 0.6B (600 M weights, frozen; ~7x Dasheng-base's compute; its fingerprints
                take ~350 MB per recorded day on disk), the final model's classifier (5 runs)
  dasheng_ft    Dasheng-base fine-tuned with LoRA together with the classifier (dasheng_ft.py,
                5 runs; inference ~1.3x frozen Dasheng)
  *_ens         all runs averaged into one model (aves_ens, dasheng_ens, fusion_ens,
                late_ens = 10 heads, ...). The heads are tiny, so this costs almost nothing at
                inference; the encoders run once either way.

Double-checkers (round 1's model 4, generalised): every alarm of a first stage gets a second
opinion from an image model looking at a 32 s low-pitch spectrogram around it, trained on
labelled rumbles and on false alarms of AVES and Dasheng first stages that never saw that day.
  resnet18, efficientnet_b0, convnext_tiny (ImageNet weights), 5 runs each;
  "+arch" uses checker run i for first-stage run i, "+arch x5" averages all 5 checkers.
  Scores: the checker's alone, or the geometric mean with the first stage's (chosen on
  validation). Applied to the 3 best first stages on validation.

On the lab PC, after candidates.py (needs its m6_dasheng and, if present, m4 fold models):

  python tools\\round2.py                       # everything; rerun to resume
  python tools\\round2.py --archs resnet18      # fewer checkers, faster
  python tools\\round2.py --skip dasheng06b     # without the big encoder (disk, time)
  python tools\\round2.py final                 # the tests for whatever has finished

Results: <out>\\_round2\\round2.md (validation) and round2_final.md (tests); progress.txt.
"""
import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import candidates as cd  # noqa: E402
import dasheng_ft as dft  # noqa: E402
import dataset_configs as dc  # noqa: E402
import experiments as ex  # noqa: E402
import improve as im  # noqa: E402
import study as st  # noqa: E402
from dataset_configs import log, fmt_dur, Day  # noqa: E402

ROOT_NAME = "_round2"
TRAINED = ["fusion", "dasheng06b", "dasheng_ft"]   # first stages trained here (can be skipped with --skip)
STAGES = ["aves", "dasheng", "fusion", "late", "dasheng06b", "dasheng_ft",
          "aves_ens", "dasheng_ens", "fusion_ens", "late_ens", "dasheng06b_ens", "dasheng_ft_ens"]
ARCHS = ["resnet18", "efficientnet_b0", "convnext_tiny"]
ARCH_LR = {"resnet18": 3e-4, "efficientnet_b0": 3e-4, "convnext_tiny": 1e-4}
CHECKED_STAGES = 3            # the checkers are tried on this many of the best first stages
NEG_PER_STAGE = 150           # false alarms per training day from each fold first stage (AVES, Dasheng)
TEST_SETS = [("sampleB", "Test 1: Sample B"), ("sampleA", "Test 2: Sample A"),
             ("vehicle", "Vehicle (false alarms only)")]
TITLES = {"aves": "Final model (AVES), the reference", "dasheng": "Dasheng (round 1)",
          "fusion": "AVES + Dasheng fingerprints, one classifier", "late": "AVES and Dasheng scores averaged",
          "aves_ens": "AVES, 5 runs averaged", "dasheng_ens": "Dasheng, 5 runs averaged",
          "fusion_ens": "AVES + Dasheng fingerprints, 5 runs averaged",
          "late_ens": "AVES and Dasheng, all 10 runs averaged",
          "dasheng06b": "Dasheng 0.6B (600 M weights, frozen)", "dasheng06b_ens": "Dasheng 0.6B, 5 runs averaged",
          "dasheng_ft": "Dasheng fine-tuned with LoRA", "dasheng_ft_ens": "Dasheng fine-tuned, 5 runs averaged"}
ENCODERS_OF = {"aves": "AVES", "dasheng": "Dasheng", "fusion": "AVES + Dasheng", "late": "AVES + Dasheng",
               "dasheng06b": "Dasheng 0.6B (about 7x Dasheng's compute)",
               "dasheng_ft": "Dasheng fine-tuned (about 1.3x Dasheng: overlapping blocks)"}


def fit(x, n):
    """Trim or pad (repeating the last value) to n steps: Dasheng and AVES can differ by one."""
    import numpy as np
    import torch
    if len(x) >= n:
        return x[:n]
    if isinstance(x, torch.Tensor):
        return torch.cat([x, x[-1:].expand(n - len(x), *x.shape[1:])]) if len(x) else torch.zeros(n, *x.shape[1:], dtype=x.dtype)
    return np.concatenate([x, np.repeat(x[-1:], n - len(x), 0)]) if len(x) else np.zeros(n, x.dtype)


class FusionSource(cd.Source):
    """AVES and Dasheng fingerprints side by side, on AVES's step count."""

    def __init__(self, aves, dasheng):
        self.aves, self.dasheng, self.dim = aves, dasheng, aves.dim + dasheng.dim

    def day(self, pool, key):
        import torch
        a = self.aves.day(pool, key).whole()
        d = fit(self.dasheng.day(pool, key).whole(), a.shape[0])
        return cd.FeatDay(torch.cat([a, d.to(a.dtype)], 1))


################################################################################
# checkers
################################################################################

def make_checker(arch, pretrained=True):
    """An ImageNet image model on one-channel spectrograms, one output (first layer's colour
    weights summed, as for round 1's ResNet-18)."""
    import torch
    import torchvision
    if arch == "resnet18":
        return cd.make_verifier(pretrained)
    tv = torchvision.models
    if arch == "efficientnet_b0":
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    elif arch == "convnext_tiny":
        m = tv.convnext_tiny(weights=tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(arch)
    c = m.features[0][0]
    new = torch.nn.Conv2d(1, c.out_channels, c.kernel_size, c.stride, c.padding, bias=c.bias is not None)
    new.weight.data = c.weight.data.sum(1, keepdim=True)
    if c.bias is not None:
        new.bias.data = c.bias.data.clone()
    m.features[0][0] = new
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, 1)
    return m


def train_checker(arch, x, y, hx, hy, seed, logf, device, pretrained=True):
    """candidates.train_verifier for any architecture (same crops, SpecAugment masks, class
    weight, early stopping on the held-out days' cuts)."""
    import numpy as np
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_checker(arch, pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ARCH_LR[arch], weight_decay=1e-4 if arch == "resnet18" else 0.05)
    pos_w = torch.tensor(float((y == 0).sum()) / max(1, int((y == 1).sum())), device=device)
    best, best_ep, best_state, since, ep, loss = -1e9, 0, None, 0, 0, None
    t0 = time.time()
    for ep in range(1, cd.VER_MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(y))
        for i in range(0, len(order), cd.VER_BATCH):
            b = order[i:i + cd.VER_BATCH]
            if len(b) < 2:
                continue
            xb = cd.crop(x[b], True, rng).to(device)
            yb = torch.from_numpy(y[b]).float().to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xb)[:, 0], yb, pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        lv = loss.item() if loss is not None else 0.0
        if len(hy) and hy.min() != hy.max():
            ap = st.average_precision(torch.from_numpy(cd.verifier_scores(model, hx, device)), torch.from_numpy(hy))
        else:
            ap = -lv
        better = ap > best + 1e-5
        logf.write(f"{arch} seed {seed} epoch {ep}: loss {lv:.4f}, held-out AP {ap:.4f}{' *' if better else ''}\n")
        if better:
            best, best_ep, since = ap, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
            if since >= cd.VER_PATIENCE:
                break
    model.load_state_dict(best_state)
    info = {"arch": arch, "epochs": ep, "best_epoch": best_ep, "heldout_ap": round(best, 5),
            "seconds": round(time.time() - t0, 1)}
    logf.write(json.dumps(info) + "\n")
    return model.eval().cpu(), info


class CutStore:
    """Spectrogram cuts around alarms, one file per recording, added to as new alarms come."""

    def __init__(self, lab, folder):
        self.lab, self.folder = lab, folder

    def path(self, part, p, k):
        return os.path.join(self.folder, part, p, f"{k}.npz")

    def get(self, part, p, k, boxes):
        import numpy as np
        boxes = [(int(a), int(b)) for a, b in boxes]
        path = self.path(part, p, k)
        have, x = {}, np.zeros((0, 128, cd.cut_cols()), np.uint8)
        if os.path.exists(path):
            z = np.load(path)
            x = z["x"]
            have = {tuple(int(v) for v in b): i for i, b in enumerate(z["boxes"])}
        missing = sorted(set(b for b in boxes if b not in have))
        if missing:
            y = dc.load_audio(self.lab.pools[p].sounds[k]).numpy()
            new = np.stack([cd.spectrogram_cut(y, (a + b + 1) / 2 * dc.FRAME_SECONDS) for a, b in missing])
            for b in missing:
                have[b] = len(have)
            x = np.concatenate([x, new])
            allb = sorted(have, key=have.get)
            self.lab.w.makedirs(os.path.dirname(path))
            with open(self.lab.w.check(path + ".tmp"), "wb") as f:
                np.savez(f, x=x, boxes=np.asarray(allb, np.int64).reshape(-1, 2))
            os.replace(self.lab.w.check(path + ".tmp"), self.lab.w.check(path))
        return x[[have[b] for b in boxes]] if boxes else x[:0]


################################################################################
# the round-2 lab
################################################################################

class R2(cd.CandLab):
    def __init__(self, a, pools, w):
        super().__init__(a, pools, w)
        self.cand = self.root                                   # round 1's folder
        self.root = os.path.join(a.out, ROOT_NAME)
        self.archs = a.archs
        self.skip = set(a.skip or [])
        self.cuts = CutStore(self, os.path.join(self.root, "cuts"))
        self._probs = {}
        self._models = {}
        self._cs = {}
        r6 = self.jload(os.path.join(self.cand, "m6_dasheng", "results.json"))
        if not r6:
            sys.exit("round 1's Dasheng model is missing: run candidates.py --only m6 first")
        self.rules = {"aves": self.final_rule, "dasheng": r6["rule"]}
        self.src = {"aves": cd.Source(a.cache), "dasheng": cd.Source(self.encoder_cache("dasheng"))}
        self.src["fusion"] = FusionSource(self.src["aves"], self.src["dasheng"])
        self.src["dasheng06b"] = cd.Source(self.encoder_cache("dasheng06b"), dim=cd.ENCODERS["dasheng06b"]["dim"])

    def report(self, now):
        try:
            with self.w.open(os.path.join(self.root, "progress.txt")) as f:
                f.write(f"Progress, updated {time.strftime('%Y-%m-%d %H:%M')}\nNow: {now}\n"
                        f"Running for {fmt_dur(time.time() - self.t0)}. Results so far: round2.md\n")
        except OSError as e:
            log(f"WARNING could not write progress.txt: {e}")
        log(f"PROGRESS: {now}")

    # ---- first-stage models ------------------------------------------------------------
    def base_models(self, base):
        import torch
        if base in self._models:
            return self._models[base]
        ctx = self.base_spec["ctx"]
        if base == "aves":
            ms = self.lab.models(self.final_name)
        elif base == "dasheng_ft":
            ms = [dft.load_ft(os.path.join(self.dir("dasheng_ft"), f"model_seed{s}.pth"), ctx, self.a.fake_embedder,
                              self.device) for s in self.seeds]
        else:
            folder = os.path.join(self.cand, "m6_dasheng") if base == "dasheng" else self.dir(base)
            ms = []
            for s in self.seeds:
                m = cd.make_ctx_model(self.src[base].dim, ctx)
                m.load_state_dict(torch.load(os.path.join(folder, f"model_seed{s}.pth"), map_location="cpu"))
                ms.append(m.eval().to(self.device))
        self._models[base] = ms
        return ms

    def train_base(self, base):
        """The final model's classifier and recipe on other fingerprints (fusion, Dasheng 0.6B)."""
        if self.trained(base):
            return
        if base == "dasheng06b":
            self.report("dasheng06b: fingerprints of every recording")
            cd.cmd_embed(self, "dasheng06b")
        src = self.src[base]
        self.report(f"{base}: building the training set")
        t0 = time.time()
        bank, win, y, ho_w, ho_y, info = cd.build_sets(self, self.base_spec, src)
        runs = []
        with self.w.open(os.path.join(self.dir(base), "training_log.txt"), "a") as logf:
            for seed in self.seeds:
                path = os.path.join(self.dir(base), f"model_seed{seed}.pth")
                if os.path.exists(path) and os.path.exists(path + ".json"):
                    runs.append(json.load(open(path + ".json")))
                    continue
                self.report(f"{base}: training run {seed}")
                m, ri = cd.train_windows(bank, win, y, ho_w, ho_y, lambda: cd.make_ctx_model(src.dim, self.base_spec["ctx"]),
                                         seed, logf, self.device)
                st.tsave(self.w, path, m.state_dict())
                st.wjson(self.w, path + ".json", ri)
                runs.append(ri)
                log(f"{base} run {seed}: {ri}")
        st.wjson(self.w, os.path.join(self.dir(base), "training.json"),
                 {"model": base, "dim": src.dim, "data": info, "runs": runs,
                  "train_minutes": round((time.time() - t0) / 60, 1)})

    def train_ft(self):
        """Dasheng-base fine-tuned with LoRA together with the classifier (dasheng_ft.py)."""
        if self.trained("dasheng_ft"):
            return
        t0 = time.time()
        folder = self.dir("dasheng_ft")
        dft.build_clips(self, os.path.join(folder, "clips"))
        runs = []
        with self.w.open(os.path.join(folder, "training_log.txt"), "a") as logf:
            for seed in self.seeds:
                path = os.path.join(folder, f"model_seed{seed}.pth")
                if os.path.exists(path) and os.path.exists(path + ".json"):
                    runs.append(json.load(open(path + ".json")))
                    continue
                self.report(f"dasheng_ft: fine-tuning run {seed}")
                state, ri = dft.train_ft(os.path.join(folder, "clips"), self.base_spec["ctx"], seed, logf, self.device,
                                         self.a.fake_embedder)
                st.tsave(self.w, path, state)
                st.wjson(self.w, path + ".json", ri)
                runs.append(ri)
                log(f"dasheng_ft run {seed}: {ri}")
        st.wjson(self.w, os.path.join(folder, "training.json"),
                 {"model": "dasheng_ft", "runs": runs, "train_minutes": round((time.time() - t0) / 60, 1)})

    # ---- scores per step, events ---------------------------------------------------------
    def split_keys(self, split):
        if split == "validation":
            return self.days("validation")
        s = next((t for t in self.lab.cfg["tests"] if t["key"] == split), None)
        return [(p, k) for p, ks in s["files"].items() for k in ks] if s else []

    def files(self, split):
        return [(p, k, Day(self.a.cache, p, k).n * dc.FRAME_SECONDS / 3600) for p, k in self.split_keys(split)]

    def base_probs(self, base, split):
        """[run] -> {(pool, key): probability per AVES step}."""
        key = (base, split)
        if key not in self._probs:
            ms = self.base_models(base)
            out = [dict() for _ in ms]
            for i, (p, k) in enumerate(self.split_keys(split)):
                n = Day(self.a.cache, p, k).n
                if base == "dasheng_ft":
                    if i % 10 == 0:
                        self.report(f"dasheng_ft: {split} recording {i + 1}")
                    for j, pr in enumerate(dft.day_probs(ms, self.pools[p].sounds[k], n, self.base_spec["ctx"], self.device)):
                        out[j][(p, k)] = fit(pr, n)
                    continue
                x = self.src[base].day(p, k).whole()
                for i, m in enumerate(ms):
                    out[i][(p, k)] = fit(ex.day_probs(m, x, self.base_spec["ctx"], self.device), n)
            self._probs[key] = out
        return self._probs[key]

    def stage_probs(self, stage, split):
        import numpy as np
        kind = stage.replace("_ens", "")
        if kind == "late":
            a, d = self.base_probs("aves", split), self.base_probs("dasheng", split)
            runs = [{k: (x[k] + y[k]) / 2 for k in x} for x, y in zip(a, d)]
            allr = a + d
        else:
            runs = self.base_probs(kind, split)
            allr = runs
        if stage.endswith("_ens"):
            return [{k: np.mean([r[k] for r in allr], 0) for k in allr[0]}]
        return runs

    def stage_events(self, stage, split, rule):
        return [{d: im.alarms(pr, rule) for d, pr in run.items()} for run in self.stage_probs(stage, split)]

    def stage_cost(self, stage):
        kind = stage.replace("_ens", "")
        heads = (10 if kind == "late" else 5) if stage.endswith("_ens") else (2 if kind == "late" else 1)
        return f"{ENCODERS_OF[kind]}; {heads} small head{'s' if heads > 1 else ''}"

    # ---- first stages on validation ----------------------------------------------------------
    def evaluate_stage(self, stage):
        path = os.path.join(self.dir(stage), "results.json")
        r = self.jload(path)
        if r and r.get("seeds") == self.seeds:
            return r
        self.report(f"{stage}: scoring the validation days")
        files = self.files("validation")
        truth = im.Truth(self.pools, files)
        probs = self.stage_probs(stage, "validation")
        if stage in self.rules:  # round 1 already tuned these rules on the same days
            rule, why = self.rules[stage], "the rule tuned before"
        else:
            rule, why = self.tune_rule(stage, probs, truth)
        sweeps = [cd.sweep_events(truth, ev) for ev in self.stage_events(stage, "validation", rule)]
        summ = st.summarize(sweeps)
        im.write_curve(self.w, self.dir(stage), stage, sweeps)
        r = {"name": stage, "kind": "stage", "title": TITLES[stage], "rule": rule, "why": why, "seeds": self.seeds,
             "summary": summ, "cost": self.stage_cost(stage), "validation_days": len(files),
             "validation_rumbles": truth.n, "finished": time.strftime("%Y-%m-%d %H:%M")}
        st.wjson(self.w, path, r)
        log(f"{stage}: {st.fmt_score(summ)}")
        return r

    def tune_rule(self, name, probs, truth):
        rules = st.all_rules(im.RULE_OPTIONS)
        rows, caches, t0 = [], [{} for _ in probs], time.time()
        for i, rule in enumerate(rules):
            summ = st.summarize([im.sweep(truth, pr, rule, cc) for pr, cc in zip(probs, caches)])
            simple = (rule["smooth"], rule["gap"], rule["score"] != "mean", -rule["min_frames"])
            rows.append((st.rule_key(rule), summ, simple, rule))
            if (i + 1) % 400 == 0:
                log(f"{name}: {i + 1}/{len(rules)} rules ({fmt_dur(time.time() - t0)})")
        win, _, why = st.choose([(k, sm, sp) for k, sm, sp, _ in rows])
        return next(r for k, _, _, r in rows if k == win), why

    # ---- checkers: training data -------------------------------------------------------------
    def fold_split(self):
        keys = self.train_days()
        return {pk: int(hashlib.md5(pk[1].encode()).hexdigest(), 16) % 2 for pk in keys}

    def fold_models(self, base):
        """Two copies of a first stage (one run each), each trained without one half of the
        training days. Model f never saw the days of fold f."""
        import torch
        fold = self.fold_split()
        ms = []
        for f in (0, 1):
            path = os.path.join(self.dir("folds"), f"{base}_fold{f}.pth")
            old = os.path.join(self.cand, "m4", f"first_stage_fold{f}.pth")  # round 1 trained AVES's already
            if base == "aves" and not os.path.exists(path) and os.path.exists(old):
                path = old
            if not os.path.exists(path):
                self.report(f"checkers: {base} first stage trained without half {f} of the training days")
                cfg0 = self.lab.cfg
                cfg = copy.deepcopy(cfg0)
                for p in cfg["train"]:
                    if p != "noise":
                        cfg["train"][p] = [k for k in cfg["train"][p] if fold.get((p, k), 1 - f) != f]
                self.lab.cfg = cfg
                try:
                    bank, win, y, ho_w, ho_y, _ = cd.build_sets(self, self.base_spec, self.src[base])
                finally:
                    self.lab.cfg = cfg0
                with self.w.open(os.path.join(self.dir("folds"), "training_log.txt"), "a") as logf:
                    m, _ = cd.train_windows(bank, win, y, ho_w, ho_y,
                                            lambda: cd.make_ctx_model(self.src[base].dim, self.base_spec["ctx"]),
                                            self.seeds[0], logf, self.device)
                del bank, win, y
                st.tsave(self.w, path, m.state_dict())
            m = cd.make_ctx_model(self.src[base].dim, self.base_spec["ctx"])
            m.load_state_dict(torch.load(path, map_location="cpu"))
            ms.append(m.eval().to(self.device))
        return ms, fold

    def checker_data(self):
        """Cuts from the training days: labelled rumbles (plus alarms that hit one) and the
        false alarms of AVES and Dasheng first stages that never saw the day; the held-out
        days' cuts are kept apart for stopping."""
        import numpy as np
        if getattr(self, "_cdata", None) is not None:
            return self._cdata
        folds = {b: self.fold_models(b) for b in ("aves", "dasheng")}
        fold = self.fold_split()
        held = sorted(self.lab.heldout)
        keys = [(pk, "train") for pk in self.train_days()] + [(pk, "held") for pk in held]
        xs, ys = {"train": [], "held": []}, {"train": [], "held": []}
        for i, ((p, k), part) in enumerate(keys):
            if i % 20 == 0:
                self.report(f"checkers: cuts of training day {i + 1}/{len(keys)}")
            n = Day(self.a.cache, p, k).n
            truth = im.Truth(self.pools, [(p, k, 1.0)])
            labels = self.pools[p].labels.get(k, [])
            r = np.random.default_rng(int(hashlib.md5(k.encode()).hexdigest(), 16) % 2**32)
            pos = [(dc.fp_data.time_to_index(l.start), dc.fp_data.time_to_index(l.start + l.duration)) for l in labels]
            if len(pos) > cd.VER_POS_PER_DAY:
                pos = [pos[j] for j in sorted(r.choice(len(pos), cd.VER_POS_PER_DAY, replace=False))]
            hits, neg = [], []
            for b, (ms, _) in folds.items():
                m = ms[fold[(p, k)]] if part == "train" else ms[0]
                x = self.src[b].day(p, k).whole()
                s, e, v = im.alarms(fit(ex.day_probs(m, x, self.base_spec["ctx"], self.device), n), self.rules[b])
                hit, _ = truth.match((p, k), s, e, v)
                hits += [(int(a), int(c)) for a, c, h in zip(s, e, hit) if h][:cd.VER_POS_PER_DAY // 4]
                nb = [(a, c, sc) for a, c, sc, h in zip(s, e, v, hit) if not h
                      and not (p == "clearing" and sc > cd.VER_SKIP_ABOVE)]
                neg += [(int(a), int(c)) for a, c, _ in sorted(nb, key=lambda t: -t[2])[:NEG_PER_STAGE]]
            pos_all = list(dict.fromkeys(pos + hits))
            neg = [b for b in dict.fromkeys(neg) if b not in set(pos_all)]
            boxes = pos_all + neg
            if not boxes:
                continue
            xs[part].append(self.cuts.get("train", p, k, boxes))
            ys[part].append(np.asarray([1] * len(pos_all) + [0] * len(neg), np.int64))
        cat = lambda v: np.concatenate(v) if v else np.zeros((0, 128, cd.cut_cols()), np.uint8)  # noqa: E731
        caty = lambda v: np.concatenate(v) if v else np.zeros(0, np.int64)  # noqa: E731
        self._cdata = (cat(xs["train"]), caty(ys["train"])), (cat(xs["held"]), caty(ys["held"]))
        return self._cdata

    def checker_trained(self, arch):
        return all(os.path.exists(os.path.join(self.dir("checkers"), f"{arch}_seed{s}.pth")) for s in self.seeds)

    def train_checkers(self, arch):
        if self.checker_trained(arch):
            return
        (x, y), (hx, hy) = self.checker_data()
        info = {"cuts": int(len(y)), "rumble_cuts": int(y.sum()), "heldout_cuts": int(len(hy))}
        log(f"checker training set: {info}")
        runs = []
        with self.w.open(os.path.join(self.dir("checkers"), "training_log.txt"), "a") as logf:
            for seed in self.seeds:
                path = os.path.join(self.dir("checkers"), f"{arch}_seed{seed}.pth")
                if os.path.exists(path) and os.path.exists(path + ".json"):
                    runs.append(json.load(open(path + ".json")))
                    continue
                self.report(f"checkers: training {arch} run {seed}")
                m, ri = train_checker(arch, x, y, hx, hy, seed, logf, self.device, pretrained=not self.a.fake_embedder)
                st.tsave(self.w, path, m.state_dict())
                st.wjson(self.w, path + ".json", ri)
                runs.append(ri)
                log(f"{arch} run {seed}: {ri}")
        st.wjson(self.w, os.path.join(self.dir("checkers"), f"{arch}_training.json"), {"data": info, "runs": runs})

    # ---- checkers: scoring alarms ------------------------------------------------------------
    def checker_scores(self, arch, split, events):
        """events: [run] -> {day: (s, e, v)}. Returns {day: {box: [score per checker run]}}."""
        import numpy as np
        import torch
        if arch not in self._models:
            ms = []
            for s in self.seeds:
                m = make_checker(arch, pretrained=False)
                m.load_state_dict(torch.load(os.path.join(self.dir("checkers"), f"{arch}_seed{s}.pth"), map_location="cpu"))
                ms.append(m.eval().to(self.device))
            self._models[arch] = ms
        ms = self._models[arch]
        out = {}
        for d in events[0]:
            boxes = sorted({(int(a), int(b)) for run in events for a, b in zip(run[d][0], run[d][1])})
            cache = self._cs.setdefault((arch, split, d), {})
            todo = [b for b in boxes if b not in cache]
            if todo:
                cuts = self.cuts.get(split, d[0], d[1], todo)
                sc = np.stack([cd.verifier_scores(m, cuts, self.device) for m in ms], 1)
                for b, row in zip(todo, sc):
                    cache[b] = row
            out[d] = cache
        return out

    def checked_events(self, stage, arch, ens, split, rule, how):
        import numpy as np
        events = self.stage_events(stage, split, rule)
        cs = self.checker_scores(arch, split, events)
        out = []
        for i, run in enumerate(events):
            ev = {}
            for d, (s, e, v) in run.items():
                rows = [cs[d][(int(a), int(b))] for a, b in zip(s, e)]
                pv = np.asarray([r.mean() if ens else r[i % len(r)] for r in rows], np.float64)
                v = np.asarray(v, np.float64)
                ev[d] = (np.asarray(s), np.asarray(e), pv if how == "verifier" else np.sqrt(np.clip(v, 0, 1) * pv))
            out.append(ev)
        return out

    def evaluate_checked(self, stage, arch, ens):
        name = f"{stage}+{arch}{'_x5' if ens else ''}"
        path = os.path.join(self.dir("checked"), f"{name}.json")
        r = self.jload(path)
        if r and r.get("seeds") == self.seeds:
            return r
        self.report(f"{name}: scoring the validation days")
        base = self.evaluate_stage(stage)
        files = self.files("validation")
        truth = im.Truth(self.pools, files)
        options = []
        for i, how in enumerate(cd.VER_COMBOS):
            sweeps = [cd.sweep_events(truth, ev) for ev in self.checked_events(stage, arch, ens, "validation", base["rule"], how)]
            options.append((how, st.summarize(sweeps), i))
        win, _, why = st.choose(options)
        summ = next(sm for h, sm, _ in options if h == win)
        r = {"name": name, "kind": "checked", "stage": stage, "arch": arch, "ens": ens, "rule": base["rule"],
             "combine": win, "why": why, "seeds": self.seeds, "summary": summ,
             "title": f"{TITLES[stage]} + {arch} checker{' (5 averaged)' if ens else ''}",
             "cost": f"{self.stage_cost(stage)}; {arch} on each alarm{' x5' if ens else ''}",
             "options": {h: st.fmt_score(sm) for h, sm, _ in options}, "finished": time.strftime("%Y-%m-%d %H:%M")}
        st.wjson(self.w, path, r)
        log(f"{name}: {st.fmt_score(summ)} (scores: {win})")
        return r

    # ---- all results ---------------------------------------------------------------------------
    def results(self):
        out = []
        for s in STAGES:
            r = self.jload(os.path.join(self.dir(s), "results.json"))
            if r:
                out.append(r)
        folder = self.dir("checked")
        if os.path.isdir(folder):
            for f in sorted(os.listdir(folder)):
                if f.endswith(".json"):
                    out.append(self.jload(os.path.join(folder, f)))
        return out

    def best_stages(self, n):
        rs = [r for r in self.results() if r["kind"] == "stage"]
        ranked = sorted(rs, key=lambda r: st.rank_key(r["summary"], r["name"]))
        return [r["name"] for r in ranked[:n]]

    def events_for(self, r, split):
        if r["kind"] == "stage":
            return self.stage_events(r["name"], split, r["rule"])
        return self.checked_events(r["stage"], r["arch"], r["ens"], split, r["rule"], r["combine"])


################################################################################
# commands
################################################################################

def record_failure(lab, what):
    tb = traceback.format_exc()
    log(f"FAILED {what}:\n{tb}")
    try:
        with lab.w.open(os.path.join(lab.root, "failed.txt"), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')} {what}\n{tb}\n")
    except OSError:
        pass


def safe(lab, what, fn, *a):
    try:
        return fn(*a)
    except Exception:
        record_failure(lab, what)
        return None


def write_report(lab):
    rs = lab.results()
    rs.sort(key=lambda r: st.rank_key(r["summary"], r["name"]))
    f3 = lambda x: "-" if not x else f"{x[0]:.3f}"  # noqa: E731
    rows = []
    for r in rs:
        s = r["summary"]
        sc = "85% not reached by every run" if not s["score"] else \
            f"**{s['score'][0]:.3f}**" + (f" ({s['score'][1]:.3f}-{s['score'][2]:.3f})" if s["runs"] > 1 else "")
        note = im.rule_text(r["rule"]) if r["kind"] == "stage" else f"scores: {r['combine']}"
        rows.append(f"| {r['name']} | {r['title']} | {sc} | {f3(s['fp_at_80'])} | {f3(s['fp_at_85'])} | "
                    f"{f3(s['fp_at_90'])} | {f3(s['max_recall'])} | {s['runs']} | {r['cost']} | {note} |")
    head = ["# Round 2 on the validation days", "",
            "Same days and score as the final model: **mean false alarms per hour at 80% and 85% recall** "
            "(lower is better), range over runs in brackets. Averaged (`_ens`, `_x5`) models are one model, "
            "so they have no range; compare them with the reference's range. Best first.", "",
            "| Name | What | Score | FA/h at 80% | at 85% | at 90% | Max recall | Runs | Inference | Rule / notes |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    with lab.w.open(os.path.join(lab.root, "round2.md")) as f:
        f.write("\n".join(head + rows) + "\n")


def cmd_all(lab):
    t0 = time.time()
    for base in TRAINED:
        if base in lab.skip:
            continue
        if base == "dasheng_ft":
            safe(lab, "dasheng_ft training", lab.train_ft)
        else:
            safe(lab, f"{base} training", lab.train_base, base)
    for s in STAGES:
        b = s.replace("_ens", "")
        if b in TRAINED and (b in lab.skip or not lab.trained(b)):
            continue
        safe(lab, f"{s} validation", lab.evaluate_stage, s)
        write_report(lab)
    top = lab.best_stages(CHECKED_STAGES)
    log(f"checkers will be tried on: {top}")
    for arch in lab.archs:
        if safe(lab, f"{arch} training", lab.train_checkers, arch) is None and not lab.checker_trained(arch):
            continue
        for s in top:
            for ens in (False, True):
                safe(lab, f"{s}+{arch} validation", lab.evaluate_checked, s, arch, ens)
                write_report(lab)
    lab.report(f"validation done in {fmt_dur(time.time() - t0)}; scoring the tests")


def cmd_final(lab):
    rs = lab.results()
    rs.sort(key=lambda r: st.rank_key(r["summary"], r["name"]))
    head = ["# Round 2 on the tests, scored once", "",
            "Each run uses the cut-off that found 80% (or 85%) of the rumbles on the validation days. Each cell: "
            "recall / precision / false alarms per hour, mean over runs. Ordered by the validation score.", "",
            "| Name | Validation score | " + " | ".join(f"{n} at 80% | at 85%" for _, n in TEST_SETS) + " |",
            "|---|---|" + "---|---|" * len(TEST_SETS)]
    rows, out = [], {}
    old = lab.jload(os.path.join(lab.root, "final.json")) or {}
    for r in rs:
        name = r["name"]
        if name in old and old[name].get("finished_validation") == r["finished"]:
            res = old[name]["sets"]
        else:
            lab.report(f"tests: {name}")
            try:
                res = {}
                for key, _ in TEST_SETS:
                    files = lab.files(key)
                    if not files:
                        continue
                    truth = im.Truth(lab.pools, files)
                    sweeps = [cd.sweep_events(truth, ev) for ev in lab.events_for(r, key)]
                    res[key] = {}
                    for rec in (80, 85):
                        per = []
                        for sw, ps in zip(sweeps, r["summary"]["per_seed"]):
                            cut = (ps.get(f"at_{rec}") or {}).get("threshold")
                            if cut is not None:
                                per.append(cd.at_cut(sw, cut))
                        mean = {k: statistics.mean([x[k] for x in per if x[k] == x[k]] or [float("nan")])
                                for k in ("recall", "precision", "fp_per_hour")}
                        res[key][str(rec)] = {"per_run": per, "mean": mean}
            except Exception:
                record_failure(lab, f"tests for {name}")
                continue
        out[name] = {"sets": res, "finished_validation": r["finished"]}
        st.wjson(lab.w, os.path.join(lab.root, "final.json"), out)
        cells = []
        for key, _ in TEST_SETS:
            for rec in ("80", "85"):
                m = ((res.get(key) or {}).get(rec) or {}).get("mean")
                cells.append("-" if not m else f"{im._f(m['recall'])} / {im._f(m['precision'])} / {im._f(m['fp_per_hour'], 2)}")
        sc = r["summary"]["score"]
        rows.append(f"| {name} | {'-' if not sc else f'{sc[0]:.3f}'} | " + " | ".join(cells) + " |")
    with lab.w.open(os.path.join(lab.root, "round2_final.md")) as f:
        f.write("\n".join(head + rows) + "\n")
    print("\n".join(head + rows))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", nargs="?", default="all", choices=["all", "report", "final"])
    p.add_argument("--archs", nargs="+", default=ARCHS, choices=ARCHS, help="checker architectures to train and try")
    p.add_argument("--skip", nargs="+", choices=TRAINED, help="first stages not to train (the slowest: dasheng06b)")
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
    w = dc.Writer([os.path.join(a.out, ROOT_NAME), os.path.join(a.out, "_embedding_cache_dasheng06b")],
                  [dc.ORIGINAL_DATA, os.path.join(root, "Training"), os.path.join(root, "Testing")])
    w.makedirs(os.path.join(a.out, ROOT_NAME))
    stdout = sys.stdout
    sys.stdout = dc.Tee(w.check(os.path.join(a.out, ROOT_NAME, "run_log.txt")))
    try:
        pools = dc.load_pools(root)
        lab = R2(a, pools, w)
        log(f"round 2: final model {lab.final_name}; checkers {lab.archs}; writing to {lab.root}")
        if a.step == "all":
            cmd_all(lab)
            cmd_final(lab)
        elif a.step == "final":
            cmd_final(lab)
        write_report(lab)
        lab.report("finished")
    finally:
        sys.stdout = stdout


if __name__ == "__main__":
    main()
