#!/usr/bin/env python
"""
Teaching AVES about rumbles, after the full rerun (study.py). Run it once the rerun has
finished; it starts from the rerun's final model (its data, listening window, mined false
alarms and alarm rule) and asks two questions:

  A  Probing: which AVES layer describes rumbles best? The rerun's recipe is trained on the
     fingerprints of layers 8, 10 and 12 (12 = the fingerprints used so far). If an
     earlier layer wins, inference can stop there (faster, too).
  C  Fine-tuning the top of AVES: the top 2 or top 4 transformer layers are trained together
     with the classifier, with a small learning rate.
  D  LoRA: AVES stays frozen; small adapters (rank 4 or 8) on the attention of the top 4
     layers are trained with the classifier.

Lower layers are never changed. Their output is computed once and cached
(<cache>\\_layers\\L08 ...), so training only runs the layers above.

Every option is scored exactly like the rerun (validation days, mean false alarms/h at 80%
and 85% recall, the same ties) and trained until the held-out training days stop improving.
The winner gets its own alarm-rule search and is scored once on the tests. Augmentation is
left out (the augmented fingerprints are of layer 12 only).

Writes only to <out>\\_finetune and <cache>\\_layers. A second run resumes.

  python tools\\finetune.py
"""
import argparse
import copy
import json
import math
import os
import shutil
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset_configs as dc  # noqa: E402
import experiments as ex  # noqa: E402
import improve as im  # noqa: E402
import study as st  # noqa: E402
from dataset_configs import log, fmt_dur, Day, FRAME_SECONDS  # noqa: E402

ROOT_NAME = "_finetune"
SEEDS = 3
PROBE_LAYERS = [8, 10]               # plus 12, the fingerprints already cached (8 and 10 also feed the fine-tuning)
FINETUNE = [dict(mode="top", n=2), dict(mode="top", n=4), dict(mode="lora", n=4, r=4), dict(mode="lora", n=4, r=8)]
LORA_ALPHA = 16
CHUNK_S = 3600                       # AVES runs on 1-hour pieces with 16 frames on each side, as the detector does
CONTEXT_FRAMES = 16
CROP = 1400                          # training sequence: about 15 minutes of fingerprints
MARGIN = 32                          # frames at the ends of a crop that are only context, not trained on
MAX_FRAMES_PER_CROP = 512
ACCUM = 4                            # crops per optimiser step
LR_LAYERS = 1e-5
LR_LORA = 1e-4
LR_HEAD = 1e-4                       # the classifier starts from the rerun's trained one
CHECK_STEPS = 300                    # optimiser steps between checks on the held-out days
PATIENCE = 8
MAX_CHECKS = 60
DISK_MARGIN_GB = 30


################################################################################
# AVES, layer by layer
################################################################################

def load_aves(fake=False):
    """The wav2vec2 model inside AVES (or a tiny random one with the same shape, for the tests)."""
    if fake:
        import torch
        from torchaudio.models import wav2vec2_model
        torch.manual_seed(0)
        return wav2vec2_model(
            extractor_mode="group_norm", extractor_conv_layer_config=[(32, 10, 5)] + [(32, 3, 2)] * 4 + [(32, 2, 2)] * 2,
            extractor_conv_bias=False, encoder_embed_dim=768, encoder_projection_dropout=0.1, encoder_pos_conv_kernel=16,
            encoder_pos_conv_groups=16, encoder_num_layers=12, encoder_num_heads=12, encoder_attention_dropout=0.1,
            encoder_ff_interm_features=64, encoder_ff_interm_dropout=0.0, encoder_dropout=0.1,
            encoder_layer_norm_first=False, encoder_layer_drop=0.0, aux_num_out=None).eval()
    from aves_torchaudio_wrapper import AvesTorchaudioWrapper
    return AvesTorchaudioWrapper().model.eval()


def unit(e):
    return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def chunks(n, chunk, ctx):
    """(a, b, s, e): run frames a..b, keep frames s..e (global indices)."""
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        yield max(0, s - ctx), min(n, e + ctx), s, e


def embed_layers(aves, y, layers, device):
    """Raw output of each of `layers` for a whole recording, as fp_replica.Embedder.whole_file
    does it (1-hour pieces, 16 frames of context, the last frame repeated)."""
    import torch
    hop = 320
    chunk, ctx = CHUNK_S * 500, CONTEXT_FRAMES * hop
    out = {l: [] for l in layers}
    top = max(layers)
    with torch.inference_mode():
        for s in range(0, y.shape[0], chunk):
            pre, cur, post = y[max(0, s - ctx):s], y[s:s + chunk], y[s + chunk:s + chunk + ctx]
            x = torch.cat([pre, cur, post])
            if x.shape[0] < hop * 2:
                continue
            feats, _ = aves.extract_features(x.view(1, -1).to(device), num_layers=top)
            a = pre.shape[0] // hop
            b = (pre.shape[0] + cur.shape[0]) // hop
            for l in layers:
                e = feats[l - 1][0].float().cpu()
                out[l].append(e[a:b])
                if post.shape[0] == 0 and e.shape[0] > 1:
                    out[l].append(e[-2:-1])
    return {l: torch.cat(v) if v else torch.empty(0, 768) for l, v in out.items()}


def layer_cache(cache, layer):
    return cache if layer == 12 else os.path.join(cache, "_layers", f"L{layer:02d}")


class LayerCaches:
    """Whole-day outputs of the probed layers, stored like the fingerprint cache: the unit
    vectors in "emb" (what a classifier reads) and their lengths in "norm" (to rebuild the raw
    output the layers above need)."""

    def __init__(self, lab, layers, fake):
        self.lab, self.layers, self.fake = lab, sorted(set(layers) - {12}), fake
        self._aves = None

    def aves(self):
        if self._aves is None:
            import torch
            self._aves = load_aves(self.fake).to("cuda" if torch.cuda.is_available() else "cpu")
        return self._aves

    def release(self):
        self._aves = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def missing(self, files):
        return [(p, k) for p, k in files
                if any(not os.path.exists(dc.cache_path(layer_cache(self.lab.a.cache, l), p, k, "day")) for l in self.layers)]

    def make(self, files, what):
        import torch
        todo = self.missing(files)
        if not todo or not self.layers:
            return
        frames = sum(Day(self.lab.a.cache, p, k).n if self.has_main(p, k) else 135_000 for p, k in todo)
        need_gb = frames * 768 * 2 * len(self.layers) / 2**30
        free = shutil.disk_usage(self.lab.a.cache).free / 2**30
        if free < need_gb + DISK_MARGIN_GB:
            raise RuntimeError(f"caching layers {self.layers} for {what} needs about {need_gb:.0f} GB "
                               f"(+{DISK_MARGIN_GB} GB margin) but only {free:.0f} GB are free")
        log(f"caching layers {self.layers} for {len(todo)} {what} recordings (about {need_gb:.0f} GB)")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        for i, (p, k) in enumerate(todo, 1):
            y = dc.load_audio(self.lab.pools[p].sounds[k])
            outs = embed_layers(self.aves(), y, self.layers, dev)
            for l, e in outs.items():
                path = dc.cache_path(layer_cache(self.lab.a.cache, l), p, k, "day")
                self.lab.w.makedirs(os.path.dirname(path))
                n = e.norm(dim=1)
                st.tsave(self.lab.w, path, {"kind": "day", "n": int(e.shape[0]), "emb": unit(e).half(),
                                           "norm": n.half(), "layer": l})
            if i % 10 == 0 or i == len(todo):
                log(f"layers cached: {i}/{len(todo)} {what} recordings ({fmt_dur(time.time() - t0)})")
        self.release()

    def has_main(self, p, k):
        return os.path.exists(dc.cache_path(self.lab.a.cache, p, k, "day"))


def raw_day(cache, layer, p, k, mmap=False):
    """[n, 768] raw output of `layer` for one day (float16)."""
    import torch
    path = dc.cache_path(layer_cache(cache, layer), p, k, "day")
    d = torch.load(path, mmap=mmap) if mmap else torch.load(path)
    if "norm" not in d:
        raise RuntimeError(f"{path} has no norms: it is not a layer cache")
    return d


################################################################################
# the model: the top AVES layers + the rerun's classifier
################################################################################

def make_top(aves, n, ctx, lora_r=None):
    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, base, r):
            super().__init__()
            self.base = base
            for prm in self.base.parameters():
                prm.requires_grad_(False)
            self.a = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
            self.b = nn.Parameter(torch.zeros(base.out_features, r))
            self.scale = LORA_ALPHA / r

        def forward(self, x):
            return self.base(x) + (x @ self.a.t() @ self.b.t()) * self.scale

    class TopModel(nn.Module):
        """Layers 13-n .. 12 of AVES, then the unit vector, then the classifier."""

        def __init__(self):
            super().__init__()
            self.n, self.ctx, self.lora_r = n, ctx, lora_r
            self.input_layer = 12 - n
            self.layers = nn.ModuleList(copy.deepcopy(aves.encoder.transformer.layers[12 - n:12]))
            if lora_r:
                for prm in self.layers.parameters():
                    prm.requires_grad_(False)
                for layer in self.layers:
                    att = layer.attention
                    att.q_proj = LoRALinear(att.q_proj, lora_r)
                    att.v_proj = LoRALinear(att.v_proj, lora_r)
            self.head = ex.make_model(ctx)

        def encode(self, x):  # [B, T, 768] raw input-layer output -> [B, T, 768] unit fingerprints
            pb = None
            for layer in self.layers:
                x, pb = layer(x, None, position_bias=pb)
            return unit(x)

        def classify(self, h, idx):  # h [T, 768], idx [K] -> logits [K, 2]
            if self.ctx == 0:
                return self.head(h[idx])
            off = torch.arange(-self.ctx, self.ctx + 1, device=h.device)
            w = (idx[:, None] + off[None, :]).clamp(0, h.shape[0] - 1)
            return self.head(h[w])

        def trainable(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

    return TopModel()


def day_fingerprints(model, raw, device):
    """Fingerprints of a whole day from the cached raw input-layer output, in the same 1-hour
    pieces (with 16 frames of context) the cache was made with."""
    import torch
    n = raw.shape[0]
    out = []
    with torch.inference_mode():
        for a, b, s, e in chunks(n, CHUNK_S * 500 // 320, CONTEXT_FRAMES):
            h = model.encode(raw[a:b].to(device).float().unsqueeze(0))[0]
            out.append(h[s - a:e - a].float().cpu())
    return torch.cat(out) if out else torch.empty(0, 768)


def day_probs_top(model, raw, device):
    h = day_fingerprints(model, raw, device)
    return ex.day_probs(model.head, h, model.ctx, device)


################################################################################
# experiments: a study spec + the layer read ("layer") + what is trained on top ("ft")
################################################################################

_study_name_of = st.name_of


def input_layer(s):
    """The cached layer an experiment reads: the probed layer, or the one below the trained layers."""
    return 12 - s["ft"]["n"] if s.get("ft") else s["layer"]


def ft_text(f):
    if not f:
        return "probe"
    return f"top{f['n']}" if f["mode"] == "top" else f"lora{f['n']}r{f['r']}"


def name_of(s):
    base = _study_name_of(s)
    if "layer" not in s:
        return base
    return f"{base}_L{s['layer']:02d}_{ft_text(s.get('ft'))}"


def describe(s):
    d = st.describe(s)
    f = s.get("ft")
    if not f:
        return f"{d}; classifier on AVES layer {s['layer']}"
    if f["mode"] == "top":
        return f"{d}; AVES layers {13 - f['n']}-12 fine-tuned with the classifier"
    return f"{d}; LoRA rank {f['r']} on the attention of AVES layers {13 - f['n']}-12, with the classifier"


class FTLab(st.Lab):
    """study.Lab with the layer caches and the fine-tuned models."""

    def __init__(self, a, pools, w, study_root, rule, base_name):
        super().__init__(a, pools, w)
        self.main_cache = a.cache
        self.study_root = study_root
        self.fixed_rule = rule
        self.base_name = base_name
        self._aves = None

    # the cache follows the layer an experiment reads
    def at_layer(self, layer):
        self.a.cache = layer_cache(self.main_cache, layer)
        self._val = {}

    def rule_for(self, s):
        c = self.choices()
        if "rules_final" in c and c["rules_final"].get("experiment") == name_of(s):
            return c["rules_final"]["rule"]
        return self.fixed_rule

    def aves(self):
        if self._aves is None:
            self._aves = load_aves(self.a.fake_embedder)
        return self._aves

    def models(self, name):
        import torch
        s = self.spec(name)
        if not s.get("ft"):
            return super().models(name)
        out = []
        for seed in self.seeds:
            f = s["ft"]
            m = make_top(self.aves(), f["n"], s["ctx"], f.get("r") if f["mode"] == "lora" else None)
            m.load_state_dict(torch.load(os.path.join(self.dir(name), f"model_seed{seed}.pth"), map_location="cpu"))
            out.append(m.eval().to(self.device))
        return out

    def probs(self, models, ctx, files):
        if not models or not hasattr(models[0], "encode"):
            return super().probs(models, ctx, files)
        out = [dict() for _ in models]
        layer = models[0].input_layer
        for p, k, _ in files:
            raw = raw_day(self.main_cache, layer, p, k)
            r = (raw["emb"].float() * raw["norm"].float()[:, None])
            for i, m in enumerate(models):
                out[i][(p, k)] = day_probs_top(m, r, self.device)
        return out

    def speed(self, name, s, files, rule):
        if not s.get("ft"):
            return dict(super().speed(name, s, files, rule), aves_layers_run=s["layer"])
        return {"aves_layers_run": 12}

    def train(self, s):
        name = name_of(s)
        if self.trained_ok(name):
            return name
        st.wjson(self.w, os.path.join(self.dir(name), "spec.json"), s)
        mined = mined_frames(self)
        log(f"=== {name}: {describe(s)}; {len(self.seeds)} runs, each until converged")
        t0 = time.time()
        runs = []
        with self.w.open(os.path.join(self.dir(name), "training_log.txt")) as logf:
            if not s.get("ft"):  # probing: the rerun's own training on another layer
                bank, win, y, ho_w, ho_y, info = st.build_sets(self, s, mined)
                logf.write(json.dumps(info) + "\n")
                for seed in self.seeds:
                    path = os.path.join(self.dir(name), f"model_seed{seed}.pth")
                    if os.path.exists(path) and os.path.exists(path + ".json"):
                        runs.append(json.load(open(path + ".json")))
                        continue
                    m = st.train_until_converged(bank, win, y, ho_w, ho_y, s["ctx"], seed, logf, self.device)
                    st.tsave(self.w, path, m.state_dict())
                    st.wjson(self.w, path + ".json", m.run_info)
                    runs.append(m.run_info)
                    log(f"run {seed}: {m.run_info}")
                del bank, win, y, ho_w, ho_y
            else:
                data = FTData(self, s, mined)
                info = data.info
                logf.write(json.dumps(info) + "\n")
                for seed in self.seeds:
                    path = os.path.join(self.dir(name), f"model_seed{seed}.pth")
                    if os.path.exists(path) and os.path.exists(path + ".json"):
                        runs.append(json.load(open(path + ".json")))
                        continue
                    m = train_top(self, s, data, seed, logf)
                    st.tsave(self.w, path, m.state_dict())
                    st.wjson(self.w, path + ".json", m.run_info)
                    runs.append(m.run_info)
                    log(f"run {seed}: {m.run_info}")
        st.wjson(self.w, os.path.join(self.dir(name), "training.json"),
                 {"name": name, "spec": s, "description": describe(s), "data": info, "runs": runs,
                  "train_minutes": round((time.time() - t0) / 60, 1)})
        return name

    def run_at(self, s):
        self.at_layer(input_layer(s))
        return self.run(s)


def mined_frames(lab):
    """The false alarms the rerun's final model was trained on (its last mistake round)."""
    rows = lab.jload(os.path.join(lab.study_root, lab.base_name, "mined.json")) or []
    return {(r["pool"], r["key"]): list(r["frames"]) for r in rows}


################################################################################
# fine-tuning: crops of the cached input layer, the layers above, the classifier
################################################################################

class FTData:
    """Labelled frames of every training day (rumbles 1; gaps, random forest frames, mined
    false alarms and noise clips 0), read from the input layer's cache on demand."""

    def __init__(self, lab, s, mined):
        import numpy as np
        self.lab, self.s = lab, s
        layer = 12 - s["ft"]["n"]
        self.days, pos, neg, held = [], [], [], []
        for p, files in lab.cfg["train"].items():
            for key in files:
                if p == "noise":
                    if s["noise"]:
                        d = raw_day(lab.main_cache, layer, p, key, mmap=True)
                        self.days.append(d)
                        neg += [(len(self.days) - 1, i) for i in range(d["n"])]
                    continue
                labels = lab.pools[p].labels.get(key, [])
                d = raw_day(lab.main_cache, layer, p, key, mmap=True)
                n = d["n"]
                pi, ni = dc.label_frames(labels)
                if (p, key) in lab.heldout:
                    rnd = dc.random_negative_frames(labels, n, key + "-heldout", n=st.HELDOUT_RANDOM_PER_DAY)
                    held.append(((p, key), [i for i in pi if i < n], sorted(set(i for i in list(ni) + list(rnd) if i < n))))
                    continue
                self.days.append(d)
                j = len(self.days) - 1
                if dc.is_usable(labels):
                    pos += [(j, i) for i in pi if i < n]
                    neg += [(j, i) for i in ni if i < n]
                if s["rand"] and dc.POOLS[p]["forest"]:
                    neg += [(j, i) for i in dc.random_negative_frames(labels, n, key, n=s["rand"])]
                neg += [(j, i) for i in mined.get((p, key), []) if i < n]
        self.pos = np.asarray(sorted(set(pos)), dtype=np.int64).reshape(-1, 2)
        self.neg = np.asarray(sorted(set(neg)), dtype=np.int64).reshape(-1, 2)
        self.label = [dict() for _ in self.days]
        for j, i in self.neg.tolist():
            self.label[j][i] = 0
        for j, i in self.pos.tolist():
            self.label[j][i] = 1
        self.held = held
        self.layer = layer
        self.info = {"rumble_frames": int(len(self.pos)), "other_frames": int(len(self.neg)),
                     "training_recordings": len(self.days), "heldout_days": len(held),
                     "heldout_rumble_frames": sum(len(a) for _, a, _ in held),
                     "heldout_other_frames": sum(len(b) for _, _, b in held),
                     "input_layer": layer, "crop_frames": CROP}
        log(f"fine-tuning data: {self.info}")
        if not len(self.pos):
            raise RuntimeError("no rumble frames to fine-tune on")

    def crop(self, rng):
        """One training sequence around a random rumble or other frame (half each)."""
        import numpy as np
        import torch
        pool = self.pos if (rng.random() < 0.5 or not len(self.neg)) else self.neg
        j, c = pool[rng.integers(len(pool))].tolist()
        d = self.days[j]
        n = d["n"]
        t = min(CROP, n)
        a = int(min(max(0, c - t // 2 + int(rng.integers(-t // 4, t // 4 + 1))), n - t))
        x = (d["emb"][a:a + t].float() * d["norm"][a:a + t].float()[:, None])
        lo = 0 if a == 0 else MARGIN
        hi = t if a + t == n else t - MARGIN
        lab = self.label[j]
        if len(lab) < hi - lo:
            idx = [i - a for i in lab if a + lo <= i < a + hi]
        else:
            idx = [i - a for i in range(a + lo, a + hi) if i in lab]
        if c - a not in idx and lo <= c - a < hi:
            idx.append(c - a)
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) > MAX_FRAMES_PER_CROP:
            keep = rng.choice(len(idx), MAX_FRAMES_PER_CROP, replace=False)
            idx = np.unique(np.concatenate([idx[keep], [c - a]]))
        y = np.asarray([lab[a + i] for i in idx], dtype=np.int64)
        return x, torch.from_numpy(idx), torch.from_numpy(y)


def train_top(lab, s, data, seed, logf):
    """Train the top layers (or their LoRA adapters) and the classifier until the average
    precision on the held-out days stops improving; the best check is kept."""
    import numpy as np
    import torch
    f = s["ft"]
    dev = lab.device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_top(lab.aves(), f["n"], s["ctx"], f.get("r") if f["mode"] == "lora" else None)
    head_path = os.path.join(lab.study_root, lab.base_name, f"model_seed{seed}.pth")
    if not os.path.exists(head_path):
        head_path = os.path.join(lab.study_root, lab.base_name, f"model_seed{lab.seeds[0]}.pth")
    model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
    model.to(dev).train()
    layer_params = [p for n, p in model.layers.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": layer_params, "lr": LR_LORA if f["mode"] == "lora" else LR_LAYERS},
                             {"params": model.head.parameters(), "lr": LR_HEAD}], weight_decay=0.01)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    held_raw = {}

    def check():
        model.eval()
        sc, ys = [], []
        for (p, k), pi, ni in data.held:
            if (p, k) not in held_raw:
                d = raw_day(lab.main_cache, data.layer, p, k)
                held_raw[(p, k)] = (d["emb"].float() * d["norm"].float()[:, None]).half()  # half: less RAM
            prob = day_probs_top(model, held_raw[(p, k)], dev)
            sc += [float(prob[i]) for i in pi] + [float(prob[i]) for i in ni]
            ys += [1] * len(pi) + [0] * len(ni)
        model.train()
        if not ys:
            return 0.0
        return st.average_precision(torch.tensor(sc), torch.tensor(ys))

    best, best_check, best_state, since, checks, steps = check(), 0, None, 0, 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    logf.write(f"seed {seed} start: held-out days AP {best:.4f} (the rerun's classifier on unchanged AVES)\n")
    log(f"seed {seed}: held-out AP before fine-tuning {best:.4f}")
    t0 = time.time()
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    while True:
        opt.zero_grad()
        total = 0.0
        for _ in range(ACCUM):
            x, idx, y = data.crop(rng)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                h = model.encode(x.to(dev).unsqueeze(0))[0].float()
                lg = model.classify(h, idx.to(dev)).float()
            yy = y.to(dev)
            l = ce(lg, yy)
            parts = [l[yy == c].mean() for c in (0, 1) if (yy == c).any()]
            loss = sum(parts) / len(parts) / ACCUM  # rumble and other frames weigh the same
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        steps += 1
        if steps % CHECK_STEPS == 0:
            checks += 1
            ap = check()
            better = ap > best + 1e-5
            msg = (f"seed {seed} check {checks} ({steps * ACCUM} crops): loss {total * 100:.2f}, "
                   f"held-out days AP {ap:.4f}{' *' if better else ''}")
            logf.write(msg + "\n")
            log(msg)
            if better:
                best, best_check, since = ap, checks, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                since += 1
            if since >= PATIENCE or checks >= MAX_CHECKS:
                break
    model.load_state_dict(best_state)
    model.eval().cpu()
    model.run_info = {"checks": checks, "best_check": best_check, "crops_seen": steps * ACCUM,
                      "heldout_ap": round(best, 5), "trainable_parameters": model.trainable(),
                      "hit_max": checks >= MAX_CHECKS and since < PATIENCE, "seconds": round(time.time() - t0, 1)}
    logf.write(json.dumps(model.run_info) + "\n")
    return model


################################################################################
# the run
################################################################################

def cmd_all(lab, base_spec):
    lab.report("caching the layers")
    lab.at_layer(12)
    train_files = [(p, k) for p, ks in lab.cfg["train"].items() for k in ks]
    val_files = [(p, k) for p, ks in lab.cfg["validation"]["files"].items() for k in ks]
    layers = sorted(set(PROBE_LAYERS) | {12 - f["n"] for f in FINETUNE})
    caches = LayerCaches(lab, layers, lab.a.fake_embedder)
    caches.make(train_files, "training")
    caches.make(val_files, "validation")

    # A: which layer
    lab.report("A: which AVES layer (probing)")
    items = []
    for l in sorted(set(PROBE_LAYERS) | {12}):
        s = dict(base_spec, layer=l, ft=None)
        r = st.safe(f"probe layer {l}", lab.run_at, s, lab=lab)
        if r:
            items.append((name_of(s), r["summary"], 0 if l == 12 else 1))  # ties stay with layer 12
    if items:
        win, ranked, why = st.choose(items)
        lab.save_choice("probe", {"winner": win, "ranked": ranked, "why": why})

    # C and D: fine-tuning
    for f in FINETUNE:
        lab.report(f"fine-tuning: {ft_text(f)}")
        s = dict(base_spec, layer=12, ft=f)
        st.safe(f"fine-tune {ft_text(f)}", lab.run_at, s, lab=lab)

    # the winner, its own alarm rule
    all_items = []
    for d in sorted(os.listdir(lab.root)):
        s = lab.spec(d) if os.path.isdir(lab.dir(d)) else None
        if not s:
            continue
        r = lab.result(d, lab.fixed_rule)
        if r:
            simple = 0 if d.endswith("_L12_probe") else (1 if not s.get("ft") else (2 if s["ft"]["mode"] == "lora" else 3))
            all_items.append((d, r["summary"], simple))
    if not all_items:
        sys.exit("nothing could be trained; see failed.txt")
    win, ranked, why = st.choose(all_items)
    ws = lab.spec(win)
    lab.at_layer(input_layer(ws))
    rule = st.safe("alarm rules of the winner", lab.tune_rules, win, "rules_final", im.RULE_OPTIONS, lab=lab) \
        or lab.fixed_rule
    lab.save_choice("final", {"winner": win, "ranked": ranked, "why": why, "rule": rule})
    st.safe("scoring the winner with its rule", lab.evaluate, win, rule, lab=lab)


def cmd_tests(lab, base_spec):
    """The rerun's final model (layer 12, from study.py) next to the winner here, on the tests."""
    c = lab.choices()
    if "final" not in c:
        return
    win, rule = c["final"]["winner"], c["final"]["rule"]
    ws = lab.spec(win)
    layer = input_layer(ws)
    lab.at_layer(layer)
    sets = [("sampleB", "Test 1: Sample B"), ("sampleA", "Test 2: Sample A"), ("vehicle", "Vehicle (false alarms only)")]
    tests = {t["key"]: t for t in lab.cfg["tests"] if any(t["files"].values())}
    files = [(p, k) for key, _ in sets if key in tests for p, ks in tests[key]["files"].items() for k in ks]
    if layer != 12:
        LayerCaches(lab, [layer], lab.a.fake_embedder).make(files, "test")
    r = lab.evaluate(win, rule)
    models = lab.models(win)
    rows, out = [], {"winner": win, "rule": rule, "sets": {}}
    for key, title in sets:
        t = tests.get(key)
        if not t:
            continue
        keys = [(p, k) for p, ks in t["files"].items() for k in ks]
        probs = lab.probs(models, ws["ctx"], [(p, k, 0) for p, k in keys])
        fl = [(p, k, len(probs[0][(p, k)]) * FRAME_SECONDS / 3600) for p, k in keys]
        truth = im.Truth(lab.pools, fl)
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
            res[rec] = {m: statistics.mean([x[m] for x in per if x[m] == x[m]] or [float("nan")])
                        for m in ("recall", "precision", "fp_per_hour")}
        out["sets"][key] = res
        rows.append(f"| {title} | " + " | ".join(
            f"{im._f(res[rec]['recall'])} / {im._f(res[rec]['precision'])} / {im._f(res[rec]['fp_per_hour'], 2)}"
            for rec in (80, 85)) + " |")
    st.wjson(lab.w, os.path.join(lab.root, "final.json"), out)
    study_final = os.path.join(lab.study_root, "final.md")
    t = ["# Tests of the fine-tuning winner", "", f"Winner: **{win}** ({describe(ws)}), alarm rule {st.rule_key(rule)}.",
         f"Means over {len(lab.seeds)} runs at the cut-offs that found 80% / 85% of the validation rumbles. "
         "Each cell: recall / precision / false alarms per hour.", "",
         "| Test | at 80% | at 85% |", "|---|---|---|"] + rows + [
        "", f"The rerun's own test results (layer 12, no fine-tuning) are in `{study_final}`, row \"10 final model\"."]
    with lab.w.open(os.path.join(lab.root, "final.md")) as fh:
        fh.write("\n".join(t) + "\n")
    print("\n".join(t))


def write_report(lab):
    rows = []
    for d in sorted(os.listdir(lab.root)):
        s = lab.spec(d) if os.path.isdir(lab.dir(d)) else None
        if not s:
            continue
        r = lab.result(d, lab.rule_for(s)) or lab.result(d, lab.fixed_rule)
        if r:
            rows.append(r)
    if not rows:
        return
    rows.sort(key=lambda r: st.rank_key(r["summary"], 0))
    g = lambda s, k=3: "-" if not s else f"{s[0]:.{k}f} ({s[1]:.{k}f}-{s[2]:.{k}f})"  # noqa: E731
    t = ["# Teaching AVES about rumbles", "",
         f"Starting point: the rerun's final model `{lab.base_name}`. Scored like the rerun (validation days, "
         f"mean false alarms per hour at 80% and 85% recall, lower is better), {len(lab.seeds)} runs each.", "",
         "| Experiment | What | Score | FA/h at 80% | at 85% | at 90% | Max recall | Trainable weights | Train min |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s, sm = r["spec"], r["summary"]
        tp = (r.get("runs") or [{}])[0].get("trainable_parameters") or "classifier only"
        t.append(f"| {r['name']} | {describe(s).split('; ')[-1]} | **{g(sm['score'])}** | {g(sm['fp_at_80'])} | "
                 f"{g(sm['fp_at_85'])} | {g(sm['fp_at_90'])} | {g(sm['max_recall'])} | {tp} | {r.get('train_minutes')} |")
    c = lab.choices()
    if "final" in c:
        t += ["", f"**Winner:** {c['final']['winner']} ({c['final']['why']})"]
    with lab.w.open(os.path.join(lab.root, "finetune.md")) as f:
        f.write("\n".join(t) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=dc.DEFAULT_SOURCE)
    p.add_argument("--out", help="the Configurations folder (default <source>\\Configurations)")
    p.add_argument("--cache", help="the embedding cache (default <out>\\_embedding_cache)")
    p.add_argument("--seeds", type=int, default=SEEDS, help=f"training runs per option (default {SEEDS})")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.out = a.out or os.path.join(a.source, "Configurations")
    a.cache = a.cache or os.path.join(a.out, "_embedding_cache")
    a.deadline, a.no_augmentation, a.round1, a.start_from = None, True, [], None
    im.keep_awake()
    root = dc.find_dataset_root(a.source)
    study_root = os.path.join(a.out, st.ROOT_NAME)
    ft_root = os.path.join(a.out, ROOT_NAME)
    w = dc.Writer([ft_root, os.path.join(a.cache, "_layers")],
                  [dc.ORIGINAL_DATA, os.path.join(root, "Training"), os.path.join(root, "Testing")])
    w.makedirs(ft_root)
    stdout = sys.stdout
    sys.stdout = dc.Tee(w.check(os.path.join(ft_root, "run_log.txt")))
    try:
        sc = json.load(open(os.path.join(study_root, "choices.json"))) if os.path.exists(
            os.path.join(study_root, "choices.json")) else {}
        if "final" not in sc:
            sys.exit(f"the rerun has not finished: {study_root}\\choices.json has no final choice")
        base_name, rule = sc["final"]["winner"], sc["final"]["rule"]
        base = json.load(open(os.path.join(study_root, base_name, "spec.json")))
        base_spec = dict(base, aug=None)
        log(f"teaching AVES about rumbles, starting from {base_name} with rule {st.rule_key(rule)}")
        pools = dc.load_pools(root)
        lab = FTLab(a, pools, w, study_root, rule, base_name)
        lab.root = ft_root
        lab.use_setup(base["setup"])
        lab.at_layer(12)
        # study.Lab's methods name experiments and write their report through these module names
        saved = st.name_of, st.write_report
        st.name_of, st.write_report = name_of, write_report
        try:
            try:
                cmd_all(lab, base_spec)
            except SystemExit:
                raise
            except Exception:
                st.record_failure(lab, "fine-tuning")
            st.safe("tests", cmd_tests, lab, base_spec, lab=lab)
            st.safe("finetune.md", write_report, lab, lab=lab)
            lab.report("finished: see finetune.md and final.md")
        finally:
            st.name_of, st.write_report = saved
    finally:
        sys.stdout = stdout


if __name__ == "__main__":
    main()
