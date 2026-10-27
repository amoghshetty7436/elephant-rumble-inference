#!/usr/bin/env python
"""
Tuning AVES itself for rumbles (the "AVES Tuning Plan"). Run it after the full rerun
(study.py): it starts from the rerun's final model and keeps everything else of it (its
data, listening window, mined false alarms, scoring); only AVES changes.

  Stage 1  which AVES and which layer, AVES frozen: layers 4-12, a learned mix of all
           layers, the mean of the last 4, and the other AVES models (AVES-all,
           BirdAVES-biox-base, BirdAVES-bioxn-large)
  Stage 2  light-touch tuning, AVES's own weights frozen: LoRA (top 4 layers rank 4/8/16,
           all layers rank 8 with the feed-forward too), bottleneck adapters, BitFit, and
           adapters + LoRA + learned layer mix together
  Stage 3  AVES layers trained: top 2/4/6, all 12 with layer-wise learning rates,
           gradual unfreezing, and a shorter AVES (8 or 6 layers). Every model also gets
           WiSE-FT blends with the original AVES (25/50/75%) and its weight average (EMA)
  Stage 4  training recipe on the best two so far: augmentation of the audio (volume,
           noise clips, time masking), focal loss, a second round of false-alarm mining
  Finalists  the best 3 get 3 runs each; the winner (or the rerun's final model, on a
           tie) gets its own alarm rule, then the tests, scored once

Every option starts from the rerun's trained classifier and is trained on the detector's
own 1-hour pieces of audio with AVES running live on the GPU (the 500 Hz audio of every day is
cached first, about 86 MB a day). Options are screened with 1 run each (at most
--run-minutes); an option moves on only if it beats the current best by more than the
rerun's own run-to-run spread. Scored exactly like the rerun.

Writes only to <out>\\_aves_tuning and <cache>\\_audio. Reads the L: drive, never writes
there. A second run resumes; --finish-by stops new work in time for the finalists and
the tests.

  python tools\\aves_tuning.py --quick                          (10-minute smoke test)
  python tools\\aves_tuning.py --finish-by "2026-10-30 09:00"
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
import finetune as ft  # noqa: E402
import improve as im  # noqa: E402
import study as st  # noqa: E402
from dataset_configs import log, fmt_dur, FRAME_SECONDS  # noqa: E402

ROOT_NAME = "_aves_tuning"
HOP = 320                            # audio samples per AVES frame (0.64 s at 500 Hz)
CHUNK_S = 3600                       # AVES runs on 1-hour pieces with 16 frames on each side, as the detector does
CONTEXT_FRAMES = 16
PIECE_FRAMES = CHUNK_S * 500 // HOP  # 5,625 frames: training uses exactly the detector's 1-hour pieces, so a
                                     # fingerprint in training is the one the detector sees (AVES's attention
                                     # spans the whole piece; shorter pieces give other fingerprints)
MAX_FRAMES_PER_PIECE = 1024
ACCUM = 2                            # pieces per optimiser step
CHECK_STEPS = 150                    # optimiser steps between checks on the held-out days
PATIENCE = 6
MAX_CHECKS = 40
HELD_MAX_DAYS = 10                   # held-out days used for the checks (each check runs AVES on all of them)
RUN_MINUTES = 75                     # screening: longest training run
FINAL_RUN_MINUTES = 150              # finalists
FINAL_SEEDS = 3
FINALISTS = 3
FINALIST_RESERVE_S = 10 * 3600       # stages 1-4 stop starting runs this long before --finish-by
TEST_RESERVE_S = 2 * 3600            # finalists stop adding runs this long before it
FIRST_GUESS_S = 60 * 60
DISK_MARGIN_GB = 30
CHECKPOINT_FROM = 6                  # with this many trained AVES layers or more, recompute instead of storing (GPU memory)

LR_HEAD = 1e-4
LR_LAYERS = 1e-5
LR_ALL = 2e-5                        # top layer when all layers train; each lower one x LLRD
LLRD = 0.8
LR_LORA = 1e-4
LR_ADAPTER = 1e-4
LR_BITFIT = 1e-4
LR_MIX = 1e-3
LORA_ALPHA = 16
ADAPTER_DIM = 64
GRADUAL_EVERY = 2                    # gradual unfreezing: one more layer every this many checks
WISE_ALPHAS = [0.25, 0.5, 0.75]
EMA_DECAY = 0.999
FOCAL_GAMMA = 2.0
AUG_GAIN_DB = 6.0
AUG_NOISE_P = 0.5
AUG_SNR_DB = (-5.0, 15.0)
AUG_MASK_P = 0.05                    # share of frames hidden (spans of AUG_MASK_SPAN)
AUG_MASK_SPAN = 5

BACKBONES = {
    "bio": ("ported_aves", "aves-base-bio"),           # the AVES used so far
    "all": ("ported_aves", "aves-base-all"),
    "biox-base": ("birdaves", "birdaves-biox-base"),
    "bioxn-large": ("birdaves", "birdaves-bioxn-large"),
}
WEIGHTS_URL = "https://storage.googleapis.com/esp-public-files"

SIMPLE = {"head": 0, "lora": 1, "bitfit": 1, "adapter": 2, "combo": 2, "layers": 3, "all": 3, "gradual": 3}
SHORT_BONUS = 0.5                    # a shorter AVES wins ties against a full one of the same kind
FINAL_REF = "rerun_final"


################################################################################
# AVES models
################################################################################

def fake_backbone(dim=768, layers=12):
    import torch
    from torchaudio.models import wav2vec2_model
    torch.manual_seed(0)
    return wav2vec2_model(
        extractor_mode="group_norm", extractor_conv_layer_config=[(32, 10, 5)] + [(32, 3, 2)] * 4 + [(32, 2, 2)] * 2,
        extractor_conv_bias=False, encoder_embed_dim=dim, encoder_projection_dropout=0.1, encoder_pos_conv_kernel=16,
        encoder_pos_conv_groups=16, encoder_num_layers=layers, encoder_num_heads=dim // 64, encoder_attention_dropout=0.1,
        encoder_ff_interm_features=64, encoder_ff_interm_dropout=0.0, encoder_dropout=0.1,
        encoder_layer_norm_first=False, encoder_layer_drop=0.0, aux_num_out=None).eval()


def load_backbone(name, fake=False):
    """The wav2vec2 model of an AVES variant (downloaded once into torch's hub folder, on C:)."""
    import torch
    if fake:
        return fake_backbone(1024, 24) if name == "bioxn-large" else ft.load_aves(fake=True)
    from torchaudio.models import wav2vec2_model
    folder, stem = BACKBONES[name]
    d = os.path.join(torch.hub.get_dir(), "fruitpunch_elephants")
    paths = {}
    for kind, suffix in (("config", ".torchaudio.model_config.json"), ("weights", ".torchaudio.pt")):
        dst = os.path.join(d, stem + suffix)
        if not os.path.exists(dst):
            os.makedirs(d, exist_ok=True)
            log(f"downloading {stem + suffix}")
            torch.hub.download_url_to_file(f"{WEIGHTS_URL}/{folder}/{stem}{suffix}", dst)
        paths[kind] = dst
    cfg = json.load(open(paths["config"]))
    cfg["encoder_layer_drop"] = 0.0  # every layer always runs (training would otherwise skip some at random)
    m = wav2vec2_model(**cfg, aux_num_out=None)
    m.load_state_dict(torch.load(paths["weights"], map_location="cpu"))
    return m.eval()


def conv_frames(n):
    """Frames AVES's front end makes from n samples."""
    for k, s in [(10, 5)] + [(3, 2)] * 4 + [(2, 2)] * 2:
        n = (n - k) // s + 1
        if n <= 0:
            return 0
    return n


def day_frames(n_samples):
    """Frames of a whole day as fp_replica.Embedder.whole_file makes them (= the fingerprint cache)."""
    chunk, ctx = CHUNK_S * 500, CONTEXT_FRAMES * HOP
    total = 0
    for s in range(0, n_samples, chunk):
        pre = min(ctx, s)
        cur = min(chunk, n_samples - s)
        post = min(ctx, max(0, n_samples - s - chunk))
        if pre + cur + post < HOP * 2:
            continue
        e = conv_frames(pre + cur + post)
        a, b = pre // HOP, (pre + cur) // HOP
        total += max(0, min(b, e) - a)
        if post == 0 and e > 1:
            total += 1
    return total


################################################################################
# the tuned model: AVES (as the option changes it), the layer read, the classifier
################################################################################

def make_tuned(t, ctx, backbone, shared=False):
    """t: the experiment's "tune" dict. shared=True keeps the backbone object (frozen AVES only)."""
    import torch
    import torch.nn as nn
    import torch.utils.checkpoint  # noqa: F401

    class LoRALinear(nn.Module):
        def __init__(self, base, r):
            super().__init__()
            self.base = base
            self.a = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
            self.b = nn.Parameter(torch.zeros(base.out_features, r))
            self.scale = LORA_ALPHA / r

        def forward(self, x):
            return self.base(x) + (x @ self.a.t() @ self.b.t()) * self.scale

    class Adapted(nn.Module):
        """An AVES layer with a small bottleneck after it (starts as "no change")."""

        def __init__(self, layer, dim):
            super().__init__()
            self.layer = layer
            self.down = nn.Linear(dim, ADAPTER_DIM)
            self.up = nn.Linear(ADAPTER_DIM, dim)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        def forward(self, x, attention_mask=None, position_bias=None, key_padding_mask=None):
            x, pb = self.layer(x, attention_mask, position_bias=position_bias)
            return x + self.up(torch.nn.functional.gelu(self.down(x))), pb

    class Tuned(nn.Module):
        def __init__(self):
            super().__init__()
            self.t, self.ctx = t, ctx
            bb = backbone if shared else copy.deepcopy(backbone)
            for p in bb.parameters():
                p.requires_grad_(False)
            layers = bb.encoder.transformer.layers
            keep = t.get("keep") or len(layers)
            if keep < len(layers) and not shared:  # a shared AVES is only read up to `keep`
                bb.encoder.transformer.layers = nn.ModuleList(list(layers)[:keep])
            self.bb = bb
            self.keep = keep
            self.dim = bb.encoder.feature_projection.projection.out_features
            m, n = t["method"], t.get("n") or keep
            first = keep - n
            L = self.bb.encoder.transformer.layers
            if m in ("lora", "combo"):
                for i in range(first, keep):
                    att = L[i].attention
                    att.q_proj = LoRALinear(att.q_proj, t["r"])
                    att.v_proj = LoRALinear(att.v_proj, t["r"])
                    if t.get("ff") or m == "combo":
                        ffn = L[i].feed_forward
                        ffn.intermediate_dense = LoRALinear(ffn.intermediate_dense, t["r"])
                        ffn.output_dense = LoRALinear(ffn.output_dense, t["r"])
            if m in ("adapter", "combo"):
                for i in range(first, keep):
                    L[i] = Adapted(L[i], self.dim)
            if m == "bitfit":
                for i in range(first, keep):
                    for name, p in L[i].named_parameters():
                        if name.endswith("bias") or "layer_norm" in name:
                            p.requires_grad_(True)
            if m in ("layers", "all", "gradual"):
                for i in range(first, keep):
                    for p in L[i].parameters():
                        p.requires_grad_(True)
            mix = t["mix"]
            self.mixw = nn.Parameter(torch.zeros(keep)) if mix == "wsum" or m == "combo" else None
            self.proj = nn.Linear(self.dim, 768) if self.dim != 768 else None
            self.head = ex.make_model(ctx)
            self.tuned_names = sorted(k for k, p in self.named_parameters() if p.requires_grad)
            self.first_trainable = first if m != "head" else keep
            self.time_mask = False
            if m == "gradual":  # starts with only the top layer
                self.unfrozen = 1
                self.set_gradual(1)

        def set_gradual(self, k):
            L = self.bb.encoder.transformer.layers
            self.unfrozen = min(k, self.keep)
            for i in range(self.keep):
                for p in L[i].parameters():
                    p.requires_grad_(i >= self.keep - self.unfrozen)
            self.first_trainable = self.keep - self.unfrozen

        def fingerprints(self, wave, rng=None):
            """wave [S] -> unit fingerprints [T, 768]. AVES below the first trained layer runs
            without gradients."""
            bb = self.bb
            L = list(bb.encoder.transformer.layers)[:self.keep]
            mix = "wsum" if self.mixw is not None else self.t["mix"]
            want = set(range(self.keep)) if mix == "wsum" else ({self.keep - 4, self.keep - 3, self.keep - 2, self.keep - 1}
                                                              if mix == "last4" else {self.keep - 1})
            outs = {}
            grad_from = self.first_trainable
            with torch.set_grad_enabled(torch.is_grad_enabled() and grad_from == -1):
                x, _ = bb.feature_extractor(wave.view(1, -1), None)
                x = bb.encoder.feature_projection(x)
                if self.time_mask and rng is not None:
                    x = time_mask(x, rng)
                x = bb.encoder.transformer._preprocess(x)
            pb = None
            for i, layer in enumerate(L):
                train_here = torch.is_grad_enabled() and i >= grad_from
                with torch.set_grad_enabled(train_here):
                    if train_here and self.keep - grad_from >= CHECKPOINT_FROM:
                        # many trained layers on a 1-hour piece: keep only each layer's input, recompute the rest
                        x, pb = torch.utils.checkpoint.checkpoint(
                            lambda z, b, f=layer: f(z, None, position_bias=b), x, pb, use_reentrant=False)
                    else:
                        x, pb = layer(x, None, position_bias=pb)
                if i in want:
                    outs[i] = x
            if mix == "wsum":
                w = torch.softmax(self.mixw, 0)
                h = sum(w[i] * outs[i].float() for i in range(self.keep))
            elif mix == "last4":
                h = sum(outs[i].float() for i in sorted(want)) / len(want)
            else:
                h = outs[self.keep - 1].float()
            h = ft.unit(h[0])
            if self.proj is not None:
                h = ft.unit(self.proj(h))
            return h

        def classify(self, h, idx):
            if self.ctx == 0:
                return self.head(h[idx])
            off = torch.arange(-self.ctx, self.ctx + 1, device=h.device)
            w = (idx[:, None] + off[None, :]).clamp(0, h.shape[0] - 1)
            return self.head(h[w])

        def trainable(self):
            return sum(p.numel() for k, p in self.named_parameters() if k in set(self.tuned_names))

        def saved_state(self):
            keys = set(self.tuned_names)
            return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                    if k in keys or k.startswith("head.") or k.startswith("proj.")}

    return Tuned()


def time_mask(x, rng):
    """Hide random spans of frames (as wav2vec 2.0 does when it learns), here with zeros."""
    import torch
    T = x.shape[1]
    n = int(T * AUG_MASK_P / AUG_MASK_SPAN)
    if n <= 0 or T <= AUG_MASK_SPAN:
        return x
    keep = torch.ones(T, device=x.device, dtype=x.dtype)
    for s in rng.integers(0, T - AUG_MASK_SPAN, n).tolist():
        keep[s:s + AUG_MASK_SPAN] = 0
    return x * keep[None, :, None]


def day_fingerprints(model, y, device):
    """Fingerprints of a whole day in the 1-hour pieces (16 frames of context) the detector uses."""
    import torch
    chunk, ctx = CHUNK_S * 500, CONTEXT_FRAMES * HOP
    out = []
    amp = device == "cuda"
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for s in range(0, y.shape[0], chunk):
            pre, cur, post = y[max(0, s - ctx):s], y[s:s + chunk], y[s + chunk:s + chunk + ctx]
            x = torch.cat([pre, cur, post]).float()
            if x.shape[0] < HOP * 2:
                continue
            e = model.fingerprints(x.to(device)).float().cpu()
            a, b = pre.shape[0] // HOP, (pre.shape[0] + cur.shape[0]) // HOP
            out.append(e[a:b])
            if post.shape[0] == 0 and e.shape[0] > 1:
                out.append(e[-2:-1])
    return torch.cat(out) if out else torch.empty(0, 768)


def day_probs(model, y, device):
    return ex.day_probs(model.head, day_fingerprints(model, y, device), model.ctx, device)


################################################################################
# the 500 Hz audio of every day, cached once
################################################################################

def audio_path(cache, p, k):
    return dc.cache_path(os.path.join(cache, "_audio"), p, k, "audio")


class Audio:
    def __init__(self, lab):
        self.lab = lab
        self.cache = lab.a.cache

    def path(self, p, k):
        return audio_path(self.cache, p, k)

    def load(self, p, k, mmap=False):
        import torch
        return torch.load(self.path(p, k), mmap=mmap) if mmap else torch.load(self.path(p, k))

    def make(self, files, what):
        import torch
        todo = [(p, k) for p, k in files if not os.path.exists(self.path(p, k))]
        if not todo:
            return
        need_gb = len(todo) * 0.09
        self.lab.w.makedirs(os.path.join(self.cache, "_audio"))
        free = shutil.disk_usage(os.path.join(self.cache, "_audio")).free / 2**30
        if free < need_gb + DISK_MARGIN_GB:
            raise RuntimeError(f"caching the audio of {len(todo)} {what} recordings needs about {need_gb:.0f} GB "
                               f"(+{DISK_MARGIN_GB} GB margin) but only {free:.0f} GB are free")
        log(f"caching the 500 Hz audio of {len(todo)} {what} recordings (about {need_gb:.0f} GB)")
        t0 = time.time()
        for i, (p, k) in enumerate(todo, 1):
            y = dc.load_audio(self.lab.pools[p].sounds[k]).to(torch.float16)
            path = self.path(p, k)
            self.lab.w.makedirs(os.path.dirname(path))
            st.tsave(self.lab.w, path, y.contiguous())
            if i % 10 == 0 or i == len(todo):
                log(f"audio cached: {i}/{len(todo)} {what} recordings ({fmt_dur(time.time() - t0)})")


################################################################################
# training data: labelled frames of every training day, read from the cached audio
################################################################################

class AudioData:
    """The rerun's training frames (rumbles 1; gaps, random forest frames, mined false alarms
    and noise clips 0) as positions in the cached audio."""

    def __init__(self, lab, s, mined):
        import numpy as np
        self.lab, self.s = lab, s
        au = lab.audio
        self.days, self.keys, pos, neg, held = [], [], [], [], []
        self.noise = []
        for p, files in lab.cfg["train"].items():
            for key in files:
                y = au.load(p, key, mmap=True)
                n = day_frames(y.shape[0])
                if p == "noise":
                    self.noise.append(y)
                    if s["noise"]:
                        self.days.append(y)
                        self.keys.append((p, key))
                        neg += [(len(self.days) - 1, i) for i in range(n)]
                    continue
                labels = lab.pools[p].labels.get(key, [])
                pi, ni = dc.label_frames(labels)
                if (p, key) in lab.heldout:
                    rnd = dc.random_negative_frames(labels, n, key + "-heldout", n=st.HELDOUT_RANDOM_PER_DAY)
                    held.append(((p, key), [i for i in pi if i < n], sorted(set(i for i in list(ni) + list(rnd) if i < n))))
                    continue
                self.days.append(y)
                self.keys.append((p, key))
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
        held.sort(key=lambda h: -len(h[1]))  # the days with most rumbles first
        self.held = held[:HELD_MAX_DAYS]
        self.info = {"rumble_frames": int(len(self.pos)), "other_frames": int(len(self.neg)),
                     "training_recordings": len(self.days), "heldout_days": len(self.held),
                     "heldout_rumble_frames": sum(len(a) for _, a, _ in self.held),
                     "heldout_other_frames": sum(len(b) for _, _, b in self.held)}
        log(f"training data: {self.info}")
        if not len(self.pos):
            raise RuntimeError("no rumble frames to train on")

    def crop(self, rng, augment=False):
        """The detector's 1-hour piece around a random rumble or other frame (half each):
        (audio with its 16 frames of context, frames before the piece, frames in the piece,
        labelled frame positions in the piece, their labels)."""
        import numpy as np
        import torch
        pool = self.pos if (rng.random() < 0.5 or not len(self.neg)) else self.neg
        j, c = pool[rng.integers(len(pool))].tolist()
        y = self.days[j]
        n = day_frames(y.shape[0])
        h0 = (c // PIECE_FRAMES) * PIECE_FRAMES
        chunk, ctx = CHUNK_S * 500, CONTEXT_FRAMES * HOP
        s = h0 * HOP
        pre0 = max(0, s - ctx)
        wave = y[pre0:s + chunk + ctx].float().clone()
        if augment:
            wave = augment_wave(wave, self.noise, rng)
        last = s + chunk >= y.shape[0]
        cnt = min(PIECE_FRAMES, n - h0)
        # frames next to a piece edge see a cut-off listening window in training; the detector sees
        # the neighbouring piece there, so they are left out (except at the day's own ends)
        lo = 0 if h0 == 0 else self.s["ctx"]
        hi = cnt if last else cnt - self.s["ctx"]
        lab = self.label[j]
        idx = [i - h0 for i in lab if h0 + lo <= i < h0 + hi] if len(lab) < hi - lo else \
            [i - h0 for i in range(h0 + lo, h0 + hi) if i in lab]
        idx = np.asarray(sorted(idx), dtype=np.int64)
        if len(idx) > MAX_FRAMES_PER_PIECE:
            idx = np.unique(idx[rng.choice(len(idx), MAX_FRAMES_PER_PIECE, replace=False)])
        yl = np.asarray([lab[h0 + i] for i in idx], dtype=np.int64)
        return wave, (s - pre0) // HOP, cnt, last, torch.from_numpy(idx), torch.from_numpy(yl)


def augment_wave(wave, noise, rng):
    """Volume change, and (half the time) a noise clip mixed in at a random level."""
    import torch
    wave = wave * float(10 ** (rng.uniform(-AUG_GAIN_DB, AUG_GAIN_DB) / 20))
    if noise and rng.random() < AUG_NOISE_P:
        z = noise[int(rng.integers(len(noise)))]
        if z.shape[0] > 0:
            n = wave.shape[0]
            start = int(rng.integers(max(1, z.shape[0])))
            idx = (torch.arange(n) + start) % z.shape[0]
            zz = z[idx].float()
            pw, pz = wave.pow(2).mean().clamp_min(1e-12), zz.pow(2).mean().clamp_min(1e-12)
            snr = rng.uniform(*AUG_SNR_DB)
            wave = wave + zz * torch.sqrt(pw / pz / (10 ** (snr / 10)))
    return wave


################################################################################
# training one option
################################################################################

def param_groups(model):
    t = model.t
    m = t["method"]
    named = dict(model.named_parameters())
    head = [p for k, p in named.items() if k.startswith("head.") or k.startswith("proj.")]
    groups = [{"params": head, "lr": LR_HEAD}]
    if model.mixw is not None:
        groups.append({"params": [model.mixw], "lr": LR_MIX})
    bb = [(k, p) for k, p in named.items() if k.startswith("bb.") and k in set(model.tuned_names)]
    if m in ("all", "gradual"):
        for i in range(model.keep):
            ps = [p for k, p in bb if f"transformer.layers.{i}." in k]
            if ps:
                groups.append({"params": ps, "lr": LR_ALL * LLRD ** (model.keep - 1 - i)})
    elif bb:
        lr = {"lora": LR_LORA, "adapter": LR_ADAPTER, "combo": LR_ADAPTER, "bitfit": LR_BITFIT}.get(m, LR_LAYERS)
        groups.append({"params": [p for _, p in bb], "lr": lr})
    return groups


def head_start(lab, seed):
    import torch
    path = os.path.join(lab.study_root, lab.base_name, f"model_seed{seed}.pth")
    if not os.path.exists(path):
        path = os.path.join(lab.study_root, lab.base_name, f"model_seed{lab.base_seeds[0]}.pth")
    return torch.load(path, map_location="cpu")


def train_tuned(lab, s, data, seed, logf, max_minutes):
    """Train until the held-out days' average precision stops improving (the best check is
    kept), or for at most max_minutes."""
    import numpy as np
    import torch
    t = s["tune"]
    r = t.get("recipe") or {}
    dev = lab.device
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_tuned(t, s["ctx"], lab.backbone(t["backbone"]))
    model.head.load_state_dict(head_start(lab, seed))
    model.to(dev)
    model.bb.eval()  # no dropout inside AVES
    model.head.train()
    opt = torch.optim.AdamW(param_groups(model), weight_decay=0.01)
    amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()
           if k in set(model.tuned_names)} if t["method"] in ("layers", "all", "gradual") else None
    held = {}

    def check():
        model.eval()
        sc, ys = [], []
        for (p, k), pi, ni in data.held:
            if (p, k) not in held:
                held[(p, k)] = lab.audio.load(p, k)
            prob = day_probs(model, held[(p, k)].float(), dev)
            sc += [float(prob[i]) for i in pi if i < len(prob)] + [float(prob[i]) for i in ni if i < len(prob)]
            ys += [1] * sum(1 for i in pi if i < len(prob)) + [0] * sum(1 for i in ni if i < len(prob))
        model.head.train()
        return st.average_precision(torch.tensor(sc), torch.tensor(ys)) if ys else 0.0

    focal = bool(r.get("focal"))
    model.time_mask = bool(r.get("aug"))
    best = check()
    best_state, best_ema = model.saved_state(), None
    since, checks, steps, best_check = 0, 0, 0, 0
    logf.write(f"seed {seed} start: held-out days AP {best:.4f} (the rerun's classifier on this AVES)\n")
    log(f"seed {seed}: held-out AP before training {best:.4f}")
    t0 = time.time()
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    stop = "patience"
    while True:
        opt.zero_grad()
        total = 0.0
        for _ in range(ACCUM):
            wave, a, n, last, idx, y = data.crop(rng, augment=bool(r.get("aug")))
            if not len(idx):
                continue
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                e = model.fingerprints(wave.to(dev), rng)
                h = e[a:a + n]
                if last and e.shape[0] > 1 and h.shape[0] < n:
                    h = torch.cat([h, e[-2:-1]])  # as the detector does at the end of a day
                if h.shape[0] < n:
                    h = torch.cat([h, h[-1:].expand(n - h.shape[0], -1)])
                lg = model.classify(h.float(), idx.to(dev)).float()
            yy = y.to(dev)
            l = ce(lg, yy)
            if focal:
                pt = torch.softmax(lg, 1).gather(1, yy[:, None])[:, 0].detach()
                l = l * (1 - pt) ** FOCAL_GAMMA
            parts = [l[yy == c].mean() for c in (0, 1) if (yy == c).any()]
            loss = sum(parts) / len(parts) / ACCUM  # rumble and other frames weigh the same
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        scaler.step(opt)
        scaler.update()
        steps += 1
        if ema is not None:
            with torch.no_grad():
                sd = model.state_dict()
                for k in ema:
                    ema[k].mul_(EMA_DECAY).add_(sd[k].float(), alpha=1 - EMA_DECAY)
        if steps % CHECK_STEPS == 0:
            checks += 1
            ap = check()
            better = ap > best + 1e-5
            msg = (f"seed {seed} check {checks} ({steps * ACCUM} pieces, {fmt_dur(time.time() - t0)}): "
                   f"loss {total * 100:.2f}, held-out days AP {ap:.4f}{' *' if better else ''}")
            logf.write(msg + "\n")
            log(msg)
            if better:
                best, best_check, since = ap, checks, 0
                best_state = model.saved_state()
                if ema is not None:
                    best_ema = {k: v.detach().cpu().clone() for k, v in ema.items()}
            else:
                since += 1
            if t["method"] == "gradual" and checks % GRADUAL_EVERY == 0 and model.unfrozen < model.keep:
                model.set_gradual(model.unfrozen + 1)
                logf.write(f"gradual: {model.unfrozen} layers now trained\n")
            if since >= PATIENCE:
                break
            if checks >= MAX_CHECKS:
                stop = "max_checks"
                break
            if time.time() - t0 > max_minutes * 60:
                stop = "time"
                break
    info = {"checks": checks, "best_check": best_check, "pieces_seen": steps * ACCUM, "heldout_ap": round(best, 5),
            "trainable_parameters": model.trainable(), "stopped_by": stop, "seconds": round(time.time() - t0, 1),
            "seconds_per_step": round((time.time() - t0) / max(1, steps), 3)}
    if model.mixw is not None:
        import torch as _t
        info["layer_mix"] = [round(float(x), 3) for x in _t.softmax(best_state["mixw"].float(), 0)]
    logf.write(json.dumps(info) + "\n")
    ema_state = None
    if best_ema is not None:
        ema_state = dict(best_state)
        ema_state.update({k: v for k, v in best_ema.items()})
    return best_state, ema_state, info


################################################################################
# false alarms of a tuned model (stage 4's second round of mining)
################################################################################

def mine_tuned(lab, name):
    import numpy as np
    r = lab.result(name)
    rule = r["rule"]
    cuts = [(x.get("at_80") or {}).get("threshold") for x in r["summary"]["per_seed"]]
    cuts = [c for c in cuts if c is not None]
    cut = statistics.mean(cuts) if cuts else 0.5
    old = ft.mined_frames(lab)
    models = lab.models(name)
    mined, n_new = {}, 0
    for p, keys in lab.cfg["train"].items():
        if p == "noise":
            continue
        for key in keys:
            if (p, key) in lab.heldout:
                continue
            y = lab.audio.load(p, key).float()
            prob = np.mean([day_probs(m, y, lab.device) for m in models], axis=0)
            labels = lab.pools[p].labels.get(key, [])
            got = {i: float(prob[i]) for i in old.get((p, key), []) if i < len(prob)}
            for a0, b0, sc in zip(*im.alarms(prob, rule)):
                if sc < cut or sc >= st.MINE_SKIP_ABOVE:
                    continue
                t0, t1 = a0 * FRAME_SECONDS - st.MINE_PAD_S, b0 * FRAME_SECONDS + st.MINE_PAD_S
                if any(l.start < t1 and l.end > t0 for l in labels):
                    continue
                seg = np.arange(int(a0), int(b0) + 1)
                for i in seg[np.argsort(-prob[seg])[:st.MINE_FRAMES_PER_ALARM]].tolist():
                    n_new += i not in got
                    got[i] = float(prob[i])
            if got:
                mined[(p, key)] = sorted(sorted(got, key=lambda i: -got[i])[:st.MINE_MAX_PER_DAY])
    log(f"mined {n_new} new false-alarm frames with {name} at cut-off {cut:.3f}")
    del models
    return mined, n_new


################################################################################
# experiments
################################################################################

def name_of(s):
    t = s.get("tune")
    return t["name"] if t else st.name_of(s)


def what(t):
    bb = {"bio": "AVES", "all": "AVES-all", "biox-base": "BirdAVES-biox-base", "bioxn-large": "BirdAVES-bioxn-large"}[t["backbone"]]
    mix = t["mix"]
    read = {"wsum": "learned mix of all layers", "last4": "mean of the last 4 layers"}.get(mix, f"layer {t.get('keep')}")
    if t.get("derive"):
        d = t["derive"]
        return (f"{d['parent']} blended {int(d['alpha'] * 100)}% with the original AVES (WiSE-FT)" if d["kind"] == "wise"
                else f"{d['parent']}, weight average (EMA)")
    m, n = t["method"], t.get("n")
    body = {"head": "frozen, classifier only",
            "lora": f"LoRA rank {t.get('r')} on {'attention + feed-forward of ' if t.get('ff') else 'attention of '}the top {n} layers",
            "adapter": f"adapters after the top {n} layers",
            "bitfit": f"biases and layer norms of the top {n} layers",
            "combo": f"adapters + LoRA rank {t.get('r')} + learned layer mix, top {n} layers",
            "layers": f"top {n} layers trained",
            "all": f"all {n} layers trained, layer-wise learning rates",
            "gradual": f"all {n} layers, unfrozen one at a time from the top"}[m]
    rec = t.get("recipe") or {}
    extra = [k for k, v in (("audio augmentation", rec.get("aug")), ("focal loss", rec.get("focal")),
                            ("mined again", rec.get("remine"))) if v]
    return f"{bb}, {read}: {body}" + (f" ({', '.join(extra)})" if extra else "")


def simplicity(t):
    if t.get("derive"):
        return SIMPLE.get(t.get("parent_method", "layers"), 3) + 0.25
    v = SIMPLE[t["method"]] + 0.1 * bool(t.get("recipe"))
    full = 24 if t["backbone"] == "bioxn-large" else 12
    return v - (SHORT_BONUS if (t.get("keep") or full) < full else 0) + (0.2 if t["backbone"] != "bio" else 0)


def tune(name, stage, rep, method, **kw):
    t = dict(rep, name=name, stage=stage, method=method)
    t.update(kw)
    return t


def layers_of(backbone):
    return 24 if backbone == "bioxn-large" else 12


class TuneLab(st.Lab):
    def __init__(self, a, pools, w, root, study_root, rule, base_name, base_seeds):
        super().__init__(a, pools, w)
        self.root = root
        self.study_root = study_root
        self.fixed_rule = rule
        self.base_name = base_name
        self.base_seeds = base_seeds
        self.audio = Audio(self)
        self._bb = {}
        self.reserve = FINALIST_RESERVE_S
        self.exp_seconds = []
        for n in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            tr = self.jload(os.path.join(root, n, "training.json"))
            if tr and tr.get("train_minutes") and tr.get("trained_seeds") == 1:
                self.exp_seconds.append(tr["train_minutes"] * 60 + 240)

    def backbone(self, name):
        if name not in self._bb:
            self._bb = {name: load_backbone(name, self.a.fake_embedder)}  # one at a time in memory
        return self._bb[name]

    def per_experiment(self):
        return statistics.mean(self.exp_seconds[-10:]) if self.exp_seconds else FIRST_GUESS_S

    def can_start(self, n_after=0, what="experiment"):
        need = (1 + n_after) * self.per_experiment() + self.reserve
        ok = self.time_left() > need
        if not ok:
            log(f"TIME: not starting {what}: {fmt_dur(max(0, self.time_left()))} left before --finish-by, "
                f"about {fmt_dur(need)} needed")
        return ok

    def rule_for(self, s):
        c = self.choices()
        if "rules_final" in c and c["rules_final"].get("experiment") == name_of(s):
            return c["rules_final"]["rule"]
        return self.fixed_rule

    def trained_ok(self, name):
        s = self.spec(name)
        if s and s["tune"].get("derive"):
            return os.path.exists(os.path.join(self.dir(name), "training.json")) and \
                self.trained_ok(s["tune"]["derive"]["parent"])
        return super().trained_ok(name)

    # ---- models ---------------------------------------------------------------------
    def load_one(self, name, seed):
        import torch
        s = self.spec(name)
        t = s["tune"]
        d = t.get("derive")
        if d:
            parent = self.spec(d["parent"])["tune"]
            m = make_tuned(parent, s["ctx"], self.backbone(parent["backbone"]))
            st0 = {k: v.clone() for k, v in m.state_dict().items()}
            if d["kind"] == "ema":
                sd = torch.load(os.path.join(self.dir(d["parent"]), f"model_seed{seed}.ema.pth"), map_location="cpu")
                m.load_state_dict(sd, strict=False)
            else:
                sd = torch.load(os.path.join(self.dir(d["parent"]), f"model_seed{seed}.pth"), map_location="cpu")
                blend = {}
                for k, v in sd.items():
                    if k.startswith("bb.") and k in st0 and v.dtype.is_floating_point:
                        blend[k] = d["alpha"] * v.float() + (1 - d["alpha"]) * st0[k].float()
                    else:
                        blend[k] = v
                m.load_state_dict(blend, strict=False)
            return m
        shared = t["method"] == "head"
        m = make_tuned(t, s["ctx"], self.backbone(t["backbone"]), shared=shared)
        sd = torch.load(os.path.join(self.dir(name), f"model_seed{seed}.pth"), map_location="cpu")
        missing = [k for k in m.tuned_names if k not in sd]
        if missing:
            raise RuntimeError(f"{name} seed {seed}: saved model lacks {missing[:3]}")
        m.load_state_dict(sd, strict=False)
        return m

    def models(self, name):
        return [self.load_one(name, seed).eval().to(self.device) for seed in self.seeds]

    def probs(self, models, ctx, files):
        out = [dict() for _ in models]
        for p, k, _ in files:
            y = self.audio.load(p, k).float()
            for i, m in enumerate(models):
                out[i][(p, k)] = day_probs(m, y, self.device)
        return out

    def speed(self, name, s, files, rule):
        t = s["tune"]
        src = self.spec(t["derive"]["parent"])["tune"] if t.get("derive") else t
        full = layers_of(src["backbone"])
        keep = src.get("keep") or full
        rel = keep / 12 * (3.3 if src["backbone"] == "bioxn-large" else 1.0)
        return {"aves_layers_run": keep, "aves_time_vs_now": round(rel, 2),
                "extra_inference_cost": "adapters, about 1-2%" if src["method"] in ("adapter", "combo") else "none"}

    # ---- training -------------------------------------------------------------------
    def train(self, s):
        import torch
        name = name_of(s)
        if self.trained_ok(name):
            return name
        st.wjson(self.w, os.path.join(self.dir(name), "spec.json"), s)
        t = s["tune"]
        if t.get("derive"):
            parent = self.spec(t["derive"]["parent"])
            self.train(parent)
            st.wjson(self.w, os.path.join(self.dir(name), "training.json"),
                     {"name": name, "spec": s, "description": what(t), "runs": [], "train_minutes": 0,
                      "trained_seeds": len(self.seeds)})
            return name
        mined = ft.mined_frames(self)
        n_new = 0
        if (t.get("recipe") or {}).get("remine"):
            mp = os.path.join(self.dir(name), "mined.json")
            if os.path.exists(mp):
                mined = {(r["pool"], r["key"]): r["frames"] for r in json.load(open(mp))}
            else:
                mined, n_new = mine_tuned(self, t["recipe"]["remine"])
                st.wjson(self.w, mp, [{"pool": p, "key": k, "frames": v} for (p, k), v in sorted(mined.items())])
        log(f"=== {name}: {what(t)}; {len(self.seeds)} run(s)")
        t0 = time.time()
        data = AudioData(self, s, mined)
        runs, trained = [], 0
        max_min = self.a.final_run_minutes if len(self.seeds) > 1 else self.a.run_minutes
        with self.w.open(os.path.join(self.dir(name), "training_log.txt"), "a") as logf:
            logf.write(json.dumps(dict(data.info, newly_mined=n_new)) + "\n")
            for seed in self.seeds:
                path = os.path.join(self.dir(name), f"model_seed{seed}.pth")
                if os.path.exists(path) and os.path.exists(path + ".json"):
                    runs.append(json.load(open(path + ".json")))
                    continue
                state, ema_state, info = train_tuned(self, s, data, seed, logf, max_min)
                st.tsave(self.w, path, state)
                if ema_state is not None:
                    st.tsave(self.w, os.path.join(self.dir(name), f"model_seed{seed}.ema.pth"), ema_state)
                st.wjson(self.w, path + ".json", info)
                runs.append(info)
                trained += 1
                log(f"run {seed}: {info}")
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        st.wjson(self.w, os.path.join(self.dir(name), "training.json"),
                 {"name": name, "spec": s, "description": what(t), "data": data.info, "runs": runs,
                  "train_minutes": round((time.time() - t0) / 60, 1), "trained_seeds": trained})
        return name

    def run(self, s, n_after=0):
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
            if trained_now and not s["tune"].get("derive"):
                self.exp_seconds.append(time.time() - t0)
            return r
        except Exception:
            st.record_failure(self, name)
            return None

    def result_of(self, name):
        return self.result(name, self.rule_for(self.spec(name))) if self.spec(name) else None


################################################################################
# the stages
################################################################################

def ref_result(lab):
    """The rerun's final model, as the rerun scored it (5 runs)."""
    d = os.path.join(lab.study_root, lab.base_name)
    r = lab.jload(os.path.join(d, f"results_{st.rule_key(lab.fixed_rule)}.json"))
    return r


def beats(r, inc, margin):
    """Does result r beat the incumbent by more than the margin?"""
    if not r or not r["summary"]["score"]:
        return False
    if not inc or not inc["summary"]["score"]:
        return True
    return r["summary"]["score"][0] < inc["summary"]["score"][0] - margin


def best_of(results):
    ok = [r for r in results if r and r["summary"]["score"]]
    return min(ok, key=lambda r: r["summary"]["score"][0]) if ok else None


def run_stage(lab, key, label, specs):
    lab.report(f"{label}: {len(specs)} options")
    results = []
    for i, s in enumerate(specs):
        r = lab.run(s, n_after=len(specs) - i - 1)
        if r:
            results.append(r)
    lab.save_choice(key, {"winner": (best_of(results) or {}).get("name"), "options": [name_of(s) for s in specs],
                          "done": [r["name"] for r in results],
                          "why": "lowest score of the stage (1 run each)"})
    return results


def stage_specs(base, quick):
    rep0 = {"backbone": "bio", "mix": "layer", "keep": 12}
    s1 = [tune("s1_bio_L12_head", 1, rep0, "head")]
    s1 += [tune(f"s1_bio_L{l:02d}_head", 1, dict(rep0, keep=l), "head") for l in (10, 8, 6, 4)]
    s1 += [tune("s1_bio_wsum_head", 1, dict(rep0, mix="wsum"), "head"),
           tune("s1_bio_last4_head", 1, dict(rep0, mix="last4"), "head"),
           tune("s1_all_L12_head", 1, dict(rep0, backbone="all"), "head"),
           tune("s1_bioxbase_L12_head", 1, dict(rep0, backbone="biox-base"), "head"),
           tune("s1_bioxnlarge_L24_head", 1, dict(rep0, backbone="bioxn-large", keep=24), "head")]
    if quick:
        s1 = [s1[0], s1[5]]
    return [dict(base, tune=t) for t in s1]


def rep_tag(rep):
    bb = rep["backbone"].replace("-", "")
    return f"{bb}_{'L%02d' % rep['keep'] if rep['mix'] == 'layer' else rep['mix']}"


def s2_specs(base, rep, quick):
    k, tag = rep["keep"], rep_tag(rep)
    ts = [tune(f"s2_{tag}_lora4r4", 2, rep, "lora", n=min(4, k), r=4),
          tune(f"s2_{tag}_lora4r8", 2, rep, "lora", n=min(4, k), r=8),
          tune(f"s2_{tag}_lora4r16", 2, rep, "lora", n=min(4, k), r=16),
          tune(f"s2_{tag}_loraAllr8ff", 2, rep, "lora", n=k, r=8, ff=True),
          tune(f"s2_{tag}_adapters", 2, rep, "adapter", n=k),
          tune(f"s2_{tag}_bitfit", 2, rep, "bitfit", n=k),
          tune(f"s2_{tag}_combo", 2, rep, "combo", n=k, r=8)]
    if quick:
        ts = [ts[1]]
    return [dict(base, tune=t) for t in ts]


def s3_specs(base, rep, quick):
    k, tag = rep["keep"], rep_tag(rep)
    ts = [tune(f"s3_{tag}_top2", 3, rep, "layers", n=min(2, k)),
          tune(f"s3_{tag}_top4", 3, rep, "layers", n=min(4, k)),
          tune(f"s3_{tag}_top6", 3, rep, "layers", n=min(6, k)),
          tune(f"s3_{tag}_all", 3, rep, "all", n=k),
          tune(f"s3_{tag}_gradual", 3, rep, "gradual", n=k)]
    if rep["mix"] == "layer":
        for keep in (8, 6):
            if keep < k:
                short = dict(rep, keep=keep)
                ts.append(tune(f"s3_{rep_tag(short)}_short_all", 3, short, "all", n=keep))
    if quick:
        ts = [ts[0]]
    return [dict(base, tune=t) for t in ts]


def derived_specs(base, parent_spec, quick):
    p = parent_spec["tune"]
    out = []
    for alpha in ([0.5] if quick else WISE_ALPHAS):
        out.append(dict(base, ctx=parent_spec["ctx"], tune=dict(
            name=f"{p['name']}_wise{int(alpha * 100)}", stage=p["stage"], backbone=p["backbone"], mix=p["mix"],
            keep=p.get("keep"), method="derived", parent_method=p["method"],
            derive={"kind": "wise", "alpha": alpha, "parent": p["name"]})))
    if not quick:
        out.append(dict(base, ctx=parent_spec["ctx"], tune=dict(
            name=f"{p['name']}_ema", stage=p["stage"], backbone=p["backbone"], mix=p["mix"], keep=p.get("keep"),
            method="derived", parent_method=p["method"], derive={"kind": "ema", "alpha": 1.0, "parent": p["name"]})))
    return out


def s4_specs(base, lab, ranked, quick):
    out = []
    for i, r in enumerate(ranked[:1 if quick else 2]):
        t = dict(r["spec"]["tune"])
        recipes = [("aug", {"aug": True})] + ([] if quick else [("remine", {"remine": t["name"]})])
        if i == 0 and not quick:
            recipes.append(("focal", {"focal": True}))
        for tag, rec in recipes:
            out.append(dict(base, tune=dict(t, name=f"s4_{t['name'][3:]}_{tag}", stage=4, recipe=rec)))
    return out


def cmd_all(lab, base):
    ref = ref_result(lab)
    margin = ref["summary"]["score"][3] if ref and ref["summary"]["score"] else 0.0
    log(f"the rerun's final model: {st.fmt_score(ref['summary']) if ref else 'no result found'}; "
        f"an option moves on if it beats the best so far by more than {margin:.3f} (its run-to-run spread)")
    inc = ref
    quick = lab.a.quick

    # stage 1: which AVES and which layer
    r1 = run_stage(lab, "stage1", "Stage 1: which AVES, which layer (AVES frozen)", stage_specs(base, quick))
    w1 = best_of(r1)
    rep = {"backbone": "bio", "mix": "layer", "keep": 12}
    if w1 and beats(w1, inc, margin):
        t = w1["spec"]["tune"]
        rep = {"backbone": t["backbone"], "mix": t["mix"], "keep": t["keep"]}
        inc = w1
    lab.save_choice("representation", {"winner": rep_tag(rep), "rep": rep,
                                       "why": f"stage 1 winner {w1 and w1['name']}" if rep_tag(rep) != "bio_L12"
                                       else "nothing beat AVES layer 12 by more than the spread"})

    # stage 2 and 3
    r2 = run_stage(lab, "stage2", "Stage 2: light-touch tuning", s2_specs(base, rep, quick))
    r3 = run_stage(lab, "stage3", "Stage 3: AVES layers trained", s3_specs(base, rep, quick))
    derived = []
    for r in r3:
        derived += derived_specs(base, r["spec"], quick)
    r3d = run_stage(lab, "stage3_blends", "Stage 3: WiSE-FT blends and weight averages", derived)

    # stage 4: the recipe on the best two trainable options so far
    trainable = sorted([r for r in r2 + r3 if r["summary"]["score"]], key=lambda r: r["summary"]["score"][0])
    r4 = run_stage(lab, "stage4", "Stage 4: training recipe", s4_specs(base, lab, trainable, quick)) if trainable else []

    # finalists
    lab.reserve = TEST_RESERVE_S
    pool = [r for r in r1 + r2 + r3 + r3d + r4 if r and r["summary"]["score"]]
    pool.sort(key=lambda r: r["summary"]["score"][0])
    finalists = [r["name"] for r in pool[:FINALISTS]]
    lab.save_choice("finalists", {"winner": ", ".join(finalists), "finalists": finalists,
                                  "why": f"the best {FINALISTS} screened options"})
    return finalists


def cmd_finalists(lab, finalists, seeds):
    one = lab.seeds
    lab.seeds = seeds
    lab.report(f"Finalists: {len(finalists)} options, {len(seeds)} runs each")
    results = []
    for i, name in enumerate(finalists):
        s = lab.spec(name)
        r = lab.result(name, lab.fixed_rule)
        if not r:
            # a finalist adds len(seeds) - 1 runs; per_experiment() is the time of one screening run
            more = len(seeds) - 1
            if not lab.trained_ok(name) and not lab.can_start((len(finalists) - i) * more - 1, f"{name} (more runs)"):
                continue
            try:
                lab.train(s)
                r = lab.evaluate(name, lab.fixed_rule)
            except Exception:
                st.record_failure(lab, f"{name} (more runs)")
                r = None
        if r:
            results.append(r)
    items = [(r["name"], r["summary"], simplicity(r["spec"]["tune"])) for r in results]
    ref = ref_result(lab)
    if ref:
        items.append((FINAL_REF, ref["summary"], -10))  # the rerun's final model wins any tie
    if not items:
        lab.seeds = one
        return None
    win, ranked, why = st.choose(items)
    lab.save_choice("final", {"winner": win, "ranked": ranked, "why": why, "seeds": seeds})
    if win != FINAL_REF:
        rule = st.safe("alarm rules of the winner", lab.tune_rules, win, "rules_final", im.RULE_OPTIONS, lab=lab) \
            or lab.fixed_rule
        c = lab.choices()
        c["final"]["rule"] = rule
        st.wjson(lab.w, os.path.join(lab.root, "choices.json"), c)
    return win


def cmd_tests(lab):
    c = lab.choices()
    if "final" not in c or c["final"]["winner"] == FINAL_REF:
        msg = ["# Tests of the AVES tuning winner", "",
               "No tuned AVES beat the rerun's final model by more than its run-to-run spread, so the final model "
               f"stays. Its test results are in `{os.path.join(lab.study_root, 'final.md')}`."]
        with lab.w.open(os.path.join(lab.root, "final.md")) as fh:
            fh.write("\n".join(msg) + "\n")
        print("\n".join(msg))
        return
    win, rule = c["final"]["winner"], c["final"].get("rule") or lab.fixed_rule
    lab.seeds = c["final"]["seeds"]
    ws = lab.spec(win)
    sets = [("sampleB", "Test 1: Sample B"), ("sampleA", "Test 2: Sample A"), ("vehicle", "Vehicle (false alarms only)")]
    tests = {t["key"]: t for t in lab.cfg["tests"] if any(t["files"].values())}
    files = [(p, k) for key, _ in sets if key in tests for p, ks in tests[key]["files"].items() for k in ks]
    lab.audio.make(files, "test")
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
    t = ["# Tests of the AVES tuning winner", "", f"Winner: **{win}** ({what(ws['tune'])}), alarm rule {st.rule_key(rule)}.",
         f"Means over {len(lab.seeds)} runs at the cut-offs that found 80% / 85% of the validation rumbles. "
         "Each cell: recall / precision / false alarms per hour.", "",
         "| Test | at 80% | at 85% |", "|---|---|---|"] + rows + [
        "", f"The rerun's final model on the same tests: `{os.path.join(lab.study_root, 'final.md')}`, row \"10 final model\"."]
    with lab.w.open(os.path.join(lab.root, "final.md")) as fh:
        fh.write("\n".join(t) + "\n")
    print("\n".join(t))


def write_report(lab):
    if not os.path.isdir(lab.root):
        return
    rows = []
    for d in sorted(os.listdir(lab.root)):
        s = lab.spec(d) if os.path.isdir(lab.dir(d)) else None
        if not s or "tune" not in s:
            continue
        r = lab.jload(os.path.join(lab.dir(d), f"results_{st.rule_key(lab.rule_for(s))}.json")) or \
            lab.jload(os.path.join(lab.dir(d), f"results_{st.rule_key(lab.fixed_rule)}.json"))
        if r:
            rows.append(r)
    ref = ref_result(lab)
    g = lambda s, k=3: "-" if not s else f"{s[0]:.{k}f}"  # noqa: E731
    t = ["# Tuning AVES for rumbles", "",
         f"Starting point: the rerun's final model `{lab.base_name}`. Scored like the rerun (validation days, mean false "
         "alarms per hour at 80% and 85% recall, lower is better). Screening uses 1 run per option; finalists 3.", "",
         "| Stage | Experiment | What | Runs | Score | FA/h at 80% | at 85% | at 90% | Max recall | Trained weights "
         "| AVES time vs now | Train min |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    if ref:
        sm = ref["summary"]
        t.append(f"| ref | {FINAL_REF} | the rerun's final model (AVES frozen, layer 12) | {sm['runs']} | **{g(sm['score'])}** "
                 f"| {g(sm['fp_at_80'])} | {g(sm['fp_at_85'])} | {g(sm['fp_at_90'])} | {g(sm['max_recall'])} "
                 f"| classifier only | 1.0 | - |")
    rows.sort(key=lambda r: (r["spec"]["tune"]["stage"], st.rank_key(r["summary"], 0)))
    for r in rows:
        tt, sm = r["spec"]["tune"], r["summary"]
        tp = (r.get("runs") or [{}])[0].get("trainable_parameters", "-") if r.get("runs") else "-"
        sp = r.get("speed") or {}
        t.append(f"| {tt['stage']} | {r['name']} | {what(tt)} | {sm['runs']} | **{g(sm['score'])}** | {g(sm['fp_at_80'])} "
                 f"| {g(sm['fp_at_85'])} | {g(sm['fp_at_90'])} | {g(sm['max_recall'])} | {tp} "
                 f"| {sp.get('aves_time_vs_now', '-')} | {r.get('train_minutes', '-')} |")
    c = lab.choices()
    for k in ("representation", "finalists", "final"):
        if k in c:
            t += ["", f"**{k}:** {c[k].get('winner')} ({c[k].get('why')})"]
    with lab.w.open(os.path.join(lab.root, "aves_tuning.md")) as f:
        f.write("\n".join(t) + "\n")


################################################################################
# main
################################################################################

def quick_setup(lab):
    """--quick: a few training days (with their held-out days), 3 validation days, short
    checks; only to see that every kind of model runs and how fast a training step is."""
    global CHECK_STEPS, PATIENCE, MAX_CHECKS, HELD_MAX_DAYS
    CHECK_STEPS, PATIENCE, MAX_CHECKS, HELD_MAX_DAYS = 10, 1, 2, 2
    v = lab.cfg["validation"]["files"]
    keep, n = {}, 0
    for p, ks in v.items():
        keep[p] = ks[:max(0, 3 - n)]
        n += len(keep[p])
    train = {}
    for p, ks in lab.cfg["train"].items():
        held = [k for k in ks if (p, k) in lab.heldout][:1]
        train[p] = [k for k in ks if (p, k) not in lab.heldout][:3] + held
    lab.cfg = dict(lab.cfg, train=train, validation=dict(lab.cfg["validation"], files=keep))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=dc.DEFAULT_SOURCE)
    p.add_argument("--out", help="the Configurations folder (default <source>\\Configurations)")
    p.add_argument("--cache", help="the embedding cache (default <out>\\_embedding_cache)")
    p.add_argument("--finish-by", help="stop starting new work in time, e.g. '2026-10-30 09:00' ('none': no limit)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-minutes", type=float, default=RUN_MINUTES, help="longest screening run")
    p.add_argument("--final-run-minutes", type=float, default=FINAL_RUN_MINUTES, help="longest finalist run")
    p.add_argument("--quick", action="store_true", help="a short smoke test into _aves_tuning_quick")
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.out = a.out or os.path.join(a.source, "Configurations")
    a.cache = a.cache or os.path.join(a.out, "_embedding_cache")
    a.deadline = None if not a.finish_by or a.finish_by.lower() == "none" else st.parse_deadline(a.finish_by)
    a.seeds = 1
    if a.quick:
        a.run_minutes = min(a.run_minutes, 3)
        a.deadline = None
    im.keep_awake()
    root = dc.find_dataset_root(a.source)
    study_root = os.path.join(a.out, st.ROOT_NAME)
    tr_root = os.path.join(a.out, ROOT_NAME + ("_quick" if a.quick else ""))
    w = dc.Writer([tr_root, os.path.join(a.cache, "_audio")],
                  [dc.ORIGINAL_DATA, os.path.join(root, "Training"), os.path.join(root, "Testing")])
    w.makedirs(tr_root)
    stdout = sys.stdout
    sys.stdout = dc.Tee(w.check(os.path.join(tr_root, "run_log.txt")))
    saved = st.name_of, st.write_report
    try:
        sc = json.load(open(os.path.join(study_root, "choices.json"))) if os.path.exists(
            os.path.join(study_root, "choices.json")) else {}
        if "final" not in sc:
            sys.exit(f"the rerun has not finished: {study_root}\\choices.json has no final choice")
        base_name, rule = sc["final"]["winner"], sc["final"]["rule"]
        base = json.load(open(os.path.join(study_root, base_name, "spec.json")))
        base_spec = dict(base, aug=None)
        base_seeds = sorted(int(f[len("model_seed"):-4]) for f in os.listdir(os.path.join(study_root, base_name))
                            if f.startswith("model_seed") and f.endswith(".pth"))
        log(f"tuning AVES, starting from {base_name} with rule {st.rule_key(rule)}; writing to {tr_root}"
            + (f"; finishing by {time.strftime('%a %d %b %H:%M', time.localtime(a.deadline))}" if a.deadline else ""))
        pools = dc.load_pools(root)
        lab = TuneLab(a, pools, w, tr_root, study_root, rule, base_name, base_seeds)
        lab.use_setup(base["setup"])
        if a.quick:
            quick_setup(lab)
        st.name_of, st.write_report = name_of, write_report
        train_files = [(p, k) for p, ks in lab.cfg["train"].items() for k in ks]
        val_files = [(p, k) for p, ks in lab.cfg["validation"]["files"].items() for k in ks]
        lab.report("caching the audio")
        lab.audio.make(train_files, "training")
        lab.audio.make(val_files, "validation")
        finalists = []
        try:
            finalists = cmd_all(lab, base_spec)
        except SystemExit:
            raise
        except Exception:
            st.record_failure(lab, "the stages")
        if not a.quick:
            if not finalists:
                finalists = (lab.choices().get("finalists") or {}).get("finalists") or []
            st.safe("finalists", cmd_finalists, lab, finalists, list(range(a.seed, a.seed + FINAL_SEEDS)), lab=lab)
            st.safe("tests", cmd_tests, lab, lab=lab)
        st.safe("aves_tuning.md", write_report, lab, lab=lab)
        if a.quick:
            sps = [json.load(open(os.path.join(lab.dir(d), f))) for d in os.listdir(tr_root) if os.path.isdir(lab.dir(d))
                   for f in os.listdir(lab.dir(d)) if f.startswith("model_seed") and f.endswith(".pth.json")]
            for x in sps:
                log(f"seconds per training step: {x.get('seconds_per_step')} ({x.get('trainable_parameters')} weights trained)")
        lab.report("finished: see aves_tuning.md and final.md")
    finally:
        st.name_of, st.write_report = saved
        sys.stdout = stdout


if __name__ == "__main__":
    main()
