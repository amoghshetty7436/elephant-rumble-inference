"""
Dasheng-base fine-tuned with LoRA (Hu et al. 2021) together with the final model's classifier,
for round2.py. Inference costs the same as frozen Dasheng (LoRA merges into the weights),
plus about a third for overlapping blocks.

  data   82 s clips (128 steps of 0.64 s) of 1 kHz audio from the training days: around
         labelled rumbles and at random places, plus the noise clips (drawn `noise` times as
         often); every step labelled rumble / not / ignored (next to a rumble's edge, padding,
         the first and last `ctx` steps of a clip). The held-out training days give the
         clips for early stopping, as everywhere else.
  model  Dasheng-base (frozen) + LoRA rank 8 on every attention and MLP layer; its
         fingerprints (unit length, as stored for round 1) go through the final model's
         classifier (17 steps, 10.9 s) at every step of the clip.
  train  AdamW (LoRA 2e-4, classifier 1e-3), class-balanced cross-entropy, checked on the
         held-out clips by average precision, best check kept.
  infer  blocks of 128 steps every 96 steps, keeping the middle 96 of each.
"""
import copy
import hashlib
import json
import math
import os
import time

import numpy as np

import candidates as cd
import dataset_configs as dc
import experiments as ex
import study as st
from dataset_configs import log

SR = 1000
STEP = 640                     # samples per 0.64 s step at 1 kHz
CLIP = 128                     # steps per training clip and inference block (82 s)
HOP = 96                       # inference: a block every 96 steps, middle 96 kept
RUMBLE_CLIPS_PER_DAY = 40
RANDOM_CLIPS_PER_DAY = 20
NOISE_CLIPS_MAX = 2000         # at most this many clips from the noise recordings
RANK, ALPHA = 8, 16
LR_LORA, LR_HEAD = 2e-4, 1e-3
BATCH = 8
CHECK_EVERY = 2000             # clips between checks on the held-out clips
PATIENCE = 6
MAX_CHECKS = 60
TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def lora_linear(base, r=RANK, alpha=ALPHA):
    import torch
    import torch.nn as nn

    class LoRA(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            for p in self.base.parameters():
                p.requires_grad_(False)
            self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.scale = alpha / r
            self.drop = nn.Dropout(0.05)

        def forward(self, x):
            return self.base(x) + (self.drop(x) @ self.lora_a.t() @ self.lora_b.t()) * self.scale

    return LoRA()


def add_lora(model):
    """Freeze the encoder and put LoRA adapters on its attention and MLP layers."""
    for p in model.parameters():
        p.requires_grad_(False)
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if any(full.endswith(t) for t in TARGETS) and child.__class__.__name__ == "Linear":
                setattr(mod, child_name, lora_linear(child))
    return model


def make_encoder(fake):
    if fake:  # the tests: a tiny random Dasheng, nothing downloaded
        from dasheng.pretrained.pretrained import Dasheng
        return Dasheng(embed_dim=64, depth=2, num_heads=4)
    from dasheng import dasheng_base
    return dasheng_base()


def make_ft(ctx, fake):
    import torch
    import torch.nn as nn
    enc = add_lora(make_encoder(fake))
    dim = enc.embed_dim if hasattr(enc, "embed_dim") else 768

    class FT(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc
            self.head = cd.make_ctx_model(dim, ctx)
            self.ctx, self.dim = ctx, dim

        def tokens(self, audio):  # [B, CLIP*STEP] -> [B, CLIP, dim], unit length
            self.enc.eval()  # batch-norm statistics and dropout of the encoder stay as pretrained
            e = self.enc(audio)
            return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        def forward(self, audio):  # -> [B, CLIP, 2]
            h = self.head.frame_features(self.tokens(audio))
            pad = torch.cat([h[:, :1].expand(-1, self.ctx, -1), h, h[:, -1:].expand(-1, self.ctx, -1)], 1)
            win = pad.unfold(1, 2 * self.ctx + 1, 1).transpose(2, 3)  # [B, L, W, d]
            b, l = win.shape[:2]
            return self.head.head(win.reshape(b * l, -1)).reshape(b, l, 2)

    return FT()


def trainable_state(model):
    return {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k or k.startswith("head.")}


def load_ft(path, ctx, fake, device):
    import torch
    m = make_ft(ctx, fake)
    m.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    return m.eval().to(device)


################################################################################
# clips
################################################################################

def day_labels(labels, n):
    y = np.zeros(n, np.int8)
    pi, _ = dc.label_frames(labels)
    pi = np.asarray([i for i in pi if i < n], dtype=np.int64)
    if len(pi):
        y[pi] = 1
        edge = np.unique(np.clip(np.concatenate([pi - 1, pi + 1]), 0, n - 1))
        y[edge[y[edge] == 0]] = -1
    return y


def build_clips(lab, folder):
    """Writes <folder>/clips.f16 ([N, CLIP*STEP] audio), labels.npy ([N, CLIP] int8) and
    meta.json (part and weight per clip). Done once; resumes are skipped."""
    meta_path = os.path.join(folder, "meta.json")
    if os.path.exists(meta_path):
        return
    import fp_replica
    cfg, pools, held, s = lab.lab.cfg, lab.pools, lab.lab.heldout, lab.base_spec
    ctx = s["ctx"]
    lab.w.makedirs(folder)
    raw = os.path.join(folder, "clips.f16")
    ys, parts, weights = [], [], []
    keys = [(p, k) for p, ks in cfg["train"].items() for k in ks]
    noise_keys = [pk for pk in keys if pk[0] == "noise"]
    noise_w = float(s["noise"] or 0)
    with open(lab.w.check(raw + ".tmp"), "wb") as f:
        for i, (p, k) in enumerate(keys):
            if p == "noise" and not noise_w:
                continue
            if i % 20 == 0:
                lab.report(f"dasheng_ft: audio clips of training recording {i + 1}/{len(keys)}")
            y = fp_replica.load_audio(pools[p].sounds[k], sr_out=SR).numpy()
            n = max(1, len(y) // STEP)
            r = np.random.default_rng(int(hashlib.md5(k.encode()).hexdigest(), 16) % 2**32)
            if p == "noise":
                lab_full = np.zeros(n, np.int8)
                per = max(1, NOISE_CLIPS_MAX // max(1, len(noise_keys)))
                starts = list(range(0, n, CLIP))[:per]
            else:
                labels = pools[p].labels.get(k, [])
                if not dc.is_usable(labels) and not dc.POOLS[p]["forest"]:
                    continue
                lab_full = day_labels(labels, n)
                starts = []
                if len(labels):
                    centers = [dc.fp_data.time_to_index(l.start + l.duration / 2) for l in labels]
                    pick = r.choice(len(centers), min(len(centers), RUMBLE_CLIPS_PER_DAY), replace=False)
                    starts += [min(max(0, centers[j] - int(r.integers(ctx + 4, CLIP - ctx - 4))), max(0, n - CLIP))
                               for j in pick]
                starts += list(r.integers(0, max(1, n - CLIP), RANDOM_CLIPS_PER_DAY))
            part = "held" if (p, k) in held else "train"
            for a in sorted(set(int(v) for v in starts)):
                seg = np.zeros(CLIP * STEP, np.float16)
                piece = y[a * STEP:(a + CLIP) * STEP]
                seg[:len(piece)] = piece
                lab_c = np.full(CLIP, -1, np.int8)
                m = min(CLIP, n - a)
                lab_c[:m] = lab_full[a:a + m]
                lab_c[:ctx] = -1
                lab_c[CLIP - ctx:] = -1
                f.write(seg.tobytes())
                ys.append(lab_c)
                parts.append(part)
                weights.append(noise_w if p == "noise" else 1.0)
    os.replace(lab.w.check(raw + ".tmp"), lab.w.check(raw))
    with open(lab.w.check(os.path.join(folder, "labels.npy")), "wb") as f:
        np.save(f, np.stack(ys) if ys else np.zeros((0, CLIP), np.int8))
    st.wjson(lab.w, meta_path, {"parts": parts, "weights": weights, "clips": len(ys)})
    log(f"dasheng_ft: {len(ys)} clips ({parts.count('held')} from held-out days)")


def load_clips(folder):
    meta = json.load(open(os.path.join(folder, "meta.json")))
    y = np.load(os.path.join(folder, "labels.npy"))
    x = np.memmap(os.path.join(folder, "clips.f16"), dtype=np.float16, mode="r", shape=(len(y), CLIP * STEP))
    return x, y, np.asarray(meta["parts"]), np.asarray(meta["weights"], np.float64)


################################################################################
# training and inference
################################################################################

def train_ft(folder, ctx, seed, logf, device, fake):
    import torch
    x, y, parts, weights = load_clips(folder)
    tr, ho = np.where(parts == "train")[0], np.where(parts == "held")[0]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_ft(ctx, fake).to(device)
    lora = [p for n, p in model.named_parameters() if "lora_" in n]
    head = list(model.head.parameters())
    opt = torch.optim.AdamW([{"params": lora, "lr": LR_LORA}, {"params": head, "lr": LR_HEAD}], weight_decay=1e-2)
    yt = y[tr]
    n_pos, n_neg = int((yt == 1).sum()), int((yt == 0).sum())
    cw = torch.tensor([(n_pos + n_neg) / (2 * max(1, n_neg)), (n_pos + n_neg) / (2 * max(1, n_pos))],
                      dtype=torch.float32, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=cw, ignore_index=-1)
    p = weights[tr] / weights[tr].sum()
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def batch(idx):
        return torch.from_numpy(np.asarray(x[np.sort(idx)], np.float32)).to(device), \
            torch.from_numpy(np.asarray(y[np.sort(idx)], np.int64)).to(device)

    def check():
        model.eval()
        sc, yy = [], []
        with torch.no_grad(), torch.autocast("cuda", enabled=amp):
            for i in range(0, len(ho), BATCH):
                xb, yb = batch(ho[i:i + BATCH])
                pr = torch.softmax(model(xb).float(), -1)[..., 1]
                keep = yb >= 0
                sc.append(pr[keep].cpu())
                yy.append(yb[keep].cpu())
        model.train()
        if not sc or not torch.cat(yy).numel():
            return 0.0
        return st.average_precision(torch.cat(sc), torch.cat(yy))

    seen, checks, best, best_check, best_state, since, t0 = 0, 0, -1.0, 0, None, 0, time.time()
    model.train()
    while True:
        idx = rng.choice(tr, BATCH, p=p)
        xb, yb = batch(idx)
        with torch.autocast("cuda", enabled=amp):
            out = model(xb)
        loss = crit(out.float().reshape(-1, 2), yb.reshape(-1))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(lora + head, 5.0)
        scaler.step(opt)
        scaler.update()
        seen += len(idx)
        if seen >= (checks + 1) * CHECK_EVERY:
            checks += 1
            ap = check() if len(ho) else -loss.item()
            better = ap > best + 1e-5
            logf.write(f"seed {seed} check {checks}: loss {loss.item():.4f}, held-out AP {ap:.4f}{' *' if better else ''}\n")
            logf.flush()
            if better:
                best, best_check, since = ap, checks, 0
                best_state = copy.deepcopy(trainable_state(model))
            else:
                since += 1
            if since >= PATIENCE or checks >= MAX_CHECKS:
                break
    info = {"checks": checks, "best_check": best_check, "clips_seen": seen, "heldout_ap": round(best, 5),
            "seconds": round(time.time() - t0, 1), "train_clips": int(len(tr)), "heldout_clips": int(len(ho))}
    logf.write(json.dumps(info) + "\n")
    return best_state, info


def day_tokens(model, y, n, device):
    """Fine-tuned fingerprints for a whole recording ([n, dim]), blocks of CLIP steps every HOP."""
    import torch
    steps = max(n, 1)
    total = max(CLIP, int(math.ceil((steps - CLIP) / HOP)) * HOP + CLIP) if steps > CLIP else CLIP
    audio = np.zeros(total * STEP, np.float32)
    audio[:min(len(y), total * STEP)] = y[:total * STEP]
    starts = list(range(0, total - CLIP + 1, HOP))
    m = (CLIP - HOP) // 2
    out = torch.zeros(total, model.dim)
    with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
        for b in range(0, len(starts), BATCH):
            group = starts[b:b + BATCH]
            xb = torch.from_numpy(np.stack([audio[s * STEP:(s + CLIP) * STEP] for s in group])).to(device)
            e = model.tokens(xb).float().cpu()
            for j, s in enumerate(group):
                a = 0 if s == 0 else m
                z = CLIP if s == starts[-1] else CLIP - m
                out[s + a:s + z] = e[j, a:z]
    return out[:n]


def day_probs(models, path, n, ctx, device):
    """[probability per step] for each fine-tuned run; the audio is read once."""
    import fp_replica
    y = fp_replica.load_audio(path, sr_out=SR).numpy()
    return [ex.day_probs(m.head, day_tokens(m, y, n, device), ctx, device) for m in models]
