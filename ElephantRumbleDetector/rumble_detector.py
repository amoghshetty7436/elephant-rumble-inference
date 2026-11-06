#!/usr/bin/env python
"""
Elephant rumble detector (Elephant Listening Project).

Finds forest elephant rumbles in sound recordings and writes Raven selection tables.
The model is the improved FruitPunch AI detector (E2_ctx8_noise10_rand3000_gap0.5_hard0_augnone,
run 0) with its alarm rule and the cut-off that found 80% of the rumbles on the validation days.
It is the same detector that was run on nn_202306_jun_dep17 and dz_202512_dec_dep10.

Most people never type these commands: they double-click "2 - Find rumbles.bat". For reference:

  python rumble_detector.py                       ask for the recordings and the results folder, then run
  python rumble_detector.py "L:\\path\\to\\site"     the same, with the recordings already given
  python rumble_detector.py setup                 download AVES (once) and check that everything works
  python rumble_detector.py run INPUT... --out RESULTS
  python rumble_detector.py table --out RESULTS --min-score 0.95

INPUT can be a deployment folder (with one subfolder per site), one site folder, or one
recording. Every sound file underneath is used (.wav, .flac, .aif, .aiff), at any sample rate
and of any length. Nothing is ever written next to the recordings: only under RESULTS.
Running the same command again carries on where it stopped.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")
AVES_DIR = os.path.join(MODEL_DIR, "aves")
AVES_URL = "https://storage.googleapis.com/esp-public-files/ported_aves"
AVES_FILES = ("aves-base-bio.torchaudio.model_config.json", "aves-base-bio.torchaudio.pt")

AUDIO_EXT = (".wav", ".flac", ".aif", ".aiff")
AVES_SR = 500  # every recording is resampled to 500 Hz and fed to AVES as if it were 16 kHz
SAMPLES_PER_FRAME = 320
FRAME_SECONDS = SAMPLES_PER_FRAME / AVES_SR  # 0.64 s per probability
PROJ_DIM = 64
RAVEN_HEADER = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)",
                "High Freq (Hz)", "Begin Path", "File Offset (s)", "Begin File", "Score"]
TABLE_SUFFIX = ".rumbles.txt"
PROBS_SUFFIX = ".probs.npy"
FOLDER_TABLE = "_all_rumbles.txt"
TIMING_COLUMNS = ["finished", "site", "file", "sample_rate", "hours", "frames", "read_resample_s", "waited_s",
                  "aves_s", "classify_s", "write_s", "total_s", "detections", "max_prob"]
STALLED = 3        # exit code: nothing moved on for too long, start again
DRIVE_GONE = 4     # exit code: the recordings folder can't be reached any more
DRIVE_WAIT_S = 600  # how long to wait for a vanished folder to come back before stopping


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fmt_dur(s):
    s = max(0, int(round(s)))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d} s"
    if s < 86400:
        return f"{s // 3600} h {s % 3600 // 60:02d} min"
    return f"{s // 86400} d {s % 86400 // 3600} h"


def long_path(path):
    """Windows refuses paths over 260 characters unless they start with \\\\?\\."""
    if os.name == "nt" and len(os.path.abspath(path)) >= 240 and not path.startswith("\\\\?\\"):
        p = os.path.abspath(path)
        return "\\\\?\\UNC\\" + p[2:] if p.startswith("\\\\") else "\\\\?\\" + p
    return path


def _norm(p):
    p = os.path.abspath(p) if not re.match(r"^[A-Za-z]:[\\/]", p) else p
    return p.replace("\\", "/").rstrip("/").lower()


def is_under(path, root):
    p, r = _norm(path), _norm(root)
    return p == r or p.startswith(r + "/")


def clean_path(text):
    """A path as people paste it: quotes, spaces, or a trailing slash around it."""
    t = text.strip().strip('"').strip("'").strip()
    return t.rstrip("\\/") if len(t) > 3 else t


class Tee:
    """Everything printed also goes to <results>\\run_log.txt."""

    def __init__(self, path):
        self.f, self.out = open(path, "a", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.out.flush()


################################################################################
# finding the recordings
################################################################################

def is_audio(name):
    return name.lower().endswith(AUDIO_EXT) and not name.startswith("._")


def find_audio(roots):
    """[(root, path)] of every recording under the roots, sorted by folder then name."""
    out = []
    for root in roots:
        if os.path.isfile(root):
            if not is_audio(root):
                sys.exit(f"{root} is not a sound file this detector reads ({', '.join(AUDIO_EXT)}).")
            out.append((root, root))
            continue
        if not os.path.isdir(root):
            sys.exit(f"Can't find {root}.\nCheck the spelling, and that the drive (for example L:) is connected.")

        def unlisted(err):  # os.walk skips a folder it can't open without saying so
            log(f"WARNING: could not open the folder {err.filename} ({err.strerror}). "
                f"Its recordings are left out this time; run again later to include them.")

        for dirpath, dirs, names in os.walk(root, onerror=unlisted):
            dirs.sort()
            out += [(root, os.path.join(dirpath, n)) for n in sorted(names) if is_audio(n)]
    return out


def out_base(out, root, path, several):
    """Where a recording's results go, without the extension. One input: <out>\\<subfolders>\\<name>.
    Several inputs: <out>\\<input folder's name>\\<subfolders>\\<name>."""
    if os.path.isfile(root) and root == path:
        sub = os.path.basename(path)
        top = os.path.basename(os.path.dirname(os.path.abspath(path)))
    else:
        sub = os.path.relpath(path, root)
        top = os.path.basename(os.path.normpath(root))
    sub = os.path.splitext(sub)[0]
    return os.path.join(out, top, sub) if several else os.path.join(out, sub)


def audio_info(path):
    import soundfile as sf
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return dict(sr=0, channels=0, hours=0.0, bytes=0, error=f"can't open it ({e.strerror})")
    if size == 0:
        return dict(sr=0, channels=0, hours=0.0, bytes=0, error="the file is empty (0 bytes)")
    try:
        i = sf.info(path)
        return dict(sr=i.samplerate, channels=i.channels, hours=i.frames / i.samplerate / 3600, bytes=size, error="")
    except Exception as e:  # noqa: BLE001 - a broken file is reported, not fatal
        return dict(sr=0, channels=0, hours=0.0, bytes=size,
                    error=f"not a readable sound file, maybe damaged or not fully copied ({str(e)[:120]})")


def site_of(path):
    """The site code at the start of an ELP file name (nn06a_..., dzan_...), or else the folder name."""
    m = re.match(r"([a-z]+\d+[a-z]?)_", os.path.basename(path).lower())
    return m.group(1) if m else os.path.basename(os.path.dirname(os.path.abspath(path)))


def plain_error(path, e):
    """A reason anyone can act on."""
    if isinstance(e, FileNotFoundError) or not os.path.exists(path):
        return "the file is no longer there (moved, renamed or deleted, or the drive disconnected)"
    if isinstance(e, PermissionError):
        return "no permission to read it"
    info = audio_info(path)
    if info["error"]:
        return info["error"]
    return f"{type(e).__name__}: {str(e)[:200]}"


################################################################################
# audio, AVES and the classifier
################################################################################

def load_audio(path):
    """Mono float32 tensor at 500 Hz, resampled an hour at a time to keep memory low.
    Exactly as the model's training and test days were read (fp_replica.load_audio)."""
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    with sf.SoundFile(long_path(path)) as f:
        sr, total = f.samplerate, f.frames
        if total < sr * 2 * FRAME_SECONDS:  # under ~1.3 s (e.g. a header-only file): nothing to analyse
            return torch.zeros(0)
        ratio = Fraction(AVES_SR, sr)
        up, down = ratio.numerator, ratio.denominator
        block, pad = 3600 * sr, 10 * sr
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
    return y.to(torch.float16).to(torch.float32)  # the training audio was cached as float16


def get_aves_files():
    """AVES (378 MB) lives in model\\aves next to this script; download it the first time."""
    import urllib.request
    os.makedirs(AVES_DIR, exist_ok=True)
    for name in AVES_FILES:
        dst = os.path.join(AVES_DIR, name)
        if os.path.exists(dst):
            continue
        # a copy in the old place (the user's torch cache) saves the download
        old = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "fruitpunch_elephants", name)
        if os.path.exists(old):
            import shutil
            log(f"copying {name} from {old}")
            shutil.copyfile(old, dst + ".part")
            os.replace(dst + ".part", dst)
            continue
        log(f"downloading {name} (once; the big one is 378 MB)...")
        try:
            with urllib.request.urlopen(f"{AVES_URL}/{name}", timeout=60) as r, open(dst + ".part", "wb") as f:
                size, done, shown = int(r.headers.get("Content-Length", 0)), 0, 0
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    done += len(b)
                    if size and done * 10 // size > shown:
                        shown = done * 10 // size
                        log(f"  {done * 100 // size}%")
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Could not download AVES ({e}).\nThis needs the internet once. Check the connection and "
                     f"run '1 - Install.bat' again, or copy the two AVES files into {AVES_DIR}.")
        os.replace(dst + ".part", dst)
    return [os.path.join(AVES_DIR, n) for n in AVES_FILES]


class Embedder:
    """AVES embeddings, one per 0.64 s, exactly as fp_replica.Embedder computes them."""

    def __init__(self, device):
        import torch
        from torchaudio.models import wav2vec2_model
        config, weights = get_aves_files()
        self.device = device
        self.model = wav2vec2_model(**json.load(open(config)), aux_num_out=None)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model = self.model.eval().to(device)

    def __call__(self, y):
        import torch
        if y.shape[0] < SAMPLES_PER_FRAME * 2:
            return torch.empty(0, 768)
        with torch.inference_mode():
            e = self.model.extract_features(y.view(1, -1).to(self.device))[0][-1][0].float().cpu()
        return e / e.norm(p=2, dim=1, keepdim=True)

    def whole_file(self, y, beat=None, chunk_seconds=3600, context=16):
        """An hour at a time, with ~10 s of audio on each side of every hour."""
        import torch
        chunk, ctx = chunk_seconds * AVES_SR, context * SAMPLES_PER_FRAME
        out, n, shown = [], (y.shape[0] + chunk - 1) // chunk, time.time()
        for s in range(0, y.shape[0], chunk):
            if beat:
                beat(f"running AVES ({s // chunk + 1} of {n} pieces)")
            if time.time() - shown > 30:  # slow (on a processor): show that it is moving
                log(f"    {s / AVES_SR / 3600:.1f} of {y.shape[0] / AVES_SR / 3600:.1f} hours of this recording analysed...")
                shown = time.time()
            pre, cur, post = y[max(0, s - ctx):s], y[s:s + chunk], y[s + chunk:s + chunk + ctx]
            e = self(torch.cat([pre, cur, post]))
            a = pre.shape[0] // SAMPLES_PER_FRAME
            b = (pre.shape[0] + cur.shape[0]) // SAMPLES_PER_FRAME
            out.append(e[a:b])
            if post.shape[0] == 0 and e.shape[0] > 1:
                out.append(e[-2:-1])  # AVES drops the last frame without audio after it
        return torch.cat(out) if out else torch.empty(0, 768)


def make_classifier(ctx):
    import torch.nn as nn

    class ContextClassifier(nn.Module):
        """Squeeze each 0.64 s fingerprint to 64 numbers, then decide from 2*ctx+1 of them side by side."""

        def __init__(self):
            super().__init__()
            self.ctx = ctx
            self.drop = nn.Dropout(0.2)
            self.act = nn.LeakyReLU()
            self.proj = nn.Linear(768, PROJ_DIM)
            self.linear1 = nn.Linear((2 * ctx + 1) * PROJ_DIM, 192)
            self.linear2 = nn.Linear(192, 2)

        def frame_features(self, x):
            return self.act(self.proj(self.drop(x)))

        def head(self, flat):
            return self.linear2(self.drop(self.act(self.linear1(self.drop(flat)))))

    return ContextClassifier()


def limit_cores(cores):
    """Use at most `cores` processor cores, leaving the rest free for other work. Must run before
    numpy/scipy/torch are first imported for the environment variables to count."""
    if not cores:
        return
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(cores)
    import torch
    torch.set_num_threads(cores)
    try:
        torch.set_num_interop_threads(max(1, min(cores, 2)))
    except RuntimeError:  # already set
        pass


USE_CPU_FILE = os.path.join(HERE, "use_cpu.txt")  # written by the installer when the graphics card fails its test


def pick_device(want):
    import torch
    if want == "cpu" or (want == "auto" and os.path.exists(USE_CPU_FILE)):
        return "cpu"
    if want == "gpu" and not torch.cuda.is_available():
        sys.exit("--device gpu was asked for, but PyTorch can't see an NVIDIA graphics card on this computer.")
    return "cuda" if torch.cuda.is_available() else "cpu"


class Detector:
    def __init__(self, model_dir=MODEL_DIR, device=None, fake=False):
        import torch
        s = json.load(open(os.path.join(model_dir, "settings.json")))
        self.name, self.rule, self.cut_off, self.ctx = s["name"], s["rule"], s["cut_off"], s["context_frames"]
        self.model_path = os.path.join(model_dir, s["model_file"])
        if not os.path.exists(self.model_path):
            sys.exit(f"The model file is missing: {self.model_path}\nDownload the package again.")
        self.device = device or pick_device("auto")
        m = make_classifier(self.ctx)
        m.load_state_dict(torch.load(self.model_path, map_location="cpu", weights_only=True))
        self.model = m.eval().to(self.device)
        self.embed = FakeEmbedder() if fake else Embedder(self.device)
        if self.device == "cuda":
            where = f"the graphics card ({torch.cuda.get_device_name(0)})"
        else:
            why = "as asked" if torch.cuda.is_available() else "no NVIDIA graphics card found"
            where = (f"the processor ({why}), using {torch.get_num_threads()} of its {os.cpu_count()} cores; "
                     f"this works but is much slower than a graphics card")
        log(f"detector {self.name}; running on {where}")

    def probs(self, y, beat=None):
        """Rumble probability of every 0.64 s frame (numpy float32)."""
        import numpy as np
        import torch
        try:
            emb = self.embed.whole_file(y, beat)
        except torch.cuda.OutOfMemoryError:  # a small graphics card: smaller pieces
            torch.cuda.empty_cache()
            log("  the graphics card ran out of memory; trying again in 15-minute pieces")
            emb = self.embed.whole_file(y, beat, chunk_seconds=900)
        if emb.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        ctx, chunk = self.ctx, 20_000
        with torch.no_grad():
            h = self.model.frame_features(emb.to(self.device).float())
            hp = torch.cat([h[:1].expand(ctx, -1), h, h[-1:].expand(ctx, -1)])  # repeat the edge frames
            win = hp.unfold(0, 2 * ctx + 1, 1).transpose(1, 2)
            lg = torch.cat([self.model.head(win[i:i + chunk].reshape(-1, (2 * ctx + 1) * PROJ_DIM))
                            for i in range(0, win.shape[0], chunk)])
            return torch.softmax(lg, 1)[:, 1].float().cpu().numpy()


class FakeEmbedder:
    """For the tests: random-ish fingerprints from the audio, no AVES download."""

    def whole_file(self, y, beat=None, chunk_seconds=None):
        import torch
        n = y.shape[0] // SAMPLES_PER_FRAME
        if n < 2:
            return torch.empty(0, 768)
        frames = y[:n * SAMPLES_PER_FRAME].view(n, SAMPLES_PER_FRAME)
        g = torch.Generator().manual_seed(0)
        e = frames.abs().mean(1, keepdim=True) * torch.randn(1, 768, generator=g) + 0.01 * torch.randn(n, 768, generator=g)
        return e / e.norm(dim=1, keepdim=True)


################################################################################
# the alarm rule and the tables
################################################################################

def alarms(prob, rule):
    """(first frames, last frames, scores) of a recording's detections (improve.alarms)."""
    import numpy as np
    p = prob
    s = rule["smooth"]
    if s > 1 and len(p):
        k = np.ones(s)
        p = np.convolve(prob, k, "same") / np.convolve(np.ones_like(prob), k, "same")
    d = np.diff(np.concatenate([[0], (p > rule["on"]).astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    if len(st) and rule["gap"] > 0:
        first = np.flatnonzero(np.concatenate([[True], st[1:] - en[:-1] - 1 > rule["gap"]]))
        last = np.concatenate([first[1:] - 1, [len(st) - 1]])
        st, en = st[first], en[last]
    keep = en - st + 1 >= rule["min_frames"]
    st, en = st[keep], en[keep]
    if not len(st):
        return st, en, np.zeros(0)
    if rule["score"] == "mean":
        cs = np.concatenate([[0.0], np.cumsum(prob, dtype=np.float64)])
        sc = (cs[en + 1] - cs[st]) / (en - st + 1)
    else:
        ext = np.concatenate([prob, [0.0]]).astype(np.float64)
        sc = np.maximum.reduceat(ext, np.column_stack([st, en + 1]).ravel())[::2]
    return st, en, sc


def file_rows(prob, rule, min_score, src):
    return [(a * FRAME_SECONDS, b * FRAME_SECONDS, src, a * FRAME_SECONDS, float(s))
            for a, b, s in zip(*alarms(prob, rule)) if s >= min_score]


def write_table(path, rows):
    """rows: (begin s in the table, end s, full audio path, offset s in that file, score)."""
    os.makedirs(long_path(os.path.dirname(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(long_path(tmp), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(RAVEN_HEADER)
        for i, (bt, et, src, off, sc) in enumerate(rows, 1):
            wr.writerow([i, "Spectrogram 1", 1, f"{bt:.3f}", f"{et:.3f}", 5, 250, src, f"{off:.3f}",
                         os.path.basename(src), f"{sc:.4f}"])
    os.replace(long_path(tmp), long_path(path))


def folder_table(folder, files, rule, min_score):
    """One Raven table for a folder: its recordings one after another, as ELP's day tables are."""
    import numpy as np
    rows, elapsed = [], 0.0
    for src, base in files:
        p = base + PROBS_SUFFIX
        if not os.path.exists(long_path(p)):
            continue
        prob = np.load(long_path(p)).astype(np.float32)
        for bt, et, s, off, sc in file_rows(prob, rule, min_score, src):
            rows.append((elapsed + bt, elapsed + et, s, off, sc))
        elapsed += len(prob) * FRAME_SECONDS
    write_table(os.path.join(folder, FOLDER_TABLE), rows)
    return len(rows)


def by_out_folder(jobs):
    groups = {}
    for src, base in jobs:
        groups.setdefault(os.path.dirname(base), []).append((src, base))
    return groups


################################################################################
# run
################################################################################

class Watchdog:
    """Ends the process (exit code STALLED) when nothing has moved on for `limit` seconds, for
    example a read from the network drive that never returns, so that it can be started again."""

    def __init__(self, limit, exit=os._exit):
        self.limit, self.exit = limit, exit
        self.last, self.what = time.time(), "starting"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)
        if limit > 0:
            self.thread.start()

    def beat(self, what):
        self.last, self.what = time.time(), what

    def watch(self):
        while not self.stop.wait(min(30.0, self.limit / 4)):
            if time.time() - self.last > self.limit:
                log(f"STOPPED: nothing happened for {fmt_dur(time.time() - self.last)} while {self.what}. "
                    f"Starting again; finished recordings are kept.")
                sys.stdout.flush()
                self.exit(STALLED)
                return


def keep_awake():
    """Ask Windows not to go to sleep while this runs (the screen may still turn off)."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000000 | 0x00000001))
        except Exception:  # noqa: BLE001
            pass


def no_quick_edit():
    """In a Windows console a single click pauses the program until Esc is pressed. Turn that off."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-10)
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(h, ctypes.byref(mode)):
            k.SetConsoleMode(h, (mode.value & ~0x0040) | 0x0080)
    except Exception:  # noqa: BLE001
        pass


def wait_for(root, wd, limit=DRIVE_WAIT_S):
    """True once the input folder (or file) can be reached; waits up to `limit` s if it vanished."""
    if os.path.exists(root):
        return True
    log(f"WARNING: {root} can't be reached any more (drive disconnected, or the folder was moved?). "
        f"Waiting up to {fmt_dur(limit)} for it to come back...")
    t0 = time.time()
    while time.time() - t0 < limit:
        wd.beat(f"waiting for {root} to come back")
        time.sleep(min(15, limit / 10))
        if os.path.exists(root):
            log(f"{root} is back; carrying on.")
            return True
    return False


def cmd_run(a):
    keep_awake()
    no_quick_edit()
    os.makedirs(a.out, exist_ok=True)
    stdout, sys.stdout = sys.stdout, Tee(os.path.join(a.out, "run_log.txt"))
    wd = Watchdog(a.stall_minutes * 60)
    try:
        return run(a, wd)
    finally:
        wd.stop.set()
        sys.stdout.f.close()
        sys.stdout = stdout


def run(a, wd):
    import numpy as np
    log(f"=== started; recordings: {'; '.join(a.inputs)}")
    log("looking for recordings...")
    wd.beat("listing the recordings")
    files = find_audio(a.inputs)
    if not files:
        log("No sound files found there (looked for " + ", ".join(AUDIO_EXT) + " in every subfolder).")
        return 1
    several = len(a.inputs) > 1
    jobs = [(p, out_base(a.out, r, p, several)) for r, p in files]
    root_of = {p: r for r, p in files}
    todo = [(p, b) for p, b in jobs if a.redo or not os.path.exists(long_path(b + TABLE_SUFFIX))]
    n_sites = len({os.path.dirname(b) for _, b in jobs})
    log(f"{len(files)} recordings in {n_sites} folder(s): {len(files) - len(todo)} already done, {len(todo)} to do")
    log(f"results go to {a.out}")
    det = Detector(a.model_dir, device=pick_device(a.device), fake=a.fake_embedder)
    min_score = det.cut_off if a.min_score is None else a.min_score
    info = {"detector": det.name, "model": det.model_path, "rule": det.rule, "cut_off": det.cut_off,
            "min_score_in_tables": min_score, "inputs": a.inputs, "machine": machine()}
    with open(os.path.join(a.out, "detector.json"), "w") as f:
        json.dump(info, f, indent=1)
    write_about(a.out, min_score, det.cut_off)
    timing_path = os.path.join(a.out, "timings.csv")
    new = not os.path.exists(timing_path)
    tf = open(timing_path, "a", newline="")
    tw = csv.writer(tf)
    if new:
        tw.writerow(TIMING_COLUMNS)
    if os.path.exists(os.path.join(a.out, "failed.txt")):
        os.remove(os.path.join(a.out, "failed.txt"))  # this run retries them; it lists what still fails
    done_hours, t_start, failed = 0.0, time.time(), []
    total_hours = 0.0
    if a.estimate and todo:
        log("reading the length of each recording (for the time left)...")
        for k, (p, _) in enumerate(todo):
            wd.beat("reading the length of each recording")
            total_hours += audio_info(p)["hours"]
        log(f"{total_hours:,.1f} hours of audio to do ({total_hours / 24:,.1f} days)")
    left_in_folder = {}
    for _, b in todo:
        left_in_folder[os.path.dirname(b)] = left_in_folder.get(os.path.dirname(b), 0) + 1
    pool = cf.ThreadPoolExecutor(max(1, a.prefetch))  # read + resample the next files meanwhile
    ahead = {}

    def queue(i):
        if i < len(todo) and i not in ahead:
            ahead[i] = pool.submit(timed_load, todo[i][0])

    for i in range(min(len(todo), a.prefetch)):
        queue(i)
    stopped = False
    for i, (src, base) in enumerate(todo):
        if not wait_for(root_of[src], wd):
            log(f"STOPPED: {root_of[src]} is still not reachable. Connect the drive and start again; "
                f"the {i} recordings finished in this run are kept.")
            stopped = True
            break
        queue(i + a.prefetch)
        try:
            t0 = time.time()
            wd.beat(f"reading {src} from the drive")
            y, t_load = ahead.pop(i).result()
            waited = time.time() - t0
            sr = audio_info(src)["sr"]
            hours = y.shape[0] / AVES_SR / 3600
            t1 = time.time()
            prob = det.probs(y, wd.beat)
            t3 = time.time()
            wd.beat(f"writing the results of {src}")
            os.makedirs(long_path(os.path.dirname(base)), exist_ok=True)
            np.save(long_path(base + PROBS_SUFFIX + ".tmp.npy"), prob.astype(np.float16))
            os.replace(long_path(base + PROBS_SUFFIX + ".tmp.npy"), long_path(base + PROBS_SUFFIX))
            rows = file_rows(prob, det.rule, min_score, src)
            write_table(base + TABLE_SUFFIX, rows)
            t4 = time.time()
        except Exception as e:  # noqa: BLE001 - one bad file must not stop a long run
            why = plain_error(src, e)
            log(f"[{i + 1}/{len(todo)}] SKIPPED {src}: {why}")
            failed.append((src, why))
            with open(os.path.join(a.out, "failed.txt"), "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{src}\t{why}\n")
            d = os.path.dirname(base)
            left_in_folder[d] -= 1
            if left_in_folder[d] == 0:
                finish_folder(d, jobs, det, min_score, a.out, info, wd)
            continue
        del y
        tw.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), site_of(src), src, sr, f"{hours:.4f}", len(prob),
                     f"{t_load:.1f}", f"{waited:.1f}", f"{t3 - t1:.1f}", "0", f"{t4 - t3:.2f}",
                     f"{t4 - t0:.1f}", len(rows), f"{float(prob.max()) if len(prob) else 0:.4f}"])
        tf.flush()
        done_hours += hours
        el = time.time() - t_start
        left = (f", about {fmt_dur(max(0.0, total_hours - done_hours) * el / done_hours)} left"
                if total_hours and done_hours > 0 else "")
        note = " (too short to analyse)" if hours < 2 * FRAME_SECONDS / 3600 else ""
        log(f"[{i + 1}/{len(todo)}] {os.path.basename(src)}: {hours:.1f} h, {len(rows)} rumbles{note}; "
            f"took {fmt_dur(t4 - t0)}{left}")
        d = os.path.dirname(base)
        left_in_folder[d] -= 1
        if left_in_folder[d] == 0:
            finish_folder(d, jobs, det, min_score, a.out, info, wd)
    pool.shutdown(wait=False, cancel_futures=True)
    tf.close()
    missing = write_summary(a.out, jobs, info, min_score, failed)
    log(f"done with {done_hours:,.1f} h of audio in {fmt_dur(time.time() - t_start)}")
    if failed:
        log(f"WARNING: {len(failed)} recording(s) could not be analysed; the list and the reasons are in "
            f"{os.path.join(a.out, 'failed.txt')}")
    if missing - len(failed) > 0 and not stopped:
        log(f"{missing - len(failed)} recording(s) have no results yet. Running the detector again on the same folder "
            f"retries just those.")
    if not missing:
        log(f"all {len(jobs)} recordings have their results")
    log(f"summary: {os.path.join(a.out, 'summary.md')}")
    return DRIVE_GONE if stopped else 0


def timed_load(path):
    t0 = time.time()
    y = load_audio(path)
    return y, time.time() - t0


def finish_folder(d, jobs, det, min_score, out, info, wd):
    wd.beat(f"writing the table for the whole folder {d}")
    files = by_out_folder(jobs).get(d, [])
    if any(os.path.exists(long_path(b + PROBS_SUFFIX)) for _, b in files):
        n = folder_table(d, files, det.rule, min_score)
        log(f"folder done: {os.path.join(d, FOLDER_TABLE)} ({n} rumbles)")
    write_summary(out, jobs, info, min_score, [])


def machine():
    import platform
    import torch
    return {"computer": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (processor)",
            "cores_used": torch.get_num_threads(), "cores": os.cpu_count()}


def write_summary(out, jobs, info, min_score, failed):
    """summary.md and summary.csv: per site, recordings, hours and rumbles. Returns how many
    recordings have no results."""
    last = {}
    path = os.path.join(out, "timings.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            last[r["file"]] = r
    failed_now = {}
    fpath = os.path.join(out, "failed.txt")
    if os.path.exists(fpath):
        for line in open(fpath, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                failed_now[parts[1]] = parts[2]
    sites = {}
    for src, base in jobs:
        s = sites.setdefault(site_of(src), dict(files=0, done=0, hours=0.0, rumbles=0, short=0, failed=0))
        s["files"] += 1
        table = long_path(base + TABLE_SUFFIX)
        if not os.path.exists(table):
            s["failed"] += src in failed_now
            continue
        s["done"] += 1
        s["rumbles"] += sum(1 for _ in open(table, encoding="utf-8")) - 1
        r = last.get(src)
        if r:
            s["hours"] += float(r["hours"])
            s["short"] += float(r["hours"]) < 2 * FRAME_SECONDS / 3600
    rows = [dict(site=name, recordings=s["files"], done=s["done"], could_not_read=s["failed"],
                 hours=round(s["hours"], 2), days=round(s["hours"] / 24, 2), rumbles=s["rumbles"],
                 rumbles_per_day=round(s["rumbles"] / (s["hours"] / 24), 1) if s["hours"] else "",
                 too_short=s["short"]) for name, s in sorted(sites.items())]
    if not rows:
        return 0
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tot = {k: sum(r[k] for r in rows) for k in ("recordings", "done", "could_not_read", "hours", "rumbles", "too_short")}
    missing = tot["recordings"] - tot["done"]
    md = ["# Rumble detection summary", "",
          f"Updated {time.strftime('%Y-%m-%d %H:%M')}. Detector: {info['detector']}. "
          f"Tables keep detections with a Score of at least {min_score:.4f}"
          + (" (the standard cut-off)." if abs(min_score - info["cut_off"]) < 1e-9 else
             f" (the standard cut-off is {info['cut_off']:.4f})."), "",
          f"- **{tot['done']} of {tot['recordings']} recordings done**, {tot['hours']:,.1f} hours of audio "
          f"({tot['hours'] / 24:,.1f} days), **{tot['rumbles']:,} rumbles found**"]
    if tot["could_not_read"]:
        md.append(f"- **{tot['could_not_read']} recordings could not be read**: see failed.txt for which and why")
    if missing - tot["could_not_read"] > 0:
        md.append(f"- {missing - tot['could_not_read']} recordings not done yet: run the detector again on the same folder")
    if tot["too_short"]:
        md.append(f"- {tot['too_short']} recordings were too short to analyse (under 2 seconds); they have empty tables")
    md += ["", "| Site | Recordings | Done | Could not read | Hours | Days | Rumbles | Rumbles per day |",
           "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['site']} | {r['recordings']} | {r['done']} | {r['could_not_read']} | {r['hours']:,.1f} | {r['days']:.1f} "
           f"| {r['rumbles']:,} | {r['rumbles_per_day']} |" for r in rows]
    md += ["", "The same table is in summary.csv (opens in Excel). Every recording, with timings: timings.csv. "
               "Everything printed on screen: run_log.txt."]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return missing


def write_about(out, min_score, cut_off):
    text = f"""WHAT IS IN THIS FOLDER
======================

Made by the elephant rumble detector (ElephantRumbleDetector, README.md).

The folders here mirror the folders of the recordings. In each one:

  _all_rumbles.txt          ONE RAVEN TABLE FOR THE WHOLE FOLDER (usually one site): every rumble
                            found in its recordings, one recording after another. Open it in Raven
                            Pro over the folder's sound files ("File > Open Sound Files", select them
                            all, then "File > Open Selection Table").
  <recording>.rumbles.txt   The Raven table of one recording.
  <recording>.probs.npy     The rumble probability of every 0.64 s. You don't need to open these;
                            they let the cut-off be changed later in seconds (see the README).

In this folder:

  summary.md / summary.csv  Per site: recordings done, hours, rumbles found, rumbles per day.
                            summary.csv opens in Excel; summary.md opens in Notepad.
  failed.txt                Only there if some recordings could not be read, with the reason.
  run_log.txt               Everything shown on screen while it ran.
  timings.csv               One line per recording: how long it took, how many rumbles.
  detector.json             Which detector, settings and computer made these results.

The Score column: how sure the detector is (0 to 1). These tables keep detections scoring at
least {min_score:.4f} (the standard cut-off is {cut_off:.4f}, which found 80% of the rumbles on the
validation days). Each detection box spans 5-250 Hz; the times are what to check.
"""
    with open(os.path.join(out, "ABOUT THESE RESULTS.txt"), "w", encoding="utf-8") as f:
        f.write(text)


def cmd_table(a):
    import numpy as np
    dj = os.path.join(a.out, "detector.json")
    if not os.path.exists(dj):
        sys.exit(f"{a.out} doesn't look like a results folder (no detector.json in it).")
    info = json.load(open(dj))
    rule = info["rule"]
    min_score = info["cut_off"] if a.min_score is None else a.min_score
    src_of = {}
    timing = os.path.join(a.out, "timings.csv")
    if os.path.exists(timing):  # the original audio paths, for the Begin Path column
        for r in csv.DictReader(open(timing)):
            src_of[os.path.splitext(os.path.basename(r["file"]))[0]] = r["file"]
    jobs = []
    for dirpath, dirs, names in os.walk(a.out):
        dirs.sort()
        for n in sorted(names):
            if n.endswith(PROBS_SUFFIX):
                b = os.path.join(dirpath, n[:-len(PROBS_SUFFIX)])
                jobs.append((src_of.get(os.path.basename(b), os.path.basename(b) + ".wav"), b))
    n = 0
    for src, base in jobs:
        prob = np.load(long_path(base + PROBS_SUFFIX)).astype(np.float32)
        rows = file_rows(prob, rule, min_score, src)
        write_table(base + TABLE_SUFFIX, rows)
        n += len(rows)
    for d, files in by_out_folder(jobs).items():
        folder_table(d, files, rule, min_score)
    info["min_score_in_tables"] = min_score
    with open(dj, "w") as f:
        json.dump(info, f, indent=1)
    write_summary(a.out, jobs, info, min_score, [])
    write_about(a.out, min_score, info["cut_off"])
    log(f"rewrote {len(jobs)} tables keeping Scores of at least {min_score}: {n} rumbles. summary.md updated.")
    return 0


################################################################################
# setup check, and the questions for people who double-click
################################################################################

def cmd_setup(a):
    import numpy as np
    import torch
    log(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, {os.cpu_count()} processor cores")
    if torch.cuda.is_available():
        log(f"graphics card: {torch.cuda.get_device_name(0)} - good, this will be fast")
    else:
        log("no NVIDIA graphics card found: the detector will run on the processor (works, but much slower)")
    get_aves_files()
    det = Detector(a.model_dir, device=pick_device(a.device))
    t = np.arange(60 * 8000) / 8000  # one minute of made-up 8 kHz audio
    y = torch.from_numpy((0.05 * np.random.default_rng(0).standard_normal(len(t))).astype(np.float32))
    from scipy.signal import resample_poly
    y = torch.from_numpy(resample_poly(y.numpy(), 1, 16).astype(np.float32))
    t0 = time.time()
    prob = det.probs(y)
    log(f"test on one minute of audio: {len(prob)} frames in {time.time() - t0:.1f} s")
    if len(prob) < 90:
        sys.exit("SOMETHING IS WRONG: the test gave too few results. Please send this window to Amogh.")
    log("ALL GOOD. The detector is ready: double-click '2 - Find rumbles.bat'.")
    return 0


def default_out(inp):
    home = os.path.expanduser("~")
    for desk in (os.path.join(os.environ.get("OneDrive", ""), "Desktop"), os.path.join(home, "Desktop")):
        if desk and os.path.isdir(desk):
            break
    else:
        desk = home
    name = os.path.splitext(os.path.basename(os.path.normpath(inp)))[0] or "results"
    return os.path.join(desk, "Rumble Results", name)


def ask(question):
    try:
        return input(question)
    except EOFError:
        return ""


def interactive(given=None):
    print()
    print("  ELEPHANT RUMBLE DETECTOR")
    print("  ========================")
    print()
    while True:
        if given:
            inp, given = clean_path(given), None
        else:
            print("  Which recordings? Paste the path of a deployment folder, a site folder or one")
            print("  sound file (in File Explorer: click the address bar, Ctrl+C, then right-click here).")
            inp = clean_path(ask("  Recordings: "))
        if not inp:
            continue
        if os.path.isdir(inp) or (os.path.isfile(inp) and is_audio(inp)):
            break
        print(f"\n  Can't find a folder or sound file at:\n    {inp}\n  Check it and paste it again.\n")
    print(f"\n  Recordings: {inp}")
    out_d = default_out(inp)
    print("\n  Where should the results go? Press Enter to use:")
    print(f"    {out_d}")
    print("  or paste another folder.")
    while True:
        out = clean_path(ask("  Results folder: ")) or out_d
        out = os.path.abspath(out)
        if is_under(out, inp) and os.path.isdir(inp):
            print("  The results can't go inside the recordings folder. Pick another place.")
            continue
        break
    print(f"\n  Results: {out}\n")
    n = os.cpu_count() or 1
    if has_nvidia_gpu():
        cores = 0
        print("  An NVIDIA graphics card was found: it does the heavy work, so there is no need")
        print("  to choose processor cores (the detector only uses a few of them).\n")
    else:
        cores = ask_cores(n)
    if os.path.exists(os.path.join(out, "detector.json")):
        print("  This results folder already has results: recordings that are done will be skipped.\n")
    cmd = [sys.executable, os.path.abspath(__file__), "run", inp, "--out", out] + (["--cores", str(cores)] if cores else [])
    cmd += os.environ.get("RUMBLE_TEST_ARGS", "").split()  # only for the tests
    return run_with_restarts(cmd, out)


def has_nvidia_gpu():
    """Without importing PyTorch (slow): does nvidia-smi see a card, and hasn't the installer
    switched this computer to the processor?"""
    if os.path.exists(USE_CPU_FILE) or os.environ.get("RUMBLE_TEST_ARGS"):
        return False
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and "GPU" in r.stdout
    except Exception:  # noqa: BLE001 - no driver, not found, ...
        return False


def ask_cores(n):
    print(f"  This computer has {n} processor cores. How many may the detector use?")
    print(f"  Press Enter to use all {n}, or type a smaller number to keep some free for other work.")
    while True:
        c = ask("  Cores: ").strip()
        if not c:
            cores = 0
            break
        if c.isdigit() and 1 <= int(c) <= n:
            cores = int(c)
            break
        print(f"  Type a number from 1 to {n}, or just press Enter.")
    print(f"\n  Cores: {cores or n}\n")
    return cores


def run_with_restarts(cmd, out):
    restarts = 0
    while True:
        code = subprocess.call(cmd)
        if code == STALLED and restarts < 50:
            restarts += 1
            log("starting again where it stopped...")
            continue
        break
    print()
    if code == 0:
        print("  FINISHED. The results folder will open now; start with 'ABOUT THESE RESULTS.txt'")
        print("  and summary.md.")
    elif code == DRIVE_GONE:
        print("  The recordings stopped being reachable. Connect the drive and run the detector again")
        print("  on the same folder: it carries on where it stopped.")
    else:
        print("  It stopped with a problem (see the messages above). Running it again carries on where it")
        print("  stopped. If it keeps happening, send run_log.txt from the results folder to Amogh.")
    if os.name == "nt" and os.path.isdir(out):
        try:
            os.startfile(out)
        except Exception:  # noqa: BLE001
            pass
    return code


def interactive_cutoff():
    print()
    print("  CHANGE THE CUT-OFF OF FINISHED RESULTS")
    print("  ======================================")
    print("  This rewrites the Raven tables of a results folder with another cut-off, in seconds.")
    print("  Lower (e.g. 0.95): more rumbles found, but more false alarms. Higher (e.g. 0.999): fewer of both.")
    print()
    while True:
        out = clean_path(ask("  Results folder (paste its path): "))
        if out and os.path.exists(os.path.join(out, "detector.json")):
            break
        print("  That isn't a results folder (it has no detector.json). Paste the folder that has summary.md in it.")
    info = json.load(open(os.path.join(out, "detector.json")))
    print(f"\n  Now: {info.get('min_score_in_tables', info['cut_off'])}. Standard: {info['cut_off']}.")
    while True:
        v = ask(f"  New cut-off between 0 and 1 (press Enter for the standard {info['cut_off']}): ").strip()
        try:
            v = float(v) if v else info["cut_off"]
            if 0 <= v <= 1:
                break
        except ValueError:
            pass
        print("  Type a number such as 0.95.")
    return main(["table", "--out", out, "--min-score", str(v)])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["cutoff"]:
        return interactive_cutoff()
    if not argv or argv[0] not in ("setup", "run", "table", "-h", "--help"):
        if len(argv) > 1:
            print(f"  (Several items were dropped; using only the first: {argv[0]})")
        return interactive(argv[0] if argv else None)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["setup", "run", "table"])
    p.add_argument("inputs", nargs="*", help="deployment folders, site folders or single recordings")
    p.add_argument("--out", help="where the results go (can be on the network drive)")
    p.add_argument("--min-score", type=float, default=None,
                   help="keep detections scoring at least this (default: the standard cut-off; 0 keeps all)")
    p.add_argument("--model-dir", default=MODEL_DIR, help=argparse.SUPPRESS)
    p.add_argument("--cores", type=int, default=0,
                   help="processor cores the detector may use (default: all); leaves the rest free")
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                   help="auto (default): the NVIDIA graphics card if there is one, else the processor")
    p.add_argument("--prefetch", type=int, default=2, help="recordings read ahead in the background")
    p.add_argument("--redo", action="store_true", help="redo recordings that already have results")
    p.add_argument("--stall-minutes", type=float, default=20,
                   help=f"stop with exit code {STALLED} after this long without progress (0: never)")
    p.add_argument("--no-estimate", dest="estimate", action="store_false", help="don't work out the time left")
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.cores:
        a.cores = max(1, min(a.cores, os.cpu_count() or a.cores))
        limit_cores(a.cores)
    if a.step == "setup":
        return cmd_setup(a)
    if not a.out:
        p.error("--out is needed")
    a.out = os.path.abspath(a.out)
    if a.step == "table":
        return cmd_table(a)
    if not a.inputs:
        p.error("give at least one folder or recording")
    a.inputs = [os.path.abspath(clean_path(i)) for i in a.inputs]  # full paths in the Raven tables
    for r in a.inputs:
        if os.path.isdir(r) and is_under(a.out, r):
            sys.exit(f"The results folder {a.out} is inside the recordings folder {r}; pick a folder outside it.")
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
