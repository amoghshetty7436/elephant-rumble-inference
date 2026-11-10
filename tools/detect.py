"""Run the rerun's final detector on recordings and time every step.

This is the detector ELP would run: AVES fingerprints, the final model (the study's best
run, with its listening window), the final alarm rule and the validation cut-off, written
as one Raven table per recording. It reads the model, rule and cut-off from
Configurations\\_study\\final_detections\\final_model.json, so it needs the study to have
finished. Nothing in the existing code changes.

Steps for each recording, each one timed:
  read      read the sound file and resample it to 500 Hz (fp_replica.load_audio)
  aves      AVES fingerprints, an hour at a time, as in training (fp_replica.Embedder)
  classify  the final model's rumble probability for every 0.64 s frame
  alarms    the final alarm rule and the cut-off
  write     the Raven table

Ways to run it faster, to compare:
  --readers N         read and resample the next N recordings while AVES works (default 3;
                      0 = read each one only when it is needed)
  --half / --full     AVES in half precision (float16 on the GPU, bfloat16 on the CPU) or full;
                      half is the default on the GPU. With half, the first recording is also
                      run at full precision and both are compared
  --workers N         N recordings side by side in separate processes (CPU)
  --threads T         CPU threads per process (default: all cores / workers)

On the lab PC:

  python tools\\detect.py L:\\path\\to\\day.wav
  python tools\\detect.py L:\\path\\to\\folder --device cpu --workers 6 --threads 4

Results: <out> (default <source>\\Configurations\\_detect\\<date-time>\\): one
<recording>.rumbles.txt per recording, timing.md and timing.json.
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset_configs as dc  # noqa: E402
from dataset_configs import log, fmt_dur  # noqa: E402

FRAME_SECONDS = 0.64
SOUND_EXT = (".wav", ".flac", ".aif", ".aiff")
STEPS = ("read", "aves", "classify", "alarms", "write")


def find_recordings(inputs):
    out = []
    for p in inputs:
        if os.path.isdir(p):
            for d, _, fs in os.walk(p):
                out += [os.path.join(d, f) for f in sorted(fs) if f.lower().endswith(SOUND_EXT)]
        elif os.path.isfile(p):
            out.append(p)
        else:
            sys.exit(f"not found: {p}")
    return out


def ctx_of(name):
    m = re.search(r"_ctx(\d+)_", name)
    return int(m.group(1)) if m else 0


def load_final(path):
    """(experiment name, model path, rule, cut-off) from the study's final_model.json."""
    with open(path) as f:
        fm = json.load(f)
    for k in ("experiment", "model", "rule", "cut_off"):
        if k not in fm:
            sys.exit(f"{path} has no '{k}'; is it the study's final_model.json?")
    return fm["experiment"], fm["model"], fm["rule"], fm["cut_off"]


class Detector:
    """AVES + the final model + the alarm rule, loaded once per process."""

    def __init__(self, final, device, threads, fake_embedder=False, half=False):
        import torch
        import experiments as ex
        if threads:
            torch.set_num_threads(threads)
        self.name, model_path, self.rule, self.cut = load_final(final)
        self.ctx, self.device, self.half = ctx_of(self.name), device, half
        self.checked_half = False
        self.embed = dc.FakeEmbedder() if fake_embedder else self._aves(device)
        m = ex.make_model(self.ctx)
        m.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = m.eval().to(device)

    def fingerprints(self, y, half):
        import torch
        if half:
            dt = torch.float16 if self.device == "cuda" else torch.bfloat16
            with torch.autocast(self.device, dtype=dt):
                emb = self.embed.whole_file(y)
        else:
            emb = self.embed.whole_file(y)
        if self.device == "cuda":
            torch.cuda.synchronize()
        # the study's fingerprints were stored as float16; round the same way
        return emb.float().half().float()

    def alarms(self, prob):
        import improve as im
        starts, ends, scores = im.alarms(prob, self.rule) if len(prob) else ([], [], [])
        return [(a, b, s) for a, b, s in zip(starts, ends, scores) if s >= self.cut]

    def compare_full(self, y, emb_half, kept_half):
        """Fingerprints and alarms of half vs full precision on one recording."""
        import experiments as ex
        full = self.fingerprints(y, False)
        n = min(full.shape[0], emb_half.shape[0])
        cos = (full[:n] * emb_half[:n]).sum(1) / (full[:n].norm(dim=1) * emb_half[:n].norm(dim=1)).clamp_min(1e-12)
        kept_full = self.alarms(ex.day_probs(self.model, full, self.ctx, self.device)) if n else []
        same = len({(a, b) for a, b, _ in kept_full} & {(a, b) for a, b, _ in kept_half})
        c = {"cosine_mean": round(float(cos.mean()), 5) if n else None,
             "cosine_min": round(float(cos.min()), 5) if n else None,
             "alarms_full": len(kept_full), "alarms_half": len(kept_half), "alarms_identical": same}
        log(f"half precision vs full on the first recording: fingerprint similarity mean "
            f"{c['cosine_mean']}, worst {c['cosine_min']}; alarms {c['alarms_half']} vs {c['alarms_full']}, "
            f"{same} identical")
        return c

    @staticmethod
    def _aves(device):
        import fp_replica
        return fp_replica.Embedder(device)

    def read(self, path):
        import fp_replica
        t0 = time.time()
        y = fp_replica.load_audio(path)
        return y, time.time() - t0

    def run(self, path, y, out_dir, w):
        """Everything after reading; returns ({step: seconds}, alarms kept)."""
        import experiments as ex
        t, t0 = {}, time.time()
        emb = self.fingerprints(y, self.half)
        t["aves"] = time.time() - t0
        t0 = time.time()
        prob = ex.day_probs(self.model, emb, self.ctx, self.device) if emb.shape[0] else []
        t["classify"] = time.time() - t0
        t0 = time.time()
        kept = self.alarms(prob)
        t["alarms"] = time.time() - t0
        if self.half and not self.checked_half:
            self.checked_half = True
            self.half_check = self.compare_full(y, emb, kept)
        t0 = time.time()
        write_raven(w, os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + ".rumbles.txt"),
                    path, kept)
        t["write"] = time.time() - t0
        return t, len(kept)


def write_raven(w, dst, path, kept):
    with w.open(dst) as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)",
                     "High Freq (Hz)", "Begin Path", "File Offset (s)", "Begin File", "Score"])
        for i, (a, b, s) in enumerate(kept, 1):
            bt, et = a * FRAME_SECONDS, b * FRAME_SECONDS  # as the study's final_detections
            wr.writerow([i, "Spectrogram 1", 1, f"{bt:.3f}", f"{et:.3f}", 5, 250, path, f"{bt:.3f}",
                         os.path.basename(path), f"{s:.4f}"])


def hours_of(path):
    import soundfile as sf
    return sf.info(path).duration / 3600


def process_files(det, files, out_dir, w, readers, label=""):
    """Run det over files in order; with readers > 0, the next `readers` files are read and
    resampled in background threads while AVES works."""
    rows = []
    pool = ThreadPoolExecutor(readers) if readers > 0 and len(files) > 1 else None
    ahead = [pool.submit(det.read, f) for f in files[:readers + 1]] if pool else []
    for i, path in enumerate(files):
        t0 = time.time()
        if pool:
            y, read_s = ahead.pop(0).result()
            j = i + readers + 1
            if j < len(files):
                ahead.append(pool.submit(det.read, files[j]))
        else:
            y, read_s = det.read(path)
        wait_s = time.time() - t0
        t, n = det.run(path, y, out_dir, w)
        hours = y.shape[0] / 500 / 3600
        row = {"file": path, "hours": round(hours, 3), "alarms": n, "read": read_s, "read_wait": wait_s, **t,
               "wall": time.time() - t0}
        rows.append(row)
        log(f"{label}{os.path.basename(path)} ({hours:.1f} h): {n} alarms; "
            + ", ".join(f"{k} {row[k]:.1f} s" for k in STEPS) + f"; waited {wait_s:.1f} s for reading")
    if pool:
        pool.shutdown()
    return rows


def _worker(args):
    final, device, threads, files, out_dir, roots, readers, fake, half, k = args
    w = dc.Writer(roots, [])
    det = Detector(final, device, threads, fake, half)
    return process_files(det, files, out_dir, w, readers, label=f"[worker {k}] "), getattr(det, "half_check", None)


def summarize(rows, wall, a, device_name):
    hours = sum(r["hours"] for r in rows) or 1e-9
    per24 = {k: sum(r[k] for r in rows) / hours * 24 for k in STEPS + ("read_wait",)}
    s = {"files": len(rows), "audio_hours": round(hours, 2), "wall_s": round(wall, 1),
         "wall_s_per_24h": round(wall / hours * 24, 1), "device": device_name, "workers": a.workers,
         "threads": a.threads, "readers": a.readers, "half": a.half, "step_s_per_24h": {k: round(v, 2) for k, v in per24.items()},
         "files_detail": rows}
    if rows:
        s["median_file_wall_s"] = round(statistics.median(r["wall"] for r in rows), 1)
    return s


def write_report(w, out_dir, s, det_info):
    st = s["step_s_per_24h"]
    lines = [f"# Detector timing: {s['files']} recording(s), {s['audio_hours']} h of audio", "",
             f"Model `{det_info['experiment']}`, rule {det_info['rule']}, cut-off {det_info['cut_off']}.",
             f"Device {s['device']}, {s['workers']} process(es), {s['threads'] or 'all'} thread(s) each, "
             f"{s['readers']} recording(s) read ahead, AVES in {'half' if s['half'] else 'full'} precision.", "",
             "| Step | Time per 24 h of audio |", "|---|---|"]
    for k in STEPS:
        lines.append(f"| {k} | {fmt_dur(st[k])} |")
    lines += [f"| waited for reading | {fmt_dur(st['read_wait'])} |", "",
              f"**Wall clock: {fmt_dur(s['wall_s'])} in all, {fmt_dur(s['wall_s_per_24h'])} per 24 h of audio.** "
              "With several processes, step times add up across processes, so they can exceed the wall clock.", ""]
    c = s.get("half_check")
    if c:
        lines += [f"Half precision against full, on the first recording: fingerprint similarity {c['cosine_mean']} "
                  f"(worst {c['cosine_min']}); {c['alarms_half']} alarms against {c['alarms_full']}, "
                  f"{c['alarms_identical']} of them identical.", ""]
    with w.open(os.path.join(out_dir, "timing.md")) as f:
        f.write("\n".join(lines) + "\n")
    w.json(os.path.join(out_dir, "timing.json"), {**s, **det_info})
    print("\n".join(lines))


def main(argv=None):
    import multiprocessing as mp
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="sound files or folders")
    p.add_argument("--source", default=dc.DEFAULT_SOURCE)
    p.add_argument("--final", help="final_model.json (default <source>\\Configurations\\_study\\"
                                   "final_detections\\final_model.json)")
    p.add_argument("--out", help="where the Raven tables and timing go "
                                 "(default <source>\\Configurations\\_detect\\<date-time>)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--workers", type=int, default=1, help="recordings processed side by side")
    p.add_argument("--threads", type=int, default=0, help="CPU threads per process (0: all cores / workers)")
    p.add_argument("--readers", type=int, default=3, help="recordings read ahead in the background (0: none)")
    p.add_argument("--half", action="store_true", default=None,
                   help="AVES in half precision, compared with full on the first file (default on the GPU)")
    p.add_argument("--full", dest="half", action="store_false", help="AVES in full precision")
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    import torch
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if a.half is None:
        a.half = device == "cuda"  # identical alarms on the lab PC's RTX 4060, 3x faster AVES
    if device == "cuda" and a.workers > 1:
        sys.exit("--workers is for the CPU; on the GPU one process is fastest")
    if device == "cpu" and not a.threads:
        a.threads = max(1, (os.cpu_count() or 1) // a.workers)
    a.final = a.final or os.path.join(a.source, "Configurations", "_study", "final_detections", "final_model.json")
    if not os.path.exists(a.final):
        sys.exit(f"{a.final} not found: run tools\\study.py first, or give --final")
    out = a.out or os.path.join(a.source, "Configurations", "_detect", time.strftime("%Y-%m-%d_%H%M%S"))
    forbidden = [dc.ORIGINAL_DATA]
    try:
        root = dc.find_dataset_root(a.source)
        forbidden += [os.path.join(root, "Training"), os.path.join(root, "Testing")]
    except SystemExit:
        pass
    w = dc.Writer([out], forbidden)
    w.makedirs(out)
    files = find_recordings(a.inputs)
    if not files:
        sys.exit("no sound files found")
    name, model_path, rule, cut = load_final(a.final)
    det_info = {"experiment": name, "model": model_path, "rule": rule, "cut_off": cut}
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else f"CPU ({os.cpu_count()} logical cores)"
    log(f"{len(files)} recording(s); model {name}; {dev_name}; {a.workers} process(es) x "
        f"{a.threads or 'default'} thread(s); {a.readers} read ahead; {'half' if a.half else 'full'} precision")
    t0 = time.time()
    if a.workers <= 1:
        det = Detector(a.final, device, a.threads, a.fake_embedder, a.half)
        load_s = time.time() - t0
        log(f"models loaded in {load_s:.1f} s")
        t0 = time.time()
        rows = process_files(det, files, out, w, a.readers)
        checks = [getattr(det, "half_check", None)]
    else:
        shares = [files[k::a.workers] for k in range(a.workers)]
        jobs = [(a.final, device, a.threads, sh, out, [out], a.readers, a.fake_embedder, a.half, k)
                for k, sh in enumerate(shares) if sh]
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            parts = pool.map(_worker, jobs)
        rows = [r for part, _ in parts for r in part]
        checks = [c for _, c in parts]
    wall = time.time() - t0
    s = summarize(rows, wall, a, dev_name)
    s["half_check"] = next((c for c in checks if c), None)
    write_report(w, out, s, det_info)
    log(f"wrote {len(rows)} Raven table(s), timing.md and timing.json in {out}")
    return s


if __name__ == "__main__":
    main()
