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
  --prefetch          read the next recording while AVES works on this one (default on)
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

    def __init__(self, final, device, threads, fake_embedder=False):
        import torch
        import experiments as ex
        if threads:
            torch.set_num_threads(threads)
        self.name, model_path, self.rule, self.cut = load_final(final)
        self.ctx, self.device = ctx_of(self.name), device
        self.embed = dc.FakeEmbedder() if fake_embedder else self._aves(device)
        m = ex.make_model(self.ctx)
        m.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = m.eval().to(device)

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
        import torch
        import experiments as ex
        import improve as im
        t, t0 = {}, time.time()
        emb = self.embed.whole_file(y)
        if self.device == "cuda":
            torch.cuda.synchronize()
        # the study's fingerprints were stored as float16; round the same way
        emb = emb.half().float()
        t["aves"] = time.time() - t0
        t0 = time.time()
        prob = ex.day_probs(self.model, emb, self.ctx, self.device) if emb.shape[0] else []
        t["classify"] = time.time() - t0
        t0 = time.time()
        starts, ends, scores = im.alarms(prob, self.rule) if len(prob) else ([], [], [])
        kept = [(a, b, s) for a, b, s in zip(starts, ends, scores) if s >= self.cut]
        t["alarms"] = time.time() - t0
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


def process_files(det, files, out_dir, w, prefetch, label=""):
    """Run det over files in order; with prefetch, the next file is read while AVES works."""
    rows = []
    pool = ThreadPoolExecutor(1) if prefetch and len(files) > 1 else None
    nxt = pool.submit(det.read, files[0]) if pool else None
    for i, path in enumerate(files):
        t0 = time.time()
        if pool:
            y, read_s = nxt.result()
            if i + 1 < len(files):
                nxt = pool.submit(det.read, files[i + 1])
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
    final, device, threads, files, out_dir, roots, prefetch, fake, k = args
    w = dc.Writer(roots, [])
    det = Detector(final, device, threads, fake)
    return process_files(det, files, out_dir, w, prefetch, label=f"[worker {k}] ")


def summarize(rows, wall, a, device_name):
    hours = sum(r["hours"] for r in rows) or 1e-9
    per24 = {k: sum(r[k] for r in rows) / hours * 24 for k in STEPS + ("read_wait",)}
    s = {"files": len(rows), "audio_hours": round(hours, 2), "wall_s": round(wall, 1),
         "wall_s_per_24h": round(wall / hours * 24, 1), "device": device_name, "workers": a.workers,
         "threads": a.threads, "prefetch": a.prefetch, "step_s_per_24h": {k: round(v, 2) for k, v in per24.items()},
         "files_detail": rows}
    if rows:
        s["median_file_wall_s"] = round(statistics.median(r["wall"] for r in rows), 1)
    return s


def write_report(w, out_dir, s, det_info):
    st = s["step_s_per_24h"]
    lines = [f"# Detector timing: {s['files']} recording(s), {s['audio_hours']} h of audio", "",
             f"Model `{det_info['experiment']}`, rule {det_info['rule']}, cut-off {det_info['cut_off']}.",
             f"Device {s['device']}, {s['workers']} process(es), {s['threads'] or 'all'} thread(s) each, "
             f"reading ahead {'on' if s['prefetch'] else 'off'}.", "",
             "| Step | Time per 24 h of audio |", "|---|---|"]
    for k in STEPS:
        lines.append(f"| {k} | {fmt_dur(st[k])} |")
    lines += [f"| waited for reading | {fmt_dur(st['read_wait'])} |", "",
              f"**Wall clock: {fmt_dur(s['wall_s'])} in all, {fmt_dur(s['wall_s_per_24h'])} per 24 h of audio.** "
              "With several processes, step times add up across processes, so they can exceed the wall clock.", ""]
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
    p.add_argument("--no-prefetch", dest="prefetch", action="store_false",
                   help="read each recording only when the previous one is done")
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    import torch
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and a.workers > 1:
        sys.exit("--workers is for the CPU; on the GPU one process is fastest")
    if device == "cpu" and not a.threads:
        a.threads = max(1, (os.cpu_count() or 1) // a.workers)
    a.final = a.final or os.path.join(a.source, "Configurations", "_study", "final_detections", "final_model.json")
    if not os.path.exists(a.final):
        sys.exit(f"{a.final} not found: run tools\\study.py first, or give --final")
    out = a.out or os.path.join(a.source, "Configurations", "_detect", time.strftime("%Y-%m-%d_%H%M"))
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
        f"{a.threads or 'default'} thread(s); reading ahead {'on' if a.prefetch else 'off'}")
    t0 = time.time()
    if a.workers <= 1:
        det = Detector(a.final, device, a.threads, a.fake_embedder)
        load_s = time.time() - t0
        log(f"models loaded in {load_s:.1f} s")
        t0 = time.time()
        rows = process_files(det, files, out, w, a.prefetch)
    else:
        shares = [files[k::a.workers] for k in range(a.workers)]
        jobs = [(a.final, device, a.threads, sh, out, [out], a.prefetch, a.fake_embedder, k)
                for k, sh in enumerate(shares) if sh]
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            rows = [r for part in pool.map(_worker, jobs) for r in part]
    wall = time.time() - t0
    s = summarize(rows, wall, a, dev_name)
    write_report(w, out, s, det_info)
    log(f"wrote {len(rows)} Raven table(s), timing.md and timing.json in {out}")
    return s


if __name__ == "__main__":
    main()
