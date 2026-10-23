#!/usr/bin/env python
"""
How fast is the detector on a CPU? A one-off measurement (about an hour) on one real
24-hour recording, to check ELP's target (3 months x 50 sites in under 5 days on 30 CPUs).

Nearly all the time is AVES (the classifier takes about 0.1 s a day), so this times:

  1  reading + resampling the 24 h file (what happens before AVES)
  2  AVES on the CPU, the way the detector runs it (1-hour pieces), with 1, 4, 8, 16 and
     all threads, on a 1-hour excerpt
  3  several AVES processes side by side (each with 4 threads): the throughput of a
     whole machine working through a folder
  4  shorter pieces (5 min) in batches, and how close their fingerprints are
  5  8-bit AVES weights, and how close their fingerprints are
  6  the GPU, for comparison (if there is one)

and estimates the time for 4,500 recording-days on 30 cores. It writes nothing outside
<out>\\_cpu_speed (cpu_speed.md, cpu_speed.json). Run it when nothing else is using the PC:

  python tools\\cpu_speed.py
"""
import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset_configs as dc  # noqa: E402
from dataset_configs import log, fmt_dur  # noqa: E402

SR = 500
HOP = 320                  # samples per fingerprint (0.64 s)
TARGET_DAYS = 4500         # 3 months x 50 sites
TARGET_CORES = 30
TARGET_LIMIT_DAYS = 5


def unit(e):
    return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def cosine(a, b):
    n = min(a.shape[0], b.shape[0])
    return (unit(a[:n].float()) * unit(b[:n].float())).sum(1)


def embed_pieces(aves, y, piece_frames=470, ctx_frames=32, batch=8):
    """AVES on ~5-minute pieces (with ~20 s of context on each side), several at once."""
    import torch
    n = y.shape[0] // HOP
    pieces = []
    for f0 in range(0, n, piece_frames):
        f1 = min(n, f0 + piece_frames)
        a, b = max(0, f0 - ctx_frames), min(n, f1 + ctx_frames + 1)
        pieces.append((f0, f1, a, b))
    out = [None] * len(pieces)
    by_len = {}  # pieces of the same length go through AVES together (the first and last are shorter)
    for i, p in enumerate(pieces):
        by_len.setdefault(p[3] - p[2], []).append(i)
    with torch.inference_mode():
        for same in by_len.values():
            for k in range(0, len(same), batch):
                idx = same[k:k + batch]
                x = torch.stack([y[pieces[i][2] * HOP:pieces[i][3] * HOP] for i in idx])
                e = aves(x)
                for j, i in enumerate(idx):
                    f0, f1, a, _ = pieces[i]
                    out[i] = e[j, f0 - a:f1 - a]
    return unit(torch.cat(out).float())


def _worker(args):
    """One of several side-by-side processes: load AVES, wait for the others, embed the excerpt."""
    path, threads, barrier = args
    import numpy as np
    import torch
    torch.set_num_threads(threads)
    import fp_replica
    y = torch.from_numpy(np.load(path))
    e = fp_replica.Embedder("cpu")
    barrier.wait()  # all start together, so they really compete for the cores
    t0 = time.time()
    e.whole_file(y)
    return time.time() - t0


def main(argv=None):
    import numpy as np
    import soundfile as sf
    import torch
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=dc.DEFAULT_SOURCE)
    p.add_argument("--out", help="the Configurations folder (default <source>\\Configurations)")
    p.add_argument("--excerpt-minutes", type=float, default=60, help="audio used for the CPU timings")
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.out = a.out or os.path.join(a.source, "Configurations")
    root = dc.find_dataset_root(a.source)
    dst = os.path.join(a.out, "_cpu_speed")
    w = dc.Writer([dst], [dc.ORIGINAL_DATA, os.path.join(root, "Training"), os.path.join(root, "Testing")])
    w.makedirs(dst)
    stdout = sys.stdout
    sys.stdout = dc.Tee(w.check(os.path.join(dst, "run_log.txt")))
    res = {"cpu": None, "logical_cores": os.cpu_count(), "torch_threads_default": torch.get_num_threads()}
    try:
        import platform
        res["cpu"] = platform.processor() or platform.machine()
        pools = dc.load_pools(root)
        # the longest validation recording (Complex/General are 24 h days)
        cands = [(sf.info(pools[pn].sounds[k]).duration, pn, k) for pn in ("complex", "general")
                 for k in pools[pn].files]
        dur, pool, key = max(cands)
        path = pools[pool].sounds[key]
        res["file"] = {"path": path, "hours": round(dur / 3600, 2), "bytes": os.path.getsize(path),
                       "sample_rate": sf.info(path).samplerate}
        log(f"CPU speed test on {path} ({dur / 3600:.1f} h, {os.path.getsize(path) / 2**30:.2f} GB)")

        # 1 reading + resampling
        t0 = time.time()
        y = dc.load_audio(path)
        res["read_resample_s_per_24h"] = round((time.time() - t0) * 24 / (dur / 3600), 1)
        log(f"1 reading + resampling: {res['read_resample_s_per_24h']} s per 24 h")
        n_ex = int(a.excerpt_minutes * 60 * SR) // HOP * HOP
        ex_y = y[:n_ex].clone()
        ex_hours = n_ex / SR / 3600
        ex_path = os.path.join(dst, "excerpt.npy")
        np.save(w.check(ex_path), ex_y.numpy())

        if a.fake_embedder:
            embedder, aves = dc.FakeEmbedder(), None
        else:
            import fp_replica
            embedder = fp_replica.Embedder("cpu")
            aves = embedder.aves

        def per24(seconds):
            return seconds / ex_hours * 24

        # 2 threads
        res["threads"] = {}
        ref = None
        n_logical = os.cpu_count() or 4
        for th in sorted({1, 4, 8, 16, n_logical} & set(range(1, n_logical + 1))):
            torch.set_num_threads(th)
            t0 = time.time()
            e = embedder.whole_file(ex_y)
            s = per24(time.time() - t0)
            ref = e if ref is None else ref
            res["threads"][th] = {"s_per_24h": round(s, 1), "core_min_per_24h": round(s * th / 60, 1)}
            log(f"2 AVES on CPU, {th} threads: {fmt_dur(s)} per 24 h ({s * th / 60:.0f} thread-minutes)")
        torch.set_num_threads(n_logical)

        def optional(step, fn):
            try:
                fn()
            except Exception as err:  # one failed extra test must not lose the rest
                res.setdefault("failed", {})[step] = repr(err)
                log(f"{step} failed, skipped: {err!r}")

        # 3 side-by-side processes, 4 threads each
        def parallel():
            n_proc = max(1, n_logical // 4)
            t0 = time.time()
            ctx = mp.get_context("spawn")
            with ctx.Manager() as mgr, ctx.Pool(n_proc) as pool_:
                barrier = mgr.Barrier(n_proc)
                # a generous time limit, so a crashed process can't hang the test forever
                limit = 600 + 20 * res["threads"][1]["s_per_24h"] * ex_hours / 24
                secs = pool_.map_async(_worker, [(ex_path, 4, barrier)] * n_proc).get(timeout=limit)
            wall = time.time() - t0
            hours_done = n_proc * ex_hours
            res["parallel"] = {"processes": n_proc, "threads_each": 4, "wall_s": round(wall, 1),
                               "slowest_s": round(max(secs), 1), "audio_hours": round(hours_done, 2),
                               "machine_s_per_24h": round(max(secs) / hours_done * 24, 1)}
            log(f"3 {n_proc} processes x 4 threads: the whole PC does 24 h of audio every "
                f"{fmt_dur(res['parallel']['machine_s_per_24h'])}")
        if not a.fake_embedder:
            optional("3 parallel", parallel)

        # 4 shorter pieces
        def pieces():
            for piece_min in (5, 10):
                pf = int(piece_min * 60 * SR / HOP)
                t0 = time.time()
                e = embed_pieces(aves, ex_y, piece_frames=pf)
                s = per24(time.time() - t0)
                c = cosine(e, ref)
                res[f"pieces_{piece_min}min"] = {"s_per_24h": round(s, 1), "cos_mean": round(float(c.mean()), 5),
                                                  "cos_min": round(float(c.min()), 5)}
                log(f"4 {piece_min}-minute pieces, batched, all threads: {fmt_dur(s)} per 24 h; "
                    f"fingerprints vs 1-hour pieces: cosine {float(c.mean()):.4f} (min {float(c.min()):.4f})")

        # 5 8-bit
        def int8():
            import fp_replica
            q = torch.ao.quantization.quantize_dynamic(aves, {torch.nn.Linear}, dtype=torch.qint8)
            qe = fp_replica.Embedder.__new__(fp_replica.Embedder)
            qe.device, qe.aves = "cpu", q
            t0 = time.time()
            e = qe.whole_file(ex_y)
            s = per24(time.time() - t0)
            c = cosine(e, ref)
            res["int8"] = {"s_per_24h": round(s, 1), "cos_mean": round(float(c.mean()), 5), "cos_min": round(float(c.min()), 5)}
            log(f"5 8-bit AVES, all threads: {fmt_dur(s)} per 24 h; cosine {float(c.mean()):.4f}")
        if aves is not None:
            optional("4 pieces", pieces)
            optional("5 int8", int8)

        # 6 GPU
        def gpu():
            import fp_replica
            ge = fp_replica.Embedder("cuda")
            ge.whole_file(ex_y[:SR * 600])
            torch.cuda.synchronize()
            t0 = time.time()
            ge.whole_file(y)
            torch.cuda.synchronize()
            res["gpu_s_per_24h"] = round((time.time() - t0) * 24 / (dur / 3600), 1)
            res["gpu"] = torch.cuda.get_device_name(0)
            log(f"6 GPU ({res['gpu']}): {res['gpu_s_per_24h']} s per 24 h (AVES only)")
        if torch.cuda.is_available() and not a.fake_embedder:
            optional("6 GPU", gpu)

        # the estimate
        best_thread = min(v["core_min_per_24h"] for v in res["threads"].values())
        par = res.get("parallel")
        if par:
            core_min = par["machine_s_per_24h"] / 60 * n_logical
        else:
            core_min = best_thread
        read_core_min = res["read_resample_s_per_24h"] / 60
        per_day_core_min = core_min + read_core_min
        days = TARGET_DAYS * per_day_core_min / TARGET_CORES / 60 / 24
        res["estimate"] = {"core_minutes_per_24h": round(per_day_core_min, 1), "days_for_target": round(days, 2),
                           "note": f"{TARGET_DAYS} recording-days on {TARGET_CORES} cores like these "
                                   f"({n_logical} logical cores counted as cores)"}
        t = ["# CPU speed of the detector", "",
             f"Measured on {res['cpu']} ({n_logical} logical cores), file `{os.path.basename(path)}` "
             f"({res['file']['hours']} h, {res['file']['bytes'] / 2**30:.2f} GB). AVES is nearly all the time; "
             "the classifier adds about 0.1 s per 24 h.", "",
             "| Test | Time per 24 h of audio | Notes |", "|---|---|---|",
             f"| Reading + resampling | {fmt_dur(res['read_resample_s_per_24h'])} | one thread |"]
        for th, v in res["threads"].items():
            t.append(f"| AVES, {th} thread{'s' if th > 1 else ''} (1-hour pieces, as the detector does it) | {fmt_dur(v['s_per_24h'])} | "
                     f"{v['core_min_per_24h']} thread-minutes |")
        if par:
            t.append(f"| {par['processes']} files side by side, 4 threads each | {fmt_dur(par['machine_s_per_24h'])} "
                     "for the whole PC | throughput on a folder of files |")
        for k in ("pieces_5min", "pieces_10min", "int8"):
            if k in res:
                t.append(f"| {k.replace('_', ' ')} | {fmt_dur(res[k]['s_per_24h'])} | fingerprints vs normal: cosine "
                         f"{res[k]['cos_mean']:.4f} (worst {res[k]['cos_min']:.4f}) |")
        if "gpu_s_per_24h" in res:
            t.append(f"| GPU ({res['gpu']}) | {fmt_dur(res['gpu_s_per_24h'])} | AVES only |")
        t += ["", f"**Estimate:** about {per_day_core_min:.0f} core-minutes per 24 h of audio, including reading. "
                  f"{TARGET_DAYS} recording-days on {TARGET_CORES} cores like these: about **{days:.1f} days** "
                  f"(target: under {TARGET_LIMIT_DAYS}). Logical cores (threads) are counted as cores, and the "
                  "lab's processors may be slower or faster than this one."]
        with w.open(os.path.join(dst, "cpu_speed.md")) as f:
            f.write("\n".join(t) + "\n")
        w.json(os.path.join(dst, "cpu_speed.json"), res)
        print("\n".join(t))
        try:
            os.remove(w.check(ex_path))
        except OSError:
            pass
    finally:
        sys.stdout = stdout


if __name__ == "__main__":
    mp.freeze_support()
    main()
