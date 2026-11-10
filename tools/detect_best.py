#!/usr/bin/env python
"""
Run round 2's best detector on recordings: late_ens + EfficientNet x5 by default, i.e.
the 5 AVES heads and the 5 Dasheng heads averaged into one score every 0.64 s, the late_ens
alarm rule, then every alarm re-scored by 5 EfficientNet-B0 checkers (averaged) on a 32 s
spectrogram around it, combined with the first stage's score as round 2 chose on validation.
The tables keep the alarms at the cut-off that found 80% of the rumbles on the validation days.

Same output layout as detect.py, so compare_hori_harm.py reads it as it is:
  <out>\\<input folder>\\<subfolders>\\<name>.rumbles.txt   Raven table at the cut-off
  <name>.probs.npy    the first stage's probability every 0.64 s (float16)
  <name>.events.npy   every alarm of the rule, at any score: [first frame, last frame,
                      first-stage score, final score] (so other cut-offs need no rerun)
  timings.csv, summary.md (time per step and per 24 h, projections), detector.json, run_log.txt

Commands (lab PC, from the repository folder, the round 2 venv):
  export  copy the chosen model out of Configurations\\_round2 into one folder (once; run does it
          too when the folder is missing), and check on two test recordings that this script
          gives back round 2's own scores from the raw audio.
  run     detect. --hh TABLE keeps only the recordings in that Hori-Harm table (the days its
          analysts reviewed). --sample N keeps N of them, spread over the sites, for timing.
          Rerunning skips recordings already done, so a sample run is the start of the full run.

  python tools\\detect_best.py run L:\\...\\nn_202306_jun_dep17 --hh HH_TABLE.txt --sample 12 --out OUT
  python tools\\detect_best.py run L:\\...\\nn_202306_jun_dep17 --hh HH_TABLE.txt --out OUT
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import random
import shutil
import statistics
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset_configs as dc  # noqa: E402
import detect  # noqa: E402
from dataset_configs import log, fmt_dur, FRAME_SECONDS  # noqa: E402

DEFAULT_NAME = "late_ens+efficientnet_b0_x5"
STAGE_ENCODERS = {"aves_ens": ["aves"], "dasheng_ens": ["dasheng"], "late_ens": ["aves", "dasheng"]}
EVENTS_SUFFIX = ".events.npy"
TABLE_SUFFIX, PROBS_SUFFIX = detect.TABLE_SUFFIX, detect.PROBS_SUFFIX
TIMING_COLUMNS = ["finished", "site", "file", "hours", "frames", "read_resample_s", "waited_s", "aves_s",
                  "dasheng_s", "heads_s", "cuts_s", "checker_s", "write_s", "total_s", "alarms", "detections"]


def default_configs():
    return os.path.join(dc.DEFAULT_SOURCE, "Configurations")


################################################################################
# export: round 2's model into one folder
################################################################################

def open_round2(source, fake, seeds=None):
    import round2 as r2
    import study as st
    out = os.path.join(source, "Configurations")
    a = argparse.Namespace(source=source, out=out, cache=os.path.join(out, "_embedding_cache"),
                           seeds=seeds or st.SEEDS, seed=0, fake_embedder=fake, archs=r2.ARCHS, skip=None)
    root = dc.find_dataset_root(source)
    w = dc.Writer([os.path.join(out, r2.ROOT_NAME)], [dc.ORIGINAL_DATA])
    return r2.R2(a, dc.load_pools(root), w)


def parse_name(name):
    """'late_ens+efficientnet_b0_x5' -> ('late_ens', 'efficientnet_b0'); 'late_ens' -> ('late_ens', None)."""
    stage, _, rest = name.partition("+")
    if stage not in STAGE_ENCODERS:
        sys.exit(f"{name}: only {', '.join(STAGE_ENCODERS)} first stages (with or without a checker) are supported")
    if rest and not rest.endswith("_x5"):
        sys.exit(f"{name}: only the averaged checkers (+<arch>_x5) are supported")
    return stage, (rest[:-3] if rest else None)


def cmd_export(a):
    import torch
    lab = open_round2(a.source, a.fake_embedder, a.seeds)
    stage, arch = parse_name(a.name)
    sr = lab.jload(os.path.join(lab.dir(stage), "results.json"))
    if not sr:
        sys.exit(f"round 2 has no results for {stage} yet")
    r = lab.jload(os.path.join(lab.dir("checked"), f"{a.name}.json")) if arch else sr
    if not r:
        sys.exit(f"round 2 has no results for {a.name}: is it in round2.md?")
    final = (lab.jload(os.path.join(lab.root, "final.json")) or {}).get(a.name, {}).get("sets", {})
    os.makedirs(a.model, exist_ok=True)
    heads = []
    for enc in STAGE_ENCODERS[stage]:
        for i, m in enumerate(lab.base_models(enc)):
            f = f"{enc}_head{i}.pth"
            torch.save(m.state_dict(), os.path.join(a.model, f))
            heads.append({"encoder": enc, "file": f, "kind": "final" if enc == "aves" else "ctx", "dim": 768})
    checkers = []
    if arch:
        for s in lab.seeds:
            f = f"{arch}_checker{s}.pth"
            shutil.copyfile(os.path.join(lab.dir("checkers"), f"{arch}_seed{s}.pth"), os.path.join(a.model, f))
            checkers.append(f)
    per = r["summary"]["per_seed"][0]
    info = {"name": a.name, "title": r.get("title"), "stage": stage, "ctx": lab.base_spec["ctx"], "rule": r["rule"],
            "arch": arch, "combine": r.get("combine"), "heads": heads, "checkers": checkers,
            "cut_off": (per.get("at_80") or {}).get("threshold"), "cut_off_85": (per.get("at_85") or {}).get("threshold"),
            "validation_score": r["summary"]["score"], "tests": final, "aves_final_model": lab.final_name,
            "exported": time.strftime("%Y-%m-%d %H:%M"), "from": lab.root, "fake": bool(a.fake_embedder)}
    if info["cut_off"] is None:
        sys.exit(f"{a.name} never reached 80% recall on validation: no cut-off to use")
    with open(os.path.join(a.model, "model.json"), "w") as f:
        json.dump(info, f, indent=1)
    log(f"exported {a.name} to {a.model}: {len(heads)} heads, {len(checkers)} checkers, cut-off {info['cut_off']:.4f}")
    check_against_round2(lab, a.model, a.fake_embedder)


def check_against_round2(lab, model_dir, fake, n=2):
    """From the raw audio of n test recordings, this script's first-stage probabilities and alarm
    scores must equal round 2's (which used the cached fingerprints)."""
    import numpy as np
    import improve as im
    det = BestDetector(model_dir, fake=fake, device=lab.device)
    split = next(s["key"] for s in lab.lab.cfg["tests"] if s["files"])
    keys = [(p, k) for p, k in lab.split_keys(split) if lab.pools[p].sounds.get(k)][:n]
    worst_p = worst_s = 0.0
    for p, k in keys:
        n_steps = dc.Day(lab.a.cache, p, k).n
        ref = []
        for enc in STAGE_ENCODERS[det.info["stage"]]:
            x = lab.src[enc].day(p, k).whole()
            ref += [lab_fit(lab, ex_probs(m, x, det.ctx, lab.device), n_steps) for m in lab.base_models(enc)]
        ref = np.mean(ref, 0)
        y500, y1k, _ = load_two(lab.pools[p].sounds[k])
        prob, ev, _ = det.score(y500, y1k)
        worst_p = max(worst_p, float(np.abs(prob[:len(ref)] - ref).max()) if len(ref) else 0.0)
        s, e, v = im.alarms(ref, det.rule)
        if det.checkers:
            cuts = lab.cuts.get(split, p, k, list(zip(s, e)))
            v = det.combine(np.asarray(v, np.float64), det.check(cuts))
        mine = {(int(a), int(b)): sc for a, b, _, sc in ev}
        common = [(int(a), int(b), sc) for a, b, sc in zip(s, e, v) if (int(a), int(b)) in mine]
        if common:
            worst_s = max(worst_s, max(abs(mine[(a, b)] - sc) for a, b, sc in common))
        log(f"check {p}/{k}: {len(prob)} steps (round 2: {len(ref)}), {len(ev)} alarms (round 2: {len(s)}, "
            f"{len(common)} identical boxes)")
    ok = worst_p < 0.02 and worst_s < 0.02
    log(f"check against round 2: largest difference {worst_p:.4f} in probability, {worst_s:.4f} in alarm score: "
        + ("OK" if ok else "TOO LARGE, tell Claude before trusting the results"))
    return worst_p, worst_s


def lab_fit(lab, x, n):
    import round2 as r2
    return r2.fit(x, n)


def ex_probs(m, x, ctx, device):
    import experiments as ex
    return ex.day_probs(m, x, ctx, device)


################################################################################
# the detector
################################################################################

def load_two(path, block_seconds=3600, pad_seconds=10):
    """The recording at 500 Hz (AVES, checkers) and 1 kHz (Dasheng) from one read of the file,
    each exactly as fp_replica.load_audio makes it; and the seconds it took."""
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    t0 = time.time()
    outs = {500: [], 1000: []}
    with sf.SoundFile(dc.long_path(path)) as f:
        sr, total = f.samplerate, f.frames
        block, pad = block_seconds * sr, pad_seconds * sr
        for start in range(0, total, block):
            a, b = max(0, start - pad), min(total, start + block + pad)
            f.seek(a)
            x = f.read(b - a, dtype="float32", always_2d=True)[:, 0]
            for rate, out in outs.items():
                ratio = Fraction(rate, sr)
                up, down = ratio.numerator, ratio.denominator
                y = resample_poly(x, up, down)
                skip = (start - a) * up // down
                keep = (min(start + block, total) - start) * up // down
                out.append(torch.from_numpy(y[skip:skip + keep].astype(np.float32)))
    res = [torch.cat(outs[r]).to(torch.float16).to(torch.float32) if outs[r] else torch.zeros(0) for r in (500, 1000)]
    return res[0], res[1], time.time() - t0


class BestDetector:
    def __init__(self, model_dir, fake=False, device=None):
        import torch
        import candidates as cd
        import experiments as ex
        import round2 as r2
        self.info = json.load(open(os.path.join(model_dir, "model.json")))
        self.ctx, self.rule, self.cut_off = self.info["ctx"], self.info["rule"], self.info["cut_off"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.heads = {}
        for h in self.info["heads"]:
            m = ex.make_model(self.ctx) if h["kind"] == "final" else cd.make_ctx_model(h["dim"], self.ctx)
            m.load_state_dict(torch.load(os.path.join(model_dir, h["file"]), map_location="cpu"))
            self.heads.setdefault(h["encoder"], []).append(m.eval().to(self.device))
        self.checkers = []
        for f in self.info["checkers"]:
            m = r2.make_checker(self.info["arch"], pretrained=False)
            m.load_state_dict(torch.load(os.path.join(model_dir, f), map_location="cpu"))
            self.checkers.append(m.eval().to(self.device))
        self.enc = {}
        if "aves" in self.heads:
            self.enc["aves"] = dc.get_embedder(fake)
        if "dasheng" in self.heads:
            self.enc["dasheng"] = cd.FakeEncoder(1000) if fake else cd.DashengEmbedder(self.device)
        self.times = {}
        log(f"model {self.info['name']} ({self.info.get('title')}): {sum(map(len, self.heads.values()))} heads, "
            f"{len(self.checkers)} checkers ({self.info.get('combine')}), rule {self.rule}, "
            f"cut-off {self.cut_off:.4f}, on {self.device}")

    def check(self, cuts):
        import numpy as np
        import candidates as cd
        if not len(cuts):
            return np.zeros(0)
        return np.mean([cd.verifier_scores(m, cuts, self.device) for m in self.checkers], 0)

    def combine(self, v, pv):
        import numpy as np
        return pv if self.info["combine"] == "verifier" else np.sqrt(np.clip(v, 0, 1) * pv)

    def score(self, y500, y1k):
        """(first-stage probability per step, events [n, 4], seconds per step)."""
        import numpy as np
        import torch
        import candidates as cd
        import improve as im
        import round2 as r2
        t = {}
        probs, n = [], None
        for enc in ("aves", "dasheng"):
            if enc not in self.heads:
                continue
            t0 = time.time()
            x = self.enc[enc].whole_file(y500 if enc == "aves" else y1k).half()   # cached as float16 in round 2
            if self.device == "cuda":
                torch.cuda.synchronize()
            t[f"{enc}_s"] = time.time() - t0
            t0 = time.time()
            n = x.shape[0] if n is None else n                                    # AVES's step count rules
            probs += [r2.fit(ex_probs(m, x, self.ctx, self.device), n) if x.shape[0] else np.zeros(n, np.float32)
                      for m in self.heads[enc]]
            t["heads_s"] = t.get("heads_s", 0.0) + time.time() - t0
        prob = np.mean(probs, 0).astype(np.float32) if n else np.zeros(0, np.float32)
        t0 = time.time()
        s, e, v = im.alarms(prob, self.rule) if len(prob) else ([], [], [])
        v = np.asarray(v, np.float64)
        t["heads_s"] = t.get("heads_s", 0.0) + time.time() - t0
        final = v
        if self.checkers and len(v):
            t0 = time.time()
            y = y500.numpy()
            cuts = np.stack([cd.spectrogram_cut(y, (int(a) + int(b) + 1) / 2 * FRAME_SECONDS) for a, b in zip(s, e)])
            t["cuts_s"] = time.time() - t0
            t0 = time.time()
            final = self.combine(v, self.check(cuts))
            t["checker_s"] = time.time() - t0
        ev = np.stack([np.asarray(s, np.float64), np.asarray(e, np.float64), v, final], 1) if len(v) else np.zeros((0, 4))
        return prob, ev, t


def event_rows(ev, min_score, src):
    return [(a * FRAME_SECONDS, b * FRAME_SECONDS, src, a * FRAME_SECONDS, sc)
            for a, b, _, sc in ev if sc >= min_score]


################################################################################
# choosing the recordings
################################################################################

def hh_files(path):
    """Recording names in a Hori-Harm table, from the rows an analyst counted (as compare_hori_harm)."""
    names = set()
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if (r.get("Count") or "").strip():
                names.add(r["Begin File"].strip().lower())
    return names


def spread(files, n, seed=17):
    """n recordings taken in turn from each site, at random within a site."""
    by = {}
    for x in files:
        by.setdefault(detect.site_of(x[1]), []).append(x)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    out, sites = [], sorted(by)
    while len(out) < n and any(by.values()):
        for s in sites:
            if by[s] and len(out) < n:
                out.append(by[s].pop())
    return sorted(out, key=lambda x: x[1])


################################################################################
# run
################################################################################

def cmd_run(a):
    import improve as im
    im.keep_awake()
    detect.no_quick_edit()
    os.makedirs(a.out, exist_ok=True)
    stdout, sys.stdout = sys.stdout, dc.Tee(os.path.join(a.out, "run_log.txt"))
    try:
        run(a)
    finally:
        sys.stdout.f.close()
        sys.stdout = stdout


def run(a):
    import numpy as np
    if not os.path.exists(os.path.join(a.model, "model.json")):
        log(f"{a.model} has no model yet: exporting {a.name} from round 2 first")
        cmd_export(a)
    det = BestDetector(a.model, fake=a.fake_embedder)
    files = detect.find_audio(a.inputs)
    log(f"{len(files)} recordings under {', '.join(a.inputs)}")
    if a.hh:
        want = hh_files(a.hh)
        files = [x for x in files if os.path.basename(x[1]).lower() in want]
        log(f"{len(files)} of them are in the Hori-Harm table ({len(want)} recordings listed there)")
    pool_files = files
    if a.sample:
        files = spread(files, a.sample)
        log(f"sample: {len(files)} recordings from {len({detect.site_of(p) for _, p in files})} sites")
    jobs = [(p, detect.out_base(a.out, r, p)) for r, p in files]
    todo = [(p, b) for p, b in jobs if a.redo or not os.path.exists(dc.long_path(b + TABLE_SUFFIX))]
    log(f"{len(jobs) - len(todo)} already done, {len(todo)} to do; results in {a.out}")
    min_score = det.cut_off if a.min_score is None else a.min_score
    info = {"experiment": det.info["name"], "kind": "checked" if det.checkers else "stage", "rule": det.rule,
            "cut_off": det.cut_off, "min_score_in_tables": min_score, "model": a.model, "inputs": a.inputs,
            "hh_table": a.hh, "machine": detect.machine(), "model_info": {k: det.info[k] for k in (
                "title", "combine", "validation_score", "tests", "exported") if k in det.info}}
    with open(os.path.join(a.out, "detector.json"), "w") as f:
        json.dump(info, f, indent=1)
    tpath = os.path.join(a.out, "timings.csv")
    new = not os.path.exists(tpath)
    tf = open(tpath, "a", newline="")
    tw = csv.writer(tf)
    if new:
        tw.writerow(TIMING_COLUMNS)
    pool = cf.ThreadPoolExecutor(max(1, a.prefetch))
    ahead = {}

    def queue(i):
        if i < len(todo) and i not in ahead:
            ahead[i] = pool.submit(load_two, todo[i][0])

    for i in range(min(len(todo), a.prefetch)):
        queue(i)
    t_start, done_h, failed = time.time(), 0.0, 0
    for i, (src, base) in enumerate(todo):
        queue(i + a.prefetch)
        try:
            t0 = time.time()
            y500, y1k, t_load = ahead.pop(i).result()
            waited = time.time() - t0
            hours = y500.shape[0] / 500 / 3600
            prob, ev, t = det.score(y500, y1k)
            t1 = time.time()
            os.makedirs(os.path.dirname(dc.long_path(base)), exist_ok=True)
            for suffix, arr in ((PROBS_SUFFIX, prob.astype(np.float16)), (EVENTS_SUFFIX, ev.astype(np.float32))):
                np.save(dc.long_path(base + suffix + ".tmp.npy"), arr)
                os.replace(dc.long_path(base + suffix + ".tmp.npy"), dc.long_path(base + suffix))
            rows = event_rows(ev, min_score, src)
            detect.write_table(base + TABLE_SUFFIX, rows)        # last: its presence means "done"
            write_s = time.time() - t1
        except Exception as e:  # noqa: BLE001 - one bad file must not stop the run
            log(f"FAILED {src}: {e}")
            failed += 1
            with open(os.path.join(a.out, "failed.txt"), "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{src}\t{e}\n")
            continue
        total = time.time() - t0
        tw.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), detect.site_of(src), src, f"{hours:.4f}", len(prob),
                     f"{t_load:.1f}", f"{waited:.1f}"] + [f"{t.get(k, 0.0):.2f}" for k in TIMING_COLUMNS[7:12]]
                    + [f"{write_s:.2f}", f"{total:.1f}", len(ev), len(rows)])
        tf.flush()
        done_h += hours
        el = time.time() - t_start
        log(f"[{i + 1}/{len(todo)}] {os.path.basename(src)}: {hours:.1f} h, {len(rows)} rumbles ({len(ev)} alarms checked); "
            f"read {t_load:.0f} s (waited {waited:.0f} s), AVES {t.get('aves_s', 0):.0f} s, Dasheng {t.get('dasheng_s', 0):.0f} s, "
            f"checker {t.get('cuts_s', 0) + t.get('checker_s', 0):.0f} s; {fmt_dur(el / max(done_h, 1e-9) * 24)} per 24 h so far, "
            f"about {fmt_dur(el / (i + 1) * (len(todo) - i - 1))} left")
    pool.shutdown()
    tf.close()
    log(f"finished: {done_h:.0f} h of audio in {fmt_dur(time.time() - t_start)}" + (f"; {failed} failed, see failed.txt" if failed else ""))
    path = write_summary(a, jobs, pool_files, time.time() - t_start, len(todo) - failed)
    log(f"wrote {path}")


def write_summary(a, jobs, pool_files, wall, n_run):
    """summary.md: time per step, per 24 h of audio, wall clock with the reads in the background,
    and how long the rest would take."""
    rows = {}
    path = os.path.join(a.out, "timings.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            rows[r["file"]] = r
    mine = [rows[p] for p, _ in jobs if p in rows]
    hours = sum(float(r["hours"]) for r in mine)
    md = ["# Best detector: timing and results", "", f"Updated {time.strftime('%Y-%m-%d %H:%M')}. "
          f"Model: see detector.json. {len(mine)} of {len(jobs)} recordings done, {hours:,.1f} h of audio.", ""]
    if mine and hours:
        def per24(k):
            return sum(float(r[k]) for r in mine) / hours * 24
        steps = [("read + resample (in the background)", "read_resample_s"), ("waiting for the read", "waited_s"),
                 ("AVES", "aves_s"), ("Dasheng", "dasheng_s"), ("10 heads + alarm rule", "heads_s"),
                 ("spectrogram cuts", "cuts_s"), ("checkers", "checker_s"), ("writing", "write_s"),
                 ("**total per recording**", "total_s")]
        md += ["| Step | Seconds per 24 h of audio | Median per recording |", "|---|---|---|"]
        md += [f"| {n} | {per24(k):.1f} | {statistics.median(float(r[k]) for r in mine):.1f} |" for n, k in steps]
        tot = per24("total_s")
        alarms = sum(int(r["alarms"]) for r in mine)
        dets = sum(int(r["detections"]) for r in mine)
        md += ["", f"- Alarms checked: {alarms / hours * 24:.0f} per day; rumbles kept at the cut-off: {dets / hours * 24:.1f} per day.",
               f"- Reading the next recordings happens while the GPU works on this one, so the time per recording is "
               f"the total line, not read + total."]
        if n_run:
            md += [f"- This run: {n_run} recordings in {fmt_dur(wall)} of wall clock."]
        left = [x for x in pool_files if x[1] not in rows]
        if left:
            md += [f"- **The other {len(left):,} recordings{' in the Hori-Harm table' if a.hh else ''}: about "
                   f"{fmt_dur(len(left) * tot * (hours / len(mine)) / 24)}** at this speed (if they are as long as these)."]
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return os.path.join(a.out, "summary.md")


################################################################################

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["export", "run"])
    p.add_argument("inputs", nargs="*", help="run: deployment folders (searched with subfolders) or files")
    p.add_argument("--out", help="run: where the results go")
    p.add_argument("--name", default=DEFAULT_NAME, help=f"round 2's model (default {DEFAULT_NAME}; or e.g. late_ens)")
    p.add_argument("--source", default=dc.DEFAULT_SOURCE, help=f"the Datasets copy round 2 ran on (default {dc.DEFAULT_SOURCE})")
    p.add_argument("--model", help="the exported model folder (default <source>\\Configurations\\_round2\\export_<name>)")
    p.add_argument("--hh", help="run: only the recordings in this Hori-Harm table")
    p.add_argument("--sample", type=int, default=0, help="run: only this many recordings, spread over the sites")
    p.add_argument("--min-score", type=float, default=None, help="tables keep alarms scoring at least this (default: the cut-off)")
    p.add_argument("--prefetch", type=int, default=2, help="recordings read + resampled ahead in the background")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--seeds", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.model = a.model or os.path.join(a.source, "Configurations", "_round2", "export_" + a.name.replace("+", "_"))
    if a.step == "run":
        if not a.inputs or not a.out:
            p.error("run needs the deployment folder and --out")
        a.out = os.path.abspath(a.out)
        for r in a.inputs:
            if dc.is_under(a.out, r):
                sys.exit(f"--out {a.out} is inside the input folder {r}; pick a folder outside the recordings.")
        cmd_run(a)
    else:
        cmd_export(a)


if __name__ == "__main__":
    main()
