"""Speed-up experiment: does playing the audio at another speed help AVES or Dasheng?

Every model so far plays the recordings sped up so that rumbles (10-30 Hz) land where the
encoders were trained: AVES at 32x (rumbles at 320-960 Hz), Dasheng at 16x. The speed-up
decides both the pitch the encoder hears and how much real time each fingerprint covers.
This script tries other speeds:

  AVES     16x, 24x, 48x, 64x   (rumbles at 160-480 ... 640-1920 Hz)
  Dasheng   8x, 32x

Each one's fingerprints are moved onto the usual 0.64 s grid (linear interpolation in time,
unit length again), so the labels, the final model's classifier and recipe, the alarm rules
and the scoring are exactly round 2's. Each speed gets 5 runs and their average (_ens),
scored on the validation days and then once on the tests, next to the 32x AVES and 16x
Dasheng references.

Speeds run one after another. A speed's fingerprints (about 80 GB) are deleted once its
tests are scored, unless --keep-fingerprints, so the disk never holds more than one set.

    python tools\\speedup.py                       # everything, resumes if stopped
    python tools\\speedup.py --speeds aves16 aves64  # only some speeds
    python tools\\speedup.py report                # rewrite speedup.md from what's done

Results: <out>\\_speedup\\speedup.md (validation) and speedup_final.md (tests).
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import candidates as cd  # noqa: E402
import dataset_configs as dc  # noqa: E402
import improve as im  # noqa: E402
import round2 as r2  # noqa: E402
import study as st  # noqa: E402
from dataset_configs import log, fmt_dur, Day  # noqa: E402

ROOT_NAME = "_speedup"
# name -> encoder, speed. Audio is played as 16 kHz, so it is loaded at 16000 / speed Hz.
SPEEDS = {"aves16": ("aves", 16), "aves24": ("aves", 24), "aves48": ("aves", 48), "aves64": ("aves", 64),
          "dasheng8": ("dasheng", 8), "dasheng32": ("dasheng", 32)}
REFERENCES = ["aves", "aves_ens", "dasheng", "dasheng_ens"]   # 32x AVES and 16x Dasheng, as in round 2
NATIVE_STEP = {"aves": 320, "dasheng": 640}                 # samples per fingerprint at 16 kHz


def sample_rate(speed):
    return Fraction(16000, speed)


def frame_seconds(enc, speed):
    return float(NATIVE_STEP[enc] / sample_rate(speed))


def to_grid(e, n):
    """[m, d] fingerprints at any step -> [n, d] on the 0.64 s grid, unit length."""
    import torch
    if n == 0 or e.shape[0] == 0:
        return torch.zeros(n, e.shape[1])
    x = torch.nn.functional.interpolate(e.float().T[None], size=n, mode="linear", align_corners=False)[0].T
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-6)


def describe(name):
    enc, speed = SPEEDS[name]
    lo, hi = 10 * speed, 30 * speed
    return f"{'AVES' if enc == 'aves' else 'Dasheng'} at {speed}x (rumbles heard at {lo}-{hi} Hz, one fingerprint per {frame_seconds(enc, speed):.2f} s)"


def cost(name):
    enc, speed = SPEEDS[name]
    ref = 32 if enc == "aves" else 16
    return f"{'AVES' if enc == 'aves' else 'Dasheng'}, about {ref / speed:.2g}x its usual compute"


class Speedup(r2.R2):
    def __init__(self, a, pools, w):
        super().__init__(a, pools, w)
        self.root = os.path.join(a.out, ROOT_NAME)
        self.speeds = a.speeds
        for name in SPEEDS:
            self.src[name] = cd.Source(self.encoder_cache(name))
            r2.TITLES[name] = describe(name)
            r2.TITLES[name + "_ens"] = describe(name) + ", 5 runs averaged"
            r2.ENCODERS_OF[name] = cost(name)

    def report(self, now):
        try:
            with self.w.open(os.path.join(self.root, "progress.txt")) as f:
                f.write(f"Progress, updated {time.strftime('%Y-%m-%d %H:%M')}\nNow: {now}\n"
                        f"Running for {fmt_dur(time.time() - self.t0)}. Results so far: speedup.md\n")
        except OSError as e:
            log(f"WARNING could not write progress.txt: {e}")
        log(f"PROGRESS: {now}")

    # ---- fingerprints ---------------------------------------------------------------------
    def embedder(self, enc):
        if self.a.fake_embedder:
            return dc.FakeEmbedder()          # one fingerprint per 320 samples at any rate
        if enc == "aves":
            import fp_replica
            return fp_replica.Embedder(self.device)
        return cd.DashengEmbedder(self.device)

    def embed(self, name):
        import torch
        import fp_replica
        enc, speed = SPEEDS[name]
        cache = self.encoder_cache(name)
        keys = self.days("train") + self.days("validation") + self.days("tests")
        todo = [(p, k) for p, k in keys if not os.path.exists(dc.cache_path(cache, p, k, "day"))]
        log(f"{name}: {len(todo)} of {len(keys)} recordings to fingerprint")
        if not todo:
            return
        embed, t0 = self.embedder(enc), time.time()
        for i, (p, k) in enumerate(todo, 1):
            n = Day(self.a.cache, p, k).n
            y = fp_replica.load_audio(self.pools[p].sounds[k], sr_out=sample_rate(speed))
            e = to_grid(embed.whole_file(y), n)
            dst = dc.cache_path(cache, p, k, "day")
            self.w.makedirs(os.path.dirname(dst))
            torch.save({"kind": "day", "n": n, "emb": e.half(), "speed": speed}, self.w.check(dst + ".tmp"))
            os.replace(self.w.check(dst + ".tmp"), self.w.check(dst))
            if i % 10 == 0 or i == len(todo):
                left = (time.time() - t0) / i * (len(todo) - i)
                self.report(f"{name} fingerprints: {i}/{len(todo)} recordings, about {fmt_dur(left)} left")

    def train_base(self, base):
        if base in SPEEDS and not self.trained(base):
            self.report(f"{base}: fingerprints of every recording")
            self.embed(base)
        super().train_base(base)

    def tested(self, name):
        return name in (self.jload(os.path.join(self.root, "final.json")) or {})

    def drop_fingerprints(self, name):
        path = self.encoder_cache(name)
        if os.path.isdir(path) and not self.a.keep_fingerprints:
            log(f"{name}: tests scored, deleting its fingerprints ({path})")
            shutil.rmtree(self.w.check(path), ignore_errors=True)
            self._probs = {k: v for k, v in self._probs.items() if k[0] != name}

    # ---- results ------------------------------------------------------------------------------
    def stage_names(self):
        return REFERENCES + [s for n in SPEEDS for s in (n, n + "_ens")]

    def results(self):
        out = []
        for s in self.stage_names():
            r = self.jload(os.path.join(self.dir(s), "results.json"))
            if r:
                out.append(r)
        return out


################################################################################
# commands
################################################################################

def score_tests(lab, r):
    """Each test set at the cut-offs that found 80% / 85% on validation (round 2's cmd_final)."""
    res = {}
    for key, _ in r2.TEST_SETS:
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
    return res


def test_stages(lab, names):
    path = os.path.join(lab.root, "final.json")
    out = lab.jload(path) or {}
    for name in names:
        r = lab.jload(os.path.join(lab.dir(name), "results.json"))
        if not r or (name in out and out[name].get("finished_validation") == r["finished"]):
            continue
        lab.report(f"tests: {name}")
        try:
            out[name] = {"sets": score_tests(lab, r), "finished_validation": r["finished"]}
        except Exception:
            r2.record_failure(lab, f"tests for {name}")
            continue
        st.wjson(lab.w, path, out)


def write_reports(lab):
    rs = lab.results()
    rs.sort(key=lambda r: st.rank_key(r["summary"], r["name"]))
    f3 = lambda x: "-" if not x else f"{x[0]:.3f}"  # noqa: E731
    rows = []
    for r in rs:
        s = r["summary"]
        sc = "85% not reached by every run" if not s["score"] else \
            f"**{s['score'][0]:.3f}**" + (f" ({s['score'][1]:.3f}-{s['score'][2]:.3f})" if s["runs"] > 1 else "")
        rows.append(f"| {r['name']} | {r['title']} | {sc} | {f3(s['fp_at_80'])} | {f3(s['fp_at_85'])} | "
                    f"{f3(s['fp_at_90'])} | {f3(s['max_recall'])} | {s['runs']} | {r['cost']} | {im.rule_text(r['rule'])} |")
    head = ["# Speed-up experiment on the validation days", "",
            "Same days and score as round 2: **mean false alarms per hour at 80% and 85% recall** (lower is "
            "better), range over runs in brackets. `aves` (32x) and `dasheng` (16x) are the usual speeds. Best first.", "",
            "| Name | What | Score | FA/h at 80% | at 85% | at 90% | Max recall | Runs | Inference | Rule |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    with lab.w.open(os.path.join(lab.root, "speedup.md")) as f:
        f.write("\n".join(head + rows) + "\n")
    fin = lab.jload(os.path.join(lab.root, "final.json")) or {}
    head = ["# Speed-up experiment on the tests, scored once", "",
            "Each run uses the cut-off that found 80% (or 85%) of the rumbles on the validation days. Each cell: "
            "recall / precision / false alarms per hour, mean over runs. Ordered by the validation score.", "",
            "| Name | Validation score | " + " | ".join(f"{n} at 80% | at 85%" for _, n in r2.TEST_SETS) + " |",
            "|---|---|" + "---|---|" * len(r2.TEST_SETS)]
    rows = []
    for r in rs:
        res = (fin.get(r["name"]) or {}).get("sets")
        if res is None:
            continue
        cells = []
        for key, _ in r2.TEST_SETS:
            for rec in ("80", "85"):
                m = ((res.get(key) or {}).get(rec) or {}).get("mean")
                cells.append("-" if not m else f"{im._f(m['recall'])} / {im._f(m['precision'])} / {im._f(m['fp_per_hour'], 2)}")
        sc = r["summary"]["score"]
        rows.append(f"| {r['name']} | {'-' if not sc else f'{sc[0]:.3f}'} | " + " | ".join(cells) + " |")
    with lab.w.open(os.path.join(lab.root, "speedup_final.md")) as f:
        f.write("\n".join(head + rows) + "\n")


def cmd_all(lab):
    t0 = time.time()
    for s in REFERENCES:
        r2.safe(lab, f"{s} validation", lab.evaluate_stage, s)
    test_stages(lab, REFERENCES)
    write_reports(lab)
    for name in lab.speeds:
        names = [name, name + "_ens"]
        if all(lab.tested(n) for n in names):
            lab.drop_fingerprints(name)
            continue
        if r2.safe(lab, f"{name} training", lab.train_base, name) is None and not lab.trained(name):
            continue
        for s in names:
            r2.safe(lab, f"{s} validation", lab.evaluate_stage, s)
            write_reports(lab)
        test_stages(lab, names)
        write_reports(lab)
        if all(lab.tested(n) for n in names):
            lab.drop_fingerprints(name)
        lab._models.pop(name, None)
    lab.report(f"all speeds done in {fmt_dur(time.time() - t0)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", nargs="?", default="all", choices=["all", "report"])
    p.add_argument("--speeds", nargs="+", default=list(SPEEDS), choices=list(SPEEDS), help="which speeds, in this order")
    p.add_argument("--keep-fingerprints", action="store_true", help="don't delete a speed's fingerprints after its tests")
    p.add_argument("--source", default=dc.DEFAULT_SOURCE, help=f"the Datasets copy (default {dc.DEFAULT_SOURCE})")
    p.add_argument("--out", help="the Configurations folder (default <source>\\Configurations)")
    p.add_argument("--cache", help="the AVES cache (default <out>\\_embedding_cache)")
    p.add_argument("--seeds", type=int, default=st.SEEDS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.out = a.out or os.path.join(a.source, "Configurations")
    a.cache = a.cache or os.path.join(a.out, "_embedding_cache")
    a.archs, a.skip = [], []
    im.keep_awake()
    root = dc.find_dataset_root(a.source)
    w = dc.Writer([os.path.join(a.out, ROOT_NAME)] + [os.path.join(a.out, f"_embedding_cache_{n}") for n in SPEEDS],
                  [dc.ORIGINAL_DATA, os.path.join(root, "Training"), os.path.join(root, "Testing")])
    w.makedirs(os.path.join(a.out, ROOT_NAME))
    stdout = sys.stdout
    sys.stdout = dc.Tee(w.check(os.path.join(a.out, ROOT_NAME, "run_log.txt")))
    try:
        pools = dc.load_pools(root)
        lab = Speedup(a, pools, w)
        log(f"speed-up experiment: speeds {lab.speeds}; writing to {lab.root}")
        if a.step == "all":
            cmd_all(lab)
        write_reports(lab)
        lab.report("finished")
    finally:
        sys.stdout = stdout


if __name__ == "__main__":
    main()
