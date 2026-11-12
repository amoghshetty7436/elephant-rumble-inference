#!/usr/bin/env python
"""
Precision-recall curves on the validation and test days for round 2's detectors, from round 2's
own saved models and scores (nothing is retrained). round2.py only keeps the points at the 80% and
85% cut-offs; this writes every cut-off, in the same layout as the rerun's pr_curve.csv files:

  <out>\\<name>\\<set>_pr_curve.csv   run, threshold, recall, precision, fp_per_hour, f1, detections,
                                    rumbles_found, true_detections, false_alarms, rumbles, hours
  <out>\\<name>\\cut_offs.csv         each run's validated cut-offs (80% and 85% recall on validation)
  <out>\\summary.md                  every set at the 80% cut-off, to check against round2_final.md

Default models: the best detector (late_ens+efficientnet_b0_x5) and round 2's reference "aves"
(the rerun's final model, 5 runs; run 0 is the one in the ElephantRumbleDetector package).

On the lab PC, from the repository folder, with the round 2 venv:
  python tools\\pr_curves.py
  python tools\\pr_curves.py --names late_ens+efficientnet_b0_x5 aves late_ens --out D:\\pr_curves
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset_configs as dc  # noqa: E402
from dataset_configs import log  # noqa: E402

DEFAULT_NAMES = ["late_ens+efficientnet_b0_x5", "aves"]
SETS = ["validation", "sampleA", "sampleB", "vehicle"]
COLUMNS = ["run", "threshold", "recall", "precision", "fp_per_hour", "f1", "detections", "rumbles_found",
           "true_detections", "false_alarms", "rumbles", "hours"]


def open_lab(source, fake=False, seeds=None):
    import round2 as r2
    import study as st
    out = os.path.join(source, "Configurations")
    a = argparse.Namespace(source=source, out=out, cache=os.path.join(out, "_embedding_cache"),
                           seeds=seeds or st.SEEDS, seed=0, fake_embedder=fake, archs=r2.ARCHS, skip=None)
    root = dc.find_dataset_root(source)
    w = dc.Writer([os.path.join(out, r2.ROOT_NAME)], [dc.ORIGINAL_DATA])
    return r2.R2(a, dc.load_pools(root), w)


def num(x, d=6):
    x = float(x)
    return "" if x != x else round(x, d)


def curves(lab, name, sets, out):
    import candidates as cd
    import improve as im
    r = next((r for r in lab.results() if r and r["name"] == name), None)
    if r is None:
        log(f"{name}: round 2 has no results for it (see round2.md for the names); skipped")
        return []
    folder = os.path.join(out, name)
    os.makedirs(folder, exist_ok=True)
    per_seed = r["summary"]["per_seed"]
    with open(os.path.join(folder, "cut_offs.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["run", "cut_off_80", "cut_off_85"])
        for i, ps in enumerate(per_seed):
            wr.writerow([i] + [(ps.get(f"at_{k}") or {}).get("threshold", "") for k in (80, 85)])
    summary = []
    for split in sets:
        files = lab.files(split)
        if not files:
            log(f"{name}: no {split} days; skipped")
            continue
        truth = im.Truth(lab.pools, files)
        sweeps = [cd.sweep_events(truth, ev) for ev in lab.events_for(r, split)]
        path = os.path.join(folder, f"{split}_pr_curve.csv")
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(COLUMNS)
            for run, sw in enumerate(sweeps):
                for i, t in enumerate(sw["threshold"]):
                    wr.writerow([run, num(t, 3), num(sw["recall"][i]), num(sw["precision"][i]), num(sw["fp_per_hour"][i]),
                                 num(sw["f1"][i]), int(sw["detections"][i]), int(sw["rumbles_found"][i]),
                                 int(sw["true_detections"][i]), int(sw["fp"][i]), int(sw["rumbles"]), num(truth.hours, 3)])
        for run, (sw, ps) in enumerate(zip(sweeps, per_seed)):
            cut = (ps.get("at_80") or {}).get("threshold")
            if cut is not None:
                p = cd.at_cut(sw, cut)
                summary.append((name, split, run, cut, p["recall"], p["precision"], p["fp_per_hour"]))
        log(f"{name}: {split} ({len(files)} days, {truth.n} rumbles, {truth.hours:.1f} h) -> {path}")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--names", nargs="+", default=DEFAULT_NAMES, help="round 2 names, as in round2.md")
    p.add_argument("--sets", nargs="+", default=SETS, choices=SETS)
    p.add_argument("--source", default=dc.DEFAULT_SOURCE, help=f"the Datasets copy round 2 ran on (default {dc.DEFAULT_SOURCE})")
    p.add_argument("--out", help="where the curves go (default <source>\\Configurations\\_round2\\pr_curves)")
    p.add_argument("--seeds", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    import round2 as r2
    out = a.out or os.path.join(a.source, "Configurations", r2.ROOT_NAME, "pr_curves")
    os.makedirs(out, exist_ok=True)
    lab = open_lab(a.source, a.fake_embedder, a.seeds)
    rows = []
    for name in a.names:
        rows += curves(lab, name, a.sets, out)
    lines = ["# Precision-recall curves: check", "",
             "Each run at its validated cut-off (80% recall on validation). These should equal round2_final.md.", "",
             "| Model | Set | Run | Cut-off | Recall | Precision | False alarms per hour |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {n} | {s} | {r} | {c:.4f} | {rc:.3f} | {pr:.3f} | {fa:.3f} |" for n, s, r, c, rc, pr, fa in rows]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    log(f"done: {out}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
