#!/usr/bin/env python
"""
Score two published forest-elephant rumble detectors on our test sets, the same way
our own detector was scored (study.py final), so the numbers sit side by side:

  EarthToolsMaker   YOLOv8 on 164 s spectrogram images (FruitPunch AI challenge follow-up)
                    github.com/earthtoolsmaker/forest-elephants-rumble-detection
  DeepForestSound   AST + LoRA, 10 s chunks, trained in Kibale (Uganda)
                    github.com/MNHN-OFVI/DeepForestSound

Same yardstick as ours: ELP's matching rule (more than 1% overlap in time, as a share of
the shorter box; a detector without frequencies counts as covering 5-250 Hz, exactly as
ours does), the cut-off is the one that finds 80% (or 85%) of the rumbles on the
validation days (RumbleComplex + RumbleGeneral), and the tests are scored with it once.
Each tool's own suggested cut-off (EarthToolsMaker 0.25, DeepForestSound 0.30) is shown too.

Caveat: EarthToolsMaker's shipped model was trained on the FruitPunch "Testing" folder,
which holds 24 of the 43 Sample A days. Sample A is reported on all days and on the 19
days it never saw; Sample B (2023) and the validation days are unseen by both tools.

Steps (main environment, where dataset_configs.py runs):

  python tools\\baselines\\compare.py list  --source L:\\...\\Dataset --out D:\\baselines
      -> D:\\baselines\\files.tsv (every validation, test and vehicle sound file)

then each detector in its own environment (see its script's header):

  python tools\\baselines\\run_etm.py --etm-repo ... --list D:\\baselines\\files.tsv --out D:\\baselines\\etm
  python tools\\baselines\\run_dfs.py --dfs-repo ... --list D:\\baselines\\files.tsv --out D:\\baselines\\dfs

and back in the main environment:

  python tools\\baselines\\compare.py score --source L:\\...\\Dataset --out D:\\baselines ^
      [--ours <study folder>\\final.json]
      -> D:\\baselines\\baselines.md, baselines.json, curves\\<detector>_<set>.csv
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import dataset_configs as dc  # noqa: E402

VALIDATION = ("complex", "general")
TESTS = ("sampleB", "sampleA", "vehicle")
DETECTORS = {
    "etm": dict(name="EarthToolsMaker (YOLOv8)", own_cutoff=0.25),
    "dfs": dict(name="DeepForestSound (AST + LoRA)", own_cutoff=0.30),
}
TARGETS = (0.80, 0.85)
THRESHOLDS = sorted({round(x / 100, 3) for x in range(1, 100)} | {round(0.9 + x / 1000, 3) for x in range(0, 100)})
OVERLAP = 0.01


def etm_trained_days():
    """Days in the FruitPunch "Testing" folder: what EarthToolsMaker's shipped model trained on."""
    days, on = set(), False
    with open(os.path.join(os.path.dirname(HERE), "fruitpunch_files_used.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("##"):
                on = line == "## TESTING"
            elif on and line.lower().endswith(".wav"):
                days.add(dc.day_of(dc.file_key(line)))
    return days


################################################################################
# list
################################################################################

def cmd_list(a):
    import soundfile as sf
    pools = dc.load_pools(dc.find_dataset_root(a.source))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "files.tsv")
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["pool", "key", "hours", "path"])
        for p in VALIDATION + TESTS:
            for k in pools[p].files:
                info = sf.info(pools[p].sounds[k])
                w.writerow([p, k, f"{info.frames / info.samplerate / 3600:.5f}", pools[p].sounds[k]])
                n += 1
    dc.log(f"wrote {path}: {n} sound files")


def read_list(out):
    with open(os.path.join(out, "files.tsv"), newline="", encoding="utf-8") as f:
        return [(r["pool"], r["key"], float(r["hours"])) for r in csv.DictReader(f, delimiter="\t")]


################################################################################
# score
################################################################################

def read_detections(folder, files):
    """{(pool, key): (starts, ends, scores)} in seconds; None if a file has not been run."""
    import numpy as np
    out = {}
    for p, k, _ in files:
        path = os.path.join(folder, p, k + ".csv")
        if not os.path.exists(path):
            return None
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        out[(p, k)] = tuple(np.array([float(r[c]) for r in rows], dtype=np.float64)
                            for c in ("t_start", "t_end", "score"))
    return out


def sweep(truth, dets):
    """Precision, recall and false alarms per hour at every cut-off in THRESHOLDS.
    Each box (EarthToolsMaker) or 10 s chunk (DeepForestSound) is one detection, as the
    tools write them: joining touching chunks would let one long box cover many rumbles
    under the 1% rule, which flatters a detector instead of measuring it."""
    import numpy as np
    rows = []
    for cut in THRESHOLDS:
        n_det = tp = found = 0
        for day, (s, e, v) in dets.items():
            keep = v >= cut - 1e-12
            s, e, v = s[keep], e[keep], v[keep]
            # Truth.match works in frames of FRAME_SECONDS
            hit, best = truth.match(day, s / dc.FRAME_SECONDS, e / dc.FRAME_SECONDS, v)
            n_det += len(s)
            tp += int(hit.sum())
            found += int(np.isfinite(best).sum())
        fp = n_det - tp
        rows.append(dict(threshold=cut, detections=n_det, false_alarms=fp, rumbles=truth.n, rumbles_found=found,
                         recall=found / truth.n if truth.n else float("nan"),
                         precision=tp / n_det if n_det else float("nan"),
                         fp_per_hour=fp / truth.hours if truth.hours else float("nan")))
    return rows


def cut_for_recall(rows, target):
    ok = [r for r in rows if r["recall"] >= target - 1e-9]
    return max(ok, key=lambda r: r["threshold"])["threshold"] if ok else None


def at(rows, cut):
    return None if cut is None else min(rows, key=lambda r: abs(r["threshold"] - cut))


def fmt(v, d=3):
    return "-" if v is None or v != v else f"{v:.{d}f}"


def cell(r, vehicle=False):
    if r is None:
        return "never reaches it"
    return fmt(r["fp_per_hour"], 2) if vehicle else \
        f"{fmt(r['recall'])} / {fmt(r['precision'])} / {fmt(r['fp_per_hour'], 2)}"


def ours_rows(path):
    """Our final model's test cells from study.py's final.json (step 10)."""
    with open(path, encoding="utf-8") as f:
        steps = json.load(f)["steps"]
    final = [s for s in steps if s["step"].startswith("10")] or steps[-1:]
    return final[0]["sets"] if final else {}


def cmd_score(a):
    import improve as im
    pools = dc.load_pools(dc.find_dataset_root(a.source))
    listed = read_list(a.out)
    by_pool = {p: [f for f in listed if f[0] == p] for p in VALIDATION + TESTS}
    trained = etm_trained_days()
    sets = {
        "validation": [f for f in listed if f[0] in VALIDATION],
        "sampleB": by_pool["sampleB"],
        "sampleB_no_odd": [f for f in by_pool["sampleB"] if dc.day_of(f[1]) != dc.SAMPLE_B_ODD_DAY],
        "sampleA": by_pool["sampleA"],
        "sampleA_unseen_by_etm": [f for f in by_pool["sampleA"] if dc.day_of(f[1]) not in trained],
        "vehicle": by_pool["vehicle"],
    }
    titles = {"validation": "Validation (Complex + General)", "sampleB": "Test 1: Sample B",
              "sampleB_no_odd": f"Sample B without {dc.SAMPLE_B_ODD_DAY}", "sampleA": "Test 2: Sample A (all days)",
              "sampleA_unseen_by_etm": "Sample A, days EarthToolsMaker never trained on",
              "vehicle": "Vehicle (false alarms per hour only)"}
    truths = {n: im.Truth(pools, fs) for n, fs in sets.items() if fs}
    os.makedirs(os.path.join(a.out, "curves"), exist_ok=True)

    result, md = {}, []
    for det, spec in DETECTORS.items():
        dets = read_detections(os.path.join(a.out, det), listed)
        if dets is None:
            dc.log(f"{spec['name']}: not every listed file has been run yet, skipped")
            continue
        curves = {}
        for n, truth in truths.items():
            curves[n] = sweep(truth, {(p, k): dets[(p, k)] for p, k, _ in sets[n]})
            with open(os.path.join(a.out, "curves", f"{det}_{n}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(curves[n][0]))
                w.writeheader()
                w.writerows(curves[n])
        cuts = {f"at_{int(t * 100)}": cut_for_recall(curves["validation"], t) for t in TARGETS}
        cuts["own"] = spec["own_cutoff"]
        res = {"cutoffs": cuts, "sets": {}}
        for n, rows in curves.items():
            res["sets"][n] = {c: at(rows, v) for c, v in cuts.items()}
            res["sets"][n]["max_recall"] = max(r["recall"] for r in rows)
        result[det] = res
        dc.log(f"{spec['name']}: scored")

    head = ["| Detector | Cut-off | " + " | ".join(titles[n] for n in sets) + " |",
            "|---|---|" + "---|" * len(sets)]
    md += ["# Published detectors on our tests", "",
           "Scored like our final model (study.py final): ELP's 1% overlap rule, cut-off chosen on the "
           "validation days to find 80% or 85% of their rumbles, then applied once to each test. "
           "Each cell: recall / precision / false alarms per hour (vehicle: false alarms per hour). "
           "\"own\" = the tool's suggested cut-off.", "",
           f"EarthToolsMaker's model trained on {len(trained)} FruitPunch test days, "
           f"{len(sets['sampleA']) - len(sets['sampleA_unseen_by_etm'])} of them in Sample A: its "
           "\"Sample A (all days)\" number is flattered, use the unseen-days column.", ""] + head
    for det, res in result.items():
        for c in ("at_80", "at_85", "own"):
            md.append(f"| {DETECTORS[det]['name']} | {c.replace('at_', '')}{'% recall' if c != 'own' else ''} "
                      f"({fmt(res['cutoffs'][c], 3)}) | "
                      + " | ".join(cell(res["sets"][n][c], n == "vehicle") if n in res["sets"] else "-"
                                   for n in sets) + " |")
        md.append(f"| {DETECTORS[det]['name']} | highest recall at any cut-off | "
                  + " | ".join(fmt(res["sets"][n]["max_recall"]) if n in res["sets"] and n != "vehicle" else "-"
                               for n in sets) + " |")
    if a.ours:
        try:
            ours = ours_rows(a.ours)
            md += ["", "Our final model (study.py final, mean of its runs) on the same tests:", "",
                   "| Test | at 80% | at 85% |", "|---|---|---|"]
            for key in TESTS:
                s = ours.get(key)
                if s:
                    md.append(f"| {titles[key]} | " + " | ".join(
                        (cell(m, key == "vehicle") if m else "-")
                        for m in (s.get(str(r), s.get(r, {})).get("mean") for r in (80, 85))) + " |")
        except (OSError, KeyError, ValueError) as e:
            md += ["", f"(could not read our numbers from {a.ours}: {e})"]
    md += ["", "Days and rumbles per set: " + ", ".join(
        f"{titles[n]} {len(fs)} files / {truths[n].n} rumbles" for n, fs in sets.items() if n in truths), "",
        "Precision-recall curves: curves\\<detector>_<set>.csv."]
    with open(os.path.join(a.out, "baselines.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(a.out, "baselines.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("\n".join(md))
    dc.log(f"wrote {os.path.join(a.out, 'baselines.md')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "score"])
    ap.add_argument("--source", default=dc.DEFAULT_SOURCE, help="dataset folder (as for dataset_configs.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ours", help="study.py's final.json, to print our final model's numbers alongside")
    a = ap.parse_args(argv)
    {"list": cmd_list, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
