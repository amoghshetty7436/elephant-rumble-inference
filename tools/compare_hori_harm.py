"""Compare our rumble detector with Hori-Harm on days an analyst reviewed.

The Hori-Harm table (from Bobbi) lists every Hori-Harm detection on a random sample of
days, and its Count column says how many rumbles each one holds (0 = false alarm).
Our results folder holds one Raven table per recording (<name>.rumbles.txt) and, next to
it, the frame probabilities (<name>.probs.npy), from which detections at any cut-off
can be made again with the detector's own rule.

What it measures, on the recordings in both:
  * how many of Hori-Harm's confirmed rumbles we also detect (at every cut-off);
  * how many of Hori-Harm's false alarms we also fire on;
  * what our detections land on: a confirmed rumble, a Hori-Harm false alarm, or no
    Hori-Harm box at all (unknown: our false alarm or a rumble Hori-Harm missed);
  * the same for Hori-Harm at higher score cut-offs, for a like-for-like view.
Two boxes match when they overlap, allowing --slack seconds between them.

Usage (Windows, from the repository folder):
  python tools\\compare_hori_harm.py --hh "PATH\\nn_..._counted.txt" --ours "L:\\...\\nn_202306_jun_dep17" --out "C:\\...\\hh_comparison"

Writes to --out: report.md (read this first), and Excel-friendly CSV files and Raven
tables (see the end of report.md for the list). Needs only numpy.
"""
import argparse
import csv
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# works from tools\ in the repository, or copied into the ElephantRumbleDetector folder
sys.path.insert(0, os.path.join(HERE, "..", "ElephantRumbleDetector"))
sys.path.insert(0, HERE)
import rumble_detector as rd  # noqa: E402  the detector's own alarm rule and table format

DEFAULT_CUTOFFS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.993, 0.995, 0.997, 0.999]
HH_CUTOFFS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SLACKS = [0.0, 1.0, 2.0, 5.0]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------- reading

def read_hh(path):
    """Reviewed Hori-Harm detections: dicts with file, path, site, begin, end, count, score, notes.
    Rows without a Count (an un-reviewed copy of a site) and exact duplicates are dropped."""
    rows, seen, info = [], set(), {"rows": 0, "no_count": 0, "duplicates": 0}
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            info["rows"] += 1
            c = (r.get("Count") or "").strip()
            if c == "":
                info["no_count"] += 1
                continue
            b = float(r["File Offset (s)"])
            e = b + float(r["End Time (s)"]) - float(r["Begin Time (s)"])
            key = (r["Begin File"], round(b, 3), round(e, 3))
            if key in seen:
                info["duplicates"] += 1
                continue
            seen.add(key)
            rows.append({"file": r["Begin File"], "path": r["Begin Path"], "site": r.get("Site") or r["Begin File"][:5],
                         "begin": b, "end": e, "count": int(float(c)), "score": float(r["Score"]),
                         "notes": "; ".join(x for x in (r.get("Ambiguous", ""), r.get("Notes", "")) if x.strip())})
    return rows, info


def index_ours(folder):
    """recording file name (.wav) -> its table path, for every table under the results folder."""
    out = {}
    for root, _, files in os.walk(folder):
        for n in files:
            if n.endswith(rd.TABLE_SUFFIX):
                out[n[: -len(rd.TABLE_SUFFIX)] + ".wav"] = os.path.join(root, n)
    return out


def read_table(path):
    """Our detections in one table: list of (begin s, end s, score)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [(float(r["File Offset (s)"]), float(r["File Offset (s)"]) + float(r["End Time (s)"])
                 - float(r["Begin Time (s)"]), float(r["Score"])) for r in csv.DictReader(f, delimiter="\t")]


def from_events(path):
    """All alarms detect_best.py saved, with their final (checked) scores."""
    ev = np.load(path).astype(np.float64)
    return [(a * rd.FRAME_SECONDS, b * rd.FRAME_SECONDS, float(s)) for a, b, _, s in ev]


def detector_info(folder):
    """detector.json of the run that made the results folder (it sits in --out, one level up)."""
    for d in (folder, os.path.dirname(os.path.normpath(folder))):
        p = os.path.join(d, "detector.json")
        if os.path.exists(p):
            with open(p) as f:
                return rd.json.load(f)
    return {}


def from_probs(path, rule):
    """All detections the rule makes from the saved probabilities, at any score."""
    prob = np.load(path).astype(np.float32)
    st, en, sc = rd.alarms(prob, rule)
    return [(a * rd.FRAME_SECONDS, b * rd.FRAME_SECONDS, float(s)) for a, b, s in zip(st, en, sc)]


# ---------------------------------------------------------------------------- matching

def overlaps(b1, e1, b2, e2, slack):
    return b1 <= e2 + slack and b2 <= e1 + slack


def match_file(hh, ours, slack):
    """For each Hori-Harm box: the best score of our detections touching it (-1 if none).
    For each of our detections: 'rumble' / 'hh_false_alarm' / 'no_hh_box', and what it touches."""
    hh_best = []
    for h in hh:
        s = [sc for b, e, sc in ours if overlaps(h["begin"], h["end"], b, e, slack)]
        hh_best.append(max(s) if s else -1.0)
    kinds = []
    for b, e, sc in ours:
        touch = [h for h in hh if overlaps(h["begin"], h["end"], b, e, slack)]
        if any(h["count"] > 0 for h in touch):
            kinds.append("rumble")
        elif touch:
            kinds.append("hh_false_alarm")
        else:
            kinds.append("no_hh_box")
    return hh_best, kinds


# ---------------------------------------------------------------------------- report helpers

def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "-"


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_raven(path, rows):
    """rows: dicts with path, file, begin, end, score, and extra columns. Raven opens the
    recordings through Begin Path + File Offset; the empty Count column is for the reviewer."""
    extra = [k for k in rows[0] if k not in ("path", "file", "begin", "end", "score")] if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(rd.RAVEN_HEADER + extra + ["Count", "Reviewer notes"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, "Spectrogram 1", 1, f"{r['begin']:.3f}", f"{r['end']:.3f}", 5, 250, r["path"],
                        f"{r['begin']:.3f}", r["file"], f"{r['score']:.4f}"] + [r[k] for k in extra] + ["", ""])


def table_md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hh", required=True, help="Hori-Harm selection table with the Count column")
    p.add_argument("--ours", required=True, help="our results folder for the same deployment")
    p.add_argument("--out", required=True, help="folder for the report (made if needed)")
    p.add_argument("--slack", type=float, default=2.0, help="seconds allowed between two boxes that match")
    p.add_argument("--cutoffs", default=",".join(str(c) for c in DEFAULT_CUTOFFS))
    p.add_argument("--hh-cutoffs", default=",".join(str(c) for c in HH_CUTOFFS),
                   help="Hori-Harm score cut-offs to show (its table starts at 0.4)")
    p.add_argument("--main-cutoff", type=float, default=None, help="default: the model's own cut-off")
    p.add_argument("--tables-only", action="store_true", help="ignore the .probs.npy files (one cut-off only)")
    p.add_argument("--sample", type=int, default=200, help="how many of our unmatched detections to sample for review")
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    settings = rd.json.load(open(os.path.join(rd.MODEL_DIR, "settings.json")))
    run_info = detector_info(a.ours)
    if run_info.get("kind") == "checked" or run_info.get("model_info"):  # detect_best.py's results
        settings = {"name": run_info.get("experiment"), "rule": run_info["rule"], "cut_off": run_info["cut_off"]}
        log(f"results of {run_info.get('experiment')}: rule and cut-off {settings['cut_off']:.4f} from its detector.json")
    rule, main_cut = settings["rule"], a.main_cutoff or settings["cut_off"]
    cutoffs = sorted(set([float(c) for c in a.cutoffs.split(",")] + [main_cut]))

    log("reading the Hori-Harm table...")
    hh, hh_info = read_hh(a.hh)
    by_file = {}
    for h in hh:
        by_file.setdefault(h["file"], []).append(h)
    log(f"{hh_info['rows']:,} rows; {len(hh):,} reviewed detections on {len(by_file):,} recordings")
    log("listing our tables...")
    ours_idx = index_ours(a.ours)
    log(f"{len(ours_idx):,} of our tables found")
    common = sorted(f for f in by_file if f in ours_idx)
    missing = sorted(f for f in by_file if f not in ours_idx)

    # our detections per common file, at all scores if the probabilities are there
    use_probs, check = not a.tables_only, {"files": 0, "same": 0, "differ": [], "borderline": 0}
    ours, no_probs = {}, []
    for i, f in enumerate(common, 1):
        table = read_table(ours_idx[f])
        probs = ours_idx[f][: -len(rd.TABLE_SUFFIX)] + rd.PROBS_SUFFIX
        events = ours_idx[f][: -len(rd.TABLE_SUFFIX)] + ".events.npy"
        if use_probs and (os.path.exists(events) or os.path.exists(probs)):
            # detect_best.py's alarms carry the checker's scores, which the probabilities alone can't give
            det = from_events(events) if os.path.exists(events) else from_probs(probs, rule)
            # self-check: the probabilities give back exactly the saved table at the cut-off it used
            # The probabilities are saved as float16 (about 0.0005 apart near 1), the tables were made
            # from float32: a detection scoring within that of the cut-off can fall either side.
            def near(x):
                return abs(x[2] - main_cut) < 0.001
            again = [d for d in det if d[2] >= main_cut and not near(d)]
            saved = [d for d in table if not near(d)]
            check["borderline"] += sum(1 for d in det if d[2] >= main_cut and near(d)) - sum(map(near, table))
            check["files"] += 1
            same = len(again) == len(saved) and all(abs(x[0] - y[0]) < 0.01 and abs(x[1] - y[1]) < 0.01
                                                     for x, y in zip(again, saved))
            if same:
                check["same"] += 1
            else:
                check["differ"].append(f)
        else:
            no_probs.append(f)
            det = table
        ours[f] = det
        if i % 200 == 0:
            log(f"read {i:,} of {len(common):,} recordings")
    if use_probs and no_probs:
        log(f"WARNING: {len(no_probs):,} recordings have no .probs.npy next to their table "
            f"(first: {no_probs[0]}), so only the main cut-off can be compared")
    have_low = check["files"] == len(common) and check["files"] > 0
    if not have_low:
        cutoffs = [c for c in cutoffs if c >= main_cut]
    log(f"matching (slack {a.slack:g} s)...")

    # matching at the chosen slack, and the slack sensitivity at the main cut-off
    res = {}
    for f in common:
        res[f] = match_file(by_file[f], ours[f], a.slack)
    slack_rows = []
    for sl in sorted(set(SLACKS + [a.slack])):
        found = fa = tot = 0
        for f in common:
            det = [d for d in ours[f] if d[2] >= main_cut]
            hb, _ = match_file(by_file[f], det, sl)
            for h, s in zip(by_file[f], hb):
                if h["count"] > 0:
                    tot += 1
                    found += s >= 0
                else:
                    fa += s >= 0
        slack_rows.append([f"{sl:g}", found, tot, pct(found, tot), fa])

    hh_c = [h for f in common for h in by_file[f]]
    best = [s for f in common for s in res[f][0]]
    n_true = sum(h["count"] > 0 for h in hh_c)
    n_calls = sum(h["count"] for h in hh_c)
    n_fa = len(hh_c) - n_true
    days = len(common)

    # ours at each cut-off
    cut_rows = []
    for c in cutoffs:
        found = sum(1 for h, s in zip(hh_c, best) if h["count"] > 0 and s >= c)
        calls = sum(h["count"] for h, s in zip(hh_c, best) if h["count"] > 0 and s >= c)
        fa = sum(1 for h, s in zip(hh_c, best) if h["count"] == 0 and s >= c)
        k = {"rumble": 0, "hh_false_alarm": 0, "no_hh_box": 0}
        for f in common:
            for (b, e, s), kind in zip(ours[f], res[f][1]):
                if s >= c:
                    k[kind] += 1
        n = sum(k.values())
        cut_rows.append([c, n, round(n / days, 2), found, pct(found, n_true), calls, pct(calls, n_calls), fa,
                         pct(fa, n_fa), k["rumble"], k["hh_false_alarm"], k["no_hh_box"],
                         pct(k["rumble"], n), pct(n - k["hh_false_alarm"], n)])
    cut_header = ["our cut-off", "our detections", "per day", "HH rumbles we find", "% of HH rumbles",
                  "rumbles (calls) we find", "% of calls", "HH false alarms we also fire on", "% of HH false alarms",
                  "ours on a HH rumble", "ours on a HH false alarm", "ours on no HH box",
                  "our precision, lowest", "our precision, highest"]
    write_csv(os.path.join(a.out, "our_cutoffs.csv"), cut_header, cut_rows)

    # Hori-Harm at each score cut-off (its recall is out of its own confirmed rumbles at 0.4)
    hh_rows = []
    for c in sorted({float(x) for x in a.hh_cutoffs.split(",")}):
        sel = [h for h in hh_c if h["score"] >= c]
        t = sum(h["count"] > 0 for h in sel)
        hh_rows.append([c, len(sel), round(len(sel) / days, 2), t, pct(t, n_true), pct(t, len(sel))])
    hh_header = ["HH score cut-off", "HH detections", "per day", "confirmed rumbles kept", "% of HH rumbles",
                 "HH precision"]
    write_csv(os.path.join(a.out, "hori_harm_cutoffs.csv"), hh_header, hh_rows)

    # per site at the main cut-off
    sites = {}
    for f in common:
        st = sites.setdefault(f[:5], {"days": 0, "hh": 0, "true": 0, "found": 0, "fa": 0, "fa_hit": 0, "ours": 0,
                                      "on_r": 0, "on_fa": 0, "none": 0})
        st["days"] += 1
        for h, s in zip(by_file[f], res[f][0]):
            st["hh"] += 1
            if h["count"] > 0:
                st["true"] += 1
                st["found"] += s >= main_cut
            else:
                st["fa"] += 1
                st["fa_hit"] += s >= main_cut
        for (b, e, s), kind in zip(ours[f], res[f][1]):
            if s >= main_cut:
                st["ours"] += 1
                st[{"rumble": "on_r", "hh_false_alarm": "on_fa", "no_hh_box": "none"}[kind]] += 1
    site_header = ["site", "days", "HH detections", "HH confirmed rumbles", "we find", "% found",
                   "HH false alarms", "we also fire on", "our detections", "ours on a HH rumble",
                   "ours on a HH false alarm", "ours on no HH box"]
    site_rows = [[s, v["days"], v["hh"], v["true"], v["found"], pct(v["found"], v["true"]), v["fa"], v["fa_hit"],
                  v["ours"], v["on_r"], v["on_fa"], v["none"]] for s, v in sorted(sites.items())]
    write_csv(os.path.join(a.out, "per_site.csv"), site_header, site_rows)

    # recall by Hori-Harm score and by number of calls in the box
    band_rows = []
    for lo, hi in [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        t = [s for h, s in zip(hh_c, best) if h["count"] > 0 and lo <= h["score"] < hi]
        band_rows.append([f"{lo:.1f}-{min(hi, 1):.1f}", len(t), sum(s >= main_cut for s in t),
                          pct(sum(s >= main_cut for s in t), len(t))])
    calls_rows = []
    for k in sorted({h["count"] for h in hh_c if h["count"] > 0}):
        t = [s for h, s in zip(hh_c, best) if h["count"] == k]
        calls_rows.append([k, len(t), sum(s >= main_cut for s in t), pct(sum(s >= main_cut for s in t), len(t))])
    flag = [s for h, s in zip(hh_c, best) if h["count"] > 0 and h["notes"]]

    # every Hori-Harm box with our result, and the Raven tables to review
    write_csv(os.path.join(a.out, "hori_harm_boxes_matched.csv"),
              ["site", "file", "begin_s", "end_s", "hh_score", "count", "analyst_notes", "our_best_score",
               f"found_at_{main_cut}"],
              [[h["site"], h["file"], f"{h['begin']:.2f}", f"{h['end']:.2f}", h["score"], h["count"], h["notes"],
                f"{s:.4f}" if s >= 0 else "", int(s >= main_cut)] for h, s in zip(hh_c, best)])
    missed = [dict(path=h["path"], file=h["file"], begin=h["begin"], end=h["end"], score=h["score"],
                   **{"Site": h["site"], "HH count": h["count"], "Our best score": f"{s:.4f}" if s >= 0 else "none",
                      "Analyst notes": h["notes"]})
              for h, s in zip(hh_c, best) if h["count"] > 0 and s < main_cut]
    path_of = {f: by_file[f][0]["path"] for f in common}
    unmatched = [dict(path=path_of[f], file=f, begin=b, end=e, score=s, Site=f[:5])
                 for f in common for (b, e, s), kind in zip(ours[f], res[f][1]) if s >= main_cut and kind == "no_hh_box"]
    on_fa = [dict(path=path_of[f], file=f, begin=b, end=e, score=s, Site=f[:5])
             for f in common for (b, e, s), kind in zip(ours[f], res[f][1]) if s >= main_cut and kind == "hh_false_alarm"]
    random.Random(a.seed).shuffle(sample := list(unmatched))
    sample = sorted(sample[: a.sample], key=lambda r: (r["file"], r["begin"]))
    write_raven(os.path.join(a.out, "hori_harm_rumbles_we_missed.txt"), missed)
    write_raven(os.path.join(a.out, "ours_not_in_hori_harm.txt"), unmatched)
    write_raven(os.path.join(a.out, "ours_not_in_hori_harm_sample.txt"), sample)
    write_raven(os.path.join(a.out, "ours_on_hori_harm_false_alarms.txt"), on_fa)

    # ---------------------------------------------------------------- report.md
    m = next(r for r in cut_rows if r[0] == main_cut)
    hh04 = hh_rows[0]
    near = min((r for r in hh_rows if r[1] > 0), key=lambda r: abs(r[1] - m[1]))
    rep = [f"# Our detector vs Hori-Harm: {os.path.basename(os.path.normpath(a.ours))}", "",
           f"Made {time.strftime('%Y-%m-%d %H:%M')} by tools/compare_hori_harm.py.",
           f"Hori-Harm table: `{a.hh}`  ", f"Our results: `{a.ours}`  ",
           f"Our model: {settings['name']}, rule {rule}; main cut-off {main_cut}. "
           f"Two boxes match when they overlap or are at most {a.slack:g} s apart.", "",
           "## What is compared", "",
           f"- Hori-Harm table: {hh_info['rows']:,} rows. Dropped {hh_info['no_count']:,} rows with no Count (not reviewed: "
           f"an un-reviewed copy of a site) and {hh_info['duplicates']:,} exact duplicates, leaving **{len(hh):,} reviewed "
           f"detections on {len(by_file):,} recordings**.",
           f"- **{days:,} of those recordings are in our results** and are compared below. "
           f"{len(missing)} are not in our results"
           + (f" (sites {', '.join(sorted({f[:5] for f in missing}))})." if missing else "."),
           f"- On those {days:,} recordings Hori-Harm has **{n_true:,} confirmed rumble boxes** ({n_calls:,} calls; a box "
           f"can hold several) and **{n_fa:,} false alarms**.",
           ("- Self-check: on all " + f"{check['files']:,}" + " recordings, the saved probabilities give back exactly "
            "the saved table at the main cut-off" + (
                f" (apart from {check['borderline']:+d} detections scoring within 0.001 of it: the probabilities "
                "are stored rounded to float16, so a score that close can land on either side)."
                if check["borderline"] else ".") if have_low and not check["differ"] else
            f"- Self-check: {len(check['differ'])} of {check['files']:,} recordings' probabilities do NOT give back their "
            f"table (first: {', '.join(check['differ'][:5])}). Look at these before trusting the other cut-offs."
            if check["differ"] else "- Only our tables were used (no probabilities): one cut-off only."),
           "", "## Main result (our cut-off " + str(main_cut) + ")", "",
           table_md(["", "Hori-Harm (score >= 0.4)", "Ours"], [
               ["Detections", f"{hh04[1]:,}", f"{m[1]:,}"],
               ["Detections per day", hh04[2], m[2]],
               ["Hori-Harm's confirmed rumbles found", f"{n_true:,} (100%, by definition)", f"{m[3]:,} ({m[4]})"],
               ["Calls found", f"{n_calls:,}", f"{m[5]:,} ({m[6]})"],
               ["Hori-Harm's false alarms raised", f"{n_fa:,}", f"{m[7]:,} ({m[8]})"],
               ["Precision", hh04[5], f"between {m[12]} and {m[13]}"]]),
           "",
           f"Of our {m[1]:,} detections, {m[9]:,} are on a confirmed rumble, {m[10]:,} on a Hori-Harm false alarm, and "
           f"**{m[11]:,} on no Hori-Harm box**. Those last ones are either our false alarms or rumbles Hori-Harm missed "
           f"(scored under 0.4 or never boxed); nobody has looked at them yet. Our precision is therefore between "
           f"{m[12]} (all of them are false alarms) and {m[13]} (all of them are rumbles). "
           f"Listening to `ours_not_in_hori_harm_sample.txt` ({len(sample)} picked at random) pins it down.", "",
           f"For a like-for-like view: Hori-Harm with about as many detections as ours (score >= {near[0]}: "
           f"{near[1]:,}) keeps {near[3]:,} of its confirmed rumbles ({near[4]}), where we find {m[3]:,} ({m[4]}).", "",
           "**Read this with care:** the rumbles here are only the ones Hori-Harm found. Every confirmed rumble is "
           "by construction a Hori-Harm detection, so this measures how much of Hori-Harm's catch we also catch, "
           "not our true recall; rumbles that both detectors missed are not counted anywhere. "
           "Also, only recordings with at least one Hori-Harm detection are in the table, so a sampled day where "
           "Hori-Harm found nothing is missing here.", "",
           "## Our detector at other cut-offs", "",
           table_md(cut_header, cut_rows) if have_low else
           "(needs the .probs.npy files next to the tables; only the main cut-off was available)", "",
           "## Hori-Harm at higher score cut-offs", "",
           table_md(hh_header, hh_rows), "",
           "## How sure is the matching? (main cut-off, different slack)", "",
           table_md(["slack s", "HH rumbles found", "of", "%", "HH false alarms also raised"], slack_rows), "",
           "## Which of Hori-Harm's rumbles do we miss?", "",
           "By Hori-Harm's score (a rough guide to how clear the rumble is):", "",
           table_md(["HH score", "confirmed rumbles", "we find", "%"], band_rows), "",
           "By the number of calls the analyst counted in the box:", "",
           table_md(["calls in box", "boxes", "we find", "%"], calls_rows), "",
           f"Boxes with an analyst note (\"faint\", \"q\", \"insect?\" ...): we find {sum(s >= main_cut for s in flag)} of "
           f"{len(flag)} ({pct(sum(s >= main_cut for s in flag), len(flag))}).", "",
           "## Per site", "", table_md(site_header, site_rows), "",
           "## Files written", "",
           "- `report.md`: this report.",
           "- `our_cutoffs.csv`, `hori_harm_cutoffs.csv`, `per_site.csv`: the tables above, for Excel.",
           "- `hori_harm_boxes_matched.csv`: every reviewed Hori-Harm box with the best score of our detections on it.",
           f"- `hori_harm_rumbles_we_missed.txt` ({len(missed):,}): Raven table of the confirmed rumbles we miss at {main_cut}.",
           f"- `ours_not_in_hori_harm.txt` ({len(unmatched):,}) and `ours_not_in_hori_harm_sample.txt` ({len(sample)}): "
           "Raven tables of our detections with no Hori-Harm box, all and a random sample, with an empty Count column "
           "for a reviewer.",
           f"- `ours_on_hori_harm_false_alarms.txt` ({len(on_fa):,}): our detections on boxes the analysts called false alarms.",
           ]
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n")
    log(f"{m[3]:,} of {n_true:,} Hori-Harm rumbles found ({m[4]}); {m[11]:,} of our {m[1]:,} detections on no Hori-Harm box")
    log(f"report: {os.path.join(a.out, 'report.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
