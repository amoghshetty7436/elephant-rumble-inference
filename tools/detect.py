#!/usr/bin/env python
"""
Run the final rumble detector (the full rerun's final model and alarm rule, see
walkthrough/18) on new deployments: folders of recordings at any sample rate and of any
length. Raven tables go wherever --out points, which can be the network drive.

Any sample rate works without changes: every file is resampled to 500 Hz with
scipy's resample_poly (a proper anti-alias filter), exactly as the training and test
days were (fp_replica.load_audio). 32 kHz is divided by 64, 8 kHz by 16, 4 kHz by 8.

Three commands:

  survey  list every recording (count, hours, sample rates, formats, lengths, sizes) per
          site folder, time one file per sample rate (reading over the network, resampling,
          AVES, writing the table to --out and to a local folder), and estimate the total
          time. Writes survey.md and survey_files.csv to --out.
  run     detect rumbles in every recording. For each file: <name>.rumbles.txt (Raven table)
          and <name>.probs.npy (the rumble probability of every 0.64 s frame, ~0.3 MB a day,
          so the cut-off can be changed later without rerunning AVES). After each folder: one
          combined Raven table for the folder. Rerunning skips the files already done.
  table   rewrite the Raven tables from the saved .probs.npy files with another cut-off
          (seconds, no GPU).

  python tools\\detect.py survey L:\\path\\to\\deployment --out L:\\path\\to\\results
  python tools\\detect.py run    L:\\path\\to\\deployment --out L:\\path\\to\\results
  python tools\\detect.py table  --out L:\\path\\to\\results --min-score 0.95

The output folder mirrors the input folder's subfolders. Nothing is written outside --out.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset_configs as dc  # noqa: E402
from dataset_configs import log, fmt_dur, FRAME_SECONDS  # noqa: E402

AUDIO_EXT = (".wav", ".flac", ".aif", ".aiff")
DEFAULT_MODEL = os.path.join(dc.DEFAULT_SOURCE, "Configurations", "_study", "final_detections", "final_model.json")
RAVEN_HEADER = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)",
                "High Freq (Hz)", "Begin Path", "File Offset (s)", "Begin File", "Score"]
TABLE_SUFFIX = ".rumbles.txt"
PROBS_SUFFIX = ".probs.npy"
FOLDER_TABLE = "_all_rumbles.txt"
# measured on the lab PC (improve.md): AVES on the GPU takes about 31 s per 24 h of audio
AVES_S_PER_24H = 31.0


################################################################################
# finding the recordings
################################################################################

def find_audio(roots):
    """[(root, path)] of every recording under the roots, sorted by folder then name."""
    out = []
    for root in roots:
        if os.path.isfile(root):
            out.append((os.path.dirname(root), root))
            continue
        if not os.path.isdir(root):
            sys.exit(f"Can't find {root}. Is the drive connected?")
        for dirpath, dirs, names in os.walk(root):
            dirs.sort()
            out += [(root, os.path.join(dirpath, n)) for n in sorted(names)
                    if n.lower().endswith(AUDIO_EXT) and not n.startswith("._")]
    return out


def out_base(out, root, path):
    """<out>\\<input root's name>\\<subfolders>\\<file name>, without the extension."""
    sub = os.path.relpath(path, root)
    return os.path.join(out, os.path.basename(os.path.normpath(root)), os.path.splitext(sub)[0])


def audio_info(path):
    import soundfile as sf
    try:
        i = sf.info(path)
        return dict(sr=i.samplerate, channels=i.channels, hours=i.frames / i.samplerate / 3600,
                    format=f"{i.format} {i.subtype}", bytes=os.path.getsize(path), error="")
    except Exception as e:  # noqa: BLE001 - a broken file is reported, not fatal
        return dict(sr=0, channels=0, hours=0.0, format="?", bytes=os.path.getsize(path), error=str(e)[:200])


def site_of(path):
    """The site code at the start of an ELP file name (nn06a_..., dzan_...), or the folder."""
    m = re.match(r"([a-z]+\d+[a-z]?)_", os.path.basename(path).lower())
    return m.group(1) if m else os.path.basename(os.path.dirname(path))


################################################################################
# the model
################################################################################

class Detector:
    def __init__(self, model_json, device=None, fake=False):
        import torch
        import experiments as ex
        info = json.load(open(model_json))
        model_path = info["model"]
        if not os.path.exists(model_path):  # the folder was copied somewhere else
            model_path = os.path.join(os.path.dirname(os.path.dirname(model_json)), info["experiment"],
                                      os.path.basename(info["model"]))
        spec_path = os.path.join(os.path.dirname(model_path), "spec.json")
        if os.path.exists(spec_path):
            self.ctx = json.load(open(spec_path))["ctx"]
        else:
            self.ctx = int(re.search(r"ctx(\d+)", info["experiment"]).group(1))
        self.rule, self.cut_off, self.name = info["rule"], info["cut_off"], info["experiment"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m = ex.make_model(self.ctx)
        m.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = m.eval().to(self.device)
        self.embed = dc.get_embedder(fake)
        self.model_path = model_path
        log(f"model {self.name} (run {info.get('run')}), window {(2 * self.ctx + 1) * FRAME_SECONDS:.2f} s, "
            f"rule {self.rule}, cut-off {self.cut_off:.4f}, on {self.device}")

    def embeddings(self, y):
        return self.embed.whole_file(y)

    def probs(self, emb):
        import experiments as ex
        return ex.day_probs(self.model, emb, self.ctx, self.device)


def load(path):
    """500 Hz mono audio of one recording, and how long reading + resampling took."""
    t0 = time.time()
    y = dc.load_audio(path)
    return y, time.time() - t0


def detections(prob, rule, min_score):
    import improve as im
    return [(a, b, s) for a, b, s in zip(*im.alarms(prob, rule)) if s >= min_score]


def write_table(path, rows):
    """rows: (begin s in table, end s, full audio path, offset s in that file, score)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(dc.long_path(tmp), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(RAVEN_HEADER)
        for i, (bt, et, src, off, sc) in enumerate(rows, 1):
            wr.writerow([i, "Spectrogram 1", 1, f"{bt:.3f}", f"{et:.3f}", 5, 250, src, f"{off:.3f}",
                         os.path.basename(src), f"{sc:.4f}"])
    os.replace(dc.long_path(tmp), dc.long_path(path))


def file_rows(prob, rule, min_score, src):
    # times as in study.py's final_detections tables, which the test scores were computed on
    return [(a * FRAME_SECONDS, b * FRAME_SECONDS, src, a * FRAME_SECONDS, s)
            for a, b, s in detections(prob, rule, min_score)]


def folder_table(folder, files, rule, min_score):
    """One Raven table for a folder: its files one after another, as ELP's day tables are."""
    import numpy as np
    rows, elapsed = [], 0.0
    for src, base in files:
        p = base + PROBS_SUFFIX
        if not os.path.exists(dc.long_path(p)):
            continue
        prob = np.load(dc.long_path(p)).astype(np.float32)
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

def cmd_run(a):
    import numpy as np
    import improve as im
    im.keep_awake()
    files = find_audio(a.inputs)
    jobs = [(p, out_base(a.out, r, p)) for r, p in files]
    todo = [(p, b) for p, b in jobs if a.redo or not os.path.exists(dc.long_path(b + TABLE_SUFFIX))]
    log(f"{len(files)} recordings, {len(files) - len(todo)} already done, {len(todo)} to do; results in {a.out}")
    det = Detector(a.model, fake=a.fake_embedder)
    min_score = det.cut_off if a.min_score is None else a.min_score
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "detector.json"), "w") as f:
        json.dump({"experiment": det.name, "model": det.model_path, "rule": det.rule,
                   "cut_off": det.cut_off, "min_score_in_tables": min_score, "inputs": a.inputs}, f, indent=1)
    timing_path = os.path.join(a.out, "timings.csv")
    new = not os.path.exists(timing_path)
    tf = open(timing_path, "a", newline="")
    tw = csv.writer(tf)
    if new:
        tw.writerow(["file", "hours", "read_resample_s", "waited_s", "aves_s", "classify_s", "write_s", "detections"])
    done_hours, t_start, failed = 0.0, time.time(), []
    total_hours = sum(audio_info(p)["hours"] for p, _ in todo) if a.estimate else 0.0
    left_in_folder = {d: sum(1 for _, b in todo if os.path.dirname(b) == d) for d in {os.path.dirname(b) for _, b in todo}}
    # read + resample the next files in the background while AVES works on this one
    pool = cf.ThreadPoolExecutor(max(1, a.prefetch))
    ahead = {}

    def queue(i):
        if i < len(todo) and i not in ahead:
            ahead[i] = pool.submit(load, todo[i][0])

    for i in range(min(len(todo), a.prefetch)):
        queue(i)
    for i, (src, base) in enumerate(todo):
        queue(i + a.prefetch)
        try:
            t0 = time.time()
            y, t_load = ahead.pop(i).result()
            waited = time.time() - t0
            hours = y.shape[0] / 500 / 3600
            t1 = time.time()
            emb = det.embeddings(y)
            t2 = time.time()
            prob = det.probs(emb)
            t3 = time.time()
            os.makedirs(os.path.dirname(dc.long_path(base)), exist_ok=True)
            np.save(dc.long_path(base + PROBS_SUFFIX + ".tmp.npy"), prob.astype(np.float16))
            os.replace(dc.long_path(base + PROBS_SUFFIX + ".tmp.npy"), dc.long_path(base + PROBS_SUFFIX))
            rows = file_rows(prob, det.rule, min_score, src)
            write_table(base + TABLE_SUFFIX, rows)
            t4 = time.time()
        except Exception as e:  # noqa: BLE001 - one bad file must not stop a 3-day run
            log(f"FAILED {src}: {e}")
            failed.append(f"{src}\t{e}")
            with open(os.path.join(a.out, "failed.txt"), "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{src}\t{e}\n")
            continue
        del y, emb
        tw.writerow([src, f"{hours:.3f}", f"{t_load:.1f}", f"{waited:.1f}", f"{t2 - t1:.1f}", f"{t3 - t2:.2f}",
                     f"{t4 - t3:.2f}", len(rows)])
        tf.flush()
        done_hours += hours
        el = time.time() - t_start
        rate = el / max(done_hours, 1e-9) * 24
        left = f", about {fmt_dur((total_hours - done_hours) * el / done_hours)} left" if total_hours else ""
        log(f"[{i + 1}/{len(todo)}] {os.path.basename(src)}: {hours:.1f} h, {len(rows)} rumbles; "
            f"read+resample {t_load:.0f} s (waited {waited:.0f} s), AVES {t2 - t1:.0f} s, writing {t4 - t3:.1f} s; "
            f"{fmt_dur(rate)} per 24 h so far{left}")
        d = os.path.dirname(base)
        left_in_folder[d] -= 1
        if left_in_folder[d] == 0:
            n = folder_table(d, by_out_folder(jobs)[d], det.rule, min_score)
            log(f"folder table {os.path.join(d, FOLDER_TABLE)}: {n} rumbles")
    pool.shutdown()
    tf.close()
    log(f"finished: {done_hours:.0f} h of audio in {fmt_dur(time.time() - t_start)}"
        + (f"; {len(failed)} failed, see failed.txt" if failed else ""))


def cmd_table(a):
    import numpy as np
    info = json.load(open(os.path.join(a.out, "detector.json")))
    rule = info["rule"]
    min_score = info["cut_off"] if a.min_score is None else a.min_score
    jobs = []
    for dirpath, dirs, names in os.walk(a.out):
        dirs.sort()
        for n in sorted(names):
            if n.endswith(PROBS_SUFFIX):
                jobs.append((None, os.path.join(dirpath, n[:-len(PROBS_SUFFIX)])))
    src_of = {}
    timing = os.path.join(a.out, "timings.csv")
    if os.path.exists(timing):  # the original audio paths, for the Begin Path column
        for r in csv.DictReader(open(timing)):
            src_of[os.path.splitext(os.path.basename(r["file"]))[0]] = r["file"]
    jobs = [(src_of.get(os.path.basename(b), os.path.basename(b) + ".wav"), b) for _, b in jobs]
    n = 0
    for src, base in jobs:
        prob = np.load(dc.long_path(base + PROBS_SUFFIX)).astype(np.float32)
        rows = file_rows(prob, rule, min_score, src)
        write_table(base + TABLE_SUFFIX, rows)
        n += len(rows)
    for d, files in by_out_folder(jobs).items():
        folder_table(d, files, rule, min_score)
    info["min_score_in_tables"] = min_score
    with open(os.path.join(a.out, "detector.json"), "w") as f:
        json.dump(info, f, indent=1)
    log(f"rewrote {len(jobs)} tables with Score >= {min_score}: {n} rumbles")


################################################################################
# survey
################################################################################

def time_one(det, path, out_dir):
    """Seconds for each step on one file: raw network read, read + resample, AVES, classifier,
    and writing its outputs to --out compared with a local folder."""
    import numpy as np
    t0 = time.time()
    n = 0
    with open(dc.long_path(path), "rb") as f:  # the first read comes from the network
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            n += len(b)
    raw = time.time() - t0
    y, t_load = load(path)  # now mostly from Windows' file cache: the resampling part
    hours = y.shape[0] / 500 / 3600
    t1 = time.time()
    emb = det.embeddings(y)
    t2 = time.time()
    prob = det.probs(emb)
    t3 = time.time()
    rows = file_rows(prob, det.rule, det.cut_off, path)

    def write_test(folder):
        os.makedirs(folder, exist_ok=True)
        t = time.time()
        for k in range(20):
            b = os.path.join(folder, f"_write_test_{k}")
            np.save(b + PROBS_SUFFIX, prob.astype(np.float16))
            write_table(b + TABLE_SUFFIX, rows)
        s = (time.time() - t) / 20
        for k in range(20):
            for suf in (PROBS_SUFFIX, TABLE_SUFFIX):
                try:
                    os.remove(os.path.join(folder, f"_write_test_{k}{suf}"))
                except OSError:
                    pass
        return s

    w_out = write_test(os.path.join(out_dir, "_write_test"))
    shutil.rmtree(os.path.join(out_dir, "_write_test"), ignore_errors=True)
    local = tempfile.mkdtemp(prefix="rumble_write_test_")
    w_local = write_test(local)
    shutil.rmtree(local, ignore_errors=True)
    return dict(file=path, hours=hours, mb=n / 1e6, raw_read_s=raw, mb_per_s=n / 1e6 / max(raw, 1e-9),
                resample_s=t_load, aves_s=t2 - t1, classify_s=t3 - t2, detections=len(rows),
                write_out_s=w_out, write_local_s=w_local)


def cmd_survey(a):
    files = find_audio(a.inputs)
    if not files:
        sys.exit("No recordings found.")
    log(f"{len(files)} recordings; reading their headers...")
    rows = []
    for i, (root, p) in enumerate(files):
        info = audio_info(p)
        rows.append(dict(root=root, path=p, site=site_of(p), **info))
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(files)}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "survey_files.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = ["# Survey of the recordings", "", f"Made {time.strftime('%Y-%m-%d %H:%M')}.", ""]
    timings = {}
    for root in sorted({r["root"] for r in rows}):
        rs = [r for r in rows if r["root"] == root]
        ok = [r for r in rs if not r["error"]]
        hours = sum(r["hours"] for r in ok)
        sites = sorted({r["site"] for r in ok})
        md += [f"## {root}", "",
               f"- **{len(rs)} recordings**, {len(sites)} sites, **{hours:,.0f} hours** ({hours / 24:,.1f} recording-days), "
               f"{sum(r['bytes'] for r in rs) / 1e9:,.1f} GB",
               f"- sample rates: {', '.join(f'{sr} Hz x {sum(1 for r in ok if r['sr'] == sr)}' for sr in sorted({r['sr'] for r in ok}))}",
               f"- formats: {', '.join(sorted({r['format'] for r in ok}))}; channels: {sorted({r['channels'] for r in ok})}",
               f"- file length: median {statistics.median([r['hours'] for r in ok]):.2f} h, "
               f"shortest {min(r['hours'] for r in ok):.2f} h, longest {max(r['hours'] for r in ok):.2f} h"
               if ok else "- no readable files"]
        bad = [r for r in rs if r["error"]]
        short = [r for r in ok if r["hours"] < 0.01]
        if bad:
            md.append(f"- **{len(bad)} files could not be read** (listed in survey_files.csv, column error)")
        if short:
            md.append(f"- {len(short)} files are under 36 s long and will give no detections")
        md += ["", "| Site | Files | Hours | Days | Sample rate | First file | Last file |", "|---|---|---|---|---|---|---|"]
        for s in sites:
            ss = [r for r in ok if r["site"] == s]
            h = sum(r["hours"] for r in ss)
            md.append(f"| {s} | {len(ss)} | {h:,.1f} | {h / 24:.1f} | {', '.join(str(x) for x in sorted({r['sr'] for r in ss}))} "
                      f"| {os.path.basename(ss[0]['path'])} | {os.path.basename(ss[-1]['path'])} |")
        md.append("")

    if not a.no_timing:
        det = Detector(a.model, fake=a.fake_embedder)
        for sr in sorted({r["sr"] for r in rows if not r["error"]}):
            ok = [r for r in rows if r["sr"] == sr and not r["error"] and r["hours"] > 0.5]
            if not ok:
                continue
            pick = sorted(ok, key=lambda r: r["hours"])[len(ok) // 2]  # a typical-length file
            log(f"timing one {sr} Hz file: {pick['path']}")
            timings[sr] = time_one(det, pick["path"], a.out)
            log(f"  {timings[sr]}")
        md += ["## Measured on one file per sample rate", "",
               "| Sample rate | File | Hours | Network read | Read speed | Resample | AVES | Classifier | Write to --out | Write locally |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for sr, t in timings.items():
            md.append(f"| {sr} Hz | {os.path.basename(t['file'])} | {t['hours']:.1f} | {t['raw_read_s']:.1f} s "
                      f"({t['mb']:.0f} MB) | {t['mb_per_s']:.0f} MB/s | {t['resample_s']:.1f} s | {t['aves_s']:.1f} s "
                      f"| {t['classify_s']:.2f} s | {t['write_out_s'] * 1000:.0f} ms | {t['write_local_s'] * 1000:.0f} ms |")
        md.append("")

    md += ["## Estimated time for `run`", "",
           "Per 24 h of audio: reading + resampling happens in the background while AVES works on the "
           "previous file, so each file costs about the slower of the two, plus writing.", "",
           "| Folder | Hours | Read + resample per 24 h | AVES per 24 h | Estimate |", "|---|---|---|---|---|"]
    grand = 0.0
    for root in sorted({r["root"] for r in rows}):
        tot, reads, aveses = 0.0, [], []
        for sr in sorted({r["sr"] for r in rows if r["root"] == root and not r["error"]}):
            h = sum(r["hours"] for r in rows if r["root"] == root and r["sr"] == sr and not r["error"])
            t = timings.get(sr)
            if t:
                k = 24 / t["hours"]
                read = (t["raw_read_s"] + t["resample_s"]) * k
                aves = (t["aves_s"] + t["classify_s"]) * k
                write = t["write_out_s"] * k
            else:  # no timing: guess from 16-bit audio at ~100 MB/s and the measured AVES speed
                read = sr * 2 * 86400 / 100e6 + sr * 86400 / 40e6
                aves, write = AVES_S_PER_24H, 0.5
            tot += h / 24 * (max(read, aves) + write)
            reads.append(f"{fmt_dur(read)} ({sr} Hz)")
            aveses.append(fmt_dur(aves))
        grand += tot
        hours = sum(r["hours"] for r in rows if r["root"] == root and not r["error"])
        md.append(f"| {root} | {hours:,.0f} | {', '.join(reads)} | {', '.join(aveses)} | **{fmt_dur(tot)}** |")
    md += ["", f"**All folders: about {fmt_dur(grand)}** on this PC, run one after the other.",
           "" if timings else "(Guessed, not measured: run survey without --no-timing on the PC that will do the work.)"]
    with open(os.path.join(a.out, "survey.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    log(f"wrote {os.path.join(a.out, 'survey.md')} and survey_files.csv")


################################################################################

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["survey", "run", "table"])
    p.add_argument("inputs", nargs="*", help="deployment folders (searched with subfolders) or single files")
    p.add_argument("--out", required=True, help="where the results go (can be on the network drive)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"final_model.json of the rerun (default {DEFAULT_MODEL})")
    p.add_argument("--min-score", type=float, default=None,
                   help="keep detections scoring at least this (default: the model's cut-off at 80%% recall "
                        "on the validation days; 0 keeps every detection of the alarm rule)")
    p.add_argument("--prefetch", type=int, default=2, help="files read + resampled ahead in the background")
    p.add_argument("--redo", action="store_true", help="redo files that already have a table")
    p.add_argument("--no-estimate", dest="estimate", action="store_false", help="don't read every header first for the time left")
    p.add_argument("--no-timing", action="store_true", help="survey: only list the files, don't time one")
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.step != "table" and not a.inputs:
        p.error("give at least one deployment folder")
    a.out = os.path.abspath(a.out)
    for r in a.inputs:
        if dc.is_under(a.out, r):
            sys.exit(f"--out {a.out} is inside the input folder {r}; pick a folder outside the recordings.")
    {"survey": cmd_survey, "run": cmd_run, "table": cmd_table}[a.step](a)


if __name__ == "__main__":
    main()
