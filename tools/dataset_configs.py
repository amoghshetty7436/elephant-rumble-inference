#!/usr/bin/env python
"""
Build training/testing "configurations" of ELP's rumble data, train FruitPunch's
classifier on each one, and score each one two ways (FruitPunch's balanced
frames, and ELP's whole-day detector validation).

Nothing is written next to the original data. Every file this script writes
goes under --out (default <source>\\Configurations) or --cache, and it refuses
to run if either is inside L:\\ELP\\Detectors\\Elephant.

Configurations (see walkthrough/13 and the "Training and testing configurations" doc):

  A  train on clearing + forest            (what FruitPunch did)
  B  train on forest only
  C  train on clearing only
  D  train on forest + a few clearing days (as many clearing rumbles as forest ones)
  E  best of A-D + noise clips and "not rumble" frames from anywhere in forest days
  F  best + RumbleComplex + RumbleGeneral moved into training
     (validation becomes ~15 forest training days held out by site)
  G  best + Sample A moved into training (then only Sample B tests it)
  H  best + Complex + General + Sample A

  Test 1 = Sample B (Additional2024), Test 2 = Sample A (Grid-Forest, 43 schedule days),
  validation = Complex + General (used to pick the score threshold),
  Vehicle = the 3 Kakum days, scored only for false alarms per hour.

Steps (each can be re-run; finished work is kept):

  plan     make one folder per configuration, with copies of the original selection
           tables, the exact list of sound files used, and a README
  embed    run AVES over every day once; shared by all configurations (GPU, slow)
  run      train + score each configuration; results go in <configuration>\\results
  compare  one table across all configurations: <out>\\comparison.md
  all      plan, embed, run, compare

Example (Windows):

  python tools\\dataset_configs.py all --configs A B C D
  python tools\\dataset_configs.py all --configs E F G H --best B
"""
import argparse
import csv
import datetime
import json
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fp_data  # noqa: E402
import elp_score  # noqa: E402

DEFAULT_SOURCE = r"L:\ELP\Detectors\Other\Amogh\Dataset"
ORIGINAL_DATA = r"L:\ELP\Detectors\Elephant"  # read-only: never written to

FRAME_SECONDS = 0.64  # one AVES embedding per 320 samples at "500 Hz"
RANDOM_NEG_PER_DAY = 200  # E: extra "not rumble" frames per forest day
RANDOM_NEG_MIN_GAP = 30.0  # ...at least this many seconds from any rumble
HELDOUT_FOREST_DAYS = 15  # F/H validation
EXTRA_NEG_SHARE = 1 / 3  # E: share of the balanced negatives taken from noise/random frames

# Sample A: Schedule_File.xlsx (Nouabale-Ndoki rows) minus training day nn04a_20180608
SAMPLE_A_DAYS = """
nn01c_20180111 nn01c_20201017 nn01d_20210112 nn01e_20200101 nn02b_20200120 nn02d_20180112
nn02d_20181022 nn02d_20201013 nn03c_20180114 nn03c_20200124 nn03e_20181015 nn04a_20180116
nn04a_20180122 nn04a_20180710 nn04a_20180922 nn04a_20181021 nn04a_20191102 nn04a_20200104
nn04a_20200314 nn04a_20200429 nn04a_20201029 nn04a_20210124 nn04d_20200109 nn04e_20200120
nn05d_20201031 nn06a_20191008 nn06b_20180126 nn06b_20201005 nn06b_20201030 nn06b_20210131
nn06d_20200118 nn07b_20191022 nn07b_20200111 nn07b_20201027 nn07c_20180110 nn08c_20200104
nn08c_20201005 nn08c_20201031 nn08c_20210117 nn09a_20201025 nn09a_20201029 nn10a_20191006
nn10a_20201013""".split()

# The truth table calls this recording 20200428; the sound file (and the schedule) say 20200429.
NAME_FIXES = {"nn04a_20200428_": "nn04a_20200429_"}

SAMPLE_B_ODD_DAY = "nn08c_20230104"  # reported with and without

# Nouabale (nn) site_descriptions.xlsx
HABITAT = {}
for _h, _sites in {
    "mono-dominant": "01a 01f 01g 02a 02c 03b 03c 03e 04d 06a 06c 06e 06f 07c 09a 10b",
    "mixed-closed": "01c 01d 01e 02b 02d 02e 02f 03a 03d 03g 04a 04b 04c 04f 05a 05b 05c 05d 05f "
                    "06b 06d 07a 07b 08b 09b 09c",
    "mixed-open": "01b 02g 04e 08a 10a",
    "swamp": "03f 05e 08c",
}.items():
    HABITAT.update({"nn" + s: _h for s in _sites.split()})

NOISE_FOLDERS = ["thunder_8kHz", "rain_8kHz", "wind_8kHz", "vehicle_8kHz", "vehicle_500Hz",
                 "human_voice", "wings_8kHz", "silence_8kHz"]

# Every dataset, by its folders relative to the Datasets root (Training\..., Testing\...).
POOLS = {
    "clearing": dict(
        title="Clearing (Central African Republic, 56 days)",
        sounds=[r"Training\Sound_Files\rumble_clearing"],
        tables=[r"Training\Selection_Tables\rumble_clearing\rumble_clearing_00-24hr_56days.txt"],
        forest=False),
    "forest": dict(
        title="Forest training set (Nouabale-Ndoki)",
        sounds=[r"Training\Sound_Files\rumble_forest\TrainingSet_sounds"],
        tables=[r"Training\Selection_Tables\rumble_forest\nn_ele_hb_00-24hr_TrainingSet_Removed-nn03c20180611.txt"],
        forest=True),
    "complex": dict(
        title="RumbleComplex-Forest (30 days)",
        sounds=[r"Testing\Sounds\RumbleComplex-Forest"],
        tables=[r"Testing\Truth_Tables\nn_ele_00-24hr_ComplexTest_inc3NoEleDay.txt"],
        forest=True),
    "general": dict(
        title="RumbleGeneral-Forest (29 days)",
        sounds=[r"Testing\Sounds\RumbleGeneral-Forest"],
        tables=[r"Testing\Truth_Tables\nn_ele_00-24hr_GeneralTest_v4.txt"],
        forest=True),
    "sampleA": dict(
        title="Sample A: Nouabale-Ndoki_Grid-Forest (43 schedule days)",
        sounds=[r"Testing\Sounds\Nouabale-Ndoki_Grid-Forest\24-hr"],
        tables=[r"Testing\Truth_Tables\Nouabale-Ndoki_Grid_RandSmaple\NN_Random_Sample_Hand_ele.txt",
                r"Testing\Truth_Tables\Nouabale-Ndoki_Grid_RandSmaple\NN_Random_Sample_Hand_nn04aSupplemental_with_20200428.txt"],
        forest=True),
    "sampleB": dict(
        title="Sample B: Nouabale-Ndoki_Grid-Forest-Additional2024 (reviewed days)",
        sounds=[r"Testing\Sounds\Nouabale-Ndoki_Grid-Forest-Additional2024\sounds"],
        tables=[r"Testing\Truth_Tables\Nouabale-Ndoki_Grid-Forest-Additional2024\Nouabale-Ndoki_Grid-Forest-Additional2024_HB_eleOnly_combined.txt"],
        extra_folders=[r"Testing\Truth_Tables\Nouabale-Ndoki_Grid-Forest-Additional2024\raw_reviewed_tables"],
        forest=True),
    "vehicle": dict(
        title="Vehicle (3 Kakum days, false alarms only)",
        sounds=[r"Testing\Sounds\Vehicle"], tables=[], forest=False),
    "noise": dict(
        title="Noise clips (TrainingClips)",
        sounds=[r"Training\Sound_Files\TrainingClips" + "\\" + n for n in NOISE_FOLDERS], tables=[],
        forest=False),
}
WHOLE_DAY_POOLS = {"complex", "general", "sampleA", "sampleB", "vehicle", "noise"}

BASE_CONFIGS = {"A": ["clearing", "forest"], "B": ["forest"], "C": ["clearing"],
                "D": ["forest", "clearing-matched"]}
ALL_LETTERS = "ABCDEFGH"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


################################################################################
# safety: every write goes through Writer, which only writes under its roots
################################################################################

def _norm(p):
    p = os.path.abspath(p) if not re.match(r"^[A-Za-z]:[\\/]", p) else p
    return p.replace("\\", "/").rstrip("/").lower()


def is_under(path, root):
    p, r = _norm(path), _norm(root)
    return p == r or p.startswith(r + "/")


class Writer:
    def __init__(self, roots, forbidden):
        for r in roots:
            for f in forbidden:
                if is_under(r, f):
                    sys.exit(f"Refusing to write under {f}: {r} is inside it. "
                             f"Pick an --out / --cache outside the original data.")
        self.roots = roots

    def check(self, path):
        if not any(is_under(path, r) for r in self.roots):
            raise RuntimeError(f"internal error: refusing to write outside {self.roots}: {path}")
        return long_path(path)

    def makedirs(self, path):
        os.makedirs(self.check(path), exist_ok=True)

    def open(self, path, mode="w"):
        self.makedirs(os.path.dirname(path))
        return open(self.check(path), mode, newline="", encoding=None if "b" in mode else "utf-8")

    def copy(self, src, dst):
        self.makedirs(os.path.dirname(dst))
        dst = self.check(dst)
        if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
            shutil.copy2(src, dst)

    def json(self, path, obj):
        with self.open(path + ".tmp") as f:
            json.dump(obj, f, indent=1)
        os.replace(self.check(path + ".tmp"), self.check(path))


def long_path(path):
    """Windows refuses paths over 260 characters unless they start with \\\\?\\."""
    if os.name == "nt" and len(os.path.abspath(path)) >= 240 and not path.startswith("\\\\?\\"):
        p = os.path.abspath(path)
        return "\\\\?\\UNC\\" + p[2:] if p.startswith("\\\\") else "\\\\?\\" + p
    return path


def rel(path_with_backslashes):
    return os.path.join(*path_with_backslashes.split("\\"))


def find_dataset_root(source):
    for cand in (source, os.path.join(source, "Datasets")):
        if os.path.isdir(os.path.join(cand, "Training")) and os.path.isdir(os.path.join(cand, "Testing")):
            return cand
    sys.exit(f"Can't find Training and Testing folders in {source} (or {source}\\Datasets).")


################################################################################
# labels
################################################################################

def file_key(name):
    """'L:\\...\\nn04a_20200428_221358.wav' -> 'nn04a_20200429_221358' (lower case, no extension)."""
    k = elp_score.normalize_file_name(name)
    for a, b in NAME_FIXES.items():
        k = k.replace(a, b)
    return k


def day_of(key):
    """'nn04a_20200429_221358' -> 'nn04a_20200429'."""
    return "_".join(key.split("_")[:2])


def _f(row, *names):
    for n in names:
        v = row.get(n)
        if v not in (None, ""):
            try:
                return float(v)
            except ValueError:
                pass
    return None


def read_truth(paths):
    """Rumble boxes from Raven tables, per sound file.

    Handles every truth-table layout in the datasets (Begin File or filename,
    End-Begin or a duration column, a missing Low Freq column). Rows marked
    DUMMY_NoEles are placeholders for elephant-free days: they are dropped, and
    the day is remembered as reviewed. Returns ({file key: [Label]}, {dummy file keys}).
    """
    by_file, dummy = defaultdict(list), set()
    for path in paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for raw in reader:
                row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items() if k}
                name = row.get("begin file") or row.get("filename") or row.get("begin path")
                if not name:
                    continue
                key = file_key(name)
                if any("dummy" in v.lower() for v in row.values()):
                    dummy.add(key)
                    continue
                start = _f(row, "file offset (s)")
                dur = _f(row, "duration", "delta time (s)")
                b, e = _f(row, "begin time (s)"), _f(row, "end time (s)")
                if dur is None and b is not None and e is not None:
                    dur = e - b
                if start is None or dur is None:
                    continue
                lf = _f(row, "low freq (hz)") or 0.0
                hf = _f(row, "high freq (hz)") or 250.0
                by_file[key].append(fp_data.Label(start, dur, lf, hf, key))
    for v in by_file.values():
        v.sort(key=lambda l: l.start)
    return dict(by_file), dummy


def find_wavs(folder):
    """{file key: full path} for .wav files under a folder (recursive)."""
    out = {}
    if not os.path.isdir(folder):
        return out
    for dirpath, _, names in os.walk(folder):
        for n in names:
            if n.lower().endswith(".wav"):
                out[file_key(n)] = os.path.join(dirpath, n)
    return out


################################################################################
# what is in each dataset
################################################################################

class Pool:
    """One dataset: its sound files, labels, and which files count."""

    def __init__(self, name, root):
        self.name, self.spec = name, POOLS[name]
        self.root = root
        self.sound_dirs = [os.path.join(root, rel(s)) for s in self.spec["sounds"]]
        self.table_paths = [os.path.join(root, rel(t)) for t in self.spec["tables"]]
        missing = [p for p in self.sound_dirs + self.table_paths if not os.path.exists(p)]
        self.missing = missing
        self.labels, self.dummy = read_truth([p for p in self.table_paths if os.path.exists(p)])
        self.sounds = {}
        self.sound_dir_of = {}
        for d, s in zip(self.sound_dirs, self.spec["sounds"]):
            for k, p in find_wavs(d).items():
                self.sounds[k], self.sound_dir_of[k] = p, s
        self.files = self._select()

    def _select(self):
        n = self.name
        all_files = sorted(self.sounds)
        if n == "sampleA":
            days = set(SAMPLE_A_DAYS)
            return [k for k in all_files if day_of(k) in days]
        if n == "sampleB":
            folder = os.path.join(self.root, rel(self.spec["extra_folders"][0]))
            days = {day_of(file_key(f.split(".wav")[0])) for f in os.listdir(folder)} if os.path.isdir(folder) else None
            return [k for k in all_files if days is None or day_of(k) in days]
        if n == "vehicle":
            return [k for k in all_files if not k.startswith("nn")]  # Kakum days: no elephants
        if n in ("clearing", "forest"):
            return [k for k in all_files if k in self.labels]  # browsed days only
        return all_files  # complex, general: every day was reviewed (no-elephant days have DUMMY rows)

    def n_rumbles(self, files=None):
        return sum(len(self.labels.get(k, [])) for k in (self.files if files is None else files))


def heldout_forest_days(forest, seed=0, n_days=None):
    """~15 forest training days, whole sites at a time, for F/H validation."""
    n_days = n_days or HELDOUT_FOREST_DAYS
    by_site = defaultdict(list)
    for k in forest.files:
        by_site[k[:5]].append(k)
    sites = sorted(by_site)
    random.Random(seed).shuffle(sites)
    out = []
    for s in sites:
        if len(out) >= n_days:
            break
        out += by_site[s]
    return sorted(out)


def matched_clearing_days(clearing, forest, seed=0):
    """Random clearing days until they hold at least as many rumbles as the forest set."""
    target = forest.n_rumbles()
    days = [k for k in clearing.files if is_usable(clearing.labels.get(k, []))]
    random.Random(seed).shuffle(days)
    out, n = [], 0
    for k in days:
        if n >= target:
            break
        out.append(k)
        n += len(clearing.labels[k])
    return sorted(out)


def is_usable(labels):
    """FruitPunch's rule for a training day: at least 2 rumbles and 2 gap windows."""
    return len(labels) >= 2 and len(fp_data.negative_windows(labels)) >= 2


################################################################################
# configurations
################################################################################

def load_pools(root):
    pools = {n: Pool(n, root) for n in POOLS}
    for p in pools.values():
        if p.missing:
            log(f"WARNING {p.name}: not found: {p.missing}")
        log(f"{p.name}: {len(p.files)} sound files used ({len(p.sounds)} in folder), "
            f"{p.n_rumbles()} rumbles")
    return pools


def resolve_components(letter, best, out):
    """Training components of a configuration, e.g. ['forest', 'clearing-matched', 'noise']."""
    if letter in BASE_CONFIGS:
        return list(BASE_CONFIGS[letter])
    if not best:
        return None
    if best in BASE_CONFIGS:
        base = list(BASE_CONFIGS[best])
    else:  # e.g. --best E: reuse what E was built from
        cfg = find_config_dir(out, best)
        if not cfg:
            sys.exit(f"--best {best}: no {best}_* folder in {out}; run `plan --configs {best}` first.")
        base = json.load(open(os.path.join(cfg, "config.json")))["components"]
    if letter == "E":
        if best not in BASE_CONFIGS:
            sys.exit("--best for E must be one of A, B, C, D")
        return base + ["noise"]
    extra = {"F": ["complex", "general"], "G": ["sampleA"], "H": ["complex", "general", "sampleA"]}[letter]
    return base + [c for c in extra if c not in base]


def find_config_dir(out, letter):
    if not os.path.isdir(out):
        return None
    for d in sorted(os.listdir(out)):
        if d.startswith(letter + "_") and os.path.exists(os.path.join(out, d, "config.json")):
            return os.path.join(out, d)
    return None


def design(letter, components, pools, seed):
    """Which files go to training, validation and testing for one configuration."""
    train, n_clear = {}, 0
    heldout = []
    for c in components:
        if c == "clearing-matched":
            train["clearing"] = matched_clearing_days(pools["clearing"], pools["forest"], seed)
            n_clear = len(train["clearing"])
        elif c == "noise":
            train["noise"] = list(pools["noise"].files)
        else:
            train[c] = list(pools[c].files)
    moved = {"complex", "general"} <= set(components)
    if moved:  # Complex + General are training now: validate on held-out forest training days
        heldout = heldout_forest_days(pools["forest"], seed)
        if "forest" in train:
            train["forest"] = [k for k in train["forest"] if k not in set(heldout)]
        validation = {"name": f"{len(heldout)} held-out forest training days", "files": {"forest": heldout}}
    else:
        validation = {"name": "Complex + General", "files": {"complex": pools["complex"].files,
                                                             "general": pools["general"].files}}
    tests = [{"name": "Test 1: Sample B", "key": "sampleB", "files": {"sampleB": pools["sampleB"].files}}]
    if "sampleA" not in components:
        tests.append({"name": "Test 2: Sample A", "key": "sampleA", "files": {"sampleA": pools["sampleA"].files}})
    tests.append({"name": "Vehicle (false alarms only)", "key": "vehicle", "files": {"vehicle": pools["vehicle"].files}})

    # never score a day the model trained on
    train_days = {day_of(k) for fs in train.values() for k in fs}
    dropped = []
    for s in [validation] + tests:
        for p, fs in s["files"].items():
            keep = [k for k in fs if day_of(k) not in train_days]
            dropped += [k for k in fs if day_of(k) in train_days]
            s["files"][p] = keep
    parts = []
    for c in components:
        parts.append(f"clearing{n_clear}days" if c == "clearing-matched" else c)
    name = f"{letter}_train-" + "+".join(parts)
    if moved:
        name += f"_val-{len(heldout)}forestdays"
    return {"letter": letter, "name": name, "components": components, "seed": seed,
            "train": train, "validation": validation, "tests": tests,
            "dropped_from_scoring_because_trained_on": sorted(dropped),
            "random_negatives": "noise" in components}


################################################################################
# plan: folders, copied tables, file lists, README
################################################################################

def role_path(role, original):
    """'Testing\\Sounds\\RumbleComplex-Forest' as training data -> 'Training\\Sounds\\RumbleComplex-Forest'."""
    parts = original.split("\\")
    return os.path.join(role, *parts[1:])


def write_pool_files(w, cfg_dir, role, pool, files, copy_audio):
    by_dir = defaultdict(list)
    for k in files:
        by_dir[pool.sound_dir_of[k]].append(k)
    for d, ks in sorted(by_dir.items()):
        folder = os.path.join(cfg_dir, role_path(role, d))
        with w.open(os.path.join(folder, "sound_files.txt")) as f:
            f.write("\n".join(pool.sounds[k] for k in sorted(ks)) + "\n")
        if copy_audio:
            for k in ks:
                w.copy(pool.sounds[k], os.path.join(folder, os.path.basename(pool.sounds[k])))
    for t, path in zip(pool.spec["tables"], pool.table_paths):
        if os.path.exists(path):
            w.copy(path, os.path.join(cfg_dir, role_path(role, t)))
    for extra in pool.spec.get("extra_folders", []):
        src = os.path.join(pool.root, rel(extra))
        if os.path.isdir(src):
            for n in os.listdir(src):
                w.copy(os.path.join(src, n), os.path.join(cfg_dir, role_path(role, extra), n))


def pool_rows(cfg, pools):
    rows = []
    for role, groups in [("Training", [{"name": "", "files": cfg["train"]}]),
                         ("Validation", [cfg["validation"]]), ("Testing", cfg["tests"])]:
        for g in groups:
            for p, files in g["files"].items():
                pool = pools[p]
                rows.append({
                    "role": role, "set": g["name"], "dataset": pool.spec["title"],
                    "folders": "; ".join(sorted({pool.sound_dir_of[k] for k in files})) or pool.spec["sounds"][0],
                    "files": len(files), "rumbles": pool.n_rumbles(files),
                    "no_rumble_days": sum(1 for k in files if not pool.labels.get(k)),
                    "fruitpunch_usable": sum(1 for k in files if is_usable(pool.labels.get(k, []))),
                })
    return rows


def cmd_plan(a, pools, w):
    letters = a.configs
    for letter in letters:
        comps = resolve_components(letter, a.best, a.out)
        if comps is None:
            log(f"{letter}: skipped, needs --best (the winner of A-D, or E)")
            continue
        cfg = design(letter, comps, pools, a.seed)
        old = find_config_dir(a.out, letter)
        if old and os.path.basename(old) != cfg["name"]:
            log(f"{letter}: NOTE an older {os.path.basename(old)} exists; leaving it, writing {cfg['name']}")
        cfg_dir = os.path.join(a.out, cfg["name"])
        w.makedirs(cfg_dir)
        for p, files in cfg["train"].items():
            write_pool_files(w, cfg_dir, "Training", pools[p], files, a.copy_audio)
        for p, files in cfg["validation"]["files"].items():
            write_pool_files(w, cfg_dir, "Validation", pools[p], files, a.copy_audio)
        for t in cfg["tests"]:
            for p, files in t["files"].items():
                write_pool_files(w, cfg_dir, "Testing", pools[p], files, a.copy_audio)
        cfg["source_root"] = pools["forest"].root
        cfg["created"] = datetime.datetime.now().isoformat(timespec="seconds")
        cfg["datasets"] = pool_rows(cfg, pools)
        w.json(os.path.join(cfg_dir, "config.json"), cfg)
        write_readme(w, cfg_dir, cfg)
        log(f"{letter}: wrote {cfg_dir}")


def write_readme(w, cfg_dir, cfg):
    t = [f"# Configuration {cfg['name']}", "",
         f"Training data: {' + '.join(cfg['components'])}.", "",
         "Each folder below mirrors the original `L:\\ELP\\Detectors\\Elephant\\Datasets` layout, "
         "with `Training`, `Validation` or `Testing` in front to say what the data is used for here. "
         "The selection tables are unchanged copies of the originals. `sound_files.txt` lists the "
         "exact .wav files used (the audio itself was not copied).", "",
         "| Used for | Set | Dataset | Original folder | Days | Rumbles | Days with no rumbles | Days FruitPunch can train on |",
         "|---|---|---|---|---|---|---|---|"]
    for r in cfg["datasets"]:
        t.append(f"| {r['role']} | {r['set']} | {r['dataset']} | `{r['folders']}` | {r['files']} | "
                 f"{r['rumbles']} | {r['no_rumble_days']} | {r['fruitpunch_usable']} |")
    t += ["", "Notes:",
          "- Training follows FruitPunch: a day is used only if it has at least 2 rumbles and 2 gaps "
          "between rumbles; negatives come from the gaps; negatives are cut to as many as positives.",
          "- DUMMY_NoEles rows are not rumbles: those days count as reviewed days with no elephants.",
          "- Validation picks the score threshold (best F1 by ELP scoring); the tests never influence it."]
    if cfg["random_negatives"]:
        t.append(f"- Extra negatives: every noise clip, plus up to {RANDOM_NEG_PER_DAY} frames per forest "
                 f"training day at least {RANDOM_NEG_MIN_GAP:.0f} s from any rumble; together they are "
                 f"{EXTRA_NEG_SHARE:.0%} of the balanced negatives.")
    if cfg["dropped_from_scoring_because_trained_on"]:
        t.append(f"- Left out of scoring because the model trained on the same day: "
                 f"{', '.join(cfg['dropped_from_scoring_because_trained_on'])}")
    t += ["", "Results, once `run` has finished: `results\\summary.md`."]
    with w.open(os.path.join(cfg_dir, "README.md")) as f:
        f.write("\n".join(t) + "\n")


################################################################################
# embed: one shared cache of AVES embeddings
################################################################################

def label_frames(labels):
    """Frame indices of rumbles and of FruitPunch's gap negatives for one day."""
    def idx(ls):
        out = []
        for l in ls:
            i0 = max(0, fp_data.time_to_index(l.start))
            i1 = max(i0, fp_data.time_to_index(l.start + l.duration))
            out.extend(range(i0, i1))
        return out
    return idx(labels), idx(fp_data.negative_windows(labels)) if labels else []


def random_negative_frames(labels, n_frames, key, n=RANDOM_NEG_PER_DAY):
    """Frames at least RANDOM_NEG_MIN_GAP seconds from every rumble (same pick every run)."""
    bad = bytearray(n_frames)
    pad = int(RANDOM_NEG_MIN_GAP / FRAME_SECONDS)
    for l in labels:
        i0 = max(0, fp_data.time_to_index(l.start) - pad)
        i1 = min(n_frames, fp_data.time_to_index(l.start + l.duration) + pad + 1)
        bad[i0:i1] = b"\x01" * max(0, i1 - i0)
    ok = [i for i in range(n_frames) if not bad[i]]
    rng = random.Random(f"{key}-random-negatives")
    return sorted(rng.sample(ok, min(n, len(ok))))


def cache_path(cache, pool, key, kind):
    """kind 'day' = every frame of the day; 'frames' = only the frames training needs."""
    return os.path.join(cache, pool, f"{key}.{kind}.pt")


def needed_files(pools, seed):
    """(pool, key, whole_day) for everything any configuration may need."""
    heldout = set(heldout_forest_days(pools["forest"], seed))
    jobs = []
    for name, pool in pools.items():
        for k in pool.files:
            jobs.append((name, k, name in WHOLE_DAY_POOLS or (name == "forest" and k in heldout)))
    return jobs


class FakeEmbedder:
    """Stand-in for AVES used by the tests: per-frame band energies, no model download."""

    def whole_file(self, y):
        import torch
        n = y.shape[0] // 320
        if n == 0:
            return torch.empty(0, 768)
        fr = y[:n * 320].view(n, 320)
        spec = torch.fft.rfft(fr, dim=1).abs()[:, :161]  # 0-250 Hz at "500 Hz"
        feat = torch.log1p(torch.cat([spec, spec[:, :96] ** 2, torch.zeros(n, 768 - 161 - 96)], 1))
        return feat / feat.norm(dim=1, keepdim=True).clamp_min(1e-6)


def get_embedder(fake):
    if fake:
        return FakeEmbedder()
    import fp_replica
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("embedding on", device, torch.cuda.get_device_name(0) if device == "cuda" else "(slow without a GPU)")
    return fp_replica.Embedder(device)


def load_audio(path):
    import fp_replica
    return fp_replica.load_audio(path)


def frame_metrics(logits, y):
    """Rumble-class precision/recall/F1, accuracy and ROC AUC, as fp_replica.report prints them."""
    import torch
    import fp_replica
    pred = logits.argmax(1)
    tp = ((pred == 1) & (y == 1)).sum().item()
    p = tp / max(1, (pred == 1).sum().item())
    r = tp / max(1, (y == 1).sum().item())
    return {"support": int((y == 1).sum().item()), "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "accuracy": (pred == y).float().mean().item(),
            "roc_auc": fp_replica.roc_auc(torch.softmax(logits, 1)[:, 1], y)}


def cmd_embed(a, pools, w):
    import torch
    jobs = []
    for pool, key, whole in needed_files(pools, a.seed):
        dst = cache_path(a.cache, pool, key, "day" if whole else "frames")
        if os.path.exists(dst) or (not whole and os.path.exists(cache_path(a.cache, pool, key, "day"))):
            continue
        jobs.append((pool, key, whole, dst))
    log(f"{len(jobs)} files to embed (files already in {a.cache} are skipped)")
    if not jobs:
        return
    embed = get_embedder(a.fake_embedder)
    t_start = time.time()
    for n, (pool, key, whole, dst) in enumerate(jobs, 1):
        t0 = time.time()
        y = load_audio(pools[pool].sounds[key])
        day = embed.whole_file(y)
        if whole:
            item = {"kind": "day", "n": day.shape[0], "emb": day.half()}
        else:
            labels = pools[pool].labels.get(key, [])
            pos, neg = label_frames(labels)
            rnd = random_negative_frames(labels, day.shape[0], key) if POOLS[pool]["forest"] else []
            idx = sorted({i for i in pos + neg + rnd if i < day.shape[0]})
            idx_t = torch.tensor(idx, dtype=torch.long)
            item = {"kind": "frames", "n": day.shape[0], "idx": idx_t, "emb": day[idx_t].half()}
        w.makedirs(os.path.dirname(dst))
        torch.save(item, w.check(dst + ".tmp"))
        os.replace(dst + ".tmp", dst)
        el = time.time() - t_start
        log(f"{n}/{len(jobs)} {pool} {key}: {day.shape[0]} frames ({time.time() - t0:.0f} s), "
            f"about {el / n * (len(jobs) - n) / 60:.0f} min left")


class Day:
    """Embeddings of one day from the cache, looked up by frame index."""

    def __init__(self, cache, pool, key):
        import torch
        path = cache_path(cache, pool, key, "day")
        if not os.path.exists(path):
            path = cache_path(cache, pool, key, "frames")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing: run the embed step first")
        d = torch.load(path)
        self.kind, self.n, self.emb = d["kind"], d["n"], d["emb"]
        if self.kind == "frames":
            self.pos = {int(i): j for j, i in enumerate(d["idx"].tolist())}

    def frames(self, idx):
        import torch
        idx = [i for i in idx if i < self.n]
        if self.kind == "day":
            return self.emb[torch.tensor(idx, dtype=torch.long)] if idx else self.emb[:0]
        rows = [self.pos[i] for i in idx if i in self.pos]
        return self.emb[torch.tensor(rows, dtype=torch.long)] if rows else self.emb[:0]

    def whole(self):
        if self.kind != "day":
            raise RuntimeError("this day was cached as frames only")
        return self.emb

    @property
    def hours(self):
        return self.n * FRAME_SECONDS / 3600


################################################################################
# run: train and score one configuration
################################################################################

def training_data(cfg, pools, cache, seed):
    import torch
    pos, neg, extra = [], [], []
    used, skipped = defaultdict(int), defaultdict(int)
    for p, files in cfg["train"].items():
        for k in files:
            day = Day(cache, p, k)
            if p == "noise":
                extra.append(day.whole())
                used[p] += 1
                continue
            labels = pools[p].labels.get(k, [])
            if is_usable(labels):
                pi, ni = label_frames(labels)
                pos.append(day.frames(pi))
                neg.append(day.frames(ni))
                used[p] += 1
            else:
                skipped[p] += 1
            if cfg["random_negatives"] and POOLS[p]["forest"]:
                extra.append(day.frames(random_negative_frames(labels, day.n, k)))
    pos, neg = torch.cat(pos), torch.cat(neg)
    extra = torch.cat(extra) if extra else neg[:0]
    g = torch.Generator().manual_seed(seed)
    n_extra = min(extra.shape[0], int(pos.shape[0] * EXTRA_NEG_SHARE)) if cfg["random_negatives"] else 0
    n_gap = min(neg.shape[0], pos.shape[0] - n_extra)
    negs = torch.cat([neg[torch.randperm(neg.shape[0], generator=g)[:n_gap]],
                      extra[torch.randperm(extra.shape[0], generator=g)[:n_extra]]])
    x = torch.cat([pos, negs]).float()
    y = torch.cat([torch.ones(pos.shape[0], dtype=torch.long), torch.zeros(negs.shape[0], dtype=torch.long)])
    info = {"days_used": dict(used), "days_skipped_fruitpunch_rule": dict(skipped),
            "rumble_frames": pos.shape[0], "gap_negative_frames_available": neg.shape[0],
            "negatives_used_from_gaps": n_gap, "negatives_used_from_noise_and_random": n_extra}
    return torch.utils.data.TensorDataset(x, y), info


def train(ds, minutes, seed, logf):
    """FruitPunch's training loop (fp_replica.cmd_train): 95/5 split, Adam, batch 64, time budget."""
    import torch
    from fp_replica import ElephantRumbleClassifier
    torch.manual_seed(seed)
    tr, va = torch.utils.data.random_split(ds, [0.95, 0.05], generator=torch.Generator().manual_seed(42))
    loader = torch.utils.data.DataLoader(tr, batch_size=64, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))
    xv, yv = ds.tensors[0][va.indices], ds.tensors[1][va.indices]
    y = ds.tensors[1]
    weights = y.shape[0] / (2 * torch.bincount(y, minlength=2).float())
    model = ElephantRumbleClassifier()
    crit = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters())
    t0, min_val = time.time(), float("inf")
    for epoch in range(20000):
        model.train()
        tot = cnt = 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.shape[0]
            cnt += xb.shape[0]
        model.eval()
        with torch.no_grad():
            val = crit(model(xv), yv).item()
        msg = f"epoch {epoch + 1}: train loss {tot / max(1, cnt) * 100:.2f}, held-out 5% loss {val * 100:.2f}"
        log(msg)
        logf.write(msg + "\n")
        if val > min_val * 1.15 * 2:
            logf.write("held-out loss rising, stopping\n")
            break
        min_val = min(min_val, val)
        if time.time() - t0 > minutes * 60:
            break
    model.eval()
    return model


def logits_of(model, x, batch=100_000):
    import torch
    with torch.no_grad():
        return torch.cat([model(x[i:i + batch].float()) for i in range(0, x.shape[0], batch)]) \
            if x.shape[0] else torch.empty(0, 2)


def fruitpunch_metrics(pos_logits, neg_logits, seed):
    """FruitPunch's way: every rumble frame vs as many gap-negative frames, each on its own."""
    import torch
    if not pos_logits:
        return None
    pos, neg = torch.cat(pos_logits), torch.cat(neg_logits)
    g = torch.Generator().manual_seed(seed)
    neg = neg[torch.randperm(neg.shape[0], generator=g)[:pos.shape[0]]]
    y = torch.cat([torch.ones(pos.shape[0], dtype=torch.long), torch.zeros(neg.shape[0], dtype=torch.long)])
    return frame_metrics(torch.cat([pos, neg]), y)


def detect(model, cfg_set, pools, cache, w, out_path, seed):
    """Whole-day detections (FruitPunch's rule: 4+ frames in a row) -> Raven table + Boxes,
    plus FruitPunch's balanced-frame metrics from the same logits."""
    import torch
    from fp_replica import segments
    boxes, hours, files = [], 0.0, []
    pos_lg, neg_lg = [], []
    cols = ["Selection", "View", "Channel", "Begin Time (s)", "End Time (s)", "Low Freq (Hz)",
            "High Freq (Hz)", "Begin Path", "File Offset (s)", "Begin File", "Score"]
    sel, elapsed = 0, 0.0
    with w.open(out_path) as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(cols)
        for p, keys in cfg_set["files"].items():
            for k in keys:
                day = Day(cache, p, k)
                lg = logits_of(model, day.whole())
                prob = torch.softmax(lg, 1)[:, 1]
                labels = pools[p].labels.get(k, [])
                if labels:
                    pi, ni = label_frames(labels)
                    pos_lg.append(lg[[i for i in pi if i < lg.shape[0]]])
                    neg_lg.append(lg[[i for i in ni if i < lg.shape[0]]])
                for s0, s1 in segments(lg[:, 1] > lg[:, 0]):
                    bt, et = s0 * FRAME_SECONDS, s1 * FRAME_SECONDS
                    sc = prob[s0:s1 + 1].mean().item()
                    sel += 1
                    path = pools[p].sounds[k]
                    wr.writerow([sel, "Spectrogram 1", 1, f"{elapsed + bt:.3f}", f"{elapsed + et:.3f}", 5, 250,
                                 path, f"{bt:.3f}", os.path.basename(path), f"{sc:.4f}"])
                    boxes.append(elp_score.Box(k, bt, et, 5.0, 250.0, sc))
                elapsed += day.hours * 3600
                hours += day.hours
                files.append((p, k, day.hours))
    return boxes, files, fruitpunch_metrics(pos_lg, neg_lg, seed)


def truth_boxes(pools, files):
    out = []
    for p, k, _ in files:
        for l in pools[p].labels.get(k, []):
            out.append(elp_score.Box(k, l.start, l.start + l.duration, l.low_freq, l.high_freq))
    return out


def elp_sweep(truth, dets, files, keep=None):
    files = [f for f in files if keep is None or keep(f[1])]
    if not files:
        return None
    hours = sum(h for _, _, h in files)
    rows = elp_score.score(truth, dets, [k for _, k, _ in files], hours_per_file=hours / len(files))
    for r in rows:
        p, rc = r["precision"], r["recall"]
        r["f1"] = 2 * p * rc / (p + rc) if p == p and rc == rc and p + rc > 0 else 0.0
        r["hours"] = hours
        r["days"] = len(files)
    return rows


def at(rows, thr):
    return min(rows, key=lambda r: abs(r["threshold"] - thr)) if rows else None


def write_csv(w, path, rows):
    with w.open(path) as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def cmd_run(a, pools, w):
    import torch
    for letter in a.configs:
        cfg_dir = find_config_dir(a.out, letter)
        if not cfg_dir:
            log(f"{letter}: no configuration folder yet (run `plan` first), skipping")
            continue
        cfg = json.load(open(os.path.join(cfg_dir, "config.json")))
        res = os.path.join(cfg_dir, "results")
        if os.path.exists(os.path.join(res, "metrics.json")) and not a.rerun:
            log(f"{letter}: results already there (use --rerun to redo)")
            continue
        log(f"=== {cfg['name']}")
        ds, tinfo = training_data(cfg, pools, a.cache, a.seed)
        log(f"training frames: {tinfo}")
        with w.open(os.path.join(res, "training_log.txt")) as logf:
            logf.write(json.dumps(tinfo, indent=1) + "\n")
            model = train(ds, a.minutes, a.seed, logf)
        torch.save(model.state_dict(), w.check(os.path.join(res, "model.pth")))

        sets = [dict(cfg["validation"], key="validation")] + cfg["tests"]
        metrics = {"name": cfg["name"], "components": cfg["components"], "training": tinfo, "sets": {}}
        sweeps = {}
        for s in sets:
            log(f"scoring {s['name']}")
            dets, files, fpm = detect(model, s, pools, a.cache, w,
                                      os.path.join(res, "detections", f"{s['key']}_detections.txt"), a.seed)
            truth = truth_boxes(pools, files)
            rows = elp_sweep(truth, dets, files)
            sweeps[s["key"]] = (rows, truth, dets, files)
            if rows:
                write_csv(w, os.path.join(res, f"elp_method_{s['key']}_all_thresholds.csv"), rows)
            metrics["sets"][s["key"]] = {"name": s["name"], "fruitpunch_method": fpm,
                                         "days": len(files), "hours": sum(h for *_, h in files)}

        val_rows = sweeps["validation"][0]
        # detections only exist where the mean rumble probability is over 0.5, so lower
        # thresholds change nothing; among equal F1s keep the lowest threshold
        cands = [r for r in val_rows or [] if r["threshold"] >= 0.5]
        best = max(cands, key=lambda r: (r["f1"], -r["threshold"])) if cands else None
        thr = best["threshold"] if best else 0.5
        metrics["threshold_from_validation"] = thr
        for key, (rows, truth, dets, files) in sweeps.items():
            m = metrics["sets"][key]
            m["elp_at_validation_threshold"] = at(rows, thr)
            m["elp_at_0.5"] = at(rows, 0.5)
            if rows and key != "vehicle":
                m["elp_best_f1_on_this_set_for_reference"] = max(rows, key=lambda r: r["f1"])
            if key == "sampleB":
                m[f"elp_without_{SAMPLE_B_ODD_DAY}"] = at(
                    elp_sweep(truth, dets, files, lambda k: day_of(k) != SAMPLE_B_ODD_DAY), thr)
            if key in ("sampleA", "sampleB"):
                m["elp_by_habitat"] = {h: at(elp_sweep(truth, dets, files, lambda k, h=h: HABITAT.get(k[:5]) == h), thr)
                                       for h in sorted(set(HABITAT.values()))}
        w.json(os.path.join(res, "metrics.json"), metrics)
        write_summary(w, os.path.join(res, "summary.md"), cfg, metrics)
        log(f"{letter}: results in {res}")


def _p(v, digits=3):
    return "-" if v is None or v != v else f"{v:.{digits}f}"


def write_summary(w, path, cfg, m):
    thr = m["threshold_from_validation"]
    t = [f"# Results: {cfg['name']}", "",
         f"Trained on: {' + '.join(cfg['components'])}. "
         f"{m['training']['rumble_frames']} rumble frames, as many negative frames.", "",
         f"Score threshold picked on validation ({cfg['validation']['name']}): **{thr:.2f}** "
         "(the one with the best F1 under ELP scoring).", "",
         "## ELP method (whole days, detector validation rules)", "",
         "A detection counts when it overlaps a truth box by more than 1% in time and in frequency. "
         "Precision = detections that hit a rumble / all detections; recall = rumbles found / all rumbles; "
         "FP/h = false detections per hour of audio.", "",
         "| Set | Days | Hours | Rumbles | Detections | Precision | Recall | F1 | FP/h |",
         "|---|---|---|---|---|---|---|---|---|"]
    for key, s in m["sets"].items():
        r = s["elp_at_validation_threshold"]
        if not r:
            continue
        t.append(f"| {s['name']} | {s['days']} | {s['hours']:.1f} | {r['rumbles']} | {r['detections']} | "
                 f"{_p(r['precision'])} | {_p(r['recall'])} | {_p(r['f1'])} | {_p(r['fp_per_hour'], 2)} |")
    t += ["", "Same sets at threshold 0.50 (FruitPunch's default):", "",
          "| Set | Precision | Recall | F1 | FP/h |", "|---|---|---|---|---|"]
    for key, s in m["sets"].items():
        r = s["elp_at_0.5"]
        if r:
            t.append(f"| {s['name']} | {_p(r['precision'])} | {_p(r['recall'])} | {_p(r['f1'])} | {_p(r['fp_per_hour'], 2)} |")
    t += ["", "Best F1 if the threshold were tuned on each set itself (for reference only; optimistic):", "",
          "| Set | Threshold | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for key, s in m["sets"].items():
        r = s.get("elp_best_f1_on_this_set_for_reference")
        if r:
            t.append(f"| {s['name']} | {r['threshold']:.2f} | {_p(r['precision'])} | {_p(r['recall'])} | {_p(r['f1'])} |")
    for key, s in m["sets"].items():
        if f"elp_without_{SAMPLE_B_ODD_DAY}" in s:
            r = s[f"elp_without_{SAMPLE_B_ODD_DAY}"]
            if r:
                t += ["", f"Sample B without {SAMPLE_B_ODD_DAY}: precision {_p(r['precision'])}, "
                          f"recall {_p(r['recall'])}, F1 {_p(r['f1'])}, FP/h {_p(r['fp_per_hour'], 2)} "
                          f"({r['rumbles']} rumbles)."]
        if "elp_by_habitat" in s:
            t += ["", f"{s['name']} by habitat (threshold {thr:.2f}):", "",
                  "| Habitat | Days | Rumbles | Precision | Recall | F1 | FP/h |", "|---|---|---|---|---|---|---|"]
            for h, r in s["elp_by_habitat"].items():
                if r:
                    t.append(f"| {h} | {r['days']} | {r['rumbles']} | {_p(r['precision'])} | {_p(r['recall'])} | "
                             f"{_p(r['f1'])} | {_p(r['fp_per_hour'], 2)} |")
    t += ["", "## FruitPunch method (balanced frames)", "",
          "Every rumble frame plus as many frames from the gaps between rumbles, each classified on its own. "
          "This is how FruitPunch reported results; it ignores whole-day false alarms, so it looks better "
          "than the ELP method.", "",
          "| Set | Rumble frames | Precision | Recall | F1 | Accuracy | ROC AUC |", "|---|---|---|---|---|---|---|"]
    for key, s in m["sets"].items():
        f = s["fruitpunch_method"]
        if f:
            t.append(f"| {s['name']} | {f['support']} | {_p(f['precision'])} | {_p(f['recall'])} | "
                     f"{_p(f['f1'])} | {_p(f['accuracy'])} | {_p(f['roc_auc'])} |")
    tr = m["training"]
    t += ["", "## Training", "",
          f"- Days used per dataset: {tr['days_used']}",
          f"- Days skipped (fewer than 2 rumbles or 2 gaps, FruitPunch's rule): {tr['days_skipped_fruitpunch_rule']}",
          f"- Negatives: {tr['negatives_used_from_gaps']} from gaps between rumbles, "
          f"{tr['negatives_used_from_noise_and_random']} from noise clips and random forest frames",
          "", "Files here: `model.pth`, `training_log.txt`, `detections\\*.txt` (Raven tables with a Score "
          "column), `elp_method_*_all_thresholds.csv` (every threshold), `metrics.json`."]
    with w.open(path) as f:
        f.write("\n".join(t) + "\n")


################################################################################
# compare
################################################################################

def cmd_compare(a, w):
    rows = []
    for d in sorted(os.listdir(a.out)) if os.path.isdir(a.out) else []:
        mp = os.path.join(a.out, d, "results", "metrics.json")
        if os.path.exists(mp):
            rows.append(json.load(open(mp)))
    if not rows:
        log("no results yet")
        return
    head = ["Configuration", "Threshold", "Val F1",
            "Test 1 (B) P", "R", "F1", "FP/h", "Test 2 (A) P", "R", "F1", "FP/h",
            "Vehicle FP/h", "FruitPunch-method F1 Test 1", "AUC Test 1"]
    t = ["# All configurations", "",
         "ELP method at the threshold each configuration picked on its own validation set. "
         "Pick the winner on Test 1 (Sample B); Test 2 (Sample A) is missing where Sample A was trained on.", "",
         "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    csv_rows = []
    for m in rows:
        s = m["sets"]

        def g(key, field, digits=3):
            r = (s.get(key) or {}).get("elp_at_validation_threshold")
            return _p(r[field], digits) if r else "-"
        fp1 = (s.get("sampleB") or {}).get("fruitpunch_method") or {}
        cells = [m["name"], f"{m['threshold_from_validation']:.2f}", g("validation", "f1"),
                 g("sampleB", "precision"), g("sampleB", "recall"), g("sampleB", "f1"), g("sampleB", "fp_per_hour", 2),
                 g("sampleA", "precision"), g("sampleA", "recall"), g("sampleA", "f1"), g("sampleA", "fp_per_hour", 2),
                 g("vehicle", "fp_per_hour", 2), _p(fp1.get("f1")), _p(fp1.get("roc_auc"))]
        t.append("| " + " | ".join(cells) + " |")
        csv_rows.append(dict(zip(["configuration", "threshold", "val_f1", "b_precision", "b_recall", "b_f1",
                                  "b_fp_per_hour", "a_precision", "a_recall", "a_f1", "a_fp_per_hour",
                                  "vehicle_fp_per_hour", "b_fruitpunch_f1", "b_auc"], cells)))
    with w.open(os.path.join(a.out, "comparison.md")) as f:
        f.write("\n".join(t) + "\n")
    write_csv(w, os.path.join(a.out, "comparison.csv"), csv_rows)
    log("wrote", os.path.join(a.out, "comparison.md"))
    print("\n".join(t))


################################################################################

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["plan", "embed", "run", "compare", "all"])
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help=f"your copy of the Datasets folder (read only; default {DEFAULT_SOURCE})")
    p.add_argument("--out", help="where the configuration folders go (default <source>\\Configurations)")
    p.add_argument("--cache", help="embedding cache, shared by all configurations "
                                   "(default <out>\\_embedding_cache; needs ~60 GB)")
    p.add_argument("--configs", nargs="+", default=list("ABCD"), type=str.upper,
                   help="which configurations (default A B C D)")
    p.add_argument("--best", type=str.upper, help="winner of A-D (for E), or of A-E (for F, G, H)")
    p.add_argument("--copy-audio", action="store_true",
                   help="also copy the .wav files into each configuration folder (hundreds of GB each)")
    p.add_argument("--minutes", type=float, default=2, help="training time per configuration (FruitPunch: 2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rerun", action="store_true", help="redo configurations that already have results")
    p.add_argument("--fake-embedder", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.out = a.out or os.path.join(a.source, "Configurations")
    a.cache = a.cache or os.path.join(a.out, "_embedding_cache")
    bad = [c for c in a.configs if c not in ALL_LETTERS]
    if bad:
        sys.exit(f"unknown configurations {bad}; use letters A-H")
    root = find_dataset_root(a.source)
    # never write into the original data, or into the dataset folders of the copy
    w = Writer([a.out, a.cache], [ORIGINAL_DATA, os.path.join(root, "Training"), os.path.join(root, "Testing")])
    log(f"reading datasets from {root}; writing to {a.out}")
    pools = load_pools(root) if a.step != "compare" else None
    if a.step in ("plan", "all"):
        cmd_plan(a, pools, w)
    if a.step in ("embed", "all"):
        cmd_embed(a, pools, w)
    if a.step in ("run", "all"):
        cmd_run(a, pools, w)
    if a.step in ("compare", "all"):
        cmd_compare(a, w)


if __name__ == "__main__":
    main()
