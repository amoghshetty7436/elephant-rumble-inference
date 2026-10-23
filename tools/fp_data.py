"""
Label handling for the FruitPunch replication, without torch.

This mirrors what notebooks/training_notebook.ipynb + RavenFileHelper do:

* every row of every Raven table is a rumble ("positive") label, keyed by the
  "Begin File" name (dzan_ -> dz_), with
      start    = File Offset (s)
      duration = End Time (s) - Begin Time (s)
  and rows with no End Time are dropped;
* labels for one sound file are pooled from all tables and sorted by start;
* a file is used only if it has at least 2 labels and at least 2 negative
  windows (get_embeddings_for_labels_in_file);
* negative windows come from the gaps between labels (get_negative_labels).

It is kept free of torch/duckdb so it can be unit-tested anywhere.
"""
import csv
import glob
import os
import re
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Label:
    start: float
    duration: float
    low_freq: float
    high_freq: float
    file: str

    @property
    def end(self):
        return self.start + self.duration


def normalize_file_name(name):
    """'L:\\...\\dzan_20120201_000000.wav' -> 'dz_20120201_000000.wav' (lower case)."""
    name = os.path.basename(name.replace("\\", "/")).strip().strip('"').lower()
    return re.sub(r"^dzan", "dz", name)


def _float(v):
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def read_labels(table_paths):
    """Return {sound_file_name: [Label, ...]} pooled over all tables, sorted by start."""
    by_file = defaultdict(list)
    for path in table_paths:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for raw in csv.DictReader(f, delimiter="\t"):
                row = {(k or "").strip(): v for k, v in raw.items()}
                name = row.get("Begin File")
                begin, end = _float(row.get("Begin Time (s)")), _float(row.get("End Time (s)"))
                offset = _float(row.get("File Offset (s)"))
                if not name or begin is None or end is None or offset is None:
                    continue  # same as the notebook's "WHERE end_time is not null"
                lf, hf = _float(row.get("Low Freq (Hz)")), _float(row.get("High Freq (Hz)"))
                file = normalize_file_name(name)
                by_file[file].append(Label(offset, end - begin, lf or 0.0, hf or 0.0, file))
    for labels in by_file.values():
        labels.sort(key=lambda l: l.start)
    return dict(by_file)


def negative_windows(labels):
    """Exact port of RavenFileHelper.get_negative_labels (returns Labels)."""
    pl = sorted(labels, key=lambda l: l.start)
    if not pl:
        return []
    first = pl[0]
    d = min(first.start - 5, 60)
    neg = [Label(0, d, 25, 100, first.file)]
    for l1, l2 in zip(pl, pl[1:]):
        gap = l2.start - l1.end
        lf, hf = (l1.low_freq + l2.low_freq) / 2, (l1.high_freq + l2.high_freq) / 2
        if gap < 20:  # close rumbles often have unlabelled rumbles between them
            continue
        if gap < 130:
            s, e = l1.end + 5, l2.start - 5
            neg.append(Label(s, e - s, lf, hf, l1.file))
            continue
        if gap > 130:
            neg.append(Label(l1.end + 5, 60, lf, hf, l1.file))
            neg.append(Label(l2.start - 65, 60, lf, hf, l1.file))
        if gap > 130 * 2:
            neg.append(Label((l1.end + l2.start) / 2 - 60, 120, lf, hf, l1.file))
    return neg


def find_sound_files(roots, extensions=(".wav",)):
    """{normalized name: full path} for every sound file under the given folders."""
    found = {}
    for root in roots:
        for ext in extensions:
            for p in glob.glob(os.path.join(root, "**", "*" + ext), recursive=True):
                found[normalize_file_name(p)] = p
    return found


def usable_files(labels_by_file, sound_files):
    """Files the notebook would actually use: has a sound file, >=2 labels, >=2 negatives.

    Returns (used, skipped) where skipped maps file -> reason.
    """
    used, skipped = [], {}
    for name in sorted(sound_files):
        labels = labels_by_file.get(name)
        if not labels:
            continue
        if len(labels) < 2:
            skipped[name] = "fewer than 2 labels"
        elif len(negative_windows(labels)) < 2:
            skipped[name] = "fewer than 2 negative windows"
        else:
            used.append(name)
    for name in labels_by_file:
        if name not in sound_files:
            skipped[name] = "no sound file found"
    return used, skipped


def time_to_index(t, sr=500, samples_per_embedding=320):
    """Same as AudioFileProcessor.time_to_score_index, as an int."""
    return int(t * sr // samples_per_embedding)
