# Retraining the FruitPunch rumble classifier on ELP data: plan, replicability check, time estimates

Based on: your Word doc *Elephant Rumble Detector Improvement* (Bobbi Estabrook & Amogh Shetty, Oct 2026), and a line-by-line read of `ramayer/elephant-rumble-inference` (our fork, commit `f6dde7e`), including the saved outputs inside `notebooks/training_notebook.ipynb`.

---

## 1. The short version

- **Can training be replicated? Yes, but not by running the notebook as-is.** The method is simple and fully visible in the code. The notebook, though, is wired to the FruitPunch Google Drive layout, contains leftover debug cells that point at one specific file, and has about 10 assumptions that will break or quietly mislead on ELP's data (section 4). These are all fixable; I'd turn the notebook into one clean, configurable training script.
- **Replicating it exactly isn't the real goal.** Your doc's Table 1 shows this repo's model scoring **precision 0.245, recall 0.932** on *Nouabale-Ndoki_Grid Additional*. The README's "high 90s" scores were measured a different, much easier way (section 3). Your target is **precision > 0.7 and recall > 0.7**, so retraining should change a few things on purpose, not just copy the author.
- **Time on an RTX 4060:** a few hours of one-time preparation, then **~2 minutes per training run**, and **roughly 15–30 s per 24-hour file** for inference. Details in section 6.

---

## 2. What I need from you to start (the Word doc doesn't list the data folders)

Your doc points to a second document for the data details:
`L:\ELP\Detectors\Elephant\Datasets\Rumble_Training_&_Testing_Data_Info.docx`

Please send:

1. **That info doc** (attach it here).
2. **The folder layout** of the training and testing data. In PowerShell, run this from the dataset folder and paste the output (it lists folders only, not every file):
   ```
   Get-ChildItem -Recurse -Directory | ForEach-Object { "{0}  ({1} files)" -f $_.FullName, (Get-ChildItem $_.FullName -File).Count }
   ```
3. **One example Raven selection table** for each type you have (training clips, testing days). Just the first 10 lines is enough. I need to see the exact column names and how rumbles vs. non-target signals are tagged.
4. **Two facts:** are the training rumbles **short clips** (a few seconds each) or **long files with selection tables**? And are the testing files 24-hour, 1-hour, or mixed?

---

## 3. Why the README's ~97% and your doc's 0.245 precision are both "true"

| | How the author tested (README) | How ELP tests (your Table 1) |
|---|---|---|
| Unit scored | each 0.64 s slice of audio | each detected **rumble event** |
| Test mix | **50% rumble / 50% non-rumble slices** (balanced on purpose, `make_dataset(..., balance_classes=True)`) | real recordings: rumbles are rare, and **140+ full days have no elephants at all** |
| Negatives | only quiet gaps *near* labelled rumbles, in files that *contain* rumbles | everything, including elephant-free days |

From the notebook's saved output: the training set had **184,151 rumble slices and 1,808,388 non-rumble slices** (about 1:10). The author threw away 90% of the non-rumble slices to balance it 50/50. The model therefore saw relatively few examples of "ordinary forest noise", and **never** saw days with no elephants. On real data it flags many non-elephant sounds as rumbles, so recall is high and precision is low. That's exactly the pattern in your Table 1.

**So the improvements that matter most for precision are:**
1. **Train on many more, and harder, negatives:** ELP's "non-target signals similar to rumbles", plus slices from **elephant-free days**, and keep far more of the ordinary background.
2. **Choose the decision threshold on purpose.** Today it's "elephant score > not-elephant score". Instead we can produce a 0–1 score and pick the cut-off that meets 0.7/0.7 on a validation set. That also gives Raven a **detection score**, which your doc lists as missing from the Stanford detector.
3. **Measure the way ELP measures:** event-level precision/recall against Raven tables, the same as Table 1. Then every change is judged on the number you care about.

The official FruitPunch pipeline on GitLab (the Grumble GUI) scored **0.75 / 0.91** on the same test set, already above your precision target. It's a different codebase from this repo. If we can get access to it, it's worth reading before we invest heavily here.

---

## 4. Replicability audit: everything that blocks or skews retraining

### Will stop it from running on ELP data
| # | Where | Problem | Fix |
|---|---|---|---|
| 1 | notebook cell 1 | Data comes from a Google Drive symlink: `AI for Forest Elephants 2 Shared/01. Data/cornell_data` → `./data/Rumble` | Point a config setting at the L: drive / local folders |
| 2 | `raven_file_helper.py` `find_candidate_audio_files` | Only looks for **`*.wav`** | Also accept `.flac` and `.aiff` (your doc requires all three) |
| 3 | `raven_file_helper.py` `get_files_from_train_folder` / `_test_folder` | The train/test split is decided by whether the **path contains the text "Train" or "Test"** | Use ELP's explicit training and testing lists |
| 4 | `raven_file_helper.py` (SQL in `all_raven_files_as_one_table`) | Requires the columns `Tag 1`, `Tag 2`, `Tags`, `Notes`, `Score`, `File Offset (s)`, `Begin File` to exist. If no table has one of them, the query errors | Make missing columns optional |
| 5 | same | **Every selection in every table counts as a rumble.** If a table also marks non-target signals (gunshots, thunder, other animals), they'd be learned as elephants | Filter by tag; send non-targets to the negative class |
| 6 | notebook cells 6, 8, 9, 11, 12 and 29 | `DEBUG = True`, and the debug cells load the specific file `CEB1_20111010_000000.wav`. Cells 12 and 29 use `audio_path`, `Image` and `y`, which only exist if the debug cell ran | Remove from the training script |
| 7 | notebook cell 33 | Compares against `2024-06-30.pth` etc., downloaded from **0ape.com (offline)** | Use our local `2024-07-03.pth` as the baseline |
| 8 | notebook imports | Needs `torchlens`, `umap-learn`, `seaborn`, `tensorboard`, `scikit-learn`, `pandas`, `soundfile`, none of which are in `pyproject.toml`. Also, **torchaudio ≥ 2.9 removed `StreamReader`**, which the code depends on | Pin a working environment (Python 3.11, torch/torchaudio 2.4.x, numpy < 2) |

### Will run, but skew the results
| # | Where | Problem | Fix |
|---|---|---|---|
| 9 | notebook cell 23 | `val_loader = testing_loader`: the test set is used as the validation set during training. The model was also picked by comparing several versions on the test set (cell 33) | Hold out a validation set **from the training data**; touch the test set once, at the end |
| 10 | `raven_file_helper.py` `get_negative_labels` | Negatives are only taken from gaps of at least 20 s between labelled rumbles, in files that have at least 2 labels. Any rumble the annotators missed becomes a "not elephant" example, and elephant-free days are never used | Add ELP non-target clips and elephant-free days |
| 11 | notebook cell 14 | Throws away ~90% of negatives to balance 50/50 | Keep many more negatives; use class weights instead |
| 12 | notebook cells 14, 25 | Only `random_split` is seeded. The balancing `randperm` and the network's starting weights are not, so two runs give slightly different models | Seed everything |
| 13 | notebook cell 9 | Each label is embedded with **500 s before and 200 s after** it as context. Slow (~700 s of audio per label), and overlapping labels are embedded again and again | Embed each recording once, then cut out label slices |

### Things already fine
- AVES weights still download (Google storage, checked).
- The trained-classifier format is plain PyTorch. We have `2024-07-03.pth`, verified intact.
- The classifier itself and the training loop are standard and will train fine.

### Inference gaps vs. your doc's requirements (after retraining)
These don't block training but matter for ELP use:
- **Begin Time / End Time** are written as seconds *into each file*, but your doc needs seconds *from the first file of the table*.
- **Begin Path** is just the file name, not the full path.
- **No real score.** It's hard-coded to 0.9 (`main.py`).
- There's no subfolder scanning and no `.flac`/`.aiff` support, and files are capped at 24 h by default.
- Very short files: anything under ~1.3 s returns nothing. 20 s clips should work, but that needs testing.

Suhana's improved command line (zip in `FruitPunchAI\program_files\Suhana`) already handles different file lengths. Worth merging in rather than redoing.

---

## 5. Proposed plan

1. **Baseline.** Run the existing `2024-07-03.pth` on the test set with an ELP-style event-level scorer, and confirm we reproduce ~0.245 / 0.93. That proves our measuring stick matches Table 1.
2. **Data adapter.** Point the code at ELP's folders and tables (needs section 2).
3. **One-time embedding cache.** Run AVES once over all training/testing audio and save the results.
4. **Training script.** A clean `train.py` replacing the notebook: seeded, with a proper validation split, hard negatives, class weights, threshold tuning, and saving the model plus its metrics.
5. **Evaluate** on Nouabale-Ndoki_Grid and Grid Additional, and compare to Table 1.
6. **Iterate** on negatives, threshold and minimum event length until **P > 0.7 and R > 0.7**.
7. **Then** the inference fixes from the list above, and speed.

---

## 6. Time estimates on an RTX 4060

Anchors: the README reports **22 s per 24-hour file on an RTX 2060** and **~28 min per 24-hour file on CPU**. A desktop RTX 4060 is roughly 2× a 2060 in raw compute, with 8 GB of memory. These are estimates until we time one real file. A *laptop* 4060 is slower than the desktop card.

| Step | Estimate on RTX 4060 | Notes |
|---|---|---|
| Resample/cache each file at 500 Hz | ~5–20 s per 24 h file (CPU) | one-time |
| **AVES embeddings for training data** | **~15–25 s per 24 h of audio** → for ~190 day-files (the author's training set size) about **1–1.5 h** | one-time; if ELP's training set is short clips, far less |
| **Training the classifier** | **~1–2 min per run** | the notebook's 11 epochs took 2 min *on CPU* |
| **Testing** (inference over the test set) | ~15–30 s per 24 h file → for ~200 test days about **1–2 h** | once per model; re-scoring at new thresholds then takes seconds |
| Same test set on CPU only | ~28 min per 24 h file → **~4 days** for 200 days | why the GPU matters |

The slow part is the one-time AVES pass. After that, each experiment (new negatives, new threshold) is minutes, so we can try many.

"Two-core GPU": if you meant **two GPUs**, inference and embedding time can roughly halve by giving each GPU half the files.
