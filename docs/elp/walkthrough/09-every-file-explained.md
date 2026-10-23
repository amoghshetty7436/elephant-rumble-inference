# Every file in the retraining run, explained simply

Think of the run as a small factory with 3 machines:

```
 .wav sound + .txt answer key ──► PREPARE ──► fingerprints (cache) ──► TRAIN ──► retrained.pth
                                                                                     │
                       whole test days (.wav) ───────────────────────────────► DETECT ──► Raven tables + scores
```

Below is every file involved, in the order the data flows through them.

---

## A. The files you run or feed in

### `tools/fp_replica.py` (the main script, ~460 lines)
This is the only file you run. It has three "buttons", chosen by the first word after the script name:

| You type | It does |
|---|---|
| `prepare` | Turns sound into fingerprints and saves them |
| `train` | Teaches the small model and saves `retrained.pth` |
| `detect` | Uses a model on full days and checks it against the answer key |

Its pieces, top to bottom:

| Piece | What it does, simply |
|---|---|
| The top (`AVES_SR = 500`, `PREROLL, POSTROLL = 500, 200`) | The fixed settings copied from FruitPunch: slow the sound down to 500 samples per second, and give AVES 500 s of sound before each window and 200 s after it as context. |
| `log` | Prints a message with the time in front, e.g. `13:25:09 done`. |
| `Progress`, `fmt_time` | The `[#####.....] 40% … about 1h20m left` progress bar. |
| `load_audio` | Opens one `.wav` and reads it an hour at a time, so a 1.3 GB day never has to fit in memory at once. It keeps one channel and resamples from 8,000 (or 4,000) samples per second to 500. Then it rounds the numbers to half precision, exactly like FruitPunch's cached audio. |
| `Embedder` | Loads the AVES model onto the GPU. Give it a piece of sound and it returns one fingerprint (768 numbers) per 0.64 s, each scaled to length 1. `whole_file` does a full day hour by hour, with ~10 s of overlap at each edge so nothing is cut in half. This is the same way FruitPunch's inference program does it. |
| `window_embeddings_fp` | FruitPunch's way of fingerprinting one labelled window: play AVES the window plus up to 500 s before it and 200 s after it, then keep only the fingerprints that fall inside the window. |
| `window_embeddings_from_day` | The faster alternative (`--mode day`): fingerprint the whole day once, then cut out the window. |
| `cmd_prepare` | **The PREPARE button.** (1) Reads all the label tables. (2) Finds the `.wav` files in the train and test folders. (3) Keeps only usable days: those with ≥2 rumbles, on FruitPunch's list (`--only-files`), and not already done. (4) For each day, loads the audio, fingerprints every rumble and every "not rumble" window, and saves them as one small file, `fp_cache\fp\train\<day>.pt` or `…\test\<day>.pt`. (5) Writes `manifest.json`, which lists which days were used or skipped and why. |
| `load_split` | Loads all the saved day files for train (or test) and stacks them. It then **balances** them: it randomly keeps only as many "not rumble" fingerprints as rumble ones. |
| `balanced_class_weights` | Gives both classes equal importance (same as sklearn's `compute_class_weight('balanced')` that FruitPunch used). |
| `cmd_train` | **The TRAIN button.** It splits the training fingerprints 95% / 5% (seed 42, like FruitPunch). Then it trains the small model with Adam in batches of 64 for up to 2 minutes, printing one line per pass ("epoch"). It stops early if the check loss gets much worse. It saves `retrained.pth`, then prints the test report. With `--compare`, it prints the same report for FruitPunch's `2024-07-03.pth`. |
| `predict`, `report`, `roc_auc` | Run a model on the test fingerprints and print precision, recall, F1, accuracy and ROC AUC (how well the scores rank rumbles above non-rumbles: 1.0 is perfect, 0.5 is guessing). |
| `segments` | Finds stretches where the model says "rumble" for at least 4 fingerprints in a row (~2.5 s). That's FruitPunch's rule for what counts as one detection. |
| `cmd_detect` | **The DETECT button.** For every `.wav` in the test folder (all 61 days, even ones without elephants), it fingerprints the whole day and asks the model "rumble or not?" for every 0.64 s. It turns the "rumble" stretches into detections with a **Score** (average confidence, 0 to 1). It writes one Raven table per day plus one combined table, then grades them with `elp_score.py`. |
| `main` | Reads what you typed (`prepare --tables … --cache …`) and presses the right button. |

### `tools/fp_data.py` (the label rules, ~130 lines)
This file holds the "answer key" logic. It has no AI in it, which is why I could test it against FruitPunch's printed numbers.

| Piece | What it does, simply |
|---|---|
| `Label` | One labelled box: start time in the day, length, low and high pitch, and which day. |
| `normalize_file_name` | Makes names comparable: removes the folder, lower-cases, and turns `dzan_` into `dz_`. FruitPunch did the same. |
| `read_labels` | Reads every row of every table and groups them by day. Start = `File Offset (s)`, length = `End Time − Begin Time`. Rows with no End Time are skipped (FruitPunch did this too). Labels for the same day from different tables are pooled. |
| `negative_windows` | Picks the "not rumble" windows from the quiet gaps between rumbles, following FruitPunch's exact rules (first minute; skip gaps < 20 s; then 1, 2 or 3 windows depending on gap length). On your real tables it gives the same counts as their notebook for 187 of 189 days. |
| `find_sound_files` | Lists every `.wav` under the folders you give it, including subfolders. |
| `usable_files` | Says which days to use (≥2 rumbles and ≥2 negative windows) and why others are skipped. |
| `time_to_index` | Converts seconds to fingerprint number: × 500 ÷ 320, so 0.64 s per fingerprint. |

### `tools/fruitpunch_files_used.txt`
A plain list of the 165 training days and 24 test days FruitPunch actually used, taken from their notebook's printout. `--only-files` uses it so we train on exactly the same days.

### `tools/elp_score.py` (ELP-style grading, ~190 lines)
This is a Python version of ELP's MATLAB "Detector Validation" tool.

| Piece | What it does, simply |
|---|---|
| `read_raven_table` | Reads any Raven table into boxes: day, start, end, low/high pitch, Score. It uses `File Offset` for times, because Raven's `Begin Time` counts from the first file in the table, not the start of each day. |
| `overlap_fraction`, `matches` | A detection is a hit if it overlaps a true rumble by more than 1% in time **and** in pitch. |
| `score` | For each threshold 0, 0.05, …, 1: keep detections with Score ≥ threshold, then count hits, false alarms, and rumbles found or missed. From those it works out precision, recall and false alarms per hour. |
| `main` | Lets you run it on its own: `python tools\elp_score.py --truth … --detections … --files …`. |

### `tools/requirements-train.txt`
The 3 extra libraries to install: numpy (number crunching), scipy (resampling the audio), and soundfile (reading `.wav`).

---

## B. FruitPunch's own files that we reuse unchanged

### `elephant_rumble_inference/aves_torchaudio_wrapper.py` (47 lines)
This loads **AVES**, the big pretrained "listening" model (a HuBERT model trained on animal sounds). On first use it downloads two files (~380 MB) from Google Cloud into `C:\Users\avs97\.cache\torch\hub\fruitpunch_elephants\`. `forward` returns the last layer's output: the 768-number fingerprints. AVES is never changed by training.

### `elephant_rumble_inference/elephant_rumble_classifier.py` (66 lines)
The **small model** we actually train: 768 numbers → 192 → 2 scores ("not", "rumble"), with 20% dropout so it doesn't memorise. Its layout is identical to FruitPunch's, which is why our `retrained.pth` and their `2024-07-03.pth` are interchangeable.

The other files in that folder (`main.py`, `audio_file_processor.py`, …) are FruitPunch's original inference program. Our script doesn't use them, because they rely on a torchaudio feature that newer versions removed.

---

## C. The tests (only for checking my code)

| File | Checks |
|---|---|
| `tests/test_fp_data.py` | That label reading, the negative-window rules and the day selection behave exactly like FruitPunch's. |
| `tests/test_elp_score.py` | That the ELP grading counts hits, false alarms, thresholds and elephant-free days correctly. |

You don't need to run these. If you'd like to: `python -m pip install pytest`, then `python -m pytest tests`.

---

## D. What the run creates (in `C:\Users\avs97\fp_cache`)

| File | Made by | What it is |
|---|---|---|
| `fp\train\*.pt`, `fp\test\*.pt` | prepare | One small file per day with that day's rumble and not-rumble fingerprints (~3 GB total). This is why a restart skips finished days. |
| `fp\manifest.json` | prepare | Which days were used, which were skipped, and why. |
| `1_prepare_log.txt` | prepare | Everything prepare printed, including the time per day. |
| `retrained.pth` | train | **The new model** (~0.6 MB). |
| `2_train_results.txt` | train | The FruitPunch-style scores, for our model and for theirs. |
| `detections\<day>.rumbles.txt` | detect | One Raven table per test day. Open it in Raven with that day's `.wav` to see each detection. |
| `detections\all_detections.txt` | detect | All detections in one Raven table. |
| `detections\files.txt` | detect | The list of days that were checked (needed so elephant-free days count). |
| `detections\elp_score.csv` | detect | Precision / recall / false alarms per hour at every threshold. Opens in Excel. |
| `3_detect_results.txt` | detect | The same table as the CSV, plus the run log. |
