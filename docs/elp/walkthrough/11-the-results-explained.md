# The retraining results, explained simply

These numbers come from the files you uploaded: `1_prepare_log.txt`, `2_train_results.txt`, `3_detect_results.txt` and `retrained.pth`. The run was on 9 October 2026 on the lab PC's RTX 4060.

**The short version:**
1. We rebuilt FruitPunch's model, and it scores the same as theirs on their own test.
2. On whole days it finds almost every rumble, but at the default setting about 3 of every 4 detections are false alarms.
3. Raising the score threshold to 0.90 brings precision and recall both to about 0.70.
4. I found one more problem in the test set while checking the files (section 4).

---

## 1. `1_prepare_log.txt`: turning sound into fingerprints

| Line in the log | What it means |
|---|---|
| `35309 labels for 219 sound files in 4 tables` | All four tables together hold 35,309 rumble boxes on 219 different recordings. |
| `using cuda NVIDIA GeForce RTX 4060` | The GPU was used. |
| `train: leaving out 1 days not in tools\fruitpunch_files_used.txt: ['nn02c_20180625…']` | One training day that FruitPunch didn't have was left out on purpose, so we match them. |
| `train: 192 sound files, 165 usable, 23 skipped` | 23 training days had fewer than 2 rumbles, so FruitPunch's rules skip them. 165 days are left, exactly FruitPunch's number. |
| `test: 61 sound files, 25 usable, 5 skipped` | 25 test files have rumbles: FruitPunch's 24 test days, plus `nn04a_20180608`, which is a training day that also sits in the test folder. The training step ignores that extra day. |
| `train 12/165 ceb1_20111017…: 3437 labels -> 16713 rumble frames …` | One line per day: how many rumbles it had, how many 0.64 s fingerprints those became, and how long the day took. |
| `elapsed 1h47m … done` | The whole step took 1 hour 47 minutes for 190 days. |

## 2. `2_train_results.txt`: training and the snippet test

**The data going in:**
- `train: 165 days, 184104 rumble frames, 1807915 negative frames (ratio 9.8)`: there are about 10 "not rumble" fingerprints for every rumble one. FruitPunch's notebook printed 184,151 and 1,808,388, so these are almost identical.
- `test: 24 days, 6391 rumble frames, 95431 negative frames (ratio 14.9)`: the 24 test days, after leaving out the training day.

**The training lines** (`epoch 1` to `epoch 16`):
- An **epoch** is one pass through all the training fingerprints. 16 passes fit into the 2-minute budget.
- **Train loss** is how wrong the model is on what it's learning from. **Val loss** is how wrong it is on the 5% held back to check progress. Lower is better for both.
- Both kept falling: val loss went from 7.30 to 5.38, and the held-back check went from 17,957 to 18,060 correct out of 18,410 (98.1%). Early stopping never triggered, so the model was still improving slightly when time ran out.

**The snippet test.** This is FruitPunch's way of testing: the labelled rumble pieces of the 24 test days plus the same number of "not rumble" pieces, 6,391 each.

| | Accuracy | ROC AUC | Rumble precision | Rumble recall |
|---|---|---|---|---|
| Our retrained model | 0.958 | 0.993 | 0.977 | 0.939 |
| FruitPunch's 2024-07-03.pth | 0.953 | 0.992 | 0.981 | 0.924 |

- **Accuracy**: the share of pieces it got right.
- **ROC AUC**: how well its scores rank rumbles above non-rumbles. 1.0 is perfect and 0.5 is guessing.
- The two models are practically the same, so **the replication worked**. FruitPunch's reported ~97% is a little higher because they also tested on Dzanga and used the test set to decide when to stop training.

**`retrained.pth`**: I loaded it. It has the same layout as FruitPunch's model (768 → 192 → 2, 148,034 numbers), so it can replace `2024-07-03.pth` anywhere.

## 3. `3_detect_results.txt`: the whole-day test

**What the step did:** it ran the model over every `.wav` in the `24-hr` test folder: 61 files, around the clock. Where the model said "rumble" for at least 4 fingerprints in a row (~2.5 s), it wrote one detection with a Score from 0 to 1. Then it checked those detections against the truth tables using ELP's rule: a hit needs more than 1% overlap in time and in frequency.

Each `n/61 … detections (31 s)` line is one file. Full days took ~31 s. The twelve `nn09a` files are 4-hour pieces, so they took ~5 s.

**The threshold table.** Each row keeps only the detections whose Score is at least that threshold.

| Column | Meaning |
|---|---|
| dets | detections kept |
| TP / FP | detections that hit a real rumble / false alarms (TP + FP = dets) |
| found / FN | real rumbles caught / missed (found + FN = 920) |
| prec | TP ÷ dets: the share of detections that are real |
| recall | found ÷ 920: the share of real rumbles found |
| FP/h | false alarms per hour of audio |

The rows that matter:

| Threshold | Detections | Precision | Recall | False alarms |
|---|---|---|---|---|
| 0.5 (default) | 2,222 | 0.28 | 0.93 | about 1 every 45 min |
| 0.80 | 1,195 | 0.47 | 0.87 | about 1 every 2 h |
| 0.85 | 889 | 0.57 | 0.81 | about 1 every 3 h |
| **0.90** | **598** | **0.71** | **0.70** | **about 1 every 7 h** |
| 0.95 | 242 | 0.87 | 0.42 | about 1 every 30 h |

- Thresholds 0 to 0.50 give the same row because every detection already scores above 0.5. That's how a detection is defined.
- **0.90 is the best balance**, and the only row that reaches ELP's goal of both above 0.7. For comparison, Stanford's detector scored P 0.84 / R 0.75 on Sample A.

**Why it's so different from the snippet test:** the snippet test is half rumbles. A real day is about 99.8% "not rumble", so even a small error rate on quiet time turns into many false alarms. The model also never trained on thunder, rain or most of the night.

## 4. What I found when I checked the test set against the truth tables

I matched the detections in each file against the truth tables you uploaded.

| Group of files | Files | Detections (threshold 0.5) | Real rumbles |
|---|---|---|---|
| FruitPunch's 24 test days | 24 | 1,688 | 910 |
| Files with no rumbles in the tables | 31 | 381 | 0 |
| Files with only 1 to 3 rumbles in the tables, not in FruitPunch's 24 | 6 | 153 | 10 |

What this means:
1. **About a third of the false alarms come from the extra 37 files.** At threshold 0.5, the 381 detections on rumble-free files are all false alarms, and so are most of the 153 on the six near-empty files. If nobody reviewed those files, some of those "false alarms" could be real rumbles.
2. **Even on the 24 rumble days alone, precision is low.** There are 1,688 detections and at most ~617 hits, so precision is roughly 0.37 at threshold 0.5. This confirms that the main problem is the model, not the choice of files.
3. **A file-name mismatch, newly found.** The supplemental truth table has 57 rumbles on `nn04a_20200428_221358.wav`, but the folder's file is named `nn04a_20200429_221358.wav` (the table's own Begin Date column says 4/29/2020). Because the names don't match:
   - those 57 rumbles weren't counted (977 rumbles in the tables, 920 scored);
   - the model's 32 detections on that file were all counted as false alarms, even though many are probably real.

   It's very likely the same recording, renamed. FruitPunch's test list skips this day too, probably for the same reason. Fixing the name would make our numbers slightly better, and it adds 57 rumbles to the test.
4. **`nn04a_20180608` is a training day**, so its 64 detections shouldn't count in a test.

## 5. Caveats

- **The 0.90 threshold was chosen by looking at the test results**, which flatters it a little. A fair choice comes from separate validation days.
- **The 61 files aren't exactly ELP's Sample A.** ELP's Schedule File (`L:\ELP\Detectors\Detector_Performance\Rumbles\Schedule_File.xlsx`) lists the days that were actually reviewed. Scoring only those days gives the fair comparison with Stanford.
- **Our scorer copies ELP's MATLAB rules** (1% overlap, 0.05 steps). One detail, what the 1% is measured against, isn't written in ELP's document. It barely matters at 1%.

## 6. What it means for the project

- **Replication: done.** We have our own copy of FruitPunch's model, trained from ELP's data with their exact method, and it performs the same.
- **Honest performance:** on whole forest days, at the right threshold, it's at about P 0.70 / R 0.70. That's in ELP's target range, but below Stanford's precision.
- **Next steps, in order:**
  1. Get the Schedule File and fix the `nn04a_20200428/29` name, then re-score in seconds. No re-run is needed.
  2. Choose the threshold on held-out validation days.
  3. Improve precision: train on more "not rumble" sounds (elephant-free days, thunder, rain), use hard-negative mining, and smooth the scores over time. See [10-how-to-improve-precision-and-recall.md](10-how-to-improve-precision-and-recall.md).
