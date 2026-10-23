# Choosing the training and testing data

**Goal:** settle which data we train on and which we test on, before we try any model improvements. The data choice should be the only thing that changes between runs, so we keep FruitPunch's method fixed (same model, same "not rumble" rules, same 2-minute training) for all of these experiments.

Everything below comes from the tables and folder listing you shared. I counted the tables directly.

---

## 1. What we have

### Training pools

| Pool | Where | Days | Rumbles | Sites | Years | Notes |
|---|---|---|---|---|---|---|
| **A. Clearing** | `rumble_clearing` table + 56 sounds | 56 | **31,129** (~556 per day) | 3 (ceb1 35 days, madj 14, jobo 7) | 2011–2016 | 4 kHz. Loud, busy bais. Not every rumble marked |
| **B. Forest (PNNN training)** | `nn_ele_hb_…TrainingSet_Removed…` / `TrainingSet_v2` + `TrainingSet_sounds` (136) | 133 in the table, 109 used by FruitPunch | **3,203** (~24 per day) | 44 | 2018–2019 | 8 kHz. 237 rows marked marginal/faint, 165 tagged multi-harmonic, **3 `DUMMY_NoEles` rows = 3 elephant-free days**. Not every rumble marked |
| B2. Overlap tables | `overlapping-clean` (53 boxes, 28 days), `RumbleOverlap` (451 boxes, 4 days) | | | | | Days are mostly inside B. Boxes are loose (from a detector), so use with care |
| **C. `nn (only train files)`** | 317 sound files across 39 site folders | 317 | ? | 39 | 2020 | New days, but **no table found**, and some are Sample A test days (section 5) |
| D. `unknown\X_Datasets_To_Use` | 177 hand-browsed day tables | | | | 2017–2019 | **Not new days:** the source tables for B and for Complex/General (section 5) |
| E. Rumble clips | `TrainingClips\elephant_general_rumbles` (2,212) + `elephant_marginal_rumbles` (941) | | | | | **Not new data.** `TrainingSet_v2` has exactly 2,211 normal + 941 marginal rows, so these clips are cut from the same forest days as B |
| **F. Non-elephant clips** | `thunder` 77, `rain` 40, `wind` 46+19, `vehicle` 17+3, `human_voice` 6, `wings` 6, `silence` 63 | | 0 | | | Clean "not rumble" examples. FruitPunch never used them |

### Testing pools

| Pool | Where | Days | Rumbles | Notes |
|---|---|---|---|---|
| **T1. Sample A** (Nouabale-Ndoki_Grid) | `24-hr` (61 files) + `NN_Random_Sample_Hand_ele` + `…nn04aSupplemental…` | 61 (31 with rumbles, 30 without) | ~977 (920 scored + 57 on the misnamed nn04a day) | **Fully hand-browsed**: every rumble is marked, so every unmarked second is a true "not rumble". 2018–2021. What we've been scoring on |
| **T2. Sample B** (Additional2024) | `Additional2024\sounds` (83) + 4 truth tables + 81 raw reviewed tables | 80 reviewed (27 with rumbles, 53 without) | 1,577 | Fully hand-browsed, **recorded in 2023**, so years after every training day. See section 5 |
| T3. Complex / General | `RumbleComplex-Forest` (30), `RumbleGeneral-Forest` (29) | 59 | ? | Truth tables found (section 5). Now planned as **validation** |
| T4. Dzanga | `Testing\Sounds\RumbleOverlap` + `Truth_Tables\RumbleOverlap` | 3–4 | | Clearing. Parked, as you decided |

## 2. Is the clearing data a problem? What the numbers say

Your hunch has real support, but it isn't proven yet:

- **Positives:** in FruitPunch's training run, 165,447 of 184,104 rumble fingerprints (**90%**) came from clearings.
- **Negatives too:** 1,448,458 of 1,807,915 "not rumble" fingerprints (**80%**) came from clearing days. So the model mostly learned what a *bai* sounds like when no rumble is present, not what a quiet forest night with insects and rain sounds like. That fits the result: the false alarms are on forest background.
- **The rumbles themselves differ a little:** clearings have ~556 rumbles a day versus ~24 in the forest (sound is busier and closer); median length is 3.3 s versus 3.7–4.5 s in the forest. Their pitch range is similar (low edge ~21 Hz in both).
- **But** clearing data also gives 10× more rumble examples. It might be teaching the model the rumble shape in a way 3,200 forest rumbles can't. That's exactly why it has to be tested, as you said.

**One trap to avoid:** if "forest only" scores worse, that could be because it has 10× less data, not because the domain matters. So the experiments below include a run where clearing and forest have **the same number of rumbles**, which separates "which kind" from "how much".

## 3. The fixed yardstick (set this up first)

All configurations must be judged on the same days, in the same way, or the comparison means nothing.

1. **Main test: Sample B (T2).** It's fully labelled (80 days; 3 unreviewed `HOLD` files left out), it's from 2023 (so nothing in any training set is close to it in time), and it has elephant-free days. Every configuration gets scored on it, including the ones that move Sample A into training. That's why I want to use it as the common yardstick.
2. **Second test: Sample A (T1)**, for the configurations that don't train on it. This keeps us comparable with Stanford (P 0.84 / R 0.75) and with our baseline (P 0.71 / R 0.70 at 0.90).
3. **Validation: whole forest days that are not used for training** (updated in section 6: the RumbleComplex and RumbleGeneral days). These pick the threshold and the stopping point. Today the threshold 0.90 was picked on the test, which flatters it.
4. **Score:** ELP's rules (1% overlap), reported as precision and recall at the threshold picked on validation, plus the best F1 and false alarms per hour.
5. **Fixes before any run:** rename `nn04a_20200428` ↔ `20200429`, drop `nn04a_20180608` from testing (it's a training day), drop `DUMMY_NoEles` rows from the rumble list.

**Leakage check I ran:** training forest days and Sample A share only `nn04a_20180608` (already excluded). No test day is within a week of a training day at the same site. But 13 of Sample A's 14 sites also appear in training (only `nn03c` is new), so Sample A tests "new days at known sites". Sample B tests "new year", which is closer to how ELP will use the detector.

## 4. The configurations to compare

Each row is one training run. Same method throughout; only the data changes.

| # | Train on | Test on | What it answers |
|---|---|---|---|
| **0** | A + B (FruitPunch's 165 days) | B-val, T1, T2 | Baseline. Re-run with the new validation days so it's fair |
| **1** | B only (forest) | T1, T2 | Does the model do better without clearings at all? |
| **2** | A only (clearing) | T1, T2 | How far does clearing-only transfer to forest? Sets the floor |
| **3** | B + a random slice of A with **as many rumbles as B** (~3,200, about 6 clearing days) | T1, T2 | The fair "forest vs clearing" test: equal sizes. Compare with 1 and 2 |
| **4** | A + B, **balanced 50/50** (each training batch is half clearing, half forest, for both rumbles and "not rumble") | T1, T2 | Keep the clearing data but stop it dominating |
| **5** | Best of 0–4 **+ F (thunder, rain, wind, vehicle, voice, silence clips) + the 3 elephant-free days** as negatives | T1, T2 | Does adding clean forest-type "not rumble" sounds fix the false alarms? |
| **6** | Best of 0–5 **+ Sample A moved into training** (all 61 days: ~977 rumbles and 30 elephant-free days, all fully labelled) | T2 only | Your "move test data into training" idea. Sample A is the most valuable forest data we have, because its "not rumble" parts are guaranteed clean. Costs us the Sample A comparison with Stanford |
| **7** | Best of 0–5 **+ C and D** (extra forest days), only if section 5 shows they have tables | T1, T2 | Does more forest data keep helping? |
| 8 (optional) | Swap of 6: train on best + Sample B, test on Sample A | T1 only | Checks that the gain in 6 isn't specific to one test set |

**How to read the results:**
- If **1 beats 0** and **3** looks like 1, the clearing data is hurting: drop or down-weight it.
- If **0 beats 1** but **3 ≈ 1**, clearings help only through sheer quantity: keep them, balanced (config 4).
- If **2 is close to 0**, the forest data adds little and more forest data (6, 7) is the lever.
- **5** tells us how much of the precision problem is "never heard forest noise" rather than the clearing split.

**Order to run them:** 0, 1, 2, 3 first (one evening; preparation is the slow part and is shared), then 4 and 5, then 6/7/8 on the winner.

**Cheap to do:** every pool's fingerprints only need to be computed once. Configs 0–4 reuse the same cache; each extra run is just the ~2-minute training plus the whole-day test. The whole-day test is the slow bit (~31 s a day), so we should also cache whole-day fingerprints for the test and validation days once.

**Subsets for fast model experiments:** as you said, we'll pick these once the best split is known. I'll keep the day lists in files so cutting a smaller, representative subset later is easy.

## 5. What your file list and the Sample B tables showed (16 Oct)

Copies are in `retraining/dataset_inventory/` (`dataset_files.tsv` and `sampleB/`).

### Sample B is a good common test, with one catch
- **83 sound files, 80 reviewed.** Three are marked `HOLD` in the tracking sheet and were never browsed: `nn05b_20230116`, `nn05b_20230126`, `nn08c_20231007`. They must be left out. One day (`nn02d_20230109`) was reviewed twice.
- **27 days with rumbles, 53 elephant-free days, 1,577 rumbles**, 15 sites, January and October 2023. No day is shared with any training set.
- **The catch:** one day, `nn08c_20230104`, has **1,032 rumbles, 65% of the whole set**. If we only report the pooled numbers, Sample B mostly measures that one day. So we'll report Sample B **with and without that day**, plus the average over days.
- 98 rumbles are marked marginal, faint or "q" (questionable). We'll score with and without them, the way ELP's tables do.

### `X_Datasets_To_Use` is the original hand-browsed tables, not new days
- It holds 177 tables named `…_hele_00-24hr_…`: one hand-browsed table per **whole day**.
- 127 of those days are the forest training days (`TrainingSet_sounds`), and 49 are the RumbleComplex/RumbleGeneral test days.
- So it's where the training table was built from. It matters because if those days were browsed end to end, their unlabelled time is a clean "not rumble", which the training table alone can't tell us.

### 59 more labelled forest days we weren't using
- `RumbleComplex-Forest` (30 days) and `RumbleGeneral-Forest` (29 days) have truth tables: `Testing\Truth_Tables\nn_ele_00-24hr_ComplexTest_inc3NoEleDay.txt` and `nn_ele_00-24hr_GeneralTest_v4.txt`.
- They're 2017–2019 forest days. None is in the forest training table (`nn05d_20180321` sits in the `TrainingSet_sounds` folder but has no rows in the training table, so it's safe). Complex was picked for thunder, insects and vehicles, and includes 3 elephant-free days.
- **This changes the plan for validation:** instead of holding back 15 training days, we can use these 59 days to choose the threshold and the stopping point, and keep all the training days for training.

### `nn (only train files)` is new, but has no labels, and touches Sample A
- **317 whole days from September to December 2020**, 39 sites, 425 GB. None of them is a training day.
- I found **no table for them** in any of the folders listed.
- **4 of them are Sample A test days** (`nn02d_20201013`, `nn06b_20201005`, `nn06b_20201030`, `nn08c_20201005`), and about 17 more are within a week of a Sample A day at the same site. If labels exist and we train on them, those days must come out, or Sample A stops being a fair test.

### Other things found
- **Dzanga** (parked): its tables are in `Testing\Truth_Tables\RumbleOverlap\dzan_…_hele_…`, and the sounds are almost certainly the 4 files in `Testing\Sounds\RumbleOverlap\4kHz` and `8kHz`.
- **Vehicle** (4 files) has no truth table, and 3 of the 4 are from Kakum (Ghana), not Nouabalé-Ndoki.

## 6. What the last zip showed (16 Oct)

Copies are in `retraining/dataset_inventory/more_tables/`.

### The forest training labels are the hand-browsed whole-day tables
- I compared the original day tables in `X_Datasets_To_Use` with the training table for the 124 days they share. **112 days have exactly the same number of rumbles.** On 11 days the training table has a few more (47 in total, probably the "bje added" ones), and on 1 day it has one fewer.
- So the forest training days were most likely browsed end to end, and the time between labelled rumbles is a trustworthy "not rumble". The ELP info doc says "not all rumbles are marked", so this is my reading of the file names and counts, not a confirmed fact. It's worth asking your supervisor.
- **What it means:** for forest days we can safely take "not rumble" examples from anywhere in the day, not only next to rumbles. That's the change most likely to fix precision.

### Validation set: Complex + General, 59 days
| | Days | With rumbles | Elephant-free | Rumbles | Marginal/faint |
|---|---|---|---|---|---|
| RumbleComplex | 30 | 27 | 3 | 358 | 39 |
| RumbleGeneral | 29 | 29 | 0 | 758 | 220 |

- No day is shared with training, Sample A or Sample B.
- The General table only gives start time, duration and top frequency (no bottom frequency), so for scoring I'll treat its boxes as starting at 0 Hz. With ELP's 1% overlap rule that changes almost nothing.

### Sample A: score the 44 official days, not all 50
- ELP's `Schedule_File.xlsx` lists **44 Nouabalé-Ndoki days** as reviewed for Sample A, plus 3 Dzanga and 3 Kakum days.
- The `24-hr` folder has 50 days. **6 are not on the schedule**: `nn02d_20200112`, `nn03a_20201020`, `nn05c_20200113`, `nn05c_20201027`, `nn06a_20200122`, `nn06f_20201027`. Their false alarms may not be real false alarms, so they come out of Sample A scoring.
- Those 44 days hold **970 rumbles, exactly ELP's count** for Sample A, so our test now matches theirs. One of the 44, `nn04a_20180608`, is also a training day, so in our runs Sample A is 43 days and 967 rumbles.
- The schedule also confirms the recording is `nn04a_20200429`, so the 20200428 name in the truth table is the mistake.

### Habitat mix
| Pool | Days | Mixed closed | Mono-dominant | Mixed open | Swamp |
|---|---|---|---|---|---|
| Forest training (B) | 133 | 56% | 29% | 9% | 5% |
| Validation (Complex + General) | 59 | 49% | 36% | 5% | 10% |
| Sample A (schedule) | 44 | 65% | 18% | 6% | 9% |
| Sample B | 80 | 52% | 15% | 16% | 16% |

Training is light on swamp and open forest, and Sample B has three times as much of both. We'll report Sample B per habitat too, so we can see if those are where the false alarms come from.

## 7. The final plan

| Role | Data |
|---|---|
| **Training pools** | A clearing (56 days, 31,129 rumbles) · B forest (133 days, ~3,200 rumbles, 3 elephant-free) · F non-elephant clips |
| **Validation** (threshold, stopping) | RumbleComplex + RumbleGeneral (59 days, 1,116 rumbles) |
| **Test 1 (common yardstick)** | Sample B (80 days, 1,577 rumbles), reported with and without `nn08c_20230104`, and per habitat |
| **Test 2** | Sample A, the 43 scheduled days that aren't training days (967 rumbles), except in runs that move it into training |
| Parked | `nn (only train files)` until we know if it has labels · Dzanga · Vehicle/Kakum |

The 8 runs in section 4 stay as they are. Run 7 now means "+ the 2020 days, if they turn out to be labelled".

**Still open (a question for your supervisor):** does `nn (only train files)` (317 days, Sept–Dec 2020) have rumble labels anywhere on L:? And were the forest training days browsed end to end?
