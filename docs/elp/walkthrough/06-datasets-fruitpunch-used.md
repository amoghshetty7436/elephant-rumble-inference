# Exactly which ELP datasets FruitPunch used

How this was worked out: the training notebook in the repo still contains its saved output, which prints every recording it loaded. I matched those names against your two Raven tables, your folder listing, and *Rumble_Detector_Performance_Methods.docx*.
Full file-by-file list: [fruitpunch_files_used.txt](fruitpunch_files_used.txt)

Their folder layout (from the code) was `cornell_data/Rumble/Training/{Clearings, Sounds}` and `cornell_data/Rumble/Testing/{Dzanga, PNNN}`.

---

## Training: 2 datasets, about 190 day-long recordings

| Dataset | Selection table | Sound files on L: | Match |
|---|---|---|---|
| **Rumble Clearing** (Madj, CEB1, Jobo; 4 kHz, 24 h) | `Training\Selection_Tables\rumble_clearing\rumble_clearing_00-24hr_56days.txt` (31,129 rumbles) | `Training\Sound_Files\rumble_clearing` (56 files) | ✅ **Exact: all 56 days** in your table are the 56 clearing days in their notebook |
| **PNNN forest training** (Nouabalé-Ndoki, 8 kHz, 24 h) | tables in `Training\Selection_Tables\rumble_forest` (which of the 6, still to confirm) | most likely `Training\Sound_Files\rumble_forest\TrainingSet_sounds` (136 files) | ✅ Strong: they used ~134 nn days. Your `…overlapping-clean.txt` table points at `TrainingSet_sounds`, and **27 of its 28 days** are in their training list |

## Testing: 2 datasets, 33 recordings (28 had usable labels)

| Dataset | What they used | Where it is in ELP |
|---|---|---|
| **PNNN "Random Sample"** (`Testing/PNNN/NN_Random_Sample_Hand_ele.txt`) | 24 nn days | ✅ The same days as **Table 3 + Table 4 of the Performance Methods doc** (e.g. nn07c 2018-01-10, nn06a 2019-10-08, nn04a 2019-11-02…). That's ELP's **Nouabale-Ndoki_Grid "Sample A"**: `Testing\Truth_Tables\Nouabale-Ndoki_Grid_RandSmaple` + `Testing\Sounds\Nouabale-Ndoki_Grid-Forest\24-hr` |
| **Dzanga** (clearing, hand-browsed) | 4 days: `dz_20120201`, `dz_20120206`, `dz_20120222_000000`, `dz_20120222_134648` | ❓ **Not visible in your Testing listing.** Need to find where these live on L: |

**Good:** no recording day appears in both their training and testing lists (checked: 0 shared).

## Not used by FruitPunch (we'll leave these out for now, as you asked)
- all of `TrainingClips` (rumble clips and thunder/rain/wind/vehicle negatives)
- `rumble_forest\RumbleOverlap`, and `nn (only train files)` (unless it turns out to be the same days as `TrainingSet_sounds`)
- `Nouabale-Ndoki_Grid-Forest-Additional2024` (Sample B; recorded after their project)
- `RumbleComplex-Forest`, `RumbleGeneral-Forest`, `Vehicle`
- the `DUMMY_NoEles` "no elephant" days

---

## Two important things the Performance Methods doc tells us

### 1. Why their scores looked so good
Sample A has **42+ days**: 24 random days (most with **no elephants**) plus 18 days known to have rumbles, plus 8 extra FM days.
The notebook only builds test examples from days that have **at least 2 labelled rumbles** (`get_embeddings_for_labels_in_file`), and their test list has only 24 of Sample A's ~50 nn days. **26 days are missing**: 5 were loaded and then skipped by that ≥2-rumble rule, and the rest were probably never in their copy. Those are mostly the elephant-free random days, which are exactly where false alarms happen. On top of that, the remaining test data was balanced to 50% rumble. That's how "high 90s" turned into 0.245 precision when ELP ran it on real days.

### 2. How ELP scores a detector, so we can score ours the same way
From the doc: Mike Pitzrick's MATLAB **Detector Validation** tool.
- A detection is a **true positive if it overlaps a true rumble by more than 1% in time *and* in frequency**.
- Precision/recall are reported at **score thresholds in steps of 0.05**, using the selection table's **`Score`** column. So our detector must output a real score, not the repo's fixed 0.9.
- Also reported: false positives per hour.

I'll write a Python version of this scorer so we can check results ourselves, then confirm against the MATLAB tool on one run.

Reference numbers on Sample A (all 42 days, 970 rumbles): **Stanford P 0.842 / R 0.746**, Hori-Harm (0.4) P 0.343 / R 0.60.

---

## Plan for the replication
1. **Data:** the 56 clearing days + `TrainingSet_sounds` (training), and Sample A + the 4 Dzanga days (testing), with their tables.
2. **Run their method unchanged** (same 500 Hz, same negatives-from-gaps, same 2-layer classifier) as one clean script.
3. **Score it two ways:**
   - their way (balanced slices), to check we reproduce their ~0.9+ numbers, which proves the replication worked;
   - ELP's way, on all Sample A days, to get the honest number to compare with Stanford/Hori-Harm.
4. Only after that, add the improvements (hard negatives, threshold tuning, extra datasets).

## Confirmed: the exact tables FruitPunch used (matched against their notebook's per-day rumble counts)

**Training (166 days, all exact unless noted):**
- `rumble_clearing_00-24hr_56days.txt`: 56 of 56 days match exactly.
- `nn_ele_hb_00-24hr_TrainingSet_Removed-nn03c20180611.txt`: 109 days used, 106 match exactly. Three days are off by 2 to 7 rumbles (nn06a_20180615: 67 vs 60, nn06b_20180411: 48 vs 46), so they had a slightly different copy of this table.
- 3 rumbles from `NN_Random_Sample_Hand_nn04aSupplemental_with_20200428.txt` on nn04a_20180608 (a training day), which makes that day exact (45 + 3 = 48).
- **Not used:** `TrainingSet_v2` (only 67 days match), `overlapping-clean`, `Select_HH_forOverlap...`. Their code loads every table in the folder, so if those had been there the counts would be roughly doubled.
- 24 TrainingSet days are absent: 18 have only 1 rumble (dropped by the ≥2-rumble rule); 6 have more (e.g. nn02c_20180625 with 19, nn02e_20180629 with 14), so those sound files were probably missing from their copy.

**Testing (28 days):**
- `NN_Random_Sample_Hand_ele.txt`: 20 days, all exact. The 3 left out have 1 rumble each.
- `NN_Random_Sample_Hand_nn04aSupplemental_with_20200428.txt`: 4 days (nn04a 20180710, 20191102, 20200104, 20200314), all exact. Not used: 20180122 and 20180922 (2 rumbles each) and 20200428 (57).
- 4 Dzanga days (ignored for now, per your call).

## Still needed
1. **The names of the 6 tables** in `Training\Selection_Tables\rumble_forest`. A screenshot is fine.
2. **Where the 4 Dzanga test days are** (`dz_20120201_000000.wav` etc.) and their `dzan_…_hele_….txt` tables. In PowerShell:
   ```
   Get-ChildItem L:\ELP\Detectors -Recurse -Filter "dz*_2012020*" -ErrorAction SilentlyContinue | Select FullName
   ```
3. **The names of the files in `Testing\Truth_Tables\Nouabale-Ndoki_Grid_RandSmaple`.**
