# The retraining script, explained simply

## Part 1: What goes in

There are two kinds of input files, and the script needs both.

| Kind | What it is | Why we need it |
|---|---|---|
| **`.wav` sound files** | One file = one recorder, one full day (24 hours). Forest days are recorded 8,000 times a second, about 1.3 GB each. Clearing days are 4,000 times a second, about 0.7 GB each. | This is the actual sound the model listens to. |
| **`.txt` Raven tables** | Spreadsheets (tab-separated) that analysts made in the Raven program. Each row is one rumble they found: which day (`Begin File`), when in that day (`File Offset (s)`), how long (`End` minus `Begin Time`), and its pitch range. | They tell the model *where* the rumbles are, so it can learn what a rumble sounds like. These are the "answers". |

So the `.wav` is the question, and the `.txt` is the answer key.

### Exactly what we use (only what FruitPunch used)

**Training: 165 days**
| Sound folder | Days | Table | Rumbles |
|---|---|---|---|
| `Training\Sound_Files\rumble_clearing` | 56 clearing days (sites like CEB1, Madj, Jobo) | `rumble_clearing_00-24hr_56days.txt` | 31,129 |
| `Training\Sound_Files\rumble_forest\TrainingSet_sounds` | 109 forest days (PNNN, recorders nn01a to nn10b) | `nn_ele_hb_00-24hr_TrainingSet_Removed-nn03c20180611.txt` | 3,143 |

In total that's **34,272 rumbles in about 3,960 hours of sound**. Most rumbles are short: half are under 3.3 seconds. All the rumbles together add up to only about 34 hours.

**Testing: 24 days** (sound folder `Testing\Sounds\Nouabale-Ndoki_Grid-Forest\24-hr`)
| Table | Days | Rumbles |
|---|---|---|
| `NN_Random_Sample_Hand_ele.txt` | 20 | 843 in the table |
| `NN_Random_Sample_Hand_nn04aSupplemental_with_20200428.txt` | 4 (nn04a) | |

That's 910 rumbles on these 24 days. FruitPunch also tested on 4 Dzanga days, and we're leaving those out, as you asked.

**Why only these days?** These are exactly the days FruitPunch's notebook printed when it ran. Some days are in the folders but were skipped, for two reasons:
- **18 training days and 3 test days have only 1 rumble.** FruitPunch's code skips any day with fewer than 2.
- **6 training days were probably missing from their copy.** The list file `tools/fruitpunch_files_used.txt` keeps those out too, so we match them exactly.

## Part 2: What the script does, step by step

### Step 1: `prepare` (the slow part, run once)
For every day:
1. **Read the rumble rows** for that day from all four tables, and sort them by time.
2. **Pick "not rumble" examples** from the quiet gaps between rumbles. These are FruitPunch's exact rules:
   - Take the first minute of the day (before the first rumble).
   - If two rumbles are less than 20 s apart, skip the gap, because there are often unlabelled rumbles in between.
   - If the gap is 20 to 130 s, take the gap minus 5 s at each end.
   - If the gap is longer, take 60 s just after the first rumble and 60 s just before the next one.
   - If the gap is longer than 260 s, also take 120 s from the middle.
3. **Load the sound and slow it down.** The day is resampled to **500 samples per second**. The AVES model thinks it's hearing 16,000 per second, so to AVES the rumble sounds 32 times faster and 5 octaves higher, like a bird call, which AVES is good at.
4. **Turn sound into "fingerprints."** AVES listens to each window and outputs one list of 768 numbers for every 0.64 seconds of sound. Each list is scaled to length 1 so they can be compared fairly.
   - Like FruitPunch, each rumble or gap window is played to AVES with up to 500 s of sound before it and 200 s after it, so AVES has context. Only the fingerprints inside the window are kept.
5. **Save** the "rumble" and "not rumble" fingerprints for that day into the cache folder.

Result: about **184,000 rumble fingerprints** and **1.8 million "not rumble" fingerprints** for training. That matches FruitPunch's printed 184,151 and 1,808,388 almost exactly. The tiny difference comes from 2 days where their copy of the table had a few more rows.

### Step 2: `train` (about 3 minutes)
1. **Balance:** randomly keep only as many "not rumble" fingerprints as rumble ones (184k each), so the model doesn't just learn "always say no".
2. **Split:** use 95% for training, and hold 5% back to check progress (the same split and seed, 42, that FruitPunch used).
3. **The model** is the same small network: 768 numbers, then 192, then 2 scores ("rumble" or "not"), with 20% dropout. Only this small part learns. AVES stays frozen.
4. **Learn:** the Adam optimizer, batches of 64, for **2 minutes**, stopping early if the check loss gets much worse. These are the same settings as the notebook.
5. **Save** the model as a `.pth` file. It's the same format as their `2024-07-03.pth`, so it works with their inference program.
6. **Report:** accuracy, precision and recall on the 24 test days, balanced 50/50, which is how FruitPunch reported their ~97%. With `--compare 2024-07-03.pth`, it also scores their original model on the very same test fingerprints. If the two numbers are close, we've replicated their work.

### Step 3: `detect` (extra, for the honest ELP number)
FruitPunch's 97% only checks short balanced snippets, not whole days. This step does what ELP actually needs:
1. Play **every test day in full** (all 61 days in the folder, including days with no elephants) through AVES and the model, an hour at a time.
2. Wherever the model says "rumble" for at least 4 fingerprints in a row (about 2.5 s), write one detection into a Raven table with a **Score**: how sure the model was, from 0 to 1.
3. Compare the detections with the truth tables using ELP's rule: a hit is any detection that overlaps a real rumble by more than 1% in time and in pitch. The script prints precision, recall and false alarms per hour at thresholds 0, 0.05, …, 1. Those are the numbers that compare with Stanford's P 0.84 / R 0.75.

## Part 3: Where we differ from FruitPunch, and why

These all leave the method itself unchanged: the same data, the same rules, the same model and the same settings.

| Difference | Why | Effect |
|---|---|---|
| Training checks progress on 5% of the **training** data. FruitPunch's notebook checked on the **test** data (cell 23). | Checking on test data lets the test results leak into training, which flatters the score. | Small. Add `--val-on-test` to do it exactly their way. |
| The audio is read with `soundfile` and resampled with `scipy`. | The tool FruitPunch used (`torchaudio`'s StreamReader) was removed from new versions of PyTorch's audio library. | Very slightly different filtering, so the fingerprints aren't bit-for-bit identical. Not expected to change results. |
| Fingerprints are stored at half precision (float16). | It halves the disk use (~3 GB instead of ~6 GB). | Rounding at the 4th digit. Negligible. |
| Random choices are seeded. | So the same command gives the same model every time. | None, except that the 2-minute limit means a faster PC fits in more training. |
| No Dzanga test days. | Your call. | Test numbers cover forest only. |
| `detect` writes a real Score (FruitPunch wrote a fixed 0.9). | ELP's scoring needs a Score to make the threshold table. | This is only used for evaluation, so training is unaffected. |

There's also one optional shortcut: `--mode day` embeds each day once instead of window by window. It's about twice as fast but not identical to FruitPunch. The default is the exact way.
