# How to improve FruitPunch's detector

**The problem in one line:** on ELP's real test (whole days), FruitPunch's model had **precision 0.245**: about 3 of every 4 detections were false alarms. Recall was fine. So the biggest wins come from **teaching the model what "not a rumble" sounds like**, then from smarter decisions on top of the model's scores.

Ideas are ordered by how much I expect them to help per hour of work. Expected effects are my judgement, not measured yet.

---

## Step 0: Get an honest baseline first
Before changing anything, finish the current run and look at `elp_score.csv` (Test 2). Every idea below gets judged by whether it moves **precision and recall on whole days**, not by the 50/50 snippet test.

Also fix how we judge:
- **Split by day, not by random snippets.** Today, the 5% "check" set is random frames from the training days, and neighbouring frames are near-identical, so it's too easy. Hold out ~15 whole forest days as a **validation set** for choosing settings, and keep the test days for the final number only.
- **Score only reviewed days** (I need the Sample A day list).

## Step 1: Better "not rumble" examples (biggest expected gain, precision)
Right now, negatives come only from gaps **next to** rumbles on days **with** rumbles. The model never practises on thunder, rain, planes, insects or elephant-free days.
1. **Random negatives from whole days:** sample windows from anywhere in a day that are ≥ 30 s away from any labelled rumble, including **elephant-free days**. (`TrainingSet_sounds` has days that were removed as `DUMMY_NoEles`. Those are perfect negatives.)
2. **Hard-negative mining (the key one):** run the model over the *training* days, take every detection that doesn't overlap a labelled rumble (a false alarm), add those as "not rumble", retrain, and repeat 2–3 rounds. This directly targets what fools it.
3. **Stop throwing away 90% of negatives.** Keep more negatives than positives (e.g. 1:5 or 1:10) and use loss weights, so the model learns that rumbles are rare.

## Step 2: Smarter decisions on top of the scores (cheap, no retraining)
Today, any run of ≥ 4 frames (2.5 s) where "rumble" > "not" becomes a detection with a fixed box (5–250 Hz).
- **Smooth the scores over time** (e.g. average over 3–5 s) before deciding. That removes one-frame flickers that become false alarms.
- **Tune the threshold, minimum length and gap-merging** on the validation days, picking the setting that maximises precision with recall ≥ 0.7.
- **Use a better detection Score** (max, or mean of the top frames) so the ELP threshold sweep works well.
- **Estimate the frequency box** from the spectrogram (where the energy is under 100 Hz) instead of a fixed 5–250 Hz. ELP's 1% overlap rule is lenient, so this matters less for scoring, but it helps analysts reviewing in Raven.

## Step 3: Let the model hear more than 0.64 s at a time (model)
The classifier decides from **one** 0.64 s fingerprint. A rumble lasts 2–8 s and has a shape (rise, harmonics, fall), and thunder has a different shape.
- Replace the 768 → 192 → 2 classifier with a small **temporal model** over ~10 s of fingerprints: a 1-D convolution or tiny transformer/GRU. It's still cheap to train on the cached fingerprints.
- Try **earlier AVES layers** or a learned mix of layers. The last layer isn't always the best for a sound AVES never trained on.

## Step 4: More and better-matched training data
91% of training rumbles come from the 56 **clearing** days, but the test is **forest**, which is quieter, further away and noisier. Once we've matched FruitPunch, adding data is the next big lever:
- the remaining **forest** training days (`rumble_forest\nn (only train files)`, ~318 days) and the other forest tables (`TrainingSet_v2`, `overlapping-clean`, `Select_HH`);
- **weight forest examples higher**, or sample clearings and forest evenly;
- **clean the labels:** drop `DUMMY_NoEles` rows, and check rows tagged `marginal`/`faint`/`REMOVE`.

Keep `Additional2024` and Sample A strictly for testing.

## Step 5: Bigger changes (only if Steps 1–4 aren't enough)
- **Fine-tune the top layers of AVES** on rumbles with a small learning rate. This needs more GPU time but often gives the largest jump.
- **Try a different speed-up:** 500 Hz (5 octaves) was one choice. 1,000 Hz or 250 Hz may suit AVES better. It's a cheap test with `--mode day`.
- **Compare other audio foundation models** (e.g. Google's Perch, BirdNET embeddings) the same way.
- **Ensemble 3–5 models** trained with different seeds, and average their scores. It usually trims false alarms.

## Speed and loading (for faster experiments, and for CPU use in Africa)
- **Cache the 500 Hz audio** once per day (~86 MB, float16). After that, no run ever re-reads 1.3 GB from L:. That's most of the current time per forest day.
- **Use `--mode day`** for experiments: it embeds each day once instead of ~4× over.
- **Store whole-day fingerprints** for the validation and test days, so Steps 2 and 3 can be tried in seconds without AVES.
- **For CPU-only use:** half-precision or quantised AVES, and process several hours in parallel across CPU cores.

---

## Suggested order
| # | What | Retrain? | Effort | Expected effect |
|---|---|---|---|---|
| 0 | Baseline + day-based validation set | no | small | needed to measure anything |
| 1 | Smoothing + threshold/length tuning | no | small | precision ↑ |
| 2 | Random + elephant-free negatives, keep more negatives | yes | small | precision ↑↑ |
| 3 | Hard-negative mining (2–3 rounds) | yes | medium | precision ↑↑↑ |
| 4 | Temporal classifier over ~10 s | yes | medium | precision ↑, recall ↑ |
| 5 | More forest training data | yes | medium | recall ↑, precision ↑ |
| 6 | Fine-tune AVES / other models / ensembles | yes | large | both ↑ |

Target: precision and recall both above 0.7 on whole days (ELP's goal). Stanford's model is at P 0.84 / R 0.75.
