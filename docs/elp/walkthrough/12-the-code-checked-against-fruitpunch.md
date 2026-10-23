# The replication code, checked line by line against FruitPunch's notebook

I re-read the code on the `elp-retraining` branch (latest commit `ca6ae2c`) and compared it with FruitPunch's `notebooks/training_notebook.ipynb` and their inference program. Everything below comes from those files.

## 1. Is `fp_replica.py` the only code?

It's the only file you **run**. It uses four other files:

| File | Who wrote it | What it does |
|---|---|---|
| `tools/fp_replica.py` (477 lines) | me | The main script, with three commands: `prepare`, `train` and `detect` |
| `tools/fp_data.py` (131 lines) | me, copying the notebook's rules | Reads the Raven tables and picks the "not rumble" windows |
| `tools/elp_score.py` (191 lines) | me, copying ELP's MATLAB rules | Grades whole-day detections the way ELP does |
| `elephant_rumble_inference/aves_torchaudio_wrapper.py` | **FruitPunch, unchanged** | Loads AVES, the big frozen "listening" model |
| `elephant_rumble_inference/elephant_rumble_classifier.py` | **FruitPunch, unchanged** | The small 768 → 192 → 2 model that gets trained |

The notebook couldn't be run as it is, for two reasons:
- It uses `torchaudio.io.StreamReader`, which newer PyTorch removed.
- Its paths point to the author's own computer (`cornell_data/…`).

So `fp_replica.py` is the notebook's steps rewritten as one script. The model itself and AVES are FruitPunch's own code.

## 2. What `prepare` does (the slow step, once)

1. **Read the answer key** (`fp_data.read_labels`). Every row of all 4 tables is one rumble.
   - Start = `File Offset (s)`, length = `End Time − Begin Time`.
   - Rows with no End Time are dropped.
   - Labels for the same recording are pooled across tables.
   - All of this is the same as the notebook.
2. **Pick the days.**
   - A day needs at least 2 rumbles and 2 "not rumble" windows. That's the notebook's rule.
   - `--only-files` then keeps exactly FruitPunch's 165 training and 24 test days.
   - A day counts as training or testing depending on which sound folder its `.wav` is in.
3. **Pick the "not rumble" windows** (`fp_data.negative_windows`). These come from the gaps between rumbles, using the notebook's exact rules:
   - the first minute of the day;
   - skip gaps under 20 s;
   - 20–130 s gaps: take the gap minus 5 s at each end;
   - longer gaps: 60 s after the first rumble and 60 s before the next;
   - gaps over 260 s: also 120 s from the middle.
4. **Load the sound** (`load_audio`): one channel, resampled to **500 samples per second**, then rounded to half precision the way the notebook's cache stored it.
5. **Make fingerprints** (`window_embeddings_fp`).
   - Each window is played to AVES together with up to **500 s before it and 200 s after it**.
   - AVES gives one 768-number fingerprint per **0.64 s**, scaled to length 1.
   - Only the fingerprints inside the window are kept. That's the notebook's `get_aves_embeddings_from_file_with_buffers`.
6. **Save** each day's rumble and not-rumble fingerprints to `fp_cache\fp\train\` or `…\test\`.

Your run gave 184,104 rumble and 1,807,915 not-rumble training fingerprints. The notebook printed 184,151 and 1,808,388.

## 3. What `train` does (about 2 minutes)

| Setting | Our script | FruitPunch's notebook | Same? |
|---|---|---|---|
| Model | 768 → 192 → 2, LeakyReLU, 20% dropout (their own class) | same | ✅ |
| Balancing | keep as many "not rumble" fingerprints as rumble ones, picked at random | same | ✅ (ours uses a fixed random seed, theirs doesn't) |
| Train / validation split | 95% / 5%, seed 42 | same | ✅ |
| Batch size | 64, shuffled | 64, shuffled | ✅ |
| Optimizer | Adam, default learning rate (0.001) | same | ✅ |
| Loss | cross-entropy with "balanced" class weights | same | ✅ The data is already 50/50, so the weights come out as 1 and 1, which means no extra weighting. Their notebook printed `tensor([1., 1.])` too. |
| Number of epochs | as many as fit in **2 minutes** (limit 20,000) | same | ✅ Your PC fitted **16**. The notebook's run fitted **11**. |
| Early stopping | stop if val loss rises above 2.3 × its lowest value | same rule | ✅ It never triggered in either run. |
| **What the validation check uses** | **5% of the training fingerprints** | **the test set** (cell 23: `val_loader = testing_loader`) | ❌ **deliberate difference**, see below |
| Which model is saved | the one after the last epoch | the last epoch | ✅ |
| Test report | precision, recall, F1, accuracy and ROC AUC on the balanced test fingerprints, plus FruitPunch's model on the same fingerprints (`--compare`) | classification report on the balanced test set | ✅ |

**Check on the training curve.** The notebook's saved output shows train loss 9.01 → 7.45 → 6.92 → … → 5.36 over 11 epochs. Yours went 9.16 → 7.58 → 7.04 → … → 5.42 at epoch 11. That's nearly the same curve.

**What "validation" and "test" mean here:**
- **Validation** is a small held-back set the model is checked on after each epoch, only to decide when to stop.
- **Test** is the final exam, used once at the end.
- FruitPunch used the test set for both jobs. That lets the test slightly influence training, so I switched validation to 5% of the training data. To copy them exactly, add `--val-on-test`.
- One weakness of our way: those 5% are random fingerprints from the *same days* as training, so the check is easy (98% correct). A better validation set would be whole held-out days.

## 4. What `detect` does (the whole-day test, my addition)

FruitPunch's notebook has no whole-day test. Their separate inference program (`main.py`) does detection, and `detect` follows it:
1. It fingerprints the whole day an hour at a time, with ~10 s of overlap at each edge, like their `AudioFileProcessor`.
2. It labels a moment "rumble" when the model's rumble score beats its not-rumble score.
3. It keeps runs of **at least 4 fingerprints in a row (~2.5 s)**. That's their `find_long_enough_segments(n=3)`.
4. Each detection becomes a box from 5 to 250 Hz, the same as theirs.

The differences:
- **The Score.** Their program writes a fixed `score = 0.9` (`main.py:157`, marked "TODO"). Ours writes the model's average confidence, because ELP's threshold table needs a real score.
- **The grading** (`elp_score.py`) is new. It copies ELP's Detector Validation rules: more than 1% overlap in time and frequency, and thresholds in 0.05 steps.

## 5. Every difference from FruitPunch, in one list

| # | Difference | Why | Effect on results |
|---|---|---|---|
| 1 | Validation uses 5% of training data, not the test set | stops the test from influencing training | small; `--val-on-test` restores theirs |
| 2 | Audio read with `soundfile` and resampled with `scipy` | their `StreamReader` no longer exists | fingerprints very slightly different, not bit-identical |
| 3 | Fingerprints stored at half precision | half the disk space | rounding at the 4th digit |
| 4 | Random choices use fixed seeds | same command gives the same model | none, except that a faster PC fits more epochs (16 vs 11) |
| 5 | Dzanga test days left out | your decision | the snippet test covers forest only |
| 6 | `nn04a_20180608` is ignored in the test | it's a training day that also sits in the test folder | keeps training data out of the test |
| 7 | `detect` writes a real Score, and `elp_score.py` grades the ELP way | ELP's method needs a score | only affects evaluation |
| 8 | Optional `--mode day` (fingerprint each day once) | faster | **not used in your run**: you ran the default exact mode |

Nothing about the data, the "not rumble" rules, the model, the optimizer, the batch size, the 2-minute budget or the stopping rule was changed.
