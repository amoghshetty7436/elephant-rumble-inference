# Retraining FruitPunch's model on your PC

The script is `tools/fp_replica.py`. It is the training notebook turned into one script, using only the tables FruitPunch used (see [06](06-datasets-fruitpunch-used.md)). There are three steps:

1. **prepare**: turn the labelled parts of every day into AVES embeddings. This is the slow step, and it runs on the GPU.
2. **train**: train the small classifier (2 minutes), then report results the way FruitPunch did.
3. **detect**: run the model over whole test days, write Raven tables with a real Score, and score them the way ELP does.

The code is in [elephant-rumble-inference-elp-retraining.zip](/mnt/project-files/retraining/elephant-rumble-inference-elp-retraining.zip) (unzip it and work inside that folder). It is also on the local `elp-retraining` branch of the fork, which I haven't pushed yet.

## What I checked, and what I couldn't
- **Checked:** the label and negative-window logic reproduces FruitPunch's notebook output. For 187 of 189 days, both the rumble count and the negative-window count match exactly. Training comes out at 184,104 rumble frames, against 184,151 in their run. The unit tests pass.
- **Also checked (2026-10-09):** I installed PyTorch on the cloud machine (CPU only) and ran all three steps end to end on made-up recordings. These were 1-hour, 8 kHz .wav files with fake 4-second rumbles, plus Raven tables in ELP's format. `prepare` (both modes), `train` (including `--compare` with the real `2024-07-03.pth` and `--val-on-test`) and `detect` with ELP scoring all ran without errors. FruitPunch's original model scored 99.6% on our fingerprints of the fake rumbles. That suggests our audio-to-AVES pipeline produces the same kind of fingerprints theirs did.
- **Still not checked:** the real ELP recordings and the GPU. The trial run in step 3 covers both.

## 1. Set up the new PC (once, about 30 minutes)
1. **NVIDIA driver:** install the latest "Game Ready" or "Studio" driver for the RTX 4060 from nvidia.com/drivers. You don't need a separate CUDA install, because PyTorch brings its own.
2. **Python 3.12:** download it from python.org (Windows installer, 64-bit). On the first screen, tick **"Add python.exe to PATH"**, then Install.
3. **VS Code:** install it from code.visualstudio.com. Open it, go to Extensions (Ctrl+Shift+X), and install **Python** (by Microsoft).
4. **The code:** unzip `elephant-rumble-inference-elp-retraining.zip` to e.g. `C:\work\`. In VS Code, use **File > Open Folder** and pick `C:\work\elephant-rumble-inference`.
5. **Terminal:** in VS Code, open **Terminal > New Terminal** (a PowerShell panel at the bottom), and run these one at a time:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
   If activation says "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer Y, and try again. The prompt now starts with `(.venv)`.
6. **Libraries** (PyTorch is a ~3 GB download):
   ```
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install -r tools\requirements-train.txt
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   The last line should print `True NVIDIA GeForce RTX 4060...`. If it prints `False`, update the driver from step 1.
7. When VS Code asks "select interpreter", pick the one in `.venv`. Every time you open a new terminal, run `.venv\Scripts\activate` first.

The first run also downloads the AVES model (~380 MB) from Google Cloud.

## 2. Copy the data to the SSD, then set the paths
Your C: and D: are both on the same 1 TB NVMe SSD, so either one works. Use whichever has **~300 GB free**: about 40 GB of clearing days, about 175 GB of forest training days, and about 80 GB of test days. In the VS Code terminal (this takes a while, and you can re-run it to resume):
```
$L = "L:\ELP\Detectors\Elephant\Datasets"
$D = "D:\ELPdata"
robocopy "$L\Training\Selection_Tables" "$D\Training\Selection_Tables" /E
robocopy "$L\Testing\Truth_Tables" "$D\Testing\Truth_Tables" /E
robocopy "$L\Training\Sound_Files\rumble_clearing" "$D\Training\Sound_Files\rumble_clearing" /E /MT:8
robocopy "$L\Training\Sound_Files\rumble_forest\TrainingSet_sounds" "$D\Training\Sound_Files\rumble_forest\TrainingSet_sounds" /E /MT:8
robocopy "$L\Testing\Sounds\Nouabale-Ndoki_Grid-Forest\24-hr" "$D\Testing\Sounds\Nouabale-Ndoki_Grid-Forest\24-hr" /E /MT:8
```
For the trial run you don't need to wait for all of it: the trial only reads the first 2 days of each folder.

Then set the paths (run these in every new terminal before the commands below):
```
$D = "D:\ELPdata"
$TABLES = "$D\Training\Selection_Tables\rumble_clearing\rumble_clearing_00-24hr_56days.txt",
          "$D\Training\Selection_Tables\rumble_forest\nn_ele_hb_00-24hr_TrainingSet_Removed-nn03c20180611.txt",
          "$D\Testing\Truth_Tables\Nouabale-Ndoki_Grid_RandSmaple\NN_Random_Sample_Hand_ele.txt",
          "$D\Testing\Truth_Tables\Nouabale-Ndoki_Grid_RandSmaple\NN_Random_Sample_Hand_nn04aSupplemental_with_20200428.txt"
$TRAIN = "$D\Training\Sound_Files\rumble_clearing", "$D\Training\Sound_Files\rumble_forest\TrainingSet_sounds"
$TEST  = "$D\Testing\Sounds\Nouabale-Ndoki_Grid-Forest\24-hr"
$CACHE = "D:\fp_cache"
```
Fix any folder name that doesn't match your drive. All four tables go to `prepare`, because the notebook pools every table's labels for each day. Whether a day counts as training or testing depends only on which sound folder its .wav file is in.

## 3. Trial run (2 days per split)
```
python tools\fp_replica.py prepare --tables $TABLES --train-sounds $TRAIN --test-sounds $TEST --cache "$CACHE-trial" --only-files tools\fruitpunch_files_used.txt --limit-files 2
python tools\fp_replica.py train --cache "$CACHE-trial" --out "$CACHE-trial\trial.pth" --minutes 0.2
```
If either one fails, paste me the error. Each logged line also shows how long that day took, which gives us a real time estimate.

## 4. Full run
```
python tools\fp_replica.py prepare --tables $TABLES --train-sounds $TRAIN --test-sounds $TEST --cache $CACHE --only-files tools\fruitpunch_files_used.txt
python tools\fp_replica.py train --cache $CACHE --out "$CACHE\retrained.pth" --compare "<path to>\2024-07-03.pth"
python tools\fp_replica.py detect --model "$CACHE\retrained.pth" --sounds $TEST --out "$CACHE\detections" --truth $TABLES[2] $TABLES[3]
```
- `--only-files` keeps exactly FruitPunch's days. Without it, 6 extra training days with rumbles would be included; FruitPunch's copy seemed to be missing them.
- `prepare` can be stopped and restarted, because finished days are skipped.
- `--compare` scores FruitPunch's original `2024-07-03.pth` on the same test frames. This is the replication check: the two models should score about the same.
- `detect` runs over **all** days in the 24-hr folder, elephant-free days included, and prints precision, recall and false alarms per hour at each threshold from 0 to 1 in 0.05 steps. Those are the numbers that compare with Stanford (P 0.842 / R 0.746).

## Estimated time on an RTX 4060 (not measured; the trial run will tell)
| Step | What it does | Estimate |
|---|---|---|
| prepare (default, exact) | Each labelled window is embedded with 500 s / 200 s of context, like the notebook. That's ~460 days of audio | 2 to 4 h |
| prepare `--mode day` | Each day is embedded once, an hour at a time, like inference. That's ~190 days of audio. Also pass `--mode day` to `train` | 1 to 1.5 h |
| train | 2-minute budget, like the notebook | ~3 min incl. loading |
| detect | 61 test days | 20 to 40 min |

These times assume the data is on the SSD (step 2). Reading from the L: drive would be slower.

## Differences from the notebook (all deliberate)
- **Early stopping uses 5% of the training data.** The notebook stopped on the test set (cell 23), which leaks test data. Add `--val-on-test` to copy the notebook exactly.
- **Seeded:** the same command gives the same model. The only exception is the 2-minute budget: a faster PC fits in more epochs.
- **Audio** is read with `soundfile` and resampled with scipy. The notebook used `torchaudio.io.StreamReader`, which newer torchaudio removed. The resampling filter is slightly different, so the embeddings won't be bit-identical.
- **Embeddings are cached as float16** (~3 GB instead of ~6 GB).
- **Detections have a real Score:** the mean rumble probability over the detection. The original wrote a fixed 0.9.
