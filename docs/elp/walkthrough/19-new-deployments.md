# Running the final detector on new deployments (new desktop)

The two deployments:

| | Dzanga, December 2025 (dep10, part 2) | Nouabalé-Ndoki, June 2023 (dep17) |
|---|---|---|
| Folder | `L:\ELP\DATA_ARCHIVE\x_15TB_Archiving_Stage_Space\2020_SPAC_DSPA_136895\dz_202512_dec_dep10\part2_sounds` | `L:\ELP\Projects\Nouabale\nn_sounds\nn_202306_jun_dep17` |
| Recordings | 32 kHz, 6-hour files, 34 sites | 8 kHz, 24-hour files, 43 sites |

## Does the different sample rate need code changes?

No. The final model reads audio with `fp_replica.load_audio`, which resamples whatever rate the file has to 500 Hz with `scipy.signal.resample_poly` (a proper anti-alias filter): 32 kHz is divided by 64, 8 kHz by 16. The training and test days went through exactly the same function, and they already mixed rates (the noise clips include 8 kHz and 500 Hz folders). File length doesn't matter either: a 6-hour file is processed like a quarter of a day.

The old `elephant-rumble-inference` command (main.py) is **not** what we use here: it runs FruitPunch's 2024 model, resamples with ffmpeg instead, and cuts files at 24 hours. The new script `tools\detect.py` runs the rerun's final model (`E2_ctx8_noise10_rand3000_gap0.5_hard0_augnone`, best run, with its alarm rule and the cut-off that found 80% of the rumbles on the validation days).

## Writing to the network: is it slower?

Writing the results is not the slow part. Each recording gives a Raven table of a few KB and a probability file of about 0.3 MB per day, so writing to L: costs well under a second per file, against about 30 s of AVES per day. `survey` measures it on your PC (the "Write to --out" and "Write locally" columns).

What *can* be slow is **reading** the audio from L:: a 24-hour 8 kHz file is about 0.7 GB, and a day at 32 kHz is about 2.8 GB. That has to come over the network whether the outputs go to L: or to C:, so writing locally and copying afterwards doesn't save time. The script reads and resamples the next two files in the background while the GPU works on the current one, so the GPU stays the bottleneck as long as the network keeps up: about 25 MB/s for the 8 kHz files (easy) and about 90 MB/s for the 32 kHz ones (right at what a wired gigabit link gives, about 100 MB/s). So the Dzanga files may be limited by the network rather than the GPU. If `survey` shows a much lower read speed (Wi-Fi, or a slow share), the fix is copying the *audio* to a local SSD first, not the results.

So: write straight to the network. If you'd still like local-then-copy, use `--out C:\RumbleResults\...` and the `robocopy` line at the end.

---

## Part 1: Setting up the new desktop (once)

Everything is typed into **PowerShell** (Start → type PowerShell). Paste one grey box at a time and wait for it to finish.

### 1. Check the graphics card and the L: drive

```
nvidia-smi
Test-Path "L:\ELP\Projects\Nouabale\nn_sounds\nn_202306_jun_dep17"
Test-Path "L:\ELP\Detectors\Other\Amogh\Dataset\Configurations\_study\final_detections\final_model.json"
```

- `nvidia-smi` should print a table with the GPU's name and a "CUDA Version". If it says "not recognized", install the NVIDIA driver first (https://www.nvidia.com/Download/index.aspx), restart, and try again. **Note the GPU name**: step 4 depends on it.
- Both `Test-Path` lines should print `True`. If they print `False`, L: isn't connected on this PC. On the old PC, run `net use` and look for the `L:` line: it shows the network path (something like `\\server\share`). Then on the new PC:

```
net use L: \\server\share /persistent:yes
```
(with the path from the old PC; it may ask for your lab username and password).

### 2. Install Git and Python 3.11

```
winget install --id Git.Git -e
winget install --id Python.Python.3.11 -e
```

Close PowerShell and open a new one (so it finds the new programs), then check:

```
git --version
py -3.11 --version
```

If `winget` isn't recognized, install them by hand: Git from https://git-scm.com/download/win and Python 3.11 from https://www.python.org/downloads/release/python-3119/ ("Windows installer (64-bit)", tick **"Add python.exe to PATH"**).

### 3. Get the code (the branch with detect.py)

```
cd "$HOME\Desktop"
git clone https://github.com/amoghshetty7436/elephant-rumble-inference.git
cd elephant-rumble-inference
git checkout claude/project-thread-qzngu1
```

### 4. Make the Python environment and install PyTorch for the GPU

```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If `Activate.ps1` gives a red error about "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, and run the `Activate.ps1` line again. You should see `(.venv)` at the start of the line. **Every new PowerShell window needs the `cd` and `Activate.ps1` lines again** before running the detector.

Then install PyTorch. Pick by the GPU name from step 1:

Most NVIDIA cards (RTX 20xx, 30xx, 40xx, A-series):
```
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

An **RTX 50xx** (5070, 5080, 5090) or other Blackwell card needs a newer build:
```
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Then the rest:
```
pip install -r tools\requirements-train.txt
```

Check that PyTorch sees the GPU (it should print `True` and the GPU name):
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 5. The AVES model (378 MB)

It downloads by itself the first time the detector runs. To skip the download, copy it from the old PC: the folder `C:\Users\<you>\.cache\torch\hub\fruitpunch_elephants` (it holds `aves-base-bio.torchaudio.pt` and `aves-base-bio.torchaudio.model_config.json`) goes to the same place on the new PC:

```
New-Item -ItemType Directory -Force "$HOME\.cache\torch\hub\fruitpunch_elephants"
```
and copy the two files into it.

The rumble model itself stays on L: (`Configurations\_study\...`); the script reads it from there.

### 6. Test it

```
python -m pytest tests\test_detect.py -q
```
It should say `2 passed`.

---

## Part 2: Survey the two deployments (about 10 minutes)

This lists every recording, times one file of each sample rate (network read, resampling, AVES, writing to L: and to C:), and estimates the total time. Results go to `survey.md` in the output folder.

```
cd "$HOME\Desktop\elephant-rumble-inference"
.\.venv\Scripts\Activate.ps1
$out = "L:\ELP\Detectors\Other\Amogh\Detections"
$dz = "L:\ELP\DATA_ARCHIVE\x_15TB_Archiving_Stage_Space\2020_SPAC_DSPA_136895\dz_202512_dec_dep10\part2_sounds"
$nn = "L:\ELP\Projects\Nouabale\nn_sounds\nn_202306_jun_dep17"
python tools\detect.py survey $dz $nn --out "$out\_survey"
notepad "$out\_survey\survey.md"
```

(`$out` is a suggestion; any folder outside the sound folders works. Send me `survey.md` and I'll check the numbers.)

## Part 3: Run them

One after the other, in one go (leave the PC plugged in; the script keeps Windows awake, but pause Windows Update):

```
python tools\detect.py run $nn --out "$out\nn_202306_jun_dep17" ; python tools\detect.py run $dz --out "$out\dz_202512_dec_dep10_part2"
```

It prints one line per recording with the time per 24 h and the time left. If it stops (crash, restart), **run the same line again**: it skips the recordings that already have a table.

What you get, mirroring the sound folders:

| File | What |
|---|---|
| `<recording>.rumbles.txt` | Raven table of that recording's rumbles |
| `<recording>.probs.npy` | The rumble probability of every 0.64 s; lets you change the cut-off later in seconds |
| `_all_rumbles.txt` (one per folder) | One Raven table for the whole folder, recordings one after another |
| `summary.md` / `summary.csv` | Per site: recordings done, hours, rumbles (total and per day), time taken, seconds per 24 h of audio. Also checks that every finished recording has its table and matching probabilities. Updated after every site and at the end |
| `timings.csv` | One line per recording: when it finished, site, sample rate, hours, read + resample, waiting for the network, AVES, classifier and writing times, total time, number of rumbles, highest probability |
| `run_log.txt` | Everything printed on screen, for every run (appended) |
| `detector.json` | Which model, rule and cut-off made these tables, and the PC, GPU, Python and PyTorch versions |
| `failed.txt` | Any recording that couldn't be read (the run carries on) |

### Changing the cut-off afterwards

The tables keep detections scoring at least the validated cut-off (80% recall on the validation days). To keep more (higher recall, more false alarms) or fewer, rewrite the tables from the saved probabilities, no GPU needed:

```
python tools\detect.py table --out "$out\nn_202306_jun_dep17" --min-score 0.95
```

`--min-score 0` keeps every detection the alarm rule makes. See `results/final-model-thresholds/thresholds.md` for what each cut-off gave on the tests.

### If you'd rather write locally and copy afterwards

```
python tools\detect.py run $nn --out "C:\RumbleResults\nn_202306_jun_dep17"
robocopy "C:\RumbleResults\nn_202306_jun_dep17" "$out\nn_202306_jun_dep17" /E /Z /R:3 /W:10
```
