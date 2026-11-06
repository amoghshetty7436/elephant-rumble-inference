# Elephant Rumble Detector

This program listens to your recordings and finds **forest elephant rumbles**. For every
recording it makes a **Raven selection table** with one box for each rumble it finds, so you can
check them in Raven Pro.

It works on Windows computers with or without a graphics card. You don't need to know any
programming: you double-click a file, pick your recordings in a window, and press **Start**.

---

## What's in this folder

| File | What it is for |
|---|---|
| **1 - Install.bat** | Sets the detector up. Run it **once** on each computer. |
| **2 - Find rumbles.bat** | Opens the detector's window. Use it every time you have recordings to check. |
| README.md | These instructions. |
| rumble_detector.py, rumble_gui.py, model\\, requirements.txt | The detector itself. Don't move or change these. |

**Keep the whole folder together.** Copy it to your own computer (for example to the Desktop)
rather than running it from the network drive.

---

## Step 1: Install (once per computer, 5 to 20 minutes)

1. **Unzip the folder first.** Right-click `ElephantRumbleDetector.zip`, choose **Extract All...**,
   and pick a place on this computer (for example the Desktop). Double-clicking the files while
   they are still inside the zip doesn't work.
2. Make sure the computer is connected to the internet.
3. Double-click **`1 - Install.bat`**.
   - If Windows shows a blue box saying "Windows protected your PC", click **More info**, then
     **Run anyway**.
4. Wait. A black window shows what is happening. It:
   - installs Python 3.11 if the computer doesn't have it,
   - checks whether the computer has an NVIDIA graphics card and installs the matching version
     (about 2.5 GB with a graphics card, 200 MB without),
   - downloads the AVES sound model (378 MB),
   - runs a short test.
5. When it says **INSTALLED**, press any key to close the window.

If it says **THE INSTALL DID NOT FINISH**, check the internet connection and double-click
`1 - Install.bat` again, because it picks up where it stopped. If it fails a second time, take a screenshot of
the window and send it to Amogh.

> If the installer had to install Python, it may ask you to close the window and run
> `1 - Install.bat` once more. That's normal.

---

## Step 2: Find rumbles

1. Double-click **`2 - Find rumbles.bat`**. The "Find elephant rumbles" window opens.

2. **Recordings:** click **Folder...** (or **One file...**) and pick what to analyse:

   | You choose | Example | It analyses |
   |---|---|---|
   | A whole deployment (a folder with one subfolder per site) | `L:\ELP\Projects\Nouabale\nn_sounds\nn_202306_jun_dep17` | every recording of every site |
   | One site folder | `L:\ELP\Projects\Nouabale\nn_sounds\nn_202306_jun_dep17\nn03g_202306_jun` | every recording of that site |
   | One recording | `...\nn03g_202306_jun\nn03g_20230825_134525.wav` | just that recording |

   You can also paste a path into the box, or drag a folder or file onto `2 - Find rumbles.bat`.

3. **Save results in:** it fills in `Desktop\Rumble Results\<name of what you chose>`. Keep it, or
   click **Folder...** to pick another place (it can be on the network drive, but not inside the
   recordings folder).

4. The grey line under it says whether the **graphics card** or the **processor** will be used.
   Without a graphics card there is also a **Cores to use** box: leave it at the maximum, or lower
   it (for example to half) to keep the computer free for other work.

5. Press **Start**. Two bars show the progress:
   - **All recordings:** how many recordings and hours of audio are done, and the time left.
   - **This recording:** the recording being analysed right now.

   The box at the bottom lists each finished recording and how many rumbles it found.
   **Keep the computer plugged in.** The detector stops Windows from going to sleep, but don't
   shut the computer down or let Windows Update restart it.

6. When it's finished it asks whether to open the results folder (the **Open results folder**
   button does the same at any time).

### How long does it take?

The time depends on the computer and on how long the recordings are:

| Computer | One 24-hour recording | One month of one site (30 days) |
|---|---|---|
| Desktop with an NVIDIA graphics card (e.g. RTX 4060) | about 30 to 60 seconds | about 20 to 30 minutes |
| Laptop or desktop without a graphics card | about 5 to 15 minutes (4 to 8 cores) | about 3 to 8 hours |

A whole deployment of 40 sites x 3 months takes days on a processor only. Use a computer with a
graphics card for big jobs if you can. Reading the recordings over the network (L:) can also
slow it down; if it's very slow, copy the recordings to the computer first.

### Stopping and carrying on

Press **Stop** (or close the window) at any time. Next time, choose **the same recordings and the
same results folder** and press Start: it skips everything already done and carries on. It also carries on by
itself if:

- **the network drive stops responding:** it waits up to 10 minutes for the drive to come back.
  If the drive is still gone, it stops and asks you to start again later.
- **it gets stuck** (nothing happens for 20 minutes): it restarts by itself.

---

## What you get

The results folder has the same subfolders as your recordings (one per site). Start with
**`ABOUT THESE RESULTS.txt`** and **`summary.md`**.

| File | What it is |
|---|---|
| **`<site>\_all_rumbles.txt`** | **One Raven table for the whole site**, with every rumble from all its recordings, one recording after another. This is usually the one you want. |
| `<site>\<recording>.rumbles.txt` | The Raven table of one recording. |
| `<site>\<recording>.probs.npy` | The detector's raw scores. You don't need to open these; they make changing the cut-off quick. |
| **`all_rumbles.csv`** | **Every rumble in one list, for Excel** (double-click it): site, recording, **date and clock time** (worked out from the recording's file name), start and end in the recording, score, and which Raven table it is in. |
| **`summary.md`** / `summary.csv` | Per site: recordings done, hours, rumbles found, rumbles per day. `summary.csv` opens in Excel and `summary.md` in Notepad. |
| `failed.txt` | Only there if some recordings could not be read, with the reason for each. |
| `run_log.txt` | Everything the window showed. Send this to Amogh if something goes wrong. |
| `timings.csv`, `detector.json` | Details of each recording's run, and the settings and computer used. |

### Opening the results in Raven Pro

The tables are tab-separated `.txt` files, the same format as ELP's hand-made ground-truth tables,
because that is what Raven opens. (Excel opens them too: right-click > Open with > Excel. For
Excel, `all_rumbles.csv` is usually handier.)

- **One recording:** open the recording in Raven (*File > Open Sound Files*), then
  *File > Open Selection Table* and pick its `.rumbles.txt`.
- **A whole site:** open all of the site's recordings together (*File > Open Sound Files*, select them all,
  and choose to open them as one file sequence), then open `_all_rumbles.txt`.

Each box covers 5 to 250 Hz, so check the **times** rather than the frequencies. The **Score**
column is how sure the detector is, from 0 to 1.

### Recordings it couldn't use

The run doesn't stop for a bad file. It writes the file to `failed.txt` and `summary.md` with a reason, and
carries on:

| Reason in failed.txt | What it means | What to do |
|---|---|---|
| the file is empty (0 bytes) | The recorder or the copy produced an empty file. | Nothing to analyse; check the original card if it matters. |
| not a readable sound file | The file is damaged or wasn't fully copied. | Copy it again from the original and run again. |
| the file is no longer there | It was moved, renamed or deleted during the run, or the drive disconnected. | Run again: only the missing recordings are retried. |
| no permission to read it | Your account can't open that file or folder. | Ask for access, then run again. |

Recordings shorter than about 2 seconds (very small files, often about 50 bytes) aren't errors.
They get an empty table and are counted in `summary.md` as "too short to analyse".

If a whole **subfolder** can't be opened, the window shows a **WARNING** with its name, and its
recordings are left out of that run. Run again once the folder is reachable.

---

## Changing the cut-off (optional)

Each detection has a Score. By default the tables keep detections scoring **0.993 or more**.
On ELP's validation days, that setting found 80% of the rumbles.

- **Lower** cut-off (e.g. 0.95): finds more rumbles, but gives more false alarms to check.
- **Higher** cut-off (e.g. 0.999): fewer false alarms, but misses more rumbles.

To change it for results you already have, click **Change cut-off...**
in the window, choose the results folder, type the new number and click **Rewrite the tables**. It takes seconds and doesn't need the
recordings or a graphics card. You can always go back to 0.993.

What each cut-off gave with this model on ELP's test days. "Found" is the share of the hand-marked
rumbles it found (recall); "real" is the share of its detections that were rumbles (precision).
Sample A is 54 Nouabalé-Ndoki forest days; Sample B is 2024's additional test days; the vehicle
days have no elephants, so every detection there is a false alarm. (The same numbers are in
`model\test_results.csv`, and the window shows them when you pick a cut-off.)

From 0.5 to 0.95 the share of real detections rises a lot while few rumbles are lost; above
0.95 each small step trades rumbles found for fewer false alarms.

| Cut-off | Sample A: found / real | Sample A false alarms per hour | Sample B: found / real | Sample B false alarms per hour | Vehicle days false alarms per hour |
|---|---|---|---|---|---|
| 0.5 | 92% / 20% | 2.10 | 98% / 14% | 2.02 | 4.61 |
| 0.55 | 92% / 23% | 1.83 | 98% / 16% | 1.76 | 4.24 |
| 0.6 | 92% / 25% | 1.58 | 98% / 18% | 1.52 | 3.72 |
| 0.65 | 92% / 28% | 1.36 | 98% / 20% | 1.28 | 3.25 |
| 0.7 | 92% / 32% | 1.14 | 98% / 23% | 1.08 | 2.78 |
| 0.75 | 92% / 36% | 0.93 | 98% / 28% | 0.85 | 2.25 |
| 0.8 | 91% / 41% | 0.76 | 98% / 32% | 0.67 | 1.92 |
| 0.85 | 90% / 48% | 0.57 | 97% / 39% | 0.50 | 1.46 |
| 0.9 | 89% / 57% | 0.39 | 97% / 46% | 0.36 | 1.04 |
| 0.95 | 86% / 70% | 0.21 | 95% / 59% | 0.21 | 0.68 |
| 0.96 | 85% / 75% | 0.16 | 95% / 63% | 0.17 | 0.58 |
| 0.97 | 84% / 76% | 0.14 | 94% / 69% | 0.13 | 0.46 |
| 0.98 | 81% / 81% | 0.10 | 93% / 74% | 0.10 | 0.35 |
| 0.99 | 76% / 86% | 0.07 | 91% / 80% | 0.07 | 0.18 |
| **0.993 (standard)** | **74% / 88%** | 0.06 | **90% / 83%** | 0.05 | 0.15 |
| 0.995 | 73% / 90% | 0.04 | 89% / 84% | 0.05 | 0.10 |
| 0.997 | 71% / 92% | 0.03 | 88% / 88% | 0.03 | 0.08 |
| 0.998 | 69% / 93% | 0.03 | 86% / 89% | 0.03 | 0.07 |
| 0.999 | 65% / 95% | 0.02 | 85% / 92% | 0.02 | 0.06 |

---

## Common problems

| What you see | What to do |
|---|---|
| "The detector is not installed on this computer yet" | Run `1 - Install.bat` first. |
| A black window with questions opens instead of the detector's window | The computer's Python has no window toolkit (tkinter). It still works: answer the questions in the black window (paste paths with a right-click). |
| "Can't find a folder or sound file at ..." | The path is wrong or the drive isn't connected. Check that you can open it in File Explorer, then paste it again. |
| "No sound files found there" | The folder has no `.wav` / `.flac` / `.aif` files, even in its subfolders. |
| It says it runs "on the processor" but the computer has a graphics card | Only NVIDIA cards are used. If it's NVIDIA, update its driver (nvidia.com/drivers), delete the `.venv` folder and run `1 - Install.bat` again. |
| The computer becomes slow while it runs (no graphics card) | Press **Stop**, lower **Processor cores to use**, and press **Start** again. It carries on where it stopped. |
| Anything else | Send `run_log.txt` from the results folder (or a screenshot) to Amogh. |

---

## For the technical person

- **Model:** the improved FruitPunch AI detector `E2_ctx8_noise10_rand3000_gap0.5_hard0_augnone`,
  run 0 (`model\rumble_model.pth`, settings in `model\settings.json`). AVES (`aves-base-bio`)
  embeddings of audio resampled to 500 Hz (scipy `resample_poly`, any input sample rate),
  one per 0.64 s, then a classifier over 17 neighbouring frames (about 11 s). Alarm rule:
  frames over 0.2 after 5-frame smoothing, at least 3 frames, gaps of up to 3 frames joined,
  score = highest frame probability, kept if the score is at least 0.993. This is the same model, rule and
  cut-off that were run on `nn_202306_jun_dep17` and `dz_202512_dec_dep10` with `tools\detect.py`.
  Checks before sharing: on Amogh's recording `nn03g_20230825_134525.wav` the package's
  probabilities match the lab run's to within 0.0014 and the Raven table is identical, and on the
  public Sample A day `nn06a_20191008` it found 140 of the 152 hand-marked rumbles, with 59 of
  its 61 detections real (CPU, 3 min 13 s for the 24 hours on 4 cores).
- **The window** is `rumble_gui.py` (tkinter, part of Python). It runs `rumble_detector.py run`
  in the background, restarts it if it stalls, and shows its output. `rumble_detector.py` with
  no arguments asks the same questions in the console instead.
- **Command line** (inside the folder, with `.venv\Scripts\python.exe`):

  ```
  python rumble_detector.py run "L:\path\deployment" "L:\path\other" --out "D:\results" --cores 8
  python rumble_detector.py run "L:\path\site" --out "D:\results" --device cpu
  python rumble_detector.py table --out "D:\results" --min-score 0.95
  python rumble_detector.py setup
  ```

  `--device auto|gpu|cpu` (default auto), `--cores N`, `--redo` (redo finished recordings),
  `--stall-minutes 20` (exit code 3 when stuck; the double-click launcher restarts it),
  `--prefetch 2` (recordings read ahead while the GPU works). Exit code 4 means the recordings
  folder became unreachable.
- **PyTorch builds the installer picks:** torch 2.4.1 + CUDA 12.1 for most NVIDIA cards,
  torch 2.7.1 + CUDA 12.8 for RTX 50xx (Blackwell), torch 2.4.1 CPU otherwise. If the
  graphics card fails the install test, the installer writes `use_cpu.txt` and the detector
  uses the processor; delete that file to try the card again.
- **No internet on the computer:** copy `aves-base-bio.torchaudio.pt` and
  `aves-base-bio.torchaudio.model_config.json` into `model\aves\` from a computer where the
  install worked. (Python and PyTorch still need to be installed once.)
