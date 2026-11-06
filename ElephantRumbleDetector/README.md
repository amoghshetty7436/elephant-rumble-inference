# Elephant Rumble Detector

This program listens to your recordings and finds **forest elephant rumbles**. For every
recording it makes a **Raven selection table** with one box for each rumble it finds, so you can
check them in Raven Pro.

It works on Windows computers with or without a graphics card. You don't need to know any
programming: you double-click a file, then paste a folder path.

---

## What's in this folder

| File | What it is for |
|---|---|
| **1 - Install.bat** | Sets the detector up. Run it **once** on each computer. |
| **2 - Find rumbles.bat** | Runs the detector. Use it every time you have recordings to check. |
| 3 - Change cut-off (optional).bat | Makes finished results stricter or looser without running again (see below). |
| README.md | These instructions. |
| rumble_detector.py, model\\, requirements.txt | The detector itself. Don't move or change these. |

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

1. Double-click **`2 - Find rumbles.bat`**. A black window opens and asks three questions.

2. **"Recordings:"** paste the location of your recordings. You can give it any of these:

   | You give it | Example | It analyses |
   |---|---|---|
   | A whole deployment (a folder with one subfolder per site) | `L:\ELP\Projects\Nouabale\nn_sounds\nn_202306_jun_dep17` | every recording of every site |
   | One site folder | `L:\ELP\Projects\Nouabale\nn_sounds\nn_202306_jun_dep17\nn03g_202306_jun` | every recording of that site |
   | One recording | `...\nn03g_202306_jun\nn03g_20230825_134525.wav` | just that recording |

   **How to copy a path:** open the folder in File Explorer, click in the address bar at the top,
   press **Ctrl+C**. In the black window, **right-click** to paste, then press **Enter**.
   To copy the path of one file, hold **Shift**, right-click the file, and choose **Copy as path**.

   *Shortcut:* you can also drag a folder or a file onto `2 - Find rumbles.bat`.

3. **"Results folder:"** press **Enter** to save the results to
   `Desktop\Rumble Results\<name of what you gave it>`, or paste another folder (it can be on
   the network drive). Results are never saved inside the recordings folder.

4. **"Cores:"** (only asked on computers without an NVIDIA graphics card; with one, the graphics
   card does the heavy work and this question is skipped). Press **Enter** to let the detector
   use the whole processor. If you want to keep using the computer for other work while it
   runs, type a smaller number (for example half of what it says the computer has).

5. Leave it running. Each line shows a finished recording, how many rumbles it found and the
   time left. **Keep the computer plugged in.** The detector stops Windows from going to sleep, but
   don't shut the computer down or let Windows Update restart it.

6. When it says **FINISHED**, the results folder opens by itself.

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

You can close the window at any time. Next time, give it **the same recordings and the same
results folder**: it skips everything already done and carries on. It also carries on by
itself if:

- **the network drive stops responding:** it waits up to 10 minutes for the drive to come back.
  If the drive is still gone, it stops and asks you to start again later.
- **it gets stuck** (nothing happens for 20 minutes): it restarts by itself.

> Don't click inside the black window while it runs. The detector turns off the Windows setting
> that pauses programs when you click, but if the window ever seems frozen, press **Esc** once.

---

## What you get

The results folder has the same subfolders as your recordings (one per site). Start with
**`ABOUT THESE RESULTS.txt`** and **`summary.md`**.

| File | What it is |
|---|---|
| **`<site>\_all_rumbles.txt`** | **One Raven table for the whole site**, with every rumble from all its recordings, one recording after another. This is usually the one you want. |
| `<site>\<recording>.rumbles.txt` | The Raven table of one recording. |
| `<site>\<recording>.probs.npy` | The detector's raw scores. You don't need to open these; they make changing the cut-off quick. |
| **`summary.md`** / `summary.csv` | Per site: recordings done, hours, rumbles found, rumbles per day. `summary.csv` opens in Excel and `summary.md` in Notepad. |
| `failed.txt` | Only there if some recordings could not be read, with the reason for each. |
| `run_log.txt` | Everything the window showed. Send this to Amogh if something goes wrong. |
| `timings.csv`, `detector.json` | Details of each recording's run, and the settings and computer used. |

### Opening the results in Raven Pro

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

To change it for results you already have, double-click **`3 - Change cut-off (optional).bat`**,
paste the results folder, and type the new number. It takes seconds and doesn't need the
recordings or a graphics card. You can always go back to 0.993.

What each cut-off gave on ELP's test days (Nouabalé "Sample A"):

| Cut-off | Rumbles found | Detections that are real rumbles | False alarms per hour |
|---|---|---|---|
| 0.95 | 87% | 67% | 0.24 |
| 0.99 | 80% | 84% | 0.08 |
| 0.993 (standard) | about 78% | about 86% | about 0.07 |
| 0.999 | 68% | 93% | 0.02 |

---

## Common problems

| What you see | What to do |
|---|---|
| "The detector is not installed on this computer yet" | Run `1 - Install.bat` first. |
| "Can't find a folder or sound file at ..." | The path is wrong or the drive isn't connected. Check that you can open it in File Explorer, then paste it again. |
| "No sound files found there" | The folder has no `.wav` / `.flac` / `.aif` files, even in its subfolders. |
| It says it runs "on the processor" but the computer has a graphics card | Only NVIDIA cards are used. If it's NVIDIA, update its driver (nvidia.com/drivers), delete the `.venv` folder and run `1 - Install.bat` again. |
| The computer becomes slow while it runs | Close it, start again, and type fewer **Cores**. It carries on where it stopped. |
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
