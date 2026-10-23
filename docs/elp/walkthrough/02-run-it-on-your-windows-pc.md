# Running the elephant rumble detector on your Windows PC

You type everything below into **PowerShell** (Start menu → type "PowerShell").
Copy each grey box, paste it in, press Enter, and wait for it to finish before the next one.

---

## Step 0: Check whether your teammate already set it up (30 seconds)

Your FruitPunchAI folder has a `processing_log` from July 2025, so someone may already have run this on this PC. Try:

```
elephant-rumble-inference --help
```

- If it prints a long help message, **it's already installed**. Skip to Step 4.
- If it says *"not recognized"*, carry on with Step 1.

---

## Step 1: Install Python 3.11 (one time)

1. Go to https://www.python.org/downloads/release/python-3119/ and download **"Windows installer (64-bit)"**.
2. Run it and **tick "Add python.exe to PATH"** at the bottom of the first screen. That's important.
3. Close and reopen PowerShell, then check:

```
python --version
```
It should say `Python 3.11.x`.

> Why 3.11 and not the newest? This project is from 2024, and its audio library (torchaudio) dropped a feature it needs in later versions. 3.11 with the versions pinned below is the safe combination.

---

## Step 2: Make a project folder with its own Python "bubble" (one time)

This keeps the project's libraries separate from everything else on your PC.

```
cd "$HOME\Desktop\Amogh - Rumble Detection"
python -m venv elephant-env
.\elephant-env\Scripts\Activate.ps1
```

You should now see `(elephant-env)` at the start of the line.
**Every time you open a new PowerShell to use the tool, run the last two lines again** (`cd …` and `Activate.ps1`).

> If `Activate.ps1` gives a red error about "running scripts is disabled", run this once, then try again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

---

## Step 3: Install the libraries and the tool (one time)

**3a. PyTorch + torchaudio (pinned to a version that works with this code).**
Pick ONE of the two:

If your PC has an **NVIDIA graphics card** (much faster):
```
pip install torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```
Otherwise (CPU only, works everywhere, slower):
```
pip install torch==2.4.1 torchaudio==2.4.1
```

**3b. The elephant tool itself**, from your GitHub fork:
```
pip install https://github.com/amoghshetty7436/elephant-rumble-inference/archive/refs/heads/main.zip
```

**3c. Put the model file where the tool looks for it.** The original download site is offline, so we copy your `2024-07-03.pth` in by hand. Change the first path if your file is somewhere else:
```
$dest = "$HOME\.cache\torch\hub\fruitpunch_elephants"
New-Item -ItemType Directory -Force $dest
Copy-Item "$HOME\Desktop\Amogh - Rumble Detection\FruitPunchAI\<path-to>\2024-07-03.pth" $dest
```
(The other model, AVES, ~378 MB, downloads automatically the first time you run the tool. Its host is still online.)

---

## Step 4: Try it on one recording

Start with **1 hour** of audio so it's quick:

```
elephant-rumble-inference --limit-audio-hours=1 --save-raven --visualizations-per-audio-file=2 "C:\full\path\to\some_recording.wav"
```

What you get, in a new `output` folder:
- `some_recording.wav.raven.txt`: the list of detected rumbles (open it in Raven, or in Excel/Notepad).
- 2 `.png` pictures: spectrograms with detected rumbles tinted green.

If that works, drop `--limit-audio-hours=1` to process the full 24 hours.
Speed: about 20–30 s per 24 h file with an NVIDIA GPU, about 30 min on CPU.

---

## If something goes wrong

Copy the **last ~20 lines** of the red error text into the thread. Likely suspects:

| Error mentions… | What it means |
|---|---|
| `ffmpeg` / `StreamReader` / `libavutil` | torchaudio can't find the audio decoder it needs. Fixable; send me the error. |
| `numpy` version | Run `pip install "numpy<2"` and retry. |
| `0ape.com` / download failed | The model file isn't in the folder from Step 3c. Check that path. |
| `CUDA` | GPU driver issue. Use the CPU install from 3a instead. |
