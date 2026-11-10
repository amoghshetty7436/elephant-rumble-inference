# Running EarthToolsMaker and DeepForestSound on our tests: step by step

All commands are for the Windows command prompt (cmd) on the lab PC. Paths you can
change are `C:\Users\avs97\code`, `C:\Users\avs97\venvs` and `C:\Users\avs97\baselines`; keep them the same in every step.

## Step 1. Install Git (once)

Steps 3 and 4 need Git, plus Git LFS for DeepForestSound's model files. Install
Git for Windows from https://git-scm.com/download/win (the defaults are fine; it comes
with Git LFS). Close and reopen the terminal, then check:

```bat
git --version
git lfs install
```

## Step 2. Get the new code

Your copy of the repo was downloaded as a zip, so it can't be updated with git. Either:

- download the branch as a zip from
  https://github.com/amoghshetty7436/elephant-rumble-inference/archive/refs/heads/main.zip
  (signed in to GitHub), unzip it, and copy its `tools\baselines` folder into your
  existing `tools` folder; or
- clone it fresh:
  `git clone -b main https://github.com/amoghshetty7436/elephant-rumble-inference C:\Users\avs97\code\eri`
  and run the later steps from `C:\Users\avs97\code\eri`.

Either way you should end up with a `tools\baselines\` folder holding `compare.py`,
`run_etm.py`, `run_dfs.py` and `README.md`.

## Step 3. Download the two detectors

```bat
mkdir C:\Users\avs97\code
git clone https://github.com/earthtoolsmaker/forest-elephants-rumble-detection C:\Users\avs97\code\etm
git clone https://github.com/MNHN-OFVI/DeepForestSound C:\Users\avs97\code\dfs
```

Check that the DeepForestSound model is real and not a small placeholder:

```bat
dir C:\Users\avs97\code\dfs\model_weights
```

`DFS_LF.pth` should be hundreds of MB. If it is only a few hundred bytes, run
`cd C:\Users\avs97\code\dfs` then `git lfs pull`.

## Step 4. Make a Python environment for each detector

This PC has no Python 3.11, so uv fetches one (no admin rights needed). Use the
Python you normally run `study.py` with:

```bat
python -m pip install uv
python -m uv venv -p 3.11 C:\Users\avs97\venvs\etm
python -m uv venv -p 3.11 C:\Users\avs97\venvs\dfs
python -m uv pip install -p C:\Users\avs97\venvs\etm torch torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m uv pip install -p C:\Users\avs97\venvs\etm ultralytics opencv-python soundfile pandas
python -m uv pip install -p C:\Users\avs97\venvs\dfs torch==2.1.0+cu121 torchaudio==2.1.0+cu121 torchvision==0.16.0+cu121 accelerate==0.29.3 timm==0.4.5 peft==0.10.0 transformers==4.40.2 "numpy<2" soundfile pandas wget matplotlib --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
```

These environments have no `pip` of their own; always install through `python -m uv pip install -p ...`.

Check that both can see the GPU (each should print `True`):

```bat
C:\Users\avs97\venvs\etm\Scripts\python -c "import torch; print(torch.cuda.is_available())"
C:\Users\avs97\venvs\dfs\Scripts\python -c "import torch; print(torch.cuda.is_available())"
```

## Step 5. List the sound files

Back in the repo folder, using the Python you normally use for `dataset_configs.py`
and `study.py`:

```bat
python tools\baselines\compare.py list --out C:\Users\avs97\baselines
```

This writes `C:\Users\avs97\baselines\files.tsv`, listing every validation, Sample A, Sample B and
vehicle recording. If your dataset is not at `L:\ELP\Detectors\Other\Amogh\Dataset`,
add `--source <your dataset folder>`.

## Step 6. Run EarthToolsMaker

```bat
C:\Users\avs97\venvs\etm\Scripts\python tools\baselines\run_etm.py --etm-repo C:\Users\avs97\code\etm --list C:\Users\avs97\baselines\files.tsv --out C:\Users\avs97\baselines\etm
```

About a minute or two per recording day. It prints one line per day.

## Step 7. Run DeepForestSound

```bat
C:\Users\avs97\venvs\dfs\Scripts\python tools\baselines\run_dfs.py --dfs-repo C:\Users\avs97\code\dfs --list C:\Users\avs97\baselines\files.tsv --out C:\Users\avs97\baselines\dfs
```

This one is slower. It runs on the GPU, but on a CPU it would take about 45 minutes per
day. Leave it running (overnight is fine). If it stops, run the same command again: days
already done are skipped. Steps 6 and 7 can run at the same time in two cmd windows.

## Step 8. Score everything

Using your normal Python again:

```bat
python tools\baselines\compare.py score --out C:\Users\avs97\baselines --ours <your study folder>\_study\final.json
```

Point `--ours` at the `final.json` that sits next to the `final.md` of our rerun, so our
final model's numbers appear in the same report. (Add `--source` here too if you used
it in step 5.)

## Step 9. Send me the results

Upload `C:\Users\avs97\baselines\baselines.md`; `baselines.json` is useful too. I'll turn them into
the comparison with our detector.

## If something goes wrong

- **"... is a Git LFS pointer"**: go back to step 3 and run `git lfs pull`.
- **`torch.cuda.is_available()` prints False**: reinstall torch in that environment with
  the `cu121` command from step 4.
- **"not every listed file has been run yet, skipped"** in step 8: step 6 or 7 hasn't
  finished. Run it again and it will finish the missing days.
- Anything else: paste me the last 20 lines of the error.
