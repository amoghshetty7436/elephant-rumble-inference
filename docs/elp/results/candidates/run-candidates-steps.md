# Running the candidate models on the lab PC

Use the same Python (venv) you run `study.py` with, from `C:\Users\avs97\code\eri`.
The study's final model must have finished: `_study\choices.json` needs a `"final"` entry.

## 1. Get the new code

```powershell
cd C:\Users\avs97\code\eri
git pull origin main
```

## 2. Check the packages

```powershell
python -c "import torch, torchvision, scipy, dasheng; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

It should print versions and `True`. If something is missing:

- `scipy` or `dasheng`: `python -m pip install scipy einops` then `python -m pip install dasheng --no-deps`
  (`--no-deps` stops pip from swapping your GPU torch for a CPU one).
- `torchvision`: install the one that matches your torch, from the cu121 index, e.g. for torch 2.5.1:
  `python -m pip install torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --no-deps`
  (torch 2.4 → 0.19.x, 2.3 → 0.18.x, 2.2 → 0.17.x, 2.1 → 0.16.x).

Run the check again until it prints `True`.

## 3. Start everything (one command)

```powershell
python tools\candidates.py
```

Add `--source <dataset folder>` if your dataset is not at `L:\ELP\Detectors\Other\Amogh\Dataset`.

It runs, in order: audio features (models 5 and 7), model 4, 1, 2, 7, 5, then BirdAVES
fingerprints and model 6, Dasheng fingerprints and model 6, then the tests. It downloads
ImageNet ResNet-18, BirdAVES (~380 MB) and Dasheng (~350 MB) the first time.

Expected time is roughly 35 to 45 hours, so it should finish by Monday night.

## 4. While it runs

- Progress: `<dataset>\Configurations\_candidates\progress.txt`
- Results so far: `_candidates\candidates.md` (validation score per model, next to the final model)
- If the PC restarts or it stops, run the same command again. Finished parts are skipped.
- If one model fails, the others still run; the error is in `_candidates\<model>\failed.txt`.
- To run only some: `python tools\candidates.py --only m4 m1` (names: m1 m2 m4 m5 m6 m7).

## 5. Send me

`_candidates\candidates.md` and `_candidates\final.md` (the tests), plus any `failed.txt`.
