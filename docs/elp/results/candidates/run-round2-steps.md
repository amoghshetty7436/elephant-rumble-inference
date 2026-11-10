# Running round 2 on the lab PC

Start this once the deployment run has finished. Use the same venv as `candidates.py`
(`C:\Users\avs97\venvs\etm`), from `C:\Users\avs97\code\eri`.

## 1. Get the new code

```powershell
cd C:\Users\avs97\code\eri
git pull origin main
```

## 2. Run (one line; both parts resume if stopped)

```powershell
python tools\candidates.py --only m5 m7; python tools\round2.py
```

- The first part runs models 5 and 7, which crashed last time (about 6 h).
- The second part is round 2, roughly 35–50 h with everything. It downloads the ImageNet
  weights for EfficientNet-B0 and ConvNeXt-Tiny, and Dasheng 0.6B (about 2.5 GB), the first time.
- **Disk:** Dasheng 0.6B's fingerprints need about 350 MB per recorded day, roughly 140 GB
  in total, under `Configurations\_embedding_cache_dasheng06b`. Check there's room on L: first.
- To finish sooner, leave parts out:
  - `python tools\round2.py --skip dasheng06b` drops the big encoder (saves the most time and all of that disk).
  - `--archs resnet18` uses one checker instead of three.

Keep both PowerShell windows open, including the keep-awake one, as last time.

## 3. Where the results are

In `<dataset>\Configurations\_round2\`:

- `progress.txt`: what it's doing now
- `round2.md`: validation scores, best first, with the inference cost of each
- `round2_final.md`: the tests, once everything has been scored
- `failed.txt`: only if something failed (the rest keeps going)

Send me `round2.md`, `round2_final.md` and, if they exist, `failed.txt` and
`_candidates\candidates.md`.
