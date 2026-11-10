# Running the speed-up experiment on the lab PC

Start this **after round 2 has finished** (its window is back at the prompt). Same PC, venv and folder.

```powershell
cd C:\Users\avs97\code\eri
git pull origin main
$py = "C:\Users\avs97\venvs\etm\Scripts\python.exe"
& $py tools\speedup.py
```

- It tries AVES at 16x, 24x, 48x, 64x and Dasheng at 8x, 32x, one after another, each with 5 runs plus their average,
  scored on the validation days and then on the tests. Rough guess: a day in total.
- **Disk:** each speed's fingerprints take about 80 GB under `Configurations\_embedding_cache_<speed>`, and are deleted
  once that speed's tests are scored. Have about 100 GB free on L:. If space is short, the 140 GB folder
  `_embedding_cache_dasheng06b` from round 2 can be deleted once round 2's tests are done.
- It resumes if stopped. To run only some speeds: `& $py tools\speedup.py --speeds aves16 aves64`.

## Results

`L:\ELP\Detectors\Other\Amogh\Dataset\Configurations\_speedup\`
- `speedup.md`: validation, best first, next to the usual 32x AVES and 16x Dasheng
- `speedup_final.md`: the tests
- `progress.txt`, `failed.txt` (only if something failed)
