# The full rerun (study.py): how to start it and what you get

## Start it (once, then leave the PC until Tuesday)

1. Put `study.py` and `cpu_speed.py` in your `tools` folder, next to `improve.py`. It is on the branch `claude/project-thread-ls3rsg` and in `retraining/study/` of this project. No existing file changes.
2. Plug the PC in. The script stops Windows from sleeping while it runs, but it can't stop Windows Update from restarting the PC, so pause updates for a week.
3. In PowerShell, in the repository folder:

```
python tools\study.py --finish-by "2026-10-27 06:00" ; python tools\study.py --finish-by "2026-10-27 06:00" ; python tools\cpu_speed.py
```

The second command only does something if the first one stopped early, for example after a crash: it picks up where the first left off. If everything already finished, it just rescores the tests.

The rerun plans to be completely finished by **Tuesday 06:00** on the PC's clock. Then the CPU speed test runs (well under an hour, see below), so everything is ready by about 07:00.

## The CPU speed test (cpu_speed.py)

It doesn't change the detector. It measures how fast the current detector runs on your CPU, on the longest Complex/General validation day:

1. Reading and resampling the file.
2. AVES on the CPU as the detector runs it (1-hour pieces), with 1, 4, 8, 16 and all threads, on a 1-hour excerpt.
3. Several files at once (all cores split into processes of 4 threads), which is how a folder would really be processed.
4. 5- and 10-minute pieces in batches, a possible speedup, with how closely the fingerprints match the normal ones.
5. 8-bit AVES weights, with how closely the fingerprints match.
6. The GPU, for comparison.

Results go in `Configurations\_cpu_speed\cpu_speed.md` (and `.json`, `run_log.txt`): the time per 24 h of audio for each test, and an estimate for ELP's target (4,500 recording-days on 30 cores in under 5 days). Run it again any time with `python tools\cpu_speed.py`, ideally with nothing else running.

## What it does, in order

| Stage | What | Experiments |
|---|---|---|
| 1 | First round: A, B, C, D, E, E2, F, G, H, each with its own tuned alarm rule | 9 |
| 2 | All 1,960 alarm rules on the first-round winner | 0 (no training) |
| 3 | Noise clips used 0, 1, 3 or 10 times | ~3 |
| 4 | Random forest frames per day: 0, 200, 1,000, 2,000, 3,000, 5,000 | ~5 |
| 5 | Gap frames per rumble frame: 0 (none), 0.25, 0.5, 1, 2, plus the first-round amount | ~5 |
| 6 | Listening window: all 10 sizes from 0.64 s to 12.16 s | ~9 |
| 7 | Augmentation: which sounds (rumbles, forest, both, plus a no-noise control), then how loud (+10, 0, −10 dB, mixed), then how many copies (1, 2, 4) | ~10 |
| 8 | Combinations of the best two options from stages 4 to 7 | up to 16 |
| 9 | Learning from its own mistakes: 1 to 5 rounds, with the cleaned mining | 5 |
| 10 | All 1,960 alarm rules on the final model | 0 |
| 11 | Tests (Test 1, Test 2, vehicle days), scored once, for the baseline, every stage winner and the final model | 0 |

**F and H** train on the practice days, so they can't be ranked on them. They are compared on the 15 held-out forest days they are validated on, against every setup that never trained on those days (C, E, G). The result is shown separately, and the run continues from the winner among the others.

**If time runs short,** it stops starting new experiments in later stages (combinations first). It always keeps about 3 hours for stage 10 and the tests. An experiment that fails (for example, out of memory) is written to `failed.txt` and skipped, and the run carries on.

## Where the results are

Everything is in `L:\ELP\Detectors\Other\Amogh\Dataset\Configurations\_study\`:

| File | What |
|---|---|
| `progress.txt` | Current stage and time left before the finish time |
| `study.md` / `study.csv` | Every experiment: score, false alarms at 80/85/90% recall, max recall, epochs, training time, data counts, inference time |
| `choices.json` | Every decision made, and why |
| `rules_rules.csv`, `rules_rules_final.csv`, `rules_round1_rule_*.csv` | Every alarm rule's score |
| `final.md` / `final.json` | Test results |
| `final_detections\` | Raven tables of the final model on each test set |
| `<experiment>\` | Models, training logs (every check), results, precision–recall curves |
| `run_log.txt`, `failed.txt` | Everything printed, and anything that failed |

Augmented fingerprints are kept in `Configurations\_embedding_cache\_augmented\` for later use.
