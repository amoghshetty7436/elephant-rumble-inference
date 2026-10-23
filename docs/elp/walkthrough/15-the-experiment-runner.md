# The experiment runner (`tools\experiments.py`)

**What it does:** it starts from the data setup that `dataset_configs.py` picks as best, and finds the best model settings on top of it. It works in stages, keeps the winner of each stage, and judges everything **only on the validation days** (RumbleComplex + RumbleGeneral). Sample A and Sample B are not looked at until you run `final`.

It is tested only on small made-up recordings so far (14 tests pass). The real run on your PC is the first time it meets real data.

---

## What it tries

| Stage | Experiments | Question |
|---|---|---|
| 1. Context | the classifier sees 1 fingerprint (FruitPunch), or **~6 s, ~11 s, ~21 s** around each slice | Does more context help? |
| 2. Negatives | on the stage-1 winner: **1 or 5** "not rumble" frames per rumble frame × **0, 1 or 2 rounds** of hard negatives | Do more and harder negatives cut false alarms? |
| 3. Recheck | the stage-1 runner-up with the stage-2 winner's settings | Did the stages interact? |

**For every experiment:**
- It trains **3 seeds**, for 5 minutes each.
- It tunes the **decision rules** on validation: smoothing (1, 3, 5 or 7 slices), joining gaps (0, 1 or 2 slices), minimum length (4, 5 or 6 slices), the Score (mean, top-3 or peak) and the threshold. That's 108 rule sets, each with every threshold. FruitPunch's own rule is one of them, so the table also shows how much the rules alone add.
- **Picking a winner:** the best mean F1 wins. But if a simpler experiment is within one seed-spread of it, the simpler one wins, because a difference that small could just be luck.
- **Hard negatives:** the previous model's false alarms on training days. False alarms within 10 s of a labelled rumble, or scoring ≥ 0.99, are skipped, since they might be real rumbles nobody marked. Mined frames are at most a third of the negatives.

---

## How to run it (after the split run has finished)

1. Copy `experiments.py` from `retraining\experiments\` in the project files into your `tools\` folder, next to `dataset_configs.py`. It's also on GitHub, on branch `claude/project-thread-xdnlff`.
2. In VS Code's terminal:
   ```
   .venv\Scripts\activate
   python tools\experiments.py
   ```
   It reads `best_choice.txt` to find the winning setup. You can override it with `--base E`, for example.
3. Leave it overnight, with sleep set to Never as before. If it stops, run the same command again: finished experiments are kept.
4. In the morning, open `Configurations\_experiments\experiments.md`. It has one row per experiment, the ± spread between seeds, and the choices it made and why.
5. When you're happy with the chosen experiment, run the tests **once**:
   ```
   python tools\experiments.py final
   ```
   That writes `final\summary.md` (Sample A, Sample B, Vehicle) and the Raven detection tables in the chosen experiment's folder.

## What it needs
- **Disk:** the first step re-embeds the training days as whole days. Context windows and hard negatives need every slice, but the split run only kept the slices it trained on. That's about **200 MB per day**, so ~38 GB if the winner trains on clearing + forest, and ~27 GB for forest only. It goes into the same `_embedding_cache`.
- **Time:** my estimate is about **1 hour of embedding, then 6–8 hours** for all the stages. It prints progress as it goes. I can't measure this until it runs on your PC.
- **Memory:** a few GB of RAM and GPU memory for the training fingerprints.

## Files it writes (all under `Configurations\_experiments\`)
- `experiments.md` / `.csv`: the comparison table.
- `choices.txt`: what each stage picked and why.
- `chosen.txt`: the overall pick.
- `<experiment>\`: models (`model_seed0.pth` …), `results.json`, `training_log.txt`, `mined.json` (hard negatives), and `final\` after the final step.

Models without context (`ctx0_…`) are in FruitPunch's `.pth` format and work with the existing tools. Context models need `experiments.py` to run them.
