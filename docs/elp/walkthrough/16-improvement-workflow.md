# 16. The improvement workflow (plan, 20 Oct 2026)

**Goal:** starting from E2 (clearing + forest + noise), find the best combination of improvements to the FruitPunch detector, changing one thing at a time so we know what each change did.

**One command, run overnight:** `python tools\improve.py`. It resumes where it stopped if the PC restarts. Results go in `Configurations\_improve\`. It never writes to the L: drive originals.

## The rules every experiment follows

| Rule | What it means |
|---|---|
| Yardstick | False alarms per hour on the 59 validation days, at the cut-off that finds **80%** of rumbles there. Also shown: at 85% and 90%, the most recall the model reaches, F1. |
| 5 runs | Each experiment is trained 5 times (seeds 0–4), same 5-minute training budget for all. Scores are averaged; the min–max range shows how much is luck. |
| Keep rule | A change is kept only if it lowers false alarms by more than that luck range. If two options are within the range, the simpler / faster one wins. |
| Tests sealed | Test 1, Test 2 and the vehicle days are not looked at until the very end. |
| Speed | Every model's inference time is measured: classifier seconds per 24 h of audio (GPU and CPU), plus AVES time, plus model size. |

## The stages

| Stage | What changes | Options tried | Retraining? |
|---|---|---|---|
| 0. Prepare | Re-embed the training days as whole days (needed for stages 2–4) | – | No (≈1.5 h once, ≈38 GB) |
| Baseline | E2 with FruitPunch's settings, 3 runs | – | Yes |
| 1. Alarm rules | How probabilities become alarms | "on" level 0.5 / 0.45 / 0.4 / 0.35 / 0.3 / 0.25 / 0.2; minimum length 2–8 frames; smoothing 1, 3, 5, 7, 9 frames; joining alarms 0–3 frames apart; alarm score = mean / peak (≈2,000 combinations) | No: reuses the baseline models |
| 2a. Noise clips | How often each noise clip is used | ×1 (now), ×2, ×3, ×5, ×10, ×20 | Yes |
| 2b. Random forest "not rumble" frames | Frames per forest day | 0, 200 (now), 500, 1,000, 2,000 | Yes |
| 2c. Negatives per rumble | Balance of the training data | 1 : 1 (now), 1 : 2, 1 : 3, 1 : 5 | Yes |
| 3. Listening window | Seconds of audio the classifier sees at once | 0.64 (now), 1.9, 3.2, 4.5, 5.8, 7.0, 8.3, 9.6, 10.9, 12.2 s | Yes |
| 4. Combinations | Every combination of the best 2 options of stages 2a, 2b, 2c and 3 (up to 16) | – | Yes |
| 5. Its own mistakes | On the combination winner: add its false alarms on training days as "not rumble" | 1, 2, 3 rounds | Yes |
| 6. Final alarm rules | Re-tune the alarm rules for the final model | – | No |

Each stage starts from the winner of the previous one, and the alarm rules from stage 1 stay locked until stage 6.

## What you get

- `improve.md`: one row per experiment (stage, what changed, false alarms at 80/85/90% recall, max recall, F1, luck range, inference time, kept or not), plus a log of every choice and why.
- Per experiment: the 3 models, the precision–recall curve (table and chart), training log.
- `progress.txt`: what is running and an estimated finish time, updated as it goes.

## At the end: `python tools\improve.py final`

- Scores the baseline and each stage's winner on Test 1, Test 2 and the vehicle days, once, giving a "what each step added" table.
- Shows the final model's full precision–recall curve so the working cut-off can be chosen (with a small safety margin), and writes its detections as Raven tables.

## Time

About 1.5 h embedding + about 25 stage experiments, up to 16 combinations and 3 hard-negative rounds × about 30 min each (5 runs) + alarm-rule tuning ≈ 20–25 hours. Started Tuesday evening, it should finish by Thursday, ahead of Friday. `progress.txt` shows the live estimate.

## Known limitation

Changing one thing at a time can miss a combination that only works together. Stage 4 covers this by training every combination of the top 2 options from stages 2a–3.
