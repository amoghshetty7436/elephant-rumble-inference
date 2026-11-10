# Teaching AVES about rumbles (finetune.py)

The step after the full rerun. Start it on Tuesday, once the rerun and the CPU test have finished. It starts from the rerun's final model and asks whether changing AVES itself helps. Background with diagrams: the "Tuning AVES for Rumbles" page.

## Start it

1. Copy `finetune.py` into `tools`, next to `study.py`. It's on GitHub and in `retraining/study/` of this project. It imports `study.py`, `improve.py`, `experiments.py` and `dataset_configs.py`, and changes none of them.
2. In the same PowerShell window and folder as the rerun:

```
python tools\finetune.py ; python tools\finetune.py
```

The second command only does something if the first stopped early. It resumes and retrains nothing that is already done.

It needs the rerun's `_study\choices.json` with a final choice, and stops with a message if the rerun hasn't finished.

## What it does

| Step | What | Trained |
|---|---|---|
| Caching | Output of AVES layers 8 and 10 for every training and validation day, stored like the fingerprint cache | - |
| A: probing | The rerun's recipe trained on layer 8, 10 and 12 fingerprints | the classifier |
| C: top 2 | AVES layers 11-12 plus the classifier | ~14 M weights |
| C: top 4 | AVES layers 9-12 plus the classifier | ~28 M weights |
| D: LoRA | Rank 4 and rank 8 adapters on the attention of layers 9-12, plus the classifier; AVES unchanged | ~0.1 M weights |
| Winner | Every alarm rule, then the tests (Sample B, Sample A, vehicle) | - |

- **Starting point:** fine-tuning starts from the rerun's classifier, on unchanged AVES, so each run starts where the rerun ended. It is trained on 15-minute stretches with a small learning rate. Rumble and non-rumble frames count equally (the rerun's data: gaps, random forest frames, its mined false alarms and noise clips).
- **When it stops:** every 300 steps it checks the held-out training days (the same days the rerun held out). It stops after 8 checks without improvement and keeps the best check.
- **Scoring:** every option is scored like the rerun: the validation days, mean false alarms per hour at 80% and 85% recall, and the same ties. A tie goes to the simpler option, in this order: plain layer 12, another layer, LoRA, then the fine-tuned top layers.
- **Not included:** augmentation is left out, because the augmented fingerprints only exist for layer 12.
- **Training runs:** each option is trained 3 times by default (`--seeds 5` for 5).

## Time and space

- **Caching:** about 2 hours on the RTX 4060, and roughly 130 GB on the cache drive. It checks the free space first and stops with a message if there isn't enough.
- **Probing:** each layer takes about as long as one rerun experiment.
- **Fine-tuning:** each option takes very roughly 1 to 3 hours per run.
- **Total:** expect about 1 to 2 days. The real numbers will be in `run_log.txt` and `finetune.md`.

## Results

In `Configurations\_finetune\`:

| File | What |
|---|---|
| `finetune.md` | Every option: score, false alarms at 80/85/90%, max recall, trainable weights, training time |
| `final.md` / `final.json` | The winner on the tests. The rerun's own test results are in `_study\final.md` for comparison |
| `choices.json` | The probing winner and the overall winner, and why |
| `<experiment>\` | Models, training logs (every check), results, precision-recall curves |
| `run_log.txt`, `failed.txt` | Everything printed, and anything that failed |

The layer caches are in `Configurations\_embedding_cache\_layers\`.
