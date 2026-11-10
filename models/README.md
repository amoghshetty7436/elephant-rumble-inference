# Model weights

Small model files (about 1 MB or less) are kept in git. Large files and anything
built from ELP's recordings stay on the lab PC.

## In this repository

| File | What it is | Used by |
|---|---|---|
| `models/fruitpunch-2024-07-03.pth` | FruitPunch's original 2024 classifier (the baseline we started from). The original download site, 0ape.com, is offline | `tools/fp_replica.py train --compare`, `elephant-rumble-inference --model` |
| `docs/elp/retraining-results-2026-10-09/retrained.pth` | First retraining on ELP data (2026-10-09), FruitPunch's recipe | walkthrough/11 |
| `ElephantRumbleDetector/model/rumble_model.pth` | **The final detector**: `E2_ctx8_noise10_rand3000_gap0.5_hard0_augnone`, run 0 of the full rerun, with its alarm rule and cut-off in `settings.json` | `ElephantRumbleDetector/` |

AVES itself (`aves-base-bio`) is downloaded by torchaudio / the scripts on first use.

## Only on the lab PC

Under `L:\ELP\Detectors\Other\Amogh\Dataset\Configurations\`:

| Folder | What |
|---|---|
| `_study\final_detections\final_model.json` (+ heads) | The rerun's final model as `tools/detect.py` reads it |
| `_study\`, `_improve\`, `_experiments\` | Every model trained during the searches, with logs |
| `_candidates\`, `_round2\`, `_speedup\` | Candidate detectors, round 2 heads and double-checkers |
| AVES / Dasheng fingerprints | About 140 GB of cached embeddings; rebuilt by the scripts if missing |

If one of these heads should be shared, it can go into a GitHub release (or Git LFS)
rather than the main history.
