# ELP rumble detector work

Notes, reports and results from improving FruitPunch's elephant rumble detector for the
Elephant Listening Project. The code lives in `tools/` (index: `tools/README.md`) with tests
in `tests/`; the shareable detector is in `ElephantRumbleDetector/`; model weights are
listed in `models/README.md`. This folder holds everything written around the code.

| Folder | What |
|---|---|
| `walkthrough/` | Step-by-step explanations, numbered in the order the work happened (list below) |
| `reports/` | Evaluations, comparisons, the literature survey, and progress reports (`progress/`) |
| `research_notes/` | Background research (bioacoustics practice, ML methods, post-processing) |
| `results/` | Result pages and run instructions for each stage (list below) |
| `retraining-results-2026-10-09/` | First retraining run: logs and the retrained classifier |
| `release/` | Screenshots and an example output of `ElephantRumbleDetector` |
| `updates/` | Progress updates |

## Walkthrough

| # | Topic |
|---|---|
| 00-03 | Overview, the pipeline in plain words, running it on Windows, speed and input length |
| 04-08 | Retraining plan, ELP data map, FruitPunch's datasets, how to retrain, the retraining script |
| 09-12 | Every file explained, improving precision/recall, the results explained, the code checked against FruitPunch |
| 13-16 | Choosing training and testing data, improvement ideas, the experiment runner, the improvement workflow |
| 17-18 | The full rerun: plan answers and how to run it |
| 19 | Teaching AVES (finetune.py, aves_tuning.py) |
| 20 | Splitting the work across CPUs, GPUs and machines |
| 21 | Running the detector on new deployments (detect.py) |

## Results

| Path | What |
|---|---|
| `results/rumble-detector-research-explained.html` | The whole project explained in one page |
| `results/rumble-detector-training-setups.html` | The training/testing setups compared |
| `results/rumble-detector-improvement-run.html`, `results/improve-run/` | The improvement run (improve.py) |
| `results/aves-finetuning-explained.html` | Fine-tuning and LoRA explained |
| `results/final-model-thresholds/` | Precision-recall curves of the final model on Sample A, Sample B and the vehicle days |
| `results/candidate-models.html`, `results/candidates/` | Candidate detectors and their results (candidates.py, round2.py) |
| `results/round2/` | Model architectures of round 2 and the speed-up run |
| `results/baselines/` | How to run the published detectors (EarthToolsMaker, DeepForestSound) |
| `results/hh-comparison/` | Comparison with Hori-Harm: summary reports and per-site tables from two runs |

## Not in this repository

The ELP recordings, Raven label tables, truth tables per day, dataset inventories and
detection-level outputs (for example the full Hori-Harm match lists) stay on the lab PC
and in the project's shared folder. They are ELP data, and this repository is public.
