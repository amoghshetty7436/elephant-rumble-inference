# tools/: ELP research scripts

Every script used to retrain, improve, compare and run the rumble detector for the
Elephant Listening Project. The scripts stay side by side in this folder because they
import each other (`dataset_configs.py`, `fp_replica.py`, `elp_score.py` and `fp_data.py`
are shared by most of them), and the walkthroughs in `docs/elp/walkthrough/` give commands
like `python tools\study.py ...`. Tests for each script are in `tests/test_<script>.py`.

Install: PyTorch for your GPU first, then `pip install -r tools/requirements-train.txt`.
The dataset is not in the repository; most scripts default to the lab PC's
`L:\ELP\Detectors\Other\Amogh\Dataset` and take `--source` for another folder.

## In the order the work happened

| Stage | Script | What it does | Docs |
|---|---|---|---|
| Shared | `fp_data.py` | Label handling (Raven tables to frame labels), no torch needed | |
| Shared | `elp_score.py` | Scores a detector the way ELP's "Detector Validation" tool does (1% overlap rule) | walkthrough/11 |
| 1. Replicate FruitPunch | `fp_replica.py` | Re-trains FruitPunch's classifier from ELP's data: `prepare`, `train`, `detect` | walkthrough/07, 08, 12 |
| 2. Choose the data | `dataset_configs.py` | Builds the training/testing configurations, trains and scores each one | walkthrough/13 |
| 3. Tune settings | `experiments.py` | Staged search for model settings on the winning data configuration | walkthrough/15 |
| 4. Improve | `improve.py` | One change at a time on top of E2, then the best combination | walkthrough/16, `docs/elp/results/improve-run/` |
| 4. Improve | `improve_converged.py` | Same as improve.py but every model trains until it has converged | `docs/elp/reports/fair-comparison-training-length.md` |
| 5. Full rerun | `study.py` | The whole search, every model trained the same fair way, ending with test scores and `final_model.json` | walkthrough/17, 18 |
| 5. Full rerun | `cpu_speed.py` | Measures CPU inference time on one real day | walkthrough/20 |
| 6. Tune AVES | `finetune.py` | Probes AVES layers, fine-tunes its top layers or LoRA from the rerun's final model | walkthrough/19 |
| 6. Tune AVES | `aves_tuning.py` | Wider screen of AVES tuning methods (layers, other AVES models, LoRA, adapters, BitFit, WiSE-FT, EMA) | walkthrough/19 |
| 7. Candidates | `candidates.py` | Six candidate detectors trained and scored like the final model | `docs/elp/results/candidate-models.html`, `results/candidates/` |
| 7. Candidates | `round2.py` | Round 2: ensembles, AVES + Dasheng fusion, stronger double-checkers | `results/candidates/run-round2-steps.md`, `results/round2/model-architectures.html` |
| 7. Candidates | `dasheng_ft.py` | Dasheng-base fine-tuned with LoRA, used by round2.py | |
| 7. Candidates | `speedup.py` | Does playing the audio faster help AVES or Dasheng? | `results/round2/run-speedup-steps.md` |
| 8. Published detectors | `baselines/` | Runs and scores EarthToolsMaker and DeepForestSound on our tests | `baselines/README.md`, `results/baselines/` |
| 8. Published detectors | `compare_hori_harm.py` | Compares our detector with Hori-Harm on analyst-reviewed days | `results/hh-comparison/` |
| 9. Run it | `detect.py` | Runs the final detector on new deployments (`survey`, `run`, `table`) at any sample rate | walkthrough/21 |
| 9. Run it | `detect_timing.py` | Runs the final detector on recordings and times every step (readers, half precision, workers) | |

The shareable, double-click version of the final detector is in `ElephantRumbleDetector/`
at the top of the repository.

`fruitpunch_files_used.txt` lists the FruitPunch dataset files the original model was trained on.
