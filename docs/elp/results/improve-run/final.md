# What each step added (test sets, scored once)

Each run uses the cut-off that found 80% of the rumbles on the validation days. Means over 5 runs.

| Step | Experiment | Validation FA/h | Test 1: Sample B recall / precision / FA/h | Test 2: Sample A recall / precision / FA/h | Vehicle (false alarms only) recall / precision / FA/h |
|---|---|---|---|---|---|
| baseline, FruitPunch's alarm rule | ctx0_noise1_rand200_neg1_hard0 | 0.498 (0.460-0.557) | 0.890 / 0.529 / 0.30 | 0.762 / 0.683 / 0.23 | - / 0.000 / 1.33 |
| 1 alarm rules | ctx0_noise1_rand200_neg1_hard0 | 0.297 (0.275-0.317) | 0.897 / 0.582 / 0.21 | 0.760 / 0.712 / 0.18 | - / 0.000 / 0.81 |
| 2a noise clips | ctx0_noise1_rand200_neg1_hard0 | 0.297 (0.275-0.317) | 0.897 / 0.582 / 0.21 | 0.760 / 0.712 / 0.18 | - / 0.000 / 0.81 |
| 2b random frames | ctx0_noise1_rand1000_neg1_hard0 | 0.214 (0.207-0.224) | 0.914 / 0.640 / 0.16 | 0.767 / 0.781 / 0.12 | - / 0.000 / 0.74 |
| 2c negatives | ctx0_noise1_rand1000_neg1_hard0 | 0.214 (0.207-0.224) | 0.914 / 0.640 / 0.16 | 0.767 / 0.781 / 0.12 | - / 0.000 / 0.74 |
| 3 window | ctx1_noise1_rand1000_neg1_hard0 | 0.203 (0.185-0.220) | 0.912 / 0.656 / 0.15 | 0.777 / 0.798 / 0.11 | - / 0.000 / 0.66 |
| 4 combinations | ctx1_noise1_rand1000_neg1_hard0 | 0.203 (0.185-0.220) | 0.912 / 0.656 / 0.15 | 0.777 / 0.798 / 0.11 | - / 0.000 / 0.66 |
| 5 own false alarms | ctx1_noise1_rand1000_neg1_hard0 | 0.203 (0.185-0.220) | 0.912 / 0.656 / 0.15 | 0.777 / 0.798 / 0.11 | - / 0.000 / 0.66 |
| 6 final alarm rules | ctx1_noise1_rand1000_neg1_hard0 | 0.191 (0.178-0.210) | 0.906 / 0.652 / 0.15 | 0.776 / 0.795 / 0.11 | - / 0.000 / 0.64 |

The final model's detections (its best run on validation) are Raven tables in `final_detections\`, with each test set's precision-recall curve.
