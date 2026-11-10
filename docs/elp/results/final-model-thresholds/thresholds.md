# Final rerun model at different thresholds

Model: `E2_ctx8_noise10_rand3000_gap0.5_hard0_augnone`, alarm rule on > 0.2, at least 3 frames, smoothing 5, gaps of up to 3 joined, score = max. The threshold is the cut-off on each detection's score. Every number is the **mean of the 5 training runs**, from the `pr_curve.csv` files in `_study\final_detections\` (copies here).

| Threshold | Sample A recall | Sample A precision | Sample A FA/h | Sample B recall | Sample B precision | Sample B FA/h | Vehicle FA/h |
|---|---|---|---|---|---|---|---|
| 0.500 | 0.921 | 0.194 | 2.38 | 0.985 | 0.128 | 2.35 | 5.42 |
| 0.800 | 0.913 | 0.391 | 0.83 | 0.979 | 0.290 | 0.79 | 2.40 |
| 0.900 | 0.896 | 0.544 | 0.43 | 0.970 | 0.422 | 0.43 | 1.39 |
| 0.950 | 0.871 | 0.669 | 0.24 | 0.958 | 0.548 | 0.25 | 0.90 |
| 0.970 | 0.853 | 0.740 | 0.17 | 0.951 | 0.635 | 0.17 | 0.66 |
| 0.980 | 0.830 | 0.784 | 0.13 | 0.944 | 0.694 | 0.13 | 0.50 |
| 0.990 | 0.795 | 0.837 | 0.08 | 0.928 | 0.767 | 0.08 | 0.34 |
| 0.992 | 0.781 | 0.850 | 0.07 | 0.922 | 0.789 | 0.07 | 0.31 |
| 0.994 | 0.769 | 0.872 | 0.06 | 0.914 | 0.810 | 0.06 | 0.26 |
| 0.996 | 0.750 | 0.890 | 0.05 | 0.903 | 0.836 | 0.05 | 0.21 |
| 0.998 | 0.713 | 0.917 | 0.03 | 0.881 | 0.877 | 0.03 | 0.14 |
| 0.999 | 0.679 | 0.934 | 0.02 | 0.863 | 0.904 | 0.02 | 0.09 |

## How to read it

- **Below 0.5 nothing changes much.** Detections only start where frame scores rise above 0.2, so the curve is flat at the low end. The most recall it can reach on these tests is about 0.92 on Sample A and 0.99 on Sample B.
- **Almost all of the trade-off happens between 0.95 and 1.0**, because a detection's score is the highest frame score inside it, and real rumbles almost always have some frame near 1.
- **Best F1** (mean of runs): Sample A at 0.994 (recall 0.769, precision 0.872, F1 0.817); Sample B at 0.999 (recall 0.863, precision 0.904, F1 0.883).
- **Compared with Stanford** (P 0.842 / R 0.746 on ELP's 42 Sample A days, from ELP's own document): at 0.992 we get recall 0.781 / precision 0.850, and at 0.996 recall 0.750 / precision 0.890. Both points are better on both measures, but on our 54 Sample A days, not the same 42.
- Thresholds here are chosen by looking at the tests, so treat them as a description of the curve. The honest operating point is still the cut-off chosen on the validation days (see `final.md`).

## A practical problem for ELP's validator

ELP's MATLAB Detector Validation tool sweeps the `Score` column in steps of 0.05. With these scores, everything from 0.95 to 1.0 falls into one step, so ELP's tool would show almost none of this curve. Before handing the detector over, write a rescaled score into the Raven tables, for example −log10(1 − p) capped and divided so it fills 0–1, or the detection's rank-based percentile on the validation days. Then 0.05 steps cover the useful range.
