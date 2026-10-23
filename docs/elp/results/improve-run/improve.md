# Improving the rumble detector

Base data: E_train-clearing+forest+noise. Every number is on the validation days (Complex + General, 58 days, 1113 rumbles) and averaged over 5 training runs, with the range over runs in brackets. The tests are untouched until `final`.

**The yardstick is false alarms per hour at the cut-off that finds 80% of the rumbles** (lower is better). A difference smaller than the range in brackets can be luck.

**Final model: ctx1_noise1_rand1000_neg1_hard0** (1.92 s window, noise x1, 1000 random frames/day, 1:1 negatives, 0 hard-negative rounds), alarm rule: on > 0.2, at least 8 frames, smoothing 5, join gaps of 2, score = mean.

| Experiment | Stage | Window (s) | Noise x | Random/day | Neg:1 | Hard | FA/h at 80% | FA/h at 85% | FA/h at 90% | Max recall | Best F1 | Classifier s per 24 h (GPU / CPU) | Negatives per rumble (actual) | Noise+random share | Rule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ctx1_noise1_rand1000_neg1_hard0 | 3 | 1.92 | 1 | 1000 | 1 | 0 | **0.191 (0.178-0.210)** | 0.284 (0.263-0.321) | 0.439 (0.422-0.467) | 0.914 (0.897-0.923) | 0.751 (0.740-0.760) | 0.05 / 0.04 | 1.08 | 0.73 | on0.2_min8_smooth5_gap2_mean |
| ctx1_noise1_rand1000_neg1_hard3 | mistakes | 1.92 | 1 | 1000 | 1 | 3 | **0.195 (0.170-0.234)** | 0.268 (0.236-0.283) | 0.406 (0.373-0.433) | 0.949 (0.934-0.955) | 0.758 (0.741-0.767) | 0.10 / 0.04 | 1.16 | 0.69 | on0.25_min6_smooth5_gap2_mean |
| ctx6_noise1_rand1000_neg1_hard0 | 3 | 8.32 | 1 | 1000 | 1 | 0 | **0.198 (0.188-0.216)** | 0.284 (0.257-0.327) | 0.419 (0.381-0.467) | 0.944 (0.940-0.946) | 0.748 (0.731-0.756) | 0.08 / 0.07 | 1.08 | 0.73 | on0.25_min6_smooth5_gap2_mean |
| ctx7_noise1_rand1000_neg1_hard0 | 3 | 9.6 | 1 | 1000 | 1 | 0 | **0.204 (0.176-0.229)** | 0.271 (0.247-0.295) | 0.420 (0.383-0.478) | 0.947 (0.940-0.951) | 0.747 (0.732-0.765) | 0.07 / 0.09 | 1.08 | 0.73 | on0.25_min6_smooth5_gap2_mean |
| ctx4_noise1_rand1000_neg1_hard0 | 3 | 5.76 | 1 | 1000 | 1 | 0 | **0.212 (0.191-0.222)** | 0.288 (0.257-0.308) | 0.437 (0.389-0.464) | 0.949 (0.944-0.955) | 0.746 (0.736-0.756) | 0.10 / 0.06 | 1.08 | 0.73 | on0.25_min6_smooth5_gap2_mean |
| ctx0_noise1_rand1000_neg1_hard0 | 2b | 0.64 | 1 | 1000 | 1 | 0 | **0.214 (0.207-0.224)** | 0.315 (0.294-0.336) | 0.534 (0.510-0.600) | 0.931 (0.923-0.937) | 0.748 (0.738-0.756) | 0.19 / 0.06 | 1.08 | 0.73 | on0.25_min6_smooth5_gap2_mean |
| ctx9_noise1_rand1000_neg1_hard0 | 3 | 12.16 | 1 | 1000 | 1 | 0 | **0.220 (0.170-0.295)** | 0.298 (0.231-0.349) | 0.473 (0.336-0.584) | 0.946 (0.941-0.950) | 0.742 (0.699-0.773) | 0.10 / 0.09 | 1.08 | 0.73 | on0.25_min6_smooth5_gap2_mean |
| ctx5_noise1_rand1000_neg1_hard0 | 3 | 7.04 | 1 | 1000 | 1 | 0 | **0.223 (0.208-0.252)** | 0.291 (0.278-0.304) | 0.444 (0.429-0.470) | 0.952 (0.950-0.953) | 0.738 (0.723-0.747) | 0.08 / 0.08 | 1.08 | 0.73 | on0.25_min6_smooth5_gap2_mean |
| ctx8_noise1_rand1000_neg1_hard0 | 3 | 10.88 | 1 | 1000 | 1 | 0 | **0.223 (0.207-0.238)** | 0.300 (0.264-0.342) | 0.442 (0.378-0.503) | 0.949 (0.945-0.953) | 0.735 (0.728-0.748) | 0.07 / 0.10 | 1.08 | 0.73 | on0.25_min6_smooth5_gap2_mean |
| ctx1_noise2_rand1000_neg1_hard0 | combos | 1.92 | 2 | 1000 | 1 | 0 | **0.224 (0.205-0.249)** | 0.301 (0.289-0.311) | 0.465 (0.449-0.477) | 0.953 (0.941-0.965) | 0.745 (0.736-0.751) | 0.05 / 0.04 | 1.08 | 0.75 | on0.25_min6_smooth5_gap2_mean |
| ctx2_noise1_rand1000_neg1_hard0 | 3 | 3.2 | 1 | 1000 | 1 | 0 | **0.227 (0.185-0.263)** | 0.302 (0.254-0.348) | 0.457 (0.398-0.520) | 0.954 (0.951-0.959) | 0.745 (0.726-0.764) | 0.16 / 0.05 | 1.08 | 0.73 | on0.25_min6_smooth5_gap2_mean |
| ctx1_noise1_rand1000_neg1_hard2 | mistakes | 1.92 | 1 | 1000 | 1 | 2 | **0.235 (0.193-0.280)** | 0.310 (0.246-0.349) | 0.484 (0.432-0.571) | 0.953 (0.949-0.960) | 0.743 (0.735-0.765) | 0.11 / 0.04 | 1.16 | 0.69 | on0.25_min6_smooth5_gap2_mean |
| ctx6_noise2_rand1000_neg2_hard0 | combos | 8.32 | 2 | 1000 | 2 | 0 | **0.235 (0.178-0.301)** | 0.299 (0.227-0.358) | 0.472 (0.370-0.592) | 0.942 (0.934-0.951) | 0.734 (0.694-0.762) | 0.07 / 0.07 | 2.16 | 0.40 | on0.25_min6_smooth5_gap2_mean |
| ctx3_noise1_rand1000_neg1_hard0 | 3 | 4.48 | 1 | 1000 | 1 | 0 | **0.235 (0.207-0.260)** | 0.314 (0.287-0.357) | 0.447 (0.409-0.485) | 0.956 (0.950-0.963) | 0.738 (0.726-0.752) | 0.03 / 0.06 | 1.08 | 0.73 | on0.25_min6_smooth5_gap2_mean |
| ctx1_noise2_rand1000_neg2_hard0 | combos | 1.92 | 2 | 1000 | 2 | 0 | **0.237 (0.210-0.275)** | 0.325 (0.309-0.368) | 0.514 (0.454-0.570) | 0.952 (0.949-0.954) | 0.743 (0.734-0.750) | 0.04 / 0.04 | 2.16 | 0.40 | on0.25_min6_smooth5_gap2_mean |
| ctx6_noise1_rand500_neg1_hard0 | combos | 8.32 | 1 | 500 | 1 | 0 | **0.239 (0.181-0.280)** | 0.315 (0.268-0.359) | 0.505 (0.440-0.609) | 0.942 (0.936-0.948) | 0.735 (0.719-0.761) | 0.08 / 0.07 | 1.08 | 0.40 | on0.25_min6_smooth5_gap2_mean |
| ctx1_noise1_rand1000_neg2_hard0 | combos | 1.92 | 1 | 1000 | 2 | 0 | **0.245 (0.222-0.260)** | 0.347 (0.326-0.365) | 0.530 (0.495-0.569) | 0.949 (0.942-0.953) | 0.734 (0.717-0.741) | 0.06 / 0.04 | 2.16 | 0.37 | on0.25_min6_smooth5_gap2_mean |
| ctx6_noise1_rand1000_neg2_hard0 | combos | 8.32 | 1 | 1000 | 2 | 0 | **0.248 (0.214-0.290)** | 0.332 (0.298-0.388) | 0.502 (0.420-0.547) | 0.943 (0.937-0.951) | 0.726 (0.704-0.747) | 0.08 / 0.07 | 2.16 | 0.37 | on0.25_min6_smooth5_gap2_mean |
| ctx6_noise2_rand1000_neg1_hard0 | combos | 8.32 | 2 | 1000 | 1 | 0 | **0.251 (0.229-0.275)** | 0.329 (0.278-0.382) | 0.474 (0.423-0.501) | 0.955 (0.945-0.965) | 0.724 (0.705-0.740) | 0.07 / 0.07 | 1.08 | 0.75 | on0.25_min6_smooth5_gap2_mean |
| ctx0_noise1_rand500_neg1_hard0 | 2b | 0.64 | 1 | 500 | 1 | 0 | **0.252 (0.231-0.287)** | 0.375 (0.320-0.415) | 0.623 (0.592-0.645) | 0.935 (0.930-0.942) | 0.737 (0.725-0.748) | 0.06 / 0.04 | 1.08 | 0.40 | on0.25_min6_smooth5_gap2_mean |
| ctx0_noise1_rand2000_neg1_hard0 | 2b | 0.64 | 1 | 2000 | 1 | 0 | **0.259 (0.242-0.272)** | 0.385 (0.342-0.412) | 0.590 (0.546-0.641) | 0.930 (0.925-0.937) | 0.728 (0.722-0.731) | 0.06 / 0.05 | 1.08 | 0.75 | on0.25_min6_smooth5_gap2_mean |
| ctx0_noise1_rand1000_neg2_hard0 | 2c | 0.64 | 1 | 1000 | 2 | 0 | **0.260 (0.248-0.280)** | 0.391 (0.375-0.413) | 0.627 (0.569-0.719) | 0.932 (0.921-0.942) | 0.729 (0.722-0.737) | 0.06 / 0.05 | 2.16 | 0.37 | on0.25_min6_smooth5_gap2_mean |
| ctx1_noise2_rand500_neg1_hard0 | combos | 1.92 | 2 | 500 | 1 | 0 | **0.263 (0.239-0.290)** | 0.360 (0.339-0.392) | 0.562 (0.523-0.602) | 0.954 (0.946-0.963) | 0.726 (0.719-0.732) | 0.05 / 0.04 | 1.08 | 0.46 | on0.25_min6_smooth5_gap2_mean |
| ctx0_noise1_rand1000_neg3_hard0 | 2c | 0.64 | 1 | 1000 | 3 | 0 | **0.265 (0.249-0.280)** | 0.391 (0.371-0.420) | 0.655 (0.582-0.710) | 0.929 (0.919-0.942) | 0.726 (0.721-0.736) | 0.04 / 0.05 | 3.25 | 0.24 | on0.25_min6_smooth5_gap2_mean |
| ctx1_noise1_rand500_neg1_hard0 | combos | 1.92 | 1 | 500 | 1 | 0 | **0.265 (0.241-0.285)** | 0.367 (0.349-0.387) | 0.562 (0.534-0.607) | 0.953 (0.948-0.955) | 0.724 (0.715-0.731) | 0.10 / 0.04 | 1.08 | 0.40 | on0.25_min6_smooth5_gap2_mean |
| ctx1_noise1_rand500_neg2_hard0 | combos | 1.92 | 1 | 500 | 2 | 0 | **0.277 (0.265-0.288)** | 0.386 (0.338-0.449) | 0.611 (0.567-0.676) | 0.952 (0.946-0.959) | 0.716 (0.709-0.726) | 0.05 / 0.04 | 2.16 | 0.20 | on0.25_min6_smooth5_gap2_mean |
| ctx0_noise1_rand1000_neg5_hard0 | 2c | 0.64 | 1 | 1000 | 5 | 0 | **0.296 (0.260-0.343)** | 0.431 (0.378-0.485) | 0.721 (0.656-0.768) | 0.938 (0.934-0.943) | 0.712 (0.699-0.721) | 0.15 / 0.05 | 5.41 | 0.15 | on0.25_min6_smooth5_gap2_mean |
| ctx0_noise1_rand200_neg1_hard0 | 2a | 0.64 | 1 | 200 | 1 | 0 | **0.297 (0.275-0.317)** | 0.434 (0.416-0.454) | 0.759 (0.725-0.804) | 0.933 (0.926-0.942) | 0.704 (0.701-0.707) | 0.08 / 0.04 | 1.08 | 0.19 | on0.25_min6_smooth5_gap2_mean |
| ctx6_noise2_rand500_neg1_hard0 | combos | 8.32 | 2 | 500 | 1 | 0 | **0.297 (0.252-0.344)** | 0.386 (0.351-0.427) | 0.568 (0.543-0.613) | 0.948 (0.943-0.951) | 0.703 (0.681-0.730) | 0.08 / 0.07 | 1.08 | 0.46 | on0.25_min6_smooth5_gap2_mean |
| ctx1_noise2_rand500_neg2_hard0 | combos | 1.92 | 2 | 500 | 2 | 0 | **0.298 (0.265-0.336)** | 0.420 (0.364-0.476) | 0.674 (0.602-0.848) | 0.952 (0.942-0.961) | 0.712 (0.691-0.729) | 0.06 / 0.04 | 2.16 | 0.23 | on0.25_min6_smooth5_gap2_mean |
| ctx0_noise2_rand200_neg1_hard0 | 2a | 0.64 | 2 | 200 | 1 | 0 | **0.308 (0.283-0.333)** | 0.464 (0.418-0.487) | 0.729 (0.669-0.817) | 0.947 (0.944-0.949) | 0.695 (0.674-0.709) | 0.06 / 0.05 | 1.08 | 0.25 | on0.25_min6_smooth5_gap2_mean |
| ctx6_noise1_rand500_neg2_hard0 | combos | 8.32 | 1 | 500 | 2 | 0 | **0.312 (0.282-0.348)** | 0.423 (0.372-0.462) | 0.673 (0.528-0.799) | 0.947 (0.935-0.957) | 0.693 (0.682-0.718) | 0.08 / 0.07 | 2.16 | 0.20 | on0.25_min6_smooth5_gap2_mean |
| ctx6_noise2_rand500_neg2_hard0 | combos | 8.32 | 2 | 500 | 2 | 0 | **0.324 (0.289-0.389)** | 0.435 (0.387-0.500) | 0.639 (0.579-0.753) | 0.947 (0.940-0.955) | 0.695 (0.666-0.716) | 0.08 / 0.07 | 2.16 | 0.23 | on0.25_min6_smooth5_gap2_mean |
| ctx0_noise3_rand200_neg1_hard0 | 2a | 0.64 | 3 | 200 | 1 | 0 | **0.325 (0.301-0.350)** | 0.447 (0.410-0.468) | 0.736 (0.715-0.771) | 0.939 (0.932-0.947) | 0.698 (0.694-0.704) | 0.04 / 0.05 | 1.08 | 0.31 | on0.25_min6_smooth5_gap2_mean |
| ctx0_noise10_rand200_neg1_hard0 | 2a | 0.64 | 10 | 200 | 1 | 0 | **0.328 (0.309-0.355)** | 0.455 (0.426-0.505) | 0.662 (0.607-0.725) | 0.954 (0.943-0.960) | 0.689 (0.681-0.703) | 0.06 / 0.05 | 1.08 | 0.72 | on0.25_min6_smooth5_gap2_mean |
| ctx0_noise5_rand200_neg1_hard0 | 2a | 0.64 | 5 | 200 | 1 | 0 | **0.340 (0.311-0.359)** | 0.478 (0.461-0.521) | 0.780 (0.725-0.856) | 0.947 (0.940-0.950) | 0.689 (0.678-0.695) | 0.06 / 0.04 | 1.08 | 0.43 | on0.25_min6_smooth5_gap2_mean |
| ctx1_noise1_rand1000_neg1_hard1 | mistakes | 1.92 | 1 | 1000 | 1 | 1 | **0.379 (0.285-0.533)** | 0.532 (0.418-0.740) | 0.802 (0.656-1.034) | 0.909 (0.897-0.933) | 0.691 (0.640-0.711) | 0.07 / 0.04 | 1.08 | 0.73 | on0.25_min6_smooth5_gap2_mean |
| ctx0_noise20_rand200_neg1_hard0 | 2a | 0.64 | 20 | 200 | 1 | 0 | **0.390 (0.337-0.457)** | 0.547 (0.497-0.641) | 0.834 (0.741-0.930) | 0.959 (0.953-0.966) | 0.675 (0.654-0.686) | 0.06 / 0.04 | 1.08 | 0.75 | on0.25_min6_smooth5_gap2_mean |
| ctx0_noise1_rand0_neg1_hard0 | 2b | 0.64 | 1 | 0 | 1 | 0 | **0.400 (0.370-0.439)** | 0.556 (0.533-0.582) | 0.908 (0.835-0.973) | 0.951 (0.937-0.961) | 0.662 (0.650-0.684) | 0.16 / 0.04 | 1.08 | 0.06 | on0.25_min6_smooth5_gap2_mean |

AVES itself takes about 31 s per 24 h of audio on this PC, for every model.

## Choices made

- 2026-10-20 17:03 stage 1 (alarm rules): picked on > 0.25, at least 6 frames, smoothing 5, join gaps of 2, score = mean on ctx0_noise1_rand200_neg1_hard0; 0.297 false alarms/h at 80% recall (FruitPunch's rule: 0.498 false alarms/h at 80% recall)
- 2026-10-20 19:51 stage 2a (noise clips used): picked ctx0_noise1_rand200_neg1_hard0 (0.64 s window, noise x1, 200 random frames/day, 1:1 negatives, 0 hard-negative rounds)
- 2026-10-20 22:03 stage 2b (random forest frames per day): picked ctx0_noise1_rand1000_neg1_hard0 (0.64 s window, noise x1, 1000 random frames/day, 1:1 negatives, 0 hard-negative rounds)
- 2026-10-20 23:43 stage 2c (negatives per rumble): picked ctx0_noise1_rand1000_neg1_hard0 (0.64 s window, noise x1, 1000 random frames/day, 1:1 negatives, 0 hard-negative rounds)
- 2026-10-21 04:43 stage 3 (listening window): picked ctx1_noise1_rand1000_neg1_hard0 (1.92 s window, noise x1, 1000 random frames/day, 1:1 negatives, 0 hard-negative rounds); ctx6_noise1_rand1000_neg1_hard0 scored lower but within the run-to-run spread, so the simpler one is kept
- 2026-10-21 12:30 stage 4 (combinations): picked ctx1_noise1_rand1000_neg1_hard0 (1.92 s window, noise x1, 1000 random frames/day, 1:1 negatives, 0 hard-negative rounds); ctx6_noise1_rand1000_neg1_hard0 scored lower but within the run-to-run spread, so the simpler one is kept
- 2026-10-21 14:27 stage 5 (hard negatives): picked ctx1_noise1_rand1000_neg1_hard0 (1.92 s window, noise x1, 1000 random frames/day, 1:1 negatives, 0 hard-negative rounds); ctx1_noise1_rand1000_neg1_hard3 scored lower but within the run-to-run spread, so the simpler one is kept
- 2026-10-21 14:38 stage 6 (alarm rules for the final model): picked on > 0.2, at least 8 frames, smoothing 5, join gaps of 2, score = mean on ctx1_noise1_rand1000_neg1_hard0; 0.191 false alarms/h at 80% recall (FruitPunch's rule: 0.245 false alarms/h at 80% recall)
