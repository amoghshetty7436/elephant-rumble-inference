# Our detector vs Hori-Harm: nn_202306_jun_dep17

Made 2026-11-10 18:00 by tools/compare_hori_harm.py.
Hori-Harm table: `/mnt/project-files/nn_202306_jun_dep17_d00_HHv6p2_p4_rand3_counted.txt`  
Our results: `/tmp/claude-0/-home-user/50b568e8-3177-5d89-b0ab-d494490db69c/scratchpad/dep17/nn_202306_jun_dep17`  
Our model: E2_ctx8_noise10_rand3000_gap0.5_hard0_augnone (run 0), rule {'on': 0.2, 'min_frames': 3, 'smooth': 5, 'gap': 3, 'score': 'max'}; main cut-off 0.993. Two boxes match when they overlap or are at most 2 s apart.

## What is compared

- Hori-Harm table: 96,437 rows. Dropped 11,620 rows with no Count (not reviewed: an un-reviewed copy of a site) and 0 exact duplicates, leaving **84,817 reviewed detections on 1,669 recordings**.
- **1,653 of those recordings are in our results** and are compared below. 16 are not in our results (sites nn03b, nn03c, nn04c).
- On those 1,653 recordings Hori-Harm has **2,071 confirmed rumble boxes** (2,276 calls; a box can hold several) and **82,651 false alarms**.
- Only our tables were used (no probabilities): one cut-off only.

## Main result (our cut-off 0.993)

|  | Hori-Harm (score >= 0.4) | Ours |
|---|---|---|
| Detections | 84,722 | 3,413 |
| Detections per day | 51.25 | 2.06 |
| Hori-Harm's confirmed rumbles found | 2,071 (100%, by definition) | 1,844 (89.0%) |
| Calls found | 2,276 | 2,044 (89.8%) |
| Hori-Harm's false alarms raised | 82,651 | 1,320 (1.6%) |
| Precision | 2.4% | between 43.0% and 84.5% |

Of our 3,413 detections, 1,468 are on a confirmed rumble, 530 on a Hori-Harm false alarm, and **1,415 on no Hori-Harm box**. Those last ones are either our false alarms or rumbles Hori-Harm missed (scored under 0.4 or never boxed); nobody has looked at them yet. Our precision is therefore between 43.0% (all of them are false alarms) and 84.5% (all of them are rumbles). Listening to `ours_not_in_hori_harm_sample.txt` (200 picked at random) pins it down.

For a like-for-like view: Hori-Harm with about as many detections as ours (score >= 0.8: 1,612) keeps 330 of its confirmed rumbles (15.9%), where we find 1,844 (89.0%).

**Read this with care:** the rumbles here are only the ones Hori-Harm found. Every confirmed rumble is by construction a Hori-Harm detection, so this measures how much of Hori-Harm's catch we also catch, not our true recall; rumbles that both detectors missed are not counted anywhere. Also, only recordings with at least one Hori-Harm detection are in the table, so a sampled day where Hori-Harm found nothing is missing here.

## Our detector at other cut-offs

(needs the .probs.npy files next to the tables; only the main cut-off was available)

## Hori-Harm at higher score cut-offs

| HH score cut-off | HH detections | per day | confirmed rumbles kept | % of HH rumbles | HH precision |
|---|---|---|---|---|---|
| 0.4 | 84722 | 51.25 | 2071 | 100.0% | 2.4% |
| 0.5 | 34630 | 20.95 | 1533 | 74.0% | 4.4% |
| 0.6 | 13810 | 8.35 | 1055 | 50.9% | 7.6% |
| 0.7 | 5243 | 3.17 | 650 | 31.4% | 12.4% |
| 0.8 | 1612 | 0.98 | 330 | 15.9% | 20.5% |
| 0.9 | 172 | 0.1 | 108 | 5.2% | 62.8% |
| 0.95 | 42 | 0.03 | 40 | 1.9% | 95.2% |

## How sure is the matching? (main cut-off, different slack)

| slack s | HH rumbles found | of | % | HH false alarms also raised |
|---|---|---|---|---|
| 0 | 1844 | 2071 | 89.0% | 1278 |
| 1 | 1844 | 2071 | 89.0% | 1295 |
| 2 | 1844 | 2071 | 89.0% | 1320 |
| 5 | 1846 | 2071 | 89.1% | 1388 |

## Which of Hori-Harm's rumbles do we miss?

By Hori-Harm's score (a rough guide to how clear the rumble is):

| HH score | confirmed rumbles | we find | % |
|---|---|---|---|
| 0.4-0.5 | 538 | 423 | 78.6% |
| 0.5-0.6 | 478 | 415 | 86.8% |
| 0.6-0.7 | 405 | 364 | 89.9% |
| 0.7-0.8 | 320 | 313 | 97.8% |
| 0.8-0.9 | 222 | 221 | 99.5% |
| 0.9-1.0 | 108 | 108 | 100.0% |

By the number of calls the analyst counted in the box:

| calls in box | boxes | we find | % |
|---|---|---|---|
| 1 | 1912 | 1690 | 88.4% |
| 2 | 129 | 124 | 96.1% |
| 3 | 22 | 22 | 100.0% |
| 4 | 5 | 5 | 100.0% |
| 6 | 1 | 1 | 100.0% |
| 7 | 2 | 2 | 100.0% |

Boxes with an analyst note ("faint", "q", "insect?" ...): we find 73 of 130 (56.2%).

## Per site

| site | days | HH detections | HH confirmed rumbles | we find | % found | HH false alarms | we also fire on | our detections | ours on a HH rumble | ours on a HH false alarm | ours on no HH box |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nn01b | 55 | 2829 | 44 | 40 | 90.9% | 2785 | 3 | 80 | 33 | 2 | 45 |
| nn01c | 27 | 6741 | 0 | 0 | - | 6741 | 0 | 0 | 0 | 0 | 0 |
| nn01e | 2 | 4 | 0 | 0 | - | 4 | 0 | 0 | 0 | 0 | 0 |
| nn01f | 34 | 2015 | 15 | 13 | 86.7% | 2000 | 7 | 28 | 10 | 2 | 16 |
| nn01g | 25 | 324 | 0 | 0 | - | 324 | 0 | 1 | 0 | 0 | 1 |
| nn02a | 2 | 4 | 0 | 0 | - | 4 | 0 | 1 | 0 | 0 | 1 |
| nn02b | 49 | 1193 | 35 | 30 | 85.7% | 1158 | 5 | 89 | 28 | 2 | 59 |
| nn02c | 29 | 271 | 0 | 0 | - | 271 | 0 | 1 | 0 | 0 | 1 |
| nn02d | 54 | 2997 | 156 | 149 | 95.5% | 2841 | 8 | 263 | 130 | 7 | 126 |
| nn02e | 15 | 6520 | 0 | 0 | - | 6520 | 0 | 0 | 0 | 0 | 0 |
| nn02f | 51 | 2597 | 49 | 48 | 98.0% | 2548 | 2 | 74 | 33 | 2 | 39 |
| nn02g | 19 | 467 | 0 | 0 | - | 467 | 1 | 2 | 0 | 1 | 1 |
| nn03a | 49 | 1936 | 47 | 43 | 91.5% | 1889 | 2 | 37 | 27 | 2 | 8 |
| nn03d | 43 | 1489 | 4 | 3 | 75.0% | 1485 | 3 | 9 | 3 | 2 | 4 |
| nn03f | 34 | 547 | 0 | 0 | - | 547 | 2 | 4 | 0 | 1 | 3 |
| nn03g | 27 | 615 | 27 | 24 | 88.9% | 588 | 14 | 55 | 16 | 8 | 31 |
| nn04a | 47 | 2111 | 250 | 215 | 86.0% | 1861 | 16 | 293 | 167 | 10 | 116 |
| nn04b | 49 | 2052 | 12 | 10 | 83.3% | 2040 | 0 | 16 | 9 | 0 | 7 |
| nn04d | 65 | 1571 | 45 | 41 | 91.1% | 1526 | 17 | 137 | 37 | 13 | 87 |
| nn04e | 18 | 61 | 1 | 1 | 100.0% | 60 | 1 | 3 | 1 | 1 | 1 |
| nn04f | 50 | 1348 | 76 | 63 | 82.9% | 1272 | 9 | 60 | 36 | 6 | 18 |
| nn05a | 43 | 497 | 0 | 0 | - | 497 | 2 | 9 | 0 | 2 | 7 |
| nn05b | 55 | 2776 | 266 | 242 | 91.0% | 2510 | 38 | 368 | 202 | 25 | 141 |
| nn05c | 48 | 560 | 3 | 2 | 66.7% | 557 | 1 | 13 | 2 | 1 | 10 |
| nn05d | 40 | 1466 | 112 | 103 | 92.0% | 1354 | 30 | 141 | 81 | 16 | 44 |
| nn05f | 33 | 337 | 4 | 4 | 100.0% | 333 | 5 | 13 | 3 | 5 | 5 |
| nn06a | 5 | 11 | 0 | 0 | - | 11 | 0 | 0 | 0 | 0 | 0 |
| nn06b | 13 | 58 | 3 | 2 | 66.7% | 55 | 0 | 2 | 2 | 0 | 0 |
| nn06c | 26 | 133 | 0 | 0 | - | 133 | 0 | 0 | 0 | 0 | 0 |
| nn06d | 56 | 7240 | 28 | 20 | 71.4% | 7212 | 10 | 45 | 20 | 7 | 18 |
| nn06e | 51 | 2187 | 23 | 16 | 69.6% | 2164 | 7 | 42 | 11 | 7 | 24 |
| nn06f | 57 | 5742 | 15 | 10 | 66.7% | 5727 | 31 | 46 | 9 | 18 | 19 |
| nn07a | 62 | 1740 | 68 | 64 | 94.1% | 1672 | 11 | 100 | 53 | 9 | 38 |
| nn07b | 58 | 3033 | 9 | 7 | 77.8% | 3024 | 5 | 37 | 7 | 5 | 25 |
| nn07c | 53 | 4559 | 76 | 70 | 92.1% | 4483 | 22 | 173 | 63 | 19 | 91 |
| nn08a | 64 | 636 | 8 | 8 | 100.0% | 628 | 1 | 18 | 6 | 1 | 11 |
| nn08b | 62 | 1676 | 26 | 19 | 73.1% | 1650 | 11 | 65 | 18 | 9 | 38 |
| nn08c | 61 | 2428 | 493 | 435 | 88.2% | 1935 | 22 | 580 | 330 | 15 | 235 |
| nn09a | 46 | 257 | 2 | 2 | 100.0% | 255 | 1 | 20 | 2 | 1 | 17 |
| nn09b | 2 | 2 | 0 | 0 | - | 2 | 0 | 0 | 0 | 0 | 0 |
| nn09c | 59 | 11620 | 173 | 159 | 91.9% | 11447 | 1032 | 583 | 128 | 330 | 125 |
| nn10b | 15 | 72 | 1 | 1 | 100.0% | 71 | 1 | 5 | 1 | 1 | 3 |

## Files written

- `report.md`: this report.
- `our_cutoffs.csv`, `hori_harm_cutoffs.csv`, `per_site.csv`: the tables above, for Excel.
- `hori_harm_boxes_matched.csv`: every reviewed Hori-Harm box with the best score of our detections on it.
- `hori_harm_rumbles_we_missed.txt` (227): Raven table of the confirmed rumbles we miss at 0.993.
- `ours_not_in_hori_harm.txt` (1,415) and `ours_not_in_hori_harm_sample.txt` (200): Raven tables of our detections with no Hori-Harm box, all and a random sample, with an empty Count column for a reviewer.
- `ours_on_hori_harm_false_alarms.txt` (530): our detections on boxes the analysts called false alarms.
