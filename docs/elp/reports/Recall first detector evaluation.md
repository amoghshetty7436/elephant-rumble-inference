# Count false alarms at a fixed recall target

**Short answer: yes, your plan matches common practice, with one exception that has to be fixed first.** The plan is to compare models by false alarms per hour (FP/h) at a fixed recall target of about 80–90%, choose thresholds on the validation days, and score the test days once at the end. Detection fields where missing a call is costly use the same recipe. Marine mammal detectors report recall at a fixed false-alarm budget of about 20 FP/h ([Shiu et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6969184)). The Elephant Listening Project's own rumble detector deliberately ran at roughly 80–89% recall and about 20 FP/h, below the threshold that maximises F1 ([Keen et al. 2017](https://www.researchgate.net/publication/316355452_Automated_detection_of_low-frequency_rumbles_of_forest_elephants_A_critical_tool_for_their_conservation)). Wake-word systems tune their threshold on development data and report misses at a fixed false-alarm rate on a separate test set ([Coucke et al. 2019](https://arxiv.org/pdf/1811.07684)). The exception is the current rule that turns frames into detections. A detection only exists if at least 4 frames in a row have probability above 0.5, so every cut-off at or below 0.5 gives exactly the same detections. Each model's recall therefore stops at its own ceiling of about 0.85–0.93. At a 90% target, some models may never reach the target, so the comparison is partly decided by the rule rather than by the models. The sound event detection literature shows that the frame threshold, the minimum duration and smoothing settings should be tuned on validation like any other setting, and that shorter, looser settings favour recall ([Ebbers et al. 2023](https://arxiv.org/pdf/2306.15440)). **The recommendation:**

1. Make the frame-level settings sweepable.
2. Report full precision-recall curves plus FP/h at several recall levels.
3. Pick a primary target that every model can actually reach (80% under the current rule; 90% once the rule is opened up).
4. Average over several training seeds.
5. Keep the test days sealed until every choice is frozen.

## What the key terms mean

A few terms recur throughout this report.

| Term | Plain meaning | Why it matters here |
|---|---|---|
| **Recall** (also called sensitivity, true positive rate) | Share of real rumbles the detector finds | Missed rumbles are lost data for a rare species |
| **Precision** | Share of detections that are real rumbles | Depends heavily on how common rumbles are, so it changes from site to site and day to day |
| **False alarms per hour (FP/h)** | Wrong detections per hour of audio | Directly measures the analyst's review workload, and transfers across datasets better than precision |
| **Threshold / operating point** | Score cut-off that decides "rumble or not" | Each cut-off is one trade-off between recall and false alarms |
| **PR curve** | Precision plotted against recall across all thresholds | Shows the whole trade-off, not one point |
| **Validation vs test** | Data used to make choices vs data used once to report the result | Choices made while looking at the test data give scores that are too optimistic |

## Elephant work favours recall but rarely says so formally

The ELP's main published detector paper is Keen et al. 2017. It was tested on about **3,983 hours of audio from 5 Central African clearings**. The authors tried thresholds from 0.2 to 0.95. The F1 score (a single number balancing precision and recall) peaked at thresholds of 0.6–0.65, but the authors operated at **0.3–0.4**. That choice gave **79.6–89.35% recall across sites** and a headline of **83.2% recall at about 20 false positives per hour**. At threshold 0.2, recall passed 90% at every site, but false alarms rose steeply at some sites. The authors describe recall above 70% as reliable for monitoring ([Keen et al. 2017](https://www.researchgate.net/publication/316355452_Automated_detection_of_low-frequency_rumbles_of_forest_elephants_A_critical_tool_for_their_conservation)). Two details matter for your comparison:

- **The paper's "false positive rate" is not the standard one.** It is defined as FP/(FP+TP), which is 1 − precision.
- **The paper's event rule resembles yours.** It used 100 ms frames, required a detection to stay above threshold for more than 2 s, and counted a match when at least 10% of either event overlapped the other.

Later deep-learning work on elephants mostly reports other numbers:

- Bjorck et al. 2019 reported **accuracy on class-balanced windows**, although calls make up as little as 0.5% of the audio at some sites ([Bjorck et al. 2019](https://ar5iv.arxiv.org/html/1902.09069)).
- Geldenhuys & Niesler report frame-level average precision of 0.962 on savanna and Asian elephant data. They warn that numbers across elephant papers "are not directly comparable" ([Geldenhuys & Niesler 2025](https://arxiv.org/html/2410.12082)).
- The 2026 DeepForestSound detector reports AP and best-F1 per minute of audio, without a separate set for choosing thresholds ([Dubus et al. 2026](https://arxiv.org/pdf/2604.08087)).

**None of these papers sets an explicit recall target.** Elephant metrics are a mix of different and partly non-standard measures, so any new study should state recall, precision, FP/h and the matching rule explicitly.

## Other fields usually fix false alarms or precision, not recall

Looking across fields, recall-first detection is usually reported in one of three ways.

| Field | What is held fixed | What is reported | Example |
|---|---|---|---|
| Whales (right whale upcalls) | About 20 FP/h budget | Recall at that budget; also FP/h at fixed recall | DNNs reached 85% to over 90% recall at 20 FP/h, against about 65% for the best older detector ([Shiu et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6969184)) |
| Whales (humpback song) | About 90% precision, per site, set on validation | Test precision and recall, PR curves, AP | Test recall 0.72–0.95 ([Allen et al. 2021](https://repository.library.noaa.gov/view/noaa/33468/noaa_33468_DS1.pdf)) |
| Birds (BirdNET) | A target probability that a detection is correct (0.9 or 0.95) | Species-specific thresholds | [Wood & Kahl 2024](https://connormwood.com/wp-content/uploads/2024/02/wood-kahl-2024-guidelines-for-birdnet-scores.pdf) |
| Wake-word / keyword spotting | False alarms per hour (e.g. 0.5/h) | Miss rate at that rate, threshold tuned on dev | [Coucke et al. 2019](https://arxiv.org/pdf/1811.07684) |
| Household sound event detection (DCASE) | Maximum FP/h of interest (e.g. 100/h) | Area under recall-vs-FP/h curve (PSDS) | [Cornell et al. 2024](https://arxiv.org/pdf/2406.08056) |

Shiu et al. give the reasoning most relevant to you. **Precision does not tell an analyst how often false alarms occur, but FP/h does.** Their operating rule is to use a low threshold so few calls are missed, let analysts remove the false alarms, and cap the threshold with a false-alarm budget ([Shiu et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6969184)).

Your plan reverses this: it fixes recall and measures FP/h. That is the same trade-off read from the other axis. In formal terms it is the Neyman-Pearson approach: hold one error rate fixed and minimise the other. That approach is standard in screening and detection ([Tong et al. 2018](https://arxiv.org/abs/1608.03109)). It also fixes a known weakness of comparing models by F1 or F2. Those scores apply a different, model-dependent trade-off to each model, whereas the trade-off should be a property of the problem ([Hand & Christen 2018](https://openresearch-repository.anu.edu.au/items/e3bc0e58-14c1-43a5-b1fa-78b3fdbdcd10)).

What the literature does *not* contain is a formal recall standard such as "at least 90% recall". Recall targets are implicit: Keen's "above 70% is reliable", and operating points of about 80–90%. The formal targets that do appear are on precision or on FP/h.

Two cautions apply when reading other papers:

- **ROC AUC (another single-number score) can look excellent when calls are rare.** On one humpback site with 706 positives and about 100,000 negatives, ROC AUC was 0.973, but average precision was only 0.456 ([Allen et al. 2021](https://repository.library.noaa.gov/view/noaa/33468/noaa_33468_DS1.pdf)). A 2024 NeurIPS paper adds that "precision-recall is always better under imbalance" is an over-generalisation, because AP mostly rewards getting the highest-scoring detections right ([McDermott et al. 2024](https://arxiv.org/html/2401.06091v4)). For a recall-first detector, the high-recall end of the curve is what matters, and neither full-curve summary focuses on it.
- **One bird-methods paper recommends weighting recall for rare species and situations with legal implications.** Knight et al. suggest an F-beta score that favours recall for these cases. They also warn that low recall makes occupancy estimates unreliable ([Knight et al. 2017](https://ace-eco.org/vol12/iss2/art14/)). *Unverified detail:* the summary the research team retrieved gives this as β < 1. In the usual convention, β > 1 favours recall. Check the original before quoting the direction.

## Your 0.5 gate decides recall before the models do

Under the current rule, a detection exists only if at least 4 frames in a row (about 2.56 s) have probability above 0.5. The detection is then scored by the mean probability of its frames. Every detection's mean is therefore above 0.5. Sweeping a cut-off on that mean can only remove detections, never add them. This is why recall stops at about 0.85–0.93.

In threshold-free metrics like PSDS, the swept value is meant to control which events get created ([Bilen et al. 2020](https://arxiv.org/pdf/1910.08440)). In your case it cannot, so the high-recall end of every PR curve is cut off. Each model's ceiling depends partly on how its scores happen to sit around 0.5, not only on how well it ranks rumbles.

**The 0.5 threshold has no special meaning.** A classifier trained on balanced examples and deployed on audio where rumbles are rare produces scores tied to the training balance. The best cut-off for a given cost of misses versus false alarms is then not 0.5 ([Saerens et al. 2002](https://mlanthology.org/neco/2002/saerens2002neco-adjusting); [Phelps et al. 2024](https://arxiv.org/pdf/2412.16209)). Tuning confirms this in practice. On DCASE data, a threshold tuned on validation raised event F1 on the evaluation set from **51.8% at 0.5 to 57.2%** ([Ebbers et al. 2022](https://arxiv.org/pdf/2201.13148)).

The other post-processing settings matter as much as the threshold:

- **Post-processing alone can swing results enormously.** In one DCASE comparison, event F1 for the same model ranged from **15.3% to 43.9%** depending only on post-processing ([Cances et al. 2019](https://arxiv.org/pdf/1906.06909)).
- **The best settings differ for high recall.** Longer smoothing helps when few false alarms are wanted, and shorter smoothing helps when high recall is wanted ([Ebbers et al. 2023](https://arxiv.org/pdf/2306.15440)).
- **Hysteresis matches or beats a single threshold.** Hysteresis means a higher threshold to start a detection and a lower one to keep it going. It did at least as well as a single threshold in the Cances comparison, but that paper reported only F1, so **no study isolates how much recall hysteresis adds**.
- **Rumble scores can dip mid-call.** Elephant work reports long calls breaking into several fragments ([Geldenhuys & Niesler 2024](https://arxiv.org/html/2410.12082v1)). Under a strict "4 frames in a row above 0.5" rule, a rumble whose score briefly dips can be missed entirely. Hysteresis or merging nearby fragments would recover such rumbles. This is an inference from the paper, not a measured result.
- **Practitioners lower thresholds when the goal is human review.** BirdNET guidance recommends thresholds of about **0.1–0.3** when the goal is to find vocalisations ([Pérez-Granados et al. 2025](https://bou.org.uk/blog-perez-granados-optimizing-birdnet-lessons-from-a-global-initiative/)).

## Thresholds belong to validation, and one run is not enough

The machine-learning methods literature agrees on two points.

**First, anything chosen by looking at results is a tuned setting.** That includes thresholds, minimum durations, smoothing and which checkpoint to keep. Tuned settings must be fixed on validation data before the test data is touched. Otherwise the reported score is biased upward, sometimes by as much as the gap between the methods being compared ([Cawley & Talbot 2010](https://jmlr.org/papers/v11/cawley10a.html), not re-checked by the research team). Speaker-recognition benchmarks report two numbers to expose this effect: the cost at the threshold the system chose in advance, and the best cost any threshold could have achieved on the test data. The gap between them shows how much the threshold choice cost ([Sadjadi et al. 2022](https://arxiv.org/pdf/2204.10242)).

**Second, a recall target met exactly on validation will be missed about half the time on new data.** This happens because validation recall is itself a noisy estimate. Tong et al. show that naive thresholding at the empirical target fails to control the error rate. They propose choosing the threshold with a statistical safety margin ([Tong et al. 2018](https://arxiv.org/abs/1608.03109)). Illustrative arithmetic (the report's own, not from a source): with 50 validation rumbles, an observed recall of 0.90 has a 95% confidence range of roughly 0.79–0.96.

Results also change between training runs. Data sampling, random initialisation and hyperparameter choices all move benchmark scores enough to flip conclusions. Comparisons should therefore vary these sources and report spreads, not single runs ([Bouthillier et al. 2021](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html)). The research found **no source prescribing a minimum number of seeds**. "3–5 seeds" is common practice, not a cited rule.

## Recommendations for the FruitPunch rumble detector

The recommendations below are listed in the order you would apply them.

| # | Recommendation | Why |
|---|---|---|
| 1 | **Make the frame stage sweepable.** Treat the on-threshold (try about 0.1–0.5), the minimum run length (e.g. 2–6 frames) and an optional hysteresis "stay-on" threshold or gap-merge as settings tuned on validation. | Removes the recall ceiling caused by the rule, so models are compared on how well they rank rumbles. DCASE and BirdNET practice both tune these settings. |
| 2 | **Choose the target recall by what all models can reach.** Under the current 0.5/4-frame rule, use **80%** as the primary target, because every model reaches it with some margin. Once the rule is sweepable, use **90%** as primary and keep 80% as secondary. | A target that some models cannot reach makes their FP/h undefined. 80–90% matches ELP's own operating range ([Keen et al. 2017](https://www.researchgate.net/publication/316355452_Automated_detection_of_low-frequency_rumbles_of_forest_elephants_A_critical_tool_for_their_conservation)). |
| 3 | **Report a small grid, not one number.** Give FP/h at recall 0.70 / 0.80 / 0.85 / 0.90 (marking "not reached" where relevant), recall at about 5 / 10 / 20 FP/h, full PR curves, and each model's maximum reachable recall. | Mirrors marine practice (recall at a fixed FP/h budget) and your fixed-recall view in one table. The ceiling is a finding worth reporting in itself. |
| 4 | **Use FP/h and recall as headline numbers; treat precision as context.** Add AP if useful, but do not rank models by ROC AUC. | Precision changes with how many rumbles a day contains; FP/h and recall transfer across days and sites. ROC AUC flatters rare-event detectors. |
| 5 | **Pick thresholds on validation days with a margin.** Aim slightly above the target (e.g. 0.92 on validation for a 0.90 goal), or use a lower confidence bound on validation recall. | Exact targets are missed about half the time on new data ([Tong et al. 2018](https://arxiv.org/abs/1608.03109)). |
| 6 | **Use several seeds.** Train at least 3–5 seeds per model and report the mean and spread of FP/h at the target recall. Bootstrap by day or recording, not by frame. | Run-to-run variation can flip rankings. Neighbouring frames are not independent, so frame-level resampling understates uncertainty. |
| 7 | **Seal the test days.** Choose the model, checkpoint, frame settings and threshold on validation, then freeze them all. Score the test days once and report recall and FP/h at the frozen settings. Optionally also report the best result any threshold could have achieved on test, clearly labelled as such. | Standard practice in keyword spotting, sound event detection and speaker recognition. The gap shows how well the validation choices carry over. |
| 8 | **State the matching rule and the units.** Say how much overlap counts as a hit (Keen used 10%), how rumbles in overlapping detections are counted, and that FP/h is per hour of audio. | Elephant papers are hard to compare largely because these details differ. |

**When a review-time budget is known,** consider adding one more view. Expected cost per hour = (cost of a missed rumble × misses per hour) + (review cost × detections per hour). Choosing the threshold that minimises this makes the trade-off explicit ([Elkan 2001](https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations); the threshold formula is from the team's prior knowledge and was not re-checked). No source in the notes gives real cost ratios for bioacoustic review, so these would have to come from ELP.

## What remains unverified or unknown

Several gaps limit these conclusions:

- **The FruitPunch challenge report** and any ELP evaluation of AVES- or BirdNET-based detectors were not found.
- **Sources marked as not re-read during this research** should be checked before citing: Cawley & Talbot 2010, Varma & Simon 2006, Dodge et al. 2019, Davis & Goadrich 2006, the content of Saito & Rehmsmeier 2015, McClish 1989, the exact Elkan 2001 threshold formula, and the DCASE "scenario 2" PSDS parameter values. Some venue and DOI details for Saerens 2002, Cances 2019 and Ebbers 2022 were also not verified.
- **Knight et al.'s β direction** should be checked in the original, as noted above.
- **Internally inconsistent numbers:** Keen et al. give two rumble totals (110,255 and 110,042). Bjorck et al.'s results are quoted inconsistently in Geldenhuys & Niesler's comparison table.

## Conclusion

The weak point of the plan is not the metric but the place where the decision is made. FP/h at a fixed recall is a sound, field-standard way to compare models: it puts every model at the same trade-off and expresses the cost in analyst hours. Under the current rule, however, the recall ceiling is set by the post-processing before the models are compared. Opening up the frame-level threshold, minimum duration and hysteresis is likely to change the model ranking at 90% recall more than another round of classifier tuning would. That could not be measured from the notes, so test it on the validation days before freezing anything.

Elephant monitoring lacks an agreed recall standard, and a clearly reported protocol could help set one. That means FP/h at several recall levels, the ceiling each model can reach, spread across seeds and test results scored once. That would give ELP a number tied directly to review workload.
