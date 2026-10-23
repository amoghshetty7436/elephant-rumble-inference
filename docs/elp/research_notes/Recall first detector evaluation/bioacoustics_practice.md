# How bioacoustics / PAM studies evaluate detectors and set thresholds (with emphasis on rare species and recall)

## Q1. How do the Elephant Listening Project (ELP, Cornell K. Lisa Yang Center) and related groups evaluate forest elephant rumble detectors?

### Takeaway
The ELP's main published detector paper (Keen et al. 2017, JASA) reports event-level TPR/recall against "false positives per hour" and against an FPR the authors define non-standardly. It sweeps score thresholds from 0.2 to 0.95 and picks a low operating threshold (0.3–0.4) that favours recall (about 80–89%) at roughly 20 false positives per hour. Later deep-learning elephant work (Bjorck et al. 2019, Geldenhuys & Niesler 2024/25, DeepForestSound 2026) mostly reports accuracy, average precision (AP) or best-F1 instead. None of these sets an explicit recall target.

### Cited Findings
- **Keen, Shiu, Wrege & Rowland (2017)**, "Automated detection of low-frequency rumbles of forest elephants: A critical tool for their conservation," *J. Acoust. Soc. Am.* 141(4):2715–2726, doi:10.1121/1.4979476. Headline result: 83.2% true positive rate at a 5.5% "false positive rate", about **20 false positives per hour**. It was tested on about 3,983 h of recordings from 5 Central African clearings with about 110,042 rumbles. — [ResearchGate record](https://www.researchgate.net/publication/316355452_Automated_detection_of_low-frequency_rumbles_of_forest_elephants_A_critical_tool_for_their_conservation)
  - Training data: 522 h from 8 sites with 10,721 rumbles. Two test sites (Dzanga, Kessala) were not in training. An external test on the Zeppelzauer et al. 2015 data (6 h, 635 rumbles) gave about 85% recall and 88% precision. — [same](https://www.researchgate.net/publication/316355452_Automated_detection_of_low-frequency_rumbles_of_forest_elephants_A_critical_tool_for_their_conservation)
  - **Threshold sweep:** thresholds from 0.2 to 0.95 were evaluated.
    - At threshold 0.4 (the main results), TPR was 79.6–89.35% across sites and FPR 0.75–8.66%.
    - At threshold 0.2, TPR was ≥90% at every site, but FPR ranged from 2.1% to 44.4%.
    - Table III reports threshold 0.3: 87.2% TPR and 8.9% FPR.
    - At 80% recall, precision was ≥70% at all sites, and about 90% at some.
    - F1 peaked at thresholds of 0.6–0.65, so **the authors operated well below the F1-optimal threshold**, i.e. they favoured recall.
    - The authors describe recall >70% as reliable for monitoring.

    — [same](https://www.researchgate.net/publication/316355452_Automated_detection_of_low-frequency_rumbles_of_forest_elephants_A_critical_tool_for_their_conservation)
  - **Metric definition caveat:** the paper's "FPR" is defined as FP/(FP+TP), which is the false discovery rate (1 − precision), not the standard FP/(FP+TN). — [same](https://www.researchgate.net/publication/316355452_Automated_detection_of_low-frequency_rumbles_of_forest_elephants_A_critical_tool_for_their_conservation)
  - **Event rules:** 100 ms frames. A detection must stay above threshold for more than 2 s. A detection and a truth event match if at least 10% of either overlaps the other. — [same](https://www.researchgate.net/publication/316355452_Automated_detection_of_low-frequency_rumbles_of_forest_elephants_A_critical_tool_for_their_conservation)
  - **Classifiers compared by maximum F1:** SVM 0.70, random forest 0.75, AdaBoost 0.73. AdaBoost was chosen for speed. Processing ran at 24 h of audio in about 9 min on a laptop, which the authors report as more than a 300-fold gain over manual review. — [same](https://www.researchgate.net/publication/316355452_Automated_detection_of_low-frequency_rumbles_of_forest_elephants_A_critical_tool_for_their_conservation)
  - The source is internally inconsistent: Table II gives 110,255 rumbles and Table III gives 110,042. — [same](https://www.researchgate.net/publication/316355452_Automated_detection_of_low-frequency_rumbles_of_forest_elephants_A_critical_tool_for_their_conservation)
- **Wrege, Rowland, Keen & Shiu (2017)**, "Acoustic monitoring for conservation in tropical forests: examples from forest elephants," *Methods Ecol. Evol.*, doi:10.1111/2041-210X.12730.
  - This is a perspective/methods paper. It covers PAM for estimating population size, quantifying threats (e.g. gunshots) and assessing conservation strategies.
  - It calls for efficient processing of dense raw data, faster detector development and field data links.
  - The abstract and record give no detector metrics.

  — [bio.acousti.ca record](https://bio.acousti.ca/node/15452); [Methods blog post](https://methodsblog.com/2017/08/11/passive-acoustic-monitoring-elephant-conservation/)
- **Bjorck, Rappazzo, Chen, Bernstein, Wrege & Gomes (2019)**, "Automatic Detection and Compression for Passive Acoustic Monitoring of the African Forest Elephant," AAAI-19, arXiv:1902.09069.
  - **Metric:** test accuracy on *class-balanced* windows. Each call window was paired with an equal number of random empty windows, even though call prevalence is as low as 0.5% at some sites.
  - **Results:** Densenet averaged 89.72% accuracy, against 87.20% for a DNN baseline and about 73–74% for SVM/RF.
  - **Per-site accuracy:** Ceb1 93.40, Ceb4 93.68, Jobo 94.30, Dzanga 77.51.
  - A precision-recall curve is shown, but only qualitatively.
  - Segmentation was reported as accuracy, with a conv-LSTM at 91.71%.

  — [ar5iv full text](https://ar5iv.arxiv.org/html/1902.09069); [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/3820)
- **Geldenhuys & Niesler**, "Learning to rumble: automated elephant call and sub-call classification, detection and endpointing using deep architectures," *Bioacoustics* (2025), doi:10.1080/09524622.2025.2487099 (arXiv:2410.12082, 2024).
  - It uses Elephant Voices (savanna) and LDC Asian elephant data, not ELP forest data.
  - It reports framewise detection **AP = 0.962** and classification AUCs of 0.957 and 0.979.
  - Its literature table quotes Keen 2017 as 87.2% sensitivity and 91.1% specificity, and Bjorck 2019 as 90.8% precision and 96.4% recall in the text but 91.4/91.4 in the table (inconsistent).
  - It warns these numbers are "not directly comparable, as evaluations were performed on different datasets."

  — [arXiv HTML](https://arxiv.org/html/2410.12082); [T&F](https://www.tandfonline.com/doi/full/10.1080/09524622.2025.2487099)
- **Dubus et al. (2026)**, "DeepForestSound: a multi-species automatic detector for PAM in African tropical forests (Kibale NP)," arXiv:2604.08087.
  - The elephant rumble class is evaluated at the 1-minute file level with AP and best-F1. Per-species thresholds were swept over 10⁻⁷ to 1.
  - Elephant (62 positive minutes): DFS-LoRA scored F1 0.901 and AP 0.961, against Perch v2 at F1 0.224 / AP 0.117.
  - No separate threshold-selection set is described, so best-F1 may be optimistic.
  - No recall target or false positives per hour are reported.

  — [arXiv PDF](https://arxiv.org/pdf/2604.08087)

### Inferences
- In ELP practice, the operating point is set below the F1 maximum to keep recall around 80–90%. The cost is accepted as a false-positive *rate per hour* (about 20/h), which analysts then review. This matches the Cornell/Shiu framing in marine work (see Q2).
- The literature's elephant metrics are heterogeneous and partly non-standard: accuracy on balanced windows, the "FPR" defined as FDR, and per-minute AP. Cross-paper comparison is unreliable, and a new study should report recall, precision, FP/hour and the matching rule explicitly.

### Gaps
- I did not retrieve the FruitPunch AI elephant challenge report or any ELP use of BirdNET, AVES or Raven detector metrics. I found no peer-reviewed source for them.
- The Wrege 2017 full text was not read, so it is unconfirmed whether that paper states threshold or verification rules for rumble counts per day.

## Q2. Whale / marine mammal detection: FP/hour at fixed recall, PR curves, or ROC/DET?

### Takeaway
The dominant marine reporting style has two parts:
- precision-recall curves and AP;
- **recall at a fixed false-positive budget per hour**, or FP/hour at a fixed recall.

Operational thresholds are set either by a false-positive budget per hour (Shiu et al. 2020: about 20 FP/h) or by a target precision on a validation set (Allen et al. 2021: about 90% precision). ROC AUC is also reported, but is known to be misleading at low prevalence.

### Cited Findings
- **Shiu, Palmer, Roch, Fleishman, Liu, Nosal, Helble, Cholewiak, Gillespie & Klinck (2020)**, "Deep neural networks for automated detection of marine mammal species," *Scientific Reports* 10:607, doi:10.1038/s41598-020-57549-y (North Atlantic right whale upcalls, DCLDE 2013 data).
  - **Metrics:** precision = TP/(TP+FP) and recall = TP/(TP+FN). AP is the area under the PR curve; LeNet scored 0.852–0.903.
  - **FP/hour framing:** the authors argue that false positives per hour is the more practical measure, because precision does not show how often false positives occur. **About 20 FP per recording hour** is used as the acceptability benchmark and drawn on the figures.
  - **Recall at 20 FP/h:** the best DCLDE 2013 detector recalled about 65% of upcalls, against 85% to over 90% for the DNNs. No DCLDE 2013 detector exceeded 0.83 recall.
  - **Generalization test:** baseline recall was 0.492, against 0.946 and 0.883 for the DNNs.
  - **FP/hour at a fixed recall:** at 0.70 recall, one net produced <0.30 FP/h and another up to 20 FP/h.
  - **Operational rule:** use a low threshold to limit missed calls, and let analysts remove false positives. Set the threshold by the false-positive budget.
  - No ROC or DET curves are used.

  — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6969184)
- **Allen, Harvey, Harrell, Jansen, Merkens, Wall, Cattiau & Oleson (2021)**, "A Convolutional Neural Network for Automated Detection of Humpback Whale Song in a Diverse, Long-Term Passive Acoustic Dataset," *Front. Mar. Sci.* 8:607321, doi:10.3389/fmars.2021.607321.
  - **Unit of evaluation:** 75 s segments.
  - **Overall:** AP 0.97 and ROC AUC 0.992. Per-site AP ranged from 0.93 to 1.00, and per-site PR curves are shown.
  - **Thresholds:** site-specific thresholds (0.06–0.19) were chosen from validation precision-threshold curves, "targeting roughly 90% precision." The resulting test precision was 0.85–0.99 and recall 0.72–0.95.
  - **Baselines:** an energy detector scored AP 0.22 and the GPL detector AP 0.74.
  - **Low-prevalence audit (Kingman Reef):** on 100,164 negatives and 706 positives, the CNN scored ROC AUC 0.973 but AP only 0.456. This shows how ROC AUC overstates performance at low prevalence.

  — [NOAA repository PDF](https://repository.library.noaa.gov/view/noaa/33468/noaa_33468_DS1.pdf)

### Inferences
- "Recall at N FP/hour" (or FP/hour at a fixed recall) is the established operator-facing metric in marine PAM, and is directly relevant for elephant rumbles. ELP and Shiu both come from Cornell and both cite about 20 FP/h.
- The Kingman Reef numbers are strong evidence for reporting PR/AP (or FP/hour) rather than only ROC AUC for rare-call data.

### Gaps
- I did not retrieve DCLDE workshop scoring rules, Mellinger's detector-evaluation papers, or Gillespie/PAMGuard evaluation conventions within the tool budget. DET-curve use in marine PAM is therefore unverified here.

## Q3. Birds and bats: how are score thresholds chosen; is a target recall/precision used?

### Takeaway
Bird guidance has two strands:
- **Knight et al. 2017:** sweep thresholds, always report precision, recall, F-score and AUC (preferring PR-AUC), and pick the threshold to suit the application.
- **Wood & Kahl 2024:** BirdNET scores are not probabilities. Calibrate per species with logistic regression on manually verified detections, then pick the score giving a target probability of a true positive (e.g. 0.9 or 0.95), which is effectively a **precision target**.

Many recent BirdNET papers set species-specific thresholds for precision. Explicit recall targets are rare.

### Cited Findings
- **Knight, Hannah, Foley, Scott, Brigham & Bayne (2017)**, "Recommendations for acoustic recognizer performance assessment with application to five common automated signal recognition programs," *Avian Conservation and Ecology* 12(2):14, doi:10.5751/ACE-01114-120214. Recommendations:
  - Use test data that is independent of the training data.
  - Benchmark against human listening.
  - "Strongly recommend that the influence of score be included in recognizer evaluation." Evaluate across a gradient of thresholds and always report the threshold used. They note that threshold choice is "generally a subjective process based on the priorities of the user."
  - Always report four metrics: **precision, recall, F-score, AUC**. They prefer **PR-AUC over ROC-AUC** because "precision-recall curves are more appropriate for cases with class imbalance," and warn that accuracy is inflated when the target is rare.
  - Evaluate at the application level. They state that occupancy estimates "become unreliable for recognizer data with low recall." Recognizer-based occupancy came out lower than human-listening occupancy and was unstable at high thresholds.
  - Efficiency: a low threshold means "more time required to validate recognizer output," and poor recognizers' "manual postprocessing time required may outweigh the advantages of automation."

  — [ACE article](https://ace-eco.org/vol12/iss2/art14/)
- **Wood & Kahl (2024)**, "Guidelines for appropriate use of BirdNET scores and other detector outputs," *J. Ornithol.* 165:777–782, doi:10.1007/s10336-024-02144-5. Key points:
  - BirdNET confidence scores "are not probabilities, they just look like them." They are species-specific and not transferable.
  - **Option 1:** sort detections by score, verify downward, and choose a cutoff that balances review time and data quality. Thresholds can be species-specific, based on precision and recall on a test set.
  - **Option 2:** for each species, verify a random sample across the score range (oversampling high scores for rare species). Fit a logistic regression of correct/incorrect on the *logit* score, then solve threshold = (ln(p/(1−p)) − β₀)/β_score for the target p. The examples use p = 0.9 and 0.95.
  - The same probability threshold gives different recall across species.
  - Low recall matters for counts and density, but may be acceptable in detection-corrected occupancy models.
  - **Temporal rule:** e.g. require pr(TP) ≥ 0.95 on two or more survey days before declaring presence.
  - Disclose error rates for unverified detections.

  — [author PDF](https://connormwood.com/wp-content/uploads/2024/02/wood-kahl-2024-guidelines-for-birdnet-scores.pdf); [OUCI record](https://ouci.dntb.gov.ua/works/4yX5omA7)
- **Scanferla et al. (2025)**, "Determining species-specific thresholds to improve precision in passive acoustic monitoring," *Ecological Informatics* 91:103423, doi:10.1016/j.ecoinf.2025.103423.
  - They derived 72 BirdNET species-specific thresholds by binomial logistic regression from 25 h of Alpine recordings (17,737 annotated sounds).
  - The thresholds raised precision by a mean of 0.30.
  - The framing is precision-oriented; the recall trade-off is not stated in the abstract.

  — [unibz record](https://bia.unibz.it/esploro/outputs/journalArticle/Determining-species-specific-thresholds-to-improve-precision/991007132910001241)
- A 2025 *J. Ornithol.* paper, "Setting BirdNET confidence thresholds: species-specific vs. universal approaches," doi:10.1007/s10336-025-02260-w, exists, but its content could not be fetched. — [Springer](https://link.springer.com/article/10.1007/s10336-025-02260-w)
- **Bats: Mac Aodha et al. (2018)**, "Bat detective—Deep learning tools for bat acoustic signal detection," *PLoS Comput. Biol.* 14(3):e1005995.
  - Detectors were evaluated with **average precision and recall** on expert-verified test sets from different sensors and countries.
  - Commercial systems had precision >0.7 but lower recall. The CNNs did better on noisy road-transect recordings.

  — [bioRxiv abstract](https://www.biorxiv.org/content/10.1101/156869v1.full.pdf+html); [Edinburgh record](https://www.research.ed.ac.uk/en/publications/bat-detective-deep-learning-tools-for-bat-acoustic-signal-detecti/)

### Inferences
- Bird practice operationalises thresholds mostly as a **precision floor**, i.e. a calibrated probability that a detection is a true positive. This suits large multi-species surveys with limited verification. For a single rare species, the same calibration machinery could instead be inverted to hit a target recall.

### Gaps
- I could not retrieve the Bat Detective full text (matching tolerance, recall at fixed precision) or BatDetect2 (2022 preprint) evaluation details.

## Q4. Explicit guidance that recall should be prioritised for rare/endangered species or occupancy, and how

### Takeaway
Explicit statements exist but are few:
- **Knight et al. 2017** recommend an F-beta that weights recall (their text says β < 1, which in the usual F-beta convention would weight precision; see the note below) for rare species and legally sensitive cases. They also warn that low recall makes occupancy unreliable.
- **Shiu et al. 2020** recommend a low threshold plus analyst review, capped by a false-positive-per-hour budget.
- **Wood & Kahl 2024** take a more nuanced view: occupancy models can tolerate low recall, so precision may matter more there.

Human-in-the-loop verification of the false positives that a high-recall threshold produces is the standard mechanism. Its cost is the main constraint, and it can be very large for rare species: the red uakari study had 0.35% precision and needed about 80 h of review.

### Cited Findings
- Knight et al. 2017 suggest **F-beta favouring recall** for "detection of rare species or situations with legal implications." Their text gives this as β < 1. They also say to always report F1 alongside for comparability. — [ACE](https://ace-eco.org/vol12/iss2/art14/)
  - **Note for the writer:** in the standard van Rijsbergen convention, β > 1 weights recall. The β < 1 is as the fetched summary reports it, so check the original text before quoting the direction.
- Knight et al. 2017: occupancy estimates "become unreliable for recognizer data with low recall," because "species detection probability is too low for reliable occupancy estimates." — [ACE](https://ace-eco.org/vol12/iss2/art14/)
- Shiu et al. 2020: in operations, set a low threshold to reduce missed calls and let analysts remove false positives, bounded by about 20 FP/hour. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6969184)
- Keen et al. 2017 (ELP) operated at thresholds 0.3–0.4, well below the F1-optimal 0.6–0.65, which keeps recall at about 80–89%. — [ResearchGate](https://www.researchgate.net/publication/316355452_Automated_detection_of_low-frequency_rumbles_of_forest_elephants_A_critical_tool_for_their_conservation)
- Wood & Kahl 2024: whether to prioritise precision or recall depends on the application. Low recall is a problem for counts, density and behaviour, but may be acceptable in detection-corrected occupancy models. — [PDF](https://connormwood.com/wp-content/uploads/2024/02/wood-kahl-2024-guidelines-for-birdnet-scores.pdf)
- **Katsis, Rhinehart, Dorgay, Sanchez, Snaddon, Doncaster & Kitzes (2025)**, "A comparison of statistical methods for deriving occupancy estimates from machine learning outputs," *Sci. Rep.* 15:14700, doi:10.1038/s41598-025-95207-3 (Yucatán black howler monkey).
  - **Threshold:** 0.77, giving 85% recall and 92% precision on validation data, chosen as "a good balance." The authors recommend picking a range where neither metric is badly compromised.
  - **Verification strategy:** "top-ten" targeted listening per site gave ψ = 0.57 against a reference of 0.55, after about 15 min verification per site. Random verification of 10 files per site underestimated occupancy at ψ = 0.26.
  - **Rare species:** the authors argue random verification is unlikely to find positives for rare, infrequent callers, so targeted (highest-score) listening is preferable. This is a reasoned argument, not a tested result.

  — [Southampton eprint](https://eprints.soton.ac.uk/502035/1/s41598-025-95207-3.pdf)
- **Rare primate example: red uakari (*Cacajao ucayalii*)**, bioRxiv 2025, "Toward passive acoustic occupancy surveys of rare red uakari monkeys...".
  - **Classifier:** a Kaleidoscope cluster classifier returned 100,321 detections over 7,356 h, of which only 356 were true positives (precision about 0.0035). True positives fell in 54 hours.
  - **Verification:** **all detections were manually verified** by two researchers, taking about 80 h. Stopping at the first detection per station per day was estimated to cut this to about 30 h (a projection, not tested).
  - **Occupancy:** naive occupancy was 0.688, against a modelled ψ of 0.818 with p = 0.068.
  - Recall was not measured.

  — [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.10.637398.full.pdf)

### Inferences
- The practical recipe in the literature for rare-species detection runs in five steps:
  1. Sweep thresholds and report full PR curves and AP, not ROC-AUC alone.
  2. Pick a low threshold that achieves high recall, below the F1 optimum.
  3. Express the cost as false positives per hour or per day, i.e. review workload.
  4. Verify detections by hand: all of them, top-k per site/day, or a calibrated sample.
  5. Report recall explicitly, since low recall biases occupancy and counts.
- No reviewed source sets a formal fixed-recall standard (e.g. "≥95% recall"). Recall targets are implicit (Keen: >70% is "reliable"; operating points about 80–90%). The formal targets that do appear are on precision (Allen: about 90%; Wood & Kahl: pr(TP) of 0.9–0.95) or on FP/hour (Shiu: about 20/h).

### Gaps
- I found no peer-reviewed study that formally optimises the threshold against a monetised cost of manual review versus a missed call, or that adopts a cost-sensitive / fixed-recall criterion specifically for an endangered species. Such work may exist (e.g. Clare et al. 2021 or Knight et al. 2020 on false-positive occupancy, both cited by Wood & Kahl), but it was not retrieved.
