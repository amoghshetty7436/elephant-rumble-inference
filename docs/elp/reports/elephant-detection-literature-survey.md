# Elephant rumble detection: literature survey and a proposal to beat FruitPunch

*Prepared 30 October 2026 for the ELP rumble-detector project. Every source below was found and read (in full or in large part) by web search in October 2026 unless marked "not re-read". Numbers are quoted as the papers report them. Where a paper does not say something, this report says so.*

---

## 1. Summary

**What the literature says, in five points**

1. **Forest-elephant detection from long passive recordings has very few published papers.** The main ones are Keen et al. 2017 (hand-built features + AdaBoost, ~4,000 h, 5 clearings), Bjorck et al. 2019 (DenseNet + conv-LSTM, ~4,900 h, 4 clearings), the unpublished Stanford/Conservation Metrics detectors used by ELP, and DeepForestSound 2026 (LoRA-tuned Audio Spectrogram Transformer, one Ugandan forest). None of them reports results on *forest-floor* (non-clearing) recorders with a sealed, fully browsed test set the way we do. So **there is no published number to beat on our task**; the real bar is FruitPunch, Stanford (P 0.842 / R 0.746 on ELP's Sample A) and our own current model.
2. **Every recent winner adapts its backbone to the target sounds, and frozen embeddings plus a small head lose.** DeepForestSound's LoRA-tuned AST reached AP 0.961 on elephant rumbles, against 0.672 for the same frozen encoder with a linear head and 0.117 for frozen Perch 2.0 ([Dubus et al. 2026](https://arxiv.org/pdf/2604.08087)). In DCASE sound event detection, fine-tuning the transformer and then self-training on pseudo-labels gave the largest gains ([Schmid et al. 2024](https://arxiv.org/pdf/2408.00791)). In elephant call classification, frozen out-of-species embeddings come within ~2% AUC of a supervised AST but stay clearly behind on average precision ([Geldenhuys & Niesler 2026](https://arxiv.org/html/2605.00225)).
3. **A temporal model over per-frame features is the most consistently reported fix for false alarms in low-frequency detection.** For 20 Hz fin-whale notes (the closest acoustic analogue to rumbles), an LSTM over CNN features on ~111 s of context raised AUC-PR from 0.82 to ~0.95 and **cut false positives per hour by 59–85%** ([Madhusudhana et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8292017)). Bjorck et al. found the same for rumbles: conv-LSTM segmentation 91.7% vs plain LSTM 69.7%. DCASE winners all put a BiGRU on top of the transformer.
4. **Bird-trained supervised embeddings (Perch 2.0, BirdNET) transfer surprisingly well to elephants and whales, but in-domain self-supervised pretraining wins when there is a lot of unlabelled in-domain audio.** animal2vec, pretrained on 1,068 h of unlabelled meerkat audio with masking adapted to sparse calls, reached micro-AP 0.91 vs 0.30 for a speech-style baseline ([Schäfer-Zimmermann et al. 2024](https://arxiv.org/html/2406.01253v2)). ELP has hundreds of thousands of unlabelled hours. At our 32× speed-up, a 24-hour day is only 45 minutes of model audio, so this archive is a very large pretraining corpus by speech standards. **Nobody has used it for pretraining yet.**
5. **Elephant papers are mostly not comparable with each other.** They use balanced clips, accuracy, per-minute AP or ROC AUC, and different matching rules. Our protocol (whole fully browsed days, FP/h at fixed recall, sealed tests, 5 seeds) is stricter than anything published. Keep it.

**The proposal (section 6).** Build a **sequence-aware, two-stream, domain-adapted rumble detector** ("RumbleNet" as a working name):

- **(a)** AVES adapted with LoRA, plus a learned weighted mix of its layers;
- **(b)** a small parallel low-frequency spectrogram branch with PCEN (5–250 Hz), aimed at the vehicle/engine and rain/thunder false alarms;
- **(c)** a BiGRU/Conformer temporal head over 60–120 s of frames, trained on whole stretches rather than single frames;
- **(d)** noisy-student self-training on unlabelled forest days, which also cleans the "not every rumble marked" negatives in the clearing data;
- **(e)** optionally, continued self-supervised pretraining of AVES on unlabelled ELP audio at 32× speed.

Each piece has published evidence behind it, and none has been combined for elephants. The experiment plan (section 7) adds them one at a time against our existing yardstick, cheapest first, so each gain is measured, not assumed.

**What I recommend running first:**

1. The temporal head on the cached frozen fingerprints (hours, no re-extraction).
2. Let the AVES-tuning run finish. So far frozen layer 12 beats the layer mixes and the first LoRA run.
3. Combine the two only if tuning finds a winner.
4. Then the low-frequency branch and self-training.

Continued pretraining is the high-cost, high-ceiling bet. Run it only if steps 1–4 leave a clear gap.

---

## 2. Where we start from (project context, not re-derived here)

From our own runs (`results/improve-run/final.md`, `walkthrough/progress-2026-10-20.md`, `walkthrough/06…`, `walkthrough/14…`):

| Model | Test set | Recall | Precision | FA/h | Notes |
|---|---|---|---|---|---|
| FruitPunch as published (balanced clips) | FruitPunch's own split | "high 90s" | "high 90s" | – | Balanced clips, elephant-free days dropped, so not comparable |
| FruitPunch on whole days (our replication) | Sample A, thr 0.90 | 0.70 | 0.71 | – | Threshold picked on test |
| Stanford detector (ELP Performance Methods doc) | Sample A, 42 days, 970 rumbles | 0.746 | 0.842 | – | ELP's own scoring, 1% overlap |
| Hori-Harm (ELP, thr 0.4) | Sample A | 0.60 | 0.343 | – | |
| FruitPunch's data and alarm rule, retrained in the full rerun (5-seed mean) | Sample A / Sample B | 0.764 / 0.896 | 0.631 / 0.499 | 0.29 / 0.34 | 80% cut-off from validation |
| **Our current best**: full rerun final `E2_ctx8_noise10_rand3000_gap0.5_hard0` (AVES L12 frozen + MLP, 10.88 s window, 254 noise clips ×10, 3,000 random forest frames/day, 0.5 gap frames per rumble frame; rule on > 0.2, ≥ 3 frames, smooth 5, gaps ≤ 3, score = max; 5-seed mean) | Sample A | 0.763 | 0.874 | 0.06 | 80% cut-off from validation |
| same, 85% cut-off | Sample A | 0.801 | 0.829 | 0.09 | |
| same | Sample B (2023) | 0.910 | 0.819 | 0.06 | 80% cut-off (85%: 0.931 / 0.758 / 0.09) |
| same | Vehicle days | – | – | 0.25 | False alarms only (85%: 0.37) |

Validation (Complex + General): score 0.159 (mean of FA/h at 80% and 85%), **0.130 FA/h at 80% recall**, 0.316 at 90%, max recall 0.974. On Sample A this is already ahead of Stanford on both precision (0.874 vs 0.842) and recall (0.763 vs 0.746). It is not an exact comparison: Stanford's figures come from ELP's own evaluation document on the 42 Schedule-File days, while ours are on 54 days, scored by our Python copy of ELP's rule. We did not run the Stanford model ourselves. Source: `final.md` and `study.md` from the full rerun.

What we have already learned, and what the proposal must respect:

- **Context helps, up to about 11 s with concatenation.** In the full rerun, widening the MLP's window from 0.64 s to 10.88 s was one of the biggest gains (validation score 0.271 → 0.220; vehicle FA/h 0.73 → 0.44). 12.16 s was slightly worse. (The earlier, shorter improvement run had found no gain from wider windows, so the rerun's longer training mattered.) A temporal *model* (weights shared over time, gating, far fewer parameters per second of context) is the natural way to look further than concatenation allows.
- **Negatives matter most.** Noise clips, even at 7% of negatives, were the biggest single gain. Hard-negative rounds helped a little at 85–90% recall.
- **Clearing data helps through sheer quantity** (31k rumbles vs 3.2k forest). It has unmarked rumbles in its "negatives".
- **Alarm rules matter.** Tuned smoothing, minimum length and a 0.2–0.25 "on" level beat the 0.5/4-frame rule.
- **Pipeline:** 500 Hz audio presented to AVES as 16 kHz (32× speed-up). One AVES frame is 0.64 s of real audio. The target is CPU inference at ELP scale (4,500 recording-days on 30 cores in under 5 days).
- **AVES tuning run (in progress, results up to 28 Oct):** frozen layer 12 was best (score 0.134, one run). The learned layer mix scored 0.150, mean of the last 4 layers 0.213, L10 0.318, L8 0.361, and BirdAVES-biox-base and bioxn-large were worse still. The first LoRA run (rank 4, top 4 layers) stopped at its starting point: held-out AP never beat the untuned 0.9815, so it scored the same 0.134. Rank 8 and top-layer fine-tuning were still running.

---

## 3. Comparison table: elephant detection and classification papers

"Det." = detection in continuous audio. "Cls." = classification of pre-cut segments. Metrics are not comparable across rows (see section 5).

| # | Paper | Species / setting | Model | Input | Data | Headline result | Main limitations | Code / data |
|---|---|---|---|---|---|---|---|---|
| E1 | [Keen, Shiu, Wrege & Rowland 2017, JASA 141(4):2715](https://doi.org/10.1121/1.4979476) | Forest elephant, 5 Central African clearings (Gabon, CAR) | Hand-designed **harmonic** + **horizontal-energy** features per 100 ms frame. Linear SVM, random forest and AdaBoost compared (max F1 0.70 / 0.75 / 0.73); AdaBoost kept for speed | Spectrogram features, 100 ms frames; event ≥ 2 s above threshold | Train 522 h, 8 sites, 10,721 rumbles. Test 3,983 h, 5 sites, ~110k rumbles | **Det.: 83.2% recall at ~20 FP/h**, operated at thr 0.3–0.4 (recall 79.6–89.4% by site) | Much worse at the 2 test sites absent from training. Some rumbles missed even at the lowest threshold. "FPR" defined as 1 − precision | No code found |
| E2 | [Bjorck et al. 2019, AAAI](https://ar5iv.arxiv.org/html/1902.09069) | Forest elephant, Ceb1, Ceb4, Jobo (Gabon), Dzanga (CAR) | **DenseNet** for 25.5 s window classification; **conv-LSTM** for frame segmentation; learned compression | 1 kHz, FFT 512 / hop 384, < 100 Hz, 64×47 tensor; normalised by call-free frames | 4,899 labelled h, 168,249 calls; calls are 0.5–0.8% of audio except Dzanga (21.8%). Random split | Det.: **89.7% balanced-window accuracy** (Dzanga 77.5%); segmentation 91.7% (conv-LSTM) vs 69.7% (LSTM) | Balanced windows only, not whole days. Random split (leakage likely). No FP/h. No comparison with Keen | Not released |
| E3 | Gomes-Selman, Demir, Paepcke & Wrege 2020 (ESA talk; "Stanford detector"), [press release](https://esa.org/blog/2020/07/24/machine-learning-streamlines-detection-of-african-forest-elephant-vocalizations/) | Forest elephant, Nouabalé-Ndoki forest grid (our sites) | Neural network, architecture not published | – | PNNN recordings | "Reduces audio needing inspection by 98%". **ELP's scoring: P 0.842 / R 0.746 on Sample A** (internal doc) | Not peer reviewed. No score column (per ELP's doc) | Not public |
| E4 | Conservation Metrics detector, [Cornell 2018 news](https://cals.cornell.edu/news/2018/08/ai-speeds-effort-protect-endangered-elephants) | Forest elephant, Nouabalé-Ndoki, 50 sensors | "Deep neural networks", unspecified | – | ~7 TB / 3 months | Analysis 22 days vs ~3 months; "higher accuracy" (no numbers) | No published metrics | Commercial |
| E5 | [Zeppelzauer, Hensman & Stoeger 2015, Bioacoustics 24(1)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4430816) | Savanna elephant, 5 habituated animals, S. Africa | **Structure-tensor spectrogram enhancement** + Greenwood cepstral features + RBF-SVM | 0–500 Hz, 300 ms frames | ~6 h, 635 rumbles, 1 site | Det.: 88.2% detection rate; FPR 13.7% vs 24.4% without enhancement | Engine noise (car, aircraft) causes false alarms; distant rumbles lose harmonics; tiny training set (63 rumbles) | – |
| E6 | Venter & Hanekom 2010, [UP repository](https://repository.up.ac.za/handle/2263/14186) | Savanna elephant | Pitch/subharmonic-based detector (not re-read) | – | – | P/R ≈ 85.7% (as tabulated by Geldenhuys & Niesler) | Not re-read | – |
| E7 | [Geldenhuys & Niesler 2024/25, "Learning to rumble"](https://arxiv.org/abs/2410.12082) ([Bioacoustics version](https://www.dsp.sun.ac.za/~trn/reports/geldenhuys+niesler-bioacoustics25.pdf)) | Savanna (ElephantVoices, Kenya/Mozambique) and Asian (LDC, Sri Lanka) | LR, SVM, XGBoost, MLP, AlexNet, ResNet, **AST-cls**, **AST-seq** (AudioSet-pretrained) | 16 kHz log-mel, 128 bins, 25 ms / 10 ms | EV 514 calls (~1 h); LDC 4,433 calls (57.5 h) | Det. (frame, 100 ms): **AST-seq AP 0.968 (EV), 0.772 (LDC)**; ResNet 0.957 / 0.606 | Handheld, small data, CV within datasets. AST-seq over-segments long-tailed calls. Not long-term PAM | Supplemental data only |
| E8 | [Geldenhuys & Niesler 2026, "From Birdsong to Rumbles"](https://arxiv.org/html/2605.00225) | Same EV / LDC | **Frozen** embeddings (BirdNET, Perch 1/2, SurfPerch, AVES family, BirdAVES, animal2vec, BEATs, wav2vec2, HuBERT, VGGish) + LR/MLP/GRU/LSTM | Each model's native input | Same | Cls.: Perch 2.0 AUC 0.849 (EV) / 0.936 (LDC), within ~2% of supervised AST, but **AST keeps a clear AP lead (≈0.18 on LDC)**. Intermediate layers often beat the last. Marine-only models transfer poorly | Endpoints given (no detection). No rumble-specific resampling | Not stated |
| E9 | [Dubus et al. 2026, DeepForestSound](https://arxiv.org/pdf/2604.08087) | Forest + savanna elephants (genus level) plus 11 other taxa; Kibale NP, Uganda | **AST (AudioSet) fine-tuned with LoRA** (1.04M of 88M params trainable); separate low-frequency config | **Elephant LF config: 1 kHz, 128 mel bands 3–250 Hz**, 750 ms window, 35 ms hop, 10 s excerpts, SpecAugment, mixup | 1,418 elephant calls (506 PAM + 912 from camera-trap videos). Training set built semi-automatically (energy detector, then BirdNET/Perch/AST embeddings, UMAP+HDBSCAN, then manual check). Test: 1,200 one-minute files from 20 recorders, 2025 | **Elephant AP 0.961, best F1 0.901** vs DFS-Linear 0.672, Perch 2.0 0.117, RDet 0.101 | One forest. Per-minute presence scoring (not events). No validation set for thresholds. No mixup ablation | [Code + weights](https://github.com/MNHN-OFVI/DFS--multi-species-automatic-detector-for-pam-in-african-tropical-forests); audio not shareable |
| E10 | [Pickering, Martinez Balvanera, Jones & Hedwig 2025, RSEC](https://discovery.ucl.ac.uk/id/eprint/10206477) | Forest elephant, Dzanga Bai (CAR) | Frozen **VGGish, YAMNet, Perch, BirdNET** embeddings + random forest; UMAP clustering; GLMs | 4 kHz, **time-expanded** to fit model input (e.g. 5 s → 40 s for Perch) | 1,254 calls (779 rumbles, 424 roars, 51 trumpets), handheld mic | Cls.: overall accuracy 0.96–0.97; rumble F1 1.0 for all models; BirdNET best rumble silhouette (0.89) | Pre-cut calls, close-range mic. Not detection | [Code](https://github.com/AlastairPickering/elephant_transfer_learning) |
| E11 | [Dewmini, Meedeniya & Perera 2025, Sensors 25:352](https://www.mdpi.com/1424-8220/25/2/352) | Asian elephant | ElephantCallerNet (raw-waveform CNN, ACDNet-style) vs MobileNetV2, YAMNet, RawNet, ResNet18, SVM | Raw 6 s clips | 235 recordings (roar / rumble / trumpet); web-sourced in part | Cls.: 89% accuracy | Tiny; mixed web sources; no noise robustness | – |
| E12 | [Kahoro & Muriithi 2026, ICAART](https://www.scitepress.org/Papers/2026/142512/142512.pdf) | Elephant vs forest sounds, early warning | 1D CNN (int8 TFLite on RPi Pico) vs SVM, LR, RF, NB | MFCC etc., 16 kHz | 686 clips (ELP ethogram + FSC22) | Cls.: 99% accuracy | Balanced clips, secondary data, no field test. Illustrates how balanced-clip scores mislead | – |
| E13 | [Costa et al. 2023/24, seismic rumbles](https://ar5iv.labs.arxiv.org/html/2312.02831) ([journal](https://doi.org/10.3390/asi7060117)) | Asian elephant, geophone | MFCC + Ridge classifier (LazyPredict sweep); structure-tensor denoising | 475 Hz seismic | Orphanage recordings | Ridge + MFCC best (numbers not in fetched text) | Small, one site | – |
| E14 | [Bandara & Bandara 2023, Elemantra](https://arxiv.org/pdf/2310.15012v1.pdf) | Asian elephant HEC system | Rule: seismic peak 20–40 Hz over 4 s; YOLOv7-tiny for IR images | – | 44 seismic files | Seismic recall 0.82 vs STFT reference | Rule-based; preliminary | [Code](https://github.com/nuwansribandara/elemantra) |
| E15 | [Wrege et al. 2017, MEE](https://zendy.io/title/10.1111/2041-210x.12730) and [Wrege et al. 2024, PLOS ONE](https://pmc.ncbi.nlm.nih.gov/articles/PMC11280225/) | Forest elephant ecology from ELP PAM | Use ELP's production detectors with human review (details not re-read) | – | 2024: deployments 1–10, Nouabalé-Ndoki; [Dryad data](https://datadryad.org/dataset/doi:10.5061/dryad.x95x69prm) | Ecological results (activity, gunshots) | Detector performance not the focus | Detection tables on Dryad |
| E16 | FruitPunch AI "AI for Forest Elephants" ([repo](https://github.com/ramayer/elephant-rumble-inference)) | Forest (ELP clearings + PNNN) | **AVES-bio (HuBERT) frozen, 32× speed-up**, MLP on last-layer embeddings | 500 Hz shown as 16 kHz; 0.64 s frames | 56 clearing + ~110 forest days | "High 90s" on balanced clips; **P 0.245** when ELP ran it on whole days; ~0.71/0.70 at thr 0.90 (our replication) | Balanced-clip testing, threshold on test, negatives only from gaps next to rumbles | Open |

Other elephant work, cited from Geldenhuys & Niesler's comparison table and not re-read: Clemins & Johnson 2003 (HMM, 5-class accuracy 79.7%), Stöger et al. 2012 (2-class 99.0%), Silva 2017 (Asian, P/R 84%), Leonid & Jayaparvathy 2022 ([IASC 32(3)](https://techscience.com/iasc/v32n3/45906); accuracy 96.2%, recall 88.2%).

---

## 4. Related work that matters for transfer

### 4.1 Pretrained encoders for animal sound

| Encoder | What it is | Evidence relevant to us |
|---|---|---|
| **AVES** ([Hagiwara 2023](https://arxiv.org/abs/2210.14493)) | HuBERT (95M) self-supervised on animal audio at 16 kHz | Our current backbone. In marine few-shot tests AVES was "weaker when examples were scarce" than Perch 2.0 / BirdNET ([Perch whale paper](https://arxiv.org/pdf/2512.03219)). Frozen AVES ranked below Perch 2.0 on elephant call types (E8) |
| **Perch 2.0** ([van Merriënboer et al. 2025](https://arxiv.org/html/2508.04665v1)) | EfficientNet-B3, supervised on 1.54M recordings / 14,795 classes, plus self-distillation and source prediction; 32 kHz, mel 60 Hz–16 kHz | Best frozen embedding on elephant call types (E8) and on marine tasks including baleen whales (AUC 0.86–0.92 on NOAA PIPAN). **But frozen Perch 2.0 scored AP 0.117 on DFS's elephant detection** (E9). Its 60 Hz mel floor means rumbles must be sped up to be seen at all |
| **BirdNET** ([Kahl et al. 2021](https://doi.org/10.1016/j.ecoinf.2021.101236)) | ResNet-style, 48 kHz, bird-supervised | Close second to Perch 2.0 on marine tasks; best rumble cluster separation in E10. Used as a time-expanded feature extractor for forest elephants (E10) |
| **BEATs** ([Chen et al. 2022](https://arxiv.org/abs/2212.09058)) | AudioSet SSL transformer | Best non-bioacoustic model on LDC (E8). A backbone in DCASE SED winners |
| **AVEX / "What matters for bioacoustic encoding"** ([ESP 2025, ICLR 2026](https://arxiv.org/pdf/2508.11845)) | Systematic study: EfficientNet, BEATs, EAT, AVES with different pretraining | **SSL pretraining, then supervised post-training on bioacoustic + general audio, was best.** SSL models generalised better out of distribution (retrieval AUC drop ~0.01 vs ~0.09 for supervised). All at 16 kHz; no low-frequency work |
| **animal2vec** ([Schäfer-Zimmermann et al. 2024](https://arxiv.org/html/2406.01253v2)) | data2vec-2.0-style SSL transformer, SincNet front end; pretrained on **1,068 h of unlabelled in-domain meerkat audio** | Designed for **sparse, rare events**: masking changed to suit short calls in long noise, focal loss in fine-tuning. Micro-AP 0.91 vs 0.30 for a speech-style baseline; **0.83 with only 1% of labels**. Masking and frame-rate choices mattered most (+0.11 and +0.16 AP). [Code](https://github.com/livingingroups/animal2vec). Cost: ~20 days on 4×A100 for a 315M model |
| **Ghani et al. 2023** ([Sci. Rep.](https://arxiv.org/abs/2307.06292)) | Bird embeddings for transfer | Bird-trained supervised embeddings beat general audio embeddings for few-shot bioacoustic transfer |

**Takeaway for us.** No frozen encoder is close to good enough for rumble *detection*; DFS shows a gap of 0.3 AP or more between frozen and adapted. Among frozen encoders, Perch 2.0 and BirdNET look slightly better than AVES for elephants, but the frame rate (5 s windows for Perch) is wrong for event detection. **Adapting AVES (which gives 0.64 s frames at our speed-up) is the better base than switching encoders.** A frozen Perch 2.0 probe is still worth running as an external baseline.

### 4.2 Low-frequency detectors in other species (the closest analogues)

| Paper | Signal | Model | Key result for us |
|---|---|---|---|
| [Shiu et al. 2020, Sci. Rep.](https://repository.library.noaa.gov/view/noaa/27588/noaa_27588_DS1.pdf) | Right-whale upcalls, 39–358 Hz | LeNet, BirdNET, VGG, ResNet, Conv1D+GRU | Recall at 20 FP/h: 85% to over 90% vs 65% for the best old detector. **Generalised to new regions and years** (0.946 vs 0.492). **Hard-negative mining: AP 0.852 → 0.903.** Small LeNet was best |
| [Madhusudhana et al. 2021, J. R. Soc. Interface](https://pmc.ncbi.nlm.nih.gov/articles/PMC8292017) | Fin-whale 20 Hz notes, 10–54 Hz, 500 Hz audio | Small DenseNet (4 s windows) + **LSTM over 108 steps (~111 s)** | **AUC-PR 0.82 → 0.95; peak F1 0.79 → 0.91; FP/h at peak F1 cut 59–85%.** LSTM on CNN *features* beat LSTM on *scores*. Lower variance across folds. [Code](https://github.com/shyamblast/TemporalContext-2021) |
| [Allen et al. 2021, Front. Mar. Sci.](https://www.frontiersin.org/articles/10.3389/fmars.2021.607321/text) | Humpback song, 187k h | ResNet-50 | Per-site thresholds at 90% precision; ROC AUC 0.97 can hide AP 0.46 at rare-call sites (see our earlier report) |
| [Miller et al. 2023, RSEC](https://doaj.org/article/e15ca795bc38445b96d206cdc75a0175) | Blue-whale D-calls (Antarctic) | DenseNet | Beat an experienced human analyst on recall and false positives, mostly at low and medium SNR |

**Takeaway.** For long, low, slowly modulated calls, the decisive design choice was **temporal context modelled by a recurrent layer on learned features**, plus hard negatives. Bigger 2-D backbones mattered less.

### 4.3 General sound event detection (DCASE) practice

- **Architecture of current winners:** pretrained transformer (ATST-Frame, BEATs, PaSST) giving frame features → CNN → **BiGRU** → frame posteriors. Training runs in two stages (head with backbone frozen, then fine-tune the backbone), with mean-teacher consistency on unlabelled data, then **pseudo-labels from an ensemble and a second training iteration** ([Schmid et al. 2024](https://arxiv.org/pdf/2408.00791); [ATST fine-tuning, Shao et al.](https://ar5iv.labs.arxiv.org/html/2309.08153)). The ablations: **second pseudo-label iteration was the largest gain**, fine-tuning the transformer was second, and ensembling helped PSDS.
- **Frequency-aware convolutions:** frequency dynamic convolution adapts kernels to frequency position ([Nam et al. 2022](https://arxiv.org/pdf/2203.15296); [multi-dilated FDY 2024](https://arxiv.org/pdf/2406.13312)). This is relevant because a rumble's fundamental and harmonics sit at fixed, known bands, unlike in natural images.
- **Post-processing:** Sound Event Bounding Boxes decouple event extent from confidence with change-point detection. They raised PSDS1 from 0.644 to 0.686 on DCASE 2023 systems with no retraining ([Ebbers et al. 2024](https://www.arxiv.org/abs/2406.04212); [`sebbs` on PyPI](https://pypi.org/project/sebbs/)). This complements our tuned alarm rules. Our earlier report covers the related threshold and median-filter literature.

### 4.4 Front ends and normalisation

- **PCEN** replaces the log in a mel spectrogram with adaptive gain control per band. In bird flight-call detection across sensors and through the night it raised AUPRC from 56.1% to 66.0% and stabilised recall as insect noise changed ([Lostanlen et al. 2019](https://pdf.arxiv.org/pdf/1905.08352)). It helped with temporal noise changes, not with differences between sensors. **Caveat:** its smoothing time T must be longer than the event, or a 3–8 s rumble is partly absorbed into the background estimate. Our inference from the paper: T of about 10–30 s, to be tuned.
- **Structure-tensor enhancement** (E5, E13) brings out harmonic contours and halved false positives on savanna rumbles. It is an old-fashioned but cheap extra input channel.
- **Speed-up / time expansion:** FruitPunch (32×) and Pickering (time expansion into Perch/BirdNET) both shift rumbles into the band the pretrained model knows. DFS instead builds a native 1 kHz, 3–250 Hz mel front end and fine-tunes. **No paper compares the two approaches head to head.** That is an open question we can answer.

### 4.5 Label efficiency, unlabelled data and domain shift

- **Agile modelling with embeddings and active learning** ([Dumoulin et al. 2025](https://arxiv.org/html/2505.03071v3)): vector search plus a linear probe reached ROC-AUC 0.83–1.0 with under an hour of analyst time per classifier, and was 43× faster than manual scanning to find positives. Useful for **mining new labelled forest rumbles and hard negatives** quickly from unlabelled ELP days.
- **Source-free domain adaptation in bioacoustics** ([Boudiaf et al. 2023, ICML](https://arxiv.org/abs/2302.06658v1)): on natural bioacoustic shifts, some standard adaptation methods were *worse than no adaptation*, and the method rankings differed from vision. Their NOTELA method was the most robust. **Lesson:** test any adaptation or self-training step on our validation days, and never assume it helps.
- **Self-training with pseudo-labels** gave the largest single DCASE gain (above). Noisy Student ([Xie et al. 2020](https://arxiv.org/abs/1911.04252)) and Mean Teacher ([Tarvainen & Valpola 2017](https://arxiv.org/abs/1703.01780)) are the standard recipes.
- **Parameter-efficient fine-tuning:** LoRA ([Hu et al. 2021](https://arxiv.org/abs/2106.09685)) made DFS trainable on a laptop GPU (8 epochs × ~6 h). LoRA weights merge into the backbone after training, so **inference cost is unchanged**, which matters for ELP's CPU budget.
- **Layer choice:** in SSL speech models, a learned weighted sum of layers (the SUPERB protocol, [Yang et al. 2021](https://arxiv.org/abs/2105.01051)) is the standard probe. E8 found intermediate layers often beat the last for elephants.

---

## 5. How to read the elephant numbers (comparability)

| Paper | Unit scored | Negatives realistic? | Split | Comparable with our whole-day FP/h? |
|---|---|---|---|---|
| Keen 2017 | Events in continuous days, ≥ 10% overlap | Yes | By site (train 8, test 5) | **Closest.** But clearings, not forest floor |
| Bjorck 2019 | Balanced 25.5 s windows | No (50/50) | Random | No |
| Geldenhuys & Niesler | 100 ms frames, AP / AUC | Partly (handheld recordings) | CV within dataset | No |
| DFS 2026 | One-minute file presence | Yes (2025 recorders) | Different recorders and time | Partly (presence, not events) |
| FruitPunch | Balanced clips | No | By day | No |
| Stanford (ELP doc) | Events, 1% time and frequency overlap | Yes | Sample A | **Yes**, if scored on the same 42 Schedule-File days |
| Ours | Events, ELP rule, FP/h at fixed recall | Yes (fully browsed days incl. elephant-free days) | Validation days; sealed tests from other years | – |

**Implication.** To claim "beats FruitPunch and other models", score on the 42 Schedule-File days of Sample A (to match Stanford) and on Sample B. Also re-run two external baselines inside our pipeline:

- **(i)** a DFS-style LoRA-AST with its LF front end (code is public);
- **(ii)** a frozen Perch 2.0 probe on time-expanded audio.

Run them with the same data, alarm-rule tuning and seeds. That makes "other models" concrete rather than rhetorical.

---

## 6. Proposal: a sequence-aware, two-stream, domain-adapted rumble detector

### 6.1 Why our current model leaves accuracy on the table

| Weakness of current model | Evidence it matters | Fix in the proposal |
|---|---|---|
| Frozen AVES, last layer only | DFS: frozen 0.67 vs LoRA 0.96 AP on elephants; E8: intermediate layers often better; DCASE: fine-tuning is the second-largest gain. **But on our data so far, layer 12 beat every layer mix and the first LoRA run gave no gain**, so this row is the least supported by our own evidence | **(A)** LoRA on AVES (kept only if the tuning run finds a winner) |
| Each 0.64 s frame decided almost independently; context through concatenation (overparameterised, no gating) | Fin-whale LSTM: FP/h −59–85%; Bjorck conv-LSTM 92% vs 70%; every DCASE winner uses a BiGRU; rumbles come in bouts, and rain and thunder have a recognisable time course | **(C)** temporal head over 60–120 s |
| Remaining false alarms are engines/vehicles (0.64 FA/h on vehicle days), rain and thunder | Zeppelzauer: engine harmonics mimic rumbles in short-time analysis; noise clips were our biggest win | **(B)** low-frequency PCEN spectrogram branch: explicit harmonic and broadband structure; **(C)** long context separates a steady engine drone from a 3–8 s rumble |
| Labelled forest data is scarce (3.2k rumbles); clearing "negatives" contain unmarked rumbles | animal2vec 1% labels → 0.83 AP; DCASE pseudo-label iteration = largest gain; our clearing tables say "not every rumble marked" | **(D)** noisy-student self-training on unlabelled forest days; teacher relabels suspect clearing negatives |
| AVES never heard sped-up infrasound during pretraining | animal2vec, AVEX: in-domain SSL generalises best out of distribution; ELP's archive is huge at 32× | **(E)** optional continued SSL pretraining |

### 6.2 Architecture

```
24-h WAV → resample 500 Hz ──┬─► [32× speed-up] AVES (HuBERT) + LoRA (r≈8, layers 7–12)
                             │        └─ learned softmax-weighted sum of layers 4–12 → 768-d @ 0.64 s
                             │
                             └─► LF branch: STFT on 500 Hz audio, 5–250 Hz, PCEN (T≈10–30 s, tuned)
                                      → small CNN with frequency-dynamic conv → 128-d @ 0.64 s
                                                  │
                     concat + LayerNorm + linear → 256-d frame features
                                                  │
              Temporal head: 2-layer BiGRU (or 4-block Conformer), window 64–192 frames (41–123 s), 50% overlap
                                                  │
           frame posterior p(rumble) (+ optional onset/offset heads, SEBB-style boundaries)
                                                  │
         tuned alarm rule (validation) → events with score → Raven table
                                                  │
     optional: event-level verifier on 10 s around each candidate (pooled features + LF-CNN)
```

Design notes:

- **Frame rate.** Keep AVES's 0.64 s frames. ELP's 1% overlap rule is lenient, so finer boundaries add little to the score. The LF branch is pooled to the same rate.
- **Training unit.** Train on whole 2-minute stretches with frame labels, not isolated frames. This is how the temporal head learns realistic priors: rumbles are rare, and they come in bouts. Sample stretches so that about 30–50% contain a rumble, and keep the rest as random forest background plus noise-clip mixtures. This keeps our proven negatives recipe, but in sequence form.
- **Loss.** Frame BCE with positive weighting or focal loss (as in animal2vec). Ignore a ±1 frame margin at rumble edges. For clearing days, **mask** (do not penalise) frames the teacher scores high where no rumble is marked, rather than treating them as negatives (see D).
- **Augmentation.** Mix noise clips into rumble stretches at random SNR (−10 to +10 dB). Apply SpecAugment-style masks on the LF branch, random gain, and small speed perturbation (±5%, an elephant-plausible pitch change).
- **Inference cost.** LoRA merges into AVES (no extra cost). The LF CNN and BiGRU together are under 1M parameters on a 0.64 s frame rate, so they are negligible next to AVES on CPU. Check with `cpu_speed.py`.

### 6.3 What is novel here

- No published elephant detector combines an **adapted SSL waveform encoder with a temporal sequence model and self-training on unlabelled PAM data**.
- **The sped-up-SSL path versus the native low-frequency path** (FruitPunch/Pickering vs DFS) has never been compared. The two-stream design tests both and fuses them.
- **Continued SSL pretraining on sped-up infrasonic PAM audio** has not been tried for any species. animal2vec is the nearest example, on meerkats at normal speed. At 32×, ELP's archive amounts to tens of thousands of hours of model audio.
- The evaluation is itself a contribution: whole fully browsed forest days, FP/h at fixed recall, sealed test sets years apart, and seed spreads.

### 6.4 Honest risks

- **Context beyond ~11 s may add little.** Our window results flattened out after 10.88 s, and HuBERT already attends across its input. The fin-whale gain was on top of a 4 s CNN, which has much less built-in context than a transformer. *Expected gain from the temporal head: uncertain.* That is why it is tested first: it is cheap.
- **Adapting AVES may not help on our data.** In the tuning run so far, frozen L12 beats the layer mixes, and LoRA r4 didn't move off its starting point. Part (A) stays only if a remaining option wins.
- **Fine-tuning can overfit clearings.** 90% of positives come from 3 clearing sites. Use the balanced-sampling and held-out-forest-day stopping already in `finetune.py`.
- **Self-training can amplify errors.** Boudiaf et al. found adaptation can hurt. Use high-confidence thresholds, keep the student's validation days disjoint from the pseudo-labelled days, and stop if validation FP/h worsens.
- **PCEN time constant.** Too short a T erases rumbles. Tune it on validation and include a log-mel control.
- **Compute.** Continued pretraining on an RTX 4060 (8 GB) is slow. Keep it optional and small (see E7 below).

---

## 7. Experiment plan

**The yardstick does not change.**

- Validation: Complex + General days.
- Primary metric: FA/h at 80% recall; also report 85% and 90% recall and maximum recall.
- 5 seeds; mean with range.
- Alarm rules re-tuned per model on validation.
- Tests (Sample A on the 42 Schedule-File days plus all 61, Sample B, vehicle days) scored **once** at the end.
- One change at a time.

**Decision rule.** Adopt a change only if its mean FA/h at 80% recall beats the current best by more than the seed range and does not lower maximum recall. Report everything.

| Step | Experiment | Builds on | Hypothesis | Cost on lab PC (estimate) |
|---|---|---|---|---|
| **E0** | **Baselines in our pipeline:** (i) current best (L12 + MLP); (ii) frozen Perch 2.0 on time-expanded audio + same head; (iii) DFS-style LoRA-AST at 1 kHz, 3–250 Hz mel (public code) | – | Fixes the "other models" bar on identical data and scoring | (ii) ~half a day; (iii) ~1–2 days |
| **E1** | **Temporal head on cached frozen L12 fingerprints:** BiGRU (2×128) vs TCN vs 4-block Conformer; windows of 64 / 128 / 192 frames; trained on stretches | current best | Cuts FP/h at 80% by ≥ 20% and raises max recall (bridges dips mid-rumble) | Hours (no re-extraction) |
| **E2** | **Layer mix:** largely answered already: the tuning run found frozen L12 best and the learned mix worse (0.150 vs 0.134). Retry only under the E1 head | E1 | Intermediate layers might matter more once a sequence model sits on top | Hours |
| **E3** | **Adapt AVES:** only if the AVES-tuning run finds a LoRA or top-layer option that beats frozen L12, retrain it *with the E1 temporal head end to end* on stretches | E1 + tuning run | Adapted features + sequence head beat either alone (DFS, DCASE) | 1–3 h per run × 5 seeds |
| **E4** | **LF PCEN branch:** add the 5–250 Hz PCEN-CNN stream. Ablate: log-mel vs PCEN; T ∈ {2, 10, 30} s; with and without frequency-dynamic conv | E3 | Lowers vehicle-day FA/h and rain/thunder false alarms | Hours (small CNN; spectrograms are cheap) |
| **E5** | **Noisy-student self-training:** teacher = E4 ensemble (5 seeds). Pseudo-label (i) unlabelled forest days (e.g. `nn (only train files)` 2020 days, *excluding any Sample A day*), (ii) unmarked high-score frames in clearing negatives (mask or relabel). Train a student with noise augmentation; 2 iterations | E4 | Largest single gain in DCASE; adds forest positives without labelling | ~1 day per iteration |
| **E6** | **Event-level verifier + SEBB-style boundaries:** a small classifier on 10 s around each candidate, trained on validation-free mined false alarms; compare with tuned alarm rules alone | E5 | Further FP/h cut at 85–90% recall | Hours |
| **E7** *(optional, high cost)* | **Continued SSL pretraining of AVES** on unlabelled ELP audio at 32× (HuBERT-style masked prediction, or data2vec with animal2vec's sparse-event masking), e.g. 2,000–10,000 real hours (≈60–300 h of model audio), forest sites first; then repeat E3–E5 | E5 | In-domain SSL generalises best to new years and sites (AVEX, animal2vec) | Days to weeks on a 4060; better on a rented A100 |
| **E8** | **Final:** ensemble of 3–5 best seeds vs single model; CPU speed check; sealed tests once | best | Ensembling trims false alarms; confirm CPU budget | Hours |

**Ablation table to report at the end:** current best → +temporal head → +layer mix → +LoRA → +LF branch → +self-training → +verifier → +SSL, each with FA/h at 80/85/90% recall, max recall, Sample A (42 and 61 days) and Sample B P/R/FA/h, vehicle FA/h, and CPU seconds per 24 h.

**Suggested order and stopping points.**

1. Run E1 and the queued `finetune.py` this week. Both are cheap and independent.
2. If E1 helps, E3 combines them.
3. E4 and E5 are the next biggest bets.
4. Stop adding pieces when a step's gain is within seed noise.
5. Consider E7 only if the Sample B (new-year) result still lags Sample A noticeably, because that gap is the symptom SSL is best at fixing.

**What "beating FruitPunch and others" means concretely.**

- On ELP's 42-day Sample A: precision above Stanford's 0.842 at recall ≥ 0.746, or recall above 0.746 at precision ≥ 0.842.
- Lower FA/h at 80% and 90% recall than the current best, E0(ii) and E0(iii) on validation and on Sample B.
- All on seed means, with the test days scored once.

---

## 8. Open questions and unverified items

- The Stanford and Conservation Metrics detectors' architectures are not published. ELP may be able to share them, which would make E0 stronger.
- Venter & Hanekom 2010, Clemins 2003, Stöger 2012, Silva 2017 and Leonid & Jayaparvathy 2022 are cited from Geldenhuys & Niesler's table, not re-read.
- The Wrege et al. 2024 PLOS ONE detector details could not be fetched (the page was behind a browser check).
- Perch 2.0 may have seen elephant audio in training (E8 notes this), which affects its "out-of-species" status but not its use as a baseline.
- No paper compares speed-up into a bird/SSL model with a native low-frequency front end. E0(iii) vs current best plus E4 answers this for our data.
- Expected gains in section 7 are my estimates from the cited literature, not measurements on our data.

---

## References (by section)

**Elephant:**
- Keen et al. 2017, [doi:10.1121/1.4979476](https://doi.org/10.1121/1.4979476); [method summary](https://bestpractice.ai/ai-use-cases/case-studies/education/cornell-university-researchers-increase-processing-efficiency-300x-by-developing-adaboost-algorithm-to-automate-detection-of-elephant-rumblings)
- Bjorck et al. 2019, [arXiv:1902.09069](https://arxiv.org/abs/1902.09069)
- Gomes-Selman et al. 2020, [ESA](https://esa.org/blog/2020/07/24/machine-learning-streamlines-detection-of-african-forest-elephant-vocalizations/)
- Conservation Metrics, [Cornell CALS 2018](https://cals.cornell.edu/news/2018/08/ai-speeds-effort-protect-endangered-elephants)
- Zeppelzauer et al. 2015, [PMC4430816](https://pmc.ncbi.nlm.nih.gov/articles/PMC4430816)
- Venter & Hanekom 2010, [UP](https://repository.up.ac.za/handle/2263/14186)
- Geldenhuys & Niesler, [arXiv:2410.12082](https://arxiv.org/abs/2410.12082) and [arXiv:2605.00225](https://arxiv.org/html/2605.00225)
- Dubus et al. 2026, [arXiv:2604.08087](https://arxiv.org/pdf/2604.08087)
- Pickering et al. 2025, [UCL](https://discovery.ucl.ac.uk/id/eprint/10206477)
- Dewmini et al. 2025, [Sensors](https://www.mdpi.com/1424-8220/25/2/352)
- Kahoro & Muriithi 2026, [ICAART](https://www.scitepress.org/Papers/2026/142512/142512.pdf)
- Costa et al., [arXiv:2312.02831](https://ar5iv.labs.arxiv.org/html/2312.02831)
- Bandara & Bandara 2023, [arXiv:2310.15012](https://arxiv.org/pdf/2310.15012v1.pdf)
- Wrege et al. 2017, [MEE](https://zendy.io/title/10.1111/2041-210x.12730)
- Wrege et al. 2024, [PLOS ONE](https://pmc.ncbi.nlm.nih.gov/articles/PMC11280225/)
- FruitPunch, [repo](https://github.com/ramayer/elephant-rumble-inference)

**Encoders:**
- AVES, [arXiv:2210.14493](https://arxiv.org/abs/2210.14493)
- HuBERT, [arXiv:2106.07447](https://arxiv.org/abs/2106.07447)
- Perch 2.0, [arXiv:2508.04665](https://arxiv.org/html/2508.04665v1)
- Perch 2.0 marine, [arXiv:2512.03219](https://arxiv.org/pdf/2512.03219)
- BirdNET, [doi:10.1016/j.ecoinf.2021.101236](https://doi.org/10.1016/j.ecoinf.2021.101236)
- BEATs, [arXiv:2212.09058](https://arxiv.org/abs/2212.09058)
- AVEX, [arXiv:2508.11845](https://arxiv.org/pdf/2508.11845)
- animal2vec, [arXiv:2406.01253](https://arxiv.org/html/2406.01253v2)
- Ghani et al. 2023, [arXiv:2307.06292](https://arxiv.org/abs/2307.06292)
- BirdSet, [arXiv:2403.10380](https://arxiv.org/abs/2403.10380)
- BEANS, [arXiv:2210.12300](https://arxiv.org/abs/2210.12300)

**Low-frequency detection:**
- Shiu et al. 2020, [NOAA PDF](https://repository.library.noaa.gov/view/noaa/27588/noaa_27588_DS1.pdf)
- Madhusudhana et al. 2021, [PMC8292017](https://pmc.ncbi.nlm.nih.gov/articles/PMC8292017)
- Allen et al. 2021, [Frontiers](https://www.frontiersin.org/articles/10.3389/fmars.2021.607321/text)
- Miller et al. 2023, [RSEC](https://doaj.org/article/e15ca795bc38445b96d206cdc75a0175)

**SED and methods:**
- Schmid et al. 2024, [arXiv:2408.00791](https://arxiv.org/pdf/2408.00791)
- ATST-SED, [arXiv:2309.08153](https://ar5iv.labs.arxiv.org/html/2309.08153)
- FDY, [arXiv:2203.15296](https://arxiv.org/pdf/2203.15296); MDFD, [arXiv:2406.13312](https://arxiv.org/pdf/2406.13312)
- SEBB, [arXiv:2406.04212](https://www.arxiv.org/abs/2406.04212)
- PCEN, [arXiv:1905.08352](https://pdf.arxiv.org/pdf/1905.08352)
- Agile modelling, [arXiv:2505.03071](https://arxiv.org/html/2505.03071v3)
- Boudiaf et al. 2023, [arXiv:2302.06658](https://arxiv.org/abs/2302.06658v1)
- Noisy Student, [arXiv:1911.04252](https://arxiv.org/abs/1911.04252)
- Mean Teacher, [arXiv:1703.01780](https://arxiv.org/abs/1703.01780)
- LoRA, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- SUPERB, [arXiv:2105.01051](https://arxiv.org/abs/2105.01051)
- Stowell 2022 review, [PeerJ](https://peerj.com/articles/13152)

**Our earlier reports:**
- `reports/Recall first detector evaluation.md` (metrics, thresholds, post-processing)
- `reports/fair-comparison-training-length.md` (early stopping, seeds)
