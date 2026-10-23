# Fair comparison of training-data variants: training length, early stopping, seeds

Context: an MLP head on frozen AVES embeddings, Adam, batch 64, 370k–1.1M training examples. We compare data variants that differ in size, negative mix and context width. Until now each variant had a fixed 5-minute wall-clock budget. The plan is to replace that with early stopping on a held-out split and to keep the best epoch.

All sources below were found and checked by web search (October 2026). Where a paper does *not* say something, this report says so.

## Short answer: recommended settings for our case

| Setting | Recommendation |
|---|---|
| Stopping rule | Use the **same rule for every variant**: same max length, same patience, same validation set. A fixed long budget with retrospective best-checkpoint selection is equivalent and is what the Google tuning playbook prefers. |
| What to monitor | Select the best checkpoint by a **threshold-free task metric on validation** (average precision / PR-AUC for rumbles). Also log validation loss. HEAR uses the task score, not the loss. Smoothing or a `min_delta` keeps noise from triggering a stop. |
| Patience | About **10–20 evaluations**. Validation curves have many local minima (Prechelt), and published probe setups use 15 epochs (DCASE 2018) or 20 checks (HEAR). Define patience in **examples seen, not epochs**, e.g. evaluate every ~370k examples. Otherwise the 1.1M variant gets 3x more gradient steps of patience than the 370k one. |
| Max length | Set the cap high enough that **no run ever reaches it**, e.g. 100 evaluations, and log the best epoch. If a run reaches the cap, the comparison is budget-limited again. |
| Validation split | **Hold out whole recordings or days, not a random 5% of frames.** Neighbouring frames with overlapping context windows are near-duplicates. A random frame split leaks, which makes validation too optimistic and makes stopping late. Use **one common validation set for all variants**, taken from recordings that are in no variant's training pool. Then every variant is stopped against the same yardstick. |
| Test | Report on a separate test set that was never used for stopping or for choosing hyperparameters (Cawley & Talbot). |
| LR schedule | **Constant Adam LR** (or the same ReduceLROnPlateau rule for all variants). Avoid cosine or linear decay with early stopping: those schedules assume a known total budget (Li et al.). If affordable, tune the LR over a small grid (e.g. 3 values) **with the same grid for every variant**. |
| Seeds | **At least 3, ideally 5 seeds per variant.** Where possible, vary the validation and training sampling as well as the initialisation. Report mean with a bootstrap CI or all runs, not the best seed (Bouthillier, Lucic, Agarwal). |

## Evidence

### 1. Fair comparison across datasets of different size

- **Lucic et al. 2018, "Are GANs Created Equal? A Large-Scale Study"** (NeurIPS), https://arxiv.org/abs/1711.10337. The ranking of methods depends on the compute budget, and "a 'bad' model trained on a large budget can out perform a 'good' model trained on a small budget". Their recommendation is to compare *distributions* of results under a fixed budget, never the best run. This is the failure mode of our old 5-minute rule: the large variants got fewer epochs (14 against 64).
- **Melis, Dyer & Blunsom 2018, "On the State of the Art of Evaluation in Neural Language Models"** (ICLR), https://arxiv.org/abs/1707.05589. With large-scale automatic hyperparameter tuning for every model, a plain LSTM beat newer architectures. Earlier gains came from "uncontrolled sources of experimental variation", including limited compute. Lesson: give every configuration an equal tuning effort.
- **Dodge et al. 2019, "Show Your Work: Improved Reporting of Experimental Results"** (EMNLP), https://arxiv.org/abs/1909.03004. Reports *expected validation performance as a function of compute budget* (number of tuning trials or training time). Some published conclusions reverse when the budget changes. Lesson: report the budget each variant used (epochs to best, wall time).
- **Oliver et al. 2018, "Realistic Evaluation of Deep Semi-Supervised Learning Algorithms"** (NeurIPS), https://arxiv.org/abs/1804.09170. Uses the same model, the same training procedure (500k updates, no early stopping, report at lowest validation error) and the same 1000-trial tuning budget for baselines and methods. Also warns against over-tuning on "an unrealistically large validation set".
- **Data-subset papers.** DeepCore (Guo et al. 2022, https://arxiv.org/abs/2204.08499) trains every coreset size for a **fixed 200 epochs**, so smaller subsets get fewer steps, and does not discuss whether this is fair. **Okanovic et al. 2023/ICLR 2024, "Repeated Random Sampling for Minimizing the Time-to-Accuracy of Learning"** (https://arxiv.org/abs/2305.18424) keeps the same number of rounds but decays the LR "inversely proportional to the subset size". For GPT-2 they scale iterations with subset size. So the subset literature has no consensus, and the budget convention changes conclusions. This is why the max length should be generous and the stopping point should be data-driven.
- **Google "Deep Learning Tuning Playbook"** (Godbole, Dahl, Gilmer, Shallue, Nado 2023), https://github.com/google-research/tuning_playbook. "Run training for a fixed number of steps and retrospectively choose the best checkpoint". Prospective early stopping "is usually not necessary". "Never tune the max_train_steps number in a study". It warns that epoch budgets and step budgets probe different things.

### 2. Early stopping practice

- **Prechelt, "Early Stopping — But When?"** (Neural Networks: Tricks of the Trade, 1998/2012), https://link.springer.com/chapter/10.1007/978-3-642-35289-8_5 (PDF: https://publikationen.bibliothek.kit.edu/7498/1832). Defines the GL criterion (relative increase over the best-so-far validation error), the PQ criterion (generalisation loss ÷ training progress) and the UP criterion (validation error rises over *s* successive strips). "Real validation error curves almost always have more than one local minimum." Slower criteria give about 4% lower error for about 4x the training time (1296 MLP runs). This supports a patient stopping rule and restoring the best checkpoint.
- **Bengio 2012, "Practical Recommendations for Gradient-Based Training of Deep Architectures"**, https://arxiv.org/abs/1206.5533. Patience is measured in **training examples** seen after the current best, and it grows when a new best is found. Save the best model. Evaluate every N examples with N ≥ the validation set size. The LR is the most important hyperparameter to tune.
- **Validation split structure.** **Roberts et al. 2017, "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure"** (Ecography), https://doi.org/10.1111/ecog.02881. "random data splitting does not provide independent validation when a dependence structure is present", so error estimates are "too optimistic, and model selection will favour too complex models". Block by time instead. **Kapoor & Narayanan 2023, "Leakage and the reproducibility crisis in machine-learning-based science"** (Patterns), https://arxiv.org/abs/2207.07048, lists non-independence between train and evaluation data as a leakage type. **Stowell 2022, "Computational bioacoustics with deep learning: a review and roadmap"** (PeerJ), https://peerj.com/articles/13152, recommends separate train, validation and test sets. It notes that test sets ideally represent "novel conditions", and that it is "still common" to draw all splits from the same pool.
- **Choosing on validation, then reporting on it.** **Cawley & Talbot 2010, "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation"** (JMLR), https://www.jmlr.org/papers/v11/cawley10a.html. Optimising a selection criterion overfits it, with bias as large as the differences between algorithms. High-variance criteria, such as a small or noisy validation set, make this worse. So report on a separate test set.

### 3. Learning-rate schedules

- **Li, Yumer & Ramanan 2020, "Budgeted Training: Rethinking Deep Neural Network Training Under Resource Constraints"** (ICLR), https://arxiv.org/abs/1905.04753. "It is critical to adjust the learning rate schedule according to the given budget." Linear decay to zero is a robust default *when the budget is known*. A schedule tuned for one length "fails to generalize across budgets".
- **Loshchilov & Hutter 2017, "SGDR"** (ICLR), https://arxiv.org/abs/1608.03983. Cosine annealing is likewise defined over a planned number of epochs.
- The tuning playbook prefers linear or cosine decay and says some non-constant schedule matters.
- Implication: decay schedules and early stopping pull against each other. When the stopping point varies across variants, use a constant LR or a plateau rule that is identical for all variants. If we later switch to a fixed per-variant budget, use a schedule that is a function of training *progress*. Every comparison paper above holds the training procedure fixed across the methods being compared.

### 4. Bioacoustics and audio probes on frozen embeddings

- **HEAR: Turian et al. 2022, "HEAR: Holistic Evaluation of Audio Representations"**, https://arxiv.org/abs/2203.03022. This is the closest analogue: an MLP on frozen embeddings, trained with Adam for up to 500 epochs. Validation is checked every 3 epochs, and training stops after 20 checks without improvement. Stopping and selection use the **task score** on validation, not the loss. A grid of 8 of 16 points covers 4 LRs and 1–2 hidden layers. One fixed seed.
- **BEANS: Hagiwara et al. 2023, "BEANS: The Benchmark of Animal Sounds"** (ICASSP), https://arxiv.org/abs/2210.12300, and **AVES: Hagiwara 2023, "AVES: Animal Vocalization Encoder based on Self-Supervision"** (ICASSP), https://arxiv.org/abs/2210.14493. Adam, 50 epochs per LR in {1e-5, 5e-5, 1e-4}. They "picked the best model based on the validation metric". No early stopping or seeds are reported. Most BEANS datasets use random 6:2:2 splits. Note that these are *fine-tuning* setups (AVES freezes only the CNN feature encoder), not linear probes.
- **Ghani et al. 2023, "Global birdsong embeddings enable superior transfer learning for bioacoustic classification"** (Sci. Rep.), https://arxiv.org/abs/2307.06292. Linear probe on frozen embeddings, trained "to convergence". 5 seeds per configuration, results averaged. No validation set or optimiser details are reported.
- **DCASE 2018 Task 4 baseline: Serizel et al. 2018**, https://arxiv.org/abs/1807.10501. 100 epochs, early stopping with 15-epoch patience, and 20% of the weak-label training set held out for validation. The monitored metric is not stated.

### 5. Seeds and reporting variance

- **Bouthillier et al. 2021, "Accounting for Variance in Machine Learning Benchmarks"** (MLSys), https://arxiv.org/abs/2103.03098. Randomise "as many sources of variation as possible" (initialisation, data order, data splits, hyperparameter optimisation). Varying only the seed underestimates variance. Decide "A beats B" with a probability-of-improvement criterion, not mean differences. Prefer resampling to a single fixed test split.
- **Agarwal et al. 2021, "Deep Reinforcement Learning at the Edge of the Statistical Precipice"** (NeurIPS), https://arxiv.org/abs/2108.13264. With 3–10 runs, report stratified or percentile bootstrap CIs. Percentile CIs work for as few as 10 runs. Publish all runs.
- Lucic et al.: report the distribution over seeds, never the best seed.

## Uncertainties and caveats

- No paper found addresses *our exact* question: comparing data variants of different size with an MLP probe. The recommendations combine general benchmarking practice with probe-specific defaults (HEAR, DCASE).
- Early stopping versus fixed budget: the tuning playbook prefers a fixed budget with retrospective checkpoint selection. With a generous cap, the two give the same chosen checkpoint, so either is fine as long as it is applied identically.
- Defining patience in examples rather than epochs is our own reading of Bengio 2012. No paper here tests it for different dataset sizes.
- The HEAR "20 checks" patience comes from the appendix text as fetched. Check the exact value in Table 4 before quoting it.
- Whether a validation set common to all variants is better than a per-variant 5% split is a judgement, not a literature result. It does remove the confound that each variant's validation set has a different negative mix.
- Some per-paper details (e.g. BEANS split ratios, Kapoor & Narayanan taxonomy wording) were checked from abstracts or summaries, not full reads.
