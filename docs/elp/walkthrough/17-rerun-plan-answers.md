# Rerun plan: answers to your questions (23 Oct)

Nothing is coded yet. The answer to question 1 (training length and fair comparison in the literature) will follow when the research is done, in `reports/fair-comparison-training-length.md`. A first answer is below.

---

## 1. "Train until it stops improving": how does that prevent overfitting?

This is called **early stopping**. After every epoch, the model is checked on examples it is not trained on (the held-out set). While it is still learning, the held-out loss goes down. When it starts to memorise its training examples, the held-out loss stops falling and then rises. We stop once it has not improved for a set number of epochs (the "patience", for example 10), and **keep the weights from the best epoch, not the last one**. So the model can run as many epochs as it needs, up to a ceiling such as 300, but what we keep is the point just before it started to over-fit.

That is exactly what went wrong for B and D in the first round. B's held-out loss was best at epoch 26 and much worse by epoch 148, but we kept epoch 148.

**One thing I'd fix in the rerun:** right now the held-out set is a random 5% of the training *frames*. A frame's neighbours (0.64 s away, nearly the same sound) are often in the training set, so the held-out loss looks better than it really is, and early stopping can stop too late. The fix is to **hold out whole training days** (for example 5–10% of the days) for early stopping. Those days are still never the practice or test days.

**What the literature recommends for our case** (details and references in `reports/fair-comparison-training-length.md`):

- **The same stopping rule for every version:** the same maximum length, the same patience and the same validation days, always keeping the best checkpoint. A fixed time or epoch budget is exactly what distorts rankings between configs (Lucic et al. 2018; Google's Deep Learning Tuning Playbook).
- **Choose the best checkpoint by a task score, not only the loss.** For example, average precision on the held-out days, as the HEAR benchmark does for probes on frozen audio embeddings. Log the loss too.
- **Patience of about 10–20 checks, counted in examples seen rather than epochs** (Prechelt 1998; Bengio 2012). Otherwise the 1.1-million-example version gets three times as much patience as the 370,000 one.
- **A generous ceiling** (for example the equivalent of 100 epochs), and confirm that no run actually hits it.
- **Hold out whole days, not random frames**, and use one shared set of held-out days for every version, so each is stopped against the same yardstick.
- **The same learning-rate rule for every version:** constant Adam, or the same reduce-on-plateau rule. Schedules that assume a known training length don't mix well with early stopping.
- **5 runs per version**, reporting the spread. Ideally the runs also differ in which examples are sampled, not just in starting weights (Bouthillier et al. 2021).

---

## 2. Is redoing the first round needed?

From the first round's training logs (2 minutes per setup):

| Setup | Epochs run | Best epoch | Held-out loss: best → last |
|---|---|---|---|
| A clearing + forest | 15 | 15 (last) | 6.39 → 6.39 |
| B forest only | 148 | 26 | 7.99 → 13.82 |
| C clearing only | 17 | 17 (last) | 5.78 → 5.78 |
| D forest + 10 clearing days | 75 | 17 | 6.22 → 8.55 |
| E clearing + noise | 17 | 17 (last) | 5.80 → 5.80 |
| F E + Complex + General | 17 | 16 | 6.39 → 6.56 |
| G E + Sample A | 16 | 14 | 6.47 → 6.56 |
| H E + Complex + General + Sample A | 16 | 14 | 6.62 → 6.80 |
| E2 | not in the files I have | – | – |

- A, C and E were still improving when time ran out. Their best epoch was the last one.
- B and D over-trained badly, and we kept their worst version.
- F, G and H were roughly at their best. They had peaked only 1–2 epochs earlier.

**My view: yes, redo it.**
- Every later step was built on the choice of E2, and that choice came from a **single run per setup, under-trained, using FruitPunch's alarm rule**.
- E against E2 was a tie inside run-to-run luck. We now know luck is worth about ±0.02–0.05 false alarms an hour, so one run per setup can't separate them.
- The new alarm rule changed scores a lot (−40% false alarms), and it may not help every setup equally.

It is cheap: 9 setups × 5 runs, trained to convergence, then scored with the 80%-recall yardstick and the tuned rule. That's about 6–10 hours. Then the improvement run starts from whichever setup wins.

---

## 3. Without the 75% cap, how are the three kinds of "not a rumble" divided?

Instead of fixing the total and letting a cap and "fill the rest" decide the mix, **each source gets an explicit amount**, and the total is just their sum:

- **Noise:** all 254 clips once (11,340 frames). There is no more noise to add.
- **Random forest frames:** N per forest day (130 days × N).
- **Gap frames:** G per rumble, from the windows next to rumbles, never repeated.

Each config is then a clear recipe, and its negatives-per-rumble ratio is reported rather than forced. Examples:

| Recipe | Noise | Random | Gap | Total per rumble (approx.) |
|---|---|---|---|---|
| FruitPunch style | 11,340 | 0 | ~181,000 | 1.0 |
| Last winner | 11,340 | 130,000 (1,000/day) | ~51,000 | 1.0 |
| **No gap frames** (your idea) | 11,340 | ~181,000 (~1,400/day) | 0 | 1.0 |
| More random | 11,340 | 390,000 (3,000/day) | ~51,000 | 2.3 |
| More random | 11,340 | 650,000 (5,000/day) | ~51,000 | 3.7 |
| More gap | 11,340 | 130,000 | ~435,000 | 3.0 |
| More of both | 11,340 | 390,000 | ~435,000 | 4.3 |

I agree the **no-gap** config is a good test. It answers directly whether frames next to rumbles help at all, once the detector has plenty of forest background. One risk to watch: without gap frames the detector never sees the "almost a rumble" moments, so it could raise more false alarms just before and after real calls.

---

## 4. Tie-breaker: what happens now, and what I propose

**Now, for training options (steps 2–5):** each option is ranked by its average false alarms per hour at 80% recall over the 5 runs. Anything within one run-to-run standard deviation of the best counts as a tie, and the tie goes to the simplest option, in this order: fewer extra rounds, then shorter window, then fewer negatives, then fewer noise repeats, then fewer random frames.

**Now, for alarm rules (steps 1 and 6):** there is no tie rule. The rule with the lowest average false alarms wins, even by 0.001. That's how the 8-frame minimum (0.1905) beat the 2-frame minimum (0.1917), even though the 2-frame version can reach 97% of rumbles instead of 91%.

**Proposal:**
- **Rules:** treat rules within luck of the best as tied, then pick the one that **reaches the most rumbles** (highest maximum recall). Then the most false alarms at 90% recall, then the simplest.
- **Training options:** same luck band, but break ties first by **false alarms at 90% recall** (see 7), and only then by simplicity. Simplicity matters less now that training time isn't a concern and inference time hardly changes.

---

## 5. Save all 1,960 rule scores

Agreed. Every rule's scores will be written to a CSV for both rule steps, along with false alarms at 85% and 90% recall.

---

## 6. Gap-heavy configs alongside random-heavy ones

Agreed. With training to convergence, the gap-heavy configs get a fair chance this time. The recipe table in 3 includes "more gap", "more random" and "more of both", plus "no gap", so all four directions are tested on equal footing.

---

## 7. Using both 80% and 90% recall when choosing

At the moment every choice is made by looking only at the point where a model finds 80% of the rumbles. But you said you'd rather run the detector nearer 90%, and some options rank differently there:

| Option | False alarms/h at 80% | at 90% |
|---|---|---|
| Noise ×1 | **0.297** | 0.759 |
| Noise ×10 | 0.328 | **0.662** |
| Window 0.64 s | 0.214 | 0.534 |
| Window 8.32 s | **0.198** | **0.419** |

So noise ×1 wins at 80% but loses at 90%. With last run's method, an option that is clearly better where you'd actually use the detector can be thrown away.

**Proposal:** score each model by its **average false alarms across 80%, 85% and 90% recall**. That's one number covering the whole range you care about, similar to the "partial area under the curve" used in detection papers. Report the three points separately too.

**Caveat (checked 23 Oct):** not every run reached 90% last time. 37 of 39 versions had all 5 runs reach 90%. The final model (max recall 0.897–0.923) and hard-negatives round 1 (0.897–0.933) each had at least one run that never got there, and their 90% figures quietly averaged only the runs that did, which flatters them. Main cause: the 8-frame minimum in the chosen rule caps recall around 91%; the same model with a 2-frame minimum reaches 97%. **Update after discussion:** if the detector will be run at about 80–85% recall, the simpler choice is to score the **average of 80% and 85%** and report 90% for information only. Every run of every version reached 85% last time, so neither guard below is needed. The guards only matter if 90% is part of the score:
- **Rules must leave headroom:** only rules whose every run reaches at least 93% recall are eligible.
- **A run that misses a recall point fails it:** the version ranks below every version whose 5 runs all reach 90%. It is never averaged over fewer runs.

---

## 8. Cleaner "learning from its own mistakes" (hard negatives)

**What it does:** the trained model scans the training days. Every frame it scores as a likely rumble where *no rumble is labelled* is treated as a mistake and added to training as "not a rumble". Then it retrains, and this can repeat.

**What went wrong last time:**
- Round 1 added 51,627 such frames, and the model got clearly worse and very unstable (0.29–0.53 false alarms an hour across runs).
- Round 2 then mined **1.1 million** frames, because the worse model flagged far more.
- The likely cause: label tables are not perfect. Some "mistakes" are probably **faint real rumbles nobody labelled**, especially just before and after labelled ones. Teaching the model that those are "not a rumble" makes it ignore real quiet calls.

**Fixes:**
- Never mine anything within about 60 s of a labelled rumble.
- Mine **alarms, not frames:** one or a few frames per false alarm, instead of every frame in it, so one long false alarm doesn't add hundreds of near-identical examples.
- Cap the number per day (for example the 200 highest-scoring), so a few noisy days don't dominate.
- Keep the overall cap on the mined share (one third of negatives).

---

## 9. Mixing noise into audio (augmentation)

**The noise** here means the 254 clips you already have: thunder, rain, wind, vehicles, voices, wing beats, silence. I'd drop the "silence" clips for mixing. The vehicle days stay out, because they are a test set.

**How the mixing works:** it happens on the **sound itself, before AVES**, by adding the two waveforms together at a chosen loudness ratio (the signal-to-noise ratio, measured in the 10–100 Hz band where rumbles are).

1. **Rumble + noise → still labelled rumble.** For example, a rumble with rain on top at the same loudness, or with the rain 10 dB louder. The elephant is still calling, so the label stays "rumble". This teaches the detector that a rumble under rain or an engine is still a rumble, which should help **recall in bad weather or noisy sites**.
2. **Forest background + noise → still labelled not a rumble.** Random forest frames with an engine or thunder added. This teaches that those combinations, which are common in real recordings, are not elephants. That should cut **false alarms**.

**Things to decide together:**
- Which loudness ratios to use (for example a few levels between "rumble clearly louder" and "noise louder").
- Whether to use type 1, type 2 or both. I'd test both, separately.
- How many augmented copies per example (one is a good start).

**Cost:** the mixed audio has to go through AVES, but only the frames we train on, not whole days. One augmented copy of all rumble frames is about 31 hours of audio, roughly 40 minutes of AVES on the 4060 plus reading the audio from the L: drive. So this is a few hours per augmentation config, which is affordable before Tuesday. It also needs one new step in the code: reading audio snippets, mixing them and fingerprinting them, written as a new file.

Mixing noise into training audio is a standard technique in bird and animal sound detection. The research note will add references.

---

## Suggested order for the run (until Tuesday)

1. **First round again, done properly:** the 9 setups, each trained to convergence (early stopping on held-out *days*) with 5 runs, scored at 80–90% recall with the tuned rule. About 6–10 hours.
2. **Improvement run** on the winning setup, with every fix above: explicit negative recipes (3 and 6), the recall-first tie-breaks (4 and 7), all rules saved (5), cleaner hard negatives (8). About 30–40 hours.
3. **Augmentation configs (9)**, if we agree on the details, either inside step 2 as an extra stage or as a separate run afterwards.

This all fits before Tuesday. Everything goes in new files, and the existing code stays untouched.
