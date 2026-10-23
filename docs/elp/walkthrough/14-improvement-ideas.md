# Ways to improve the FruitPunch rumble model: a list to discuss

Where we are (from `elp_score.csv`, whole-day test, 61 files): at the default setting the model finds **93%** of rumbles but only **28%** of its detections are real. At threshold 0.90 it's about **P 0.71 / R 0.70**. A full day takes about **31 s** on the RTX 4060.

Each idea says why it fits *this* code, what it might gain, and what it costs. Gains are my estimates, not measured. ⭐ marks the ones I'd try first. "Split" means it depends on the outcome of the *Choose training and testing data splits* thread.

---

## A. Accuracy (mainly precision)

### A1. Fix the measurement first (no retraining, minutes)
| # | Idea | Why here | Payoff | Cost |
|---|---|---|---|---|
| 1 ⭐ | Fix the `nn04a_20200428/29` file name, drop `nn04a_20180608`, score only reviewed days | 57 rumbles aren't scored and 32 real hits count as false alarms; 381 "false alarms" sit on files nobody may have reviewed | Precision looks higher with no model change, and we know the true starting point | Re-score only, seconds |
| 2 ⭐ | Pick the threshold on validation days, not the test | 0.90 was chosen by looking at the test | An honest number; needed to judge everything below | Small. **Split** |

### A2. Better decisions on top of the same scores (no retraining, cheap)
Today `cmd_detect` marks every run of ≥ 4 frames (2.5 s) where the rumble probability is above 0.5, scores it by the mean, and gives every box 5–250 Hz.

| # | Idea | Payoff | Cost |
|---|---|---|---|
| 3 ⭐ | **Smooth the scores over 2–5 s** before thresholding, so short, borderline blips stop becoming detections and broken-up rumbles join back together | Precision ↑, little recall loss | ~10 lines |
| 4 | **Tune minimum length, gap merging and threshold together** (e.g. require 3 s, merge gaps under 1 s) | Precision ↑; fewer split detections | Small, needs cached scores (B4). **Split** for the validation days |
| 5 | Score each detection by its **peak or top-3 frames** instead of the mean | A cleaner threshold sweep | Tiny |
| 6 | **Estimate the frequency box** from the spectrogram instead of fixed 5–250 Hz | Small for ELP's 1% rule; nicer boxes in Raven for analysts | Small |

### A3. Better "not rumble" training examples (retrain the small model; biggest expected gain)
Today negatives only come from gaps **next to** rumbles, 80% of them on **clearing** days, and 90% of them are thrown away to make a 50/50 balance.

| # | Idea | Payoff | Cost |
|---|---|---|---|
| 7 ⭐ | **Hard-negative mining:** run the model on training days, add every false alarm as a "not rumble" example, retrain, repeat 2–3 times | The most direct fix for false alarms: precision ↑↑↑ | Medium: a few re-runs of detect + train. **Split** (needs fully reviewed training days, so the false alarms are really false) |
| 8 ⭐ | Add the **unused non-elephant clips** (thunder 77, rain 40, wind 65, vehicle 20, silence 63) and **elephant-free days** | Teaches the sounds behind many false alarms | Small |
| 9 | Take random negatives from **anywhere in forest days** (≥ 30 s from a rumble), not just next to rumbles | Covers night insects, rain, quiet forest | Small. **Split** |
| 10 | **Keep more negatives** (1:5 or 1:10) and weight the loss, instead of discarding 90% | The model learns rumbles are rare | Tiny code change |
| 11 | **Beware unmarked rumbles in negatives:** the clearing and training tables say "not every rumble marked", so some "not rumble" examples are actually rumbles. Take negatives only from fully reviewed stretches, or soften those labels | Cleaner training signal; may help recall too | Small. **Split** |

### A4. A better model (retrain)
| # | Idea | Why here | Payoff | Cost |
|---|---|---|---|---|
| 12 | **Train longer than 2 minutes** | Val loss was still falling at epoch 16; early stopping never fired | Small ↑; free | Change `--minutes` |
| 13 ⭐ | **Temporal head:** replace the per-frame 768 → 192 → 2 classifier with a small 1-D conv or GRU over ~10 s of fingerprints | A rumble lasts 2–8 s with a rise and fall; thunder looks different over time | Precision ↑, recall ↑ | Medium; trains in minutes on cached fingerprints |
| 14 | **Second-stage checker:** a tiny CNN on the spectrogram of each candidate detection (~2,000 per 60 days) that says keep or drop | Classic way to cut false alarms; adds almost no run time | Precision ↑↑ | Medium |
| 15 | **Use a mix of AVES layers** instead of only the last one | The last layer is tuned to AVES's own training task, not infrasound | Unknown, often ↑ | Re-extract fingerprints |
| 16 | **Ensemble 3–5 models** (different seeds or data) and average scores | Usually trims false alarms | Small ↑ | Cheap to train; inference cost of the small head only |
| 17 | **Fine-tune the top AVES layers** on rumbles with a small learning rate | Usually the biggest single jump, but AVES has never heard 500 Hz audio | ↑↑ possible | Large: GPU hours, re-run per experiment, risk of overfitting |
| 18 | Try **another sample rate** (250 or 1,000 Hz instead of 500) or another foundation model (Perch, BEATs) | 500 Hz was one choice, never compared | Unknown | Medium to large |

### A5. More and better-matched data (**Split**, being decided in the other thread)
19. More **forest** rumbles, forest weighted higher than clearing, and labels cleaned of `DUMMY_NoEles`, marginal and faint rows. Recall ↑ and precision ↑ on forest; cost depends on the split.

---

## B. Speed and power of inference (currently ~31 s per 24-hour file)

| # | Idea | Why here | Payoff | Cost |
|---|---|---|---|---|
| 1 ⭐ | **Time each step** on one file: reading from `L:`, resampling, AVES, classifier | We don't yet know what the 31 s is made of. Reading 1.4 GB of 8 kHz audio and resampling it with `resample_poly` on the CPU may be a large share | Tells us which fix matters | Tiny |
| 2 ⭐ | **Overlap reading with GPU work:** a background thread loads and resamples the next file while AVES runs on this one | Today the CPU and GPU take turns | Up to ~2× for a folder, if reading is half the time | Small |
| 3 ⭐ | **Half precision (fp16) for AVES** | The RTX 4060 is about 2× faster at fp16; `fp_replica` already rounds the audio to fp16 | ~1.5–2× on the AVES part; less power | Tiny; check detections match |
| 4 | **Batch the 24 hours** (or 5-minute pieces with overlap) into one GPU call, instead of one hour at a time | The GPU sees a batch of 1 today | ~1.3–2× on AVES | Small |
| 5 | **Resample on the GPU** (`torchaudio.functional.resample`), or read the file once with a fast decoder | Takes the CPU resampling out of the critical path | Depends on step 1 | Small |
| 6 | **Cache 500 Hz audio and whole-day fingerprints** for validation and test days | Lets A2 and A4 ideas be tested in seconds without AVES (~200 MB per day for fingerprints at fp16) | Experiments go from hours to seconds | Disk space |
| 7 | `torch.compile` + PyTorch's fast attention | Free speed on recent PyTorch | ~10–30% | Tiny, sometimes fiddly on Windows |
| 8 | **Use fewer AVES layers** (stop at layer 8 of 12) | Skips a third of the work | ~1.3–1.5× | Needs retraining the head (cheap) |
| 9 | **Skip quiet hours** with a cheap 10–40 Hz energy check before AVES | Many hours have nothing low-frequency at all | Large at quiet sites | Risk of missing faint rumbles; needs testing |
| 10 | **Distil into a small spectrogram CNN** that copies AVES + head | Runs in seconds on a laptop CPU, very low power; could even run on a field device | 10–100× faster | Large: new model, must re-check accuracy |
| 11 | **For CPU-only teams:** int8 quantization or ONNX / OpenVINO export of AVES | Today ~28 min per day on CPU | ~1.5–3× on CPU | Medium; check accuracy |
| 12 | **Remove the 24-hour cap** in the original `main.py` (`--limit-audio-hours`), so longer files aren't silently cut | Not speed, but "more powerful": works on any length | Correctness | Tiny |

---

## What I'd suggest discussing first
- **Accuracy:** A1 (fix the scoring), then A2 smoothing (#3), then hard negatives + non-elephant clips (#7, #8), then the temporal head (#13).
- **Speed:** time the steps (B1), then overlap reading + fp16 (B2, B3). That likely gets well under 31 s with almost no risk. B6 (cached fingerprints) speeds up all the accuracy experiments.
