# Can inference be faster? Does it need 24-hour files?

Short answers:
- **Speed:** yes. The code runs everything **one step at a time**, so a strong GPU sits idle much of the time. There are 6 concrete improvements, below.
- **24 hours:** it **doesn't require** 24-hour files. Any length already works. But there's a hidden **cap** at 24 hours that silently cuts off longer files, and that's worth fixing.

(These are audio recordings, not videos.)

---

## Part 1: How the current code runs (why it's slower than it could be)

For each file, `audio_file_processor.py` does this, strictly in order:

```
for each 1-hour piece of the file:
    1. CPU: read + decode + resample 1 hour of audio to 500 Hz     ← GPU waits
    2. GPU: run AVES on that 1 hour (batch of ONE)                   ← CPU waits
    3. copy results back to CPU, clear GPU memory
    4. CPU: run the small classifier
then move on to the next file (main.py:202, a plain for-loop)
```

So the CPU and GPU take turns instead of working at the same time, the GPU only ever sees one piece at once, and files are done one by one.

The README says a 24 h file takes **22 s on an RTX 2060**, a modest 2019 gaming card. On a strong GPU the AVES step gets faster, but **decoding audio on the CPU (step 1) doesn't**, so it likely becomes the bottleneck. That's an educated guess; step 0 below measures it.

---

## Part 2: How to make it faster

Ordered from biggest likely gain to smallest.

### 0. Measure first (always)
Time each of the 4 steps above on one real file on your GPU. That tells us which fix matters most, and gives a "before" number to compare against.

### 1. Process many files in parallel (biggest win if you have many files)
Right now: file 1 → file 2 → file 3… Instead:
- **Several CPU workers decode/resample files in the background** (PyTorch `DataLoader` with `num_workers=8` or so), while
- **the GPU continuously processes** whatever is ready.

That keeps the GPU busy instead of waiting. On a machine with many CPU cores this can be a several-times speed-up for a folder of recordings.

### 2. Feed the GPU bigger batches
Currently AVES gets a batch of **1** (`audio_file_processor.py:36`, `.view(1, …)`). A strong GPU is built to do many at once. Options:
- cut audio into e.g. **5-minute pieces with a little overlap** and send 16–64 of them per batch;
- this also uses **less GPU memory per piece**, because a transformer's memory cost grows fast with length (one hour = 5,625 "slices" in one go is heavy).

### 3. Use half precision (fp16 / bf16)
Modern GPUs run about 2× faster on 16-bit numbers than 32-bit. In PyTorch this is basically one line (`torch.autocast`). We'd check that the results barely change by comparing the scores before and after.

### 4. Stop the small inefficiencies
- `torch.cuda.empty_cache()` is called after **every** hour (`audio_file_processor.py:40`). That was for the author's small GPU; on a big GPU it just slows things down.
- The classifier is put on the **CPU** (`main.py:121`), so results bounce GPU → CPU every hour. Keep everything on the GPU.

### 5. Use more than one GPU (if the system has several)
Simplest form: run one copy of the tool per GPU, each handling a different subset of files. No code changes needed beyond splitting the file list.

### 6. Compile the model (smaller extra gain)
`torch.compile(model)` can squeeze out another 10–30% on recent PyTorch versions.

**Realistic expectation:** for a big folder of recordings, 1 + 2 + 3 together could plausibly be **several times faster** than now, but I won't promise a number until step 0 is measured.

---

## Part 3: Input length: any size already works, with one trap

The code reads the file in **1-hour pieces** (`frames_per_chunk = 500 × 60 × 60`, `audio_file_processor.py:52`) until the file ends. So:

| Recording length | What happens today |
|---|---|
| 10 seconds | Works (one short piece) |
| 37 minutes | Works (one short piece) |
| 5 hours 20 min | Works (5 full pieces + 1 short one) |
| 24 hours | Works |
| **30 hours** | **Only the first 24 hours are processed. The rest is silently skipped.** |
| under ~1.3 seconds | Returns nothing, with a warning (too short for AVES) |

**The trap:** `--limit-audio-hours` defaults to **24** (`main.py:104`), and the loop stops after that many pieces (`audio_file_processor.py:94`). That's a *maximum*, not a requirement, but for longer files it drops data without a clear warning.

**Fix (small):** make the default "no limit" and only cap when the user asks for it (useful for quick tests on CPU).

Two smaller rough edges at the very end of a file:
- If the last piece isn't a multiple of 320 samples, the final slice is based on incomplete audio (the code prints a warning).
- On the last piece, the code copies the second-to-last result again to make up for a slice AVES doesn't return (`audio_file_processor.py:91`). It works, but it's a patch. Worth tidying when we touch this code.

Also: the reader uses ffmpeg under the hood, so it should already accept **.flac, .mp3** and other formats, not only `.wav`. Not tested yet.

---

## Suggested plan

1. **Benchmark** the current code on one real file on your GPU (step 0).
2. **Fix the 24-hour cap** (tiny change, clear win).
3. Add **parallel decoding + batching + fp16** (steps 1–3) in a new, faster processing path. Keep the original as a reference so we can **check both give the same detections**.
4. Re-benchmark and compare.

---

## Part 4: Making it faster on CPU (for teams without a GPU)

Today: **about 28 minutes per 24-hour file on CPU** (README). Almost all of that is AVES, the big model. The small classifier is negligible.
Every option below works on an ordinary laptop. They're grouped by whether the trained model must change.

### A. No retraining needed (same model, same results, or almost)

| Idea | What it means | Likely gain | Risk |
|---|---|---|---|
| **1. Shorter pieces** | Feed AVES 2–5 minute pieces (with a little overlap) instead of 1 hour. A transformer's work grows *faster than* the length of what it reads, so many short pieces cost less in total than one huge one. Also uses far less memory, which matters on 8 GB laptops. | ~1.3–2× | Very low. Check the results match. |
| **2. Use all CPU cores well** | Decode the next piece while AVES works on this one, and on many-file runs process 2–4 files side by side. | 1.2–2× for a folder of files | Very low |
| **3. int8 "quantization"** | Store AVES's weights as small 8-bit numbers instead of 32-bit. PyTorch does this in ~3 lines (`quantize_dynamic`). | often 1.5–3× | Small accuracy change. Must compare detections before/after. |
| **4. ONNX Runtime / OpenVINO** | Convert AVES to a format run by an engine built specifically for fast CPU inference (OpenVINO is Intel's). | often 1.5–3× (can stack with 3) | Medium effort, needs testing on their actual laptops. |
| **5. Skip obviously empty audio** | A very cheap first pass (energy in the 10–40 Hz rumble band) skips stretches with nothing low-frequency at all; AVES only runs on the rest. The author floated this "two-stage" idea in notebook 5. | depends on the site, possibly large | Could miss very faint rumbles. Needs careful testing. |

### B. Needs retraining the small classifier (cheap *if* we have labelled data)

| Idea | What it means | Likely gain |
|---|---|---|
| **6. Use fewer AVES layers** | AVES has 12 layers and the code uses the output of the last one. Stopping at e.g. layer 6–8 skips a third to half of the work. The classifier must be retrained on those earlier-layer outputs. | ~1.5–2× |
| **7. Lower sample rate (250 Hz)** | Halves the audio AVES must read (see our earlier sample-rate discussion). Trade-offs: blurrier timing, loses sounds above 125 Hz. | ~2× |

Retraining the classifier itself only takes minutes, but it needs the labelled recordings (Raven files + audio). The 1,396 cached `.pt` files on your PC are *last-layer* outputs, so they can't be reused for idea 6.

### Realistic target
Ideas 1 + 2 + 3 together, all without retraining, could plausibly bring a 24 h file from ~28 min to something like **5–10 min** on a typical laptop. That's an estimate until measured.

### Keep it easy for the field team
Speed is only half of it. For a team on CPUs, likely just as important:
- **One command, sensible defaults:** auto-detect CPU/GPU, no 24 h cap, process a whole folder.
- **Resume after interruption:** skip files already done (the tool already skips a file if its Raven output exists, which is good).
- **A progress bar and time estimate** per file.
- **Simple install:** possibly a single Windows installer or a conda environment file, since ffmpeg setup is the usual pain point.
