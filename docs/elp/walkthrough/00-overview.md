# elephant-rumble-inference: repository overview

Fork: `amoghshetty7436/elephant-rumble-inference` (identical to `ramayer/elephant-rumble-inference` at commit `f6dde7e`).

## 1. What the project does, in one paragraph

You give it long field recordings (typically 24-hour `.wav` files from microphones in Central African forests). It outputs:

- a score for every ~0.64 s of audio saying "elephant" or "not elephant",
- a **Raven selection file** (a tab-separated text file that Cornell's Raven software opens) listing the time ranges it thinks are rumbles,
- optional spectrogram PNGs with the detections coloured in.

It is the author's (Ron M / ramayer) entry for the FruitPunch AI "AI for Forest Elephants" challenge, using data from Cornell's Elephant Listening Project.

## 2. The core idea

```
raw audio (any sample rate)
   │  resample to 500 Hz, but tell the model it is 16 kHz
   ▼
AVES-bio  (frozen, pretrained HuBERT transformer for animal sounds)
   │  one 768-number vector ("embedding") per 320 samples = 0.64 s of real audio
   ▼
normalise each vector to length 1
   ▼
ElephantRumbleClassifier  (tiny 2-layer neural net, the only part that is trained)
   │  2 numbers per frame: [not-elephant score, elephant score]
   ▼
frames where elephant > not-elephant, merged into segments ≥ ~2.5 s
   ▼
Raven selection file  +  optional spectrogram PNGs
```

**The sample-rate trick.** AVES expects 16 kHz audio. Feeding it 500 Hz audio while it assumes 16 kHz plays the recording 32× faster. That has two effects:

- It raises the pitch by 5 octaves, which moves 10–30 Hz infrasonic rumbles into the range AVES was trained on.
- It cuts compute 32×, since each 20 ms AVES frame now covers 0.64 s of real time.

## 3. File map

### Top level
| File | What it is |
|---|---|
| `README.md` | Project write-up: approach, UMAP plots of embeddings, metrics, performance numbers (22 s per 24 h of audio on a GPU, 28 min on a CPU). |
| `pyproject.toml` | Poetry package definition. Declares deps (torchaudio, librosa, duckdb, einops, matplotlib, numpy<2) and two command-line tools: `elephant-rumble-inference` and `elephant-rumble-test-data-downloader`. |
| `poetry.lock` | Pinned dependency versions. |
| `LICENSE` | GPL-3.0. |
| `.gitignore` | Ignores `dist/` and `__pycache__/`. |
| `tests/__init__.py` | Empty. There are **no tests**. |

### `elephant_rumble_inference/`: the installable Python package (~1,270 lines)
| File | Lines | Role |
|---|---|---|
| `__init__.py` | 14 | Re-exports the 6 public classes so `import elephant_rumble_inference as eri; eri.AvesTorchaudioWrapper()` works. |
| `main.py` | 272 | **The command-line tool.** Parses args, loads both models, classifies each wav file, writes scores / Raven file / PNGs. |
| `aves_torchaudio_wrapper.py` | 47 | Downloads AVES-bio weights from Earth Species Project's public bucket, builds it with torchaudio's `wav2vec2_model`, and returns the last transformer layer's output (768-dim per frame). |
| `elephant_rumble_classifier.py` | 66 | The trained part: `Dropout → Linear(768→192) → LeakyReLU → Dropout → Linear(192→2)`. Downloads pretrained weights (`2024-07-03.pth` etc.) from `0ape.com/pretrained_models`. |
| `audio_file_processor.py` | 100 | **The inference engine.** Streams a wav file one hour at a time at 500 Hz. Adds ~10 s of audio before and after each hour so frames at the edges have context, runs AVES and then the classifier, trims the padding, and concatenates. Also converts between time (seconds) and frame index. |
| `triple_buffered_iterator.py` | 26 | Small helper: turns a stream of chunks into `(previous, current, next)` triples so the processor can borrow edge audio from neighbouring hours. |
| `raven_file_helper.py` | 449 | Everything about Raven label files. It loads every `.txt` label file under a folder into DuckDB (an in-process SQL engine), looks up labels per wav file, and writes new Raven files. Training helpers: it finds Train vs Test files, caches downsampled audio as float16 tensors in `/tmp`, and **generates "negative" (non-elephant) examples** from gaps between labelled rumbles. |
| `audio_file_visualizer.py` | 203 | Makes the pictures. It draws a spectrogram (librosa STFT, per-frequency median normalisation), tints it green or red by classifier score, draws label boxes, and plots both score curves underneath. |
| `test_data_downloader.py` | 87 | Downloads the **public** ELP Nouabalé recordings from AWS S3 bucket `congo8khz-pnnn` (~100k objects). Also has hand-picked lists of "interesting" files. |

### `notebooks/`
| Notebook | Role |
|---|---|
| `training_notebook.ipynb` | **How the model was trained** (49 cells). Builds embeddings for every labelled rumble and the generated negatives, balances them 50/50, trains the 2-layer classifier with Adam and class-weighted cross-entropy (2-minute cap), saves a `.pth`, and evaluates it against the previous best. The bottom half holds the author's experiment notes (layer sizes tried, airplane false positives, TODOs). |
| `evaluate_model.ipynb` | Short (10 cells). Runs the full pipeline on one test file and lists predicted segments next to the true labels as DataFrames. |
| `5-compare-ssqueezepy-s-wavelets-vs-librosa-s-stft.ipynb` | Side exploration: synchrosqueezed wavelets give cleaner rumble pictures than STFT but are slower. |
| `7-compare_stft_implementations.ipynb` | Side exploration: librosa, torchaudio and ssqueezepy STFTs give the same numbers when configured alike. |
| `9-elephant-songs.ipynb` | Fun: speeds recordings up 4× and 16× so you can hear rumbles. |

### `docs/`
| File | Role |
|---|---|
| `AI For Forest Elephants Final Presentation.pdf` | The author's challenge presentation (4.8 MB). |
| `example_visualization_output.webp` | Sample output image. |
| `assets/*.png` | The README's figures: noisy spectrogram, classifier output, UMAP of embeddings, metrics. |

## 4. Two flows to keep in mind

**Inference** (what `elephant-rumble-inference file.wav` does):
`main.py` → `AvesTorchaudioWrapper` + `ElephantRumbleClassifier` (download weights) → `AudioFileProcessor.classify_wave_file_for_rumbles` (via `TripleBufferedIterator`) → `RavenFileHelper.find_continuous_segments` / `write_raven_file` → `AudioFileVisualizer`.

**Training** (the notebook):
`RavenFileHelper("./data/Rumble")` loads labels → for each Train file, positive labels plus `get_negative_labels()` → `get_downsampled_tensor` (cached 500 Hz audio) → AVES embeddings (cached under `tmp/aves_embedding_cache`) → `MyDataset` → training loop → `.pth` → sklearn metrics.

## 5. Things I noticed on the first read (to verify in the deep dives)

1. **Test set used for early stopping.** Training-notebook cell 23 sets `val_loader = testing_loader`, so the stopping rule watches the test data. The reported metrics are probably a bit optimistic.
2. **The README says "1/3 of a second" per frame, but the code gives 0.64 s.** 320 samples / 500 Hz = 0.64 s. The 1/3 s figure matches an older 1 kHz setting.
3. **The S3 downloader probably fails without AWS credentials.** `download_test_data()` builds an anonymous client, but `download_s3_bucket()` replaces it on its first line with a normal `boto3.client('s3')`. `boto3` and `tqdm` are also missing from `pyproject.toml`.
4. **`main.py` details.** `--model` is parsed but never used (`--model-name` is the real flag). Raven output uses a fixed score of 0.9 and a fixed 5–250 Hz band. The "over a second long" message is really ≥4 frames ≈ 2.6 s.
5. **Negative examples are assumed, not labelled.** They are gaps of at least 20 s between labelled rumbles, so any rumble the annotators missed becomes a "not elephant" training example.
6. **The training data is private** (FruitPunch Google Drive layout `data/Rumble/{Training,Testing}/...`). The public S3 data has no labels in this repo's format.
7. **No tests,** and the training notebook doesn't run cleanly from top to bottom (it relies on variables created inside `DEBUG` blocks).

## 6. Suggested deep-dive order

1. `audio_file_processor.py` + `triple_buffered_iterator.py`: the heart of inference
2. `aves_torchaudio_wrapper.py`: what AVES/HuBERT is and why the sample-rate trick works
3. `elephant_rumble_classifier.py`: the trained model
4. `raven_file_helper.py`: labels, DuckDB, negative-sample generation
5. `training_notebook.ipynb`: cell by cell
6. `main.py`: the CLI end to end
7. `audio_file_visualizer.py`: how the pictures are made
8. `test_data_downloader.py`, `evaluate_model.ipynb`, the side notebooks, the PDF
