# Published detectors on our tests

Runs two public forest-elephant rumble detectors on the validation days, Sample A,
Sample B and the vehicle days, and scores them exactly like our final model
(`study.py final`): ELP's 1% overlap rule, cut-off picked on the validation days
for 80% / 85% recall, tests scored once.

| Detector | What it is | Code |
|---|---|---|
| EarthToolsMaker | YOLOv8 boxes on 164 s, 0-250 Hz spectrogram images (follow-up to the FruitPunch challenge) | github.com/earthtoolsmaker/forest-elephants-rumble-detection |
| DeepForestSound | Audio Spectrogram Transformer + LoRA, one score per 10 s chunk, trained in Kibale (Uganda) | github.com/MNHN-OFVI/DeepForestSound |

**Leakage:** EarthToolsMaker's shipped model was trained on the FruitPunch "Testing"
folder, which holds 24 of the 43 Sample A days. The report shows Sample A on all
days and on the 19 days it never saw. Sample B and the validation days are new to
both tools.

## On the lab PC

Each tool needs its own Python environment (DeepForestSound needs old torch and
timm). Python 3.11 for both.

```bat
:: 0. the two projects (Git LFS is needed for DeepForestSound's weights)
git lfs install
git clone https://github.com/earthtoolsmaker/forest-elephants-rumble-detection C:\code\etm
git clone https://github.com/MNHN-OFVI/DeepForestSound C:\code\dfs

:: 1. environments
py -3.11 -m venv C:\venvs\etm
C:\venvs\etm\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
C:\venvs\etm\Scripts\pip install ultralytics opencv-python soundfile pandas

py -3.11 -m venv C:\venvs\dfs
C:\venvs\dfs\Scripts\pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
C:\venvs\dfs\Scripts\pip install timm==0.4.5 peft==0.10.0 transformers==4.40.2 "numpy<2" soundfile pandas wget matplotlib

:: 2. list the sound files (our usual environment)
python tools\baselines\compare.py list --out D:\baselines

:: 3. run the detectors (both resume if stopped)
C:\venvs\etm\Scripts\python tools\baselines\run_etm.py --etm-repo C:\code\etm --list D:\baselines\files.tsv --out D:\baselines\etm
C:\venvs\dfs\Scripts\python tools\baselines\run_dfs.py --dfs-repo C:\code\dfs --list D:\baselines\files.tsv --out D:\baselines\dfs

:: 4. score (our usual environment); --ours adds our final model's numbers
python tools\baselines\compare.py score --out D:\baselines --ours <study folder>\final.json
```

`compare.py` takes `--source` like `dataset_configs.py` (default
`L:\ELP\Detectors\Other\Amogh\Dataset`). The result is `D:\baselines\baselines.md`,
plus `baselines.json` and the precision-recall curves in `curves\`.

## What was kept from each tool, and what changed

- **EarthToolsMaker:** their own spectrogram code, 164 s windows, 10 s overlap,
  their box-to-time conversion. Changed: confidence floor 0.01 instead of 0.25 (so the
  whole curve can be scored; 0.25 is shown as their "own" cut-off), the end of each
  file is covered, and a rumble found twice in two windows' overlap is kept once.
  Their frequency conversion is upside down; it is corrected here, though the score
  only uses time.
- **DeepForestSound:** their chunking, resampling and features (checked to be
  identical, value for value, to their `run_inference.py`), only the elephant (LF)
  model. Each 10 s chunk is one detection, as the tool writes it. Touching chunks are
  not joined: under the 1% rule one long box would cover many rumbles.
- Both: a detector's box counts as covering 5-250 Hz, exactly as ours does.

Speed on a CPU (measured here): EarthToolsMaker about 2 min per 24 h day,
DeepForestSound about 45 min per day, so use the GPU for DeepForestSound.
