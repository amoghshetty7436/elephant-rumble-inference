# Comparing our detector with Hori-Harm (deployment 17)

The script is `tools/compare_hori_harm.py` on `main`. It reads Bobbi's reviewed Hori-Harm table and our results folder, matches the boxes recording by recording, and writes `report.md` plus Excel-friendly CSVs and Raven tables to review.

It needs our `.probs.npy` files (next to the `.rumbles.txt` tables on L:) to compare at cut-offs other than 0.993, so run it on the lab PC.

## Steps (PowerShell, on the lab PC)

No git needed: the script runs with the rumble detector package you already installed.

1. Download `compare_hori_harm.py` (attached in the thread) and put it **inside** the package folder `C:\Users\avs97\Downloads\ElephantRumbleDetector\ElephantRumbleDetector` (next to `rumble_detector.py`).
2. Put Bobbi's file in Downloads: `C:\Users\avs97\Downloads\nn_202306_jun_dep17_d00_HHv6p2_p4_rand3_counted.txt`.
3. Paste these lines one at a time:

```
cd "C:\Users\avs97\Downloads\ElephantRumbleDetector\ElephantRumbleDetector"
.\.venv\Scripts\python.exe compare_hori_harm.py --hh "C:\Users\avs97\Downloads\nn_202306_jun_dep17_d00_HHv6p2_p4_rand3_counted.txt" --ours "L:\ELP\Detectors\Other\Amogh\Detections\nn_202306_jun_dep17" --out "C:\Users\avs97\Downloads\hh_comparison"
notepad "C:\Users\avs97\Downloads\hh_comparison\report.md"
```

- It reads about 0.5 GB of probabilities from L: (one file per compared day), so expect a few minutes.
- The report's "What is compared" section should say **"the saved probabilities give back exactly the saved table"**. That is the self-check that the probabilities and tables agree. If it says they do NOT, send me the report before trusting the other cut-offs.
- Send me the whole `hh_comparison` folder zipped (it is small).

Options: `--slack 2` (seconds allowed between two matching boxes), `--cutoffs 0.9,0.99,...`, `--sample 200` (how many of our unmatched detections go into the review sample), `--tables-only`.

## What it writes

- `report.md`: what was compared, the main result, our detector at every cut-off, Hori-Harm at higher score cut-offs, how much the matching slack matters, which rumbles we miss (by Hori-Harm score and by number of calls), and every site.
- `our_cutoffs.csv`, `hori_harm_cutoffs.csv`, `per_site.csv`: the same tables for Excel.
- `hori_harm_boxes_matched.csv`: every reviewed Hori-Harm box, with the best score of our detections on it.
- `hori_harm_rumbles_we_missed.txt`: Raven table of the confirmed rumbles we miss.
- `ours_not_in_hori_harm.txt` / `ours_not_in_hori_harm_sample.txt`: our detections with no Hori-Harm box, all of them and a random sample, with an empty Count column. The sample is the one to ask Bobbi's team to review.
- `ours_on_hori_harm_false_alarms.txt`: our detections on boxes the analysts called false alarms.

`first-pass-tables-only/` in this folder is the same report from our tables alone (cut-off 0.993 only), run here on 10 Nov.
