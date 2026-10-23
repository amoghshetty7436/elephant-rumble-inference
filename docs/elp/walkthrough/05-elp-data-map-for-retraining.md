# ELP dataset → retraining map

Sources: *Rumble_Training_&_Testing_Data_Info.docx* and the folder listing of `L:\ELP\Detectors\Elephant\Datasets` (both shared 2026-10-09).

## 1. The big finding: this is the same data FruitPunch trained on

The repo's code and notebook outputs refer to the same files and sites as ELP's training/testing sets:
- `raven_file_helper.py` names `Rumble/Training/Clearings/rumble_clearing_00-24hr_56days.txt`, which matches ELP's **Rumble Clearing, 56 days** (Madj, CEB1, Jobo).
- The notebook's training files are `madj_…`, `CEB1_…`, `nn…`, and its testing files are `dz_…`, `nn…`. That matches ELP's clearing + PNNN training and Dzanga + PNNN testing.
- The code renames `dzan` → `dz`, which is exactly the rename your doc describes.

**So replicating their training is realistic:** we have the same inputs. Better still, ELP has things FruitPunch didn't use: labelled **non-target clips** (thunder, rain, wind, vehicles, voices, wings, silence) and **elephant-free days**. Those are exactly what should fix the low precision.

## 2. What goes where

### Training: positives (rumbles)
| Folder | What | Use |
|---|---|---|
| `Training\Sound_Files\rumble_clearing` (56) + `Selection_Tables\rumble_clearing` | 24 h, 4 kHz, clearings (Madj/CEB1/Jobo) | ✅ rumbles |
| `Training\Sound_Files\rumble_forest\nn (only train files)\nn01a…nn10b` (~318) + `Selection_Tables\rumble_forest` (6 tables) | 24 h, 8 kHz, PNNN forest | ✅ rumbles. ⚠ see the Tag 1 notes below |
| `Training\Sound_Files\rumble_forest\TrainingSet_sounds` (136) | 24 h forest | ✅ probably the same or overlapping days; need to check against the nn folder |
| `rumble_forest\RumbleOverlap` (4) | overlapping rumbles | ✅ with care: no Begin/End Time, so use File Offset + 4 s |
| `TrainingClips\elephant_general_rumbles_TrainingSet2_500Hz` (2,013) / `_8kHz` (2,212) | short rumble clips | ✅ extra positives (the 500 Hz copies are already the rate the model uses) |
| `TrainingClips\elephant_marginal_rumbles_TrainingSet2` (941) | faint rumbles | optional. Try with and without |
| `TrainingClips\elephant_overlapping_rumbles_*`, `good_rumbles_500Hz` | | ✅ |

### Training: negatives (not elephant), the new ingredient
| Folder | Use |
|---|---|
| `TrainingClips\thunder_8kHz`, `rain_8kHz`, `wind_*`, `vehicle_*`, `human_voice*`, `wings_8kHz`, `silence_*` | ✅ **hard negatives**: the sounds that fool detectors |
| PNNN tables rows with Tag 1 = **`DUMMY_NoEles`** | ✅ marks a **full day without elephants**. Every slice of that day is a clean negative |
| gaps between labelled rumbles (what FruitPunch used) | still usable, but your doc says "not all rumbles are marked" in training, so some of these "negatives" are really rumbles |

**Ignore** the `*_x7Speed`, `*_x100Speed-800MHz` and `*_d16kHz` folders. They're the same clips sped up for other detectors. Our model does its own speed-up (the 500 Hz trick), so it needs the originals.

### Testing (never used for training or tuning)
| Folder | Truth tables | Notes |
|---|---|---|
| `Testing\Sounds\Nouabale-Ndoki_Grid-Forest-Additional2024\sounds` (83) | `Truth_Tables\Nouabale-Ndoki_Grid-Forest-Additional2024` (+ `raw_reviewed_tables`) | **The set behind Table 1's 0.245 / 0.932.** Main benchmark |
| `Testing\Sounds\Nouabale-Ndoki_Grid-Forest\24-hr` (61) | `Truth_Tables\Nouabale-Ndoki_Grid_RandSmaple` | second benchmark |
| `RumbleComplex-Forest` (30), `RumbleGeneral-Forest` (29), `Vehicle` (4) | (not shown in listing) | stress tests: vehicles/insects/thunder |
| Dzanga | | ⚠ only the hours named in each table (`dzan_…_hele_hh-hh_ok.txt`) are fully labelled; score only those hours |

Before training we must confirm **no recording day appears in both training and testing**. I'll check file names automatically once I see the lists.

## 3. Bugs this data would trigger in the current code
1. **`DUMMY_NoEles` rows would be learned as rumbles.** The code treats *every* row in a table as a rumble. These need to become negatives.
2. **RumbleOverlap rows would be silently dropped.** They have no Begin/End Time, and the code filters out rows whose computed end time is empty.
3. **"Marginal"/"Faint" notes are ignored.** We should be able to switch these rumbles in or out.
4. **Only `.wav` is found.** Need `.flac`/`.aiff` if any of the sets use them.
5. **Dzanga negatives** could be taken from un-reviewed hours.

## 4. A practical bottleneck: reading from the L: drive
A 24 h, 8 kHz, 16-bit `.wav` is about **1.4 GB**. Training plus testing is roughly 500–700 day-files, which could be **0.7–1 TB** read over the network. Pulling that from a network drive may take longer than the GPU work.

**Plan:** convert every file **once** to a small 500 Hz copy on the local SSD, about 86 MB per day, so ~50 GB total. After that, everything runs from local disk.

## 5. Still needed from you
1. **The first ~10 lines of 3 Raven tables:**
   - one from `Training\Selection_Tables\rumble_forest`
   - the one in `Training\Selection_Tables\rumble_clearing`
   - one from `Testing\Truth_Tables\Nouabale-Ndoki_Grid-Forest-Additional2024`
   
   Open in Notepad and copy-paste. They're plain text.
2. **`Testing\Detector_Performance\Rumble_Detector_Performance_Methods.docx`**, so we score detections *exactly* the way Table 1 was scored (how much overlap counts as a hit).
3. **The audio format:** are the files `.wav` or `.flac`, and roughly how big is one 24 h file?
4. **What's in `metadata` and `metadata\rumbles`** (just the file names).
