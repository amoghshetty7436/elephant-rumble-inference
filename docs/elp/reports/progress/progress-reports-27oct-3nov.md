Tuesday, 27 October 2026: full rerun results, and a shift to speed
• Full rerun (23–27 Oct, lab PC). The same step-by-step search as last week, but every model was trained until it stopped improving. Full results: https://claude.ai/artifact/YXHAGBPGdxxhf5U8n9gexv
• Result. Compared with last week's final model, the new final model cuts false alarms by roughly half at about the same recall (80% cut-off, set on validation days):
 – Test 1 (Sample B): recall unchanged at 0.91, precision 0.65 → 0.82, false alarms 0.15 → 0.06 per hour.
 – Test 2 (Sample A): recall 0.78 → 0.76, precision 0.80 → 0.87, false alarms 0.11 → 0.06 per hour. At the 85% cut-off it reaches recall 0.80, precision 0.83, 0.09 per hour.
 – Vehicle days: false alarms 0.64 → 0.25 per hour.
 – Validation days: 0.13 false alarms per hour at 80% recall, and it can find up to 97% of rumbles.
• What changed:
 – A much wider listening window (10.9 s instead of 1.9 s) gave the biggest single gain.
 – 3,000 random forest frames per day as "not rumble" (up from 1,000).
 – Two of last week's conclusions flipped. Last week's fixed 5-minute training probably over-trained the wide-window models (we inferred this from the training logs; it wasn't tested directly).
• Against Stanford's detector on Sample A (precision 0.84 / recall 0.75, ELP's own figure): we're at 0.87 / 0.76. This isn't fully like-for-like, because we scored 54 days and ELP scored 42.
• Speed. We paused accuracy work to focus on speed. The GPU now processes a 24 h recording in about 4–8 s, down from 18 s, with identical detections. On the CPU, AVES takes about 79 s per day using all 24 cores.
• Started: an AVES fine-tuning run on the lab PC, planned to run until Friday. Plan: https://claude.ai/artifact/826Pf4xAyvEWhq2LZUNxDY
• Also drafted questions for the field team in Africa (their computers, internet speed, monthly data volume), to design a detector setup that fits them. [Not confirmed whether these were sent.]


Friday, 30 October 2026: AVES tuning, literature survey and new model designs
• AVES fine-tuning stopped early. A Windows Update restart cut it off on Wednesday at 01:20, about 9 h in. What it found before stopping:
 – Nothing beat the current setup (AVES layer 12). Earlier layers and other AVES versions all had clearly more false alarms (score 0.32–0.53, against 0.16 for the final model; lower is better).
 – LoRA didn't improve anything in the first runs. The actual fine-tuning stages never ran, and the decision on whether to rerun them is still open.
• Threshold table for the final model (results/final-model-thresholds/thresholds.md in the project files). At a 0.992 cut-off, Sample A reaches precision 0.85 / recall 0.78, ahead of Stanford on both. (That cut-off was read off the test curve, so it isn't an honest operating point.) Scores need rescaling before handover, because ELP's validation tool steps in 0.05s and all the useful range sits between 0.95 and 1.0.
• Literature survey: 16 elephant detectors and about 50 sources in all. No published study tests the way we do (whole forest days, scored on false alarms per hour), so Stanford's figure and our own model remain the bar. The strongest recent detector (DeepForestSound, 2026) adapts its pretrained network to elephants: 0.96 average precision, against 0.67 when that network is left unchanged.
 – Plain-language explainer: https://claude.ai/artifact/RnBbFcarsU76h3aw89im16
 – How each paper works: https://claude.ai/artifact/EGehMgHrQQYVJAbHn1XwCb
• Outside detectors: set up EarthToolsMaker and DeepForestSound on the lab PC to score them on our test sets. DeepForestSound takes about 87 s per day on the GPU. The run was paused, so there are no results yet.
• New model designs: 7 candidates, each rated on how likely it is to beat the current detector: https://claude.ai/artifact/1ot3fZokLQvrXcyj1Ww3zG. Round 1 started Friday evening with 6 of them, about 35–45 h on the lab PC.
• New deployments: started the final model on two unseen deployments, with results saved to the L: drive:
 – Nouabalé (June 2023, 43 sites): 5,152 recordings of 24 h, at 8 kHz.
 – Dzanga (December 2025, 34 sites): 12,497 recordings of 6 h, at 32 kHz.
 – About 184,000 h of audio in all.


Tuesday, 3 November 2026: round 1 results, and the deployment run restarted
• Round 1 candidates. Two designs beat the final model. Full results: https://claude.ai/artifact/3v8WubtxLEwwij9JqmCndu
 – Dasheng encoder (a pretrained audio model in place of AVES): validation score 0.129 against 0.159. Sample A recall 0.76 → 0.81, precision 0.87 → 0.91, false alarms 0.06 → 0.04 per hour. Sample B precision 0.82 → 0.88, false alarms 0.06 → 0.04. The catch is that vehicle false alarms more than double (0.25 → 0.57).
 – Detect, then double-check (a second image model re-checks each alarm): same validation score (0.158), but better on every test. Sample A recall 0.80, precision 0.91, false alarms 0.05 per hour. Sample B precision 0.87, false alarms 0.04. It's the only design that cuts vehicle false alarms (0.25 → 0.16).
 – Worse than the final model: BirdAVES and day-normalised fingerprints (both 0.196), and a 2-minute long-memory reader (0.286). Two designs crashed on very short noise clips; that is fixed and they're rerunning.
 – Caveat: the vehicle set is only 3 days.
• Round 2 started today on the second PC, about 35–50 h. It combines Dasheng with the double-check, uses AVES and Dasheng together, averages the 5 training runs into one model, tries stronger double-checkers, the larger Dasheng (0.6B) and LoRA fine-tuning of Dasheng. It also fixes a leak in round 1's double-checker training: its practice alarms came from a model that had already seen those days.
• Deployment inference:
 – The run froze on Saturday at 05:23 after 3,288 of 5,152 Nouabalé recordings, and Dzanga never started. The logs rule out a crash, restart, sleep, memory or a bad file. The most likely cause is a network-drive read that never returned.
 – The run now restarts itself after 20 minutes without progress, so a freeze costs minutes instead of days. It restarted today with 1,835 Nouabalé recordings left.
 – Speed: 13 s per 24 h recording (3,102 days of audio in 11 h 11 min). That's 2.4× faster than reading the file and running AVES one after the other.
 – The first 3,175 recordings produced 3,864 rumble detections (not yet reviewed).
 – ETA: Nouabalé this evening. Dzanga, between Thursday morning and early Friday. Dzanga is slower (about 52–77 s per day) because its 32 kHz audio is 4× bigger and the PC's 1 Gbit network link is already near its limit. [Both times are estimates, not yet confirmed.]
• Next: Round 2 results, finish both deployments, then review a sample of the deployment detections.
