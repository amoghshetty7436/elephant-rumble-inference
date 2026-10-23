# The pipeline, explained simply

This is the 6-step diagram from the overview, one step at a time, in plain language.
The example used throughout is **one 24-hour recording from a microphone in the forest**.

---

## The big picture first

Imagine hiring someone to listen to 24 hours of forest audio and write down every time an elephant "talks".
That's slow and boring, so the project splits the job between two "listeners":

1. **An expert listener (AVES)** who has heard thousands of hours of animal sounds and is very good at *describing* what a sound is like. It doesn't know which sounds are elephants.
2. **A trainee (the small classifier)** who reads the expert's descriptions and only answers one question: *"Is this an elephant, yes or no?"*

Only the trainee is trained in this project. The expert is borrowed as-is.

---

## Step 1: Speed the audio up (the "500 Hz trick")

```
raw audio  →  resample to 500 Hz, but tell the model it is 16 kHz
```

**The problem:** elephant rumbles are very deep, around 10–30 Hz, mostly too low for humans to hear. The expert (AVES) learned from birds, frogs and insects, which are much higher-pitched sounds. Played at normal speed, rumbles are "invisible" to it.

**The fix:** it's like playing a slow, deep voice recording on fast-forward.

- First the audio is stored with only **500 measurements per second** ("500 Hz"). That's plenty for deep sounds.
- The expert expects **16,000 measurements per second**. So when it receives the 500-per-second audio, it treats every second as if it were 1/32 of a second.
- The result: the audio plays **32× faster**, and the pitch goes **up 5 octaves** (each doubling of speed is one octave; 2⁵ = 32).

That turns a 20 Hz rumble into roughly a 640 Hz tone, which is right in the range where the expert works well.

**Bonus:** 32× faster also means 32× less work. 24 hours of audio "feels" like 45 minutes to the model.

---

## Step 2: The expert describes each slice of sound (AVES)

```
AVES-bio → one 768-number vector per 320 samples = 0.64 s of real audio
```

- AVES chops the audio into small slices. Each slice is 320 measurements, and at 500 per second that's **0.64 seconds of real forest time**.
- For each slice it produces a **"description"**: a list of **768 numbers**. This list is called an *embedding*.
- Think of it as a fingerprint of the sound. Two slices that sound alike get similar lists of numbers; a bird chirp and an elephant rumble get very different lists.

How many slices? One hour = 1,800,000 measurements ÷ 320 = **5,625 slices**. A 24-hour file gives **135,000 slices**, each with 768 numbers.

AVES is **frozen**: its knowledge is used, never changed. (AVES is built on HuBERT, a model originally designed for human speech, and re-trained by Earth Species Project on lots of animal sounds.)

---

## Step 3: Make every description the same "size" (normalise)

```
normalise each vector to length 1
```

Loud sounds tend to produce "bigger" numbers than quiet ones. An elephant 50 m away and one 2 km away should still count as an elephant.
So each 768-number list is scaled so its overall size is exactly 1. What's left is the **shape** of the sound (what kind of sound it is), not the **loudness**.

It's like comparing the direction two arrows point and ignoring how long they are.

---

## Step 4: The trainee decides "elephant or not" (the classifier)

```
ElephantRumbleClassifier → 2 numbers per slice: [not-elephant score, elephant score]
```

This is the only part that was trained in this project, and it's tiny:

```
768 numbers  →  192 numbers  →  2 numbers
  (input)       (middle layer)   (answer)
```

- It reads the 768-number description of one slice.
- It squeezes that into 192 numbers (finding the useful patterns).
- It outputs **two scores**: how "not elephant" and how "elephant" the slice is.

In total it has about 150,000 adjustable numbers. That's tiny by modern standards, which is why it trains in about 2 minutes.

During training it was shown thousands of slices labelled "rumble" (from human experts at Cornell) and "not rumble" (quiet gaps between rumbles). Slowly it learned which descriptions mean elephant.

---

## Step 5: Turn yes/no slices into "events" (segments)

```
frames where elephant > not-elephant, merged into segments ≥ ~2.5 s
```

After step 4 there are 135,000 little verdicts in a row, like:

```
slice:   1  2  3  4  5  6  7  8  9  10 11 12 ...
verdict: no no no ye ye ye ye ye no no ye no ...
```

- A slice counts as "yes" when its **elephant score is higher than its not-elephant score**.
- Back-to-back "yes" slices are **merged** into one event. In the example, slices 4–8 become one event.
- Events that are **too short** are thrown away. Slice 11 alone is probably a noise blip, not a rumble. The code keeps only events of at least **4 slices ≈ 2.5 seconds**, since a real rumble usually lasts a few seconds.
- Each kept event is converted from slice numbers back to times. For example, slice 4 → 4 × 0.64 = 2.56 s into the recording.

---

## Step 6: Write the results (Raven file + pictures)

```
Raven selection file  +  optional spectrogram PNGs
```

- **Raven file:** a simple table (tab-separated text) with one row per detected rumble: start time, end time, frequency range, etc. Cornell's *Raven* software opens it directly, so a researcher can jump straight to each detected rumble and listen or check it. Instead of listening to 24 hours, they review a list.
- **Pictures (optional):** a spectrogram (a picture of sound: time left to right, pitch bottom to top). Areas the model calls elephant are tinted **green** and the rest **red**. Underneath, two lines show the elephant and not-elephant scores over time.

---

## The whole thing in one sentence

> Speed the forest recording up 32× so elephant rumbles sound like ordinary animal calls. Let a pretrained animal-sound model describe every 0.64-second slice. Let a tiny trained network say "elephant or not" for each slice. Then merge the "yes" slices into events and save them as a list researchers can open in Raven.

## Where each step lives in the code

| Step | File |
|---|---|
| 1. Resample to 500 Hz, stream 1 hour at a time | `audio_file_processor.py` (`classify_wave_file_for_rumbles`) |
| 2. AVES descriptions | `aves_torchaudio_wrapper.py` |
| 3. Normalise | `audio_file_processor.py` (`normalize_aves_embeddings`) |
| 4. Elephant / not-elephant | `elephant_rumble_classifier.py` |
| 5. Merge into events | `raven_file_helper.py` (`find_continuous_segments`, `find_long_enough_segments`) and `main.py` |
| 6. Raven file / pictures | `raven_file_helper.py` (`write_raven_file`), `audio_file_visualizer.py` |
