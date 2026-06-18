# SignSpeak — Indian Sign Language Translator

A real-time, browser-based translator that turns **Indian Sign Language (ISL)**
hand signs into **text and spoken speech** using your webcam. No installs, no
server, no Python — and your video never leaves your computer.

> Why this is an *untapped* product: ISL serves ~63 million deaf/hard-of-hearing
> people in India, two-handed signs make it harder than ASL, and there is no
> good pretrained ISL model. So instead of guessing signs, SignSpeak lets the
> user **teach it their own signs in seconds** and recognises them live.

## The AI, in three layers

1. **Perception** — Google's **MediaPipe HandLandmarker** detects 21 3D points
   on each hand (up to two hands), live, in the browser.
2. **Features** — those raw points are converted into a position- and
   scale-invariant "hand fingerprint" (see `buildFeatureVector` in `app.js`), so
   the same sign looks the same anywhere on screen, near or far.
3. **Classifier** — a **k-Nearest-Neighbours** model you train yourself. Show it
   a sign a few times; it learns the fingerprint and recognises it. Temporal
   smoothing means you hold a sign briefly to "commit" it, so it doesn't fire
   randomly.

All three are written from scratch in `app.js` with heavy comments — the moat
here is the feature + classification pipeline, not a prompt.

## Run it

The webcam needs a *secure context*, so serve it over `localhost` (opening the
file directly may block the camera):

```bash
cd isl-translator
python3 -m http.server 5180
```

Then open **http://localhost:5180** and **allow camera access** when prompted.
Works best in Chrome/Edge.

## Quick start (30 seconds)

1. Click **Train signs**.
2. Type a name (e.g. `HELLO`), make the sign, and hold **Record** for ~2s while
   you slightly move/rotate your hands. Stop.
3. Repeat for 2–3 more signs (try `YES`, `NO`, and the special keys `SPACE` /
   `DELETE`).
4. Click **Translate** and sign — it shows the word, builds a sentence, and
   speaks it aloud.

Tip: record each sign from a few angles/distances for much better accuracy.

## Features

- Live hand-skeleton overlay + on-screen prediction with a confidence bar
- Sentence builder with **text-to-speech** (auto-speak toggle)
- Special signs: `SPACE` and `DELETE` act as keys
- **Sensitivity** and **Hold time** sliders to tune responsiveness
- Your trained model is saved in the browser (localStorage) and can be
  **exported/imported** as JSON to share or back up

## How it could grow

- Swap k-NN for a small neural net trained on a shared crowd-sourced ISL dataset
- Add **dynamic** (moving) signs via a sequence model over landmark frames
- Two-way mode: speech → animated sign avatar
- Mobile PWA so it works on any phone camera

## Honest limitations

- Recognises **static** hand shapes; true ISL has motion and facial grammar.
- Accuracy depends on your training samples and lighting.
- This is a prototype to prove the core idea is feasible — and it is.
