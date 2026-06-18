/* ===========================================================================
   SignSpeak — Indian Sign Language translator
   ---------------------------------------------------------------------------
   The "deep AI" core, in three parts:
     1. PERCEPTION  — MediaPipe HandLandmarker finds 21 3D points per hand,
                      up to two hands, live from the webcam.
     2. FEATURES    — we turn those raw points into a position/scale-invariant
                      vector (a "hand fingerprint") so the same sign looks the
                      same wherever you are on screen.
     3. CLASSIFIER  — a k-Nearest-Neighbour model that YOU train in the browser:
                      show it a sign a few times and it learns to recognise it.
   Everything runs client-side. No server, no Python, no data leaves the page.
   =========================================================================== */

import { FilesetResolver, HandLandmarker }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ---- DOM handles -----------------------------------------------------------
const video      = document.getElementById("video");
const overlay    = document.getElementById("overlay");
const ctx        = overlay.getContext("2d");
const statusDot  = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const handsTag   = document.getElementById("handsTag");
const predLabel  = document.getElementById("predLabel");
const confFill   = document.getElementById("confFill");
const outputText = document.getElementById("outputText");

const thresholdEl = document.getElementById("threshold");
const dwellEl     = document.getElementById("dwell");

// ---- Model + dataset state -------------------------------------------------
let handLandmarker = null;
let running = false;
let lastVideoTime = -1;

const STORAGE_KEY = "signspeak-dataset-v1";
let dataset = loadDataset();           // [{ label, vec:[...] }, ...]

// Connections between the 21 hand landmarks, for drawing the skeleton.
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],            // thumb
  [0,5],[5,6],[6,7],[7,8],            // index
  [5,9],[9,10],[10,11],[11,12],      // middle
  [9,13],[13,14],[14,15],[15,16],    // ring
  [13,17],[17,18],[18,19],[19,20],   // pinky
  [0,17]                              // palm base
];

/* ===========================================================================
   1. MODEL SETUP
   =========================================================================== */
async function init() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });
    setStatus("ready", "Model ready — starting camera…");
    await startCamera();
  } catch (err) {
    console.error(err);
    setStatus("error", "Model failed to load. Check your connection.");
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: 960, height: 720 },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth || 960;
    overlay.height = video.videoHeight || 720;
    running = true;
    setStatus("ready", "Live — show a hand");
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    setStatus("error", "Camera blocked. Allow webcam access and reload.");
  }
}

/* ===========================================================================
   2. MAIN LOOP — detect, draw, classify
   =========================================================================== */
function loop() {
  if (!running) return;
  const now = performance.now();

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = handLandmarker.detectForVideo(video, now);
    handleResult(result);
  }
  requestAnimationFrame(loop);
}

function handleResult(result) {
  const hands = result.landmarks || [];
  const handedness = result.handednesses || result.handedness || [];
  handsTag.textContent = `${hands.length} hand${hands.length === 1 ? "" : "s"}`;

  drawHands(hands);

  if (hands.length === 0) {
    onFrameVector(null);
    return;
  }

  const vec = buildFeatureVector(hands, handedness);
  onFrameVector(vec);

  if (recording) captureSample(vec);
  else classifyFrame(vec);
}

// ---- Drawing ---------------------------------------------------------------
function drawHands(hands) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  for (const lm of hands) {
    ctx.strokeStyle = "rgba(108,140,255,0.9)";
    ctx.lineWidth = 3;
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(lm[a].x * overlay.width, lm[a].y * overlay.height);
      ctx.lineTo(lm[b].x * overlay.width, lm[b].y * overlay.height);
      ctx.stroke();
    }
    ctx.fillStyle = "#36d399";
    for (const p of lm) {
      ctx.beginPath();
      ctx.arc(p.x * overlay.width, p.y * overlay.height, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

/* ===========================================================================
   3. FEATURE EXTRACTION
   ---------------------------------------------------------------------------
   Raw landmarks are absolute positions in the frame, so the SAME sign made in
   a different spot, distance, or hand size would produce different numbers.
   We fix that per hand:
     • translate so the wrist (point 0) sits at the origin   -> position-invariant
     • scale by the wrist→middle-knuckle distance            -> size/distance-invariant
   ISL uses two hands, so we reserve a slot for the "Left" and "Right" hand
   (from MediaPipe's handedness label). A leading 1/0 flags whether each hand is
   present, letting one- and two-handed signs share one fixed-length vector.
   =========================================================================== */
const PER_HAND = 21 * 3;                 // 63 numbers per hand
const VEC_LEN = 2 * (1 + PER_HAND);      // [present,L...][present,R...] = 128

function normalizeHand(lm) {
  const wrist = lm[0];
  const mid = lm[9];                     // middle-finger knuckle
  const scale = Math.hypot(mid.x - wrist.x, mid.y - wrist.y, (mid.z||0) - (wrist.z||0)) || 1e-6;
  const out = [];
  for (const p of lm) {
    out.push((p.x - wrist.x) / scale, (p.y - wrist.y) / scale, ((p.z||0) - (wrist.z||0)) / scale);
  }
  return out;
}

function buildFeatureVector(hands, handedness) {
  // slots: index 0 = Left, index 1 = Right
  const slots = [null, null];
  hands.forEach((lm, i) => {
    const label = handedness[i]?.[0]?.categoryName || (i === 0 ? "Right" : "Left");
    const slot = label === "Left" ? 0 : 1;
    const target = slots[slot] === null ? slot : (slot === 0 ? 1 : 0); // avoid collision
    slots[target] = normalizeHand(lm);
  });

  const vec = [];
  for (const hand of slots) {
    if (hand) { vec.push(1, ...hand); }
    else { vec.push(0, ...new Array(PER_HAND).fill(0)); }
  }
  return vec;                            // length VEC_LEN
}

/* ===========================================================================
   4. THE CLASSIFIER — k-Nearest Neighbours
   ---------------------------------------------------------------------------
   To recognise the current frame, we find the K stored examples closest to it
   (smallest Euclidean distance) and let them vote. Confidence = how dominant
   the winning label is, softened by how close the match actually is. Simple,
   transparent, and trains instantly — perfect for an in-browser, user-taught
   model.
   =========================================================================== */
const K = 5;

function classify(vec) {
  if (dataset.length === 0) return { label: null, confidence: 0 };

  const dists = dataset.map((ex) => ({ label: ex.label, d: euclidean(vec, ex.vec) }));
  dists.sort((a, b) => a.d - b.d);
  const top = dists.slice(0, Math.min(K, dists.length));

  // Distance-weighted vote.
  const votes = {};
  for (const { label, d } of top) {
    votes[label] = (votes[label] || 0) + 1 / (1 + d);
  }
  let bestLabel = null, bestScore = 0, total = 0;
  for (const [label, score] of Object.entries(votes)) {
    total += score;
    if (score > bestScore) { bestScore = score; bestLabel = label; }
  }

  // Closeness factor: 0 when the nearest match is far, ~1 when it's a tight fit.
  const nearest = top[0].d;
  const closeness = Math.exp(-nearest * 1.4);
  const confidence = (bestScore / total) * closeness;
  return { label: bestLabel, confidence };
}

function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const dd = a[i] - b[i]; s += dd * dd; }
  return Math.sqrt(s);
}

/* ===========================================================================
   5. TEMPORAL SMOOTHING + COMMIT
   ---------------------------------------------------------------------------
   A single frame is noisy. We require the SAME label to win for several frames
   in a row (the "hold time") above the sensitivity threshold before we commit
   it to the output — then we lock until the hand changes, so one sign doesn't
   fire repeatedly.
   =========================================================================== */
let candidate = null;
let candidateFrames = 0;
let committedLabel = null;

function classifyFrame(vec) {
  const { label, confidence } = classify(vec);
  const threshold = parseFloat(thresholdEl.value);
  const dwell = parseInt(dwellEl.value, 10);

  predLabel.textContent = label && confidence >= threshold ? label : "—";
  confFill.style.width = `${Math.round(Math.min(confidence, 1) * 100)}%`;

  if (label && confidence >= threshold) {
    if (label === candidate) candidateFrames++;
    else { candidate = label; candidateFrames = 1; }

    if (candidateFrames >= dwell && committedLabel !== label) {
      commitSign(label);
      committedLabel = label;            // lock until something else appears
    }
  } else {
    candidate = null;
    candidateFrames = 0;
    committedLabel = null;               // hand cleared/changed -> allow re-fire
  }
}

function onFrameVector(vec) {
  if (vec === null) {
    predLabel.textContent = "—";
    confFill.style.width = "0%";
    candidate = null; candidateFrames = 0; committedLabel = null;
  }
}

/* ===========================================================================
   6. OUTPUT + SPEECH
   =========================================================================== */
let sentence = "";

function commitSign(label) {
  if (label === "SPACE") sentence += " ";
  else if (label === "DELETE") sentence = sentence.replace(/\s*\S*\s*$/, "");
  else sentence += (sentence && !sentence.endsWith(" ") ? " " : "") + label;

  renderOutput();
  flashBadge();
  if (document.getElementById("autoSpeak").checked && label !== "DELETE") {
    speak(label === "SPACE" ? "" : label);
  }
}

function renderOutput() { outputText.textContent = sentence.trim(); }

function speak(text) {
  if (!text || !("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.95;
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

function flashBadge() {
  predLabel.animate(
    [{ transform: "scale(1)" }, { transform: "scale(1.25)" }, { transform: "scale(1)" }],
    { duration: 220 }
  );
}

/* ===========================================================================
   7. TRAINING (recording samples)
   =========================================================================== */
let recording = false;
let recLabel = "";
let recCount = 0;
const recStatus = document.getElementById("recStatus");
const recLabelEl = document.getElementById("recLabel");
const recCountEl = document.getElementById("recCount");
const recordBtn = document.getElementById("recordBtn");
const labelInput = document.getElementById("labelInput");

function captureSample(vec) {
  // Only store frames that actually contain a hand.
  if (vec.every((v) => v === 0)) return;
  dataset.push({ label: recLabel, vec });
  recCount++;
  recCountEl.textContent = recCount;
}

function toggleRecording() {
  if (recording) { stopRecording(); return; }
  const name = labelInput.value.trim().toUpperCase();
  if (!name) { labelInput.focus(); return; }
  recLabel = name;
  recCount = 0;
  recording = true;
  recLabelEl.textContent = name;
  recCountEl.textContent = "0";
  recStatus.classList.remove("hidden");
  recordBtn.textContent = "■ Stop";
  recordBtn.classList.add("recording");
}

function stopRecording() {
  recording = false;
  recordBtn.textContent = "● Record";
  recordBtn.classList.remove("recording");
  recStatus.classList.add("hidden");
  saveDataset();
  renderDataset();
  updateEmptyHint();
}

/* ===========================================================================
   8. DATASET PERSISTENCE + UI
   =========================================================================== */
function loadDataset() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []; }
  catch { return []; }
}
function saveDataset() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(dataset));
}

function labelCounts() {
  const counts = {};
  for (const ex of dataset) counts[ex.label] = (counts[ex.label] || 0) + 1;
  return counts;
}

function renderDataset() {
  const counts = labelCounts();
  const chips = document.getElementById("labelChips");
  const labels = Object.keys(counts).sort();
  document.getElementById("totalSamples").textContent = `${dataset.length} samples`;

  if (labels.length === 0) {
    chips.innerHTML = `<span class="empty-chips">No signs yet.</span>`;
    return;
  }
  chips.innerHTML = "";
  for (const label of labels) {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML = `<span>${label}</span><span class="n">${counts[label]}</span><span class="x" title="delete">✕</span>`;
    chip.querySelector(".x").onclick = () => {
      dataset = dataset.filter((ex) => ex.label !== label);
      saveDataset(); renderDataset(); updateEmptyHint();
    };
    chips.appendChild(chip);
  }
}

function updateEmptyHint() {
  const hint = document.getElementById("emptyHint");
  if (hint) hint.style.display = dataset.length === 0 ? "block" : "none";
}

/* ===========================================================================
   9. UI WIRING
   =========================================================================== */
function setStatus(state, text) {
  statusDot.className = "dot" + (state === "ready" ? " ready" : state === "error" ? " error" : "");
  statusText.textContent = text;
}

// Mode switching
const panes = { translate: document.getElementById("paneTranslate"), train: document.getElementById("paneTrain") };
document.querySelectorAll(".mode-btn").forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const mode = btn.dataset.mode;
    panes.translate.classList.toggle("hidden", mode !== "translate");
    panes.train.classList.toggle("hidden", mode !== "train");
    if (mode !== "train" && recording) stopRecording();
  };
});

recordBtn.onclick = toggleRecording;
labelInput.addEventListener("keydown", (e) => { if (e.key === "Enter") toggleRecording(); });

document.getElementById("speakBtn").onclick = () => speak(sentence.trim());
document.getElementById("spaceBtn").onclick = () => { sentence += " "; renderOutput(); };
document.getElementById("backBtn").onclick = () => { sentence = sentence.replace(/\s*\S*\s*$/, ""); renderOutput(); };
document.getElementById("clearBtn").onclick = () => { sentence = ""; renderOutput(); };

// Export / import / wipe
document.getElementById("exportBtn").onclick = () => {
  const blob = new Blob([JSON.stringify(dataset)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "signspeak-model.json";
  a.click();
};
document.getElementById("importBtn").onclick = () => document.getElementById("importFile").click();
document.getElementById("importFile").onchange = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(reader.result);
      if (Array.isArray(data)) {
        dataset = dataset.concat(data.filter((d) => d.label && Array.isArray(d.vec)));
        saveDataset(); renderDataset(); updateEmptyHint();
      }
    } catch { alert("Could not read that file."); }
  };
  reader.readAsText(file);
};
document.getElementById("wipeBtn").onclick = () => {
  if (confirm("Delete ALL trained signs?")) {
    dataset = []; saveDataset(); renderDataset(); updateEmptyHint();
  }
};

// Boot
renderDataset();
updateEmptyHint();
renderOutput();
init();
