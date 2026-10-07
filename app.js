import { FilesetResolver, GestureRecognizer } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const MP_MODEL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task";
const NONE_LABELS = new Set(["none", "None"]);
const OVERLAYS = { nike: "nikeOverlay", ok: "okOverlay" };  // 커스텀 제스처 → 띄울 요소
const STABLE_FRAMES = 3;   // 연속 N프레임 같은 제스처여야 표시
const HOLD_MS = 400;       // 제스처가 사라져도 잠깐 유지

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

let recognizer, model, stream;
let lastVideoTime = -1;
let candidate = null, candidateCount = 0, shown = null, lastSeen = 0;

// ---------- features.py 와 동일한 특징 변환 ----------
function landmarksToFeatures(landmarks, handedness) {
  const pts = landmarks.map((p) => [p.x - landmarks[0].x, p.y - landmarks[0].y, p.z - landmarks[0].z]);
  if (handedness === "Left") pts.forEach((p) => { p[0] = -p[0]; });
  const scale = Math.max(...pts.map(([x, y, z]) => Math.hypot(x, y, z)));
  return pts.flat().map((v) => (scale > 0 ? v / scale : v));
}

// ---------- sklearn MLP (StandardScaler → ReLU 은닉층 → softmax) ----------
function predictProba(features) {
  let h = features.map((v, i) => (v - model.mean[i]) / model.scale[i]);
  model.layers.forEach(({ W, b }, li) => {
    const out = b.slice();
    for (let i = 0; i < h.length; i++) {
      const hi = h[i], Wi = W[i];
      for (let j = 0; j < out.length; j++) out[j] += hi * Wi[j];
    }
    h = li < model.layers.length - 1 ? out.map((v) => Math.max(0, v)) : out;
  });
  const max = Math.max(...h);
  const exp = h.map((v) => Math.exp(v - max));
  const sum = exp.reduce((a, c) => a + c, 0);
  return exp.map((v) => v / sum);
}

// ---------- 카메라 ----------
async function startCamera(deviceId) {
  stream?.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: 640, height: 480 },
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  return stream.getVideoTracks()[0].getSettings().deviceId;
}

async function setupCameras() {
  let current = await startCamera();
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  // MacBook 내장 카메라를 우선 사용 (iPhone 연속성 카메라 대신)
  const builtin = cams.find((c) => /macbook|facetime|built-in|내장/i.test(c.label));
  if (builtin && builtin.deviceId !== current) current = await startCamera(builtin.deviceId);

  const select = $("cameraSelect");
  select.innerHTML = "";
  cams.forEach((c, i) => select.add(new Option(c.label || `카메라 ${i}`, c.deviceId)));
  select.value = current;
  select.onchange = () => startCamera(select.value).catch(showError);
}

// ---------- 화면 ----------
function buildBars() {
  $("bars").innerHTML = model.classes.map((c) => `
    <div class="bar" data-label="${c}">
      <div class="bar-label"><span>${c}</span><span class="pct">0%</span></div>
      <div class="bar-track"><div class="bar-fill"></div></div>
    </div>`).join("");
}

function updateBars(probs, active) {
  document.querySelectorAll(".bar").forEach((el, i) => {
    const p = probs ? probs[i] : 0;
    el.querySelector(".bar-fill").style.width = `${p * 100}%`;
    el.querySelector(".pct").textContent = `${Math.round(p * 100)}%`;
    el.classList.toggle("active", model.classes[i] === active);
  });
}

function placeOverlay(name, landmarks) {
  for (const [label, id] of Object.entries(OVERLAYS)) {
    const el = $(id);
    const on = label === name;
    el.classList.toggle("show", on);
    if (on && landmarks) {
      const xs = landmarks.map((p) => p.x), ys = landmarks.map((p) => p.y);
      el.style.left = `${((Math.min(...xs) + Math.max(...xs)) / 2) * 100}%`;
      el.style.top = `${Math.max(Math.min(...ys) - 0.03, 0.25) * 100}%`;
    }
  }
}

function drawHand(landmarks, color) {
  const w = canvas.width, h = canvas.height;
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
    ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "#ff3b30";
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ---------- 메인 루프 ----------
function loop() {
  requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  // 거울 모드로 그린 뒤 그 이미지로 인식 (Python 버전의 cv2.flip 과 동일)
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();

  const result = recognizer.recognizeForVideo(canvas, performance.now());
  const threshold = parseFloat($("threshold").value);

  let name = null, isCustom = false, source = "손을 카메라에 보여주세요", probs = null, hand = null;
  if (result.landmarks.length) {
    hand = result.landmarks[0];
    const handedness = result.handedness[0][0].categoryName;
    probs = predictProba(landmarksToFeatures(hand, handedness));
    const best = probs.indexOf(Math.max(...probs));
    const label = model.classes[best];
    if (probs[best] >= threshold && !NONE_LABELS.has(label)) {
      name = label;
      isCustom = true;
      source = `커스텀 모델 · ${(probs[best] * 100).toFixed(0)}% · ${handedness}`;
    } else {
      const g = result.gestures[0]?.[0];
      name = g?.categoryName || "None";
      source = `기본 제스처 · ${((g?.score || 0) * 100).toFixed(0)}% · ${handedness}`;
    }
    result.landmarks.forEach((lm, i) => drawHand(lm, i === 0 && isCustom ? "#bf5af2" : "#30d158"));
  }

  // 깜빡임 방지: 연속 프레임 확인 + 잠깐 유지
  const target = isCustom && OVERLAYS[name] ? name : null;
  candidateCount = target === candidate ? candidateCount + 1 : 1;
  candidate = target;
  const now = performance.now();
  if (candidate && candidateCount >= STABLE_FRAMES) { shown = candidate; lastSeen = now; }
  else if (shown && now - lastSeen > HOLD_MS) shown = null;
  placeOverlay(shown, shown === candidate ? hand : null);

  $("gestureName").textContent = name || "–";
  $("gestureName").classList.toggle("custom", isCustom);
  $("gestureSource").textContent = source;
  updateBars(probs, isCustom ? name : null);
}

function showError(e) {
  console.error(e);
  $("error").textContent = e.message || String(e);
  $("loading").textContent = "오류: " + (e.message || e);
}

async function main() {
  $("threshold").oninput = (e) => { $("thresholdValue").textContent = Number(e.target.value).toFixed(2); };
  model = await (await fetch("custom_model.json")).json();
  buildBars();
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
  recognizer = await GestureRecognizer.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MP_MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
  });
  $("loading").textContent = "카메라 권한을 허용해 주세요";
  await setupCameras();
  $("loading").remove();
  loop();
}

main().catch(showError);
