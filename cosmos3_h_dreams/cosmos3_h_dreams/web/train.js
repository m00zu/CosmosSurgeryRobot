// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

// Mouse/keyboard virtual instruments for the training ground. Pure state lives in train_core.js.
// The page emulates the headset: it sends the same xr_frame messages over /ws.
const KEYS = {
  switchInstrument: "Tab", rollLeft: "KeyQ", rollRight: "KeyE", energy: "KeyF", lockGrip: "KeyG",
  home: "KeyR", reset: "Backspace", pause: "Space", legend: "KeyH",
};
// The page moves the instruments and sends a pose at this rate: 30 Hz, or the input rate
// when that is higher (the server then has a new pose for every sample).
let ticksPerS = 30;
let stepTimer = null;
function setPageRate(hz) {
  ticksPerS = hz;
  clearInterval(stepTimer);
  stepTimer = setInterval(step, 1000 / hz);
}
const state = newState();
const pressed = new Set();
const pending = { dx: 0, dy: 0, wheel: 0 };
let shift = false;
let restore = null;      // last incident's restore poses
let paused = false;
let heldSince = 0;       // performance.now() when the page last entered a pause or incident

const video = document.getElementById("video");
const status = document.getElementById("status");
const modelStatus = document.getElementById("model-status");
const textCacheLine = document.getElementById("text-cache");
const hud = document.getElementById("hud");
const legend = document.getElementById("legend");
const overlay = document.getElementById("overlay");
const overlayTitle = document.getElementById("overlay-title");
const overlayBody = document.getElementById("overlay-body");
const resumeButton = document.getElementById("resume");
const traineeInput = document.getElementById("trainee");
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");
const modeSelect = document.getElementById("mode");
const inputRateInput = document.getElementById("input-rate");
const inputRateBudget = document.getElementById("input-rate-budget");
const inputRateAdviceLine = document.getElementById("input-rate-advice");
const modeConfirm = document.getElementById("mode-confirm");
const recordingBadge = document.getElementById("recording-badge");
const sensitivity = document.getElementById("sensitivity");
const depthStep = document.getElementById("depth-step");   // mm per wheel notch
const rollSpeed = document.getElementById("roll-speed");   // rad/s while Q/E is held
const resetButton = document.getElementById("reset");
const settingsDialog = document.getElementById("settings");
const notice = document.getElementById("notice");
const loading = document.getElementById("loading");
const loadingTitle = document.getElementById("loading-title");
const loadingBody = document.getElementById("loading-body");
const activeTags = document.querySelectorAll("#active [data-side]");
const actRows = document.querySelectorAll("#act-keys [data-tool]");
const ACT_TEXT = { jaws: "Hold to close the jaws", energy: "Hold for energy", lock: "Lock the jaws closed; they stay closed after Tab" };
let instruments = DEFAULT_INSTRUMENTS;   // replaced by /healthz `instruments`
let toolHint = null;                     // { text, untilMs }: why the last click or F did nothing
let gripScene = null;                    // the scene whose start_jaws_closed `startGrip` holds
let startGrip = [];                      // sides the active scene starts gripping with
let socket = null;
let currentMode = modeSelect.value;
let inputRate = null;           // /healthz input_rate.fps
let inputRateFrames = null;     // /healthz input_rate.chunk_frames
let inputRateBusy = false;      // a /input-rate request is in flight
let inputRateAdvised = false;   // the "falling behind" notice was shown once
let modeSwitching = false;      // a /mode request is in flight
let switchNotice = false;       // the status line shows the switch notice until the server finishes
let lastModeError = null;
let lastRecordingError = null;
const energySince = { left: null, right: null };

// Routine status goes to the Settings panel; problems also show in the notice under the video.
function show(message) { status.textContent = message; }
function notify(message) {
  show(message);
  notice.textContent = message;
  notice.hidden = false;
}
function clearNotice() { notice.hidden = true; }
function send(payload) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
}
function settings() {
  return {
    metresPerPixel: Number(sensitivity.value) / Math.max(video.clientWidth, 1),
    radiansPerPixel: 0.005,
    metresPerWheel: Number(depthStep.value) / 1000,   // applyInput's wheel input is in notches
  };
}
function locked() { return document.pointerLockElement === video; }
// Typing a name, adjusting a control or using a dialog must not move the instruments.
function editing(event) {
  return ["INPUT", "SELECT", "TEXTAREA"].includes(event.target?.tagName) || Boolean(document.querySelector("dialog[open]"));
}

function showOverlay(title, body, incident) {
  overlayTitle.textContent = title;
  overlayBody.textContent = body;
  overlay.classList.toggle("incident", incident);
  overlay.hidden = false;
}
function hideOverlay() { overlay.hidden = true; }

function setAttempt(running) {
  startButton.disabled = running;
  stopButton.disabled = !running;
}

function togglePause() {
  if (restore) return;   // only the Resume button clears an incident
  paused = !paused;
  state.held = paused;
  heldSince = performance.now();
  send({ type: paused ? "pause" : "resume" });
  if (paused) showOverlay("Paused", "Press Space or Resume to continue.", false);
  else hideOverlay();
}

function resetSimulation() {
  send({ type: "reset" });
  Object.assign(state, newState());
  clutch(vr.input);   // in VR the hands re-anchor at the home poses
  applyStartGrip(state, startGrip);
  restore = null;
  paused = false;
  hideOverlay();
}

// A held button cannot survive Tab or a lost focus; a scene's starting grip stays until a click releases it.
function releaseInstrument(side) {
  if (!state.instruments[side].gripping) state.instruments[side].jaw = false;
  state.instruments[side].energy = false;
}

function connectVideo() {
  video.src = `/video?t=${Date.now()}`;
  video.onerror = () => notify("Video connection lost · Reconnect the page");
  show("Controller: connected · Video: MJPEG");
  clearNotice();
}

function connect() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  // client=train: safety stops apply only to a client that can show and resume them.
  socket = new WebSocket(`${scheme}//${location.host}/ws?client=train`);
  socket.onopen = () => {
    show("Controller connected · Connecting video…");
    connectVideo();
  };
  socket.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "error") notify(message.message);
      else if (message.type === "incident") onIncident(message);
      else if (message.type === "safety") onSafety(message.ratios);
    } catch (_) { /* Ignore malformed diagnostics. */ }
  };
  socket.onclose = () => {
    notify("Connection lost · Reconnecting…");
    video.removeAttribute("src");
    setAttempt(false);
    setTimeout(connect, 1000);
  };
}

function onIncident(message) {
  if (course.playback) endDemo();   // the stop overlay stays; Resume hands the instruments to the mouse
  state.held = true;
  heldSince = performance.now();
  paused = false;
  restore = message.restore ?? {};
  if (locked()) document.exitPointerLock();   // the Resume button needs a free cursor
  const lines = (message.violations ?? []).map((violation) => violation.message);
  showOverlay("Stopped — unsafe action", lines.join("\n"), true);
}

function resume() {
  if (restore) {
    applyRestore(state, restore);
    restore = null;
    paused = false;
    send({ type: "resume" });
    hideOverlay();
    if (!vr.session) video.requestPointerLock();
  } else if (paused) {
    togglePause();
  }
}

function modeLabel() {
  const option = Array.from(modeSelect.options).find((item) => item.value === currentMode);
  return option ? option.text : currentMode;
}

function updateHud() {
  const now = performance.now();
  const lines = [locked() ? `Active: ${instrumentLabel(instruments, state.active)} (Tab switches)` : "Click the video to control the instruments"];
  for (const side of SIDES) {
    const instrument = state.instruments[side];
    if (instrument.energy && energySince[side] === null) energySince[side] = now;
    if (!instrument.energy) energySince[side] = null;
    const seconds = instrument.energy ? (now - energySince[side]) / 1000 : 0;
    lines.push(`${side === state.active ? ">" : " "} ${instrumentStatus(instruments, side, instrument, seconds)}`);
  }
  if (toolHint && now < toolHint.untilMs) lines.push(toolHint.text);
  hud.textContent = lines.join("\n");
  for (const tag of activeTags) tag.setAttribute("aria-current", String(tag.dataset.side === state.active));
  for (const row of actRows) row.classList.toggle("unavailable", !hasTool(instruments, state.active, row.dataset.tool));
}

// Close the jaws or switch energy on, if the active instrument has that tool.
function useTool(tool) {
  if (state.held) return;
  if (!hasTool(instruments, state.active, tool)) {
    toolHint = { text: missingToolHint(instruments, state.active, tool), untilMs: performance.now() + 2500 };
    return;
  }
  state.instruments[state.active][tool === "jaws" ? "jaw" : "energy"] = true;
}

// Label the instrument tags and the Act rows from /healthz `instruments`.
function showInstruments(next) {
  if (JSON.stringify(next) === JSON.stringify(instruments)) return;
  instruments = next;
  for (const tag of activeTags) tag.textContent = instrumentLabel(instruments, tag.dataset.side);
  for (const row of actRows) {
    const owners = SIDES.filter(side => hasTool(instruments, side, row.dataset.tool));
    row.hidden = owners.length === 0;
    if (row.tagName === "DD") {
      row.textContent = owners.length === 1
        ? `${ACT_TEXT[row.dataset.act ?? row.dataset.tool]} (${instruments[owners[0]].name})`
        : ACT_TEXT[row.dataset.act ?? row.dataset.tool];
    }
  }
}

function step() {
  if (vr.session) return;   // in VR, train_vr.js runs this tick from the XR frame loop
  const now = performance.now();
  if (!demoTick(now)) {   // a playing demo drives the instruments instead of the mouse
    const roll = ((pressed.has(KEYS.rollRight) ? 1 : 0) - (pressed.has(KEYS.rollLeft) ? 1 : 0)) * Number(rollSpeed.value) / ticksPerS;
    applyInput(state, { dx: pending.dx, dy: pending.dy, wheel: pending.wheel, shift, roll }, settings());
    tick(state, 30 / ticksPerS);
  }
  pending.dx = pending.dy = pending.wheel = 0;
  send(frameMessage(state, now));
  updateHud();
  exerciseTick(now);
}

video.addEventListener("click", () => {
  if (course.playback) takeOverDemo();   // the instruments continue from the demo's pose
  document.activeElement?.blur();
  video.requestPointerLock();
});
document.addEventListener("click", (event) => {
  event.target.closest?.("button")?.blur();   // keep Space/Enter from re-clicking it
});
document.addEventListener("mousemove", (event) => {
  if (!locked()) return;
  shift = event.shiftKey;
  pending.dx += event.movementX;
  pending.dy += event.movementY;
});
document.addEventListener("wheel", (event) => {
  if (!locked()) return;
  event.preventDefault();
  pending.wheel += wheelNotches(event.deltaY, event.deltaMode);
}, { passive: false });
document.addEventListener("mousedown", (event) => {
  if (event.button !== 0 || !locked()) return;
  const instrument = state.instruments[state.active];
  if (instrument.gripping && !state.held) {   // the scene's starting grip: this click lets go
    instrument.gripping = false;
    instrument.jaw = false;
    return;
  }
  useTool("jaws");
});
document.addEventListener("mouseup", (event) => {
  // Always release, so a jaw closed before a pause or incident cannot stay closed.
  if (event.button === 0 && !state.instruments[state.active].gripping) state.instruments[state.active].jaw = false;
});

document.addEventListener("keydown", (event) => {
  if (editing(event) || event.ctrlKey || event.metaKey || event.altKey) return;   // leave Ctrl+R etc. to the browser
  if (!Object.values(KEYS).includes(event.code)) return;
  event.preventDefault();
  pressed.add(event.code);
  if (event.repeat) return;
  if (demoKey(event.code)) return;   // Space pauses and Backspace restarts a playing demo
  switch (event.code) {
    case KEYS.switchInstrument:
      releaseInstrument(state.active);
      state.active = state.active === "left" ? "right" : "left";
      break;
    case KEYS.energy: useTool("energy"); break;
    case KEYS.lockGrip:
      if (state.held) break;
      if (hasTool(instruments, state.active, "jaws")) toggleGripLock(state, state.active);
      else toolHint = { text: missingToolHint(instruments, state.active, "jaws"), untilMs: performance.now() + 2500 };
      break;
    case KEYS.home: goHome(state, state.active); break;
    case KEYS.reset: resetSimulation(); break;
    case KEYS.pause: togglePause(); break;
    case KEYS.legend: toggleLegend(); break;
  }
});
document.addEventListener("keyup", (event) => {
  if (!Object.values(KEYS).includes(event.code)) return;
  pressed.delete(event.code);
  if (editing(event)) return;
  event.preventDefault();
  if (event.code === KEYS.energy) state.instruments[state.active].energy = false;
});
window.addEventListener("blur", () => {
  // Key-up events are lost with focus; never leave energy or a roll stuck on.
  pressed.clear();
  for (const side of SIDES) releaseInstrument(side);
});

resumeButton.addEventListener("click", resume);
resetButton.addEventListener("click", resetSimulation);
startButton.addEventListener("click", () => {
  send({ type: "start", trainee: traineeInput.value.trim() || "anonymous" });
  setAttempt(true);
});
stopButton.addEventListener("click", () => {
  send({ type: "stop" });
  setAttempt(false);
});

modeSelect.addEventListener("change", () => {
  modeConfirm.hidden = modeSelect.value === currentMode;
});
document.getElementById("mode-cancel").addEventListener("click", () => {
  modeSelect.value = currentMode;
  modeConfirm.hidden = true;
});
document.getElementById("mode-confirm-button").addEventListener("click", async () => {
  const requested = modeSelect.value;
  modeConfirm.hidden = true;
  settingsDialog.close();   // the loading screen on the video shows the switch
  modeSwitching = true;
  modeSelect.disabled = true;
  try {
    const response = await fetch("/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: requested }),
    });
    if (!response.ok) throw new Error(await response.text());
    if (response.status === 202) {
      // The select stays disabled until /healthz reports the switch finished.
      switchNotice = true;
      notify("Switching model… (first switch into a mode compiles for 1–2 minutes)");
    } else {
      currentMode = requested;
      modeSelect.disabled = false;
    }
    setAttempt(false);
    if (switchNotice) startButton.disabled = true;   // re-enabled once /healthz reports the switch done
  } catch (error) {
    modeSelect.value = currentMode;
    modeSelect.disabled = false;
    notify(`Could not change mode: ${error.message}`);
  } finally {
    modeSwitching = false;
  }
});

// Arrow clicks and Enter/blur fire "change"; typing only previews the budget.
inputRateInput.addEventListener("input", () => {
  inputRateBudget.textContent = inputRateBudgetText(Number(inputRateInput.value), inputRateFrames);
});
inputRateInput.addEventListener("change", async () => {
  const requested = Number(inputRateInput.value);
  if (!inputRateInput.checkValidity() || requested === inputRate) {
    if (!inputRateInput.checkValidity()) notify(`Input rate must be ${inputRateInput.min}–${inputRateInput.max} Hz`);
    inputRateInput.value = String(inputRate);
    inputRateBudget.textContent = inputRateBudgetText(inputRate, inputRateFrames);
    return;
  }
  inputRateBusy = true;
  inputRateInput.disabled = true;
  try {
    const response = await fetch("/input-rate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fps: requested }),
    });
    if (!response.ok) throw new Error(response.statusText || await response.text());
    inputRate = (await response.json()).fps;
    if (Math.max(30, inputRate) !== ticksPerS) setPageRate(Math.max(30, inputRate));
    inputRateAdviceLine.hidden = true;
    show(`Input rate: ${inputRate} Hz`);
  } catch (error) {
    notify(`Could not change the input rate: ${error.message}`);
  } finally {
    inputRateInput.value = String(inputRate);
    inputRateBudget.textContent = inputRateBudgetText(inputRate, inputRateFrames);
    inputRateBusy = false;
    inputRateInput.disabled = false;
  }
});

// The rate from /healthz.input_rate; locked during an attempt (its video has one rate).
function updateInputRate(health) {
  const rate = health.input_rate;
  if (!rate) return;
  inputRateInput.min = String(rate.min);
  inputRateInput.max = String(rate.max);
  inputRateFrames = rate.chunk_frames;
  // Leave the box alone while the user is typing in it or a change is in flight.
  if (Math.max(30, rate.fps) !== ticksPerS) setPageRate(Math.max(30, rate.fps));
  if (!inputRateBusy && document.activeElement !== inputRateInput) {
    inputRate = rate.fps;
    inputRateInput.value = String(rate.fps);
    inputRateBudget.textContent = inputRateBudgetText(rate.fps, rate.chunk_frames);
  }
  inputRateInput.disabled = inputRateBusy || Boolean(health.recording) || !health.ready;
  const advice = inputRateAdvice(health.latency, rate);
  inputRateAdviceLine.textContent = advice;
  inputRateAdviceLine.hidden = !advice;
  if (advice && !inputRateAdvised) {
    inputRateAdvised = true;
    notify(`Video is falling behind your input. ${advice} (⚙ Settings → Input rate)`);
  }
}

async function refreshModelStatus() {
  const asked = performance.now();
  try {
    const response = await fetch("/healthz", { cache: "no-store" });
    if (!response.ok) throw new Error("Model status unavailable");
    const health = await response.json();
    modelStatus.textContent = health.model_status;
    if (health.instruments) showInstruments(health.instruments);
    // Until the model is ready (start-up, mode switch, failure) the video area says why it is idle.
    loading.hidden = health.ready !== false;
    loadingTitle.textContent = health.model_status || "Loading model…";
    loadingBody.hidden = /failed/i.test(health.model_status || "");
    // Kept while switching (text_cache is null then) so the line does not flicker.
    if (health.text_cache) textCacheLine.textContent = textCacheText(health.text_cache);
    textCacheLine.hidden = !textCacheLine.textContent;
    if (health.mode !== undefined) {
      currentMode = health.mode;
      if (modeConfirm.hidden) modeSelect.value = health.mode;   // keep a pending choice until confirmed
    }
    const switching = Boolean(health.mode_switching);
    modeSelect.disabled = modeSwitching || switching || switchNotice;
    if (switchNotice && !switching) {
      switchNotice = false;
      modeSelect.disabled = modeSwitching;
      if (!health.mode_error) {
        show(`Model mode: ${modeLabel()}`);
        clearNotice();
      }
    }
    if (health.mode_error && health.mode_error !== lastModeError) notify(health.mode_error);
    lastModeError = health.mode_error ?? null;
    if (health.recording_error && health.recording_error !== lastRecordingError) notify(health.recording_error);
    lastRecordingError = health.recording_error ?? null;
    // A scene or mode switch, or a reset, clears the server's hold; follow it unless this
    // answer may predate the server seeing the page's own pause or incident.
    if (health.hold === null && (restore || paused) && asked > heldSince + 250) {
      restore = null;
      paused = false;
      state.held = false;
      hideOverlay();
    }
    recordingBadge.hidden = !health.recording;
    onRecordingHealth(health.recording ?? null);
    // The server can end an attempt on its own (disconnect, mode switch, failure).
    if ("recording" in health) setAttempt(Boolean(health.recording));
    if (switching || switchNotice) startButton.disabled = true;   // no model to record from mid-switch
    if (health.recording) {
      recordingBadge.textContent = health.recording.video_ok === false ? "● REC (no video)" : "● REC";
    }
    updateScenes(health);
    if (health.active_scene !== gripScene) {   // a new scene starts from its own grip (the server reset too)
      gripScene = health.active_scene;
      startGrip = (health.scenes ?? []).find(scene => scene.name === gripScene)?.start_jaws_closed ?? [];
      if (!course.playback) applyStartGrip(state, startGrip);
    }
    updateResources(health.gpu);
    if ("latency" in health) updateLatency(health.latency);
    updateInputRate(health);
  } catch (_) {
    modelStatus.textContent = "Model status unavailable · Reconnecting…";
  } finally {
    setTimeout(refreshModelStatus, 1000);
  }
}

// Generation and response time under the graphs, from /healthz.latency (null: dashes).
function updateLatency(latency) {
  const el = document.getElementById("latency");
  const view = latencyView(latency);
  document.getElementById("latency-generation").textContent = view.generation;
  document.getElementById("latency-response").textContent = view.response;
  document.getElementById("latency-status").textContent = view.word;
  if (view.status) el.dataset.status = view.status;
  else delete el.dataset.status;
  el.title = view.title;
  el.hidden = false;
}

// Resource graphs under the video: GPU memory used and host RAM over the last 5 minutes, from /api/gpu.
// The titles carry the current numbers (GPU from /healthz.gpu, host RAM from /api/gpu.host).
const RESOURCE_SPAN_S = 300;
const resources = {
  el: document.getElementById("resource-charts"), samples: [], marks: [], since: null, device: null, last: null,
  totalMib: null, polling: false,
};
function updateResources(gpu) {
  if (gpu) {
    resources.totalMib = gpu.total_mib;
    document.getElementById("gpu-now").textContent =
      `${(gpu.used_mib / 1024).toFixed(1)} / ${formatGiB(gpu.total_mib)} · peak ${formatGiB(gpu.peak_mib)}`;
  }
  if (gpu && !resources.polling) {
    resources.polling = true;
    void pollResources();
  }
}
function renderResources() {
  const body = resources.last;
  if (!body || resources.el.hidden) return;
  const t0 = body.now - RESOURCE_SPAN_S;
  const host = body.host || {};
  document.getElementById("host-now").textContent =
    host.pss_mib == null ? "—" : `${formatGiB(host.pss_mib)} · peak ${formatGiB(host.peak_mib)}`;
  const common = { samples: resources.samples, t0, t1: body.now, marks: resources.marks, details: false };
  renderChart(document.getElementById("gpu-chart"), { ...common, series: [GPU_SERIES[0]], totalMib: resources.totalMib },
    { tooltip: document.getElementById("gpu-tooltip"), height: 170 });
  renderChart(document.getElementById("host-chart"), { ...common, series: GPU_HOST_SERIES },
    { tooltip: document.getElementById("host-tooltip"), height: 170 });
}
async function pollResources() {
  try {
    // GPU -1 has no samples: the first answer only tells us the server's device.
    const params = new URLSearchParams({ gpu: String(resources.device ?? -1) });
    if (resources.since !== null) params.set("since", String(resources.since));
    const response = await fetch(`/api/gpu?${params}`, { cache: "no-store" });
    if (response.status === 404) {   // monitor off for now: hide the graphs and retry slowly
      resources.el.hidden = true;
      setTimeout(pollResources, 10000);
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const body = await response.json();
    const restarted = resources.since !== null && body.now < resources.since;
    if (resources.device === null || restarted) {   // first answer, or the server restarted: start over
      resources.device = body.device;
      resources.samples = [];
      resources.marks = [];
      resources.since = body.now - RESOURCE_SPAN_S;
      setTimeout(pollResources, 0);
      return;
    }
    const cutoff = body.now - RESOURCE_SPAN_S;
    const fresh = body.samples.filter(s => s.gpu === resources.device && s.t > resources.since);
    resources.samples = resources.samples.concat(fresh).filter(s => s.t >= cutoff);
    const seen = new Set(resources.marks.map(mark => `${mark.t}|${mark.kind}|${mark.label}`));
    resources.marks = pruneMarks(
      resources.marks.concat(body.marks.filter(mark => !seen.has(`${mark.t}|${mark.kind}|${mark.label}`))), cutoff);
    resources.since = resources.samples.length
      ? resources.samples[resources.samples.length - 1].t
      : Math.max(resources.since, cutoff);   // no samples yet: never ask for more than 5 minutes
    resources.last = body;
    resources.el.hidden = false;
    renderResources();
  } catch (_) { /* keep the last graphs; the notice shows the connection state */ }
  setTimeout(pollResources, 2000);
}
window.addEventListener("resize", renderResources);
document.getElementById("settings-button").addEventListener("click", () => settingsDialog.showModal());

void initCourses();
connect();
void refreshModelStatus();
setPageRate(30);
