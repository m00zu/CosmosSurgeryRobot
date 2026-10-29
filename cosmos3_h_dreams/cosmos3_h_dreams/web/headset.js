// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

// Video and the controller WebSocket share the same HTTPS origin.
// The server converts these poses to this checkpoint's fixed-anchor 28D actions.
const video = document.getElementById("video");
const status = document.getElementById("status");
const modelStatus = document.getElementById("model-status");
const statusCanvas = document.createElement("canvas");
statusCanvas.width = 1024;
statusCanvas.height = 128;
let statusTextureText = null;
let statusShownAt = 0;
const enterButton = document.getElementById("enter-vr");
const resetButton = document.getElementById("reset");
let socket = null;
let xrSession = null;
let xrSpace = null;
let gl = null;
let quad = null;
let statusQuad = null;
let resetHeldSince = null;
let resetSent = false;

function show(message) { status.textContent = message; }
async function refreshModelStatus() {
  try {
    const response = await fetch("/healthz", { cache: "no-store" });
    if (!response.ok) throw new Error("Model status unavailable");
    const health = await response.json();
    modelStatus.textContent = health.model_status;
    updateScenes(health);
  } catch (_) {
    modelStatus.textContent = "Model status unavailable · Reconnecting…";
  } finally {
    setTimeout(refreshModelStatus, 1000);
  }
}
function send(payload) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
}

function connectVideo() {
  video.src = `/video?t=${Date.now()}`;
  video.onerror = () => show("Video connection lost · Reconnect the page");
  show("Controller: connected · Video: MJPEG");
}

function connect() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${scheme}//${location.host}/ws`);
  socket.onopen = () => {
    show("Controller connected · Connecting video…");
    connectVideo();
  };
  socket.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      if (message.type === "error") show(message.message);
    } catch (_) { /* Ignore malformed diagnostics. */ }
  };
  socket.onclose = () => {
    show("Connection lost · Reconnecting…");
    video.removeAttribute("src");
    setTimeout(connect, 1000);
  };
}

function renderVideo(viewer, layer) {
  if (!viewer || !quad) return;
  if (video.naturalWidth > 0) xrSetImage(gl, quad, video);
  // Head-locked: /headset keeps its screen 1.2 m ahead of the eyes.
  const model = xrMatMul(viewer.transform.matrix, xrTranslateScale(0, 0, -1.2));
  if (statusTextureText !== modelStatus.textContent) {
    xrTextCanvas(statusCanvas, [modelStatus.textContent]);
    xrSetImage(gl, statusQuad, statusCanvas);
    statusTextureText = modelStatus.textContent;
    statusShownAt = performance.now();
  }
  const panel = xrMatMul(viewer.transform.matrix, xrTranslateScale(0, -0.42, -1.2, 1, 0.2));
  for (const [item, transform] of [[quad, model], [statusQuad, panel]]) {
    if (item === statusQuad && ["Model compiled", "Model ready"].includes(modelStatus.textContent)
        && performance.now() - statusShownAt >= 2000) continue;
    xrDrawQuad(gl, item, transform, viewer, layer);
  }
}

function readController(source, frame) {
  const space = source.gripSpace;
  if (!space) return null;
  const pose = frame.getPose(space, xrSpace);
  if (!pose) return null;
  const p = pose.transform.position, o = pose.transform.orientation;
  const orientation = [o.x, o.y, o.z, o.w];
  const gamepad = source.gamepad;
  return {
    position: [p.x, p.y, p.z],
    orientation,
    grip: Number(gamepad?.buttons[1]?.pressed ?? false),
    buttons: gamepad ? gamepad.buttons.map((button) => ({ p: button.pressed, v: button.value })) : [],
    axes: gamepad ? Array.from(gamepad.axes) : [],
  };
}

function onFrame(t, frame) {
  if (!xrSession) return;
  xrSession.requestAnimationFrame(onFrame);
  renderVideo(xrBeginFrame(gl, xrSession, xrSpace, frame), xrSession.renderState.baseLayer);

  const controllers = {};
  for (const source of xrSession.inputSources) {
    if (source.handedness !== "left" && source.handedness !== "right") continue;
    const reading = readController(source, frame);
    if (reading) controllers[source.handedness] = reading;
  }
  if (controllers.left || controllers.right)
    send({ type: "xr_frame", t_client: t, controllers });

  const b = controllers.right?.buttons?.[5]?.p;
  if (b) {
    if (resetHeldSince === null) resetHeldSince = t;
    if (!resetSent && t - resetHeldSince >= 1000) {
      send({ type: "reset" });
      resetSent = true;
    }
  } else {
    resetHeldSince = null;
    resetSent = false;
  }
}

async function enterVr() {
  if (xrSession) { await xrSession.end(); return; }
  const started = await xrStartSession();
  try {
    quad = xrMakeQuad(started.gl);
    statusQuad = xrMakeQuad(started.gl);
  } catch (error) {
    await started.session.end();
    throw error;
  }
  ({ gl, space: xrSpace, session: xrSession } = started);
  statusTextureText = null;
  xrSession.addEventListener("end", () => {
    xrSession = null; xrSpace = null; gl = null; quad = null;
    statusQuad = null;
    enterButton.textContent = "Enter VR";
  });
  enterButton.textContent = "Exit VR";
  xrSession.requestAnimationFrame(onFrame);
}

enterButton.addEventListener("click", () => {
  void enterVr().catch((error) => show(`Cannot enter VR: ${error.message}`));
});
resetButton.addEventListener("click", () => send({ type: "reset" }));
connect();
void refreshModelStatus();
function refreshVrSupport() {
  if (!navigator.xr) return;
  void navigator.xr.isSessionSupported("immersive-vr").then((supported) => {
    if (supported) enterButton.disabled = false;
  }).catch((error) => show(`WebXR check failed: ${error.message}`));
}
refreshVrSupport();
window.addEventListener("focus", refreshVrSupport);
navigator.xr?.addEventListener?.("devicechange", refreshVrSupport);
