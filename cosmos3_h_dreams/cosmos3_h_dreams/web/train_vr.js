// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

"use strict";

// Meta Quest controllers for the training ground. The hands drive the same `state` as the mouse (train.js) and a
// playing demo (train_courses.js), through the clutch in train_core.js, so safety stops, recording and the exercise
// checks see no difference. Drawing comes from xr_view.js. Names start with vr: the page scripts share one scope.
const VR_SCREEN_DISTANCE_M = 1.2;
const VR_RECENTRE_HOLD_MS = 1000;
const VR_BUTTON_PRIMARY = 4;     // xr-standard gamepad: X on the left controller, A on the right
const VR_BUTTON_SECONDARY = 5;   // xr-standard gamepad: Y on the left controller, B on the right
const VR_CONTROLS = [
  "Controls",
  "Move / turn hand → move / turn instrument",
  "Trigger (hold) → close jaws",
  "Grip (hold) → energy",
  "A / X → that instrument back to its start",
  "Trigger during a demo → take over",
  "Y → pause / resume",
  "B (hold 1 s) → recentre screen",
  "Meta button → leave VR (pauses)",
];
const vr = {
  session: null, gl: null, space: null, input: newVrInput(), screen: null, panels: null, placement: null,
  nextTickMs: 0, yWasDown: false, bSince: null, homeWasDown: { left: false, right: false },
};
const vrEnterButton = document.getElementById("enter-vr");
const vrCaption = document.getElementById("caption");
const vrCaptionText = document.getElementById("caption-text");

function vrPanel(width, height, offset) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return { quad: xrMakeQuad(vr.gl), canvas, text: null, offset };
}

// Redraw a panel's texture only when its text changes, then draw it relative to the screen.
function vrDrawPanel(panel, lines, align, viewer, layer) {
  const text = lines.join("\n");
  if (text !== panel.text) {
    xrTextCanvas(panel.canvas, lines, { align, font: 36 });
    xrSetImage(vr.gl, panel.quad, panel.canvas);
    panel.text = text;
  }
  xrDrawQuad(vr.gl, panel.quad, xrMatMul(vr.placement, panel.offset), viewer, layer);
}

function vrStatusLines(now) {
  const lines = [];
  if (!recordingBadge.hidden) lines.push(recordingBadge.textContent);
  for (const side of SIDES) {
    const instrument = state.instruments[side];
    const seconds = instrument.energy && energySince[side] !== null ? (now - energySince[side]) / 1000 : 0;
    lines.push(instrumentStatus(instruments, side, instrument, seconds, true));
  }
  if (!loading.hidden) lines.push(modelStatus.textContent);
  if (!vrCaption.hidden) {   // the page's caption-hold says Tab and scroll; the guidance is reworded for the hands
    if (vrCaptionText.textContent) lines.push(vrCaptionText.textContent);
    const run = course.run;
    const guidance = run && !run.ending ? exerciseGuidance(run, activeSeconds(run.clock, now, state.held), true) : "";
    if (guidance) lines.push(guidance);
  }
  if (toolHint && now < toolHint.untilMs) lines.push(toolHint.text);
  return lines;
}

function vrMessageLines() {
  if (restore) return ["Stopped — unsafe action", ...overlayBody.textContent.split("\n"), "", "Press Y to resume"];
  if (paused) return ["Paused", "", "Press Y to resume"];
  return null;
}

// Grip pose, trigger and grip per hand (null while that controller is not tracked), A/X per side, and Y and B.
// Hand tracking (controllers put down) has a grip space but no gamepad, so no Y to pause: treated as untracked.
function vrReadHands(frame) {
  const hands = { left: null, right: null };
  const primary = { left: false, right: false };
  const secondary = { left: false, right: false };
  for (const source of vr.session.inputSources) {
    const side = source.handedness;
    const pad = source.gamepad;
    if ((side !== "left" && side !== "right") || !source.gripSpace || !pad) continue;
    primary[side] = Boolean(pad.buttons[VR_BUTTON_PRIMARY]?.pressed);
    secondary[side] = Boolean(pad.buttons[VR_BUTTON_SECONDARY]?.pressed);
    const pose = frame.getPose(source.gripSpace, vr.space);
    if (!pose || pose.emulatedPosition) continue;   // an estimated position is not tracking
    const p = pose.transform.position, o = pose.transform.orientation;
    hands[side] = {
      position: [p.x, p.y, p.z], orientation: [o.x, o.y, o.z, o.w],
      trigger: pad.buttons[0]?.value ?? 0, grip: pad.buttons[1]?.value ?? 0,
    };
  }
  return { hands, home: primary, y: secondary.left, b: secondary.right };
}

// A (right) / X (left): that instrument glides home, as R does for the active one (also while paused: it glides on
// resume). applyHands leaves a homing side to tick.
function vrHome(home) {
  for (const side of SIDES) {
    if (home[side] && !vr.homeWasDown[side]) goHome(state, side);
    vr.homeWasDown[side] = home[side];
  }
}

// Y: play/pause a demo, else pause, or resume a pause or safety stop (train.js `resume` applies the restore pose).
function vrPauseOrResume() {
  if (course.playback) toggleDemoPlay();
  else if (restore || paused) resume();
  else togglePause();
}

// Headset off, Meta menu or session end: hold, as Space would. A safety stop is already held.
function vrHold() {
  if (course.playback) {
    if (course.playback.playing) toggleDemoPlay();
  } else if (!state.held) {
    togglePause();
  }
}

function vrToolHints(hands, now) {
  for (const side of SIDES) {
    for (const [tool, value] of [["jaws", hands[side]?.trigger], ["energy", hands[side]?.grip]]) {
      if (value >= PRESSED && !hasTool(instruments, side, tool) && !state.instruments[side].gripping) {
        toolHint = { text: missingToolHint(instruments, side, tool, true), untilMs: now + 2500 };
      }
    }
  }
}

function vrRecentre(viewer, held, now) {
  if (!vr.placement) vr.placement = xrPlaceAhead(viewer.transform.matrix, VR_SCREEN_DISTANCE_M);
  if (!held) {
    vr.bSince = null;
  } else if (vr.bSince === null) {
    vr.bSince = now;
  } else if (now - vr.bSince >= VR_RECENTRE_HOLD_MS) {
    vr.placement = xrPlaceAhead(viewer.transform.matrix, VR_SCREEN_DISTANCE_M);
    vr.bSince = Infinity;   // once per hold
  }
}

function vrRender(viewer, layer, now) {
  if (video.naturalWidth > 0) xrSetImage(vr.gl, vr.screen, video);
  xrDrawQuad(vr.gl, vr.screen, vr.placement, viewer, layer);
  vrDrawPanel(vr.panels.status, vrStatusLines(now), "left", viewer, layer);
  vrDrawPanel(vr.panels.controls, VR_CONTROLS, "left", viewer, layer);
  const message = vrMessageLines();
  if (message) vrDrawPanel(vr.panels.message, message, "center", viewer, layer);
}

// The page's tick at `ticksPerS`, run from the XR loop (Quest Browser may throttle window timers in VR). Frames come at
// 72-90 Hz; the schedule keeps the average rate and never builds a backlog. Poses go out at most once per XR frame, so
// an input rate above the headset's frame rate is capped at it.
function vrTick(hands, now) {
  const period = 1000 / ticksPerS;
  if (now < vr.nextTickMs) return;
  vr.nextTickMs = Math.max(vr.nextTickMs + period, now - period);
  if (course.playback && SIDES.some(side => (hands[side]?.trigger ?? 0) >= PRESSED)) takeOverDemo();
  if (demoTick(now)) {
    clutch(vr.input);   // the hands take over from wherever the demo leaves the instruments
  } else {
    applyHands(state, vr.input, hands, instruments);
    tick(state, 30 / ticksPerS);   // only an A/X glide moves: hand-driven sides already have target == pose
    vrToolHints(hands, now);
  }
  send(frameMessage(state, now));
  updateHud();
  exerciseTick(now);
}

function vrFrame(_t, frame) {
  if (!vr.session) return;
  vr.session.requestAnimationFrame(vrFrame);
  const layer = vr.session.renderState.baseLayer;
  const viewer = xrBeginFrame(vr.gl, vr.session, vr.space, frame);
  const now = performance.now();
  const { hands, home, y, b } = vrReadHands(frame);
  if (y && !vr.yWasDown) vrPauseOrResume();
  vr.yWasDown = y;
  vrHome(home);
  vrTick(hands, now);
  if (!viewer) return;
  vrRecentre(viewer, b, now);
  vrRender(viewer, layer, now);
}

function vrEnded() {
  vrHold();   // back on the mouse, paused; the instruments wait where the hands left them
  Object.assign(vr, { session: null, gl: null, space: null, screen: null, panels: null, placement: null });
  vrEnterButton.textContent = "Enter VR";
}

async function vrEnter() {
  if (vr.session) { await vr.session.end(); return; }
  if (locked()) document.exitPointerLock();
  const started = await xrStartSession();
  try {
    Object.assign(vr, started, {
      input: newVrInput(), placement: null, nextTickMs: 0, yWasDown: false, bSince: null,
      homeWasDown: { left: false, right: false },
    });
    vr.screen = xrMakeQuad(vr.gl);
    vr.panels = {
      status: vrPanel(1024, 384, xrTranslateScale(0, -0.53, 0, 1, 0.6)),        // 1 m × 0.38 m under the screen
      message: vrPanel(1024, 468, xrTranslateScale(0, 0, 0.01, 0.7, 0.5)),      // 0.7 m × 0.32 m over its centre
      controls: vrPanel(1024, 768, xrTranslateScale(-0.85, 0, 0, 0.6, 0.75)),   // 0.6 m × 0.48 m to its left
    };
  } catch (error) {
    vr.session = null;
    await started.session.end();
    throw error;
  }
  started.session.addEventListener("visibilitychange", () => {
    if (started.session.visibilityState !== "visible") vrHold();
  });
  started.session.addEventListener("end", vrEnded);
  // A recentre (hold Meta) or boundary change moves the grip poses: re-anchor the hands and re-place the screen.
  vr.space.addEventListener("reset", () => { clutch(vr.input); vr.placement = null; });
  vrEnterButton.textContent = "Exit VR";
  started.session.requestAnimationFrame(vrFrame);
}

// The button appears only where VR can work. Over plain HTTP (--http) WebXR is unavailable, so Quest Browser says why.
function vrRefreshSupport() {
  if (!window.isSecureContext) {
    if (/OculusBrowser/.test(navigator.userAgent)) {
      vrEnterButton.hidden = false;
      vrEnterButton.textContent = "VR needs HTTPS";
      vrEnterButton.title = "Start the server with --cert and --key instead of --http";
    }
    return;
  }
  void navigator.xr?.isSessionSupported("immersive-vr").then((supported) => {
    vrEnterButton.hidden = !supported;
    vrEnterButton.disabled = !supported;
  }).catch(() => {});
}

vrEnterButton.addEventListener("click", () => {
  void vrEnter().catch((error) => notify(`Cannot enter VR: ${error.message}`));
});
vrRefreshSupport();
window.addEventListener("focus", vrRefreshSupport);
navigator.xr?.addEventListener?.("devicechange", vrRefreshSupport);
